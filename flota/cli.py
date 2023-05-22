"""Module for running FLOTA tokenization."""

import random
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .enums import FlotaMode, NoiseType, ResultFileExistsMode, RunType, TokenizeMode
from .tokenizer import AutoFlotaTokenizer
from .utils import (
    ClassificationCollator,
    ClassificationDataset,
    Timer,
    TrainTestHelper,
    get_best_scores,
    read_vocab,
)


def run(  # noqa: PLR0913, PLR0915
    model_name: str,
    dataset: str,
    *,
    learning_rate: float,
    theta: float,
    batch_size: int,
    epochs: int,
    k: int | None,
    cache_size: int | None,
    random_seed: int | None,
    cuda_device: str,
    prefixes: Path | None,
    suffixes: Path | None,
    output: Path,
    noise: NoiseType,
    mode: TokenizeMode,
    results_exist: ResultFileExistsMode,
    strict: bool,
) -> None:
    """Run FLOTA tokenization."""
    if random_seed is not None:
        np.random.default_rng(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    datasets = {
        run_type: ClassificationDataset(pl.read_csv(f"{dataset}_{run_type.value}.csv"))
        for run_type in RunType
    }
    num_labels = datasets[RunType.TRAIN].n_classes
    k_supported = mode not in {TokenizeMode.BASE, TokenizeMode.FLOTA_DP}

    filename = f"{model_name}_{Path(dataset).stem}_{mode.value}"
    filename += f"_{k or 0}" if k_supported else ""
    filename += "_prefix" if prefixes else ""
    filename += "_suffix" if suffixes else ""
    filename += f"_noise_{noise.value}" if noise != NoiseType.NONE else ""
    filename += f"_seed-{random_seed}" if random_seed else ""

    results_file = Path(f"{output}/{filename}.txt")
    times_file = Path(f"{output}/{filename}_times.txt")

    best_f1_dev, best_f1_test = get_best_scores(results_file)
    print(f"Model: {model_name}")
    print(f"Mode: {mode.value}")
    print(f"K: {k or 0 if k_supported else 'n/a'}")
    print(f"Learning rate: {learning_rate:.0e}")
    print(f"Epochs: {epochs:02d}")
    print(f"Data: {dataset}")
    print(f"Number of classes: {num_labels}")
    print(f"Batch size: {batch_size:02d}")
    print(f"Random seed: {random_seed or 'n/a'}")
    print(f"Prefix vocabulary: {prefixes if mode != TokenizeMode.BASE else 'n/a'}")
    print(f"Suffix vocabulary: {suffixes if mode != TokenizeMode.BASE else 'n/a'}")
    print(f"Best F1 so far: {best_f1_dev} (dev), {best_f1_test} (test)")

    if results_file.exists():
        print(f"Result file '{results_file}' already exists:", end=" ")

        if results_exist == ResultFileExistsMode.SKIP:
            print("Skipping run.")
            sys.exit(0)

        elif results_exist == ResultFileExistsMode.OVERWRITE:
            print("Deleting old results.")
            results_file.unlink()
            times_file.unlink()

        elif results_exist == ResultFileExistsMode.APPEND:
            print("Appending results.")

    if mode == TokenizeMode.BASE:
        tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        if model_name == "gpt2":
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoFlotaTokenizer.from_pretrained(
            model_name,
            FlotaMode(mode.value),
            k=k,
            strict=strict,
            cache_size=cache_size,
            prefixes=read_vocab(prefixes) if prefixes else None,
            suffixes=read_vocab(suffixes) if suffixes else None,
        )

    train_noise = theta * (noise == NoiseType.TRAIN)
    test_noise = theta * (noise in {NoiseType.TRAIN, NoiseType.TEST})

    base_mode = mode == TokenizeMode.BASE
    train_collator = ClassificationCollator(tokenizer, train_noise, base=base_mode)
    test_collator = ClassificationCollator(tokenizer, test_noise, base=base_mode)

    collators = {
        RunType.TRAIN: train_collator,
        RunType.DEV: test_collator,
        RunType.TEST: test_collator,
    }

    loaders: dict[RunType, DataLoader] = {
        run_type: DataLoader(
            datasets[run_type],
            batch_size=batch_size,
            collate_fn=collators[run_type],
            shuffle=run_type == RunType.TRAIN,
        )
        for run_type in RunType
    }

    results_file.parent.mkdir(exist_ok=True, parents=True)
    times_file.parent.mkdir(exist_ok=True, parents=True)

    max_score_dev = 0.0
    max_score_test = 0.0

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    if model_name == "gpt2":
        model.config.pad_token_id = model.config.eos_token_id

    train_test = TrainTestHelper(model, cuda_device, learning_rate)

    print("Train model...")
    with Timer() as timer:
        for epoch in range(1, epochs + 1):
            time_train = train_test.train(loaders[RunType.TRAIN])
            f1_dev, time_dev = train_test.test(loaders[RunType.DEV])
            f1_test, time_test = train_test.test(loaders[RunType.TEST])

            max_score_dev = max(f1_dev, max_score_dev)
            max_score_test = max(f1_test, max_score_test)

            all_times = [time_train, time_dev, time_test]
            all_times = [sum(all_times), *all_times]

            result_line = f"{f1_dev:.4f} {f1_test:.4f} {learning_rate:.0e} {epoch:02d}"
            times_line = " ".join(f"{time:.4f}" for time in all_times)

            print(f"Results: {result_line}")
            print(f"Times: {times_line}")

            with results_file.open("a+") as f:
                f.write(f"{result_line}\n".replace(" ", "\t"))

            with times_file.open("a+") as f:
                f.write(f"{times_line}\n".replace(" ", "\t"))

    scores_line = f"{max_score_dev:.4f} (dev), {max_score_test:.4f} (test)"
    print(f"Done | Total duration: {timer.interval:.2f}s | Best scores: {scores_line}")


def encode(  # noqa: PLR0913
    model_name: str,
    words: list[str],
    *,
    k: int | None,
    cache_size: int | None,
    prefixes: Path | None,
    suffixes: Path | None,
    mode: FlotaMode,
    strict: bool,
) -> None:
    """Encode input words.

    Output will be separated into a single line for each word.
    """
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        model_name,
        mode,
        k=k,
        prefixes=read_vocab(prefixes) if prefixes else None,
        suffixes=read_vocab(suffixes) if suffixes else None,
        cache_size=cache_size,
        strict=strict,
    )

    for word in words:
        print(*tokenizer.encode(word))


def tokenize(  # noqa: PLR0913
    model_name: str,
    words: list[str],
    *,
    k: int | None,
    cache_size: int | None,
    prefixes: Path | None,
    suffixes: Path | None,
    mode: FlotaMode,
    strict: bool,
) -> None:
    """Tokenize input words.

    Output will be separated into a single line for each word.
    """
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        model_name,
        mode,
        k=k,
        prefixes=read_vocab(prefixes) if prefixes else None,
        suffixes=read_vocab(suffixes) if suffixes else None,
        cache_size=cache_size,
        strict=strict,
    )

    for word in words:
        print(*tokenizer.tokenize(word))


def print_version() -> None:
    """Print version and exit."""
    print(f"FLOTA CLI version: {metadata.version(__package__)}")
