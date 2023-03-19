"""Module for running FLOTA tokenization."""

from __future__ import annotations

import random
import sys
from importlib import metadata
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
import uvicorn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typer import Argument, Option

from .backend import app
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

IntOrNone = Optional[int]  # noqa: UP007
PathOrNone = Optional[Path]  # noqa: UP007
BoolOrNone = Optional[bool]  # noqa: UP007

THETA = 0.3

cli = typer.Typer()

CLI_MODEL_NAME = Argument(..., help="Name of model")
CLI_DATASET = Argument(
    ..., help="CSV data file path without suffixes '[_train|_dev|_test].csv'"
)
CLI_WORDS_ENCODE = Argument(..., help="Words to encode")
CLI_WORDS_TOKENIZE = Argument(..., help="Words to tokenize")
CLI_LEARNING_RATE = Option(1e-5, min=1e-12, help="Learning rate")
CLI_BATCH_SIZE = Option(64, min=1, help="Batch size")
CLI_EPOCHS = Option(20, min=1, help="Number of epochs")
CLI_K = Option(
    None,
    min=0,
    help="Number of maximum subwords, excluding prefix/suffix (0/None=unlimited)",
)
CLI_CACHE_SIZE = Option(
    default=None, help="FLOTA internal cache size (0=disable, None=unlimited)"
)
CLI_RANDOM_SEED = Option(None, help="Random seed")
CLI_CUDA_DEVICE = Option("0", help="Selected CUDA device")
CLI_PREFIX_VOCAB = Option(None, exists=True, help="Prefix vocabulary path")
CLI_SUFFIX_VOCAB = Option(None, exists=True, help="Suffix vocabulary path")
CLI_OUTPUT = Option("results", help="Output directory for results")
CLI_NOISE = Option(
    NoiseType.NONE.value, case_sensitive=False, help="Noise for input data"
)
CLI_MODE = Option(
    TokenizeMode.FLOTA.value, case_sensitive=False, help="FLOTA mode or base"
)
CLI_RESULTS_EXIST = Option(
    default=ResultFileExistsMode.APPEND.value,
    help="Overwrite, append or skip if results file already exists",
)
CLI_STRICT = Option(default=False, help="Use strict mode for BERT model")


def version_callback(*, value: bool) -> None:
    """Print version and exit."""
    if value:
        version = metadata.version(__package__)
        print(f"FLOTA CLI version: {version}")
        raise typer.Exit


@cli.command()
def run(  # noqa: PLR0915
    model_name: str = CLI_MODEL_NAME,
    dataset: str = CLI_DATASET,
    *,
    learning_rate: float = CLI_LEARNING_RATE,
    batch_size: int = CLI_BATCH_SIZE,
    epochs: int = CLI_EPOCHS,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    random_seed: IntOrNone = CLI_RANDOM_SEED,
    cuda_device: str = CLI_CUDA_DEVICE,
    prefix_vocab: PathOrNone = CLI_PREFIX_VOCAB,
    suffix_vocab: PathOrNone = CLI_SUFFIX_VOCAB,
    output: Path = CLI_OUTPUT,
    noise: NoiseType = CLI_NOISE,
    mode: TokenizeMode = CLI_MODE,
    results_exist: ResultFileExistsMode = CLI_RESULTS_EXIST,
    strict: bool = CLI_STRICT,
) -> None:
    """Run FLOTA tokenization."""
    if random_seed is not None:
        np.random.default_rng(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    datasets = {
        run_type: ClassificationDataset(pd.read_csv(f"{dataset}_{run_type.value}.csv"))
        for run_type in RunType
    }
    num_labels = datasets[RunType.TRAIN].n_classes
    k_supported = mode not in {TokenizeMode.BASE, TokenizeMode.FLOTA_DP}

    filename = f"{model_name}_{Path(dataset).stem}_{mode.value}"
    filename += f"_{k or 0}" if k_supported else ""
    filename += "_prefix" if prefix_vocab else ""
    filename += "_suffix" if suffix_vocab else ""
    filename += f"_seed-{random_seed}" if random_seed else ""
    filename += noise.filename_extension

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
    print(f"Prefix vocabulary: {prefix_vocab if mode != TokenizeMode.BASE else 'n/a'}")
    print(f"Suffix vocabulary: {suffix_vocab if mode != TokenizeMode.BASE else 'n/a'}")
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
            prefixes=read_vocab(prefix_vocab) if prefix_vocab else None,
            suffixes=read_vocab(suffix_vocab) if suffix_vocab else None,
        )

    train_noise = THETA * (noise == NoiseType.TRAIN)
    test_noise = THETA * (noise in {NoiseType.TRAIN, NoiseType.TEST})

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


@cli.command()
def encode(
    model_name: str = CLI_MODEL_NAME,
    words: list[str] = CLI_WORDS_ENCODE,
    *,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    prefix_vocab: PathOrNone = CLI_PREFIX_VOCAB,
    suffix_vocab: PathOrNone = CLI_SUFFIX_VOCAB,
    mode: FlotaMode = CLI_MODE,
    strict: bool = CLI_STRICT,
) -> None:
    """Encode input words.

    Output will be separated into a single line for each word.
    """
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        model_name,
        mode,
        k=k,
        prefixes=read_vocab(prefix_vocab) if prefix_vocab else None,
        suffixes=read_vocab(suffix_vocab) if suffix_vocab else None,
        cache_size=cache_size,
        strict=strict,
    )

    for word in words:
        print(*tokenizer.encode(word))


@cli.command()
def tokenize(
    model_name: str = CLI_MODEL_NAME,
    words: list[str] = CLI_WORDS_TOKENIZE,
    *,
    k: IntOrNone = CLI_K,
    cache_size: IntOrNone = CLI_CACHE_SIZE,
    prefix_vocab: PathOrNone = CLI_PREFIX_VOCAB,
    suffix_vocab: PathOrNone = CLI_SUFFIX_VOCAB,
    mode: FlotaMode = CLI_MODE,
    strict: bool = CLI_STRICT,
) -> None:
    """Tokenize input words.

    Output will be separated into a single line for each word.
    """
    tokenizer = AutoFlotaTokenizer.from_pretrained(
        model_name,
        mode,
        k=k,
        prefixes=read_vocab(prefix_vocab) if prefix_vocab else None,
        suffixes=read_vocab(suffix_vocab) if suffix_vocab else None,
        cache_size=cache_size,
        strict=strict,
    )

    for word in words:
        print(*tokenizer.tokenize(word))


@cli.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Listen host."),
    port: int = typer.Option(8000, help="Listen port."),
) -> None:
    """Run FLOTA API backend server."""
    uvicorn.run(app, host=host, port=port)


@cli.callback()
def callback(
    version: BoolOrNone = typer.Option(  # noqa: ARG001
        None, "--version", callback=version_callback, is_eager=True
    ),
) -> None:
    """Run application to print version."""
