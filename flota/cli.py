"""Module for running FLOTA tokenization."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import typer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typer import Argument, Option

from .enums import FlotaMode, NoiseType, ResultFileExistsMode, RunType, TokenizeMode
from .tokenizer import FlotaTokenizer
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

THETA = 0.3

cli = typer.Typer()


@cli.command()
def main(  # noqa: PLR0915
    model_name: str = Argument(..., help="Name of model"),
    dataset: str = Argument(
        ..., help="CSV data file path without suffixes '[_train|_dev|_test].csv'"
    ),
    *,
    learning_rate: float = Option(1e-5, min=1e-12, help="Learning rate"),
    batch_size: int = Option(64, min=1, help="Batch size"),
    epochs: int = Option(20, min=1, help="Number of epochs"),
    k: IntOrNone = Option(
        None,
        min=0,
        help="Number of maximum subwords, excluding prefix/suffix (0/None=unlimited)",
    ),
    cache_size: IntOrNone = Option(
        default=None, help="FLOTA internal cache size (0=disable, None=unlimited)"
    ),
    random_seed: IntOrNone = Option(None, help="Random seed"),
    cuda_device: str = Option("0", help="Selected CUDA device"),
    prefix_vocab: PathOrNone = Option(None, exists=True, help="Prefix vocabulary path"),
    suffix_vocab: PathOrNone = Option(None, exists=True, help="Suffix vocabulary path"),
    output: Path = Option(Path("results"), help="Output directory for results"),
    noise: NoiseType = Option(
        NoiseType.NONE.value, case_sensitive=False, help="Noise for input data"
    ),
    mode: TokenizeMode = Option(
        TokenizeMode.FLOTA.value, case_sensitive=False, help="FLOTA mode or base"
    ),
    results_exist: ResultFileExistsMode = Option(
        default=ResultFileExistsMode.APPEND.value,
        help="Overwrite, append or skip if results file already exists",
    ),
    strict: bool = Option(default=False, help="Use strict mode for BERT model"),
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

    filename = f"{model_name}_{Path(dataset).stem}_{mode.value}_{k or 0}"
    filename += "_prefix" if prefix_vocab else ""
    filename += "_suffix" if suffix_vocab else ""
    filename += f"_seed-{random_seed}" if random_seed else ""
    filename += noise.filename_extension

    results_file = Path(f"{output}/{filename}.txt")
    times_file = Path(f"{output}/{filename}_times.txt")

    best_f1_dev, best_f1_test = get_best_scores(results_file)
    print(f"Model: {model_name}")
    print(f"Mode: {mode.value}")
    print(f"K: {k or 0 if mode != TokenizeMode.BASE else 'n/a'}")
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
        tokenizer = FlotaTokenizer.from_pretrained(
            model_name,
            FlotaMode(mode.value),
            k=k,
            strict=strict,
            cache_size=cache_size,
            prefix_vocab=read_vocab(prefix_vocab) if prefix_vocab else None,
            suffix_vocab=read_vocab(suffix_vocab) if suffix_vocab else None,
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
