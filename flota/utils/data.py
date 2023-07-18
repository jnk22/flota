"""Data helpers for FLOTA application."""

from __future__ import annotations

import random
from time import perf_counter
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from polars import DataFrame
    from transformers import BatchEncoding, PreTrainedTokenizer


class ClassificationDataset(Dataset):
    """A dataset class for classification tasks.

    The dataset is created from a pandas DataFrame, with the texts and
    labels being stored in the class instance.

    Parameters
    ----------
    data
        A pandas DataFrame containing the text and label columns.

    Attributes
    ----------
    n_classes
        The number of unique classes in the data.
    texts
        A list of text data.
    labels
        A list of label data.
    """

    def __init__(self, data: DataFrame) -> None:
        """Initialize the dataset with the data."""
        self.n_classes = len(set(data["label"]))
        self.texts = list(data["text"])
        self.labels = list(data["label"])

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[list[str], list[str]]:
        """Return the item at the given index in the dataset."""
        text = self.texts[index]
        label = self.labels[index]
        return text, label


class ClassificationCollator:
    """A collator class for classification tasks.

    The collator class takes a tokenization function, a noise level, and
    keyword arguments to use when tokenizing the data.


    Parameters
    ----------
    tok
        A tokenization function to call on the text data.
    noise
        The level of noise to add to the text data.
    base
        Boolean indicating whether to use the base tokenization function.
    """

    def __init__(
        self,
        tok: PreTrainedTokenizer,
        noise: float = 0.0,
    ) -> None:
        """Initialize the collator with the tokenization function and noise level."""
        self.__tok = tok
        self.__noise = noise

    def __call__(
        self, batch: list[tuple[str, ...]]
    ) -> tuple[BatchEncoding, torch.Tensor]:
        """Process a batch of text data and labels.

        The text data is tokenized and the labels are converted to a torch tensor.

        Parameters
        ----------
        batch
            A list of tuples containing the text and label data.

        Returns
        -------
        tuple[dict, torch.Tensor]
            A tuple containing the tokenized text data and the label tensor.
        """
        texts = [
            perturb(text, self.__noise) if self.__noise else text for text, _ in batch
        ]
        tensors = self.__tok(texts, padding=True, truncation=True, return_tensors="pt")
        labels = torch.tensor([label for _, label in batch]).long()

        return tensors, labels


class Timer:
    """Timer to be used as context manager.

    Source: https://stackoverflow.com/a/69156219.
    """

    def __enter__(self) -> Timer:
        """Entry point."""
        self.start_time: float = perf_counter()
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: TracebackType | None,
    ) -> None:
        """Exit point."""
        self.interval: float = perf_counter() - self.start_time


def perturb(text: str, noise: float) -> str:
    """Add noise to text by randomly concatenating words.

    This function takes a string `text` and a float `theta` as input
    and returns a new string with some words concatenated together
    randomly. The probability of concatenation is controlled by theta.

    Parameters
    ----------
    text
        The input text.
    theta
        The probability of concatenation.

    Returns
    -------
    str
        The perturbed text.

    Examples
    --------
    >>> random.seed(0)
    >>> perturb('This is a perturbed text example with lowest probability', 0.0)
    'This is a perturbed text example with lowest probability'

    >>> perturb('This is a perturbedtext examplewith low probability', 0.2)
    'This is a perturbedtext examplewith low probability'

    >>> perturb('This is a perturbed text example with high probability', 0.8)
    'Thisisa perturbed text example withhighprobability'

    >>> perturb('This is a perturbed text example with highest probability', 1.0)
    'Thisisaperturbedtextexamplewithhighestprobability'
    """
    text_split = text.split()
    text_perturbed = [text_split[0]]

    for word in text_split[1:]:
        if random.random() <= noise:  # noqa: S311
            text_perturbed[-1] += word
        else:
            text_perturbed.append(word)

    return " ".join(text_perturbed)


def get_best_scores(file_path: Path) -> tuple[float, float] | tuple[None, None]:
    """Get the best scores from a result file.

    This function takes a file path as input and returns a tuple of
    floats or a tuple of Nones. The file is expected to be in a specific
    format, with each line in the format `float float float int`, where
    the first two floats are the results and the last elements are the
    learning rate and epoch number.
    The function returns the highest values column-wise.

    Parameters
    ----------
    file_path
        The path to the file.

    Returns
    -------
    tuple[float, float] | tuple[None, None]
        The best results as a tuple or Nones if file not found or invalid format.
    """
    try:
        result_lines = file_path.read_text().splitlines()
        split_labels = (label.strip().split() for label in result_lines)
        results = [(float(l_split[0]), float(l_split[1])) for l_split in split_labels]

        max_result_1 = max(r[0] for r in results)
        max_result_2 = max(r[1] for r in results)

    except (FileNotFoundError, ValueError):
        return None, None

    else:
        return max_result_1, max_result_2


def read_vocab(path: Path) -> list[str]:
    """Read vocabulary from file.

    Parameters
    ----------
    path
        Vocabulary file path.

    Returns
    -------
    list[str]
        List of vocabulary words.
    """
    return path.read_text().splitlines()
