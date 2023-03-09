"""Training functions for FLOTA application."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from sklearn.metrics import f1_score
from torch import optim
from tqdm import tqdm

from .data import Timer

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel


class TrainTestHelper:
    """Wrapper for training and testing models."""

    def __init__(
        self, model: PreTrainedModel, device: str, learning_rate: float
    ) -> None:
        cuda_device = torch.device(
            f"cuda:{device}" if torch.cuda.is_available() else "cpu"
        )
        self.__device = cuda_device
        self.__model = model.to(self.__device)
        self.__optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, loader: DataLoader) -> float:
        """Train model based on input loader."""
        self.__model.train()
        with Timer() as timer:
            for batch_tensors, labels in tqdm(loader):
                input_ids = batch_tensors["input_ids"].to(self.__device)
                attention_mask = batch_tensors["attention_mask"].to(self.__device)
                labels_on_device = labels.to(self.__device)

                self.__optimizer.zero_grad()
                output = self.__model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_on_device,
                )
                loss = output[0]
                loss.backward()
                self.__optimizer.step()

        return timer.interval

    def test(self, loader: DataLoader) -> tuple[float, float]:
        """Test model based on input loader."""
        self.__model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad(), Timer() as timer:
            for batch_tensors, labels in tqdm(loader):
                input_ids = batch_tensors["input_ids"].to(self.__device)
                attention_mask = batch_tensors["attention_mask"].to(self.__device)
                labels_on_device = labels.to(self.__device)

                output = self.__model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_on_device,
                )
                y_true.extend(labels.tolist())
                y_pred.extend(output[1].argmax(dim=1).tolist())

        return float(f1_score(y_true, y_pred, average="macro")), timer.interval
