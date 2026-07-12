"""Training loop and metrics for the sentiment classifier."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader

from sentlstm.data import SentimentDataset, collate_batch


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Classification accuracy of ``model`` over ``loader``."""
    model.eval()
    correct, total = 0, 0
    for ids, lengths, labels in loader:
        logits = model(ids.to(device), lengths.to(device))
        pred = logits.argmax(dim=1).cpu()
        correct += int((pred == labels).sum())
        total += labels.shape[0]
    return correct / max(total, 1)


@dataclass
class Trainer:
    """Trains the LSTM classifier with Adam and cross-entropy."""

    model: torch.nn.Module
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "test_acc": []}
    )

    def _loader(self, dataset: SentimentDataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch)

    def fit(
        self,
        train_set: SentimentDataset,
        test_set: SentimentDataset | None = None,
        epochs: int = 5,
        batch_size: int = 64,
        verbose: bool = True,
    ) -> Trainer:
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        train_loader = self._loader(train_set, batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            running, n = 0.0, 0
            for ids, lengths, labels in train_loader:
                ids, lengths, labels = (
                    ids.to(self.device),
                    lengths.to(self.device),
                    labels.to(self.device),
                )
                opt.zero_grad()
                loss = loss_fn(self.model(ids, lengths), labels)
                loss.backward()
                opt.step()
                running += loss.item() * labels.shape[0]
                n += labels.shape[0]
            train_loss = running / max(n, 1)
            self.history["train_loss"].append(train_loss)
            test_acc = float("nan")
            if test_set is not None:
                test_acc = accuracy(
                    self.model, self._loader(test_set, batch_size, False), self.device
                )
            self.history["test_acc"].append(test_acc)
            if verbose:
                print(f"epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")
        return self
