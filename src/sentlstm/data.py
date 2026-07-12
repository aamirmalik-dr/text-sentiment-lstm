"""Datasets, batching, and data sources for sentiment classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from sentlstm.tokenizer import Vocabulary

# A tiny, obviously-labeled toy corpus for offline tests and CI.
_POS = [
    "this movie was great and i loved every minute",
    "an excellent wonderful film with brilliant acting",
    "a fantastic and delightful story i really enjoyed it",
    "superb direction and a beautiful moving score",
    "one of the best films i have ever seen truly amazing",
]
_NEG = [
    "this movie was terrible and i hated every minute",
    "an awful boring film with wooden acting",
    "a dreadful and dull story i really disliked it",
    "poor direction and an annoying forgettable score",
    "one of the worst films i have ever seen truly awful",
]


@dataclass
class SentimentDataset(Dataset):
    """Tokenized texts with binary sentiment labels."""

    token_ids: list[list[int]]
    labels: np.ndarray
    texts: list[str]

    def __len__(self) -> int:
        return len(self.token_ids)

    def __getitem__(self, idx: int):
        return self.token_ids[idx], int(self.labels[idx])


def build_dataset(
    texts: list[str], labels: list[int], vocab: Vocabulary, max_len: int = 200
) -> SentimentDataset:
    """Encode texts with ``vocab`` into a :class:`SentimentDataset`."""
    token_ids = [vocab.encode(t, max_len=max_len) for t in texts]
    return SentimentDataset(token_ids, np.asarray(labels, dtype=np.int64), list(texts))


def collate_batch(
    items: list[tuple[list[int], int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of token id sequences and return ids, lengths, and labels."""
    lengths = [len(ids) for ids, _ in items]
    max_len = max(lengths) if lengths else 1
    ids = np.zeros((len(items), max_len), dtype=np.int64)
    for i, (seq, _) in enumerate(items):
        ids[i, : len(seq)] = seq
    labels = np.asarray([lab for _, lab in items], dtype=np.int64)
    return (
        torch.from_numpy(ids),
        torch.tensor(lengths, dtype=torch.long).clamp(min=1),
        torch.from_numpy(labels),
    )


def synthetic_sentiment(repeats: int = 20, seed: int = 0) -> tuple[list[str], list[int]]:
    """Return a small toy sentiment corpus (positive and negative sentences)."""
    rng = np.random.default_rng(seed)
    texts = (_POS + _NEG) * repeats
    labels = ([1] * len(_POS) + [0] * len(_NEG)) * repeats
    order = rng.permutation(len(texts))
    return [texts[i] for i in order], [labels[i] for i in order]


def load_rotten_tomatoes() -> tuple[tuple[list[str], list[int]], tuple[list[str], list[int]]]:
    """Load the public Rotten Tomatoes sentence-polarity dataset.

    Uses the HuggingFace ``datasets`` library, which downloads a small (about
    1 MB) corpus of roughly 8500 train and 1000 test sentences.

    Returns:
        Two ``(texts, labels)`` tuples for the train and test splits.

    Raises:
        RuntimeError: If the dataset cannot be loaded.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"could not load rotten_tomatoes via datasets: {exc}") from exc

    def split(name: str) -> tuple[list[str], list[int]]:
        part = ds[name]
        return list(part["text"]), [int(v) for v in part["label"]]

    return split("train"), split("test")
