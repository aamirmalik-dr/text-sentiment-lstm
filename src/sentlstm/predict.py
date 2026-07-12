"""High-level inference API: load a trained model and classify raw text.

This module wraps the tokenizer, vocabulary, and :class:`LSTMClassifier` into a
single object with a small typed surface. A trained classifier is serialized to
one ``.pt`` file that carries the model weights, the vocabulary, and the config
needed to rebuild it, so inference needs no separate vocabulary file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from sentlstm.data import collate_batch
from sentlstm.models import LSTMClassifier
from sentlstm.tokenizer import Vocabulary

DEFAULT_LABELS = ("negative", "positive")


@dataclass(frozen=True)
class Prediction:
    """A single classified text.

    Attributes:
        text: The input text.
        label: The predicted class name (for example ``"positive"``).
        label_id: The predicted class index.
        confidence: The softmax probability of the predicted class, in ``[0, 1]``.
        probabilities: Per-class softmax probabilities aligned with the label names.
    """

    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: tuple[float, ...]


class SentimentClassifier:
    """A trained sentiment model with a text-in, prediction-out interface.

    Use :meth:`load` to read a serialized model, or construct one directly from a
    fitted model and vocabulary and then call :meth:`save`.
    """

    def __init__(
        self,
        model: LSTMClassifier,
        vocab: Vocabulary,
        labels: Sequence[str] = DEFAULT_LABELS,
        max_len: int = 200,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.vocab = vocab
        self.labels = tuple(labels)
        self.max_len = max_len
        self.device = device

    @torch.no_grad()
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return per-class softmax probabilities for a batch of texts.

        Args:
            texts: One or more raw text strings.

        Returns:
            An array of shape ``(len(texts), num_classes)``.
        """
        if isinstance(texts, str):
            raise TypeError("pass a sequence of strings, not a single string")
        if not texts:
            return np.empty((0, len(self.labels)), dtype=np.float32)
        encoded = [(self.vocab.encode(t, max_len=self.max_len), 0) for t in texts]
        ids, lengths, _ = collate_batch(encoded)
        logits = self.model(ids.to(self.device), lengths.to(self.device))
        return torch.softmax(logits, dim=1).cpu().numpy()

    def predict(self, texts: Sequence[str]) -> list[Prediction]:
        """Classify a batch of texts into :class:`Prediction` records.

        Args:
            texts: One or more raw text strings.

        Returns:
            A list of predictions, one per input text.
        """
        probs = self.predict_proba(texts)
        out: list[Prediction] = []
        for text, row in zip(texts, probs, strict=True):
            idx = int(row.argmax())
            out.append(
                Prediction(
                    text=text,
                    label=self.labels[idx],
                    label_id=idx,
                    confidence=float(row[idx]),
                    probabilities=tuple(float(p) for p in row),
                )
            )
        return out

    def predict_one(self, text: str) -> Prediction:
        """Classify a single text string."""
        return self.predict([text])[0]

    def save(self, path: str | Path) -> None:
        """Serialize weights, vocabulary, and config to a single ``.pt`` file."""
        embed_dim = self.model.embedding.embedding_dim
        hidden_dim = self.model.lstm.hidden_size
        bundle = {
            "state_dict": self.model.state_dict(),
            "vocab_stoi": self.vocab.stoi,
            "config": {
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_classes": len(self.labels),
                "max_len": self.max_len,
            },
            "labels": list(self.labels),
        }
        torch.save(bundle, Path(path))

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> SentimentClassifier:
        """Load a classifier serialized by :meth:`save`.

        Args:
            path: Path to a ``.pt`` bundle.
            device: Torch device string for inference.

        Returns:
            A ready-to-use :class:`SentimentClassifier`.
        """
        bundle = torch.load(Path(path), map_location=device, weights_only=False)
        cfg = bundle["config"]
        vocab = Vocabulary.from_stoi(bundle["vocab_stoi"])
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_classes=cfg["num_classes"],
        )
        model.load_state_dict(bundle["state_dict"])
        return cls(
            model,
            vocab,
            labels=bundle.get("labels", DEFAULT_LABELS),
            max_len=cfg["max_len"],
            device=device,
        )
