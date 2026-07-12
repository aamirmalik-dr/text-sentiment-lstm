"""A simple word-level tokenizer and vocabulary."""

from __future__ import annotations

import re
from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens."""
    return _TOKEN_RE.findall(text.lower())


class Vocabulary:
    """Maps tokens to integer ids, reserving 0 for padding and 1 for unknown."""

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.itos: dict[int, str] = {0: PAD_TOKEN, 1: UNK_TOKEN}

    @classmethod
    def from_stoi(cls, stoi: dict[str, int]) -> Vocabulary:
        """Rebuild a vocabulary from a saved token-to-id mapping."""
        vocab = cls()
        vocab.stoi = dict(stoi)
        vocab.itos = {idx: tok for tok, idx in vocab.stoi.items()}
        return vocab

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    def __len__(self) -> int:
        return len(self.stoi)

    def build(
        self, texts: list[str], min_freq: int = 2, max_size: int | None = 20000
    ) -> Vocabulary:
        """Build the vocabulary from a corpus.

        Args:
            texts: Training texts.
            min_freq: Minimum token frequency to be included.
            max_size: Optional cap on vocabulary size (most frequent kept).
        """
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))
        candidates = [(tok, c) for tok, c in counter.items() if c >= min_freq]
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        if max_size is not None:
            candidates = candidates[: max_size - len(self.stoi)]
        for tok, _ in candidates:
            if tok not in self.stoi:
                idx = len(self.stoi)
                self.stoi[tok] = idx
                self.itos[idx] = tok
        return self

    def encode(self, text: str, max_len: int = 200) -> list[int]:
        """Encode text to a list of token ids, truncated to ``max_len``."""
        ids = [self.stoi.get(tok, self.unk_id) for tok in tokenize(text)]
        return ids[:max_len] if ids else [self.unk_id]
