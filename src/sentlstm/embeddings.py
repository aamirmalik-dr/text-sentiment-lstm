"""Load pretrained GloVe embeddings into a vocabulary-aligned matrix."""

from __future__ import annotations

import numpy as np
import torch

from sentlstm.tokenizer import Vocabulary


def build_embedding_matrix(
    vocab: Vocabulary, glove_path: str | None, embed_dim: int = 100, seed: int = 0
) -> torch.Tensor:
    """Build an embedding matrix aligned with ``vocab``.

    Words found in the GloVe file take their pretrained vectors; the rest are
    initialized randomly. If ``glove_path`` is None, the whole matrix is random,
    so the model still runs without the (large) GloVe download.

    Args:
        vocab: The vocabulary to align to.
        glove_path: Path to a GloVe text file (``word v1 v2 ...`` per line), or None.
        embed_dim: Embedding dimension (must match the GloVe file if given).
        seed: Random seed for the fallback initialization.

    Returns:
        A float tensor of shape ``(len(vocab), embed_dim)``.
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, size=(len(vocab), embed_dim)).astype(np.float32)
    matrix[vocab.pad_id] = 0.0

    if glove_path is None:
        return torch.from_numpy(matrix)

    found = 0
    with open(glove_path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            word = parts[0]
            idx = vocab.stoi.get(word)
            if idx is None:
                continue
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] == embed_dim:
                matrix[idx] = vec
                found += 1
    print(f"GloVe: matched {found} of {len(vocab)} vocabulary words")
    return torch.from_numpy(matrix)
