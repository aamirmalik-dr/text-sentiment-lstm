"""A bidirectional LSTM sentiment classifier."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """Embed tokens, run a bidirectional LSTM, and classify from the final state.

    Packing is used so padding tokens do not influence the recurrent states. An
    optional pretrained embedding matrix can be supplied and either frozen or
    fine-tuned.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 128,
        num_classes: int = 2,
        pretrained: torch.Tensor | None = None,
        freeze_embeddings: bool = False,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        if pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained, freeze=freeze_embeddings, padding_idx=pad_idx
            )
            embed_dim = pretrained.shape[1]
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        final = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.fc(self.dropout(final))
