"""LSTM sentiment classification in PyTorch.

An end-to-end pipeline: a from-scratch tokenizer and vocabulary, a bidirectional
LSTM classifier that can optionally load pretrained GloVe embeddings, and a
training loop with accuracy and confusion-matrix reporting. The demo runs on the
public Rotten Tomatoes sentence-polarity dataset.
"""

from sentlstm.data import (
    SentimentDataset,
    collate_batch,
    load_rotten_tomatoes,
    synthetic_sentiment,
)
from sentlstm.embeddings import build_embedding_matrix
from sentlstm.models import LSTMClassifier
from sentlstm.tokenizer import Vocabulary, tokenize
from sentlstm.train import Trainer, accuracy, set_seed

__all__ = [
    "SentimentDataset",
    "collate_batch",
    "load_rotten_tomatoes",
    "synthetic_sentiment",
    "build_embedding_matrix",
    "LSTMClassifier",
    "Vocabulary",
    "tokenize",
    "Trainer",
    "accuracy",
    "set_seed",
]

__version__ = "0.1.0"
