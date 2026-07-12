"""LSTM sentiment classification in PyTorch.

A compact, typed library: a from-scratch tokenizer and vocabulary, a
bidirectional LSTM classifier that can optionally load pretrained GloVe
embeddings, a training loop, and a high-level inference API that loads a trained
model from a single ``.pt`` file and classifies raw text. Calibration helpers
render a reliability diagram so predicted confidence can be trusted. A carved
sample of the public Rotten Tomatoes sentence-polarity dataset drives an offline
quickstart.
"""

from sentlstm.calibration import (
    ReliabilityCurve,
    plot_reliability_diagram,
    reliability_curve,
)
from sentlstm.data import (
    SentimentDataset,
    build_dataset,
    collate_batch,
    load_csv,
    load_rotten_tomatoes,
    synthetic_sentiment,
)
from sentlstm.embeddings import build_embedding_matrix
from sentlstm.models import LSTMClassifier
from sentlstm.predict import Prediction, SentimentClassifier
from sentlstm.tokenizer import Vocabulary, tokenize
from sentlstm.train import Trainer, accuracy, set_seed

__all__ = [
    "SentimentDataset",
    "build_dataset",
    "collate_batch",
    "load_csv",
    "load_rotten_tomatoes",
    "synthetic_sentiment",
    "build_embedding_matrix",
    "LSTMClassifier",
    "Vocabulary",
    "tokenize",
    "Trainer",
    "accuracy",
    "set_seed",
    "SentimentClassifier",
    "Prediction",
    "ReliabilityCurve",
    "reliability_curve",
    "plot_reliability_diagram",
]

__version__ = "0.2.0"
