"""Train the LSTM sentiment classifier and report accuracy and a confusion matrix.

Usage:
    # Public Rotten Tomatoes dataset (downloaded on demand):
    python scripts/benchmark.py --dataset rt --epochs 5

    # Fully offline toy corpus:
    python scripts/benchmark.py --dataset synthetic --epochs 5

    # With pretrained GloVe embeddings:
    python scripts/benchmark.py --dataset rt --glove data/glove.6B.100d.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from sentlstm.data import build_dataset, collate_batch, load_rotten_tomatoes, synthetic_sentiment
from sentlstm.embeddings import build_embedding_matrix
from sentlstm.models import LSTMClassifier
from sentlstm.tokenizer import Vocabulary
from sentlstm.train import Trainer, set_seed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["rt", "synthetic"], default="rt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--glove", default=None, help="optional GloVe txt path")
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    if args.dataset == "rt":
        (train_texts, train_labels), (test_texts, test_labels) = load_rotten_tomatoes()
    else:
        train_texts, train_labels = synthetic_sentiment(repeats=40)
        test_texts, test_labels = synthetic_sentiment(repeats=10, seed=1)

    vocab = Vocabulary().build(train_texts)
    train_set = build_dataset(train_texts, train_labels, vocab)
    test_set = build_dataset(test_texts, test_labels, vocab)
    print(f"vocab={len(vocab)}  train={len(train_set)}  test={len(test_set)}")

    pretrained = None
    if args.glove:
        pretrained = build_embedding_matrix(vocab, args.glove, embed_dim=args.embed_dim)

    model = LSTMClassifier(len(vocab), embed_dim=args.embed_dim, pretrained=pretrained)
    trainer = Trainer(model, lr=1e-3)
    trainer.fit(train_set, test_set, epochs=args.epochs, verbose=True)

    final_acc = trainer.history["test_acc"][-1]
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    # Confusion matrix.
    loader = DataLoader(test_set, batch_size=64, collate_fn=collate_batch)
    cm = np.zeros((2, 2), dtype=int)
    model.eval()
    with torch.no_grad():
        for ids, lengths, labels in loader:
            pred = model(ids, lengths).argmax(dim=1).numpy()
            for t, p in zip(labels.numpy(), pred, strict=False):
                cm[t, p] += 1

    plt.figure(figsize=(6, 4.5))
    plt.plot(
        range(1, len(trainer.history["test_acc"]) + 1), trainer.history["test_acc"], marker="o"
    )
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title("LSTM sentiment accuracy")
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy.png", dpi=120)
    plt.close()

    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["neg", "pos"])
    plt.yticks([0, 1], ["neg", "pos"])
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=120)
    plt.close()

    print(f"Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
