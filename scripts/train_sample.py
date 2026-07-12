"""Train on the committed sample and write the model, metrics, and hero figure.

This is the offline reproducibility entry point. It reads the carved Rotten
Tomatoes sample in ``data/``, trains the LSTM for a few epochs, serializes the
classifier to ``models/sentlstm_rt.pt``, writes ``results/metrics.json``, and
renders the reliability-diagram hero figure to ``results/reliability.png``.

Usage:
    python scripts/train_sample.py --epochs 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentlstm.calibration import plot_reliability_diagram
from sentlstm.data import build_dataset, load_csv
from sentlstm.models import LSTMClassifier
from sentlstm.predict import SentimentClassifier
from sentlstm.tokenizer import Vocabulary
from sentlstm.train import Trainer, set_seed

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--model-out", default=str(ROOT / "models" / "sentlstm_rt.pt"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    set_seed(args.seed)
    data_dir = Path(args.data_dir)
    train_texts, train_labels = load_csv(data_dir / "sample_train.csv")
    test_texts, test_labels = load_csv(data_dir / "sample_test.csv")

    vocab = Vocabulary().build(train_texts, min_freq=2)
    train_set = build_dataset(train_texts, train_labels, vocab)
    test_set = build_dataset(test_texts, test_labels, vocab)
    print(f"vocab={len(vocab)}  train={len(train_set)}  test={len(test_set)}")

    model = LSTMClassifier(len(vocab), embed_dim=args.embed_dim, hidden_dim=args.hidden_dim)
    trainer = Trainer(model, lr=1e-3, weight_decay=1e-4)
    trainer.fit(train_set, test_set, epochs=args.epochs, batch_size=32, verbose=True)

    clf = SentimentClassifier(model, vocab, labels=("negative", "positive"))
    probs = clf.predict_proba(test_texts)
    labels_arr = np.asarray(test_labels)
    acc = float((probs.argmax(axis=1) == labels_arr).mean())

    results_dir = Path(args.results_dir)
    curve = plot_reliability_diagram(probs, labels_arr, results_dir / "reliability.png", n_bins=10)

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    clf.save(model_out)
    size_kb = model_out.stat().st_size / 1024

    metrics = {
        "dataset": "rotten_tomatoes_carved_sample",
        "train_size": len(train_set),
        "test_size": len(test_set),
        "vocab_size": len(vocab),
        "epochs": args.epochs,
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "seed": args.seed,
        "test_accuracy": round(acc, 4),
        "expected_calibration_error": round(curve.ece, 4),
        "mean_confidence": round(float(probs.max(axis=1).mean()), 4),
        "model_file": model_out.name,
        "model_size_kb": round(size_kb, 1),
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\ntest accuracy = {acc:.4f}   ECE = {curve.ece:.4f}")
    print(f"saved model -> {model_out} ({size_kb:.1f} KB)")
    print(f"wrote results -> {results_dir}/metrics.json, {results_dir}/reliability.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
