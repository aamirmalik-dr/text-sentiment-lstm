"""Classify a handful of sample reviews with the committed pretrained model.

Fully offline. Loads ``models/sentlstm_rt.pt`` and prints the predicted label and
confidence for each line in ``examples/sample_reviews.txt``.

Usage:
    python examples/classify_reviews.py
"""

from __future__ import annotations

from pathlib import Path

from sentlstm import SentimentClassifier

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "sentlstm_rt.pt"
REVIEWS_PATH = Path(__file__).with_name("sample_reviews.txt")


def main() -> int:
    clf = SentimentClassifier.load(MODEL_PATH)
    reviews = [
        line.strip()
        for line in REVIEWS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"loaded {MODEL_PATH.name}  labels={clf.labels}\n")
    for pred in clf.predict(reviews):
        bar = "+" if pred.label == "positive" else "-"
        print(f"[{bar}] {pred.label:<8} {pred.confidence:5.2f}  {pred.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
