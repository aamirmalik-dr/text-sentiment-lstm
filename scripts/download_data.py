"""Download the public Rotten Tomatoes sentiment dataset to CSV files.

Uses the HuggingFace ``datasets`` library. Writes ``train.csv`` and
``test.csv`` with ``text,label`` columns. If the dataset cannot be fetched, the
tests and the synthetic demo path still work without it.

Usage:
    python scripts/download_data.py --outdir data
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sentlstm.data import load_rotten_tomatoes


def _write(path: Path, texts: list[str], labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        for t, y in zip(texts, labels, strict=True):
            writer.writerow([t, y])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="data")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        (train_texts, train_labels), (test_texts, test_labels) = load_rotten_tomatoes()
    except RuntimeError as exc:
        print(f"Download failed: {exc}")
        print("The tests and the --dataset synthetic demo path work without this file.")
        return 1

    _write(outdir / "train.csv", train_texts, train_labels)
    _write(outdir / "test.csv", test_texts, test_labels)
    print(f"Wrote {len(train_texts)} train and {len(test_texts)} test rows to {outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
