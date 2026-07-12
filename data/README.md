# Data

This directory is gitignored. No datasets are committed.

## Rotten Tomatoes (real)

`scripts/download_data.py` fetches the public Rotten Tomatoes sentence-polarity
dataset (about 8500 train and 1000 test sentences) through the HuggingFace
`datasets` library and writes `train.csv` and `test.csv`:

```bash
python scripts/download_data.py --outdir data
```

## GloVe embeddings (optional)

Pretrained GloVe vectors are optional. Download `glove.6B.100d.txt` from the
public GloVe distribution and pass it with `--glove data/glove.6B.100d.txt`. If
no GloVe file is given, the embedding layer is trained from scratch, so the
model still runs without the large download.

## Offline

The unit tests and the `--dataset synthetic` demo path use a small built-in toy
corpus and need no download.
