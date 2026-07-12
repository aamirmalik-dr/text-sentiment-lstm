# Data

This directory holds datasets. The full corpus and any downloads are gitignored.
Two small files are committed so the quickstart runs with no network.

## Committed sample (carved from Rotten Tomatoes)

`sample_train.csv` (1500 rows) and `sample_test.csv` (500 rows) are a carved,
class-balanced subset of the public Rotten Tomatoes sentence-polarity dataset
(`cornell-movie-review-data/rotten_tomatoes`). Each file has a `text,label`
header, where label 1 is positive and 0 is negative. The rows were sampled with a
fixed seed by carving equal numbers of positive and negative sentences from the
official train and test splits. This is a genuine public-data subset, not
synthetic text. It is small enough to train the committed model in under a minute
on CPU and to drive the offline example.

## Full Rotten Tomatoes dataset (optional)

`scripts/download_data.py` fetches the full corpus (about 8500 train and 1000
test sentences) through the HuggingFace `datasets` library and writes
`train.csv` and `test.csv`:

```bash
python scripts/download_data.py --outdir data
```

Train on the full corpus with `python scripts/benchmark.py --dataset rt`.

## GloVe embeddings (optional)

Pretrained GloVe vectors are optional. Download `glove.6B.100d.txt` from the
public GloVe distribution and pass it with `--glove data/glove.6B.100d.txt`. If
no GloVe file is given, the embedding layer is trained from scratch, so the model
still runs without the large download.

## Offline

The unit tests, the committed sample, and the `--dataset synthetic` demo path
need no download.
