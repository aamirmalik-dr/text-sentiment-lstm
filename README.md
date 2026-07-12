# text-sentiment-lstm

A bidirectional LSTM sentiment classifier in PyTorch, with a from-scratch word tokenizer and vocabulary, optional pretrained GloVe embeddings, and an end-to-end text to training to evaluation pipeline. The demo runs on the public Rotten Tomatoes sentence-polarity dataset.

## What it does

- Tokenizes raw text with a small regular-expression word tokenizer and builds a frequency-capped vocabulary reserving ids for padding and unknown tokens.
- Encodes and pads variable-length sequences, and packs them so padding tokens never influence the recurrent states.
- Trains a bidirectional LSTM classifier with Adam and cross-entropy, reporting per-epoch training loss and test accuracy, then writes an accuracy curve and a confusion matrix.
- Optionally loads pretrained GloVe vectors into a vocabulary-aligned embedding matrix, with words not found in the file initialized randomly. Without a GloVe file the embedding layer is trained from scratch, so the model still runs with no large download.
- Ships a fully offline path: the unit tests and the `--dataset synthetic` demo use a small built-in toy corpus and need no network access.

## What it does not do

- No transformer or pretrained language model. This is a plain recurrent baseline, not a fine-tuned BERT.
- No multi-class or aspect-based sentiment. The task is binary sentence polarity.
- No hyperparameter search. The reported numbers come from a single fixed configuration.

## Install

```
python -m venv .venv
.venv\Scripts\activate      # Windows, or: source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer.

## Run

```
python scripts/download_data.py --outdir data     # optional, writes CSVs
python scripts/benchmark.py --dataset rt --epochs 8      # real Rotten Tomatoes data
python scripts/benchmark.py --dataset synthetic         # guaranteed-offline demo
python scripts/benchmark.py --dataset rt --glove data/glove.6B.100d.txt   # with GloVe
pytest -q                                               # tests, fully offline
```

The Rotten Tomatoes corpus (about 8500 train and 1000 test sentences) downloads on demand through the HuggingFace `datasets` library. The demo notebook is `notebooks/demo.ipynb`, executed with saved outputs.

## Results

Produced by `python scripts/benchmark.py --dataset rt --epochs 8` on the public Rotten Tomatoes sentence-polarity dataset: vocabulary 8971 tokens, 8530 training and 1066 test sentences, 100-dimensional embeddings trained from scratch, hidden size 128, single fixed seed.

| Epoch | Train loss | Test accuracy |
| --- | --- | --- |
| 1 | 0.6689 | 0.6069 |
| 2 | 0.5667 | 0.6932 |
| 3 | 0.4278 | 0.6857 |
| 4 | 0.2889 | 0.7158 |
| 5 | 0.1622 | 0.7233 |
| 6 | 0.0860 | 0.7326 |
| 7 | 0.0466 | 0.7205 |
| 8 | 0.0223 | 0.7298 |

Test accuracy peaks at 0.7326 (epoch 6) and the final-epoch value is 0.7298. Training loss keeps falling toward zero while test accuracy plateaus, the expected overfitting signature for a recurrent model on a few thousand short sentences with no pretraining. This is in the usual range for a from-scratch LSTM on Rotten Tomatoes; loading pretrained GloVe vectors or adding regularization would be the natural next steps to close the train/test gap. Numbers from the offline synthetic path will differ and are only a sanity check.

## Package layout

```
src/sentlstm/       library code (tokenizer, data, embeddings, model, trainer)
scripts/            download_data.py, benchmark.py
notebooks/          demo.ipynb with executed outputs
tests/              pytest suite, runs fully offline
data/               gitignored, see data/README.md
```

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/dr-aamirmalik

## License

MIT, see LICENSE.
