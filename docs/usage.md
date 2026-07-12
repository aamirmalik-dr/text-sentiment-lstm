# Usage

`sentlstm` is a small typed library for binary sentiment classification with a
bidirectional LSTM. This page covers the public API from load to prediction to
calibration. Every snippet below runs offline against the committed sample and
pretrained model.

## Install

```
pip install -e ".[dev]"
```

Python 3.11 or newer. Torch is CPU only for the quickstart.

## Instant inference from the pretrained model

The repository ships a trained model at `models/sentlstm_rt.pt`. It carries the
weights, the vocabulary, and the config in one file, so no separate vocabulary
artifact is needed.

```python
from sentlstm import SentimentClassifier

clf = SentimentClassifier.load("models/sentlstm_rt.pt")

pred = clf.predict_one("a heartfelt and beautifully acted film")
print(pred.label, round(pred.confidence, 3))
# positive 0.97

for p in clf.predict(["dull and far too long", "sharp, funny, and moving"]):
    print(p.label, round(p.confidence, 2), p.probabilities)
```

`predict` returns a list of `Prediction` records:

| field | type | meaning |
| --- | --- | --- |
| `text` | `str` | the input text |
| `label` | `str` | predicted class name, `negative` or `positive` |
| `label_id` | `int` | predicted class index |
| `confidence` | `float` | softmax probability of the predicted class |
| `probabilities` | `tuple[float, ...]` | per-class softmax probabilities |

For raw probabilities without the record wrapper, use `predict_proba`, which
returns a NumPy array of shape `(n_texts, n_classes)`.

```python
probs = clf.predict_proba(["great", "awful"])   # shape (2, 2)
```

## Training your own classifier

The lower-level pieces (tokenizer, dataset, model, trainer) compose into a full
training run. This is what `scripts/train_sample.py` does.

```python
from sentlstm import (
    LSTMClassifier, SentimentClassifier, Trainer, Vocabulary,
    build_dataset, load_csv, set_seed,
)

set_seed(0)
train_texts, train_labels = load_csv("data/sample_train.csv")
test_texts, test_labels = load_csv("data/sample_test.csv")

vocab = Vocabulary().build(train_texts, min_freq=2)
train_set = build_dataset(train_texts, train_labels, vocab)
test_set = build_dataset(test_texts, test_labels, vocab)

model = LSTMClassifier(len(vocab), embed_dim=64, hidden_dim=64)
Trainer(model, lr=1e-3, weight_decay=1e-4).fit(train_set, test_set, epochs=10)

clf = SentimentClassifier(model, vocab, labels=("negative", "positive"))
clf.save("models/my_model.pt")
```

To train on the full public Rotten Tomatoes dataset instead of the carved
sample, use `scripts/benchmark.py --dataset rt`, which downloads the corpus on
demand through the HuggingFace `datasets` library.

## Calibration

A model that reports 0.9 confidence should be right about 90 percent of the
time. The `calibration` module measures how far the classifier departs from that
ideal.

```python
import numpy as np
from sentlstm import plot_reliability_diagram, reliability_curve
from sentlstm import SentimentClassifier, load_csv

clf = SentimentClassifier.load("models/sentlstm_rt.pt")
texts, labels = load_csv("data/sample_test.csv")
probs = clf.predict_proba(texts)

curve = reliability_curve(probs, np.array(labels), n_bins=10)
print("ECE", round(curve.ece, 3))

plot_reliability_diagram(probs, np.array(labels), "results/reliability.png")
```

`reliability_curve` returns a `ReliabilityCurve` with the per-bin confidence,
per-bin accuracy, bin counts, and the expected calibration error (ECE), the
count-weighted mean gap between confidence and accuracy. `plot_reliability_diagram`
renders the project's hero figure: a reliability diagram, a confidence
histogram, and the per-class distribution of the predicted positive probability.

On the carved sample this from-scratch model is noticeably overconfident: it
crowds its predictions near probability 1.0 while its accuracy sits far lower, so
the reliability bars fall below the diagonal and the ECE is large. This is the
expected signature of a small recurrent model trained without pretrained
embeddings, and it is exactly what the hero figure is meant to expose.

## Tokenizer notes

`tokenize` lowercases and splits on a simple word regex. `Vocabulary.build`
keeps tokens at or above `min_freq`, caps the size, and reserves id 0 for padding
and id 1 for unknown tokens. `Vocabulary.from_stoi` rebuilds a vocabulary from a
saved mapping, which is how `SentimentClassifier.load` restores the tokenizer.
