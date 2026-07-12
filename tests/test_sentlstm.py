import numpy as np
import torch
from torch.utils.data import DataLoader

from sentlstm.calibration import reliability_curve
from sentlstm.data import build_dataset, collate_batch, load_csv, synthetic_sentiment
from sentlstm.embeddings import build_embedding_matrix
from sentlstm.models import LSTMClassifier
from sentlstm.predict import Prediction, SentimentClassifier
from sentlstm.tokenizer import Vocabulary, tokenize
from sentlstm.train import Trainer, accuracy, set_seed


def test_tokenize():
    assert tokenize("Hello, World! It's great.") == ["hello", "world", "it's", "great"]


def test_vocab_build_and_encode():
    vocab = Vocabulary().build(["good good great", "bad bad awful"], min_freq=2)
    assert "good" in vocab.stoi
    assert "great" not in vocab.stoi  # below min_freq
    ids = vocab.encode("good unknownword")
    assert ids[0] == vocab.stoi["good"]
    assert ids[1] == vocab.unk_id


def test_collate_pads():
    vocab = Vocabulary().build(["a b c a b c"], min_freq=1)
    ds = build_dataset(["a b c", "a"], [1, 0], vocab)
    ids, lengths, labels = collate_batch([ds[0], ds[1]])
    assert ids.shape[0] == 2
    assert ids.shape[1] == 3
    assert lengths.tolist() == [3, 1]
    assert labels.tolist() == [1, 0]


def test_model_forward_shape():
    vocab = Vocabulary().build(["a b c a b c"], min_freq=1)
    model = LSTMClassifier(len(vocab), embed_dim=16, hidden_dim=8)
    ids = torch.randint(0, len(vocab), (4, 7))
    lengths = torch.tensor([7, 5, 3, 1])
    assert model(ids, lengths).shape == (4, 2)


def test_pretrained_embeddings_wire_in():
    vocab = Vocabulary().build(["a b c a b c"], min_freq=1)
    mat = build_embedding_matrix(vocab, glove_path=None, embed_dim=12)
    assert mat.shape == (len(vocab), 12)
    model = LSTMClassifier(len(vocab), pretrained=mat)
    ids = torch.randint(0, len(vocab), (2, 5))
    lengths = torch.tensor([5, 2])
    assert model(ids, lengths).shape == (2, 2)


def test_vocab_from_stoi_roundtrip():
    vocab = Vocabulary().build(["good good great bad bad"], min_freq=1)
    rebuilt = Vocabulary.from_stoi(vocab.stoi)
    assert rebuilt.stoi == vocab.stoi
    assert rebuilt.itos[rebuilt.stoi["good"]] == "good"
    assert rebuilt.pad_id == 0 and rebuilt.unk_id == 1


def _toy_classifier():
    train_texts, train_labels = synthetic_sentiment(repeats=20)
    vocab = Vocabulary().build(train_texts, min_freq=1)
    model = LSTMClassifier(len(vocab), embed_dim=16, hidden_dim=16)
    return SentimentClassifier(model, vocab, labels=("negative", "positive"))


def test_predict_returns_typed_records():
    clf = _toy_classifier()
    preds = clf.predict(["a great and wonderful film", "an awful boring movie"])
    assert len(preds) == 2
    assert all(isinstance(p, Prediction) for p in preds)
    p = preds[0]
    assert p.label in ("negative", "positive")
    assert 0.0 <= p.confidence <= 1.0
    assert abs(sum(p.probabilities) - 1.0) < 1e-5


def test_predict_proba_shape_and_empty():
    clf = _toy_classifier()
    probs = clf.predict_proba(["good", "bad", "ok"])
    assert probs.shape == (3, 2)
    assert clf.predict_proba([]).shape == (0, 2)


def test_save_load_roundtrip(tmp_path):
    set_seed(0)
    clf = _toy_classifier()
    texts = ["a great wonderful film", "an awful boring movie"]
    before = clf.predict_proba(texts)
    path = tmp_path / "model.pt"
    clf.save(path)
    reloaded = SentimentClassifier.load(path)
    after = reloaded.predict_proba(texts)
    assert np.allclose(before, after, atol=1e-5)
    assert reloaded.labels == ("negative", "positive")


def test_load_csv(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text("text,label\ngreat film,1\nawful film,0\n", encoding="utf-8")
    texts, labels = load_csv(csv_path)
    assert texts == ["great film", "awful film"]
    assert labels == [1, 0]


def test_reliability_curve_perfect_and_ece():
    # Perfectly calibrated: confidence 1.0 and always correct -> ECE 0.
    probs = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    labels = np.array([1, 0, 1])
    curve = reliability_curve(probs, labels, n_bins=5)
    assert curve.ece == 0.0
    assert curve.bin_count.sum() == 3
    # Overconfident and wrong -> positive ECE.
    bad = reliability_curve(np.array([[0.0, 1.0]]), np.array([0]), n_bins=5)
    assert bad.ece > 0.5


def test_trainer_learns_toy_sentiment():
    set_seed(0)
    train_texts, train_labels = synthetic_sentiment(repeats=30)
    test_texts, test_labels = synthetic_sentiment(repeats=6, seed=2)
    vocab = Vocabulary().build(train_texts, min_freq=1)
    train_set = build_dataset(train_texts, train_labels, vocab)
    test_set = build_dataset(test_texts, test_labels, vocab)
    model = LSTMClassifier(len(vocab), embed_dim=32, hidden_dim=32)
    trainer = Trainer(model, lr=1e-3)
    trainer.fit(train_set, test_set, epochs=6, batch_size=32, verbose=False)
    loader = DataLoader(test_set, batch_size=32, collate_fn=collate_batch)
    assert accuracy(model, loader) > 0.9  # toy corpus is easily separable
