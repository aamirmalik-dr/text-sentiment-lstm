import torch
from torch.utils.data import DataLoader

from sentlstm.data import build_dataset, collate_batch, synthetic_sentiment
from sentlstm.embeddings import build_embedding_matrix
from sentlstm.models import LSTMClassifier
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
