import tempfile
from pathlib import Path

from model.tokenizer import BPETokenizer, CharTokenizer, load_tokenizer

SAMPLE_TEXT = "Child: why is the sky blue\nRosey: Sunlight looks white, but it's actually made of all the colors.\n" * 20


def test_char_tokenizer_roundtrip():
    tok = CharTokenizer.from_corpus(SAMPLE_TEXT)
    ids = tok.encode(SAMPLE_TEXT)
    assert tok.decode(ids) == SAMPLE_TEXT


def test_char_tokenizer_save_load(tmp_path=None):
    tok = CharTokenizer.from_corpus(SAMPLE_TEXT)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tok.json"
        tok.save(path)
        loaded = load_tokenizer(path)
    assert isinstance(loaded, CharTokenizer)
    assert loaded.chars == tok.chars


def test_bpe_tokenizer_roundtrip_is_lossless():
    tok, ids = BPETokenizer.train(SAMPLE_TEXT, vocab_size=120)
    assert tok.decode(ids) == SAMPLE_TEXT
    # re-encoding independently should reproduce the same ids, since encode()
    # replays the learned merges in the same order they were trained
    assert tok.encode(SAMPLE_TEXT) == ids


def test_bpe_tokenizer_vocab_size_respected():
    tok, _ = BPETokenizer.train(SAMPLE_TEXT, vocab_size=120)
    assert tok.vocab_size <= 120


def test_bpe_tokenizer_compresses_relative_to_char_level():
    char_tok = CharTokenizer.from_corpus(SAMPLE_TEXT)
    bpe_tok, bpe_ids = BPETokenizer.train(SAMPLE_TEXT, vocab_size=150)
    char_ids = char_tok.encode(SAMPLE_TEXT)
    # the whole point of BPE here -- same text, meaningfully fewer tokens
    assert len(bpe_ids) < len(char_ids)


def test_bpe_tokenizer_save_load_preserves_merges():
    tok, ids = BPETokenizer.train(SAMPLE_TEXT, vocab_size=120)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "tok.json"
        tok.save(path)
        loaded = load_tokenizer(path)
    assert isinstance(loaded, BPETokenizer)
    assert loaded.encode(SAMPLE_TEXT) == ids
    assert loaded.decode(ids) == SAMPLE_TEXT


def test_bpe_tokenizer_handles_unseen_text_gracefully():
    tok, _ = BPETokenizer.train(SAMPLE_TEXT, vocab_size=120)
    # text sharing characters with the training corpus but not seen verbatim
    novel = "why is the sun yellow"
    ids = tok.encode(novel)
    assert tok.decode(ids) == novel
