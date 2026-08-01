"""Tokenizers, written from scratch -- no `tiktoken`, `sentencepiece`, or
`tokenizers` dependency. Two implementations behind a shared interface
(`vocab_size`, `encode`, `decode`, `save`/`load`):

- `CharTokenizer` -- vocab is just the unique characters in the corpus.
  Simplest possible "from scratch" tokenizer, but low sample-efficiency: the
  model spends a lot of its limited capacity re-deriving how English words
  are spelled, character by character, instead of learning at the level of
  words/concepts.
- `BPETokenizer` -- a from-scratch byte-pair-encoding trainer (the same
  algorithm behind GPT-2's tokenizer, implemented here without that
  dependency): start from single characters, repeatedly merge the most
  frequent adjacent pair into a new token, for a fixed number of merges.
  This is the standard fix for the char-level model's sample-efficiency
  problem, and was worth doing as soon as it became the bottleneck -- see
  training/README.md for the before/after.

  Unlike naive word-level tokenization (split on whitespace, one id per
  word), BPE never truly hits an out-of-vocabulary word: an unseen word just
  falls back to smaller learned subword pieces, and in the worst case, all
  the way down to individual characters, which are always in vocab. That
  property matters specifically because this app does online learning on
  live conversation (see backend/app/generation/online_trainer.py) -- a
  closed word-level vocabulary would silently drop every novel word a child
  or the developer typed, which would make online learning far less useful.

`load_tokenizer(path)` reads either kind back from disk based on a stored
`type` field, so callers don't need to know in advance which one a given
checkpoint was trained with.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class CharTokenizer:
    kind = "char"

    def __init__(self, chars: list[str]) -> None:
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    @classmethod
    def from_corpus(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls(chars)

    def encode(self, text: str) -> list[int]:
        # unknown characters (not seen during training) are dropped rather
        # than crashing -- acceptable for this prototype's generation path,
        # where the input has already passed the guardrail pipeline.
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": self.kind, "chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["chars"])


def _merge_pair(ids: list[int], a: int, b: int, new_id: int) -> list[int]:
    """Single left-to-right pass replacing every adjacent (a, b) with new_id."""
    out = []
    i = 0
    n = len(ids)
    while i < n:
        if i < n - 1 and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    kind = "bpe"

    def __init__(self, chars: list[str], merges: list[tuple[int, int, int]]) -> None:
        """`merges` is an ORDER-SENSITIVE list of (a, b, new_id) triples --
        encode() replays them in this exact order, which is what makes
        encoding deterministic and consistent with how the vocab was built.
        """
        self.chars = chars
        self.char_stoi = {ch: i for i, ch in enumerate(chars)}
        self.merges = merges

        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        for a, b, new_id in merges:
            self.itos[new_id] = self.itos[a] + self.itos[b]

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.merges)

    @classmethod
    def train(cls, text: str, vocab_size: int, min_pair_count: int = 2, log=None) -> tuple["BPETokenizer", list[int]]:
        """Returns (tokenizer, ids_for_text) -- the caller gets the training
        text's token ids for free, since training already computes them;
        re-encoding the same text afterward would just redo that work.
        """
        chars = sorted(set(text))
        char_stoi = {ch: i for i, ch in enumerate(chars)}
        ids = [char_stoi[c] for c in text]

        num_merges = max(0, vocab_size - len(chars))
        merges: list[tuple[int, int, int]] = []
        next_id = len(chars)

        for step in range(num_merges):
            if len(ids) < 2:
                break
            pair_counts = Counter(zip(ids, ids[1:]))
            best_pair, count = pair_counts.most_common(1)[0]
            if count < min_pair_count:
                break  # no more pairs worth merging
            a, b = best_pair
            ids = _merge_pair(ids, a, b, next_id)
            merges.append((a, b, next_id))
            next_id += 1
            if log and (step % 50 == 0 or step == num_merges - 1):
                log(f"bpe merge {step + 1}/{num_merges}: pair {best_pair} x{count} -> token {next_id - 1}, sequence now {len(ids):,} tokens")

        return cls(chars, merges), ids

    def encode(self, text: str) -> list[int]:
        ids = [self.char_stoi[c] for c in text if c in self.char_stoi]
        for a, b, new_id in self.merges:
            if len(ids) < 2:
                break
            ids = _merge_pair(ids, a, b, new_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"type": self.kind, "chars": self.chars, "merges": self.merges}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        merges = [tuple(m) for m in data["merges"]]
        return cls(data["chars"], merges)


def load_tokenizer(path: Path):
    """Reads back whichever tokenizer type a checkpoint was trained with."""
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("type", "char")  # older checkpoints predate the type field
    if kind == "bpe":
        return BPETokenizer.load(path)
    return CharTokenizer.load(path)
