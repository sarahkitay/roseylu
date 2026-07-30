"""Character-level tokenizer, written from scratch -- no `tiktoken`,
`sentencepiece`, or `tokenizers` dependency.

Char-level is the simplest tokenization scheme that's genuinely "from
scratch" with zero external vocabulary or pretrained merge rules: the vocab
is just every unique character seen in the training corpus. The tradeoff is
efficiency and headroom for fluency (a real product-grade model should move
to a trained BPE vocab, which needs meaningfully more data than this
prototype's corpus to pay for itself) -- see training/README.md.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
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
        path.write_text(json.dumps({"chars": self.chars}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["chars"])
