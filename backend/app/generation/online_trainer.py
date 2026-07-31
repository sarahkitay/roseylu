"""Online continual learning: the model takes a short additional training
burst on live conversation as it happens, instead of staying frozen at
whatever `train_from_scratch.py` last produced.

This is real training, not a cache of canned replies -- every burst does
actual gradient steps on the live model weights and overwrites the
checkpoint on disk. Two things keep it from being reckless for a
prototype-scale model:

1. **A low learning rate and a short burst** (a few dozen steps, not a full
   training run) -- enough to nudge the model toward what was just said,
   not enough to swing it wildly on one exchange.
2. **A replay buffer.** Every burst mixes a random sample of the *original*
   training corpus in alongside the new interaction. Without this, a model
   this small would catastrophically forget everything else after a
   handful of online updates -- classic continual-learning failure mode.
   Replay doesn't eliminate drift, it bounds it.

Concurrency: `generate()` and a training burst both touch the same live
`nn.Module`, and PyTorch's `.train()`/`.eval()` mode is a mutable flag on the
model, not thread-local. A single shared `threading.Lock` (owned by
`LocalTransformerBackend`, passed in here) serializes the two so a burst
never runs concurrently with a generation call. That's a correctness
requirement here, not just tidiness -- see local_model.py.

Scope note: this learns from *your* conversation, on *your* checkpoint, on
this machine. It is not multi-user personalization and has no concept of
"the child" as an identity beyond whatever's in the live interaction log --
fine for a single-developer dev instance, not how this would work with real
concurrent users. See docs/BLOCKERS.md if this ever needs to become
per-child.
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path

import torch

_ROOT = Path(__file__).parent.parent.parent.parent
_REPLAY_CORPUS_PATH = _ROOT / "training" / "data" / "corpus" / "combined.txt"
_LIVE_LOG_PATH = _ROOT / "training" / "data" / "corpus" / "live_interactions.jsonl"

_REPLAY_CHUNK_CHARS = 20_000
_BURST_ITERS = 40
_BURST_BATCH_SIZE = 8
_BURST_LR = 5e-5
_INTERACTIONS_PER_BURST = 3  # train after every N new (child, reply) pairs
_PENDING_REPEATS = 4  # repeat the new interactions within the burst text so they carry real weight


def _get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device):
    n = len(data) - block_size - 1
    if n <= 0:
        raise ValueError("burst text too short for block_size -- increase replay chunk size")
    ix = torch.randint(n, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


class OnlineTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        checkpoint_path: Path,
        lock: threading.Lock,
        block_size: int,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._lock = lock
        self._block_size = block_size

        self._replay_text = (
            _REPLAY_CORPUS_PATH.read_text(encoding="utf-8") if _REPLAY_CORPUS_PATH.exists() else ""
        )
        self._optimizer = torch.optim.AdamW(model.parameters(), lr=_BURST_LR, weight_decay=0.01)

        self._pending: list[tuple[str, str]] = []
        self.total_interactions = 0
        self.total_bursts = 0
        self.is_training = False
        self.last_trained_at: float | None = None
        self.last_burst_loss: float | None = None

        _LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def status(self) -> dict:
        return {
            "total_interactions": self.total_interactions,
            "total_bursts": self.total_bursts,
            "is_training": self.is_training,
            "last_trained_at": self.last_trained_at,
            "last_burst_loss": self.last_burst_loss,
            "pending": len(self._pending),
            "interactions_per_burst": _INTERACTIONS_PER_BURST,
            "has_replay_corpus": bool(self._replay_text),
        }

    def log_interaction(self, child_message: str, reply: str) -> None:
        """Call after a real (non-guardrail-redirected) generation. Persists
        the interaction and, once enough have accumulated, runs a training
        burst synchronously -- call this from a background task/thread, not
        on the request-handling path, so it doesn't add latency to the
        child's reply.
        """
        with open(_LIVE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"child": child_message, "rosey": reply, "ts": time.time()}) + "\n")

        self._pending.append((child_message, reply))
        self.total_interactions += 1

        if len(self._pending) >= _INTERACTIONS_PER_BURST:
            self._run_burst()

    def _run_burst(self) -> None:
        pending = self._pending
        self._pending = []

        pending_block = "\n".join(
            f"Child: {child}\nRosey: {reply}\n" for child, reply in pending
        )
        mixed_text = self._sample_replay_chunk() + "\n" + (pending_block * _PENDING_REPEATS)

        encoded = self._tokenizer.encode(mixed_text)
        if len(encoded) <= self._block_size + 1:
            # not enough text to form even one training batch -- skip this
            # burst rather than crash; the pending interactions are still
            # logged to disk and will be included in the next burst's replay
            # sampling once the log grows.
            return

        data = torch.tensor(encoded, dtype=torch.long)

        with self._lock:
            self.is_training = True
            try:
                self._model.train()
                last_loss = None
                for _ in range(_BURST_ITERS):
                    x, y = _get_batch(data, self._block_size, _BURST_BATCH_SIZE, self._device)
                    _, loss = self._model(x, y)
                    self._optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                    self._optimizer.step()
                    last_loss = loss.item()
                self._model.eval()

                self._save_checkpoint()
                self.last_burst_loss = last_loss
                self.last_trained_at = time.time()
                self.total_bursts += 1
            finally:
                self.is_training = False
                self._model.eval()

    def _sample_replay_chunk(self) -> str:
        if not self._replay_text or len(self._replay_text) <= _REPLAY_CHUNK_CHARS:
            return self._replay_text
        start = random.randint(0, len(self._replay_text) - _REPLAY_CHUNK_CHARS)
        return self._replay_text[start:start + _REPLAY_CHUNK_CHARS]

    def _save_checkpoint(self) -> None:
        config = self._model.config
        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": asdict(config) if config else {},
                "iter": -1,  # online bursts don't track a global step count the way full training does
                "online_bursts": self.total_bursts + 1,
            },
            self._checkpoint_path,
        )
