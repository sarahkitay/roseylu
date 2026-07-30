"""Pluggable generation backend. Every call site reaches this only after
GuardrailPipeline.evaluate() has returned ALLOW -- see docs/ARCHITECTURE.md's
"design principle carried through the code" section. This module has no
opinion about safety; that's already been decided upstream.

No hosted LLM API is used here, deliberately -- the app's generation path
runs entirely on a model trained on this machine (model/architecture.py +
training/scripts/train_from_scratch.py), never a live call to a third-party
provider. See training/README.md for what "trained from scratch" means here
and its current limits. `StubEchoBackend` is the fallback when no checkpoint
exists yet, so the rest of the system -- pipeline, persona, redirect engine,
CLI, API -- stays fully exercisable without a trained model present.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, message: str) -> str: ...


class StubEchoBackend(ModelBackend):
    """No trained local model checkpoint found. Reflects that clearly
    instead of pretending, rather than silently falling back to a hosted API.
    """

    def generate(self, system_prompt: str, message: str) -> str:
        del system_prompt
        return (
            "[no trained local model checkpoint found -- this is the "
            f"guardrail + persona layer only] You said: \"{message}\". Run "
            "training/scripts/build_corpus.py then "
            "training/scripts/train_from_scratch.py to produce a checkpoint; "
            "app/generation/local_model.py will pick it up automatically."
        )


def _build_default_backend() -> ModelBackend:
    try:
        from app.generation.local_model import LocalTransformerBackend

        return LocalTransformerBackend()
    except FileNotFoundError:
        return StubEchoBackend()


DEFAULT_BACKEND: ModelBackend = _build_default_backend()
