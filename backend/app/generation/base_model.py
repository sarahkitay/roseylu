"""Pluggable generation backend. Every call site reaches this only after
GuardrailPipeline.evaluate() has returned ALLOW -- see docs/ARCHITECTURE.md's
"design principle carried through the code" section. This module has no
opinion about safety; that's already been decided upstream.

No fine-tuned model exists yet (see docs/BLOCKERS.md re: clinical review
gating what a fine-tune is even trained on). `StubEchoBackend` is the default
so the rest of the system -- pipeline, persona, redirect engine, CLI, API --
is fully exercisable and testable today without any credentials.
`AnthropicModelBackend` is a real implementation behind the same interface,
active automatically once ANTHROPIC_API_KEY is set.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os


class ModelBackend(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, message: str) -> str: ...


class StubEchoBackend(ModelBackend):
    """No live model wired up. Reflects that clearly instead of pretending."""

    def generate(self, system_prompt: str, message: str) -> str:
        del system_prompt
        return (
            "[no generation backend configured -- this is the guardrail + "
            f"persona layer only] You said: \"{message}\". Once a real model "
            "backend is wired in (see app/generation/base_model.py), this is "
            "where its in-persona reply would appear."
        )


class AnthropicModelBackend(ModelBackend):
    def __init__(self, model: str = "claude-sonnet-5") -> None:
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic()
        self._model = model

    def generate(self, system_prompt: str, message: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": message}],
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


def _build_default_backend() -> ModelBackend:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicModelBackend()
        except ImportError:
            pass
    return StubEchoBackend()


DEFAULT_BACKEND: ModelBackend = _build_default_backend()
