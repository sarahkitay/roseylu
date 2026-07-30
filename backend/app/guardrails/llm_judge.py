"""Layer 3: LLM-as-judge for nuanced cases the first two layers are unsure
about -- specifically, combinations of individually-weak signals that only
look risky together (e.g. a secrecy phrase *and* an age/photo question in
the same message, neither of which alone would clear a tier-2 threshold).

No LLM API key is wired up in this prototype, so the default backend
(`HeuristicJudgeBackend`) approximates the *shape* of that judgment with
hand-written combination rules. `AnthropicJudgeBackend` is a real
implementation behind the same interface -- it activates automatically if
`ANTHROPIC_API_KEY` is set and the `anthropic` package is installed, so
wiring up a real judge later is a config change, not a rewrite. Until then,
treat every judge-layer score as a heuristic guess, not a model's assessment.
"""
from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

from app.models.schemas import RiskCategory


class JudgeResult:
    def __init__(self) -> None:
        self.category_scores: dict[RiskCategory, float] = {}
        self.category_evidence: dict[RiskCategory, list[str]] = {}

    def score_for(self, category: RiskCategory) -> float:
        return self.category_scores.get(category, 0.0)

    def evidence_for(self, category: RiskCategory) -> list[str]:
        return self.category_evidence.get(category, [])


class JudgeBackend(ABC):
    @abstractmethod
    def judge(self, message: str) -> JudgeResult: ...


_SECRECY = re.compile(r"\b(secret|don'?t tell|between us|just us|no one (?:else )?needs to know)\b", re.I)
_AGE_OR_CONTACT = re.compile(
    r"\b(how old|what grade|your (?:address|number|school)|meet (?:up|in person)|"
    r"(?:picture|photo|pic) of (?:you|yourself))\b",
    re.I,
)
_SELF_NEGATIVE = re.compile(r"\b(i'?m (?:ugly|fat|worthless|a failure|stupid|hideous)|nobody likes me|no one likes me)\b", re.I)
_HOPELESSNESS = re.compile(r"\b(what'?s the point|nothing matters|no point in trying|give up on (?:everything|life))\b", re.I)


class HeuristicJudgeBackend(JudgeBackend):
    """Combination-rule stand-in for a real LLM judge. See module docstring."""

    def judge(self, message: str) -> JudgeResult:
        result = JudgeResult()
        text = message

        if _SECRECY.search(text) and _AGE_OR_CONTACT.search(text):
            result.category_scores[RiskCategory.GROOMING] = 0.65
            result.category_evidence[RiskCategory.GROOMING] = [
                "combined secrecy + age/contact framing"
            ]

        if _SELF_NEGATIVE.search(text):
            result.category_scores[RiskCategory.BODY_IMAGE] = max(
                result.category_scores.get(RiskCategory.BODY_IMAGE, 0.0), 0.4
            )
            result.category_evidence.setdefault(RiskCategory.BODY_IMAGE, []).append(
                "self-negative framing"
            )

        if _HOPELESSNESS.search(text):
            result.category_scores[RiskCategory.SELF_HARM] = max(
                result.category_scores.get(RiskCategory.SELF_HARM, 0.0), 0.55
            )
            result.category_evidence.setdefault(RiskCategory.SELF_HARM, []).append(
                "hopelessness framing"
            )

        return result


class AnthropicJudgeBackend(JudgeBackend):
    """Real LLM-as-judge. Only usable once ANTHROPIC_API_KEY is set and the
    `anthropic` package is installed -- not exercised by default anywhere in
    this repo (no key is present in this environment). Wire it in by
    swapping `DEFAULT_JUDGE` in this module once credentials exist.
    """

    _SYSTEM_PROMPT = (
        "You are a safety classifier for a children's AI product. Given a "
        "message from a child, score 0.0-1.0 how strongly it signals each of: "
        "SELF_HARM, BODY_IMAGE, GROOMING, VIOLENCE, SUBSTANCE, HATE_HARASSMENT, "
        "DANGEROUS_ACTIVITY, EXPLICIT_SEXUAL. Respond as compact JSON mapping "
        "category name to score. Err toward higher scores for tier-1 "
        "categories (SELF_HARM, BODY_IMAGE, GROOMING) when ambiguous."
    )

    def __init__(self, model: str = "claude-sonnet-5") -> None:
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic()
        self._model = model

    def judge(self, message: str) -> JudgeResult:
        import json

        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )
        raw_text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        try:
            scores = json.loads(raw_text)
        except json.JSONDecodeError:
            return JudgeResult()

        result = JudgeResult()
        for name, score in scores.items():
            try:
                category = RiskCategory[name]
            except KeyError:
                continue
            result.category_scores[category] = float(score)
            result.category_evidence[category] = ["anthropic-judge"]
        return result


def _build_default_backend() -> JudgeBackend:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicJudgeBackend()
        except ImportError:
            pass
    return HeuristicJudgeBackend()


DEFAULT_JUDGE: JudgeBackend = _build_default_backend()
