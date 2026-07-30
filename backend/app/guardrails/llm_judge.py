"""Layer 3: LLM-as-judge for nuanced cases the first two layers are unsure
about -- specifically, combinations of individually-weak signals that only
look risky together (e.g. a secrecy phrase *and* an age/photo question in
the same message, neither of which alone would clear a tier-2 threshold).

This layer runs on every live request, which is exactly why it does NOT call
a hosted LLM API: the product's generation path is a locally-trained model
with no third-party dependency (see app/generation/local_model.py), and the
safety pipeline holds the same line. `HeuristicJudgeBackend` approximates
the *shape* of an LLM judge's combination-reasoning with hand-written rules.
The honest name for "judge" here is closer to "a second opinion using
different logic than the keyword layer" than "an LLM examined this." A
real learned judge -- e.g. a small classifier trained on
training/data/seed_dataset.jsonl once it's been clinically expanded -- is
the eventual replacement for this class, behind the same `JudgeBackend`
interface, not a call out to someone else's model.
"""
from __future__ import annotations

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


DEFAULT_JUDGE: JudgeBackend = HeuristicJudgeBackend()
