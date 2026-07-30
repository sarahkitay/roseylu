"""Central config: age tiers and guardrail thresholds.

Kept deliberately small and dependency-free so every other module can import
it without pulling in the rest of the app.
"""
from __future__ import annotations

from enum import Enum


class AgeTier(str, Enum):
    """Developmental tier. See docs/SAFETY_MODEL.md for the rationale."""

    EARLY = "early"    # 7-9
    MIDDLE = "middle"  # 10-12
    TEEN = "teen"      # 13-15


def tier_for_age(age: int) -> AgeTier:
    if age <= 9:
        return AgeTier.EARLY
    if age <= 12:
        return AgeTier.MIDDLE
    return AgeTier.TEEN


def tier_blend_weight(age: int, exact_birthday_age: int) -> float:
    """How far a child has drifted toward the *next* tier within their
    current one, as a 0..1 weight. Used by persona_engine to blend tone
    gradually instead of snapping at a tier boundary.

    `age` is whole years; without a finer-grained signal (e.g. months since
    last birthday) this returns 0.0 — the hook exists so persona_engine has
    a stable interface once account-age-in-days is wired up.
    """
    del age, exact_birthday_age  # placeholder until real signal exists
    return 0.0


# Guardrail decision thresholds. Tier-1 categories bypass thresholds
# entirely (any signal routes to REDIRECT) -- these only gate tier-2.
TIER1_CATEGORIES = {"SELF_HARM", "BODY_IMAGE", "GROOMING"}

KEYWORD_SCORE = 1.0        # a keyword hit is a hard signal
SEMANTIC_REDIRECT_THRESHOLD = 0.42
JUDGE_REDIRECT_THRESHOLD = 0.5
ESCALATE_THRESHOLD = 0.95  # combined score required to also log to review queue
# (deliberately above a single keyword hit's ~0.81 -- see
# pipeline.py::_should_escalate for why keyword-only escalation is
# category-specific rather than threshold-driven)
