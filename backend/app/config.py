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

# Two DISTINCT thresholds on two DIFFERENT scales -- these used to be the
# same constant, which was a real bug: SEMANTIC_MATCH_THRESHOLD gates the
# RAW cosine similarity score (0..1, direct from the classifier), while
# TIER2_REDIRECT_THRESHOLD gates the COMBINED score after layer weighting
# (pipeline.py multiplies semantic evidence by 0.5 before adding it in). A
# single perfect semantic match (raw 1.0) can only ever contribute 0.5 to
# the combined score -- so using one constant for both meant that any value
# above 0.5 made semantic-only tier-2 detection mathematically impossible,
# which is exactly what happened when this was raised to 0.55 to fix a
# false-positive problem (see SEMANTIC_MATCH_THRESHOLD's comment) and it
# silently broke real SUBSTANCE/HATE_HARASSMENT/DANGEROUS_ACTIVITY recall
# that had no other layer to fall back on. Caught by the full seed-dataset
# calibration (training/scripts/evaluate.py) -- rerun it after touching
# either of these.
SEMANTIC_MATCH_THRESHOLD = 0.55
# Raised from 0.42: two unrelated false positives (a fractions homework
# question, a templated addition answer) scored 0.44-0.45 against grooming
# canonical phrases on shared sentence scaffolding alone -- right in the
# same range as genuine positives (real tier-2 matches scored 0.94-1.0, but
# real tier-1 matches can be as low as ~0.32, so this can't be pushed much
# higher without costing tier-1 recall this layer contributes to). The
# TF-IDF stand-in's cosine similarity isn't well-calibrated enough on such
# short reference phrases for a fine-grained threshold to cleanly separate
# signal from noise -- this leans on the keyword/judge layers to cover real
# cases this layer now misses alone.
TIER2_REDIRECT_THRESHOLD = 0.42  # combined/weighted score, unrelated scale to the above
JUDGE_REDIRECT_THRESHOLD = 0.5
ESCALATE_THRESHOLD = 0.95  # combined score required to also log to review queue
# (deliberately above a single keyword hit's ~0.81 -- see
# pipeline.py::_should_escalate for why keyword-only escalation is
# category-specific rather than threshold-driven)
