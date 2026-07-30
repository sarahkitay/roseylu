"""Shared data shapes. Every guardrail layer and the response engine speaks
these types -- add a new risk category here first, everything downstream
reads from this enum so nothing can silently support an unlisted category.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.config import AgeTier


class RiskCategory(str, Enum):
    SELF_HARM = "SELF_HARM"
    BODY_IMAGE = "BODY_IMAGE"
    GROOMING = "GROOMING"
    VIOLENCE = "VIOLENCE"
    SUBSTANCE = "SUBSTANCE"
    HATE_HARASSMENT = "HATE_HARASSMENT"
    DANGEROUS_ACTIVITY = "DANGEROUS_ACTIVITY"
    EXPLICIT_SEXUAL = "EXPLICIT_SEXUAL"
    JAILBREAK = "JAILBREAK"


TIER1 = {RiskCategory.SELF_HARM, RiskCategory.BODY_IMAGE, RiskCategory.GROOMING}


class Action(str, Enum):
    ALLOW = "ALLOW"
    REDIRECT = "REDIRECT"
    ESCALATE = "ESCALATE"  # additive: still gets a REDIRECT reply, plus a queue entry


class ChildProfile(BaseModel):
    child_id: str
    age: int = Field(ge=7, le=15)
    persona_name: str = "Rosey"
    persona_traits: list[str] = Field(default_factory=lambda: ["curious", "warm", "encouraging"])

    @property
    def age_tier(self) -> AgeTier:
        from app.config import tier_for_age
        return tier_for_age(self.age)


class CategoryScore(BaseModel):
    category: RiskCategory
    score: float
    matched_layers: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class GuardrailResult(BaseModel):
    action: Action
    top_category: Optional[RiskCategory] = None
    scores: list[CategoryScore] = Field(default_factory=list)

    def score_for(self, category: RiskCategory) -> float:
        for s in self.scores:
            if s.category == category:
                return s.score
        return 0.0


class ChatRequest(BaseModel):
    child: ChildProfile
    message: str


class ChatResponse(BaseModel):
    reply: str
    action: Action
    top_category: Optional[RiskCategory] = None
