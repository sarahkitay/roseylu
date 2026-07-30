"""Runs all three guardrail layers in parallel and decides the Action.

See docs/SAFETY_MODEL.md for the full spec this implements. Key rule: tier-1
categories (SELF_HARM, BODY_IMAGE, GROOMING) bypass thresholds entirely --
any signal at all routes to REDIRECT. Tier-2 categories need to clear a
threshold. ESCALATE is additive on top of REDIRECT, never a different reply.
"""
from __future__ import annotations

from app.config import ESCALATE_THRESHOLD, KEYWORD_SCORE, SEMANTIC_REDIRECT_THRESHOLD
from app.guardrails import keyword_filter
from app.guardrails.embedding_classifier import DEFAULT_CLASSIFIER
from app.guardrails.llm_judge import DEFAULT_JUDGE
from app.models.schemas import (
    TIER1,
    Action,
    CategoryScore,
    GuardrailResult,
    RiskCategory,
)

# Layer contribution weights when combining signals into one 0..1 score.
_KEYWORD_WEIGHT = 0.9
_SEMANTIC_WEIGHT = 0.5
_JUDGE_WEIGHT = 0.6


class GuardrailPipeline:
    def evaluate(self, message: str) -> GuardrailResult:
        kw = keyword_filter.scan(message)
        semantic = DEFAULT_CLASSIFIER.score(message)
        judge = DEFAULT_JUDGE.judge(message)

        scores: dict[RiskCategory, CategoryScore] = {}

        for category in RiskCategory:
            if category == RiskCategory.JAILBREAK:
                continue  # handled separately below, it's a framing not a harm category

            combined = 0.0
            layers: list[str] = []
            evidence: list[str] = []

            if kw.hit(category):
                combined += KEYWORD_SCORE * _KEYWORD_WEIGHT
                layers.append("keyword")
                evidence.extend(kw.evidence_for(category))

            sem_score = semantic.score_for(category)
            if sem_score >= SEMANTIC_REDIRECT_THRESHOLD:
                combined += sem_score * _SEMANTIC_WEIGHT
                layers.append("semantic")
                evidence.extend(semantic.evidence_for(category))

            judge_score = judge.score_for(category)
            if judge_score > 0:
                combined += judge_score * _JUDGE_WEIGHT
                layers.append("judge")
                evidence.extend(judge.evidence_for(category))

            if layers:
                scores[category] = CategoryScore(
                    category=category,
                    score=min(1.0, combined),
                    matched_layers=layers,
                    evidence=evidence,
                )

        is_jailbreak_framed = bool(kw.jailbreak_hits)
        if is_jailbreak_framed:
            scores[RiskCategory.JAILBREAK] = CategoryScore(
                category=RiskCategory.JAILBREAK,
                score=1.0,
                matched_layers=["keyword"],
                evidence=kw.jailbreak_hits,
            )

        action, top_category = self._decide(scores)

        return GuardrailResult(
            action=action,
            top_category=top_category,
            scores=list(scores.values()),
        )

    @staticmethod
    def _decide(
        scores: dict[RiskCategory, CategoryScore],
    ) -> tuple[Action, RiskCategory | None]:
        tier1_hits = {c: s for c, s in scores.items() if c in TIER1 and s.score > 0}
        if tier1_hits:
            top = max(tier1_hits.values(), key=lambda s: s.score)
            action = Action.ESCALATE if _should_escalate(top) else Action.REDIRECT
            return action, top.category

        tier2_hits = {
            c: s
            for c, s in scores.items()
            if c not in TIER1 and c != RiskCategory.JAILBREAK and s.score >= SEMANTIC_REDIRECT_THRESHOLD
        }
        if tier2_hits:
            top = max(tier2_hits.values(), key=lambda s: s.score)
            return Action.REDIRECT, top.category

        if RiskCategory.JAILBREAK in scores:
            return Action.REDIRECT, RiskCategory.JAILBREAK

        return Action.ALLOW, None



# Escalation rules are deliberately category-specific rather than one
# global threshold -- see docs/SAFETY_MODEL.md's "escalation is the
# exception, not the rule" framing.
#
# SELF_HARM: any explicit keyword hit escalates on its own. The cost of a
# false-positive escalation (a specialist reads one extra queue entry) is
# far lower than a false negative here.
#
# GROOMING: escalates when two independent layers agree, because the
# layers that co-fire here (e.g. keyword secrecy phrase + judge-detected
# age/contact combo) represent genuinely distinct evidence, not the same
# fact restated -- see llm_judge.py's secrecy+age/contact rule.
#
# BODY_IMAGE never auto-escalates on its own text-pattern signal, at any
# combination or score. In practice its keyword and semantic layers
# routinely co-fire on the exact same phrase (a keyword hit's own wording is
# usually similar enough to also trip the semantic layer), so neither
# "2 layers agree" nor a high combined score is independent evidence here
# the way it is for GROOMING -- it's the same signal counted twice, and it's
# easy to accidentally cross even a 0.95 threshold that way. Routine
# body-image curiosity ("rate my face", "I hate my body") is common and
# mostly benign (docs/PRODUCT_VISION.md Q11); auto-escalating it would flood
# the review queue and dull it for cases that actually need a specialist's
# eyes. If a body-image message is *also* a self-harm signal, the SELF_HARM
# category scores independently and wins top_category on its own merits
# (see GuardrailPipeline._decide) -- that path already covers the genuinely
# urgent case without BODY_IMAGE needing its own escalation trigger.
_KEYWORD_ALONE_ESCALATES = {RiskCategory.SELF_HARM}
_MULTI_LAYER_ESCALATES = {RiskCategory.GROOMING}
_NEVER_AUTO_ESCALATES = {RiskCategory.BODY_IMAGE}


def _should_escalate(top: CategoryScore) -> bool:
    if top.category in _NEVER_AUTO_ESCALATES:
        return False
    if top.category in _KEYWORD_ALONE_ESCALATES and "keyword" in top.matched_layers:
        return True
    if top.category in _MULTI_LAYER_ESCALATES and len(top.matched_layers) >= 2:
        return True
    return top.score >= ESCALATE_THRESHOLD


DEFAULT_PIPELINE = GuardrailPipeline()
