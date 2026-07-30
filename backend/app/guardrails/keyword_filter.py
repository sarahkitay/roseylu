"""Layer 1: rule-based keyword/regex filtering.

Fast and literal on purpose -- this layer exists to catch the unambiguous
cases at near-zero latency, not to be clever. Semantic nuance is layer 2/3's
job.
"""
from __future__ import annotations

import re

from app.guardrails.lexicon import LEXICON
from app.models.schemas import RiskCategory


def _term_to_pattern(term: str) -> re.Pattern:
    # word-boundary-ish match; terms are stored lowercase, multi-word phrases
    # match as substrings (word boundaries on a phrase with spaces already
    # constrain it enough in practice).
    escaped = re.escape(term.lower())
    return re.compile(rf"\b{escaped}\b" if " " not in term else escaped)


_CATEGORY_PATTERNS: dict[RiskCategory, list[tuple[str, re.Pattern]]] = {
    category: [(term, _term_to_pattern(term)) for term in lex.all_terms]
    for category, lex in LEXICON.categories.items()
}

_JAILBREAK_PATTERNS: list[tuple[str, re.Pattern]] = [
    (p, _term_to_pattern(p)) for p in LEXICON.jailbreak_patterns
]


class KeywordFilterResult:
    def __init__(self) -> None:
        self.category_hits: dict[RiskCategory, list[str]] = {}
        self.jailbreak_hits: list[str] = []

    def evidence_for(self, category: RiskCategory) -> list[str]:
        return self.category_hits.get(category, [])

    def hit(self, category: RiskCategory) -> bool:
        return category in self.category_hits


def scan(message: str) -> KeywordFilterResult:
    text = message.lower()
    result = KeywordFilterResult()

    for category, patterns in _CATEGORY_PATTERNS.items():
        matches = [term for term, pattern in patterns if pattern.search(text)]
        if matches:
            result.category_hits[category] = matches

    result.jailbreak_hits = [p for p, pattern in _JAILBREAK_PATTERNS if pattern.search(text)]

    return result
