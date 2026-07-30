"""Loads risk_lexicon.yaml once and exposes it as typed Python structures.

Both the keyword filter and the semantic classifier read from this module so
there's exactly one place that parses the YAML.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.models.schemas import RiskCategory

_LEXICON_PATH = Path(__file__).parent / "wordlists" / "risk_lexicon.yaml"

# YAML category keys -> RiskCategory enum
_CATEGORY_KEY_MAP = {
    "self_harm": RiskCategory.SELF_HARM,
    "body_image": RiskCategory.BODY_IMAGE,
    "grooming": RiskCategory.GROOMING,
    "violence": RiskCategory.VIOLENCE,
    "substance": RiskCategory.SUBSTANCE,
    "hate_harassment": RiskCategory.HATE_HARASSMENT,
    "dangerous_activity": RiskCategory.DANGEROUS_ACTIVITY,
    "explicit_sexual": RiskCategory.EXPLICIT_SEXUAL,
}


@dataclass
class CategoryLexicon:
    keywords: list[str] = field(default_factory=list)
    slang: list[str] = field(default_factory=list)
    canonical: list[str] = field(default_factory=list)

    @property
    def all_terms(self) -> list[str]:
        return self.keywords + self.slang


@dataclass
class Lexicon:
    categories: dict[RiskCategory, CategoryLexicon]
    jailbreak_patterns: list[str]


def load_lexicon(path: Path = _LEXICON_PATH) -> Lexicon:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    categories: dict[RiskCategory, CategoryLexicon] = {}
    for key, category in _CATEGORY_KEY_MAP.items():
        entry = raw.get(key, {}) or {}
        categories[category] = CategoryLexicon(
            keywords=entry.get("keywords", []) or [],
            slang=entry.get("slang", []) or [],
            canonical=entry.get("canonical", []) or [],
        )

    jailbreak_patterns = (raw.get("jailbreak", {}) or {}).get("patterns", []) or []
    return Lexicon(categories=categories, jailbreak_patterns=jailbreak_patterns)


LEXICON = load_lexicon()
