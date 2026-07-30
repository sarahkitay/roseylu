"""Layer 2: semantic classifier.

Catches intent that doesn't hit a literal keyword (e.g. "do people think I'm
ugly" against a keyword list that only has "ugly"). Structured behind the
`Classifier` interface specifically so this stand-in can be swapped for a
real embedding model without touching pipeline.py.

CURRENT IMPLEMENTATION IS A TF-IDF COSINE-SIMILARITY STAND-IN, not a real
embedding model -- no embeddings API key or local model is wired up. See
docs/SAFETY_MODEL.md. Do not reason about recall as if this were a trained
semantic model; it's closer to fuzzy keyword matching than genuine semantics.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.guardrails.lexicon import LEXICON
from app.models.schemas import RiskCategory


class SemanticResult:
    def __init__(self) -> None:
        self.category_scores: dict[RiskCategory, float] = {}
        self.category_evidence: dict[RiskCategory, list[str]] = {}

    def score_for(self, category: RiskCategory) -> float:
        return self.category_scores.get(category, 0.0)

    def evidence_for(self, category: RiskCategory) -> list[str]:
        return self.category_evidence.get(category, [])


class Classifier(ABC):
    """Interface a real embedding-model-backed classifier should implement."""

    @abstractmethod
    def score(self, message: str) -> SemanticResult: ...


class TfidfSemanticClassifier(Classifier):
    def __init__(self) -> None:
        self._phrases: list[str] = []
        self._phrase_categories: list[RiskCategory] = []
        for category, lex in LEXICON.categories.items():
            for phrase in lex.canonical:
                self._phrases.append(phrase)
                self._phrase_categories.append(category)

        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        if self._phrases:
            self._phrase_matrix = self._vectorizer.fit_transform(self._phrases)
        else:  # pragma: no cover - guards against an empty lexicon
            self._phrase_matrix = None

    def score(self, message: str) -> SemanticResult:
        result = SemanticResult()
        if self._phrase_matrix is None:
            return result

        message_vec = self._vectorizer.transform([message])
        similarities = cosine_similarity(message_vec, self._phrase_matrix)[0]

        best_per_category: dict[RiskCategory, tuple[float, str]] = {}
        for sim, phrase, category in zip(similarities, self._phrases, self._phrase_categories):
            current = best_per_category.get(category)
            if current is None or sim > current[0]:
                best_per_category[category] = (sim, phrase)

        for category, (sim, phrase) in best_per_category.items():
            if sim > 0:
                result.category_scores[category] = float(sim)
                result.category_evidence[category] = [f"semantic~{sim:.2f}:'{phrase}'"]

        return result


# Module-level singleton -- the TF-IDF vectorizer is fit once against the
# lexicon at import time, not per-request.
DEFAULT_CLASSIFIER: Classifier = TfidfSemanticClassifier()
