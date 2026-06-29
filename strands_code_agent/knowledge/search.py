"""Search backends for OKF bundles.

Provides a SearchIndex ABC and a default TF-IDF implementation.
Custom backends (e.g., embedding-based) can be provided by subclassing SearchIndex.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strands_code_agent.knowledge.concept import Concept


class SearchIndex(ABC):
    """Abstract search index for an OKF bundle.

    Subclass this to provide custom search backends (e.g., fastembed, OpenAI embeddings).
    """

    @abstractmethod
    def build(self, concepts: list[Concept]) -> None:
        """Build/rebuild the index from a list of concepts."""
        ...

    @abstractmethod
    def query(self, query: str, top_k: int = 10) -> list[str]:
        """Return concept IDs ranked by relevance to the query."""
        ...


class KeywordSearchIndex(SearchIndex):
    """Keyword search with AND logic and OR fallback.

    Tries AND first (all words must match). If no results, falls back to
    OR (ranked by number of matching words). Fast and predictable.
    """

    def __init__(self) -> None:
        self._concepts: list[Concept] = []
        self._texts: list[str] = []

    def build(self, concepts: list[Concept]) -> None:
        self._concepts = concepts
        # Title repeated to boost its weight in frequency-based ranking
        self._texts = [f"{c.title} {c.title} {c.description} {c.body}".lower() for c in concepts]

    def query(self, query: str, top_k: int = 10) -> list[str]:
        words = [w.lower() for w in query.split() if len(w) >= 2]
        if not words:
            return []
        # Try AND first
        scored: list[tuple[int, str]] = []
        for text, concept in zip(self._texts, self._concepts):
            if all(w in text for w in words):
                hits = sum(text.count(w) for w in words)
                scored.append((hits, concept.id))
        # Fallback to OR if AND yields nothing
        if not scored:
            for text, concept in zip(self._texts, self._concepts):
                hits = sum(text.count(w) for w in words if w in text)
                if hits > 0:
                    scored.append((hits, concept.id))
        scored.sort(key=lambda x: -x[0])
        return [cid for _, cid in scored[:top_k]]
