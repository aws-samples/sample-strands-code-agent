"""OKF bundle navigation library for CodeAgent.

Provides a read-optimized API for traversing Open Knowledge Format bundles.

Usage:
    from strands_code_agent.knowledge import OKFBundle, Concept

    bundle = OKFBundle("path/to/bundle")
    concept = bundle["tables/orders"]
    concept.links           # outgoing links
    bundle.backlinks("tables/orders")  # who links here
    bundle.expand("tables/orders", hops=2)  # neighborhood
    bundle.context("tables/orders", max_chars=8000)  # budget-aware expansion

Custom search backend:
    from strands_code_agent.knowledge import OKFBundle, SearchIndex

    class MyEmbeddingSearch(SearchIndex):
        def build(self, concepts): ...
        def query(self, query, top_k=10): ...

    bundle = OKFBundle("path/to/bundle", search_index=MyEmbeddingSearch())
"""

from strands_code_agent.knowledge.concept import Concept
from strands_code_agent.knowledge.bundle import OKFBundle
from strands_code_agent.knowledge.search import SearchIndex, KeywordSearchIndex
from strands_code_agent.knowledge.pdf_to_okf_bundle import pdf_to_okf_bundle

__all__ = ["OKFBundle", "Concept", "SearchIndex", "KeywordSearchIndex", "pdf_to_okf_bundle"]
