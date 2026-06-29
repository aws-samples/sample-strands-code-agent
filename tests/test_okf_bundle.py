"""Tests for strands_code_agent.knowledge OKF bundle navigation library."""

import pytest
from pathlib import Path

from strands_code_agent.knowledge import OKFBundle, Concept


@pytest.fixture
def bundle_dir(tmp_path):
    """Create a minimal OKF bundle for testing."""
    # datasets/sales.md
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "sales.md").write_text(
        "---\n"
        "type: Dataset\n"
        "title: Sales\n"
        "description: All sales-related tables.\n"
        "tags: [sales]\n"
        "timestamp: 2026-05-28T00:00:00Z\n"
        "---\n\n"
        "Contains [orders](/tables/orders.md) and [customers](/tables/customers.md).\n"
    )

    # tables/orders.md
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "orders.md").write_text(
        "---\n"
        "type: Table\n"
        "title: Orders\n"
        "description: One row per completed order.\n"
        "tags: [sales, orders]\n"
        "resource: https://example.com/orders\n"
        "---\n\n"
        "# Schema\n"
        "| Column | Type |\n"
        "|--------|------|\n"
        "| order_id | STRING |\n"
        "| customer_id | STRING |\n\n"
        "Joins with [customers](customers.md).\n"
        "Part of [sales dataset](/datasets/sales.md).\n"
    )

    # tables/customers.md
    (tmp_path / "tables" / "customers.md").write_text(
        "---\n"
        "type: Table\n"
        "title: Customers\n"
        "description: Master customer dimension.\n"
        "tags: [sales, customers]\n"
        "---\n\n"
        "Referenced by [orders](orders.md).\n"
    )

    # playbooks/freshness.md
    (tmp_path / "playbooks").mkdir()
    (tmp_path / "playbooks" / "freshness.md").write_text(
        "---\n"
        "type: Playbook\n"
        "title: Freshness Alert\n"
        "description: Steps for data freshness issues.\n"
        "tags: [oncall]\n"
        "---\n\n"
        "Check the [orders table](/tables/orders.md).\n"
    )

    # index.md (should be skipped)
    (tmp_path / "index.md").write_text("# Bundle Index\n* [Sales](datasets/sales.md)\n")

    return tmp_path


@pytest.fixture
def bundle(bundle_dir):
    return OKFBundle(bundle_dir)


class TestLoading:
    def test_loads_concepts(self, bundle):
        assert len(bundle) == 4

    def test_skips_index(self, bundle):
        assert "index" not in bundle

    def test_concept_ids(self, bundle):
        assert "tables/orders" in bundle
        assert "tables/customers" in bundle
        assert "datasets/sales" in bundle
        assert "playbooks/freshness" in bundle


class TestConceptAccess:
    def test_getitem(self, bundle):
        c = bundle["tables/orders"]
        assert isinstance(c, Concept)
        assert c.title == "Orders"
        assert c.type == "Table"
        assert c.description == "One row per completed order."
        assert "sales" in c.tags
        assert c.resource == "https://example.com/orders"

    def test_getitem_missing(self, bundle):
        with pytest.raises(KeyError):
            bundle["nonexistent"]

    def test_iter(self, bundle):
        concepts = list(bundle)
        assert len(concepts) == 4
        assert all(isinstance(c, Concept) for c in concepts)


class TestLinks:
    def test_outgoing_links(self, bundle):
        c = bundle["tables/orders"]
        assert "tables/customers" in c.links
        assert "datasets/sales" in c.links

    def test_outgoing_links_absolute(self, bundle):
        c = bundle["datasets/sales"]
        assert "tables/orders" in c.links
        assert "tables/customers" in c.links

    def test_relative_links(self, bundle):
        c = bundle["tables/customers"]
        assert "tables/orders" in c.links


class TestSearch:
    def test_search_title(self, bundle):
        result = bundle._search("Orders")
        assert any(c.id == "tables/orders" for c in result)

    def test_search_body(self, bundle):
        result = bundle._search("Schema")
        assert any(c.id == "tables/orders" for c in result)

    def test_search_case_insensitive(self, bundle):
        result = bundle._search("freshness")
        assert any(c.id == "playbooks/freshness" for c in result)

    def test_search_no_results(self, bundle):
        assert bundle._search("xyznonexistent") == []


class TestToc:
    def test_toc_shows_roots(self, bundle):
        toc = bundle.toc()
        # Should show bundle size
        assert "4 concepts" in toc

    def test_toc_shows_concepts_with_children(self, bundle):
        toc = bundle.toc()
        # datasets/sales has links to orders and customers
        assert "sales" in toc.lower() or "concepts" in toc.lower()


class TestChildren:
    def test_children_of_concept_with_links(self, bundle):
        result = bundle.children("datasets/sales")
        assert "tables/orders" in result or "orders" in result.lower()
        assert "tables/customers" in result or "customers" in result.lower()

    def test_children_of_leaf(self, bundle):
        result = bundle.children("playbooks/freshness")
        # freshness links to orders, so it has "children" (links)
        assert "orders" in result.lower()

    def test_children_nonexistent(self, bundle):
        result = bundle.children("nonexistent")
        assert "not found" in result


class TestConcept:
    def test_summary(self, bundle):
        c = bundle["tables/orders"]
        s = c.summary()
        assert "Orders" in s
        assert "One row per completed order" in s

    def test_char_count(self, bundle):
        c = bundle["tables/orders"]
        assert c.char_count > 0

    def test_repr(self, bundle):
        c = bundle["tables/orders"]
        assert "tables/orders" in repr(c)
