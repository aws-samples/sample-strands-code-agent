"""OKF Bundle navigator — read-optimized API for traversing knowledge bundles."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from strands_code_agent.knowledge.concept import Concept
from strands_code_agent.knowledge.search import SearchIndex, KeywordSearchIndex

_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+\.md)(?:#[^)]*)?\)")
_RESERVED = {"index.md", "log.md"}


def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter + markdown body from text."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    fm = yaml.safe_load(parts[1]) or {}
    body = parts[2].lstrip("\n")
    return fm, body


def _extract_links(body: str, doc_dir: Path, bundle_root: Path) -> list[str]:
    """Extract internal concept IDs from markdown links in the body."""
    out: list[str] = []
    seen: set[str] = set()
    root_resolved = bundle_root.resolve()
    for m in _LINK_RE.finditer(body):
        target = m.group(2)
        if "://" in target:
            continue
        if target.startswith("/"):
            resolved = bundle_root / target.lstrip("/")
        else:
            resolved = (doc_dir / target).resolve()
        try:
            rel = resolved.resolve().relative_to(root_resolved)
        except ValueError:
            continue
        concept_id = rel.with_suffix("").as_posix()
        if concept_id not in seen and Path(rel.name).stem not in ("index", "log"):
            seen.add(concept_id)
            out.append(concept_id)
    return out


class OKFBundle:
    """A navigable knowledge bundle. Use these methods in order:

    1. find(query) → search by keywords, returns top matches
    2. read(concept_id) → read a specific concept's full content
    3. children(concept_id) → list subsections of a concept
    4. toc() → show top-level sections (use only if find() fails)

    Typical flow: find("topic") → pick best match → read(id)
    """

    def __init__(self, root: str | Path, search_index: SearchIndex | None = None) -> None:
        self.root = Path(root).resolve()
        self._concepts: dict[str, Concept] | None = None
        self._backlinks_idx: dict[str, list[str]] | None = None
        self._search_index = search_index or KeywordSearchIndex()

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._concepts is not None:
            return
        self._concepts = {}
        self._backlinks_idx = defaultdict(list)
        for md_path in sorted(self.root.rglob("*.md")):
            if md_path.name in _RESERVED:
                continue
            rel = md_path.relative_to(self.root).with_suffix("")
            concept_id = rel.as_posix()
            text = md_path.read_text(encoding="utf-8")
            fm, body = _parse_frontmatter(text)
            tags = fm.get("tags") or []
            if not isinstance(tags, list):
                tags = [str(tags)]
            links = _extract_links(body, md_path.parent, self.root)
            self._concepts[concept_id] = Concept(
                id=concept_id,
                type=str(fm.get("type") or "Unknown"),
                title=str(fm.get("title") or concept_id.split("/")[-1]),
                description=str(fm.get("description") or ""),
                tags=[str(t) for t in tags],
                resource=fm.get("resource"),
                timestamp=fm.get("timestamp"),
                body=body,
                metadata=fm,
                links=links,
            )
        # Build backlink index
        for cid, concept in self._concepts.items():
            for target in concept.links:
                if target in self._concepts:
                    self._backlinks_idx[target].append(cid)
        # Build search index
        self._search_index.build(list(self._concepts.values()))

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def __getitem__(self, concept_id: str) -> Concept:
        """Get a concept by ID. Raises KeyError if not found."""
        self._ensure_loaded()
        return self._concepts[concept_id]

    def __contains__(self, concept_id: str) -> bool:
        self._ensure_loaded()
        return concept_id in self._concepts

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._concepts)

    def __iter__(self):
        self._ensure_loaded()
        return iter(self._concepts.values())

    @property
    def concepts(self) -> dict[str, Concept]:
        """All concepts keyed by ID."""
        self._ensure_loaded()
        return self._concepts

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(self, query: str, top_k: int = 10) -> list[Concept]:
        """Search for concepts relevant to the query using the search index.

        Uses TF-IDF similarity by default. A custom search backend can be
        provided via the search_index parameter in the constructor.
        """
        self._ensure_loaded()
        ids = self._search_index.query(query, top_k=top_k)
        return [self._concepts[cid] for cid in ids if cid in self._concepts]

    # ------------------------------------------------------------------
    # Structure
    # ------------------------------------------------------------------

    def toc(self) -> str:
        """Return the top-level sections of the bundle.

        Shows only root-level concepts (those with children/subsections).
        Use children(concept_id) to drill into any section.
        """
        self._ensure_loaded()
        roots = [c for c in self._concepts.values() if c.type == "Service"]
        if not roots:
            roots = sorted(
                [c for c in self._concepts.values() if c.links],
                key=lambda c: -len(c.links),
            )[:20]
        lines = [f"Bundle: {len(self._concepts)} concepts total\n",
                 "Top-level sections (use children(id) to expand):\n"]
        for c in sorted(roots, key=lambda c: c.title):
            n_children = len(c.links)
            lines.append(f"- {c.id}: {c.title} ({n_children} subsections)")
        return "\n".join(lines)

    def children(self, concept_id: str) -> str:
        """List the direct children/subsections of a concept.

        Use this to drill into the hierarchy: start with toc() for top-level
        sections, then children(id) to see what's inside each one.

        Args:
            concept_id: The parent concept to expand.

        Returns:
            Formatted list of child concepts with their titles and types.
        """
        self._ensure_loaded()
        concept = self._concepts.get(concept_id)
        if concept is None:
            return f"Concept '{concept_id}' not found."
        if not concept.links:
            return f"'{concept.title}' has no subsections. Use read('{concept_id}') to see its content."
        lines = [f"Children of '{concept.title}' ({len(concept.links)} subsections):\n"]
        for child_id in concept.links:
            child = self._concepts.get(child_id)
            if child:
                has_children = " →" if child.links else ""
                lines.append(f"- {child.id}: {child.title}{has_children}")
            else:
                lines.append(f"- {child_id}: (not in bundle)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Agent-friendly methods (return formatted strings)
    # ------------------------------------------------------------------

    def read(self, concept_id: str) -> str:
        """Read a concept by ID and return its full content as formatted text.

        Args:
            concept_id: The concept ID (e.g., "tables/orders"). Use toc() or
                search() to discover available IDs.

        Returns:
            Formatted string with the concept's title, metadata, links, and body.
            If not found, returns suggestions based on fuzzy matching.
        """
        self._ensure_loaded()
        concept = self._concepts.get(concept_id)
        if concept is None:
            matches = [c for c in self._concepts.values()
                       if concept_id.replace("_", " ") in c.title.lower()]
            if matches:
                suggestions = "\n".join(f"  - {c.id}: {c.title}" for c in matches[:5])
                return f"Concept '{concept_id}' not found. Did you mean:\n{suggestions}"
            available = ", ".join(sorted(self._concepts.keys())[:10])
            return f"Concept '{concept_id}' not found. Available: {available}..."
        parts = [
            f"# {concept.title}",
            f"Type: {concept.type} | Tags: {', '.join(concept.tags)}",
            f"Links to: {', '.join(concept.links[:10])}",
            f"Linked from: {', '.join(self._backlinks_idx.get(concept_id, [])[:10])}",
            "",
            concept.body,
        ]
        return "\n".join(parts)

    def find(self, query: str) -> str:
        """Search the bundle for concepts relevant to a query.

        Uses TF-IDF relevance ranking by default. A custom search backend
        (e.g., embedding-based) can be provided via the search_index parameter.

        Args:
            query: Natural language search query.

        Returns:
            Formatted list of matching concepts (max 10 shown), ranked by relevance.
        """
        results = self._search(query)
        if not results:
            return f"No concepts found matching '{query}'."
        lines = [f"Found {len(results)} concepts matching '{query}':\n"]
        for c in results[:5]:
            lines.append(f"- {c.id} [{c.type}]: {c.title}")
        if len(results) > 5:
            lines.append(f"\n... and {len(results) - 5} more. Refine your query for better results.")
        return "\n".join(lines)

    def __repr__(self) -> str:
        self._ensure_loaded()
        return f"OKFBundle({self.root.name!r}, {len(self._concepts)} concepts)"
