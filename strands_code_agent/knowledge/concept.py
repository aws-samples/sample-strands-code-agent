"""Concept dataclass — the atomic unit of knowledge in an OKF bundle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Concept:
    """A single concept in an OKF bundle (one .md file).

    Attributes:
        id: Relative path without .md (e.g. "tables/orders").
        type: OKF type field (e.g. "BigQuery Table", "Playbook").
        title: Human-readable title.
        description: Short summary.
        tags: List of tags.
        resource: External URI (optional).
        timestamp: ISO 8601 datetime string (optional).
        body: Markdown body content.
        metadata: Full frontmatter dict for extensibility.
        links: Outgoing concept IDs extracted from markdown links.
    """

    id: str
    type: str = "Unknown"
    title: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)
    resource: str | None = None
    timestamp: str | None = None
    body: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    links: list[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        """Approximate character cost of including this concept in context."""
        return len(self.body) + len(self.title) + len(self.description)

    def summary(self) -> str:
        """One-line summary: title + description."""
        parts = [self.title or self.id]
        if self.description:
            parts.append(f"— {self.description}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"Concept({self.id!r}, type={self.type!r}, title={self.title!r})"
