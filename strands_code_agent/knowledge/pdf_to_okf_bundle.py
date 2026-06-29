"""Convert a structured PDF into an OKF knowledge bundle.

Uses the PDF's table-of-contents bookmarks for structure. Requires pymupdf.

Usage:
    from strands_code_agent.knowledge.pdf_converter import pdf_to_okf_bundle

    pdf_to_okf_bundle("guide.pdf", "output_bundle/")
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
import yaml


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:80]


def _write_concept(bundle_root: Path, concept_id: str, frontmatter: dict, body: str):
    path = bundle_root / f"{concept_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_text = yaml.dump(frontmatter, sort_keys=False, allow_unicode=True).rstrip()
    path.write_text(f"---\n{fm_text}\n---\n\n{body}\n", encoding="utf-8")


def pdf_to_okf_bundle(
    pdf_path: str | Path,
    bundle_path: str | Path,
    *,
    max_toc_depth: int = 3,
    max_body_chars: int = 4000,
    max_pages_per_section: int = 10,
    type_names: dict[int, str] | None = None,
) -> dict:
    """Convert a structured PDF into an OKF knowledge bundle.

    Extracts sections from the PDF's table-of-contents bookmarks and writes
    one OKF concept file per section, with cross-links between parent and
    child sections.

    Args:
        pdf_path: Path to the input PDF file.
        bundle_path: Directory to write the bundle into (created if needed).
        max_toc_depth: Maximum TOC depth to include (default 3).
        max_body_chars: Max characters per concept body (default 4000).
        max_pages_per_section: Max pages to extract text from per section (default 10).
        type_names: Mapping of TOC level → OKF type name.
            Default: {1: "Service", 2: "Topic", 3: "Section"}.

    Returns:
        Dict with keys: concepts (int), sections (int), pages (int).
    """
    import fitz  # pymupdf

    pdf_path = Path(pdf_path)
    bundle_root = Path(bundle_path)
    bundle_root.mkdir(parents=True, exist_ok=True)

    if type_names is None:
        type_names = {1: "Service", 2: "Topic", 3: "Section"}

    # Extract TOC sections
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    total_pages = len(doc)
    filtered = [(lvl, title, page) for lvl, title, page in toc if lvl <= max_toc_depth]

    sections: list[dict] = []
    for i, (level, title, start_page) in enumerate(filtered):
        end_page = filtered[i + 1][2] if i + 1 < len(filtered) else total_pages
        extract_end = min(start_page + max_pages_per_section, end_page)

        text_parts = []
        for page_num in range(start_page - 1, min(extract_end - 1, total_pages)):
            text_parts.append(doc[page_num].get_text())
        body = "\n".join(text_parts).strip()

        if len(body) > max_body_chars:
            body = body[:max_body_chars] + (
                f"\n\n[... truncated, full content spans pages {start_page}-{end_page - 1} ...]"
            )

        sections.append({
            "level": level,
            "title": title.strip(),
            "start_page": start_page,
            "end_page": end_page,
            "page_count": end_page - start_page,
            "body": body,
        })
    doc.close()

    # Build hierarchy
    children: dict[str, list[str]] = {}
    parent_stack: list[str] = []
    for section in sections:
        slug = _slugify(section["title"])
        parent_stack = parent_stack[:section["level"] - 1]
        if parent_stack:
            children.setdefault(parent_stack[-1], []).append(slug)
        parent_stack.append(slug)

    # Write concepts
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    concept_map = {_slugify(s["title"]): s for s in sections}
    written = 0

    for slug, section in concept_map.items():
        body = section["body"]
        child_slugs = children.get(slug, [])
        if child_slugs:
            body += "\n\n## Subsections\n\n"
            for cs in child_slugs:
                child = concept_map.get(cs)
                if child:
                    body += f"- [{child['title']}]({cs}.md)\n"

        _write_concept(bundle_root, slug, {
            "type": type_names.get(section["level"], "Section"),
            "title": section["title"],
            "description": (f"Pages {section['start_page']}-{section['end_page'] - 1} "
                           f"({section['page_count']} pages)."),
            "timestamp": timestamp,
        }, body)
        written += 1

    # Write index.md
    level1 = [s for s in sections if s["level"] == 1]
    index_lines = [f"# {pdf_path.stem}\n"]
    for s in level1:
        slug = _slugify(s["title"])
        index_lines.append(f"* [{s['title']}]({slug}.md) — {s['page_count']} pages")
    (bundle_root / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    return {"concepts": written, "sections": len(sections), "pages": total_pages}
