# Example: Large PDF → OKF Bundle → CodeAgent

Converts a 2500+ page PDF into a navigable OKF knowledge bundle and runs a CodeAgent that can search, read, and drill into it hierarchically.

The PDF used is the [Amazon Bedrock AgentCore Developer Guide](https://docs.aws.amazon.com/pdfs/bedrock-agentcore/latest/devguide/bedrock-agentcore-dg.pdf) (23MB, 2533 pages, 967 concepts).

## Run

```bash
pip install strands-code-agent pymupdf
python examples/pdf_to_okf_bundle/agent_with_knowledge.py
```

On first run, the script downloads the PDF and builds the bundle automatically. Subsequent runs reuse the cached bundle.

## How it works

1. **PDF → OKF bundle** (`pdf_to_okf_bundle`): Uses the PDF's table-of-contents bookmarks to extract sections as OKF concepts with cross-links between parent/child sections.

2. **CodeAgent with `OKFBundle`**: The agent gets a pre-loaded `agentcore_docs` object with 4 methods:

| Method | Purpose |
|--------|---------|
| `find(query)` | Keyword search, returns top matches |
| `read(concept_id)` | Read a concept's full content |
| `children(concept_id)` | Expand subsections |
| `toc()` | Top-level sections (fallback) |

## Performance

The agent typically answers questions in 3–10 cycles using the pattern: `find("topic") → read(best_match)`. On a 967-concept bundle extracted from a 2500-page PDF, this achieves ~60% token reduction vs a naive flat search approach.
