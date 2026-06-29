"""Use a CodeAgent with a large OKF knowledge bundle.

Downloads the Amazon Bedrock AgentCore Developer Guide PDF (if needed),
converts it to an OKF bundle, and runs a CodeAgent that can navigate it.

Requirements:
    pip install strands-code-agent pymupdf

Usage:
    python examples/pdf_to_okf_bundle/agent_with_knowledge.py
"""

from pathlib import Path

from strands_code_agent import CodeAgent, Toolkit
from strands_code_agent.knowledge import OKFBundle, pdf_to_okf_bundle
from strands_code_agent.utils import get_response_metrics

PDF_URL = "https://docs.aws.amazon.com/pdfs/bedrock-agentcore/latest/devguide/bedrock-agentcore-dg.pdf"
PDF_PATH = Path(__file__).parent / "bedrock_agentcore_guide.pdf"
BUNDLE_PATH = Path(__file__).parent / "agentcore_bundle"


def ensure_bundle():
    """Download PDF and build bundle if it doesn't exist yet."""
    if BUNDLE_PATH.exists():
        return
    if not PDF_PATH.exists():
        print("Downloading AgentCore Developer Guide PDF...")
        import urllib.request
        urllib.request.urlretrieve(PDF_URL, PDF_PATH)
    print(f"Converting PDF to OKF bundle...")
    result = pdf_to_okf_bundle(PDF_PATH, BUNDLE_PATH)
    print(f"✓ {result['concepts']} concepts from {result['pages']} pages\n")


def main():
    ensure_bundle()

    agent = CodeAgent(
        system_prompt=(
            "You are a technical assistant with access to the Amazon Bedrock AgentCore "
            "Developer Guide as an OKFBundle object called `agentcore_docs`.\n\n"
            "Navigate it with these methods (in priority order):\n"
            "1. agentcore_docs.find(query) → search by keywords, returns top matches\n"
            "2. agentcore_docs.read(concept_id) → read a concept's full content\n"
            "3. agentcore_docs.children(concept_id) → expand subsections of a concept\n"
            "4. agentcore_docs.toc() → top-level sections (only if find fails)\n\n"
            "Typical flow: find('topic') → pick best match → read(id)\n"
            "Cite concept IDs in your answers."
        ),
        toolkits=[
            Toolkit(
                initialization_code=(
                    f'from strands_code_agent.knowledge import OKFBundle\n'
                    f'agentcore_docs = OKFBundle("{BUNDLE_PATH}")'
                ),
                domain_specific_code=[OKFBundle],
            )
        ],
    )

    questions = [
        "How do I start a Code Interpreter session and execute Python code?",
        "What IAM permissions are needed for AgentCore Runtime?",
        "Explain how AgentCore Memory works — short-term vs long-term.",
    ]

    print("=" * 60)
    print("AgentCore Developer Guide Knowledge Agent")
    print(f"Bundle: {BUNDLE_PATH}")
    print("=" * 60)

    for q in questions:
        print(f"\n{'─' * 60}")
        print(f"Q: {q}")
        print(f"{'─' * 60}\n")
        response = agent(q)
        print(response.message["content"][0]["text"])

        metrics = get_response_metrics(response, price_1M_input_tokens=3.0, price_1M_output_tokens=15.0)
        print(f"\n  ⏱  {metrics['total_duration']:.1f}s | "
              f"🔄 {metrics['total_cycles']} cycles | "
              f"📥 {metrics['input_tokens']:,} in / 📤 {metrics['output_tokens']:,} out | "
              f"💰 ${metrics['cost']:.4f}")


if __name__ == "__main__":
    main()
