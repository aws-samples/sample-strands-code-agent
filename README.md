# strands-code-agent

A coding agent built on [Strands Agents SDK](https://github.com/strands-agents/sdk-python) that replaces the tool-calling paradigm with code generation as the agent's primary action interface. Rather than invoking structured tools by name and passing results through the conversation context, the agent writes Python code in a persistent REPL where domain capabilities (database queries, APIs, etc.) are exposed as importable library functions. This keeps intermediate data as native Python objects in memory and lets the agent compose multi-step logic in a single code block instead of orchestrating sequential tool calls. In empirical evaluations on the Data Agent Benchmark, this code-generation paradigm achieves higher accuracy (+7%) while consuming 78% fewer input tokens, 67% fewer output tokens, completing tasks 56% faster, and requiring 35% fewer reasoning cycles compared to an equivalent tool-calling agent. The library makes it easy to configure the Python environment with the libraries and domain-specific code your agent needs.

## Installation

```bash
pip install strands-code-agent
```

## Quick Start

```python
from strands_code_agent import CodeAgent

agent = CodeAgent()

response = agent("What is 2 ** 10?")
```

The agent receives a `python_repl` tool automatically and solves tasks by writing and executing Python code.

## CodeAgent

`CodeAgent` extends the Strands `Agent` with a built-in Python REPL and automatic system-prompt enrichment.

| Parameter | Type | Description |
|---|---|---|
| `system_prompt` | `str \| None` | Base system prompt, extended with coding instructions. |
| `tools` | `list \| None` | Additional tools alongside the built-in Python REPL. |
| `toolkits` | `list[Toolkit] \| None` | Toolkits that configure the REPL environment (see below). |
| `tmp_dir` | `bool` | If `True` (default), creates a temp directory and documents its path in the prompt. |
| `python_interpreter_class` | `type[PythonInterpreter]` | The interpreter backend. Defaults to `SandboxedPythonInterpreter` (import restrictions via allowlist). Use `ExecPythonInterpreter` for lightweight unrestricted `exec()`-based execution, or `AgentCorePythonInterpreter` for remote execution via AWS. |
| `python_interpreter_kwargs` | `dict \| None` | Extra keyword arguments forwarded to the interpreter constructor (e.g. `{"region": "us-east-1"}` for `AgentCorePythonInterpreter`). |
| `**kwargs` | | Forwarded to the Strands `Agent` base class (e.g. `model`, `callback_handler`). |

## Toolkit

A `Toolkit` bundles everything the REPL needs for a specific domain. Each field influences the `CodeAgent` in a specific way:

| Parameter | Type | Effect on `PythonInterpreter` | Effect on System Prompt |
|---|---|---|---|
| `libraries` | `list[str] \| None` | Added to `authorized_imports` — the REPL will only allow imports from this allowlist. | — |
| `initialization_code` | `str \| None` | Prepended to `state_initialization` — runs before every Agent snippet. | Documented so the agent knows which symbols are pre-loaded. |
| `usage_instructions` | `str \| None` | — | Appended as-is, giving the agent guidance on how to use the libraries. |
| `domain_specific_code` | `list \| None` | Auto-imported in `state_initialization` (modules added to `authorized_imports`). | Full signature + docstring of each symbol is documented so the agent can use them. |

### Example

```python
from strands_code_agent.toolkits import Toolkit

VISUALIZATION_TOOLKIT = Toolkit(
    # 1. libraries → PythonInterpreter.authorized_imports
    #    Allows the REPL to import these modules.
    #    Use "module.*" to allow a module and all its submodules.
    libraries=["matplotlib.*", "seaborn.*"],

    # 2. initialization_code → PythonInterpreter.state_initialization + System Prompt
    #    Runs before user code; also shown in the prompt so the agent
    #    knows plt and sns are already available.
    initialization_code="""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
""",

    # 3. usage_instructions → System Prompt only
    #    Tells the agent how to behave with these libraries.
    usage_instructions="Do not try to show any matplotlib image: the python_repl tool executes the code in a sub-process without a GUI.",
)
```

### Built-in Toolkits

The library ships with ready-to-use toolkits:

```python
from strands_code_agent.toolkits import (
    VISUALIZATION_TOOLKIT,   # matplotlib + seaborn (non-interactive backend)
    DATA_ANALYSIS_TOOLKIT,   # numpy + pandas + scipy + datetime
)
```

### Domain-Specific Code

Pass your own functions or classes via `domain_specific_code`. The `CodeAgent` will:

1. **Auto-import** them in `PythonInterpreter.state_initialization` (their modules are added to `authorized_imports`).
2. **Document** each symbol's full signature and docstring in the **System Prompt**, so the agent knows how to call them.

```python
from strands_code_agent import CodeAgent, Toolkit

def calculate_roi(investment: float, returns: float) -> float:
    """Calculate return on investment as a percentage."""
    return (returns - investment) / investment * 100


agent = CodeAgent(
    system_prompt="You are a finance assistant.",
    toolkits=[
        Toolkit(domain_specific_code=[calculate_roi])
    ],
)

response = agent("What is the ROI if I invest 1000 and get back 1250?")
```

### Combining Toolkits

```python
from strands_code_agent import CodeAgent
from strands_code_agent.toolkits import DATA_ANALYSIS_TOOLKIT, VISUALIZATION_TOOLKIT

agent = CodeAgent(
    system_prompt="You are a data analyst.",
    toolkits=[DATA_ANALYSIS_TOOLKIT, VISUALIZATION_TOOLKIT],
)
```

## Python Interpreters

The library provides three interpreter backends. All share the same interface (`execute_code`, `clear_state`) and work transparently with `CodeAgent` and `Toolkit`.

### SandboxedPythonInterpreter (default)

Local execution with import restrictions. Only modules listed in `authorized_imports` (from Toolkit `libraries`) can be imported.

```python
from strands_code_agent import CodeAgent

agent = CodeAgent()  # uses SandboxedPythonInterpreter by default
```

### ExecPythonInterpreter

Local execution via `exec()` with no import restrictions. Lightweight and fast, suitable for trusted environments.

```python
from strands_code_agent import CodeAgent
from strands_code_agent.python_environments.local_exec import ExecPythonInterpreter

agent = CodeAgent(python_interpreter_class=ExecPythonInterpreter)
```

### AgentCorePythonInterpreter

Remote execution via [Amazon Bedrock AgentCore Code Interpreter](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/code-interpreter-tool.html). Code runs in a secure, managed sandbox with 200+ pre-installed libraries (pandas, numpy, scikit-learn, torch, boto3, etc.).

```bash
pip install strands-code-agent[agentcore]
```

```python
from strands_code_agent import CodeAgent, AgentCorePythonInterpreter

agent = CodeAgent(
    python_interpreter_class=AgentCorePythonInterpreter,
    python_interpreter_kwargs={"region": "us-east-1"},
)
```

**Configuration options** (passed via `python_interpreter_kwargs`):

| Parameter | Default | Description |
|---|---|---|
| `region` | `"us-east-1"` | AWS region |
| `code_interpreter_identifier` | `"aws.codeinterpreter.v1"` | Managed or custom Code Interpreter ID |
| `session_timeout_seconds` | `900` | Session idle timeout (max 28800 = 8 hours) |

**Requirements:**
- AWS credentials configured (`aws sts get-caller-identity`)
- IAM permissions: `bedrock-agentcore:StartCodeInterpreterSession`, `InvokeCodeInterpreter`, `StopCodeInterpreterSession`

**Key differences from local interpreters:**
- `libraries` in Toolkit is ignored (the remote environment has its own pre-installed packages)
- `domain_specific_code` functions/classes are automatically serialized as source code and sent to the remote session
- Sessions are stateful across tool calls but reset on `clear_state()`

**Lifecycle and cost:**

The AgentCore Code Interpreter uses **active consumption-based pricing** — you pay only for actual CPU and memory consumed, not for idle/I/O wait time:

| Resource | Price |
|---|---|
| CPU | $0.0895 per vCPU-hour (billed per second, only during active computation) |
| Memory | $0.00945 per GB-hour (billed per second, based on peak memory) |

Key lifecycle details:
- **Lazy start** — no session created until code actually runs
- **One session per agent turn** — multiple `execute_code()` calls within the same LLM response share one session (no extra start/stop cost)
- **Reset between turns** — `clear_state()` stops the session; next call starts fresh
- **I/O wait is free** — while the agent is thinking (waiting for LLM response), no CPU is billed
- **Auto-termination** — sessions terminate after `session_timeout_seconds` of inactivity if `close()` is not called

**Example**: An agent that runs 3 code executions per request, each 2 minutes long with 60% I/O wait, using 2 vCPU and 4GB memory costs ~$0.0036 per request ($109/month at 30K executions). See [AgentCore pricing](https://aws.amazon.com/bedrock/agentcore/pricing/) for full details.

## Knowledge Module

The `strands_code_agent.knowledge` module provides tools for giving a CodeAgent access to large structured documents via the [Open Knowledge Format (OKF)](https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md).

### Convert a PDF to an OKF bundle

```python
from strands_code_agent.knowledge import pdf_to_okf_bundle

pdf_to_okf_bundle("guide.pdf", "my_bundle/")
```

Uses the PDF's table-of-contents bookmarks to extract a hierarchy of concepts. Requires `pymupdf`.

### Navigate a bundle with CodeAgent

```python
from strands_code_agent import CodeAgent, Toolkit
from strands_code_agent.knowledge import OKFBundle

agent = CodeAgent(
    system_prompt="You have access to `docs`, an OKFBundle. Use docs.find(query) then docs.read(id).",
    toolkits=[
        Toolkit(
            initialization_code='from strands_code_agent.knowledge import OKFBundle\ndocs = OKFBundle("my_bundle/")',
            domain_specific_code=[OKFBundle],
        )
    ],
)
```

The agent sees a minimal 4-method API:

| Method | Purpose |
|---|---|
| `find(query)` | Keyword search with AND logic + OR fallback. Returns top 5 matches. |
| `read(concept_id)` | Read a concept's full content with metadata and links. |
| `children(concept_id)` | List subsections of a concept for hierarchical drilling. |
| `toc()` | Show top-level sections (fallback when search fails). |

### Custom search backend

The default search uses keyword matching. Provide a custom backend for semantic search:

```python
from strands_code_agent.knowledge import OKFBundle, SearchIndex

class EmbeddingSearch(SearchIndex):
    def build(self, concepts): ...    # compute embeddings
    def query(self, query, top_k=10): ...  # cosine similarity

bundle = OKFBundle("my_bundle/", search_index=EmbeddingSearch())
```

See [`examples/pdf_to_okf_bundle/`](examples/pdf_to_okf_bundle/) for a complete working example using a 2500+ page PDF.

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/). Install it and run from the project root:

```bash
pip install pytest
python -m pytest tests/ -v
```

### Integration Tests (AgentCore)

The AgentCore interpreter tests use mocked boto3 by default and run with the standard test suite. To run a live integration test against the real AgentCore service:

```bash
pip install strands-code-agent[agentcore]
python -m pytest tests/test_agentcore_python_interpreter.py -v -k "integration"
```

**Prerequisites for live tests:**
- AWS credentials configured with AgentCore permissions
- Environment variable: `AWS_REGION` (defaults to `us-east-1`)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Example Agents

Example agents built with this library:

- [Data Analyst Agent](https://github.com/aws-solutions-library-samples/guidance-for-agentic-data-analyst-using-amazon-bedrock-agentcore-on-aws/blob/main/agent/aws_data_analyst/data_analyst_agent.py#L92)
- [Geospatial Agent](https://github.com/aws-samples/sample-geospatial-code-agent/blob/main/agent/geospatial_agent/agent.py#L93)
