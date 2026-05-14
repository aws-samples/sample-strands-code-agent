# strands-code-agent

A coding agent built on [Strands Agents SDK](https://github.com/strands-agents/sdk-python) that replaces the tool-calling paradigm with code generation as the agent's primary action interface. Rather than invoking structured tools by name and passing results through the conversation context, the agent writes Python code in a persistent REPL where domain capabilities (database queries, APIs, etc.) are exposed as importable library functions. This keeps intermediate data as native Python objects in memory and lets the agent compose multi-step logic in a single code block instead of orchestrating sequential tool calls. In empirical evaluations on the Data Agent Benchmark, this code-generation paradigm achieves higher accuracy (+7%) while consuming 78% fewer input tokens, completing tasks 56% faster, and requiring 35% fewer reasoning cycles compared to an equivalent tool-calling agent. The library makes it easy to configure the Python environment with the libraries and domain-specific code your agent needs.

## Installation

```bash
pip install strands-code-agent
```

## Quick Start

```python
from strands_code_agent import CodeAgent

agent = CodeAgent(system_prompt="You are a helpful data analyst.")

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
| `python_interpreter_class` | `type[PythonInterpreter]` | The interpreter backend. Defaults to `SandboxedPythonInterpreter` (import restrictions via allowlist). Use `ExecPythonInterpreter` for lightweight unrestricted `exec()`-based execution. |
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

## Running Tests

The test suite uses [pytest](https://docs.pytest.org/). Install it and run from the project root:

```bash
pip install pytest
python -m pytest tests/ -v
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
