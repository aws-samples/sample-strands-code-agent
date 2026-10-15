"""Side-by-side benchmark: tool-calling baseline vs. CodeAgent-with-MCP.

Runs the identical task set against the identical self-contained MCP server, once
with a plain Strands tool-calling ``Agent`` and once with a ``CodeAgent`` using an
``MCPToolkit``, then prints a comparison of tokens / cycles / latency / cost.

Both agents share one model pinned to a low temperature (``TEMPERATURE``) so the
comparison is as deterministic as the provider allows, and the whole benchmark is
repeated ``N_RUNS`` times and **averaged** — single runs are noisy, especially the
model-cycle count on the sequential task.

Also checks answer agreement: for each task it compares the final text of each
agent against an expected ground-truth substring, so the token savings are not
bought at the cost of correctness.

Requirements:
    pip install strands-code-agent "mcp[cli]"
    # plus model access (Amazon Bedrock by default)

Usage:
    python examples/mcp_tools/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.models import BedrockModel
from strands.tools.mcp import MCPClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strands_code_agent import CodeAgent, MCPToolkit  # noqa: E402
from strands_code_agent.utils import get_response_metrics  # noqa: E402

SERVER = Path(__file__).parent / "mcp_server.py"

# Low temperature makes the model's behaviour (notably whether it writes a loop
# on the sequential task) as reproducible as the provider allows. Averaging over
# several runs smooths out the residual noise.
TEMPERATURE = 0.1
N_RUNS = 5

BASELINE_PROMPT = (
    "You are a data assistant for a small online shop. Answer the user's question "
    "using the available tools. Be concise and give the final numeric/factual "
    "answer explicitly."
)
CODE_PROMPT = (
    "You are a data assistant for a small online shop. Answer the user's question "
    "by writing Python code that calls the available MCP-backed functions. Filter, "
    "join, and aggregate results in code; only print the final answer. Be concise "
    "and give the final numeric/factual answer explicitly."
)

# (question, ground-truth substring that must appear in a correct final answer)
TASKS = [
    ("How many orders have status 'pending'?", "41"),
    ("What is the total revenue (sum of 'total') from all shipped orders?", "59723"),
    ("Which customer tier has the highest total shipped revenue? "
     "Join orders to customers by customer_id.", "gold"),
    ("List the top 3 SKUs by shipped revenue, with their product names.", "MON-32"),
    ("Customer 5 was referred by another customer, who was in turn referred by "
     "someone else, and so on. Follow that referral chain back to its origin: "
     "who is the customer at the start of the chain that nobody referred?", "Turing"),
]

PRICE_IN, PRICE_OUT = 3.0, 15.0
METRIC_KEYS = ("input_tokens", "output_tokens", "total_cycles", "total_duration")


def _mcp_client() -> MCPClient:
    return MCPClient(
        lambda: stdio_client(StdioServerParameters(command=sys.executable, args=[str(SERVER)]))
    )


def _run(make_agent, tasks):
    """Run every task with a FRESH agent; return totals and per-task cycles.

    A new agent per task is essential: Strands agents keep conversation history
    across calls, so reusing one agent would (a) re-send prior turns on every
    call — inflating token/cycle counts, which ``get_response_metrics`` reports
    cumulatively — and (b) let the model answer later tasks from memory instead
    of actually using the tools. Each task must be measured in isolation.
    """
    acc = {k: 0.0 for k in METRIC_KEYS}
    per_task_cycles = []
    correct = 0
    for question, truth in tasks:
        agent = make_agent()
        response = agent(question)
        text = response.message["content"][0]["text"]
        m = get_response_metrics(response, PRICE_IN, PRICE_OUT)
        correct += int(truth.lower() in text.lower().replace(",", ""))
        for k in METRIC_KEYS:
            acc[k] += m[k]
        per_task_cycles.append(m["total_cycles"])
    acc["accuracy"] = correct / len(tasks)
    acc["cost"] = acc["input_tokens"] * PRICE_IN / 1e6 + acc["output_tokens"] * PRICE_OUT / 1e6
    return acc, per_task_cycles


def _mean_dict(dicts):
    """Element-wise mean of a list of same-keyed numeric dicts."""
    n = len(dicts)
    return {k: sum(d[k] for d in dicts) / n for k in dicts[0]}


def _mean_cols(rows):
    """Column-wise mean of a list of equal-length numeric lists."""
    n = len(rows)
    return [sum(r[i] for r in rows) / n for i in range(len(rows[0]))]


def _pct(base, new):
    if base == 0:
        return "n/a"
    return f"{(new - base) / base * 100:+.0f}%"


def main() -> None:
    base_runs, base_cycle_runs = [], []
    code_runs, code_cycle_runs = [], []

    model = BedrockModel(temperature=TEMPERATURE)
    client = _mcp_client()
    with client:
        tools = client.list_tools_sync()

        # Factories: a fresh agent per task/run (see _run). The model is
        # stateless and safely shared; only the agents accumulate history.
        def make_baseline():
            return Agent(model=model, system_prompt=BASELINE_PROMPT, tools=client.list_tools_sync())

        def make_code_agent():
            return CodeAgent(model=model, system_prompt=CODE_PROMPT, toolkits=[MCPToolkit(client)])

        print(f"MCP server: {SERVER.name} | {len(tools)} tools | {len(TASKS)} tasks")
        print(f"model temperature={TEMPERATURE} | averaging over {N_RUNS} runs\n")

        for run in range(1, N_RUNS + 1):
            print(f"run {run}/{N_RUNS} ...", flush=True)
            b_acc, b_cycles = _run(make_baseline, TASKS)
            c_acc, c_cycles = _run(make_code_agent, TASKS)
            base_runs.append(b_acc)
            code_runs.append(c_acc)
            base_cycle_runs.append(b_cycles)
            code_cycle_runs.append(c_cycles)
            print(f"  baseline  cycles={b_cycles} acc={b_acc['accuracy']:.0%}")
            print(f"  codeagent cycles={c_cycles} acc={c_acc['accuracy']:.0%}")

    baseline = _mean_dict(base_runs)
    codeagent = _mean_dict(code_runs)
    base_task_cycles = _mean_cols(base_cycle_runs)
    code_task_cycles = _mean_cols(code_cycle_runs)

    print(f"\n{'=' * 56}\nCOMPARISON — mean of {N_RUNS} runs (totals over {len(TASKS)} tasks)\n{'=' * 56}")
    rows = [
        ("input tokens", baseline["input_tokens"], codeagent["input_tokens"]),
        ("output tokens", baseline["output_tokens"], codeagent["output_tokens"]),
        ("model cycles", baseline["total_cycles"], codeagent["total_cycles"]),
        ("duration (s)", baseline["total_duration"], codeagent["total_duration"]),
        ("cost ($)", baseline["cost"], codeagent["cost"]),
    ]
    print(f"{'metric':<16}{'baseline':>14}{'codeagent':>14}{'change':>10}")
    print("-" * 54)
    for name, b, c in rows:
        fmt = "{:,.4f}" if name == "cost ($)" else "{:,.1f}"
        print(f"{name:<16}{fmt.format(b):>14}{fmt.format(c):>14}{_pct(b, c):>10}")
    print("-" * 54)
    print(f"{'accuracy':<16}{baseline['accuracy']:>13.0%}{codeagent['accuracy']:>14.0%}")

    # Per-task mean model cycles — the sequential referral task (last row) is
    # where a single code loop can replace a chain of tool calls.
    print(f"\n{'=' * 68}\nPER-TASK MEAN MODEL CYCLES (mean of {N_RUNS} runs)\n{'=' * 68}")
    print(f"{'task':<6}{'baseline':>10}{'codeagent':>11}  question")
    print("-" * 68)
    for i, ((question, _truth), b_c, c_c) in enumerate(
        zip(TASKS, base_task_cycles, code_task_cycles), start=1
    ):
        print(f"{i:<6}{b_c:>10.1f}{c_c:>11.1f}  {question[:46]}")


if __name__ == "__main__":
    main()
