# Example: MCP tools — baseline vs. tools-as-code

This directory quantifies the effect of calling **MCP tools as code** (the
[strands-code-agent](../../README.md) approach) versus the conventional
**tool-calling** loop, using the *same* MCP server and the *same* tasks.

It is built around a self-contained MCP server so the comparison is
reproducible — no credentials, no network, deterministic data.

## Files

| File | What it is |
|------|------------|
| `mcp_server.py` | A self-contained [FastMCP](https://modelcontextprotocol.io/) server. Tiny e-commerce dataset (5 customers, 6 products, **200 orders**) with 6 tools. No credentials, no network, seeded RNG. |
| `benchmark.py` | Runs both agents — a plain Strands tool-calling `Agent` and a `CodeAgent` with an `MCPToolkit` — on the identical task set and prints a side-by-side comparison of tokens / cycles / latency / cost / accuracy. |

## Why this server

The tools are shaped to expose the scaling pains that
["code execution with MCP"](https://www.anthropic.com/engineering/code-execution-with-mcp)
addresses:

1. **Large results.** `list_orders` returns up to 200 rows. In the baseline,
   every row is serialized into the model's context on every call. A CodeAgent
   can filter/aggregate those rows *inside the sandbox* and only surface the
   answer.
2. **Chaining / joins.** Some tasks require joining orders→customers and
   orders→products and aggregating. The baseline does this across many model
   turns; a CodeAgent does it in one code block.
3. **Sequential dependencies (→ fewer model cycles).** `get_referrer` reveals
   only one hop of a referral chain, and the next customer to look up is unknown
   until the current call returns. The baseline is forced to spend **one model
   turn per hop**; a CodeAgent walks the whole chain in a single `while` loop
   (one turn). At low temperature the CodeAgent reliably writes that loop (see
   the results discussion for how this depends on the model, not the library).

## Run

```bash
pip install strands-code-agent "mcp[cli]"

python examples/mcp_tools/benchmark.py
```

The agents share one model pinned to **temperature 0.1**, and the whole suite is
run **5 times and averaged** (both configurable at the top of `benchmark.py` via
`TEMPERATURE` and `N_RUNS`) — single runs are noisy, especially the model-cycle
count. The example uses the default Strands provider (Amazon Bedrock), so you need
model access configured — e.g. valid AWS credentials
(`aws sts get-caller-identity`) and Bedrock model access in your region. To use a
different provider, swap the `BedrockModel(...)` in `benchmark.py`.

### The tasks

The task set lives in `benchmark.py`. Each is phrased naturally — the agent must
discover which tools to use from their names and docstrings:

```text
1. How many orders have status 'pending'?                    (truth: 41)
2. Total revenue from all shipped orders?                    (truth: 59723.0)
3. Which customer tier has the highest shipped revenue?      (truth: gold)
   (orders ⋈ customers)
4. Top 3 SKUs by shipped revenue, with product names.        (truth: MON-32 #1)
   (orders ⋈ products)
5. Follow the referral chain from customer 5 back to the     (truth: A. Turing)
   customer nobody referred. (sequential: each hop depends on the previous one)
```

## How the library makes MCP tools callable as code

```python
from mcp import StdioServerParameters, stdio_client
from strands.tools.mcp import MCPClient
from strands_code_agent import CodeAgent, MCPToolkit

client = MCPClient(lambda: stdio_client(
    StdioServerParameters(command="python", args=["mcp_server.py"])))

with client:                                  # keep the session open while the agent runs
    agent = CodeAgent(toolkits=[MCPToolkit(client)])
    agent("Which customer tier has the highest shipped revenue?")
```

`MCPToolkit` lists the server's tools and converts each into a Python function
with a real typed signature and docstring (synthesized from the tool's JSON
Schema). The agent then writes code like:

```python
shipped = list_orders(status="shipped")          # native list[dict], stays in the sandbox
custs = {c["id"]: c for c in list_customers()}
rev = {}
for o in shipped:
    rev.setdefault(custs[o["customer_id"]]["tier"], 0.0)
    rev[custs[o["customer_id"]]["tier"]] += o["total"]
print(max(rev, key=rev.get))                      # only the answer reaches the model
```

The 200 order rows never enter the model's context.

For a **sequential** task the agent writes a loop that resolves in one turn what a
tool-calling model must unroll across many:

```python
cid = 5
chain = []
while cid is not None:                 # one model turn walks the whole chain
    r = get_referrer(cid)              # each hop depends on the previous result
    chain.append(r["name"])
    cid = r["referred_by"]
print("root:", chain[-1])              # -> Alan Turing
```

## Results

`benchmark.py` pins both agents to one shared model at **temperature 0.1** and
repeats the whole suite **5 times**, reporting the **mean**. Low temperature plus
averaging makes the numbers reproducible; single runs are noisy. Figures below are
means of 5 runs against Bedrock Claude Sonnet (your exact numbers will vary by
model, but the direction and magnitude are stable):

| metric | baseline (tool-calling) | codeagent (MCP as code) | change |
|---|---:|---:|---:|
| input tokens | 55,443 | ~25,700 | **−52% to −55%** |
| output tokens | ~2,100 | ~1,550 | **−22% to −31%** |
| model cycles | 14.0 | 12.0–12.8 | **−9% to −14%** |
| duration | ~31 s | ~22 s | **≈ −28%** |
| cost | $0.197 | ~$0.100 | **−47% to −51%** |
| **accuracy** | **80%** | **100%** | **+20pp** |

### Reading the numbers

- **Input tokens (≈−53%)** is the headline. The whole `list_orders` result
  (≈160 shipped rows) re-enters the baseline's context on every turn that reads
  it, plus all six tool definitions sit in context on every turn. The CodeAgent
  keeps the rows in the sandbox and sends back only the computed answer.
- **Model cycles (−9% to −14%)** come from the sequential **task 5**. The
  per-task mean cycles the benchmark prints make this explicit:

  | task | baseline | codeagent |
  |---|---:|---:|
  | 1 count pending | 2.0 | 2.0 |
  | 2 shipped revenue | 2.0 | 2.0 |
  | 3 revenue by tier | 2.0 | 3.0 |
  | 4 top SKUs | 2.0 | 3.0 |
  | **5 referral chain** | **6.0** | **~2.0** |

  On the chain the baseline is forced hop-by-hop (one model turn per hop, so 6
  cycles for the 5-hop chain), while the CodeAgent walks the whole chain in a
  single `while` loop (2 cycles) — *"one code block replaces many sequential tool
  calls."* It pays a small, fixed +1-cycle overhead on the two aggregation tasks
  (3 and 4), but the chain more than covers it. **At temperature 0.1 whether the
  CodeAgent writes the loop is essentially deterministic** — across runs it almost
  always loops (task-5 mean lands at ~2.0, occasionally nudged up by a stray run
  that stepped the chain). This is worth stating plainly: whether the agent loops
  is a property of the *model and its sampling temperature*, not of this library.
  At higher temperatures the choice becomes bimodal (loop → 2 cycles, or step →
  6–7), which is why the benchmark pins a low temperature and averages.
- **Accuracy (80% → 100%)** — the baseline's miss is the *total-revenue* task:
  asked to sum `total` over ≈160 rows **in its head**, the model gets it
  slightly wrong. The CodeAgent sums it in Python and is exact. Not routing data
  through the model also stops the model from doing error-prone arithmetic on it.
- **Duration (≈−28%)** is faster — fewer, heavier round-trips replaced by lighter
  ones — though it is the noisiest metric at this scale.

This mirrors the library's Data Agent Benchmark result (−78% input tokens, +7%
accuracy) and Anthropic's reported 98.7% token reduction on a data-heavy MCP
pipeline — the magnitude scales with how much data and how many sequential hops
flow through the tools.

## Notes on reproducibility

- The dataset is generated with a fixed seed (`random.Random(42)`), so counts
  (41 pending orders, $59,723 shipped revenue, gold top tier, MON-32 top SKU,
  referral root Alan Turing) are stable across runs and machines.
- The server speaks stdio and is launched with the same Python interpreter that
  runs the example (`sys.executable`), so no separate process management is
  required.
- `MCPToolkit` functions call a live client on the host, so they run on the
  local interpreters (`SandboxedPythonInterpreter` default, or
  `ExecPythonInterpreter`). `AgentCorePythonInterpreter` is not supported for
  MCP tools and raises a clear error.
