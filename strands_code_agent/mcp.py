"""Expose MCP (Model Context Protocol) servers to a ``CodeAgent`` as Python functions.

Instead of loading every MCP tool definition into the model's context and routing
every result back through the model (the default tool-calling pattern), this module
converts each MCP tool into a **local Python function** that the agent calls from the
sandboxed REPL. The function has a real typed signature and docstring (synthesized
from the tool's JSON Schema), and it returns a **native Python object** (recovered from
the MCP result's ``structuredContent``) rather than a serialized blob.

This is the "code execution with MCP" pattern (Anthropic, Nov 2025): the agent writes
code to call tools, filters/joins/aggregates results inside the sandbox, and only
surfaces the final answer — using far fewer tokens and model turns.

Usage::

    from mcp import StdioServerParameters, stdio_client
    from strands.tools.mcp import MCPClient
    from strands_code_agent import CodeAgent
    from strands_code_agent.mcp import MCPToolkit

    client = MCPClient(lambda: stdio_client(StdioServerParameters(command="python", args=["server.py"])) )
    with client:                       # the session must stay open while the agent runs
        agent = CodeAgent(toolkits=[MCPToolkit(client)])
        agent("How many orders are pending?")

The ``MCPClient`` must be open (inside its ``with`` block / after ``start()``) both when
the ``MCPToolkit`` is constructed (to list tools) and while the agent runs (to call them).
"""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Literal, Optional

from strands_code_agent.toolkits import Toolkit


# JSON Schema primitive type -> Python annotation.
_JSON_SCHEMA_TYPES: dict[str, Any] = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _annotation_for(schema: dict | None) -> Any:
    """Map a JSON Schema property schema to a Python type annotation.

    Falls back to :data:`typing.Any` for anything we can't represent precisely
    (``anyOf``/``oneOf`` unions, missing type, unknown type). Nested objects
    degrade to ``dict`` and arrays to ``list`` / ``list[T]``.
    """
    if not isinstance(schema, dict):
        return Any

    # enum -> Literal of its values (keeps the choices visible to the agent).
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and all(isinstance(v, (str, int, bool)) for v in enum):
        try:
            return Literal[tuple(enum)]  # type: ignore[valid-type]
        except TypeError:
            return Any

    typ = schema.get("type")
    if isinstance(typ, list):
        # e.g. ["string", "null"] -> Optional[str]
        non_null = [t for t in typ if t != "null"]
        if len(non_null) == 1:
            base = _JSON_SCHEMA_TYPES.get(non_null[0], Any)
            return Optional[base] if "null" in typ else base
        return Any

    if typ == "array":
        items = schema.get("items")
        item_ann = _annotation_for(items) if isinstance(items, dict) else Any
        return list if item_ann is Any else list[item_ann]  # type: ignore[valid-type]

    if isinstance(typ, str):
        return _JSON_SCHEMA_TYPES.get(typ, Any)

    return Any


def _build_signature(input_schema: dict) -> inspect.Signature:
    """Build an ``inspect.Signature`` from an MCP tool's JSON input schema.

    Required properties become positional-or-keyword parameters; optional ones
    get a default (the schema ``default`` if present, else ``None``). To keep a
    valid Python signature, required parameters are emitted before optional ones.
    """
    properties: dict = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = set(input_schema.get("required", []) if isinstance(input_schema, dict) else [])

    required_params: list[inspect.Parameter] = []
    optional_params: list[inspect.Parameter] = []
    for name, prop in properties.items():
        annotation = _annotation_for(prop)
        if name in required:
            required_params.append(
                inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
            )
        else:
            default = prop.get("default", None) if isinstance(prop, dict) else None
            # If a value is optional and has no schema default, widen to Optional.
            if "default" not in (prop if isinstance(prop, dict) else {}):
                annotation = Optional[annotation] if annotation is not Any else Any
            optional_params.append(
                inspect.Parameter(
                    name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default, annotation=annotation
                )
            )

    return inspect.Signature(required_params + optional_params)


def _build_docstring(name: str, description: str, input_schema: dict, output_schema: dict | None) -> str:
    """Synthesize a Google-style docstring from the tool's description and schemas."""
    lines: list[str] = []
    if description:
        lines.append(description.strip())

    properties: dict = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = set(input_schema.get("required", []) if isinstance(input_schema, dict) else [])
    if properties:
        lines.append("")
        lines.append("Args:")
        for pname, prop in properties.items():
            prop = prop if isinstance(prop, dict) else {}
            typ = prop.get("type", "any")
            opt = "" if pname in required else ", optional"
            desc = prop.get("description", "")
            enum = prop.get("enum")
            if enum:
                desc = (desc + " " if desc else "") + f"One of: {', '.join(map(repr, enum))}."
            lines.append(f"    {pname} ({typ}{opt}): {desc}".rstrip())

    lines.append("")
    lines.append("Returns:")
    lines.append("    The tool's result as a native Python object "
                 "(e.g. dict / list / str), parsed from the MCP response.")
    return "\n".join(lines)


# Content block types that carry no reusable native value.
def _maybe_json(text: str) -> Any:
    """Parse ``text`` as JSON if it looks like a JSON object/array/scalar; else return it unchanged.

    Only attempts a parse for strings that start with a JSON structural character,
    so plain prose is returned as-is rather than being mangled.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in '{[":-0123456789tfn':
        return text
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        return text


def _result_to_native(result: Any) -> Any:
    """Recover a native Python object from an ``MCPToolResult``.

    Preference order:
      1. ``structuredContent["result"]`` — FastMCP wraps scalar/list returns under
         a single ``result`` key, so this unwraps to the actual value.
      2. ``structuredContent`` as a whole — tools that natively return an object.
      3. joined text of the ``content`` blocks — unstructured / older servers.

    Raises ``RuntimeError`` if the tool reported an application-level error.
    """
    # ``result`` is a dict-like MCPToolResult (TypedDict).
    is_error = result.get("isError") if hasattr(result, "get") else None

    structured = result.get("structuredContent") if hasattr(result, "get") else None
    native: Any
    if isinstance(structured, dict) and set(structured.keys()) == {"result"}:
        native = structured["result"]
    elif structured is not None:
        native = structured
    else:
        # No structuredContent: fall back to the text content blocks. Many servers
        # (e.g. FastMCP tools whose return type is an untyped ``dict``/``list``)
        # return their payload only as JSON text. Try to parse each text block back
        # into a native object so the agent gets a dict/list, not a string.
        content = result.get("content", []) if hasattr(result, "get") else []
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and "text" in block]
        if not texts:
            native = content
        elif len(texts) == 1:
            native = _maybe_json(texts[0])
        else:
            parsed = [_maybe_json(t) for t in texts]
            # If every block parsed to a non-string object, hand back the list of
            # objects; otherwise join the raw text (keeps unstructured output readable).
            native = parsed if all(not isinstance(p, str) for p in parsed) else "\n".join(texts)

    if is_error:
        raise RuntimeError(f"MCP tool reported an error: {native}")
    return native


def mcp_tool_to_function(mcp_client: Any, tool: Any) -> Callable:
    """Convert a single MCP tool into a callable Python function.

    Args:
        mcp_client: An open ``strands.tools.mcp.MCPClient`` used to invoke the tool.
        tool: An ``MCPAgentTool`` (as returned by ``MCPClient.list_tools_sync()``).

    Returns:
        A function named after the tool, carrying a synthetic ``__signature__`` and
        ``__doc__`` derived from the tool's JSON Schema. Calling it invokes the tool
        synchronously via ``mcp_client.call_tool_sync`` and returns a native object.

        The function's ``__module__`` is set to ``"__main__"`` so that
        :class:`~strands_code_agent.code_agent.CodeAgent` injects it directly as a REPL
        builtin (via ``additional_functions``) rather than trying to ``import`` it.
    """
    spec = tool.tool_spec
    name: str = spec["name"]
    description: str = spec.get("description", "") or ""
    input_schema: dict = spec.get("inputSchema", {}).get("json", {}) or {}
    output_schema = spec.get("outputSchema", {}).get("json") if spec.get("outputSchema") else None

    signature = _build_signature(input_schema)

    def _call(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = {k: v for k, v in bound.arguments.items() if v is not None}
        result = mcp_client.call_tool_sync(
            tool_use_id=f"codeagent-{name}", name=name, arguments=arguments or None
        )
        return _result_to_native(result)

    _call.__name__ = name
    _call.__qualname__ = name
    _call.__module__ = "__main__"  # signal CodeAgent to inject, not import
    _call.__signature__ = signature  # type: ignore[attr-defined]
    _call.__doc__ = _build_docstring(name, description, input_schema, output_schema)
    _call.__mcp_tool__ = True  # type: ignore[attr-defined]  # marks a non-serializable MCP function
    return _call


class MCPToolkit(Toolkit):
    """A :class:`~strands_code_agent.toolkits.Toolkit` that exposes an MCP server's tools as Python functions.

    Each tool on the connected MCP server becomes a local function in the agent's REPL,
    documented in the system prompt with its signature and docstring. The agent calls
    these functions from code and gets native Python objects back, keeping large
    intermediate results out of the model's context.

    Args:
        mcp_client: An open ``strands.tools.mcp.MCPClient``. It must be started (inside
            its ``with`` block, or after ``start()``) when the toolkit is constructed —
            the tools are listed eagerly — and must remain open while the agent runs,
            because tool calls are routed to the live session.
        tool_filter: Optional iterable of tool names to include. If given, only those
            tools are exposed; unknown names are ignored.
        usage_instructions: Optional extra guidance appended to the system prompt. A
            sensible default is provided describing how to use the generated functions.

    Note:
        MCP functions close over a live client on the host, so they cannot be serialized
        for remote execution. Use a local interpreter
        (``SandboxedPythonInterpreter`` / ``ExecPythonInterpreter``);
        ``AgentCorePythonInterpreter`` is not supported for MCP tools.
    """

    _DEFAULT_USAGE = (
        "The domain functions below are backed by MCP (Model Context Protocol) tools. "
        "They are already available in the REPL by name — call them directly like "
        "ordinary Python functions and do NOT import them (there is no module to import "
        "them from). Each returns a native Python object (dict / list / scalar / str). "
        "Prefer to filter, join, and aggregate their results in code and only print the "
        "final answer, rather than printing whole large results."
    )

    def __init__(self, mcp_client: Any, tool_filter=None, usage_instructions: str | None = None):
        self.mcp_client = mcp_client
        tools = mcp_client.list_tools_sync()
        if tool_filter is not None:
            wanted = set(tool_filter)
            tools = [t for t in tools if t.tool_spec["name"] in wanted]

        functions = [mcp_tool_to_function(mcp_client, t) for t in tools]
        self.tool_names = [f.__name__ for f in functions]

        super().__init__(
            libraries=None,
            initialization_code=None,
            usage_instructions=usage_instructions or self._DEFAULT_USAGE,
            domain_specific_code=functions,
        )
