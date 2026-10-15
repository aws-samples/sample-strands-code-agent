"""Tests for strands_code_agent.mcp — MCP-tool → Python-function conversion.

These tests use a lightweight fake MCPClient/MCPAgentTool so no server or model
is required. They verify the schema→signature mapping, docstring synthesis,
native-value recovery from MCP results, and CodeAgent wiring.
"""

import inspect
from typing import Any, Literal, Optional, get_args, get_origin
from unittest.mock import patch

import pytest

from strands_code_agent.mcp import (
    MCPToolkit,
    _annotation_for,
    _build_signature,
    _result_to_native,
    mcp_tool_to_function,
)
from strands_code_agent.code_agent import CodeAgent


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class _FakeTool:
    def __init__(self, spec):
        self.tool_spec = spec


class _FakeMCPClient:
    """Records calls and returns canned MCPToolResult-like dicts."""

    def __init__(self, specs, results=None):
        self._specs = specs
        self._results = results or {}
        self.calls = []

    def list_tools_sync(self):
        return [_FakeTool(s) for s in self._specs]

    def call_tool_sync(self, tool_use_id, name, arguments=None, **kwargs):
        self.calls.append((name, arguments))
        return self._results.get(name, {"content": [], "structuredContent": {"result": None}})


def _spec(name, description="", properties=None, required=None, output=None):
    schema = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    spec = {"name": name, "description": description, "inputSchema": {"json": schema}}
    if output is not None:
        spec["outputSchema"] = {"json": output}
    return spec


# --------------------------------------------------------------------------- #
# Type mapping
# --------------------------------------------------------------------------- #
class TestAnnotationFor:
    @pytest.mark.parametrize("js,py", [
        ({"type": "integer"}, int),
        ({"type": "number"}, float),
        ({"type": "string"}, str),
        ({"type": "boolean"}, bool),
        ({"type": "object"}, dict),
    ])
    def test_primitives(self, js, py):
        assert _annotation_for(js) is py

    def test_array_untyped(self):
        assert _annotation_for({"type": "array"}) is list

    def test_array_typed(self):
        ann = _annotation_for({"type": "array", "items": {"type": "string"}})
        assert get_origin(ann) is list
        assert get_args(ann) == (str,)

    def test_enum_becomes_literal(self):
        ann = _annotation_for({"type": "string", "enum": ["a", "b"]})
        assert get_origin(ann) is Literal
        assert set(get_args(ann)) == {"a", "b"}

    def test_nullable_union(self):
        ann = _annotation_for({"type": ["string", "null"]})
        assert ann == Optional[str]

    def test_missing_type_is_any(self):
        assert _annotation_for({}) is Any
        assert _annotation_for({"anyOf": [{"type": "string"}]}) is Any


# --------------------------------------------------------------------------- #
# Signature building
# --------------------------------------------------------------------------- #
class TestBuildSignature:
    def test_required_before_optional(self):
        sig = _build_signature({
            "properties": {"opt": {"type": "string", "default": "x"}, "req": {"type": "integer"}},
            "required": ["req"],
        })
        names = list(sig.parameters)
        assert names == ["req", "opt"]  # required reordered before optional

    def test_default_applied(self):
        sig = _build_signature({"properties": {"n": {"type": "integer", "default": 5}}})
        assert sig.parameters["n"].default == 5

    def test_optional_without_default_is_none(self):
        sig = _build_signature({"properties": {"status": {"type": "string"}}})
        assert sig.parameters["status"].default is None
        assert sig.parameters["status"].annotation == Optional[str]

    def test_empty_schema(self):
        sig = _build_signature({})
        assert list(sig.parameters) == []


# --------------------------------------------------------------------------- #
# Native value recovery
# --------------------------------------------------------------------------- #
class TestResultToNative:
    def test_unwraps_result_key(self):
        assert _result_to_native({"structuredContent": {"result": 5}}) == 5

    def test_list_result(self):
        rows = [{"a": 1}, {"a": 2}]
        assert _result_to_native({"structuredContent": {"result": rows}}) == rows

    def test_object_without_result_wrapper(self):
        sc = {"name": "Ada", "age": 36}
        assert _result_to_native({"structuredContent": sc}) == sc

    def test_text_fallback(self):
        res = {"content": [{"text": "hello"}, {"text": "world"}]}
        assert _result_to_native(res) == "hello\nworld"

    def test_json_text_parsed_to_native(self):
        # A single JSON-object text block (no structuredContent) → native dict.
        res = {"content": [{"text": '{"name": "Ada", "age": 36}'}]}
        assert _result_to_native(res) == {"name": "Ada", "age": 36}

    def test_json_array_text_parsed(self):
        res = {"content": [{"text": "[1, 2, 3]"}]}
        assert _result_to_native(res) == [1, 2, 3]

    def test_multiple_json_blocks_become_list(self):
        res = {"content": [{"text": '{"a": 1}'}, {"text": '{"a": 2}'}]}
        assert _result_to_native(res) == [{"a": 1}, {"a": 2}]

    def test_plain_prose_left_as_string(self):
        res = {"content": [{"text": "just some text"}]}
        assert _result_to_native(res) == "just some text"

    def test_error_raises(self):
        with pytest.raises(RuntimeError):
            _result_to_native({"isError": True, "structuredContent": {"result": "boom"}})


# --------------------------------------------------------------------------- #
# Function conversion
# --------------------------------------------------------------------------- #
class TestMcpToolToFunction:
    def test_name_and_module(self):
        client = _FakeMCPClient([])
        fn = mcp_tool_to_function(client, _FakeTool(_spec("do_thing")))
        assert fn.__name__ == "do_thing"
        assert fn.__module__ == "__main__"
        assert getattr(fn, "__mcp_tool__", False) is True

    def test_docstring_has_description_and_args(self):
        tool = _FakeTool(_spec(
            "list_orders", "List orders.",
            properties={"status": {"type": "string", "description": "filter", "enum": ["shipped", "pending"]}},
        ))
        fn = mcp_tool_to_function(_FakeMCPClient([]), tool)
        doc = inspect.getdoc(fn)
        assert "List orders." in doc
        assert "status" in doc
        assert "shipped" in doc  # enum choices surfaced

    def test_call_routes_to_client_and_unwraps(self):
        client = _FakeMCPClient(
            [], results={"add": {"structuredContent": {"result": 7}}}
        )
        tool = _FakeTool(_spec(
            "add", "", properties={"a": {"type": "integer"}, "b": {"type": "integer"}},
            required=["a", "b"],
        ))
        fn = mcp_tool_to_function(client, tool)
        assert fn(3, 4) == 7
        assert client.calls == [("add", {"a": 3, "b": 4})]

    def test_omits_none_arguments(self):
        client = _FakeMCPClient([], results={"list_orders": {"structuredContent": {"result": []}}})
        tool = _FakeTool(_spec("list_orders", properties={"status": {"type": "string"}}))
        fn = mcp_tool_to_function(client, tool)
        fn()  # status defaults to None → should be dropped → arguments None
        assert client.calls == [("list_orders", None)]


# --------------------------------------------------------------------------- #
# MCPToolkit
# --------------------------------------------------------------------------- #
class TestMCPToolkit:
    def test_lists_and_wraps_all_tools(self):
        client = _FakeMCPClient([_spec("a"), _spec("b")])
        tk = MCPToolkit(client)
        assert tk.tool_names == ["a", "b"]
        assert len(tk.domain_specific_code) == 2
        assert tk.usage_instructions  # default usage present

    def test_tool_filter(self):
        client = _FakeMCPClient([_spec("a"), _spec("b"), _spec("c")])
        tk = MCPToolkit(client, tool_filter={"a", "c"})
        assert tk.tool_names == ["a", "c"]

    def test_custom_usage_instructions(self):
        client = _FakeMCPClient([_spec("a")])
        tk = MCPToolkit(client, usage_instructions="custom")
        assert tk.usage_instructions == "custom"


# --------------------------------------------------------------------------- #
# CodeAgent wiring
# --------------------------------------------------------------------------- #
class TestMCPToolkitWithCodeAgent:
    def _agent(self, tk, **kw):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
            return CodeAgent(toolkits=[tk], tmp_dir=False, **kw)

    def test_functions_injected_into_repl(self):
        client = _FakeMCPClient(
            [_spec("add", properties={"a": {"type": "integer"}, "b": {"type": "integer"}}, required=["a", "b"])],
            results={"add": {"structuredContent": {"result": 9}}},
        )
        tk = MCPToolkit(client)
        agent = self._agent(tk)
        out, err = agent.python_repl.execute_code("print(add(4, 5))")
        assert err == ""
        assert "9" in out

    def test_docs_in_system_prompt(self):
        client = _FakeMCPClient([_spec("list_orders", "List all orders.")])
        tk = MCPToolkit(client)
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mi:
            CodeAgent(toolkits=[tk], tmp_dir=False)
            prompt = mi.call_args[1]["system_prompt"]
        assert "list_orders" in prompt
        assert "Domain Specific Code" in prompt

    def test_agentcore_rejects_mcp_functions(self):
        client = _FakeMCPClient([_spec("a")])
        tk = MCPToolkit(client)

        # The guard matches on the class name, so name the fake accordingly.
        AgentCorePythonInterpreter = type("AgentCorePythonInterpreter", (), {})

        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
            with pytest.raises(ValueError, match="AgentCore"):
                CodeAgent(toolkits=[tk], tmp_dir=False, python_interpreter_class=AgentCorePythonInterpreter)
