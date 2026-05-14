"""Tests for strands_code_agent.code_agent — CodeAgent integration tests.

These tests verify that CodeAgent correctly wires toolkits, system prompts,
and the Python REPL together. The Strands Agent base class is mocked to
avoid requiring a live model.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from strands_code_agent.code_agent import CodeAgent, CODE_AGENT_INSTRUCTIONS
from strands_code_agent.toolkits import Toolkit, VISUALIZATION_TOOLKIT, DATA_ANALYSIS_TOOLKIT
from strands_code_agent.python_environments.local_sandboxed import SandboxedPythonInterpreter
from strands_code_agent.python_environments.local_exec import ExecPythonInterpreter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(*, system_prompt=None, tools=None, toolkits=None, tmp_dir=True, **kwargs):
    """Create a CodeAgent with the Strands Agent.__init__ mocked out."""
    with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
        return CodeAgent(
            system_prompt=system_prompt,
            tools=tools,
            toolkits=toolkits,
            tmp_dir=tmp_dir,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

class TestCodeAgentSystemPrompt:
    def test_includes_base_instructions(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(system_prompt="Be helpful.", tmp_dir=False)
            call_kwargs = mock_init.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert "Be helpful." in prompt
            assert "code agent" in prompt.lower()

    def test_none_system_prompt(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(system_prompt=None, tmp_dir=False)
            call_kwargs = mock_init.call_args[1]
            prompt = call_kwargs["system_prompt"]
            assert CODE_AGENT_INSTRUCTIONS in prompt

    def test_usage_instructions_in_prompt(self):
        tk = Toolkit(usage_instructions="Always use pandas for CSV files.")
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(toolkits=[tk], tmp_dir=False)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "Always use pandas for CSV files." in prompt

    def test_code_preamble_in_prompt(self):
        tk = Toolkit(initialization_code="import json")
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(toolkits=[tk], tmp_dir=False)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "import json" in prompt

    def test_tmp_dir_in_prompt(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            agent = CodeAgent(tmp_dir=True)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "temporary directory" in prompt.lower() or agent.tmp_dir in prompt

    def test_no_tmp_dir(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(tmp_dir=False)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "/tmp/" not in prompt


# ---------------------------------------------------------------------------
# Toolkit wiring
# ---------------------------------------------------------------------------

class TestCodeAgentToolkitWiring:
    def test_authorized_imports_from_toolkit(self):
        tk = Toolkit(libraries=["json", "os"])
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "json" in agent.python_repl.authorized_imports
        assert "os" in agent.python_repl.authorized_imports

    def test_initialization_code_runs(self):
        tk = Toolkit(
            libraries=["json"],
            initialization_code="INIT_FLAG = True",
        )
        agent = _make_agent(toolkits=[tk], tmp_dir=False)
        stdout, stderr = agent.python_repl.execute_code("print(INIT_FLAG)")
        assert "True" in stdout

    def test_no_toolkits(self):
        agent = _make_agent(toolkits=None, tmp_dir=False)
        # Should still have a working REPL
        stdout, stderr = agent.python_repl.execute_code("print(1+1)")
        assert "2" in stdout

    def test_init_code_imports_auto_authorized(self):
        """Imports in initialization_code are auto-authorized even without explicit libraries."""
        tk = Toolkit(initialization_code="import json")
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "json" in agent.python_repl.authorized_imports
        stdout, _ = agent.python_repl.execute_code("print(json.dumps({'a': 1}))")
        assert '"a"' in stdout

    def test_init_code_from_import_auto_authorized(self):
        tk = Toolkit(initialization_code="from datetime import date")
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "datetime" in agent.python_repl.authorized_imports

    def test_from_import_submodule_auto_authorized(self):
        """'from scipy import stats' must authorize 'scipy.stats' so the agent can use stats.*."""
        tk = Toolkit(initialization_code="from scipy import stats")
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "scipy" in agent.python_repl.authorized_imports
        assert "scipy.stats" in agent.python_repl.authorized_imports
        stdout, stderr = agent.python_repl.execute_code("print(stats.pearsonr([1,2,3], [4,5,6]))")
        assert stderr == ""
        assert "pvalue" in stdout.lower() or "PearsonR" in stdout

    def test_wildcard_library_authorizes_submodules(self):
        """'scipy.*' in libraries allows importing any scipy submodule without listing each one."""
        tk = Toolkit(libraries=["scipy.*"])
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "scipy.*" in agent.python_repl.authorized_imports
        stdout, stderr = agent.python_repl.execute_code(
            "from scipy import stats; print(stats.pearsonr([1,2,3], [4,5,6]))"
        )
        assert stderr == ""
        assert "PearsonR" in stdout

    def test_wildcard_library_authorizes_deep_submodules(self):
        """'numpy.*' allows access to deeply nested submodules like numpy.random."""
        tk = Toolkit(libraries=["numpy.*"], initialization_code="import numpy as np")
        agent = _make_agent(toolkits=[tk], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        stdout, stderr = agent.python_repl.execute_code("print(np.random.randint(0, 10))")
        assert stderr == ""


# ---------------------------------------------------------------------------
# Domain-specific code
# ---------------------------------------------------------------------------

def _sample_func(x: int) -> int:
    """Double the input."""
    return x * 2


class _SampleClass:
    """A sample helper class."""
    def greet(self) -> str:
        return "hello"


class TestCodeAgentDomainSpecificCode:
    def test_function_available_in_repl(self):
        tk = Toolkit(domain_specific_code=[_sample_func])
        agent = _make_agent(toolkits=[tk], tmp_dir=False)
        stdout, stderr = agent.python_repl.execute_code("print(_sample_func(5))")
        assert "10" in stdout

    def test_class_available_in_repl(self):
        tk = Toolkit(domain_specific_code=[_SampleClass])
        agent = _make_agent(toolkits=[tk], tmp_dir=False)
        stdout, stderr = agent.python_repl.execute_code("print(_SampleClass().greet())")
        assert "hello" in stdout

    def test_domain_specific_doc_in_prompt(self):
        tk = Toolkit(domain_specific_code=[_sample_func])
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(toolkits=[tk], tmp_dir=False)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "Domain Specific Code" in prompt
            assert "_sample_func" in prompt

    def test_no_domain_specific_code_no_section(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(tmp_dir=False)
            prompt = mock_init.call_args[1]["system_prompt"]
            assert "Domain Specific Code" not in prompt


# ---------------------------------------------------------------------------
# Tools wiring
# ---------------------------------------------------------------------------

class TestCodeAgentTools:
    def test_python_repl_tool_always_present(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(tmp_dir=False)
            tools = mock_init.call_args[1]["tools"]
            tool_names = [getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools]
            assert any("python_repl" in n for n in tool_names)

    def test_additional_tools_preserved(self):
        extra_tool = MagicMock()
        extra_tool.tool_name = "my_tool"
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(tools=[extra_tool], tmp_dir=False)
            tools = mock_init.call_args[1]["tools"]
            assert extra_tool in tools
            assert len(tools) == 2  # extra_tool + python_repl

    def test_kwargs_forwarded(self):
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(tmp_dir=False, callback_handler="my_handler")
            call_kwargs = mock_init.call_args[1]
            assert call_kwargs["callback_handler"] == "my_handler"


# ---------------------------------------------------------------------------
# Temp directory
# ---------------------------------------------------------------------------

class TestCodeAgentTmpDir:
    def test_tmp_dir_created(self):
        agent = _make_agent(tmp_dir=True)
        assert hasattr(agent, "tmp_dir")
        assert os.path.isdir(agent.tmp_dir)

    def test_tmp_dir_under_tmp(self):
        agent = _make_agent(tmp_dir=True)
        assert agent.tmp_dir.startswith("/tmp")

    def test_no_tmp_dir_attribute_when_disabled(self):
        agent = _make_agent(tmp_dir=False)
        assert not hasattr(agent, "tmp_dir")


# ---------------------------------------------------------------------------
# Combining multiple toolkits
# ---------------------------------------------------------------------------

class TestCodeAgentCombinedToolkits:
    def test_toolkit_with_overlapping_libraries(self):
        tk1 = Toolkit(libraries=["json", "os"])
        tk2 = Toolkit(libraries=["os", "sys"])
        agent = _make_agent(toolkits=[tk1, tk2], tmp_dir=False, python_interpreter_class=SandboxedPythonInterpreter)
        assert "json" in agent.python_repl.authorized_imports
        assert "os" in agent.python_repl.authorized_imports
        assert "sys" in agent.python_repl.authorized_imports

    def test_combined_initialization_code(self):
        tk1 = Toolkit(initialization_code="A = 1")
        tk2 = Toolkit(initialization_code="B = 2")
        agent = _make_agent(toolkits=[tk1, tk2], tmp_dir=False)
        stdout, _ = agent.python_repl.execute_code("print(A + B)")
        assert "3" in stdout


# ---------------------------------------------------------------------------
# python_interpreter_class selection
# ---------------------------------------------------------------------------

class TestCodeAgentInterpreterClass:
    def test_default_is_sandboxed(self):
        agent = _make_agent(tmp_dir=False)
        assert isinstance(agent.python_repl, SandboxedPythonInterpreter)

    def test_exec_when_requested(self):
        agent = _make_agent(tmp_dir=False, python_interpreter_class=ExecPythonInterpreter)
        assert isinstance(agent.python_repl, ExecPythonInterpreter)

    def test_both_interpreters_execute(self):
        for cls in [ExecPythonInterpreter, SandboxedPythonInterpreter]:
            agent = _make_agent(tmp_dir=False, python_interpreter_class=cls)
            stdout, _ = agent.python_repl.execute_code("print(1+1)")
            assert "2" in stdout
