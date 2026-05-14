"""Tests for ExecPythonInterpreter — the lightweight exec()-based interpreter."""

import pytest
from strands_code_agent.python_environments.base import PythonInterpreter, STDOUT_LABEL, STDERR_LABEL
from strands_code_agent.python_environments.local_exec import ExecPythonInterpreter


# ---------------------------------------------------------------------------
# Base class is abstract
# ---------------------------------------------------------------------------

class TestPythonInterpreterABC:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError, match="abstract"):
            PythonInterpreter()


# ---------------------------------------------------------------------------
# ExecPythonInterpreter — execution
# ---------------------------------------------------------------------------

class TestExecExecution:
    def test_simple_print(self):
        interp = ExecPythonInterpreter()
        stdout, stderr = interp.execute_code("print('hello')")
        assert "hello" in stdout
        assert stderr == ""

    def test_arithmetic(self):
        interp = ExecPythonInterpreter()
        stdout, _ = interp.execute_code("print(2 + 3)")
        assert "5" in stdout

    def test_syntax_error(self):
        interp = ExecPythonInterpreter()
        _, stderr = interp.execute_code("def")
        assert stderr != ""

    def test_runtime_error(self):
        interp = ExecPythonInterpreter()
        _, stderr = interp.execute_code("1 / 0")
        assert stderr != ""

    def test_undefined_variable(self):
        interp = ExecPythonInterpreter()
        _, stderr = interp.execute_code("print(undefined_var)")
        assert stderr != ""

    def test_multiline_code(self):
        interp = ExecPythonInterpreter()
        stdout, stderr = interp.execute_code("x = 10\ny = 20\nprint(x + y)")
        assert "30" in stdout
        assert stderr == ""

    def test_imports_unrestricted(self):
        """ExecPythonInterpreter does not restrict imports."""
        interp = ExecPythonInterpreter()
        stdout, stderr = interp.execute_code("import json; print(json.dumps({'a': 1}))")
        assert '"a"' in stdout
        assert stderr == ""


# ---------------------------------------------------------------------------
# ExecPythonInterpreter — state
# ---------------------------------------------------------------------------

class TestExecState:
    def test_state_persists(self):
        interp = ExecPythonInterpreter()
        interp.execute_code("x = 42")
        stdout, _ = interp.execute_code("print(x)")
        assert "42" in stdout

    def test_clear_state_resets(self):
        interp = ExecPythonInterpreter()
        interp.execute_code("x = 42")
        interp.clear_state()
        _, stderr = interp.execute_code("print(x)")
        assert stderr != ""

    def test_initialization_code_runs(self):
        interp = ExecPythonInterpreter(state_initialization="MY_VAR = 99")
        stdout, _ = interp.execute_code("print(MY_VAR)")
        assert "99" in stdout

    def test_initialization_code_after_clear(self):
        interp = ExecPythonInterpreter(state_initialization="MY_VAR = 99")
        interp.clear_state()
        stdout, _ = interp.execute_code("print(MY_VAR)")
        assert "99" in stdout


# ---------------------------------------------------------------------------
# ExecPythonInterpreter — get_tool
# ---------------------------------------------------------------------------

class TestExecTool:
    def test_tool_returns_stdout(self):
        interp = ExecPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="print('hi')")
        assert "hi" in result

    def test_tool_returns_stderr(self):
        interp = ExecPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="1/0")
        assert STDERR_LABEL in result or "ZeroDivision" in result

    def test_tool_success_message(self):
        interp = ExecPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="x = 1")
        assert "Code executed successfully" in result

    def test_custom_labels(self):
        interp = ExecPythonInterpreter(stdout_label="OUT:", stderr_label="ERR:")
        tool_fn = interp.get_tool()
        result = tool_fn(code="print('test')")
        assert "OUT:" in result

    def test_ignores_extra_kwargs(self):
        """ExecPythonInterpreter accepts and ignores kwargs like authorized_imports."""
        interp = ExecPythonInterpreter(
            authorized_imports=["json"],
            additional_functions={"foo": lambda: None},
            timeout_seconds=30,
        )
        stdout, _ = interp.execute_code("print('ok')")
        assert "ok" in stdout
