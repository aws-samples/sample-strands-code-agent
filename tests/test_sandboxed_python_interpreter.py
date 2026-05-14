"""Tests for strands_code_agent.python_environment — PythonInterpreter and get_import_string."""

from strands_code_agent.python_environments.base import STDOUT_LABEL, STDERR_LABEL
from strands_code_agent.python_environments.local_sandboxed import SandboxedPythonInterpreter
from strands_code_agent.imports import get_import_string


# ---------------------------------------------------------------------------
# get_import_string
# ---------------------------------------------------------------------------

class TestGetImportString:
    def test_single_symbol(self):
        def dummy():
            pass
        dummy.__module__ = "my_module"
        dummy.__qualname__ = "dummy"
        result = get_import_string([dummy])
        assert result == "from my_module import dummy"

    def test_multiple_symbols_same_module(self):
        def a():
            pass
        def b():
            pass
        a.__module__ = "mod"
        a.__qualname__ = "a"
        b.__module__ = "mod"
        b.__qualname__ = "b"
        result = get_import_string([a, b])
        assert "from mod import" in result
        assert "a" in result
        assert "b" in result

    def test_multiple_modules(self):
        def x():
            pass
        def y():
            pass
        x.__module__ = "alpha"
        x.__qualname__ = "x"
        y.__module__ = "beta"
        y.__qualname__ = "y"
        result = get_import_string([x, y])
        assert "from alpha import x" in result
        assert "from beta import y" in result

    def test_skips_main_module(self):
        def local():
            pass
        local.__module__ = "__main__"
        local.__qualname__ = "local"
        result = get_import_string([local])
        assert result == ""

    def test_deduplicates_same_symbol(self):
        def dup():
            pass
        dup.__module__ = "mod"
        dup.__qualname__ = "dup"
        result = get_import_string([dup, dup])
        assert result.count("dup") == 1

    def test_class_method_uses_top_level_name(self):
        def method():
            pass
        method.__module__ = "mod"
        method.__qualname__ = "MyClass.method"
        result = get_import_string([method])
        assert "from mod import MyClass" in result

    def test_empty_list(self):
        assert get_import_string([]) == ""


# ---------------------------------------------------------------------------
# PythonInterpreter — basic execution
# ---------------------------------------------------------------------------

class TestPythonInterpreterExecution:
    def test_simple_print(self):
        interp = SandboxedPythonInterpreter()
        stdout, stderr = interp.execute_code("print('hello')")
        assert "hello" in stdout
        assert stderr == ""

    def test_arithmetic(self):
        interp = SandboxedPythonInterpreter()
        stdout, stderr = interp.execute_code("print(2 + 3)")
        assert "5" in stdout

    def test_syntax_error(self):
        interp = SandboxedPythonInterpreter()
        stdout, stderr = interp.execute_code("def")
        assert stderr != ""

    def test_runtime_error(self):
        interp = SandboxedPythonInterpreter()
        stdout, stderr = interp.execute_code("1 / 0")
        assert stderr != ""

    def test_undefined_variable(self):
        interp = SandboxedPythonInterpreter()
        _, stderr = interp.execute_code("print(undefined_var)")
        assert stderr != ""

    def test_multiline_code(self):
        interp = SandboxedPythonInterpreter()
        code = "x = 10\ny = 20\nprint(x + y)"
        stdout, stderr = interp.execute_code(code)
        assert "30" in stdout
        assert stderr == ""


# ---------------------------------------------------------------------------
# PythonInterpreter — state persistence within a session
# ---------------------------------------------------------------------------

class TestPythonInterpreterState:
    def test_state_persists_across_calls(self):
        interp = SandboxedPythonInterpreter()
        interp.execute_code("x = 42")
        stdout, stderr = interp.execute_code("print(x)")
        assert "42" in stdout

    def test_clear_state_resets(self):
        interp = SandboxedPythonInterpreter()
        interp.execute_code("x = 42")
        interp.clear_state()
        _, stderr = interp.execute_code("print(x)")
        assert stderr != ""


# ---------------------------------------------------------------------------
# PythonInterpreter — initialization code
# ---------------------------------------------------------------------------

class TestPythonInterpreterInit:
    def test_initialization_code_runs(self):
        interp = SandboxedPythonInterpreter(state_initialization="MY_VAR = 99")
        stdout, _ = interp.execute_code("print(MY_VAR)")
        assert "99" in stdout

    def test_initialization_code_after_clear(self):
        interp = SandboxedPythonInterpreter(state_initialization="MY_VAR = 99")
        interp.clear_state()
        stdout, _ = interp.execute_code("print(MY_VAR)")
        assert "99" in stdout

    def test_no_initialization_code(self):
        interp = SandboxedPythonInterpreter()
        # Should not raise
        stdout, stderr = interp.execute_code("print('ok')")
        assert "ok" in stdout


# ---------------------------------------------------------------------------
# PythonInterpreter — authorized imports
# ---------------------------------------------------------------------------

class TestPythonInterpreterImports:
    def test_authorized_import_succeeds(self):
        interp = SandboxedPythonInterpreter(authorized_imports=["json"])
        stdout, stderr = interp.execute_code("import json; print(json.dumps({'a': 1}))")
        assert '"a"' in stdout
        assert stderr == ""

    def test_unauthorized_import_fails(self):
        interp = SandboxedPythonInterpreter(authorized_imports=[])
        _, stderr = interp.execute_code("import subprocess")
        assert stderr != ""


# ---------------------------------------------------------------------------
# PythonInterpreter — additional_functions
# ---------------------------------------------------------------------------

class TestPythonInterpreterAdditionalFunctions:
    def test_additional_function_available(self):
        def double(x):
            return x * 2
        interp = SandboxedPythonInterpreter(additional_functions={"double": double})
        stdout, stderr = interp.execute_code("print(double(5))")
        assert "10" in stdout
        assert stderr == ""


# ---------------------------------------------------------------------------
# PythonInterpreter — get_tool (python_repl)
# ---------------------------------------------------------------------------

class TestPythonReplTool:
    def test_tool_returns_stdout(self):
        interp = SandboxedPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="print('hi')")
        assert STDOUT_LABEL in result or "hi" in result

    def test_tool_returns_stderr(self):
        interp = SandboxedPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="1/0")
        assert STDERR_LABEL in result or "ZeroDivision" in result

    def test_tool_success_message(self):
        interp = SandboxedPythonInterpreter()
        tool_fn = interp.get_tool()
        result = tool_fn(code="x = 1")
        assert "Code executed successfully" in result

    def test_custom_labels(self):
        interp = SandboxedPythonInterpreter(stdout_label="OUT:", stderr_label="ERR:")
        tool_fn = interp.get_tool()
        result = tool_fn(code="print('test')")
        assert "OUT:" in result


# ---------------------------------------------------------------------------
# PythonInterpreter — __str__
# ---------------------------------------------------------------------------

class TestPythonInterpreterStr:
    def test_str_with_all_fields(self):
        interp = SandboxedPythonInterpreter(
            state_initialization="x = 1",
            authorized_imports=["json"],
            additional_functions={"foo": lambda: None},
        )
        s = str(interp)
        assert "json" in s
        assert "foo" in s
        assert "x = 1" in s

    def test_str_empty(self):
        interp = SandboxedPythonInterpreter()
        s = str(interp)
        # SandboxedPythonInterpreter always includes EXTRA_BUILTINS
        assert "Additional Functions" in s

    def test_str_imports_only(self):
        interp = SandboxedPythonInterpreter(authorized_imports=["os"])
        s = str(interp)
        assert "os" in s
        assert "Init Code" not in s
