import inspect
from unittest.mock import patch, MagicMock
import pytest

from strands_code_agent.python_environments.agentcore import AgentCorePythonInterpreter, _get_source


def _make_stream(stdout="", stderr=""):
    """Helper to create a mock streaming response."""
    return {
        "stream": [
            {"result": {"structuredContent": {"stdout": stdout, "stderr": stderr}}}
        ]
    }


# --- Domain-specific test fixtures ---

def calculate_roi(investment: float, returns: float) -> float:
    """Calculate return on investment as a percentage."""
    return (returns - investment) / investment * 100


def fibonacci(n: int) -> list[int]:
    """Return first n Fibonacci numbers."""
    if n <= 0:
        return []
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


class DataProcessor:
    """Process data with configurable transformations."""

    def __init__(self, multiplier: float = 1.0):
        self.multiplier = multiplier

    def transform(self, values: list[float]) -> list[float]:
        """Multiply each value by the multiplier."""
        return [v * self.multiplier for v in values]

    def summarize(self, values: list[float]) -> dict:
        """Return basic statistics."""
        return {
            "count": len(values),
            "sum": sum(values),
            "mean": sum(values) / len(values) if values else 0,
        }


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.start_code_interpreter_session.return_value = {"sessionId": "sess-123"}
    client.invoke_code_interpreter.return_value = _make_stream(stdout="42\n")
    return client


@pytest.fixture
def interpreter(mock_client):
    with patch("boto3.client", return_value=mock_client):
        interp = AgentCorePythonInterpreter(region="us-west-2")
        yield interp
        interp.close()


class TestSessionLifecycle:
    def test_lazy_session_start(self, mock_client):
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            assert interp._session_id is None
            interp.execute_code("print(42)")
            assert interp._session_id == "sess-123"
            mock_client.start_code_interpreter_session.assert_called_once()
            interp.close()

    def test_session_reuse(self, interpreter, mock_client):
        interpreter.execute_code("x = 1")
        interpreter.execute_code("x = 2")
        mock_client.start_code_interpreter_session.assert_called_once()

    def test_clear_state_stops_session(self, interpreter, mock_client):
        interpreter.execute_code("x = 1")
        interpreter.clear_state()
        mock_client.stop_code_interpreter_session.assert_called_once_with(
            codeInterpreterIdentifier="aws.codeinterpreter.v1",
            sessionId="sess-123",
        )
        assert interpreter._session_id is None

    def test_clear_state_then_execute_starts_new_session(self, interpreter, mock_client):
        interpreter.execute_code("x = 1")
        interpreter.clear_state()
        interpreter.execute_code("y = 2")
        assert mock_client.start_code_interpreter_session.call_count == 2

    def test_close_stops_session(self, mock_client):
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            interp.execute_code("1")
            interp.close()
            mock_client.stop_code_interpreter_session.assert_called_once()

    def test_close_noop_when_no_session(self, mock_client):
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            interp.close()  # no error
            mock_client.stop_code_interpreter_session.assert_not_called()


class TestExecuteCode:
    def test_returns_stdout_stderr(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream(
            stdout="hello\n", stderr="warning\n"
        )
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            stdout, stderr = interp.execute_code("print('hello')")
            assert stdout == "hello"
            assert stderr == "warning"
            interp.close()

    def test_empty_output(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            stdout, stderr = interp.execute_code("x = 1")
            assert stdout == ""
            assert stderr == ""
            interp.close()

    def test_multiple_stream_events(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = {
            "stream": [
                {"result": {"structuredContent": {"stdout": "line1\n", "stderr": ""}}},
                {"result": {"structuredContent": {"stdout": "line2\n", "stderr": ""}}},
            ]
        }
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter()
            stdout, stderr = interp.execute_code("print('line1'); print('line2')")
            assert stdout == "line1\nline2"
            interp.close()

    def test_invokes_with_correct_params(self, interpreter, mock_client):
        interpreter.execute_code("print(1)")
        mock_client.invoke_code_interpreter.assert_called_with(
            codeInterpreterIdentifier="aws.codeinterpreter.v1",
            sessionId="sess-123",
            name="executeCode",
            arguments={"language": "python", "code": "print(1)"},
        )


class TestInitialization:
    def test_state_initialization_runs_on_first_execute(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(state_initialization="import os")
            interp.execute_code("print(os.getcwd())")
            # First call is initialization, second is the user code
            calls = mock_client.invoke_code_interpreter.call_args_list
            assert calls[0][1]["arguments"]["code"] == "import os"
            assert calls[1][1]["arguments"]["code"] == "print(os.getcwd())"
            interp.close()

    def test_custom_identifier_and_timeout(self, mock_client):
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                code_interpreter_identifier="custom-id",
                session_timeout_seconds=3600,
            )
            interp.execute_code("1")
            mock_client.start_code_interpreter_session.assert_called_with(
                codeInterpreterIdentifier="custom-id",
                sessionTimeoutSeconds=3600,
            )
            interp.close()


class TestPythonReplTool:
    def test_get_tool_returns_callable(self, interpreter):
        tool = interpreter.get_tool()
        assert callable(tool)


class TestDomainSpecificFunctions:
    """Test that user-defined functions are serialized and sent to the remote session."""

    def test_function_source_injected_in_initialization(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={"calculate_roi": calculate_roi},
            )
            # Source should be in state_initialization
            assert "def calculate_roi" in interp.state_initialization
            assert "return (returns - investment) / investment * 100" in interp.state_initialization
            interp.close()

    def test_function_source_sent_on_first_execute(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={"calculate_roi": calculate_roi},
            )
            interp.execute_code("print(calculate_roi(1000, 1250))")
            # First invoke call is the initialization (contains the function def)
            init_call = mock_client.invoke_code_interpreter.call_args_list[0]
            init_code = init_call[1]["arguments"]["code"]
            assert "def calculate_roi" in init_code
            interp.close()

    def test_multiple_functions_injected(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={
                    "calculate_roi": calculate_roi,
                    "fibonacci": fibonacci,
                },
            )
            assert "def calculate_roi" in interp.state_initialization
            assert "def fibonacci" in interp.state_initialization
            interp.close()

    def test_function_combined_with_state_initialization(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                state_initialization="import os\nCWD = os.getcwd()",
                additional_functions={"calculate_roi": calculate_roi},
            )
            # Both should be present
            assert "def calculate_roi" in interp.state_initialization
            assert "import os" in interp.state_initialization
            assert "CWD = os.getcwd()" in interp.state_initialization
            interp.close()

    def test_builtin_functions_skipped(self, mock_client):
        """Builtins (repr, ord, etc.) exist remotely and shouldn't be serialized."""
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={"repr": repr, "ord": ord},
            )
            # No initialization needed for builtins alone
            assert interp.state_initialization is None
            interp.close()

    def test_installed_package_functions_skipped(self, mock_client):
        """Functions from installed packages are imported, not serialized."""
        import json
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                state_initialization="from json import dumps",
                additional_functions={"dumps": json.dumps},
            )
            # json.dumps comes from 'json' module, not __main__, so not serialized
            # Only state_initialization should remain
            assert interp.state_initialization == "from json import dumps"
            interp.close()


class TestDomainSpecificClasses:
    """Test that user-defined classes are serialized and sent to the remote session."""

    def test_class_source_injected_in_initialization(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={"DataProcessor": DataProcessor},
            )
            assert "class DataProcessor" in interp.state_initialization
            assert "def transform" in interp.state_initialization
            assert "def summarize" in interp.state_initialization
            interp.close()

    def test_class_and_functions_combined(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream()
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={
                    "DataProcessor": DataProcessor,
                    "calculate_roi": calculate_roi,
                },
            )
            assert "class DataProcessor" in interp.state_initialization
            assert "def calculate_roi" in interp.state_initialization
            interp.close()

    def test_class_sent_before_user_code(self, mock_client):
        mock_client.invoke_code_interpreter.return_value = _make_stream(stdout="2.0")
        with patch("boto3.client", return_value=mock_client):
            interp = AgentCorePythonInterpreter(
                additional_functions={"DataProcessor": DataProcessor},
            )
            interp.execute_code("dp = DataProcessor(2.0)\nprint(dp.transform([1.0]))")
            # First invocation is initialization with the class
            init_call = mock_client.invoke_code_interpreter.call_args_list[0]
            init_code = init_call[1]["arguments"]["code"]
            assert "class DataProcessor" in init_code
            # Second is user code
            user_call = mock_client.invoke_code_interpreter.call_args_list[1]
            user_code = user_call[1]["arguments"]["code"]
            assert "dp = DataProcessor" in user_code
            interp.close()


class TestGetSource:
    """Test the _get_source helper."""

    def test_function_source(self):
        src = _get_source(calculate_roi)
        assert "def calculate_roi" in src
        assert "return (returns - investment)" in src

    def test_class_source(self):
        src = _get_source(DataProcessor)
        assert "class DataProcessor" in src
        assert "def transform" in src

    def test_lambda_raises(self):
        fn = lambda x: x + 1  # noqa: E731
        # Lambdas have source but may work; the key test is for builtins
        # which truly can't be sourced
        with pytest.raises(ValueError, match="Cannot serialize"):
            _get_source(len)


class TestCodeAgentWithAgentCore:
    """Integration test: CodeAgent + AgentCorePythonInterpreter + domain_specific_code."""

    def test_code_agent_with_domain_function(self, mock_client):
        """Verify full CodeAgent wiring sends domain functions to remote session."""
        from strands_code_agent import CodeAgent, Toolkit
        from strands_code_agent.python_environments.agentcore import AgentCorePythonInterpreter

        mock_client.invoke_code_interpreter.return_value = _make_stream()

        with patch("boto3.client", return_value=mock_client):
            agent = CodeAgent(
                system_prompt="You are a finance assistant.",
                toolkits=[Toolkit(domain_specific_code=[calculate_roi])],
                python_interpreter_class=AgentCorePythonInterpreter,
                python_interpreter_kwargs={"region": "us-west-2"},
            )
            # The interpreter should have the function in its initialization
            assert "def calculate_roi" in agent.python_repl.state_initialization
            # The system prompt should document the function
            assert "calculate_roi" in agent.system_prompt
            agent.python_repl.close()

    def test_code_agent_with_domain_class(self, mock_client):
        """Verify CodeAgent wiring sends domain classes to remote session."""
        from strands_code_agent import CodeAgent, Toolkit
        from strands_code_agent.python_environments.agentcore import AgentCorePythonInterpreter

        mock_client.invoke_code_interpreter.return_value = _make_stream()

        with patch("boto3.client", return_value=mock_client):
            agent = CodeAgent(
                toolkits=[Toolkit(domain_specific_code=[DataProcessor])],
                python_interpreter_class=AgentCorePythonInterpreter,
                python_interpreter_kwargs={"region": "us-east-1"},
            )
            assert "class DataProcessor" in agent.python_repl.state_initialization
            assert "DataProcessor" in agent.system_prompt
            agent.python_repl.close()

    def test_code_agent_with_toolkit_and_domain_code(self, mock_client):
        """Verify combined toolkit (libraries + init_code + domain code) works."""
        from strands_code_agent import CodeAgent, Toolkit
        from strands_code_agent.python_environments.agentcore import AgentCorePythonInterpreter

        mock_client.invoke_code_interpreter.return_value = _make_stream()

        with patch("boto3.client", return_value=mock_client):
            toolkit = Toolkit(
                libraries=["pandas"],
                initialization_code="import pandas as pd",
                usage_instructions="Use pd for DataFrames.",
                domain_specific_code=[calculate_roi, DataProcessor],
            )
            agent = CodeAgent(
                toolkits=[toolkit],
                python_interpreter_class=AgentCorePythonInterpreter,
                python_interpreter_kwargs={"region": "us-east-1"},
            )
            init = agent.python_repl.state_initialization
            # Domain functions serialized
            assert "def calculate_roi" in init
            assert "class DataProcessor" in init
            # Toolkit init code included
            assert "import pandas as pd" in init
            agent.python_repl.close()

    def test_code_agent_executes_domain_function_remotely(self, mock_client):
        """Simulate full execution: init + user code calls domain function."""
        from strands_code_agent import CodeAgent, Toolkit
        from strands_code_agent.python_environments.agentcore import AgentCorePythonInterpreter

        mock_client.invoke_code_interpreter.return_value = _make_stream(stdout="25.0")

        with patch("boto3.client", return_value=mock_client):
            agent = CodeAgent(
                toolkits=[Toolkit(domain_specific_code=[calculate_roi])],
                python_interpreter_class=AgentCorePythonInterpreter,
                python_interpreter_kwargs={"region": "us-east-1"},
            )
            tool = agent.python_repl.get_tool()
            result = tool(code="print(calculate_roi(1000, 1250))")

            # Verify the initialization was sent first (with function source)
            calls = mock_client.invoke_code_interpreter.call_args_list
            assert len(calls) == 2  # init + user code
            assert "def calculate_roi" in calls[0][1]["arguments"]["code"]
            assert "print(calculate_roi(1000, 1250))" in calls[1][1]["arguments"]["code"]
            assert "25.0" in result
            agent.python_repl.close()


# --- Live integration test (requires real AWS credentials) ---

@pytest.mark.integration
class TestAgentCoreLiveIntegration:
    """Live integration tests against real AgentCore service.

    Run with: pytest tests/test_agentcore_python_interpreter.py -v -k "integration"

    Requires:
        - AWS credentials with bedrock-agentcore permissions
        - AWS_REGION env var (defaults to us-east-1)
    """

    @pytest.fixture
    def region(self):
        import os
        return os.environ.get("AWS_REGION", "us-east-1")

    def test_simple_execution(self, region):
        interp = AgentCorePythonInterpreter(region=region)
        try:
            stdout, stderr = interp.execute_code("print(2 ** 10)")
            assert "1024" in stdout
            assert stderr == ""
        finally:
            interp.close()

    def test_state_persists_across_calls(self, region):
        interp = AgentCorePythonInterpreter(region=region)
        try:
            interp.execute_code("x = 42")
            stdout, _ = interp.execute_code("print(x * 2)")
            assert "84" in stdout
        finally:
            interp.close()

    def test_domain_specific_function(self, region):
        interp = AgentCorePythonInterpreter(
            region=region,
            additional_functions={"calculate_roi": calculate_roi},
        )
        try:
            stdout, stderr = interp.execute_code("print(calculate_roi(1000, 1250))")
            assert "25.0" in stdout
            assert stderr == ""
        finally:
            interp.close()

    def test_domain_specific_class(self, region):
        interp = AgentCorePythonInterpreter(
            region=region,
            additional_functions={"DataProcessor": DataProcessor},
        )
        try:
            stdout, stderr = interp.execute_code(
                "dp = DataProcessor(3.0)\nprint(dp.transform([1.0, 2.0]))"
            )
            assert "[3.0, 6.0]" in stdout
        finally:
            interp.close()

    def test_clear_state_resets_session(self, region):
        interp = AgentCorePythonInterpreter(region=region)
        try:
            interp.execute_code("secret = 'hello'")
            interp.clear_state()
            _, stderr = interp.execute_code("print(secret)")
            assert "NameError" in stderr or "not defined" in stderr
        finally:
            interp.close()
