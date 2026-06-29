import inspect
import logging

import boto3

from strands_code_agent.python_environments.base import PythonInterpreter, STDOUT_LABEL, STDERR_LABEL

logger = logging.getLogger(__name__)


def _get_source(obj) -> str:
    """Get source code of a function or class, with a clear error if impossible."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError) as e:
        raise ValueError(
            f"Cannot serialize {obj!r} for remote execution: {e}. "
            "Only functions/classes with inspectable source code can be used as "
            "domain_specific_code with AgentCorePythonInterpreter."
        ) from e


class AgentCorePythonInterpreter(PythonInterpreter):
    """Remote Python interpreter using Amazon Bedrock AgentCore Code Interpreter.

    Executes code in a managed sandbox environment via the AgentCore API.
    Sessions are stateful — variables persist across execute_code() calls
    until clear_state() is called or the session times out.

    Domain-specific functions/classes passed via additional_functions are
    serialized as source code and injected into the remote session at startup.

    Args:
        state_initialization: Code to run at session start (e.g., imports).
        region: AWS region (default: "us-east-1").
        code_interpreter_identifier: Resource ID (default: "aws.codeinterpreter.v1").
        session_timeout_seconds: Session idle timeout (default: 900, max: 28800).
        additional_functions: Dict of {name: callable} to serialize and inject remotely.
    """

    def __init__(
        self,
        state_initialization=None,
        stdout_label=STDOUT_LABEL,
        stderr_label=STDERR_LABEL,
        region="us-east-1",
        code_interpreter_identifier="aws.codeinterpreter.v1",
        session_timeout_seconds=900,
        additional_functions=None,
        **kwargs,
    ):
        # Build the full initialization: user init + serialized domain functions
        full_init = self._build_initialization(state_initialization, additional_functions)
        super().__init__(full_init, stdout_label, stderr_label)
        self._region = region
        self._identifier = code_interpreter_identifier
        self._timeout = session_timeout_seconds
        self._client = None
        self._session_id = None

    @staticmethod
    def _build_initialization(state_initialization, additional_functions):
        """Combine state_initialization with serialized domain-specific code.

        Symbols from builtins or installed packages (site-packages / stdlib)
        are skipped since they're available on the remote server. All other
        symbols (user-defined, from local modules) are serialized as source code.
        """
        import sys

        parts = []
        if additional_functions:
            # Paths that indicate "not user code" — stdlib + site-packages
            stdlib_paths = {p for p in sys.path if "site-packages" in p}
            stdlib_paths.update(p for p in sys.path if "lib/python" in p or "lib\\python" in p)

            sources = []
            for name, obj in additional_functions.items():
                if not callable(obj):
                    continue
                module = getattr(obj, "__module__", None)
                if module == "builtins":
                    continue
                try:
                    source_file = inspect.getfile(obj)
                except (TypeError, OSError):
                    continue
                # Skip if source is in stdlib or site-packages
                if any(source_file.startswith(p) for p in stdlib_paths if p):
                    continue
                sources.append(_get_source(obj))
            if sources:
                parts.append("\n".join(sources))
        if state_initialization:
            parts.append(state_initialization)
        return "\n".join(parts) if parts else None

    def _ensure_session(self):
        if self._session_id is not None:
            return
        self._client = boto3.client("bedrock-agentcore", region_name=self._region)
        resp = self._client.start_code_interpreter_session(
            codeInterpreterIdentifier=self._identifier,
            sessionTimeoutSeconds=self._timeout,
        )
        self._session_id = resp["sessionId"]
        logger.info("Started AgentCore session: %s", self._session_id)
        if self.state_initialization:
            self._run_code(self.state_initialization)

    def execute_code(self, code) -> tuple[str, str]:
        self._ensure_session()
        return self._run_code(code)

    def _run_code(self, code: str) -> tuple[str, str]:
        response = self._client.invoke_code_interpreter(
            codeInterpreterIdentifier=self._identifier,
            sessionId=self._session_id,
            name="executeCode",
            arguments={"language": "python", "code": code},
        )
        stdout_parts, stderr_parts = [], []
        for event in response["stream"]:
            result = event.get("result", {})
            structured = result.get("structuredContent", {})
            if structured.get("stdout"):
                stdout_parts.append(structured["stdout"])
            if structured.get("stderr"):
                stderr_parts.append(structured["stderr"])
        return "".join(stdout_parts).strip(), "".join(stderr_parts).strip()

    def clear_state(self):
        self._stop_session()

    def close(self):
        self._stop_session()

    def _stop_session(self):
        if self._session_id and self._client:
            try:
                self._client.stop_code_interpreter_session(
                    codeInterpreterIdentifier=self._identifier,
                    sessionId=self._session_id,
                )
                logger.info("Stopped AgentCore session: %s", self._session_id)
            except Exception as e:
                logger.warning("Failed to stop session %s: %s", self._session_id, e)
        self._session_id = None

    def __del__(self):
        self._stop_session()
