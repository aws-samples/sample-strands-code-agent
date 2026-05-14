import warnings

from smolagents.local_python_executor import LocalPythonExecutor

from strands_code_agent.python_environments.base import PythonInterpreter, STDOUT_LABEL, STDERR_LABEL


# Python builtins that smolagents' executor doesn't allow-list by default
EXTRA_BUILTINS = {"repr": repr, "ascii": ascii, "ord": ord, "chr": chr, "hex": hex, "oct": oct, "bin": bin}


class SandboxedPythonInterpreter(PythonInterpreter):
    def __init__(self, state_initialization=None, stdout_label=STDOUT_LABEL, stderr_label=STDERR_LABEL, authorized_imports=None, additional_functions=None, timeout_seconds=60):
        super().__init__(state_initialization, stdout_label, stderr_label)

        self.authorized_imports = authorized_imports or []
        self.additional_functions = {**EXTRA_BUILTINS, **(additional_functions or {})}
        self.timeout_seconds = timeout_seconds

        self._init_executor()

    def _init_executor(self):
        # smolagents' LocalPythonExecutor introspects all attributes of authorized modules
        # via getattr(), which triggers DeprecationWarnings on access, not use.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.executor = LocalPythonExecutor(
                additional_authorized_imports=self.authorized_imports,
                additional_functions=self.additional_functions,
                timeout_seconds=self.timeout_seconds,
            )
            self.executor.send_tools({})
            if self.state_initialization:
                self.executor(self.state_initialization)

    def clear_state(self):
        self._init_executor()

    def execute_code(self, code):
        stdout, stderr = "", ""
        try:
            result = self.executor(code)
            stdout = result.logs.strip() if result.logs else ""
        except Exception as e:
            # Salvage any print output captured before the error
            if hasattr(self.executor, "state") and "_print_outputs" in self.executor.state:
                stdout = str(self.executor.state["_print_outputs"]).strip()
            stderr = str(e)
        return stdout, stderr
    
    def __str__(self) -> str:
        buffer = []
        if self.authorized_imports:
            buffer.append(f"# Authorized Imports: {', '.join(self.authorized_imports)}")
        if self.additional_functions:
            buffer.append(f"# Additional Functions: {', '.join(self.additional_functions.keys())}")
        if self.state_initialization:
            buffer.append(f"""# Init Code:
```python
{self.state_initialization}
```
""")
        return '\n'.join(buffer) if buffer else "Standard PythonInterpreter"
