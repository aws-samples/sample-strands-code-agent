from abc import ABC, abstractmethod

from strands import tool


STDOUT_LABEL = "STDOUT:\n"
STDERR_LABEL = "STDERR:"


class PythonInterpreter(ABC):
    def __init__(self, state_initialization=None, stdout_label=STDOUT_LABEL, stderr_label=STDERR_LABEL, authorized_imports=None, additional_functions=None, timeout_seconds=60):
        self.state_initialization = state_initialization
        self.stdout_label = stdout_label
        self.stderr_label = stderr_label

    @abstractmethod
    def clear_state(self):
        ...

    @abstractmethod
    def execute_code(self, code) -> tuple[str, str]:
        ...

    def get_tool(self):
        @tool
        def python_repl(code: str) -> str:
            """
            Executes Python code in a REPL environment with state persistence.

            Args:
                code: The Python code to execute
            """
            stdout_output, stderr_output = self.execute_code(code)

            observation = []
            if stdout_output:
                observation.append(f"{self.stdout_label}{stdout_output}")
            if stderr_output:
                observation.append(f"{self.stderr_label}{stderr_output}")
            if not observation:
                observation.append("Code executed successfully.")
            return '\n'.join(observation)

        return python_repl
