import io
from contextlib import redirect_stdout, redirect_stderr

from strands_code_agent.python_environments.base import PythonInterpreter, STDOUT_LABEL, STDERR_LABEL


class ExecPythonInterpreter(PythonInterpreter):
    def __init__(self, state_initialization=None, stdout_label=STDOUT_LABEL, stderr_label=STDERR_LABEL, **kwargs):
        super().__init__(state_initialization, stdout_label, stderr_label)
        
        self.state = {}
        if self.state_initialization:
            self.execute_code(self.state_initialization)
    
    def clear_state(self):
        self.state.clear()
        if self.state_initialization:
            self.execute_code(self.state_initialization)
    
    def execute_code(self, code):
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.state)
        except Exception as e:
            stderr_buffer.write(str(e))
        return stdout_buffer.getvalue().strip(), stderr_buffer.getvalue().strip()
