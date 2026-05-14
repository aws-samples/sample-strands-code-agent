import tempfile

from strands import Agent
from jinja2 import Template

from strands_code_agent.document_code import get_documentation
from strands_code_agent.python_environments.local_sandboxed import SandboxedPythonInterpreter
from strands_code_agent.imports import get_import_string, extract_imports


CODE_AGENT_INSTRUCTIONS = """
You are a code agent. You solve tasks by writing and executing Python code using the python_repl tool.

The Python interpreter state resets completely with each new user message, but it persists across multiple tool invocations within a single response.
You can perform multi-step computation within a single turn, but do not assume that results from a previous turn are still in memory.
"""

CODE_PREAMBLE_TEMPLATE = Template("""
The python environment of the python_repl tool is initialised with the following code (you do not need to rewrite this code):
```python
{{CODE_PREAMBLE}}
```
""")

TEMP_DIR_TEMPLATE = Template("""
If you need to generate a file use this temporary directory: {{TEMP_DIR}}
The user has no access to this temporary directory.
""")

DOMAIN_SPECIFIC_DOC_TEMPLATE = Template("""
You can use the following Domain Specific Code:
{{SYMBOLS_DOCUMENTATION}}
""")


class CodeAgent(Agent):
    """A coding agent that extends Strands Agent with a sandboxed Python REPL and domain-specific symbol documentation.

    CodeAgent wraps a :class:`PythonInterpreter` as a built-in ``python_repl`` tool and
    assembles a system prompt from the provided toolkits. Each :class:`Toolkit` can contribute:

    - **libraries** – module names authorized for import in the sandboxed interpreter.
    - **initialization_code** – Python code executed at interpreter startup (e.g. imports, config).
      Any modules imported in this code are automatically authorized.
    - **usage_instructions** – free-text guidance appended to the system prompt.
    - **domain_specific_code** – callable symbols whose source and docstrings are documented in
      the system prompt and made available in the interpreter.

    The interpreter state persists across tool invocations within a single agent turn but
    resets completely between user messages.

    Args:
        system_prompt: Optional base system prompt prepended before the coding instructions.

        tools: Additional tools to include alongside the built-in Python REPL.

        toolkits: :class:`Toolkit` instances that supply libraries, initialization code,
            usage instructions, and domain-specific symbols to the REPL environment.

        tmp_dir: If ``True`` (default), creates a temporary directory under ``/tmp`` and
            documents its path in the system prompt so the agent can write files there.

        timeout_seconds: Maximum execution time in seconds for each ``python_repl``
            invocation. Defaults to ``60``.

        python_interpreter_class: The :class:`PythonInterpreter` subclass to use for
            code execution. Defaults to :class:`ExecPythonInterpreter` (lightweight,
            unrestricted ``exec()``-based). Use :class:`SandboxedPythonInterpreter`
            for import restrictions and sandboxed execution.

        **kwargs: Additional arguments forwarded to the Strands :class:`Agent` base class
            (e.g. ``model``, ``callback_handler``).
    """
    def __init__(self,
                 system_prompt:str|None=None,
                 tools:list|None=None,
                 toolkits:list|None=None,
                 tmp_dir=True,
                 timeout_seconds=60,
                 python_interpreter_class=SandboxedPythonInterpreter,
                 **kwargs):
        authorized_imports = set()
        initialization_code = []
        usage_instructions = []
        domain_specific_code = []
        if toolkits is not None:
            for toolkit in toolkits:
                if toolkit.libraries is not None:
                    authorized_imports.update(toolkit.libraries)
                if toolkit.initialization_code is not None:
                    initialization_code.append(toolkit.initialization_code.strip())
                if toolkit.usage_instructions is not None:
                    usage_instructions.append(toolkit.usage_instructions.strip())
                if toolkit.domain_specific_code is not None:
                    domain_specific_code.extend(toolkit.domain_specific_code)

        additional_functions = {}
        domain_specific_doc = ""
        if domain_specific_code:
            authorized_imports.update(sym.__module__ for sym in domain_specific_code if sym.__module__ != "__main__")
            initialization_code.append(get_import_string(domain_specific_code))
            sym_doc =  "\n".join([get_documentation(sym) for sym in domain_specific_code])
            domain_specific_doc = DOMAIN_SPECIFIC_DOC_TEMPLATE.render(SYMBOLS_DOCUMENTATION=sym_doc)
            additional_functions = {sym.__qualname__.split(".")[0]: sym for sym in domain_specific_code}
        
        code_preamble = "\n".join(initialization_code)
        # Auto-authorize any modules imported in initialization code so users
        # don't have to duplicate them in both `libraries` and `initialization_code`.
        authorized_imports.update(extract_imports(code_preamble))
        code_preamble_doc = CODE_PREAMBLE_TEMPLATE.render(CODE_PREAMBLE=code_preamble) if code_preamble else ""

        tmp_dir_doc = ""
        if tmp_dir:
            self.tmp_dir = tempfile.mkdtemp(dir='/tmp')
            tmp_dir_doc = TEMP_DIR_TEMPLATE.render(TEMP_DIR=self.tmp_dir)

        system_prompt = '\n'.join([
            system_prompt if system_prompt is not None else "",
            CODE_AGENT_INSTRUCTIONS,
            code_preamble_doc,
            "\n".join(usage_instructions),
            tmp_dir_doc,
            domain_specific_doc
        ])

        self.python_repl = python_interpreter_class(
            code_preamble,
            authorized_imports=authorized_imports,
            additional_functions=additional_functions,
            timeout_seconds=timeout_seconds,
        )
        python_repl_tool = self.python_repl.get_tool()
        if tools is not None:
            tools.append(python_repl_tool)
        else:
            tools = [python_repl_tool]
        
        kwargs.update({
            "system_prompt": system_prompt,
            "tools": tools
        })
        super().__init__(**kwargs)
