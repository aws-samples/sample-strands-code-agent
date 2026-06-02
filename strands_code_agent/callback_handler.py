from typing import Any
import ast

from rich.syntax import Syntax
from rich.json import JSON
from rich.markdown import Markdown
from rich.pretty import Pretty
from rich.console import Console


def format_message(text):
    # Python data-structure
    try:
        return Pretty(ast.literal_eval(text))
    except (ValueError, SyntaxError):
        pass

    # JSON?
    try:
        return JSON(text, indent=4)
    except ValueError:
        pass

    # Markdown? (rough heuristic)
    import re
    if re.search(r"(^#{1,6} |\*\*|__|\[.+\]\(.+\)|```)", text, re.MULTILINE):
        return Markdown(text)

    return text


class CodeAgentCallbackHandler:
    def __init__(self, code_tools=None, output_prefix="STDOUT:", format_text=True, **kwargs) -> None:
        self.console = Console()
        if code_tools is None:
            code_tools = {
                'python_repl': 'python'
            }
        self.code_tools = code_tools
        self.output_prefix = output_prefix
        if format_text:
            self.format_text = format_message
        else:
            self.format_text = lambda x: x

    def __call__(self, **kwargs: Any) -> None:
        if 'message' not in kwargs:
            return
        
        message = kwargs['message']
        role = message['role']
        for content_item in  message['content']:
            if 'text' in content_item:
                self.console.print(f"\n[{role.title()}]", end=" ")
                self.console.print(self.format_text(content_item['text'].strip()), end="\n\n")

            if 'toolUse' in content_item:
                tool_use = content_item['toolUse']
                name = tool_use['name']
                self.console.print(f"\n[Tool] {name}")
                if name in self.code_tools:
                    language = self.code_tools[name]
                    syntax = Syntax(tool_use['input']['code'], language)
                    self.console.print(syntax)
                else:
                    for var, value in tool_use['input'].items():
                        self.console.print(f"\t- {var}: {value}")
            
            if 'toolResult' in content_item:
                tool_result = content_item['toolResult']
                self.console.print(f"\n[Tool Result] Status: {tool_result['status']}")
                for tool_item in tool_result['content']:
                    if 'text' in tool_item:
                        text = tool_item['text'].strip()
                        if text.startswith(self.output_prefix):
                            body = text.removeprefix(self.output_prefix).strip()
                            self.console.print(self.output_prefix)
                            self.console.print(self.format_text(body))
                        else:
                            self.console.print(text)
