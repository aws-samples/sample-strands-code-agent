import ast
from typing import Callable, Type, Union
from collections import defaultdict


def extract_imports(code: str) -> set[str]:
    """Extract top-level module names from import statements in a code snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.add(node.module)
                for alias in node.names:
                    modules.add(f"{node.module}.{alias.name}")
    return modules


def get_import_string(symbols: list[Union[Callable, Type]]) -> str:
    """Generate minimal import statements for a list of symbols."""
    by_module: dict[str, list[str]] = defaultdict(list)
    for sym in symbols:
        if sym.__module__ == "__main__":
            continue
        by_module[sym.__module__].append(sym.__qualname__.split(".")[0])

    lines = []
    for module, names in sorted(by_module.items()):
        lines.append(f"from {module} import {', '.join(sorted(set(names)))}")
    return "\n".join(lines)
