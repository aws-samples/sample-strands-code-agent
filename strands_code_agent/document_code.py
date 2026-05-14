import inspect
from typing import Callable, Type, Union


def format_function(func, indent: str = "") -> str:
    try:
        sig = inspect.signature(func)
        result = f"{indent}def {func.__name__}{sig}:\n"
    except (ValueError, TypeError):
        result = f"{indent}def {func.__name__}(...):\n"
    
    doc = inspect.getdoc(func)
    if doc:
        doc_lines = doc.split('\n')
        result += f'{indent}    """\n'
        for line in doc_lines:
            result += f"{indent}    {line}\n"
        result += f'{indent}    """\n'
    
    result += f"{indent}    ...\n"
    return result


def get_documentation(obj: Union[Callable, Type]) -> str:
    """
    Extract documentation from a Python function or class.
    Returns formatted text suitable for a coding agent.
    """
    if inspect.isclass(obj):
        # Class header with constructor signature
        try:
            sig = inspect.signature(obj)
            output = f"class {obj.__name__}{sig}:\n"
        except (ValueError, TypeError):
            output = f"class {obj.__name__}:\n"
        
        # Class docstring
        class_doc = inspect.getdoc(obj)
        if class_doc:
            output += '    """\n'
            for line in class_doc.split('\n'):
                output += f"    {line}\n"
            output += '    """\n\n'
        
        # Methods (public + key dunder methods)
        important_dunders = {
            '__init__', '__call__', '__enter__', '__exit__',
            '__iter__', '__next__', '__getitem__', '__setitem__',
            '__len__', '__contains__', '__repr__', '__str__'
        }
        
        for name, method in inspect.getmembers(obj, predicate=inspect.isfunction):
            if name.startswith('_') and name not in important_dunders:
                continue
            output += format_function(method, indent="    ") + "\n"
        
        return output.rstrip() + "\n"
    
    elif callable(obj):
        return format_function(obj)
    
    else:
        raise TypeError(f"Expected a function or class, got {type(obj)}")
