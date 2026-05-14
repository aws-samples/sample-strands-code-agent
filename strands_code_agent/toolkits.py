from dataclasses import dataclass


@dataclass
class Toolkit:
    """A toolkit that bundles libraries, initialization code, usage instructions, and domain-specific
    symbols for use by a CodeAgent's Python REPL environment.

    Args:
        libraries: Library names to authorize as imports in the Python executor.
        initialization_code: Python code snippet to run when initializing the environment.
        usage_instructions: Extra instructions included in the system prompt about how to use the provided libraries.
        domain_specific_code: Callable symbols to be fully imported and documented in the agent's context.
    """
    libraries: list[str] | None = None
    initialization_code: str | None = None
    usage_instructions: str | None = None
    domain_specific_code: list | None = None


VISUALIZATION_TOOLKIT = Toolkit(
    libraries = ['matplotlib.*', 'seaborn.*'],
    initialization_code = """
# Visualization Libraries
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
""",
    usage_instructions="""
Do not try to show any matplotlib image: the python_repl tool executes the code in a sub-process without a GUI.
""")


DATA_ANALYSIS_TOOLKIT = Toolkit(
    libraries = ['numpy.*', 'pandas.*', 'scipy.*', 'datetime'],
    initialization_code = """
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
""")
