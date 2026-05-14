"""Tests that mirror every code example in README.md.

Each test corresponds to a specific README section and verifies the example
works as documented (with the model mocked out).
"""

import os
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Quick Start — public API imports
# ---------------------------------------------------------------------------

class TestReadmeQuickStart:
    def test_public_imports(self):
        from strands_code_agent import CodeAgent, Toolkit
        assert CodeAgent is not None
        assert Toolkit is not None

    def test_code_agent_with_system_prompt(self):
        from strands_code_agent import CodeAgent
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
            agent = CodeAgent(system_prompt="You are a helpful data analyst.")
        # The REPL should be functional
        stdout, _ = agent.python_repl.execute_code("print(2 ** 10)")
        assert "1024" in stdout


# ---------------------------------------------------------------------------
# Toolkit — construction with all fields (README "Toolkit" section)
# ---------------------------------------------------------------------------

class TestReadmeToolkitConstruction:
    def test_visualization_toolkit_pattern(self):
        from strands_code_agent.toolkits import Toolkit

        tk = Toolkit(
            libraries=["matplotlib", "matplotlib.pyplot", "seaborn"],
            initialization_code="""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
""",
            usage_instructions="Do not try to show any matplotlib image.",
        )
        assert "matplotlib" in tk.libraries
        assert "Agg" in tk.initialization_code
        assert "Do not" in tk.usage_instructions


# ---------------------------------------------------------------------------
# Built-in Toolkits — importable
# ---------------------------------------------------------------------------

class TestReadmeBuiltinToolkits:
    def test_builtin_toolkits_importable(self):
        from strands_code_agent.toolkits import (
            VISUALIZATION_TOOLKIT,
            DATA_ANALYSIS_TOOLKIT,
        )
        assert VISUALIZATION_TOOLKIT.libraries is not None
        assert DATA_ANALYSIS_TOOLKIT.libraries is not None


# ---------------------------------------------------------------------------
# Domain-Specific Code — calculate_roi example
# ---------------------------------------------------------------------------

def calculate_roi(investment: float, returns: float) -> float:
    """Calculate return on investment as a percentage."""
    return (returns - investment) / investment * 100


class TestReadmeDomainSpecificCode:
    def test_calculate_roi_available_in_repl(self):
        from strands_code_agent import CodeAgent, Toolkit
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
            agent = CodeAgent(
                system_prompt="You are a finance assistant.",
                toolkits=[Toolkit(domain_specific_code=[calculate_roi])],
                tmp_dir=False,
            )
        stdout, stderr = agent.python_repl.execute_code(
            "print(calculate_roi(1000, 1250))"
        )
        assert "25" in stdout
        assert stderr == ""

    def test_calculate_roi_documented_in_prompt(self):
        from strands_code_agent import CodeAgent, Toolkit
        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None) as mock_init:
            CodeAgent(
                system_prompt="You are a finance assistant.",
                toolkits=[Toolkit(domain_specific_code=[calculate_roi])],
                tmp_dir=False,
            )
        prompt = mock_init.call_args[1]["system_prompt"]
        assert "calculate_roi" in prompt
        assert "investment" in prompt
        assert "return on investment" in prompt.lower()


# ---------------------------------------------------------------------------
# Combining Toolkits — DATA_ANALYSIS_TOOLKIT + VISUALIZATION_TOOLKIT
# ---------------------------------------------------------------------------

class TestReadmeCombiningToolkits:
    def test_data_analysis_plus_visualization(self):
        from strands_code_agent import CodeAgent
        from strands_code_agent.toolkits import DATA_ANALYSIS_TOOLKIT, VISUALIZATION_TOOLKIT
        from strands_code_agent.python_environments.local_sandboxed import SandboxedPythonInterpreter

        with patch("strands_code_agent.code_agent.Agent.__init__", return_value=None):
            agent = CodeAgent(
                system_prompt="You are a data analyst.",
                toolkits=[DATA_ANALYSIS_TOOLKIT, VISUALIZATION_TOOLKIT],
                python_interpreter_class=SandboxedPythonInterpreter,
            )
        assert "pandas" in agent.python_repl.authorized_imports
        assert "matplotlib" in agent.python_repl.authorized_imports
        assert "seaborn" in agent.python_repl.authorized_imports
        # Both init codes ran — pd and plt should be available
        stdout, _ = agent.python_repl.execute_code("print(pd.DataFrame({'a': [1]}).shape)")
        assert "(1, 1)" in stdout
        stdout, _ = agent.python_repl.execute_code("print(plt.figure())")
        assert "Figure" in stdout
