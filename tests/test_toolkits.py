"""Tests for strands_code_agent.toolkits — Toolkit class and built-in toolkit instances."""

import pytest
from dataclasses import fields
from strands_code_agent.toolkits import (
    Toolkit,
    VISUALIZATION_TOOLKIT,
    DATA_ANALYSIS_TOOLKIT,
)
from strands_code_agent.imports import extract_imports


# ---------------------------------------------------------------------------
# Toolkit construction
# ---------------------------------------------------------------------------

class TestToolkitInit:
    def test_defaults_are_none(self):
        tk = Toolkit()
        assert tk.libraries is None
        assert tk.initialization_code is None
        assert tk.usage_instructions is None
        assert tk.domain_specific_code is None

    def test_all_fields_stored(self):
        libs = ["numpy", "pandas"]
        init = "import numpy as np"
        usage = "Use np for numpy."
        dsc = [int]  # any callable
        tk = Toolkit(
            libraries=libs,
            initialization_code=init,
            usage_instructions=usage,
            domain_specific_code=dsc,
        )
        assert tk.libraries is libs
        assert tk.initialization_code is init
        assert tk.usage_instructions is usage
        assert tk.domain_specific_code is dsc

    def test_libraries_only(self):
        tk = Toolkit(libraries=["os"])
        assert tk.libraries == ["os"]
        assert tk.initialization_code is None

    def test_empty_lists_are_not_none(self):
        tk = Toolkit(libraries=[], domain_specific_code=[])
        assert tk.libraries == []
        assert tk.domain_specific_code == []


# ---------------------------------------------------------------------------
# Toolkit is a dataclass
# ---------------------------------------------------------------------------

class TestToolkitDataclass:
    def test_is_dataclass(self):
        assert len(fields(Toolkit)) == 4

    def test_equality(self):
        a = Toolkit(libraries=["os"])
        b = Toolkit(libraries=["os"])
        assert a == b

    def test_inequality(self):
        a = Toolkit(libraries=["os"])
        b = Toolkit(libraries=["sys"])
        assert a != b

    def test_repr(self):
        tk = Toolkit(libraries=["os"])
        r = repr(tk)
        assert "Toolkit" in r
        assert "os" in r


# ---------------------------------------------------------------------------
# extract_imports
# ---------------------------------------------------------------------------

class TestExtractImports:
    def test_import_statement(self):
        assert extract_imports("import json") == {"json"}

    def test_from_import(self):
        assert extract_imports("from datetime import date") == {"datetime", "datetime.date"}

    def test_multiple_imports(self):
        code = "import os\nimport sys\nfrom json import dumps"
        assert extract_imports(code) == {"os", "sys", "json", "json.dumps"}

    def test_dotted_import(self):
        assert extract_imports("import matplotlib.pyplot") == {"matplotlib.pyplot"}

    def test_no_imports(self):
        assert extract_imports("x = 1") == set()

    def test_syntax_error_returns_empty(self):
        assert extract_imports("def (broken") == set()

    def test_empty_string(self):
        assert extract_imports("") == set()

    def test_from_import_includes_submodule_path(self):
        """'from scipy import stats' must produce both 'scipy' and 'scipy.stats'."""
        result = extract_imports("from scipy import stats")
        assert "scipy" in result
        assert "scipy.stats" in result

    def test_from_import_multiple_names(self):
        """'from datetime import date, timedelta' produces dotted paths for each name."""
        result = extract_imports("from datetime import date, timedelta")
        assert result == {"datetime", "datetime.date", "datetime.timedelta"}

    def test_from_dotted_module_import(self):
        """'from os.path import join' produces both 'os.path' and 'os.path.join'."""
        result = extract_imports("from os.path import join")
        assert "os.path" in result
        assert "os.path.join" in result


# ---------------------------------------------------------------------------
# Built-in toolkits
# ---------------------------------------------------------------------------

class TestBuiltinToolkits:
    def test_visualization_toolkit_libraries_use_wildcards(self):
        assert "matplotlib.*" in VISUALIZATION_TOOLKIT.libraries
        assert "seaborn.*" in VISUALIZATION_TOOLKIT.libraries

    def test_visualization_toolkit_init_code(self):
        assert "matplotlib.use('Agg')" in VISUALIZATION_TOOLKIT.initialization_code
        assert "import seaborn as sns" in VISUALIZATION_TOOLKIT.initialization_code

    def test_visualization_toolkit_usage_instructions(self):
        assert "Do not try to show" in VISUALIZATION_TOOLKIT.usage_instructions

    def test_data_analysis_toolkit_libraries_use_wildcards(self):
        assert "numpy.*" in DATA_ANALYSIS_TOOLKIT.libraries
        assert "pandas.*" in DATA_ANALYSIS_TOOLKIT.libraries
        assert "scipy.*" in DATA_ANALYSIS_TOOLKIT.libraries
        assert "datetime" in DATA_ANALYSIS_TOOLKIT.libraries

    def test_data_analysis_toolkit_init_code(self):
        assert "import numpy as np" in DATA_ANALYSIS_TOOLKIT.initialization_code
        assert "import pandas as pd" in DATA_ANALYSIS_TOOLKIT.initialization_code

    @pytest.mark.parametrize("tk", [VISUALIZATION_TOOLKIT, DATA_ANALYSIS_TOOLKIT])
    def test_builtin_toolkits_have_no_domain_specific_code(self, tk):
        assert tk.domain_specific_code is None
