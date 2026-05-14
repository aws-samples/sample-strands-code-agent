"""Tests for strands_code_agent.document_code — format_function and get_documentation."""

import pytest
from strands_code_agent.document_code import format_function, get_documentation


# ---------------------------------------------------------------------------
# Helpers used as test subjects
# ---------------------------------------------------------------------------

def simple_func(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


def no_doc_func(a, b):
    return a - b


def multiline_doc_func():
    """First line.

    Second paragraph with details.
    """
    pass


class SampleClass:
    """A sample class for testing."""

    def __init__(self, value: int):
        """Initialise with a value."""
        self.value = value

    def public_method(self) -> int:
        """Return the value."""
        return self.value

    def _private_method(self):
        """Should be excluded."""
        pass

    def __len__(self) -> int:
        """Important dunder — should be included."""
        return 1

    def __secret(self):
        """Name-mangled — should be excluded."""
        pass


class BareClass:
    pass


# ---------------------------------------------------------------------------
# format_function
# ---------------------------------------------------------------------------

class TestFormatFunction:
    def test_includes_signature(self):
        out = format_function(simple_func)
        assert "def simple_func(x: int, y: int) -> int:" in out

    def test_includes_docstring(self):
        out = format_function(simple_func)
        assert "Add two numbers." in out

    def test_no_docstring(self):
        out = format_function(no_doc_func)
        assert "def no_doc_func(a, b):" in out
        assert '"""' not in out

    def test_multiline_docstring(self):
        out = format_function(multiline_doc_func)
        assert "First line." in out
        assert "Second paragraph" in out

    def test_indent(self):
        out = format_function(simple_func, indent="    ")
        for line in out.splitlines():
            assert line.startswith("    ") or line == ""

    def test_ends_with_ellipsis(self):
        out = format_function(simple_func)
        assert "    ...\n" in out

    def test_builtin_without_signature(self):
        # builtins like `len` may not expose a signature via inspect
        out = format_function(len)
        assert "def len" in out


# ---------------------------------------------------------------------------
# get_documentation — functions
# ---------------------------------------------------------------------------

class TestGetDocumentationFunction:
    def test_simple_function(self):
        doc = get_documentation(simple_func)
        assert "def simple_func" in doc
        assert "Add two numbers." in doc

    def test_no_doc_function(self):
        doc = get_documentation(no_doc_func)
        assert "def no_doc_func" in doc


# ---------------------------------------------------------------------------
# get_documentation — classes
# ---------------------------------------------------------------------------

class TestGetDocumentationClass:
    def test_class_header(self):
        doc = get_documentation(SampleClass)
        assert "class SampleClass" in doc

    def test_class_docstring(self):
        doc = get_documentation(SampleClass)
        assert "A sample class for testing." in doc

    def test_includes_init(self):
        doc = get_documentation(SampleClass)
        assert "def __init__" in doc

    def test_includes_public_method(self):
        doc = get_documentation(SampleClass)
        assert "def public_method" in doc

    def test_excludes_private_method(self):
        doc = get_documentation(SampleClass)
        assert "_private_method" not in doc

    def test_includes_important_dunder(self):
        doc = get_documentation(SampleClass)
        assert "def __len__" in doc

    def test_bare_class(self):
        doc = get_documentation(BareClass)
        assert "class BareClass" in doc

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="Expected a function or class"):
            get_documentation(42)

    def test_rejects_string(self):
        with pytest.raises(TypeError):
            get_documentation("hello")
