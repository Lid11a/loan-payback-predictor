# tests/test_utils.py
"""
Tests for small utility helpers.

The test suite focuses on validating stable behavior of simple helper functions
that are expected to remain unchanged across refactorings.
"""

from src.models.predict import safe_name


def test_safe_name_basic():
    """
    The test verifies basic whitespace replacement behavior.
    """

    assert safe_name("model v1") == "model_v1"


def test_safe_name_special_chars():
    """
    The test checks that non-alphanumeric characters are replaced with underscores.
    """

    assert safe_name("model@#$%name") == "model_name"


def test_safe_name_trim_underscores():
    """
    The test verifies that leading and trailing underscores are removed.
    """

    assert safe_name("__model__") == "model"


def test_safe_name_numeric():
    """
    The test checks that numeric characters are preserved in the output.
    """

    assert safe_name("model_123") == "model_123"
