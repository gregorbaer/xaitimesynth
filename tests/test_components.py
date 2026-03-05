"""Tests for component definition functions.

Most component functions are trivial dict constructors - tested via builder integration.
Only manual() has validation logic worth testing directly.
"""

import pytest

from xaitimesynth.components import manual


def test_manual_requires_values_or_generator():
    """manual() raises if neither values nor generator provided."""
    with pytest.raises(ValueError, match="Either 'values' or 'generator'"):
        manual()
