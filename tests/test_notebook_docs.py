"""Tests that documentation notebooks execute without errors.

Discovers all .ipynb files under docs/, converts each to a Python script
using nbconvert, and runs the script in a subprocess. This validates that
notebook code is correct without modifying saved notebook outputs.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from nbconvert import PythonExporter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"


def discover_notebooks():
    """Find all .ipynb files under docs/."""
    return sorted(DOCS_DIR.rglob("*.ipynb"))


@pytest.mark.parametrize(
    "notebook_path",
    discover_notebooks(),
    ids=[str(p.relative_to(PROJECT_ROOT)) for p in discover_notebooks()],
)
def test_notebook_executes(notebook_path, tmp_path):
    """Test that a documentation notebook runs without errors."""

    script, _ = PythonExporter().from_filename(str(notebook_path))

    # Strip get_ipython() calls (IPython magics don't work as plain Python)
    script = "\n".join(
        line for line in script.splitlines() if "get_ipython()" not in line
    )

    script_path = tmp_path / "notebook_script.py"
    script_path.write_text(script)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"Notebook {notebook_path.name} failed:\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
