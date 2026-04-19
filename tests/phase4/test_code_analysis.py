"""Phase 4 — CodeAnalysisAdapter tests."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from mimir.adapters.code_analysis import CodeAnalysisAdapter

_SIMPLE_MODULE = textwrap.dedent("""\
    import os
    import sys
    from pathlib import Path

    class RiskCalculator:
        def calculate(self, value: float) -> float:
            if value > 0:
                return value * 1.05
            return value

    def helper(x: int) -> int:
        for i in range(x):
            if i % 2 == 0:
                continue
        return x
""")


@pytest.fixture
def module_file(tmp_path: Path) -> Path:
    f = tmp_path / "risk_engine.py"
    f.write_text(_SIMPLE_MODULE)
    return f


@pytest.mark.phase4
def test_analyze_returns_chunk(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert chunk.source_type == "code_analysis"


@pytest.mark.phase4
def test_analyze_content_includes_module_name(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "risk_engine.py" in chunk.content


@pytest.mark.phase4
def test_analyze_detects_classes(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "RiskCalculator" in chunk.content


@pytest.mark.phase4
def test_analyze_detects_functions(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "helper" in chunk.content


@pytest.mark.phase4
def test_analyze_detects_imports(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "os" in chunk.content


@pytest.mark.phase4
def test_analyze_reports_complexity(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "Cyclomatic complexity:" in chunk.content
    complexity = int(chunk.metadata["complexity"])
    assert complexity >= 3  # if + for + if + continue = 4 branch nodes


@pytest.mark.phase4
def test_analyze_syntax_error_returns_none(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(\n  # unclosed")
    chunk = CodeAnalysisAdapter().analyze(bad)
    assert chunk is None


@pytest.mark.phase4
def test_analyze_missing_file_returns_none() -> None:
    chunk = CodeAnalysisAdapter().analyze("/nonexistent/module.py")
    assert chunk is None


@pytest.mark.phase4
def test_analyze_metadata_fields(module_file: Path) -> None:
    chunk = CodeAnalysisAdapter().analyze(module_file)
    assert chunk is not None
    assert "RiskCalculator" in chunk.metadata["classes"]
    assert "helper" in chunk.metadata["functions"]
