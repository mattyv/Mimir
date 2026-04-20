"""Code analysis adapter — extracts structure from Python source files via AST."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

from mimir.adapters.base import Chunk


def _count_complexity(tree: ast.Module) -> int:
    """Count branching nodes as a proxy for cyclomatic complexity."""
    branch_types = (
        ast.If,
        ast.For,
        ast.While,
        ast.ExceptHandler,
        ast.With,
    )
    return sum(1 for _ in ast.walk(tree) if isinstance(_, branch_types))


class CodeAnalysisAdapter:
    """Analyze Python source files and return structural summary Chunks."""

    def analyze(self, file_path: str | Path) -> Chunk | None:
        """Parse *file_path* with the Python AST and return a summary Chunk.

        Returns None if the file cannot be parsed (SyntaxError or read error).
        """
        path = Path(file_path)
        try:
            source = path.read_text()
            tree = ast.parse(source, filename=str(path))
        except (SyntaxError, OSError):
            return None

        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        functions = [
            n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
        ]
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imports.extend(
                    f"{module}.{alias.name}" if module else alias.name for alias in node.names
                )

        complexity = _count_complexity(tree)
        content = (
            f"Module: {path.name}\n"
            f"Language: Python\n"
            f"Classes: {', '.join(classes) or 'none'}\n"
            f"Functions: {', '.join(functions) or 'none'}\n"
            f"Imports: {', '.join(sorted(set(imports))) or 'none'}\n"
            f"Cyclomatic complexity: {complexity}"
        )
        return Chunk(
            id=f"code_{str(path).replace('/', '_').replace('.', '_')}",
            source_type="code_analysis",
            content=content,
            acl=[f"code:{path.parent}"],
            retrieved_at=datetime.now(UTC),
            reference=f"code://{path}",
            metadata={
                "file": str(path),
                "classes": classes,
                "functions": functions,
                "complexity": complexity,
            },
        )
