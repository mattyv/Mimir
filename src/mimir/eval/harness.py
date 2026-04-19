"""Eval harness — load frozen questions and run them against an LLM.

The harness:
1. Loads eval questions from the YAML file.
2. Passes each question to the LLM (with optional context from the world model).
3. Records the response and a numeric score (1–5) from a judge function.
4. Returns an EvalReport summarising all runs.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    notes: str = ""


@dataclass
class EvalResult:
    question_id: str
    category: str
    question: str
    response: str
    score: float | None = None  # 1–5 or None if not yet judged


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)
    total: int = 0
    scored: int = 0

    @property
    def mean_score(self) -> float:
        scores = [r.score for r in self.results if r.score is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def by_category(self) -> dict[str, list[EvalResult]]:
        cats: dict[str, list[EvalResult]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r)
        return cats


def load_questions(yaml_path: Path) -> list[EvalQuestion]:
    """Load eval questions from the frozen YAML file."""
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    questions = []
    for q in data.get("questions", []):
        questions.append(
            EvalQuestion(
                id=q["id"],
                category=q["category"],
                question=str(q["question"]).strip(),
                notes=str(q.get("notes", "")).strip(),
            )
        )
    return questions


def checksum_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_eval(
    questions: list[EvalQuestion],
    llm: Any,
    *,
    judge: Callable[[EvalQuestion, str], float] | None = None,
    context_prefix: str = "",
) -> EvalReport:
    """Run all questions through *llm* and return an EvalReport.

    Args:
        questions:      List of EvalQuestion objects.
        llm:            Object with a `.complete(prompt) -> str` method.
        judge:          Optional scorer that returns a float 1–5.
        context_prefix: Prepended to each question (e.g. world model context).
    """
    report = EvalReport(total=len(questions))
    for q in questions:
        prompt = f"{context_prefix}\n\n{q.question}".strip() if context_prefix else q.question
        response = llm.complete(prompt)
        score: float | None = judge(q, response) if judge else None
        result = EvalResult(
            question_id=q.id,
            category=q.category,
            question=q.question,
            response=response,
            score=score,
        )
        report.results.append(result)
        if score is not None:
            report.scored += 1
    return report
