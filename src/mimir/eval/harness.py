"""Eval harness — load frozen questions and run them against an LLM.

The harness:
1. Loads eval questions from the YAML file.
2. Passes each question to the LLM (with optional context from the world model).
3. Records the response and a numeric score (1–5) from a judge function.
4. Returns an EvalReport summarising all runs.
"""

from __future__ import annotations

import hashlib
import random
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


@dataclass
class ComparisonPair:
    question: EvalQuestion
    response_a: str
    response_b: str
    score_a: float | None = None
    score_b: float | None = None


@dataclass
class ComparisonReport:
    label_a: str
    label_b: str
    pairs: list[ComparisonPair] = field(default_factory=list)
    total: int = 0
    scored: int = 0

    @property
    def mean_score_a(self) -> float:
        scores = [p.score_a for p in self.pairs if p.score_a is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def mean_score_b(self) -> float:
        scores = [p.score_b for p in self.pairs if p.score_b is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def by_category(self) -> dict[str, list[ComparisonPair]]:
        cats: dict[str, list[ComparisonPair]] = {}
        for p in self.pairs:
            cats.setdefault(p.question.category, []).append(p)
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


def build_graph_context(
    question: EvalQuestion,
    tools_fn: Any,  # callable(tool_name: str, args: dict) -> dict
) -> str:
    """Build graph context string for a question by calling MCP tools.

    Dispatch strategy based on category:
    - "factual_lookup" / "policy_recall": search for keywords, return top entities
    - "relationship_traversal": search + get_neighborhood of best match
    - "cross_cutting" / "decision_history": search + cascade_risk
    - default: search only

    Returns a compact string suitable for prepending to the question.
    """
    import re

    # Extract 2-3 keywords from question (simple heuristic: longest words)
    words = re.findall(r"\b[a-zA-Z_]{4,}\b", question.question)
    query = " ".join(sorted(set(words), key=len, reverse=True)[:3])
    if not query:
        return ""

    try:
        search_result = tools_fn("search", {"query": query, "limit": 5})
        entities = search_result.get("results", [])
    except Exception:
        return ""

    if not entities:
        return ""

    lines: list[str] = ["## Knowledge Graph Context"]

    top_id: str = entities[0].get("id", "")
    top_name: str = entities[0].get("name", "")

    # Base: list matching entities
    lines.append(f"\nRelevant entities for '{query}':")
    for e in entities[:5]:
        lines.append(
            f"  - {e.get('name')} ({e.get('entity_type', '?')}): "
            f"{(e.get('description') or '')[:100]}"
        )

    cat = question.category

    if cat == "relationship_traversal" and top_id:
        try:
            nb = tools_fn("get_neighborhood", {"entity_id": top_id, "depth": 2})
            edges = nb.get("edges", [])[:20]
            if edges:
                lines.append(f"\nRelationships around '{top_name}':")
                for edge in edges:
                    lines.append(
                        f"  {edge.get('subject')} --[{edge.get('predicate')}]--> {edge.get('object')}"
                    )
        except Exception:
            pass

    elif cat in ("cross_cutting", "decision_history") and top_id:
        try:
            risk = tools_fn("entity_cascade_risk", {"entity_id": top_id})
            downstream = risk.get("downstream_entities", [])[:10]
            if downstream:
                lines.append(
                    f"\nDownstream from '{top_name}': {', '.join(str(d) for d in downstream)}"
                )
        except Exception:
            pass

    return "\n".join(lines)


def run_eval(
    questions: list[EvalQuestion],
    llm: Any,
    *,
    judge: Callable[[EvalQuestion, str], float] | None = None,
    context_prefix: str = "",
    graph_context_fn: Callable[[EvalQuestion], str] | None = None,
) -> EvalReport:
    """Run all questions through *llm* and return an EvalReport.

    Args:
        questions:        List of EvalQuestion objects.
        llm:              Object with a `.complete(prompt) -> str` method.
        judge:            Optional scorer that returns a float 1–5.
        context_prefix:   Prepended to each question (e.g. world model context).
        graph_context_fn: Optional function(question) → context string; takes
                          priority over context_prefix when provided.
    """
    report = EvalReport(total=len(questions))
    for q in questions:
        ctx = graph_context_fn(q) if graph_context_fn else context_prefix
        prompt = f"{ctx}\n\n{q.question}".strip() if ctx else q.question
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


def run_comparison(
    questions: list[EvalQuestion],
    llm_a: Any,
    llm_b: Any,
    *,
    label_a: str = "baseline",
    label_b: str = "with_mimir",
    context_fn_a: Callable[[EvalQuestion], str] | None = None,
    context_fn_b: Callable[[EvalQuestion], str] | None = None,
    judge: Callable[[EvalQuestion, str, str], tuple[float, float]] | None = None,
    seed: int | None = None,
) -> ComparisonReport:
    """Run questions against two LLM configs for A/B blind comparison.

    Args:
        questions:    List of EvalQuestion objects.
        llm_a:        LLM for config A (e.g., LLM without world model context).
        llm_b:        LLM for config B (e.g., LLM with world model context).
        label_a:      Human-readable name for config A.
        label_b:      Human-readable name for config B.
        context_fn_a: Optional function(question) → context string for A.
        context_fn_b: Optional function(question) → context string for B.
        judge:        Optional callable(question, resp_x, resp_y) → (score_x, score_y).
                      Responses are shuffled before judging so the judge is blind.
        seed:         Optional random seed for reproducible shuffling.
    """
    if seed is not None:
        random.seed(seed)

    report = ComparisonReport(label_a=label_a, label_b=label_b, total=len(questions))

    # Collect all pairs first
    raw_pairs: list[tuple[EvalQuestion, str, str]] = []
    for q in questions:
        ctx_a = context_fn_a(q) if context_fn_a else ""
        ctx_b = context_fn_b(q) if context_fn_b else ""
        prompt_a = f"{ctx_a}\n\n{q.question}".strip() if ctx_a else q.question
        prompt_b = f"{ctx_b}\n\n{q.question}".strip() if ctx_b else q.question
        resp_a = llm_a.complete(prompt_a)
        resp_b = llm_b.complete(prompt_b)
        raw_pairs.append((q, resp_a, resp_b))

    # Shuffle order for blind grading (judge doesn't know which is A or B)
    shuffled = list(raw_pairs)
    random.shuffle(shuffled)

    # Build lookup from question id → original pair order
    pair_map: dict[str, tuple[str, str]] = {q.id: (a, b) for q, a, b in raw_pairs}

    for q, resp_x, resp_y in shuffled:
        pair = ComparisonPair(
            question=q, response_a=pair_map[q.id][0], response_b=pair_map[q.id][1]
        )

        if judge is not None:
            score_x, score_y = judge(q, resp_x, resp_y)
            # Determine which shuffled slot corresponds to A vs B
            orig_a, orig_b = pair_map[q.id]
            if resp_x == orig_a:
                pair.score_a, pair.score_b = score_x, score_y
            else:
                pair.score_a, pair.score_b = score_y, score_x
            report.scored += 1

        report.pairs.append(pair)

    return report
