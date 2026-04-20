"""Phase 15 — eval harness tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mimir.eval.harness import (
    ComparisonReport,
    EvalQuestion,
    EvalReport,
    EvalResult,
    checksum_file,
    load_questions,
    run_comparison,
    run_eval,
)
from tests.conftest import FakeLLM

_EVAL_PATH = Path(__file__).parent.parent.parent / "eval" / "frozen_questions.yaml"
_CHECKSUM_PATH = Path(__file__).parent.parent.parent / ".eval_checksum"


# ── load_questions ────────────────────────────────────────────────────────────


@pytest.mark.phase15
def test_load_questions_returns_20() -> None:
    questions = load_questions(_EVAL_PATH)
    assert len(questions) == 20


@pytest.mark.phase15
def test_load_questions_all_have_ids() -> None:
    questions = load_questions(_EVAL_PATH)
    for q in questions:
        assert q.id


@pytest.mark.phase15
def test_load_questions_all_have_categories() -> None:
    questions = load_questions(_EVAL_PATH)
    valid_cats = {
        "factual_lookup",
        "relationship_traversal",
        "policy_recall",
        "decision_history",
        "cross_cutting",
    }
    for q in questions:
        assert q.category in valid_cats


@pytest.mark.phase15
def test_load_questions_all_have_text() -> None:
    questions = load_questions(_EVAL_PATH)
    for q in questions:
        assert len(q.question) > 0


@pytest.mark.phase15
def test_load_questions_five_categories() -> None:
    questions = load_questions(_EVAL_PATH)
    cats = {q.category for q in questions}
    assert len(cats) == 5


@pytest.mark.phase15
def test_load_questions_four_per_category() -> None:
    questions = load_questions(_EVAL_PATH)
    from collections import Counter

    counts = Counter(q.category for q in questions)
    for cat, count in counts.items():
        assert count == 4, f"Expected 4 questions in {cat}, got {count}"


# ── checksum_file ─────────────────────────────────────────────────────────────


@pytest.mark.phase15
def test_checksum_file_consistent() -> None:
    h1 = checksum_file(_EVAL_PATH)
    h2 = checksum_file(_EVAL_PATH)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


@pytest.mark.phase15
def test_checksum_matches_stored() -> None:
    stored = _CHECKSUM_PATH.read_text().strip()
    computed = checksum_file(_EVAL_PATH)
    assert computed == stored


# ── run_eval ──────────────────────────────────────────────────────────────────


@pytest.mark.phase15
def test_run_eval_returns_report() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="What is X?")]
    llm = FakeLLM({"dummy": "answer"})
    report = run_eval(questions, llm)
    assert isinstance(report, EvalReport)
    assert report.total == 1
    assert len(report.results) == 1


@pytest.mark.phase15
def test_run_eval_all_questions_answered() -> None:
    questions = load_questions(_EVAL_PATH)
    llm = FakeLLM()
    report = run_eval(questions, llm)
    assert len(report.results) == 20


@pytest.mark.phase15
def test_run_eval_with_judge() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="What is X?")]
    llm = FakeLLM()

    def judge(q: EvalQuestion, response: str) -> float:
        return 3.0

    report = run_eval(questions, llm, judge=judge)
    assert report.results[0].score == 3.0
    assert report.scored == 1


@pytest.mark.phase15
def test_run_eval_mean_score() -> None:
    questions = [
        EvalQuestion(id="T1", category="factual_lookup", question="Q1?"),
        EvalQuestion(id="T2", category="factual_lookup", question="Q2?"),
    ]
    llm = FakeLLM()
    scores = [2.0, 4.0]
    call_count = [0]

    def judge(q: EvalQuestion, response: str) -> float:
        score = scores[call_count[0]]
        call_count[0] += 1
        return score

    report = run_eval(questions, llm, judge=judge)
    assert report.mean_score == 3.0


@pytest.mark.phase15
def test_run_eval_mean_score_no_judge() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="Q?")]
    llm = FakeLLM()
    report = run_eval(questions, llm)
    assert report.mean_score == 0.0


@pytest.mark.phase15
def test_run_eval_by_category() -> None:
    questions = load_questions(_EVAL_PATH)
    llm = FakeLLM()
    report = run_eval(questions, llm)
    cats = report.by_category()
    assert set(cats.keys()) == {
        "factual_lookup",
        "relationship_traversal",
        "policy_recall",
        "decision_history",
        "cross_cutting",
    }


@pytest.mark.phase15
def test_run_eval_context_prefix() -> None:
    question = EvalQuestion(id="T1", category="factual_lookup", question="What is X?")
    llm = FakeLLM()
    report = run_eval([question], llm, context_prefix="Context: some info.")
    assert len(report.results) == 1


# ── EvalResult / EvalQuestion dataclasses ────────────────────────────────────


@pytest.mark.phase15
def test_eval_result_fields() -> None:
    r = EvalResult(
        question_id="FL-01",
        category="factual_lookup",
        question="Who owns X?",
        response="Team A",
        score=4.0,
    )
    assert r.question_id == "FL-01"
    assert r.score == 4.0


@pytest.mark.phase15
def test_eval_question_fields() -> None:
    q = EvalQuestion(id="FL-01", category="factual_lookup", question="Who owns X?")
    assert q.id == "FL-01"
    assert q.notes == ""


# ── run_comparison (blind grading) ────────────────────────────────────────────


@pytest.mark.phase15
def test_run_comparison_returns_report() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="Q?")]
    llm_a = FakeLLM()
    llm_b = FakeLLM()
    report = run_comparison(questions, llm_a, llm_b)
    assert isinstance(report, ComparisonReport)
    assert report.total == 1
    assert len(report.pairs) == 1


@pytest.mark.phase15
def test_run_comparison_pair_has_both_responses() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="Q?")]
    llm_a = FakeLLM()
    llm_b = FakeLLM()
    report = run_comparison(questions, llm_a, llm_b)
    pair = report.pairs[0]
    assert pair.response_a is not None
    assert pair.response_b is not None


@pytest.mark.phase15
def test_run_comparison_with_judge() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="Q?")]
    llm_a = FakeLLM()
    llm_b = FakeLLM()

    def judge(q: EvalQuestion, x: str, y: str) -> tuple[float, float]:
        return 3.0, 4.0

    report = run_comparison(questions, llm_a, llm_b, judge=judge, seed=42)
    assert report.scored == 1
    pair = report.pairs[0]
    assert pair.score_a is not None
    assert pair.score_b is not None


@pytest.mark.phase15
def test_run_comparison_mean_scores() -> None:
    questions = [
        EvalQuestion(id="T1", category="factual_lookup", question="Q1?"),
        EvalQuestion(id="T2", category="factual_lookup", question="Q2?"),
    ]
    llm_a = FakeLLM()
    llm_b = FakeLLM()
    scores = [(2.0, 4.0), (4.0, 2.0)]
    call_count = [0]

    def judge(q: EvalQuestion, x: str, y: str) -> tuple[float, float]:
        pair = scores[call_count[0]]
        call_count[0] += 1
        return pair

    report = run_comparison(questions, llm_a, llm_b, judge=judge, seed=0)
    assert report.mean_score_a is not None
    assert report.mean_score_b is not None


@pytest.mark.phase15
def test_run_comparison_no_judge_scores_are_none() -> None:
    questions = [EvalQuestion(id="T1", category="factual_lookup", question="Q?")]
    report = run_comparison(questions, FakeLLM(), FakeLLM())
    pair = report.pairs[0]
    assert pair.score_a is None
    assert pair.score_b is None


@pytest.mark.phase15
def test_run_comparison_by_category() -> None:
    questions = load_questions(_EVAL_PATH)
    report = run_comparison(questions, FakeLLM(), FakeLLM())
    cats = report.by_category()
    assert "factual_lookup" in cats
    assert len(cats) == 5


@pytest.mark.phase15
def test_comparison_report_labels() -> None:
    report = run_comparison(
        [EvalQuestion(id="T1", category="factual_lookup", question="Q?")],
        FakeLLM(),
        FakeLLM(),
        label_a="llm_alone",
        label_b="llm_with_graph",
    )
    assert report.label_a == "llm_alone"
    assert report.label_b == "llm_with_graph"
