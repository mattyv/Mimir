"""Phase 15 — graph context builder and run_eval graph_context_fn tests."""

from __future__ import annotations

from typing import Any

import pytest

from mimir.eval.harness import EvalQuestion, build_graph_context, run_eval
from tests.conftest import FakeLLM

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_tools_fn(
    search_results: list[dict[str, Any]] | None = None,
    neighborhood_edges: list[dict[str, Any]] | None = None,
    cascade_downstream: list[str] | None = None,
) -> Any:
    """Return a callable mimicking the MCP tools dispatcher."""

    def tools_fn(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "search":
            return {"results": search_results or [], "count": len(search_results or [])}
        if tool_name == "get_neighborhood":
            return {"edges": neighborhood_edges or [], "nodes": []}
        if tool_name == "entity_cascade_risk":
            return {"downstream_entities": cascade_downstream or [], "cascade_risk": 0.5}
        return {}

    return tools_fn


# ── build_graph_context ───────────────────────────────────────────────────────


@pytest.mark.phase15
def test_build_graph_context_empty_result() -> None:
    """tools_fn returns empty search results → build_graph_context returns empty string."""
    question = EvalQuestion(id="T1", category="factual_lookup", question="What is FooBar?")
    tools_fn = _make_tools_fn(search_results=[])
    ctx = build_graph_context(question, tools_fn)
    assert ctx == ""


@pytest.mark.phase15
def test_build_graph_context_factual_lookup() -> None:
    """tools_fn returns entities → context contains 'Knowledge Graph Context'."""
    question = EvalQuestion(id="T2", category="factual_lookup", question="Who owns the payment service?")
    entities = [
        {
            "id": "eid-1",
            "name": "PaymentService",
            "entity_type": "auros:TradingService",
            "description": "Handles all payments",
        }
    ]
    tools_fn = _make_tools_fn(search_results=entities)
    ctx = build_graph_context(question, tools_fn)
    assert "Knowledge Graph Context" in ctx
    assert "PaymentService" in ctx


@pytest.mark.phase15
def test_build_graph_context_relationship_traversal() -> None:
    """category=relationship_traversal → context includes edges section when edges exist."""
    question = EvalQuestion(
        id="T3",
        category="relationship_traversal",
        question="What does the risk engine depend on?",
    )
    entities = [
        {
            "id": "eid-risk",
            "name": "RiskEngine",
            "entity_type": "auros:TradingService",
            "description": "Core risk computation",
        }
    ]
    edges = [
        {"subject": "eid-risk", "predicate": "auros:dependsOn", "object": "eid-db"},
    ]
    tools_fn = _make_tools_fn(search_results=entities, neighborhood_edges=edges)
    ctx = build_graph_context(question, tools_fn)
    assert "Knowledge Graph Context" in ctx
    assert "Relationships around" in ctx
    assert "auros:dependsOn" in ctx


@pytest.mark.phase15
def test_build_graph_context_no_edges_relationship_traversal() -> None:
    """category=relationship_traversal with no edges → no Relationships section."""
    question = EvalQuestion(
        id="T4",
        category="relationship_traversal",
        question="What does this engine connect to?",
    )
    entities = [
        {
            "id": "eid-x",
            "name": "SomeEngine",
            "entity_type": "auros:TradingService",
            "description": "",
        }
    ]
    tools_fn = _make_tools_fn(search_results=entities, neighborhood_edges=[])
    ctx = build_graph_context(question, tools_fn)
    assert "Knowledge Graph Context" in ctx
    assert "Relationships around" not in ctx


@pytest.mark.phase15
def test_build_graph_context_cross_cutting() -> None:
    """category=cross_cutting with downstream → context includes Downstream section."""
    question = EvalQuestion(
        id="T5",
        category="cross_cutting",
        question="What systems depend on the settlement engine?",
    )
    entities = [
        {
            "id": "eid-settle",
            "name": "SettlementEngine",
            "entity_type": "auros:TradingService",
            "description": "Handles trade settlement",
        }
    ]
    tools_fn = _make_tools_fn(
        search_results=entities,
        cascade_downstream=["eid-clearing", "eid-reporting"],
    )
    ctx = build_graph_context(question, tools_fn)
    assert "Downstream from" in ctx
    assert "eid-clearing" in ctx


# ── run_eval with graph_context_fn ────────────────────────────────────────────


@pytest.mark.phase15
def test_run_eval_with_graph_context_fn() -> None:
    """run_eval with graph_context_fn → context is incorporated into prompt."""
    captured_prompts: list[str] = []

    class CapturingLLM:
        def complete(self, prompt: str, **_kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "response"

    questions = [
        EvalQuestion(id="GC1", category="factual_lookup", question="What is X?"),
    ]

    def graph_context_fn(q: EvalQuestion) -> str:
        return "## Knowledge Graph Context\n  - ContextEntity (auros:Service): desc"

    run_eval(questions, CapturingLLM(), graph_context_fn=graph_context_fn)

    assert len(captured_prompts) == 1
    assert "Knowledge Graph Context" in captured_prompts[0]
    assert "What is X?" in captured_prompts[0]


@pytest.mark.phase15
def test_run_eval_graph_context_fn_overrides_context_prefix() -> None:
    """graph_context_fn takes priority over context_prefix."""
    captured_prompts: list[str] = []

    class CapturingLLM:
        def complete(self, prompt: str, **_kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "response"

    questions = [
        EvalQuestion(id="GC2", category="factual_lookup", question="What is Y?"),
    ]

    def graph_context_fn(q: EvalQuestion) -> str:
        return "GraphCtx"

    run_eval(
        questions,
        CapturingLLM(),
        context_prefix="StaticPrefix",
        graph_context_fn=graph_context_fn,
    )

    assert len(captured_prompts) == 1
    # graph_context_fn result present, static prefix NOT present
    assert "GraphCtx" in captured_prompts[0]
    assert "StaticPrefix" not in captured_prompts[0]


@pytest.mark.phase15
def test_run_eval_graph_context_fn_empty_skips_prefix() -> None:
    """When graph_context_fn returns empty string, no extra prefix is added."""
    captured_prompts: list[str] = []

    class CapturingLLM:
        def complete(self, prompt: str, **_kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "response"

    questions = [
        EvalQuestion(id="GC3", category="factual_lookup", question="Simple question?"),
    ]

    run_eval(questions, CapturingLLM(), graph_context_fn=lambda q: "")

    assert captured_prompts[0] == "Simple question?"


@pytest.mark.phase15
def test_run_eval_no_graph_context_fn_uses_context_prefix() -> None:
    """Without graph_context_fn, context_prefix is used as before."""
    llm = FakeLLM()
    questions = [EvalQuestion(id="GC4", category="factual_lookup", question="Q?")]
    report = run_eval(questions, llm, context_prefix="SomePrefix")
    # The report should contain exactly one result (backward-compat)
    assert len(report.results) == 1
    assert report.total == 1
