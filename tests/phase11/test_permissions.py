"""Phase 11 — permission and ACL tests."""

from __future__ import annotations

import pytest

from mimir.permissions.acl import AccessDecision, can_write, check_access, filter_entities

# ── check_access ──────────────────────────────────────────────────────────────


@pytest.mark.phase11
def test_public_node_always_allowed() -> None:
    d = check_access([], "public", set())
    assert d.allowed
    assert d.reason == "public"


@pytest.mark.phase11
def test_internal_node_allowed_with_matching_group() -> None:
    d = check_access(["space:trading-eng"], "internal", {"space:trading-eng", "space:risk"})
    assert d.allowed
    assert "space:trading-eng" in d.matched_groups


@pytest.mark.phase11
def test_internal_node_denied_without_matching_group() -> None:
    d = check_access(["space:trading-eng"], "internal", {"space:other"})
    assert not d.allowed
    assert d.reason == "no_matching_group"
    assert d.matched_groups == []


@pytest.mark.phase11
def test_restricted_node_denied_empty_caller() -> None:
    d = check_access(["team:admin"], "restricted", set())
    assert not d.allowed


@pytest.mark.phase11
def test_restricted_node_allowed_with_match() -> None:
    d = check_access(["team:admin"], "restricted", {"team:admin"})
    assert d.allowed


@pytest.mark.phase11
def test_multiple_matched_groups_all_returned() -> None:
    d = check_access(["g1", "g2", "g3"], "internal", {"g1", "g3", "g4"})
    assert d.allowed
    assert set(d.matched_groups) == {"g1", "g3"}


@pytest.mark.phase11
def test_access_decision_dataclass() -> None:
    d = AccessDecision(allowed=True, reason="test", matched_groups=["x"])
    assert d.allowed
    assert d.reason == "test"


# ── filter_entities ───────────────────────────────────────────────────────────


def _row(acl: list[str], sensitivity: str) -> dict:
    return {
        "id": "e1",
        "payload": {"visibility": {"acl": acl, "sensitivity": sensitivity}},
    }


@pytest.mark.phase11
def test_filter_entities_returns_visible() -> None:
    rows = [
        _row(["space:trading-eng"], "internal"),
        _row(["space:risk"], "internal"),
    ]
    visible = filter_entities(rows, {"space:trading-eng"})
    assert len(visible) == 1
    assert visible[0]["payload"]["visibility"]["acl"] == ["space:trading-eng"]


@pytest.mark.phase11
def test_filter_entities_public_always_visible() -> None:
    rows = [_row([], "public"), _row(["restricted-group"], "restricted")]
    visible = filter_entities(rows, set())
    assert len(visible) == 1


@pytest.mark.phase11
def test_filter_entities_empty_caller_sees_nothing_internal() -> None:
    rows = [_row(["space:trading-eng"], "internal")]
    assert filter_entities(rows, set()) == []


@pytest.mark.phase11
def test_filter_entities_empty_list() -> None:
    assert filter_entities([], {"any-group"}) == []


@pytest.mark.phase11
def test_filter_entities_all_visible() -> None:
    rows = [_row(["g1"], "internal"), _row(["g2"], "internal")]
    visible = filter_entities(rows, {"g1", "g2"})
    assert len(visible) == 2


# ── can_write ─────────────────────────────────────────────────────────────────


@pytest.mark.phase11
def test_can_write_allowed_no_extra_requirement() -> None:
    assert can_write(["space:trading-eng"], "internal", {"space:trading-eng"})


@pytest.mark.phase11
def test_can_write_denied_no_group_match() -> None:
    assert not can_write(["space:trading-eng"], "internal", {"space:other"})


@pytest.mark.phase11
def test_can_write_denied_missing_required_group() -> None:
    assert not can_write(
        ["space:trading-eng"],
        "internal",
        {"space:trading-eng"},
        require_group="team:admin",
    )


@pytest.mark.phase11
def test_can_write_allowed_with_required_group() -> None:
    assert can_write(
        ["space:trading-eng"],
        "internal",
        {"space:trading-eng", "team:admin"},
        require_group="team:admin",
    )


@pytest.mark.phase11
def test_can_write_public_node() -> None:
    assert can_write([], "public", set())
