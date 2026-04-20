"""Phase 1 — Pydantic model unit and property tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from mimir.models.base import Grounding, GroundingTier, Source, Temporal, Visibility
from mimir.models.nodes import (
    Constraint,
    Decision,
    Entity,
    Observation,
    Process,
    Property,
    Relationship,
)

pytestmark = pytest.mark.phase1

_NOW = datetime(2026, 4, 19, tzinfo=UTC)
_YESTERDAY = _NOW - timedelta(days=1)
_TOMORROW = _NOW + timedelta(days=1)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _grounding(tier: GroundingTier = GroundingTier.source_cited) -> Grounding:
    return Grounding(tier=tier, depth=0, stop_reason="test")


def _source(src_type: str = "confluence") -> Source:  # type: ignore[return]
    valid = ("confluence", "github", "slack", "interview", "code_analysis")
    assert src_type in valid
    return Source(type=src_type, reference="ref://test", retrieved_at=_NOW)  # type: ignore[arg-type]


def _temporal(
    valid_from: datetime = _NOW,
    valid_until: datetime | None = None,
) -> Temporal:
    return Temporal(valid_from=valid_from, valid_until=valid_until)


def _visibility() -> Visibility:
    return Visibility(acl=["space:test"], sensitivity="internal")


def _entity(**kwargs: object) -> Entity:
    defaults: dict[str, object] = dict(
        id="test_entity",
        type="schema:Organization",
        name="Test Entity",
        description="A test entity",
        created_at=_NOW,
        confidence=0.9,
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    defaults.update(kwargs)
    return Entity(**defaults)  # type: ignore[arg-type]


# ── GroundingTier ordering ─────────────────────────────────────────────────────


def test_grounding_tier_ordering() -> None:
    assert GroundingTier.ungrounded < GroundingTier.source_cited
    assert GroundingTier.source_cited < GroundingTier.schema_typed
    assert GroundingTier.schema_typed < GroundingTier.wikidata_linked
    assert GroundingTier.wikidata_linked < GroundingTier.fully_grounded


def test_grounding_tier_all_five_values() -> None:
    tiers = list(GroundingTier)
    assert len(tiers) == 5


def test_grounding_tier_le_operator() -> None:
    assert GroundingTier.ungrounded <= GroundingTier.source_cited
    assert GroundingTier.schema_typed <= GroundingTier.schema_typed


def test_grounding_tier_gt_operator() -> None:
    assert GroundingTier.fully_grounded > GroundingTier.wikidata_linked


def test_grounding_tier_ge_operator() -> None:
    assert GroundingTier.wikidata_linked >= GroundingTier.schema_typed
    assert GroundingTier.schema_typed >= GroundingTier.schema_typed


def test_grounding_tier_comparison_with_non_tier_returns_not_implemented() -> None:
    tier = GroundingTier.schema_typed
    assert tier.__lt__("schema_typed") is NotImplemented
    assert tier.__le__("schema_typed") is NotImplemented
    assert tier.__gt__("schema_typed") is NotImplemented
    assert tier.__ge__("schema_typed") is NotImplemented


# ── Grounding model ────────────────────────────────────────────────────────────


def test_grounding_valid() -> None:
    g = Grounding(tier=GroundingTier.schema_typed, depth=1, stop_reason="leaf")
    assert g.tier == GroundingTier.schema_typed


def test_grounding_wikidata_id_optional() -> None:
    g = Grounding(
        tier=GroundingTier.wikidata_linked, depth=2, stop_reason="found", wikidata_id="Q42"
    )
    assert g.wikidata_id == "Q42"


# ── Source model ───────────────────────────────────────────────────────────────


def test_source_valid_types_accepted() -> None:
    for src_type in ("confluence", "github", "slack", "interview", "code_analysis"):
        s = Source(type=src_type, reference="ref://x", retrieved_at=_NOW)  # type: ignore[arg-type]
        assert s.type == src_type


def test_source_invalid_type_rejected() -> None:
    with pytest.raises(ValidationError):
        Source(type="jira", reference="ref://x", retrieved_at=_NOW)  # type: ignore[arg-type]


# ── Temporal model ─────────────────────────────────────────────────────────────


def test_temporal_valid_from_required() -> None:
    with pytest.raises((ValidationError, TypeError)):
        Temporal()  # type: ignore[call-arg]


def test_temporal_valid_until_null_means_current() -> None:
    t = Temporal(valid_from=_NOW)
    assert t.valid_until is None


def test_temporal_ordering_valid() -> None:
    t = Temporal(valid_from=_YESTERDAY, valid_until=_NOW)
    assert t.valid_from < t.valid_until  # type: ignore[operator]


def test_temporal_ordering_enforced() -> None:
    with pytest.raises(ValidationError):
        Temporal(valid_from=_TOMORROW, valid_until=_YESTERDAY)


def test_temporal_equal_from_until_allowed() -> None:
    t = Temporal(valid_from=_NOW, valid_until=_NOW)
    assert t.valid_from == t.valid_until


def test_temporal_superseded_by_optional() -> None:
    t = Temporal(valid_from=_NOW, superseded_by="axiom_b")
    assert t.superseded_by == "axiom_b"


# ── Visibility model ───────────────────────────────────────────────────────────


def test_visibility_sensitivity_required() -> None:
    with pytest.raises((ValidationError, TypeError)):
        Visibility(acl=["space:test"])  # type: ignore[call-arg]


def test_visibility_valid_sensitivities() -> None:
    for sens in ("public", "internal", "restricted"):
        v = Visibility(acl=[], sensitivity=sens)  # type: ignore[arg-type]
        assert v.sensitivity == sens


def test_visibility_invalid_sensitivity_rejected() -> None:
    with pytest.raises(ValidationError):
        Visibility(acl=[], sensitivity="confidential")  # type: ignore[arg-type]


# ── Entity model ───────────────────────────────────────────────────────────────


def test_entity_valid_iri_accepted() -> None:
    e = _entity(type="schema:Organization")
    assert e.type == "schema:Organization"


def test_entity_auros_iri_accepted() -> None:
    e = _entity(type="auros:TradingService")
    assert e.type == "auros:TradingService"


def test_entity_provisional_iri_accepted() -> None:
    e = _entity(type="auros:provisional:custom_entity")
    assert e.type == "auros:provisional:custom_entity"


def test_entity_free_string_rejected() -> None:
    with pytest.raises(ValidationError):
        _entity(type="Organization")


def test_entity_unknown_namespace_rejected() -> None:
    with pytest.raises(ValidationError):
        _entity(type="owl:Class")


def test_entity_confidence_range() -> None:
    _entity(confidence=0.0)
    _entity(confidence=1.0)
    _entity(confidence=0.5)


def test_entity_confidence_above_one_rejected() -> None:
    with pytest.raises(ValidationError):
        _entity(confidence=1.01)


def test_entity_confidence_below_zero_rejected() -> None:
    with pytest.raises(ValidationError):
        _entity(confidence=-0.01)


def test_entity_vocabulary_version_present() -> None:
    e = _entity(vocabulary_version="0.1.0")
    assert e.vocabulary_version == "0.1.0"


# ── Property model ─────────────────────────────────────────────────────────────


def test_property_key_iri_validated() -> None:
    p = Property(
        entity_id="test_entity",
        key="schema:programmingLanguage",
        value="Python",
        value_type="str",
        confidence=0.9,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    assert p.key == "schema:programmingLanguage"


def test_property_non_iri_key_rejected() -> None:
    with pytest.raises(ValidationError):
        Property(
            entity_id="test_entity",
            key="programmingLanguage",  # not an IRI
            value="Python",
            value_type="str",
            confidence=0.9,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )


# ── Relationship model ─────────────────────────────────────────────────────────


def test_relationship_valid() -> None:
    r = Relationship(
        subject_id="service_a",
        predicate="auros:dependsOn",
        object_id="service_b",
        confidence=0.85,
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    assert r.predicate == "auros:dependsOn"


def test_relationship_predicate_validated() -> None:
    with pytest.raises(ValidationError):
        Relationship(
            subject_id="service_a",
            predicate="dependsOn",  # not an IRI
            object_id="service_b",
            confidence=0.85,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )


# ── Observation model ──────────────────────────────────────────────────────────


def test_observation_closed_vocabulary_valid() -> None:
    for obs_type in (
        "strength",
        "risk",
        "anti_pattern",
        "maturity",
        "smell",
        "opportunity",
        "inconsistency",
        "functional_state",
    ):
        o = Observation(
            entity_id="svc",
            type=obs_type,  # type: ignore[arg-type]
            description="...",
            confidence=0.7,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
        assert o.type == obs_type


def test_observation_closed_vocabulary_rejects_unknown() -> None:
    with pytest.raises(ValidationError):
        Observation(
            entity_id="svc",
            type="note",  # not in closed set
            description="...",
            confidence=0.7,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )


# ── Constraint model ───────────────────────────────────────────────────────────


def test_constraint_valid_types() -> None:
    for ct in ("performance", "availability", "legal", "physical", "social"):
        c = Constraint(
            entity_id="svc",
            constraint_type=ct,  # type: ignore[arg-type]
            condition="latency < 5ms",
            threshold=5,
            source=_source(),
            grounding=_grounding(),
            temporal=_temporal(),
            visibility=_visibility(),
            vocabulary_version="0.1.0",
        )
        assert c.constraint_type == ct


# ── Process model ──────────────────────────────────────────────────────────────


def test_process_valid() -> None:
    p = Process(
        id="order_lifecycle",
        name="Order Lifecycle",
        stages=["submit", "route", "fill", "settle"],
        inputs=["order_request"],
        outputs=["settlement_report"],
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    assert p.slo is None


def test_process_slo_optional() -> None:
    p = Process(
        id="trade_flow",
        name="Trade Flow",
        stages=["receive", "validate"],
        inputs=["trade"],
        outputs=["ack"],
        slo="p99 < 10ms",
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    assert p.slo == "p99 < 10ms"


# ── Decision model ─────────────────────────────────────────────────────────────


def test_decision_valid() -> None:
    d = Decision(
        id="venue_selection_001",
        what="Use CME for equity options clearing",
        why="Lower fees and better SLA than alternatives",
        tradeoffs=["single point of failure", "US-only coverage"],
        when=_NOW,
        who=["dmitri_v", "sarah_k"],
        source=_source(),
        grounding=_grounding(),
        temporal=_temporal(),
        visibility=_visibility(),
        vocabulary_version="0.1.0",
    )
    assert len(d.tradeoffs) == 2


# ── Property-based tests ───────────────────────────────────────────────────────


_VALID_IRI_ST = st.one_of(
    st.builds(
        lambda local: f"schema:{local}",
        st.from_regex(r"[A-Za-z][A-Za-z0-9]{0,20}", fullmatch=True),
    ),
    st.builds(
        lambda local: f"auros:{local}",
        st.from_regex(r"[A-Za-z][A-Za-z0-9]{0,20}", fullmatch=True),
    ),
)


@given(dt1=st.datetimes(timezones=st.just(UTC)), dt2=st.datetimes(timezones=st.just(UTC)))
def test_temporal_ordering_preserved(dt1: datetime, dt2: datetime) -> None:
    """valid_from <= valid_until is always preserved in well-formed Temporal objects."""
    valid_from = min(dt1, dt2)
    valid_until = max(dt1, dt2)
    t = Temporal(valid_from=valid_from, valid_until=valid_until)
    assert t.valid_from <= t.valid_until  # type: ignore[operator]


@given(dt1=st.datetimes(timezones=st.just(UTC)), dt2=st.datetimes(timezones=st.just(UTC)))
@settings(max_examples=100)
def test_temporal_reversed_ordering_raises(dt1: datetime, dt2: datetime) -> None:
    """Temporal raises when valid_from > valid_until."""
    if dt1 <= dt2:
        return  # only test the invalid case
    with pytest.raises(ValidationError):
        Temporal(valid_from=dt1, valid_until=dt2)


@given(iri=_VALID_IRI_ST)
def test_entity_accepts_any_valid_iri_type(iri: str) -> None:
    """Entity accepts any IRI in the core namespaces as its type field."""
    e = _entity(type=iri)
    assert e.type == iri


@given(
    e_id=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(whitelist_categories=("Ll", "Nd"), whitelist_characters="_"),
    ),
)
def test_entity_id_stored_as_given(e_id: str) -> None:
    """Entity id is stored as provided (no normalisation at model level)."""
    e = _entity(id=e_id)
    assert e.id == e_id
