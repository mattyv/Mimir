"""ACL evaluation — caller_groups vs node visibility.

An ACL on a node is a list of strings like:
    ["space:trading-eng", "repo:risk-infra/panic_server", "internal"]

A caller presents a set of groups they belong to.  Access is granted when
the intersection of caller_groups and node_acl is non-empty, OR when the
node sensitivity is "public".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SensitivityLevel = Literal["public", "internal", "restricted"]


@dataclass
class AccessDecision:
    allowed: bool
    reason: str
    matched_groups: list[str]


def check_access(
    node_acl: list[str],
    sensitivity: SensitivityLevel,
    caller_groups: set[str],
) -> AccessDecision:
    """Return an AccessDecision for a caller with *caller_groups*.

    Public nodes are always readable.
    Internal/restricted nodes require at least one matching group.
    """
    if sensitivity == "public":
        return AccessDecision(allowed=True, reason="public", matched_groups=[])

    matched = sorted(caller_groups & set(node_acl))
    if matched:
        return AccessDecision(allowed=True, reason="acl_match", matched_groups=matched)

    return AccessDecision(
        allowed=False,
        reason="no_matching_group",
        matched_groups=[],
    )


def filter_entities(
    entities: list[dict[str, Any]],
    caller_groups: set[str],
) -> list[dict[str, Any]]:
    """Return only the entities the caller can read.

    Each entity row must have a ``payload`` dict containing ``visibility``
    with keys ``acl`` (list[str]) and ``sensitivity`` (str).
    """
    visible: list[dict[str, Any]] = []
    for row in entities:
        vis = row.get("payload", {}).get("visibility", {})
        acl: list[str] = vis.get("acl", [])
        sensitivity: SensitivityLevel = vis.get("sensitivity", "internal")
        decision = check_access(acl, sensitivity, caller_groups)
        if decision.allowed:
            visible.append(row)
    return visible


def can_write(
    node_acl: list[str],
    sensitivity: SensitivityLevel,
    caller_groups: set[str],
    *,
    require_group: str | None = None,
) -> bool:
    """Return True if the caller can write to this node.

    Write access follows the same ACL rules as read, with an optional
    additional group requirement (e.g. an admin group).
    """
    decision = check_access(node_acl, sensitivity, caller_groups)
    if not decision.allowed:
        return False
    return not (require_group and require_group not in caller_groups)
