"""Deterministic role-contract checks for explicit structured violations.

This module deliberately avoids semantic interpretation of natural-language prose.
It only evaluates facts the application can know from the immutable role contract,
structured attempted actions, and recognized provenance markers.
"""

from __future__ import annotations

from collections.abc import Iterable

from .schemas import (
    CoherenceSignal,
    InteractionTurn,
    RoleContract,
    SignalSeverity,
    SignalSource,
    SignalType,
)


_BOOLEAN_PROVENANCE_KEYS = (
    "attempted_contract_mutation",
    "self_assigned_authority",
    "escalation_required",
    "escalation_performed",
    "evidence_required",
    "evidence_present",
)


def _is_true(value: str | None) -> bool:
    return value is not None and value.strip().lower() == "true"


def _is_valid_boolean_marker(value: str) -> bool:
    return value.strip().lower() in {"true", "false"}


def _signal_id(turn: InteractionTurn, signal_type: SignalType, ordinal: int) -> str:
    return f"signal:{turn.turn_id}:{signal_type.value}:{ordinal}"


def _signal(
    turn: InteractionTurn,
    signal_type: SignalType,
    severity: SignalSeverity,
    evidence_reference: str,
    explanation: str,
    ordinal: int,
) -> CoherenceSignal:
    return CoherenceSignal(
        signal_id=_signal_id(turn, signal_type, ordinal),
        turn_id=turn.turn_id,
        signal_type=signal_type,
        severity=severity,
        source=SignalSource.DETERMINISTIC,
        evidence_reference=evidence_reference,
        explanation=explanation,
    )


def check_turn_against_contract(
    contract: RoleContract,
    turn: InteractionTurn,
) -> tuple[CoherenceSignal, ...]:
    """Return deterministic coherence signals for one interaction turn.

    The checker intentionally limits itself to structured facts. It does not infer
    mission drift, scope drift, or behavioral inconsistency from prose.
    """

    signals: list[CoherenceSignal] = []

    def add(
        signal_type: SignalType,
        severity: SignalSeverity,
        evidence_reference: str,
        explanation: str,
    ) -> None:
        signals.append(
            _signal(
                turn=turn,
                signal_type=signal_type,
                severity=severity,
                evidence_reference=evidence_reference,
                explanation=explanation,
                ordinal=len(signals) + 1,
            )
        )

    permitted_actions = set(contract.permitted_actions)
    prohibited_actions = set(contract.prohibited_actions)

    for index, action in enumerate(turn.attempted_actions):
        if action in prohibited_actions:
            add(
                SignalType.PROHIBITED_ACTION,
                SignalSeverity.CRITICAL,
                f"attempted_actions[{index}]",
                f"Action '{action}' is explicitly prohibited by the role contract.",
            )
        elif action not in permitted_actions:
            add(
                SignalType.UNAUTHORIZED_ACTION,
                SignalSeverity.HIGH,
                f"attempted_actions[{index}]",
                f"Action '{action}' is not authorized by the role contract.",
            )

    asserted_role_id = turn.provenance.get("asserted_role_id")
    if asserted_role_id is not None and asserted_role_id != contract.role_id:
        add(
            SignalType.UNKNOWN_IDENTIFIER,
            SignalSeverity.HIGH,
            "provenance.asserted_role_id",
            (
                f"Turn asserted role_id '{asserted_role_id}', but the authoritative "
                f"role_id is '{contract.role_id}'."
            ),
        )

    asserted_contract_version = turn.provenance.get("asserted_contract_version")
    if (
        asserted_contract_version is not None
        and asserted_contract_version != contract.version
    ):
        add(
            SignalType.CONTRACT_MUTATION,
            SignalSeverity.CRITICAL,
            "provenance.asserted_contract_version",
            (
                f"Turn asserted contract version '{asserted_contract_version}', but "
                f"the authoritative version is '{contract.version}'."
            ),
        )

    for key in _BOOLEAN_PROVENANCE_KEYS:
        value = turn.provenance.get(key)
        if value is not None and not _is_valid_boolean_marker(value):
            add(
                SignalType.MALFORMED_RECORD,
                SignalSeverity.HIGH,
                f"provenance.{key}",
                f"Marker '{key}' must be either 'true' or 'false'.",
            )

    if _is_true(turn.provenance.get("attempted_contract_mutation")):
        add(
            SignalType.CONTRACT_MUTATION,
            SignalSeverity.CRITICAL,
            "provenance.attempted_contract_mutation",
            "The turn explicitly records an attempt to mutate the authoritative role contract.",
        )

    if _is_true(turn.provenance.get("self_assigned_authority")):
        add(
            SignalType.AUTHORITY_EXPANSION,
            SignalSeverity.CRITICAL,
            "provenance.self_assigned_authority",
            "The turn explicitly records self-assigned authority beyond the role contract.",
        )

    if _is_true(turn.provenance.get("escalation_required")) and not _is_true(
        turn.provenance.get("escalation_performed")
    ):
        add(
            SignalType.MISSING_ESCALATION,
            SignalSeverity.CRITICAL,
            "provenance.escalation_required",
            "Required human escalation was not recorded as performed.",
        )

    if _is_true(turn.provenance.get("evidence_required")) and not _is_true(
        turn.provenance.get("evidence_present")
    ):
        add(
            SignalType.MISSING_EVIDENCE,
            SignalSeverity.HIGH,
            "provenance.evidence_required",
            "Required evidence was not recorded as present.",
        )

    return tuple(signals)


def has_hard_violation(signals: Iterable[CoherenceSignal]) -> bool:
    """Return True when deterministic signals include a high-consequence violation."""

    return any(
        signal.source is SignalSource.DETERMINISTIC
        and signal.severity is SignalSeverity.CRITICAL
        for signal in signals
    )
