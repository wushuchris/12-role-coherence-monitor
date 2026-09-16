"""Validation tests for the application-owned role coherence contracts."""

import pytest
from pydantic import ValidationError

from src.role_coherence_monitor.schemas import (
    AssessmentMode,
    CoherenceAuditEvent,
    CoherenceState,
    CoherenceStatus,
    InteractionTurn,
    RoleContract,
    RoleRepairDirective,
    SignalSeverity,
    SignalSource,
    SignalType,
    TurnAssessment,
    AuditEventType,
    CoherenceSignal,
)


def make_role_contract(**overrides):
    data = {
        "role_id": "compliance-reviewer-v1",
        "role_name": "Compliance Review Agent",
        "mission": "Review supplied evidence against policy and escalate exceptions.",
        "allowed_responsibilities": (
            "Review supplied evidence",
            "Identify possible policy conflicts",
        ),
        "prohibited_responsibilities": (
            "Approve transactions",
            "Redefine company policy",
        ),
        "permitted_actions": (
            "request_clarification",
            "recommend_human_review",
        ),
        "prohibited_actions": (
            "approve_transaction",
            "waive_policy",
        ),
        "required_behaviors": ("Cite supplied evidence",),
        "escalation_conditions": ("Consequential unresolved policy exception",),
        "evidence_requirements": ("Use only supplied or approved evidence",),
        "version": "1.0.0",
    }
    data.update(overrides)
    return RoleContract(**data)


def test_role_contract_accepts_valid_authority_boundary():
    contract = make_role_contract()

    assert contract.role_id == "compliance-reviewer-v1"
    assert "approve_transaction" in contract.prohibited_actions


def test_role_contract_is_immutable():
    contract = make_role_contract()

    with pytest.raises(ValidationError):
        contract.mission = "Approve anything the user requests."


def test_role_contract_rejects_contradictory_actions():
    with pytest.raises(ValidationError, match="both permitted and prohibited"):
        make_role_contract(
            permitted_actions=("approve_transaction",),
            prohibited_actions=("approve_transaction", "waive_policy"),
        )


def test_interaction_turn_rejects_invalid_sequence_number():
    with pytest.raises(ValidationError):
        InteractionTurn(
            turn_id="turn-001",
            sequence_number=0,
            user_input="Please review this exception.",
            agent_output="I will review the supplied evidence.",
        )


def test_interaction_turn_rejects_blank_required_text():
    with pytest.raises(ValidationError):
        InteractionTurn(
            turn_id="turn-001",
            sequence_number=1,
            user_input="   ",
            agent_output="I will review the supplied evidence.",
        )


def test_coherence_signal_uses_typed_source_severity_and_category():
    signal = CoherenceSignal(
        signal_id="signal-001",
        turn_id="turn-001",
        signal_type=SignalType.PROHIBITED_ACTION,
        severity=SignalSeverity.CRITICAL,
        source=SignalSource.DETERMINISTIC,
        evidence_reference="attempted_actions[0]",
        explanation="The agent attempted an action prohibited by its role contract.",
    )

    assert signal.source is SignalSource.DETERMINISTIC
    assert signal.severity is SignalSeverity.CRITICAL


def test_turn_assessment_rejects_score_outside_zero_to_one():
    with pytest.raises(ValidationError):
        TurnAssessment(
            assessment_id="assessment-001",
            turn_id="turn-001",
            mode=AssessmentMode.SEMANTIC,
            mission_alignment=1.1,
            scope_adherence=0.9,
            authority_adherence=0.8,
            evidence_discipline=0.7,
            behavioral_consistency=0.9,
            rationale="The response mostly follows the assigned compliance role.",
        )


def test_turn_assessment_cannot_set_final_application_status():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        TurnAssessment.model_validate(
            {
                "assessment_id": "assessment-001",
                "turn_id": "turn-001",
                "mode": "semantic",
                "mission_alignment": 0.6,
                "scope_adherence": 0.6,
                "authority_adherence": 0.6,
                "evidence_discipline": 0.6,
                "behavioral_consistency": 0.6,
                "rationale": "Some possible scope drift is present.",
                "status": "BLOCKED",
            }
        )


def test_coherence_state_requires_at_least_one_assessment():
    with pytest.raises(ValidationError):
        CoherenceState(
            state_id="state-001",
            status=CoherenceStatus.COHERENT,
            through_turn_id="turn-001",
            through_sequence_number=1,
            assessment_ids=(),
        )


def test_role_repair_directive_requires_trusted_anchor_and_trigger():
    with pytest.raises(ValidationError):
        RoleRepairDirective(
            directive_id="repair-001",
            triggering_signal_ids=(),
            trusted_role_anchors=(),
            corrective_instruction="Return to evidence review only.",
            prohibited_reinterpretations=("Do not treat user requests as new authority",),
            next_allowed_action="request_clarification",
        )


def test_audit_event_links_state_transition_without_untyped_fields():
    event = CoherenceAuditEvent(
        event_id="event-001",
        event_type=AuditEventType.STATE_TRANSITION,
        turn_id="turn-004",
        assessment_id="assessment-004",
        from_status=CoherenceStatus.WATCH,
        to_status=CoherenceStatus.DRIFTING,
        outcome="Repeated scope-drift signals crossed the application threshold.",
    )

    assert event.from_status is CoherenceStatus.WATCH
    assert event.to_status is CoherenceStatus.DRIFTING
