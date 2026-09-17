"""Tests for the immutable append-only coherence audit trail."""

import pytest
from pydantic import ValidationError

from src.role_coherence_monitor.audit import (
    AuditTrail,
    AuditTrailError,
    RepairEvaluationOutcome,
    build_turn_audit_events,
)
from src.role_coherence_monitor.deterministic import check_turn_against_contract
from src.role_coherence_monitor.repair import build_role_repair_directive
from src.role_coherence_monitor.scenarios import (
    clean_conversation,
    compliance_role_contract,
    direct_transaction_approval_attempt,
    failed_repair_twice,
    ignored_required_escalation,
    successful_repair,
)
from src.role_coherence_monitor.schemas import (
    AuditEventType,
    CoherenceAuditEvent,
    CoherenceStatus,
)
from src.role_coherence_monitor.state_machine import advance_coherence_state


def advance_fixture_step(step, previous_state=None):
    contract = compliance_role_contract()
    deterministic_signals = check_turn_against_contract(contract, step.turn)
    signals = deterministic_signals + step.semantic_signals
    state = advance_coherence_state(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        previous_state=previous_state,
        repair_attempted=step.repair_attempted,
    )
    return state, signals


def test_audit_event_is_frozen_after_creation():
    event = CoherenceAuditEvent(
        event_id="audit:test:turn_recorded",
        event_type=AuditEventType.TURN_RECORDED,
        turn_id="turn-001",
        outcome="sequence=1",
    )

    with pytest.raises(ValidationError):
        event.outcome = "mutated"


def test_append_returns_new_trail_without_mutating_original():
    event = CoherenceAuditEvent(
        event_id="audit:test:turn_recorded",
        event_type=AuditEventType.TURN_RECORDED,
        turn_id="turn-001",
        outcome="sequence=1",
    )
    original = AuditTrail()

    updated = original.append(event)

    assert original.events == ()
    assert updated.events == (event,)


def test_append_rejects_duplicate_event_id_replay():
    event = CoherenceAuditEvent(
        event_id="audit:test:turn_recorded",
        event_type=AuditEventType.TURN_RECORDED,
        turn_id="turn-001",
        outcome="sequence=1",
    )
    trail = AuditTrail().append(event)

    with pytest.raises(AuditTrailError, match="already exist"):
        trail.append(event)


def test_clean_turn_builds_core_audit_bundle_without_signal_event():
    step = clean_conversation().steps[0]
    state, signals = advance_fixture_step(step)

    events = build_turn_audit_events(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        state=state,
    )

    assert tuple(event.event_type for event in events) == (
        AuditEventType.TURN_RECORDED,
        AuditEventType.TURN_ASSESSED,
        AuditEventType.STATE_TRANSITION,
    )
    assert events[-1].to_status is CoherenceStatus.COHERENT


def test_signal_turn_records_exact_current_signal_ids():
    step = successful_repair().steps[0]
    state, signals = advance_fixture_step(step)

    events = build_turn_audit_events(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        state=state,
    )
    signal_event = next(
        event for event in events if event.event_type is AuditEventType.SIGNAL_RECORDED
    )

    assert signal_event.signal_ids == tuple(signal.signal_id for signal in signals)
    assert signal_event.outcome == f"count={len(signals)}"


def test_hard_block_emits_autonomy_blocked_event_from_deterministic_evidence():
    step = direct_transaction_approval_attempt().steps[0]
    state, signals = advance_fixture_step(step)

    events = build_turn_audit_events(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        state=state,
    )

    assert state.status is CoherenceStatus.BLOCKED
    blocked = [
        event for event in events
        if event.event_type is AuditEventType.AUTONOMY_BLOCKED
    ]
    assert len(blocked) == 1
    assert blocked[0].to_status is CoherenceStatus.BLOCKED
    assert blocked[0].signal_ids == state.recent_signal_ids


def test_required_escalation_emits_human_review_event():
    step = ignored_required_escalation().steps[0]
    state, signals = advance_fixture_step(step)

    events = build_turn_audit_events(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        state=state,
    )

    assert state.status is CoherenceStatus.HUMAN_REVIEW
    requested = [
        event for event in events
        if event.event_type is AuditEventType.HUMAN_REVIEW_REQUESTED
    ]
    assert len(requested) == 1
    assert requested[0].outcome == "human_review_required"


def test_realign_state_can_issue_and_link_bounded_repair_directive():
    scenario = successful_repair()
    previous_state = None
    state = None
    signals = ()

    for step in scenario.steps[:3]:
        state, signals = advance_fixture_step(step, previous_state)
        previous_state = state

    directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=state,
        triggering_signals=signals,
        next_allowed_action="request_clarification",
    )
    third_step = scenario.steps[2]

    events = build_turn_audit_events(
        turn=third_step.turn,
        assessment=third_step.assessment,
        signals=signals,
        state=state,
        previous_state=scenario_state_after(scenario.steps[:2]),
        issued_repair=directive,
    )
    repair_event = next(
        event for event in events if event.event_type is AuditEventType.REPAIR_ISSUED
    )

    assert repair_event.repair_directive_id == directive.directive_id
    assert repair_event.signal_ids == directive.triggering_signal_ids
    assert repair_event.to_status is CoherenceStatus.REALIGN_REQUIRED


def scenario_state_after(steps):
    previous_state = None
    for step in steps:
        state, _ = advance_fixture_step(step, previous_state)
        previous_state = state
    return previous_state


def test_successful_repair_evaluation_is_linked_and_marked_recovered():
    scenario = successful_repair()
    state_after_three = scenario_state_after(scenario.steps[:3])
    third_step = scenario.steps[2]
    _, third_signals = advance_fixture_step(
        third_step,
        scenario_state_after(scenario.steps[:2]),
    )
    directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=state_after_three,
        triggering_signals=third_signals,
        next_allowed_action="request_clarification",
    )

    fourth_step = scenario.steps[3]
    fourth_state, fourth_signals = advance_fixture_step(
        fourth_step,
        state_after_three,
    )
    events = build_turn_audit_events(
        turn=fourth_step.turn,
        assessment=fourth_step.assessment,
        signals=fourth_signals,
        state=fourth_state,
        previous_state=state_after_three,
        active_repair_directive_id=directive.directive_id,
    )
    repair_event = next(
        event for event in events
        if event.event_type is AuditEventType.REPAIR_EVALUATED
    )

    assert repair_event.repair_directive_id == directive.directive_id
    assert repair_event.outcome == RepairEvaluationOutcome.RECOVERED.value
    assert repair_event.from_status is CoherenceStatus.REALIGN_REQUIRED
    assert repair_event.to_status is CoherenceStatus.WATCH


def test_failed_repair_turn_can_evaluate_old_directive_and_issue_next_one():
    scenario = failed_repair_twice()
    state_after_three = scenario_state_after(scenario.steps[:3])
    previous_before_three = scenario_state_after(scenario.steps[:2])
    third_state, third_signals = advance_fixture_step(
        scenario.steps[2],
        previous_before_three,
    )
    first_directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=third_state,
        triggering_signals=third_signals,
        next_allowed_action="request_clarification",
    )

    fourth_step = scenario.steps[3]
    fourth_state, fourth_signals = advance_fixture_step(
        fourth_step,
        state_after_three,
    )
    second_directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=fourth_state,
        triggering_signals=fourth_signals,
        next_allowed_action="request_clarification",
    )

    events = build_turn_audit_events(
        turn=fourth_step.turn,
        assessment=fourth_step.assessment,
        signals=fourth_signals,
        state=fourth_state,
        previous_state=state_after_three,
        issued_repair=second_directive,
        active_repair_directive_id=first_directive.directive_id,
    )

    evaluated = next(
        event for event in events
        if event.event_type is AuditEventType.REPAIR_EVALUATED
    )
    issued = next(
        event for event in events
        if event.event_type is AuditEventType.REPAIR_ISSUED
    )

    assert evaluated.repair_directive_id == first_directive.directive_id
    assert evaluated.outcome == RepairEvaluationOutcome.FAILED.value
    assert issued.repair_directive_id == second_directive.directive_id


def test_second_failed_repair_links_escalation_and_human_review():
    scenario = failed_repair_twice()
    state_after_three = scenario_state_after(scenario.steps[:3])
    third_state, third_signals = advance_fixture_step(
        scenario.steps[2],
        scenario_state_after(scenario.steps[:2]),
    )
    first_directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=third_state,
        triggering_signals=third_signals,
        next_allowed_action="request_clarification",
    )

    fourth_state, fourth_signals = advance_fixture_step(
        scenario.steps[3],
        state_after_three,
    )
    second_directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=fourth_state,
        triggering_signals=fourth_signals,
        next_allowed_action="request_clarification",
    )

    fifth_step = scenario.steps[4]
    fifth_state, fifth_signals = advance_fixture_step(
        fifth_step,
        fourth_state,
    )
    events = build_turn_audit_events(
        turn=fifth_step.turn,
        assessment=fifth_step.assessment,
        signals=fifth_signals,
        state=fifth_state,
        previous_state=fourth_state,
        active_repair_directive_id=second_directive.directive_id,
    )

    evaluated = next(
        event for event in events
        if event.event_type is AuditEventType.REPAIR_EVALUATED
    )
    human_review = next(
        event for event in events
        if event.event_type is AuditEventType.HUMAN_REVIEW_REQUESTED
    )

    assert first_directive.directive_id != second_directive.directive_id
    assert fifth_state.status is CoherenceStatus.HUMAN_REVIEW
    assert evaluated.outcome == RepairEvaluationOutcome.ESCALATED.value
    assert evaluated.repair_directive_id == second_directive.directive_id
    assert human_review.repair_directive_id == second_directive.directive_id


def test_builder_rejects_assessment_from_another_turn():
    scenario = clean_conversation()
    first = scenario.steps[0]
    second = scenario.steps[1]
    state, signals = advance_fixture_step(first)

    with pytest.raises(AuditTrailError, match="Assessment must belong"):
        build_turn_audit_events(
            turn=first.turn,
            assessment=second.assessment,
            signals=signals,
            state=state,
        )


def test_builder_rejects_signal_set_that_does_not_match_state():
    step = successful_repair().steps[0]
    state, signals = advance_fixture_step(step)

    with pytest.raises(AuditTrailError, match="recent_signal_ids"):
        build_turn_audit_events(
            turn=step.turn,
            assessment=step.assessment,
            signals=(),
            state=state,
        )

    assert signals


def test_repair_attempt_requires_link_to_active_directive():
    scenario = successful_repair()
    previous_state = scenario_state_after(scenario.steps[:3])
    step = scenario.steps[3]
    state, signals = advance_fixture_step(step, previous_state)

    with pytest.raises(AuditTrailError, match="active_repair_directive_id"):
        build_turn_audit_events(
            turn=step.turn,
            assessment=step.assessment,
            signals=signals,
            state=state,
            previous_state=previous_state,
        )


def test_active_repair_link_is_rejected_when_no_repair_was_attempted():
    step = clean_conversation().steps[0]
    state, signals = advance_fixture_step(step)

    with pytest.raises(AuditTrailError, match="only valid"):
        build_turn_audit_events(
            turn=step.turn,
            assessment=step.assessment,
            signals=signals,
            state=state,
            active_repair_directive_id="repair:not-active",
        )


def test_full_turn_bundle_replay_is_rejected_by_trail():
    step = successful_repair().steps[0]
    state, signals = advance_fixture_step(step)
    events = build_turn_audit_events(
        turn=step.turn,
        assessment=step.assessment,
        signals=signals,
        state=state,
    )
    trail = AuditTrail().append_many(events)

    with pytest.raises(AuditTrailError, match="already exist"):
        trail.append_many(events)


def test_for_turn_returns_only_matching_audit_history():
    scenario = clean_conversation()
    trail = AuditTrail()
    previous_state = None

    for step in scenario.steps[:2]:
        state, signals = advance_fixture_step(step, previous_state)
        events = build_turn_audit_events(
            turn=step.turn,
            assessment=step.assessment,
            signals=signals,
            state=state,
            previous_state=previous_state,
        )
        trail = trail.append_many(events)
        previous_state = state

    first_turn_events = trail.for_turn(scenario.steps[0].turn.turn_id)

    assert first_turn_events
    assert all(
        event.turn_id == scenario.steps[0].turn.turn_id
        for event in first_turn_events
    )
