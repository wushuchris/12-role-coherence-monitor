"""Tests for deterministic longitudinal coherence-state transitions."""

import pytest
from pydantic import ValidationError

from src.role_coherence_monitor.schemas import (
    AssessmentMode,
    CoherenceSignal,
    CoherenceStatus,
    InteractionTurn,
    SignalSeverity,
    SignalSource,
    SignalType,
    TurnAssessment,
)
from src.role_coherence_monitor.state_machine import (
    CoherencePolicy,
    StateMachineError,
    advance_coherence_state,
)


def make_turn(sequence_number: int, *, turn_id: str | None = None) -> InteractionTurn:
    resolved_turn_id = turn_id or f"turn-{sequence_number:03d}"
    return InteractionTurn(
        turn_id=resolved_turn_id,
        sequence_number=sequence_number,
        user_input="Please review the supplied compliance evidence.",
        agent_output="I will review the evidence against the policy boundary.",
    )


def make_assessment(
    turn: InteractionTurn,
    *,
    assessment_id: str | None = None,
    score: float = 0.95,
) -> TurnAssessment:
    return TurnAssessment(
        assessment_id=assessment_id or f"assessment-{turn.sequence_number:03d}",
        turn_id=turn.turn_id,
        mode=AssessmentMode.HYBRID,
        mission_alignment=score,
        scope_adherence=score,
        authority_adherence=score,
        evidence_discipline=score,
        behavioral_consistency=score,
        rationale="Synthetic state-machine test assessment.",
    )


def make_signal(
    turn: InteractionTurn,
    *,
    signal_type: SignalType,
    severity: SignalSeverity,
    source: SignalSource = SignalSource.DETERMINISTIC,
) -> CoherenceSignal:
    return CoherenceSignal(
        signal_id=f"signal-{turn.sequence_number:03d}-{signal_type.value}-{source.value}",
        turn_id=turn.turn_id,
        signal_type=signal_type,
        severity=severity,
        source=source,
        evidence_reference="synthetic.test",
        explanation="Synthetic state-machine test signal.",
    )


def advance(
    sequence_number: int,
    *,
    previous_state=None,
    score: float = 0.95,
    signals=(),
    repair_attempted: bool = False,
):
    turn = make_turn(sequence_number)
    assessment = make_assessment(turn, score=score)
    return advance_coherence_state(
        turn=turn,
        assessment=assessment,
        signals=signals,
        previous_state=previous_state,
        repair_attempted=repair_attempted,
    )


def test_clean_initial_turn_is_coherent():
    state = advance(1)

    assert state.status is CoherenceStatus.COHERENT
    assert state.consecutive_warning_turns == 0
    assert state.previous_status is None


def test_single_soft_warning_moves_to_watch():
    state = advance(1, score=0.75)

    assert state.status is CoherenceStatus.WATCH
    assert state.consecutive_warning_turns == 1


def test_repeated_soft_warnings_progress_to_drifting_then_realign_required():
    first = advance(1, score=0.75)
    second = advance(2, previous_state=first, score=0.75)
    third = advance(3, previous_state=second, score=0.75)

    assert first.status is CoherenceStatus.WATCH
    assert second.status is CoherenceStatus.DRIFTING
    assert third.status is CoherenceStatus.REALIGN_REQUIRED
    assert third.consecutive_warning_turns == 3


def test_severe_dimension_failure_jumps_directly_to_drifting():
    state = advance(1, score=0.40)

    assert state.status is CoherenceStatus.DRIFTING


def test_deterministic_critical_prohibited_action_blocks_immediately():
    turn = make_turn(1)
    assessment = make_assessment(turn)
    signal = make_signal(
        turn,
        signal_type=SignalType.PROHIBITED_ACTION,
        severity=SignalSeverity.CRITICAL,
    )

    state = advance_coherence_state(
        turn=turn,
        assessment=assessment,
        signals=(signal,),
    )

    assert state.status is CoherenceStatus.BLOCKED


def test_deterministic_missing_escalation_requires_human_review_immediately():
    turn = make_turn(1)
    assessment = make_assessment(turn)
    signal = make_signal(
        turn,
        signal_type=SignalType.MISSING_ESCALATION,
        severity=SignalSeverity.CRITICAL,
    )

    state = advance_coherence_state(
        turn=turn,
        assessment=assessment,
        signals=(signal,),
    )

    assert state.status is CoherenceStatus.HUMAN_REVIEW


def test_semantic_critical_signal_cannot_directly_block_autonomy():
    turn = make_turn(1)
    assessment = make_assessment(turn)
    signal = make_signal(
        turn,
        signal_type=SignalType.PROHIBITED_ACTION,
        severity=SignalSeverity.CRITICAL,
        source=SignalSource.SEMANTIC,
    )

    state = advance_coherence_state(
        turn=turn,
        assessment=assessment,
        signals=(signal,),
    )

    assert state.status is CoherenceStatus.WATCH


def test_recovery_from_drifting_is_gradual_not_instant():
    drifting = advance(1, score=0.40)
    recovering = advance(2, previous_state=drifting, score=0.95)
    coherent = advance(3, previous_state=recovering, score=0.95)

    assert drifting.status is CoherenceStatus.DRIFTING
    assert recovering.status is CoherenceStatus.WATCH
    assert coherent.status is CoherenceStatus.COHERENT


def test_successful_repair_returns_to_watch_before_coherent():
    watch = advance(1, score=0.75)
    drifting = advance(2, previous_state=watch, score=0.75)
    realign = advance(3, previous_state=drifting, score=0.75)

    repaired = advance(
        4,
        previous_state=realign,
        score=0.95,
        repair_attempted=True,
    )
    coherent = advance(5, previous_state=repaired, score=0.95)

    assert realign.status is CoherenceStatus.REALIGN_REQUIRED
    assert repaired.status is CoherenceStatus.WATCH
    assert repaired.repair_attempts == 1
    assert coherent.status is CoherenceStatus.COHERENT


def test_failed_repairs_escalate_to_human_review_after_bounded_attempts():
    watch = advance(1, score=0.75)
    drifting = advance(2, previous_state=watch, score=0.75)
    realign = advance(3, previous_state=drifting, score=0.75)

    first_failed_repair = advance(
        4,
        previous_state=realign,
        score=0.75,
        repair_attempted=True,
    )
    second_failed_repair = advance(
        5,
        previous_state=first_failed_repair,
        score=0.75,
        repair_attempted=True,
    )

    assert first_failed_repair.status is CoherenceStatus.REALIGN_REQUIRED
    assert first_failed_repair.repair_attempts == 1
    assert second_failed_repair.status is CoherenceStatus.HUMAN_REVIEW
    assert second_failed_repair.repair_attempts == 2


def test_blocked_state_remains_blocked_on_later_clean_turn():
    turn = make_turn(1)
    blocked = advance_coherence_state(
        turn=turn,
        assessment=make_assessment(turn),
        signals=(
            make_signal(
                turn,
                signal_type=SignalType.CONTRACT_MUTATION,
                severity=SignalSeverity.CRITICAL,
            ),
        ),
    )

    later = advance(2, previous_state=blocked, score=0.95)

    assert blocked.status is CoherenceStatus.BLOCKED
    assert later.status is CoherenceStatus.BLOCKED


def test_repair_attempt_is_rejected_outside_realign_required():
    coherent = advance(1)

    with pytest.raises(StateMachineError, match="only valid after REALIGN_REQUIRED"):
        advance(2, previous_state=coherent, repair_attempted=True)


def test_non_monotonic_turn_sequence_is_rejected():
    first = advance(2)
    turn = make_turn(2, turn_id="turn-replayed")

    with pytest.raises(StateMachineError, match="advance monotonically"):
        advance_coherence_state(
            turn=turn,
            assessment=make_assessment(turn, assessment_id="assessment-replayed"),
            previous_state=first,
        )


def test_mismatched_assessment_turn_is_rejected():
    turn = make_turn(1)
    other_turn = make_turn(2)

    with pytest.raises(StateMachineError, match="TurnAssessment.turn_id"):
        advance_coherence_state(
            turn=turn,
            assessment=make_assessment(other_turn),
        )


def test_policy_rejects_inverted_score_thresholds():
    with pytest.raises(ValidationError, match="must be lower"):
        CoherencePolicy(
            warning_score_threshold=0.70,
            drift_score_threshold=0.80,
        )
