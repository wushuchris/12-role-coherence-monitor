"""Deterministic longitudinal coherence-state transitions.

The state machine owns final coherence status. Semantic assessments and coherence
signals provide evidence, but they cannot directly assign control-plane states.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    CoherenceSignal,
    CoherenceState,
    CoherenceStatus,
    InteractionTurn,
    SignalSeverity,
    SignalSource,
    SignalType,
    TurnAssessment,
)


class StateMachineError(ValueError):
    """Raised when longitudinal inputs violate state-machine invariants."""


class CoherencePolicy(BaseModel):
    """Immutable application-owned thresholds for longitudinal coherence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    warning_score_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    drift_score_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    warning_turns_to_drift: int = Field(default=2, ge=2)
    warning_turns_to_realign: int = Field(default=3, ge=3)
    max_repair_attempts: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def validate_threshold_order(self) -> "CoherencePolicy":
        if self.drift_score_threshold >= self.warning_score_threshold:
            raise ValueError(
                "drift_score_threshold must be lower than warning_score_threshold"
            )
        if self.warning_turns_to_realign <= self.warning_turns_to_drift:
            raise ValueError(
                "warning_turns_to_realign must exceed warning_turns_to_drift"
            )
        return self


DEFAULT_COHERENCE_POLICY = CoherencePolicy()

_BLOCK_IMMEDIATELY = {
    SignalType.PROHIBITED_ACTION,
    SignalType.AUTHORITY_EXPANSION,
    SignalType.CONTRACT_MUTATION,
}

_HUMAN_REVIEW_IMMEDIATELY = {
    SignalType.MISSING_ESCALATION,
}

_WARNING_SEVERITIES = {
    SignalSeverity.MEDIUM,
    SignalSeverity.HIGH,
    SignalSeverity.CRITICAL,
}


def _assessment_scores(assessment: TurnAssessment) -> tuple[float, ...]:
    return (
        assessment.mission_alignment,
        assessment.scope_adherence,
        assessment.authority_adherence,
        assessment.evidence_discipline,
        assessment.behavioral_consistency,
    )


def _state_id(turn: InteractionTurn, assessment: TurnAssessment) -> str:
    return f"state:{turn.turn_id}:{assessment.assessment_id}"


def _validate_inputs(
    turn: InteractionTurn,
    assessment: TurnAssessment,
    signals: tuple[CoherenceSignal, ...],
    previous_state: CoherenceState | None,
    repair_attempted: bool,
) -> None:
    if assessment.turn_id != turn.turn_id:
        raise StateMachineError(
            "TurnAssessment.turn_id must match the InteractionTurn.turn_id"
        )

    mismatched_signal_ids = [
        signal.signal_id for signal in signals if signal.turn_id != turn.turn_id
    ]
    if mismatched_signal_ids:
        raise StateMachineError(
            "All coherence signals must belong to the current turn: "
            f"{mismatched_signal_ids}"
        )

    if previous_state is not None:
        if turn.sequence_number <= previous_state.through_sequence_number:
            raise StateMachineError(
                "Interaction turns must advance monotonically by sequence number"
            )
        if assessment.assessment_id in previous_state.assessment_ids:
            raise StateMachineError("Assessment IDs may not be replayed")

    if repair_attempted and (
        previous_state is None
        or previous_state.status is not CoherenceStatus.REALIGN_REQUIRED
    ):
        raise StateMachineError(
            "A repair attempt is only valid after REALIGN_REQUIRED"
        )


def _immediate_control_state(
    signals: tuple[CoherenceSignal, ...],
) -> CoherenceStatus | None:
    deterministic_critical_types = {
        signal.signal_type
        for signal in signals
        if signal.source is SignalSource.DETERMINISTIC
        and signal.severity is SignalSeverity.CRITICAL
    }

    if deterministic_critical_types & _BLOCK_IMMEDIATELY:
        return CoherenceStatus.BLOCKED

    if deterministic_critical_types & _HUMAN_REVIEW_IMMEDIATELY:
        return CoherenceStatus.HUMAN_REVIEW

    return None


def advance_coherence_state(
    *,
    turn: InteractionTurn,
    assessment: TurnAssessment,
    signals: Iterable[CoherenceSignal] = (),
    previous_state: CoherenceState | None = None,
    policy: CoherencePolicy = DEFAULT_COHERENCE_POLICY,
    repair_attempted: bool = False,
) -> CoherenceState:
    """Advance application-owned coherence state by one validated turn.

    The transition policy distinguishes clean behavior, isolated warnings,
    sustained warnings, severe drift, immediate hard violations, repair, and
    gradual recovery. LLM-generated assessments never set the final status.
    """

    current_signals = tuple(signals)
    _validate_inputs(
        turn=turn,
        assessment=assessment,
        signals=current_signals,
        previous_state=previous_state,
        repair_attempted=repair_attempted,
    )

    previous_status = previous_state.status if previous_state is not None else None
    previous_warning_turns = (
        previous_state.consecutive_warning_turns if previous_state is not None else 0
    )
    previous_repair_attempts = (
        previous_state.repair_attempts if previous_state is not None else 0
    )
    repair_attempts = previous_repair_attempts + int(repair_attempted)

    scores = _assessment_scores(assessment)
    severe_drift = any(score < policy.drift_score_threshold for score in scores)
    score_warning = any(score < policy.warning_score_threshold for score in scores)
    signal_warning = any(
        signal.severity in _WARNING_SEVERITIES for signal in current_signals
    )
    warning_present = score_warning or signal_warning
    warning_turns = previous_warning_turns + 1 if warning_present else 0

    immediate_state = _immediate_control_state(current_signals)

    if previous_status is CoherenceStatus.BLOCKED:
        status = CoherenceStatus.BLOCKED
    elif previous_status is CoherenceStatus.HUMAN_REVIEW:
        status = CoherenceStatus.HUMAN_REVIEW
    elif immediate_state is not None:
        status = immediate_state
    elif previous_status is CoherenceStatus.REALIGN_REQUIRED:
        if not repair_attempted:
            status = CoherenceStatus.REALIGN_REQUIRED
        elif not warning_present:
            status = CoherenceStatus.WATCH
        elif repair_attempts >= policy.max_repair_attempts:
            status = CoherenceStatus.HUMAN_REVIEW
        else:
            status = CoherenceStatus.REALIGN_REQUIRED
    elif previous_status is CoherenceStatus.DRIFTING:
        if not warning_present:
            status = CoherenceStatus.WATCH
        elif severe_drift or warning_turns >= policy.warning_turns_to_realign:
            status = CoherenceStatus.REALIGN_REQUIRED
        else:
            status = CoherenceStatus.DRIFTING
    elif previous_status is CoherenceStatus.WATCH:
        if severe_drift:
            status = CoherenceStatus.DRIFTING
        elif not warning_present:
            status = CoherenceStatus.COHERENT
        elif warning_turns >= policy.warning_turns_to_drift:
            status = CoherenceStatus.DRIFTING
        else:
            status = CoherenceStatus.WATCH
    else:
        if severe_drift:
            status = CoherenceStatus.DRIFTING
        elif warning_present:
            status = CoherenceStatus.WATCH
        else:
            status = CoherenceStatus.COHERENT

    prior_assessment_ids = (
        previous_state.assessment_ids if previous_state is not None else ()
    )

    return CoherenceState(
        state_id=_state_id(turn, assessment),
        status=status,
        through_turn_id=turn.turn_id,
        through_sequence_number=turn.sequence_number,
        assessment_ids=prior_assessment_ids + (assessment.assessment_id,),
        recent_signal_ids=tuple(signal.signal_id for signal in current_signals),
        previous_status=previous_status,
        consecutive_warning_turns=warning_turns,
        repair_attempts=repair_attempts,
    )
