"""Append-only audit trail for role-coherence control decisions.

Audit events are derived from validated application state. They do not decide
coherence status and cannot reinterpret role authority. The trail is immutable:
append operations return a new trail and reject duplicate event IDs.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import (
    AuditEventType,
    CoherenceAuditEvent,
    CoherenceSignal,
    CoherenceState,
    CoherenceStatus,
    InteractionTurn,
    RoleRepairDirective,
    TurnAssessment,
)


class AuditTrailError(ValueError):
    """Raised when audit linkage or append-only invariants are violated."""


class RepairEvaluationOutcome(StrEnum):
    """Application-derived outcome of a bounded repair attempt."""

    RECOVERED = "repair_recovered"
    FAILED = "repair_failed"
    ESCALATED = "repair_escalated"
    BLOCKED = "repair_blocked"


class AuditTrail(BaseModel):
    """Immutable append-only sequence of validated audit events."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    events: tuple[CoherenceAuditEvent, ...] = ()

    @model_validator(mode="after")
    def validate_unique_event_ids(self) -> "AuditTrail":
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("Audit event IDs must be unique")
        return self

    def append(self, event: CoherenceAuditEvent) -> "AuditTrail":
        return self.append_many((event,))

    def append_many(
        self,
        events: Iterable[CoherenceAuditEvent],
    ) -> "AuditTrail":
        new_events = tuple(events)
        existing_ids = {event.event_id for event in self.events}
        incoming_ids = [event.event_id for event in new_events]

        duplicate_existing = sorted(existing_ids & set(incoming_ids))
        if duplicate_existing:
            raise AuditTrailError(
                "Audit event IDs already exist in the trail: "
                f"{duplicate_existing}"
            )

        if len(incoming_ids) != len(set(incoming_ids)):
            raise AuditTrailError("Incoming audit event IDs must be unique")

        return AuditTrail(events=self.events + new_events)

    def for_turn(self, turn_id: str) -> tuple[CoherenceAuditEvent, ...]:
        return tuple(event for event in self.events if event.turn_id == turn_id)


def _event_id(
    turn: InteractionTurn,
    event_type: AuditEventType,
    suffix: str | None = None,
) -> str:
    base = f"audit:{turn.turn_id}:{event_type.value}"
    return f"{base}:{suffix}" if suffix else base


def _repair_attempted(
    previous_state: CoherenceState | None,
    state: CoherenceState,
) -> bool:
    if previous_state is None:
        return False
    return (
        previous_state.status is CoherenceStatus.REALIGN_REQUIRED
        and state.repair_attempts == previous_state.repair_attempts + 1
    )


def _repair_outcome(state: CoherenceState) -> RepairEvaluationOutcome:
    if state.status in {CoherenceStatus.COHERENT, CoherenceStatus.WATCH}:
        return RepairEvaluationOutcome.RECOVERED
    if state.status is CoherenceStatus.REALIGN_REQUIRED:
        return RepairEvaluationOutcome.FAILED
    if state.status is CoherenceStatus.HUMAN_REVIEW:
        return RepairEvaluationOutcome.ESCALATED
    if state.status is CoherenceStatus.BLOCKED:
        return RepairEvaluationOutcome.BLOCKED
    raise AuditTrailError(
        f"Unsupported repair outcome state: {state.status.value}"
    )


def _validate_turn_linkage(
    *,
    turn: InteractionTurn,
    assessment: TurnAssessment,
    signals: tuple[CoherenceSignal, ...],
    state: CoherenceState,
    previous_state: CoherenceState | None,
) -> None:
    if assessment.turn_id != turn.turn_id:
        raise AuditTrailError("Assessment must belong to the audited turn")

    wrong_turn_signals = [
        signal.signal_id for signal in signals if signal.turn_id != turn.turn_id
    ]
    if wrong_turn_signals:
        raise AuditTrailError(
            "All audited signals must belong to the current turn: "
            f"{wrong_turn_signals}"
        )

    if state.through_turn_id != turn.turn_id:
        raise AuditTrailError("Coherence state must terminate at the audited turn")

    if assessment.assessment_id not in state.assessment_ids:
        raise AuditTrailError(
            "Current assessment must be linked by the coherence state"
        )

    signal_ids = tuple(signal.signal_id for signal in signals)
    if signal_ids != state.recent_signal_ids:
        raise AuditTrailError(
            "Audited signals must exactly match the state's recent_signal_ids"
        )

    if previous_state is None:
        if state.previous_status is not None:
            raise AuditTrailError(
                "Initial audited state may not declare a previous status"
            )
    else:
        if state.previous_status is not previous_state.status:
            raise AuditTrailError(
                "State.previous_status must match the supplied previous state"
            )
        if turn.sequence_number <= previous_state.through_sequence_number:
            raise AuditTrailError(
                "Audited turns must advance monotonically"
            )


def _validate_issued_repair(
    *,
    turn: InteractionTurn,
    signals: tuple[CoherenceSignal, ...],
    state: CoherenceState,
    issued_repair: RoleRepairDirective,
) -> None:
    if state.status is not CoherenceStatus.REALIGN_REQUIRED:
        raise AuditTrailError(
            "A repair directive may only be audited from REALIGN_REQUIRED"
        )

    current_signal_ids = {signal.signal_id for signal in signals}
    missing_triggers = [
        signal_id
        for signal_id in issued_repair.triggering_signal_ids
        if signal_id not in current_signal_ids
    ]
    if missing_triggers:
        raise AuditTrailError(
            "Repair directive triggers must be present in the audited turn: "
            f"{missing_triggers}"
        )

    if not issued_repair.directive_id:
        raise AuditTrailError("Repair directive ID is required")


def build_turn_audit_events(
    *,
    turn: InteractionTurn,
    assessment: TurnAssessment,
    signals: Iterable[CoherenceSignal],
    state: CoherenceState,
    previous_state: CoherenceState | None = None,
    issued_repair: RoleRepairDirective | None = None,
    active_repair_directive_id: str | None = None,
) -> tuple[CoherenceAuditEvent, ...]:
    """Build the immutable audit-event bundle for one control-plane turn.

    Repair evaluation is inferred from the validated repair-attempt counter. When
    a repair attempt occurred, active_repair_directive_id is required so the
    outcome links back to the directive that was actually issued.
    """

    current_signals = tuple(signals)
    _validate_turn_linkage(
        turn=turn,
        assessment=assessment,
        signals=current_signals,
        state=state,
        previous_state=previous_state,
    )

    if issued_repair is not None:
        _validate_issued_repair(
            turn=turn,
            signals=current_signals,
            state=state,
            issued_repair=issued_repair,
        )

    repair_was_attempted = _repair_attempted(previous_state, state)
    if repair_was_attempted and not active_repair_directive_id:
        raise AuditTrailError(
            "Repair evaluation requires active_repair_directive_id"
        )
    if not repair_was_attempted and active_repair_directive_id is not None:
        raise AuditTrailError(
            "active_repair_directive_id is only valid when a repair was attempted"
        )

    events: list[CoherenceAuditEvent] = [
        CoherenceAuditEvent(
            event_id=_event_id(turn, AuditEventType.TURN_RECORDED),
            event_type=AuditEventType.TURN_RECORDED,
            turn_id=turn.turn_id,
            outcome=f"sequence={turn.sequence_number}",
        )
    ]

    if current_signals:
        events.append(
            CoherenceAuditEvent(
                event_id=_event_id(turn, AuditEventType.SIGNAL_RECORDED),
                event_type=AuditEventType.SIGNAL_RECORDED,
                turn_id=turn.turn_id,
                signal_ids=tuple(
                    signal.signal_id for signal in current_signals
                ),
                outcome=f"count={len(current_signals)}",
            )
        )

    events.append(
        CoherenceAuditEvent(
            event_id=_event_id(turn, AuditEventType.TURN_ASSESSED),
            event_type=AuditEventType.TURN_ASSESSED,
            turn_id=turn.turn_id,
            signal_ids=assessment.signal_ids,
            assessment_id=assessment.assessment_id,
            outcome=f"mode={assessment.mode.value}",
        )
    )

    events.append(
        CoherenceAuditEvent(
            event_id=_event_id(turn, AuditEventType.STATE_TRANSITION),
            event_type=AuditEventType.STATE_TRANSITION,
            turn_id=turn.turn_id,
            signal_ids=state.recent_signal_ids,
            assessment_id=assessment.assessment_id,
            from_status=state.previous_status,
            to_status=state.status,
            outcome="state_advanced",
        )
    )

    if issued_repair is not None:
        events.append(
            CoherenceAuditEvent(
                event_id=_event_id(
                    turn,
                    AuditEventType.REPAIR_ISSUED,
                    issued_repair.directive_id,
                ),
                event_type=AuditEventType.REPAIR_ISSUED,
                turn_id=turn.turn_id,
                signal_ids=issued_repair.triggering_signal_ids,
                assessment_id=assessment.assessment_id,
                from_status=state.previous_status,
                to_status=state.status,
                repair_directive_id=issued_repair.directive_id,
                outcome="directive_issued",
            )
        )

    if repair_was_attempted:
        repair_outcome = _repair_outcome(state)
        events.append(
            CoherenceAuditEvent(
                event_id=_event_id(
                    turn,
                    AuditEventType.REPAIR_EVALUATED,
                    active_repair_directive_id,
                ),
                event_type=AuditEventType.REPAIR_EVALUATED,
                turn_id=turn.turn_id,
                signal_ids=state.recent_signal_ids,
                assessment_id=assessment.assessment_id,
                from_status=state.previous_status,
                to_status=state.status,
                repair_directive_id=active_repair_directive_id,
                outcome=repair_outcome.value,
            )
        )

    if (
        state.status is CoherenceStatus.HUMAN_REVIEW
        and state.previous_status is not CoherenceStatus.HUMAN_REVIEW
    ):
        events.append(
            CoherenceAuditEvent(
                event_id=_event_id(
                    turn,
                    AuditEventType.HUMAN_REVIEW_REQUESTED,
                ),
                event_type=AuditEventType.HUMAN_REVIEW_REQUESTED,
                turn_id=turn.turn_id,
                signal_ids=state.recent_signal_ids,
                assessment_id=assessment.assessment_id,
                from_status=state.previous_status,
                to_status=state.status,
                repair_directive_id=active_repair_directive_id,
                outcome="human_review_required",
            )
        )

    if (
        state.status is CoherenceStatus.BLOCKED
        and state.previous_status is not CoherenceStatus.BLOCKED
    ):
        events.append(
            CoherenceAuditEvent(
                event_id=_event_id(
                    turn,
                    AuditEventType.AUTONOMY_BLOCKED,
                ),
                event_type=AuditEventType.AUTONOMY_BLOCKED,
                turn_id=turn.turn_id,
                signal_ids=state.recent_signal_ids,
                assessment_id=assessment.assessment_id,
                from_status=state.previous_status,
                to_status=state.status,
                outcome="autonomy_blocked",
            )
        )

    return tuple(events)
