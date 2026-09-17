"""Typed contracts for the Role Coherence Monitor.

These schemas define the application-owned boundaries for role identity,
interaction history, coherence evidence, longitudinal state, repair, and audit.
They intentionally do not implement drift detection logic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid")


class SignalSource(StrEnum):
    """Where a coherence signal originated."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"


class SignalSeverity(StrEnum):
    """Application-visible severity for a coherence signal."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SignalType(StrEnum):
    """Supported coherence-signal categories."""

    PROHIBITED_ACTION = "prohibited_action"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    AUTHORITY_EXPANSION = "authority_expansion"
    CONTRACT_MUTATION = "contract_mutation"
    MISSING_ESCALATION = "missing_escalation"
    MISSING_EVIDENCE = "missing_evidence"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    MALFORMED_RECORD = "malformed_record"
    INVALID_STATE_TRANSITION = "invalid_state_transition"
    MISSION_DRIFT = "mission_drift"
    SCOPE_DRIFT = "scope_drift"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    EVIDENCE_DEGRADATION = "evidence_degradation"
    RECOVERY = "recovery"


class AssessmentMode(StrEnum):
    """How a turn assessment was assembled."""

    DETERMINISTIC = "deterministic"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class CoherenceStatus(StrEnum):
    """Application-owned longitudinal coherence states."""

    COHERENT = "COHERENT"
    WATCH = "WATCH"
    DRIFTING = "DRIFTING"
    REALIGN_REQUIRED = "REALIGN_REQUIRED"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    BLOCKED = "BLOCKED"


class AuditEventType(StrEnum):
    """Append-only audit event categories."""

    TURN_RECORDED = "turn_recorded"
    SIGNAL_RECORDED = "signal_recorded"
    TURN_ASSESSED = "turn_assessed"
    STATE_TRANSITION = "state_transition"
    REPAIR_ISSUED = "repair_issued"
    REPAIR_EVALUATED = "repair_evaluated"
    HUMAN_REVIEW_REQUESTED = "human_review_requested"
    AUTONOMY_BLOCKED = "autonomy_blocked"


class RoleContract(StrictModel):
    """Immutable application-owned definition of an agent role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role_id: NonEmptyStr
    role_name: NonEmptyStr
    mission: NonEmptyStr
    allowed_responsibilities: tuple[NonEmptyStr, ...] = Field(min_length=1)
    prohibited_responsibilities: tuple[NonEmptyStr, ...] = Field(min_length=1)
    permitted_actions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    prohibited_actions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    required_behaviors: tuple[NonEmptyStr, ...] = Field(min_length=1)
    escalation_conditions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    evidence_requirements: tuple[NonEmptyStr, ...] = Field(min_length=1)
    version: NonEmptyStr

    @model_validator(mode="after")
    def reject_contradictory_contract(self) -> "RoleContract":
        responsibility_overlap = set(self.allowed_responsibilities) & set(
            self.prohibited_responsibilities
        )
        if responsibility_overlap:
            raise ValueError(
                "Responsibilities cannot be both allowed and prohibited: "
                f"{sorted(responsibility_overlap)}"
            )

        action_overlap = set(self.permitted_actions) & set(self.prohibited_actions)
        if action_overlap:
            raise ValueError(
                "Actions cannot be both permitted and prohibited: "
                f"{sorted(action_overlap)}"
            )

        return self


class InteractionTurn(StrictModel):
    """One ordered interaction observed by the monitor."""

    turn_id: NonEmptyStr
    sequence_number: int = Field(ge=1)
    user_input: NonEmptyStr
    agent_output: NonEmptyStr
    attempted_actions: tuple[NonEmptyStr, ...] = ()
    context_summary: NonEmptyStr | None = None
    provenance: dict[NonEmptyStr, NonEmptyStr] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CoherenceSignal(StrictModel):
    """Evidence that supports or challenges role coherence."""

    signal_id: NonEmptyStr
    turn_id: NonEmptyStr
    signal_type: SignalType
    severity: SignalSeverity
    source: SignalSource
    evidence_reference: NonEmptyStr
    explanation: NonEmptyStr


class TurnAssessment(StrictModel):
    """Validated per-turn coherence dimensions without control authority."""

    assessment_id: NonEmptyStr
    turn_id: NonEmptyStr
    mode: AssessmentMode
    mission_alignment: Score
    scope_adherence: Score
    authority_adherence: Score
    evidence_discipline: Score
    behavioral_consistency: Score
    signal_ids: tuple[NonEmptyStr, ...] = ()
    rationale: NonEmptyStr


class CoherenceState(StrictModel):
    """Application-owned longitudinal state through a specific turn."""

    state_id: NonEmptyStr
    status: CoherenceStatus
    through_turn_id: NonEmptyStr
    through_sequence_number: int = Field(ge=1)
    assessment_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    recent_signal_ids: tuple[NonEmptyStr, ...] = ()
    previous_status: CoherenceStatus | None = None
    consecutive_warning_turns: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RoleRepairDirective(StrictModel):
    """Bounded corrective instruction derived from trusted role anchors."""

    directive_id: NonEmptyStr
    triggering_signal_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    trusted_role_anchors: tuple[NonEmptyStr, ...] = Field(min_length=1)
    corrective_instruction: NonEmptyStr
    prohibited_reinterpretations: tuple[NonEmptyStr, ...] = Field(min_length=1)
    next_allowed_action: NonEmptyStr
    human_review_required: bool = False


class CoherenceAuditEvent(StrictModel):
    """Append-only linkage between turns, evidence, state, repair, and outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: NonEmptyStr
    event_type: AuditEventType
    turn_id: NonEmptyStr | None = None
    signal_ids: tuple[NonEmptyStr, ...] = ()
    assessment_id: NonEmptyStr | None = None
    from_status: CoherenceStatus | None = None
    to_status: CoherenceStatus | None = None
    repair_directive_id: NonEmptyStr | None = None
    outcome: NonEmptyStr | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
