"""End-to-end application service for longitudinal role-coherence monitoring.

The service composes existing primitives without duplicating their authority:
deterministic checks establish structured violations, the semantic assessor
interprets language, the state machine owns control status, the repair policy
owns bounded realignment directives, and the audit layer records validated
outcomes. Session state is explicit and immutable.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from .audit import AuditTrail, build_turn_audit_events
from .deterministic import check_turn_against_contract
from .repair import RepairPolicyError, build_role_repair_directive
from .schemas import (
    AssessmentMode,
    CoherenceAuditEvent,
    CoherenceSignal,
    CoherenceState,
    CoherenceStatus,
    InteractionTurn,
    RoleContract,
    RoleRepairDirective,
    SignalSeverity,
    SignalType,
    TurnAssessment,
)
from .semantic import SemanticAssessor, to_turn_assessment
from .state_machine import CoherencePolicy, DEFAULT_COHERENCE_POLICY, advance_coherence_state


class MonitorConfigurationError(ValueError):
    """Raised when application-owned monitor configuration is invalid."""


class MonitorProcessingError(ValueError):
    """Raised when a turn cannot be processed without violating monitor policy."""


class MonitorSession(BaseModel):
    """Immutable externally visible state of one monitored conversation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    history: tuple[InteractionTurn, ...] = ()
    state: CoherenceState | None = None
    audit_trail: AuditTrail = Field(default_factory=AuditTrail)
    active_repair_directive: RoleRepairDirective | None = None


class MonitorResult(BaseModel):
    """Complete validated output from processing one interaction turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn: InteractionTurn
    assessment: TurnAssessment
    signals: tuple[CoherenceSignal, ...]
    state: CoherenceState
    issued_repair: RoleRepairDirective | None = None
    audit_events: tuple[CoherenceAuditEvent, ...] = Field(min_length=1)
    session: MonitorSession


class RoleCoherenceMonitor:
    """Application service that orchestrates the role-coherence control loop."""

    def __init__(
        self,
        *,
        contract: RoleContract,
        semantic_assessor: SemanticAssessor,
        repair_next_action: str,
        policy: CoherencePolicy = DEFAULT_COHERENCE_POLICY,
    ) -> None:
        if repair_next_action not in contract.permitted_actions:
            raise MonitorConfigurationError(
                f"Repair action '{repair_next_action}' is not permitted by the role contract"
            )

        self._contract = contract
        self._semantic_assessor = semantic_assessor
        self._repair_next_action = repair_next_action
        self._policy = policy

    @property
    def contract(self) -> RoleContract:
        return self._contract

    @property
    def policy(self) -> CoherencePolicy:
        return self._policy

    def new_session(self) -> MonitorSession:
        """Return an empty immutable session."""

        return MonitorSession()

    def process_turn(
        self,
        *,
        turn: InteractionTurn,
        session: MonitorSession | None = None,
        repair_attempted: bool = False,
    ) -> MonitorResult:
        """Process one turn transactionally through the complete control loop.

        The supplied session is never mutated. If any component raises, no updated
        MonitorSession is produced. A pending repair directive must be explicitly
        acknowledged on the next turn via repair_attempted=True.
        """

        current_session = session or self.new_session()
        self._validate_session_turn(
            session=current_session,
            turn=turn,
            repair_attempted=repair_attempted,
        )

        deterministic_signals = check_turn_against_contract(
            self._contract,
            turn,
        )

        semantic_result = self._semantic_assessor.assess(
            contract=self._contract,
            turn=turn,
            history=current_session.history,
        )
        self._validate_semantic_evidence(semantic_result)

        assessment = to_turn_assessment(
            semantic_result,
            assessment_id=f"assessment:{turn.turn_id}",
            mode=AssessmentMode.HYBRID,
        )

        signals = deterministic_signals + semantic_result.signals

        state = advance_coherence_state(
            turn=turn,
            assessment=assessment,
            signals=signals,
            previous_state=current_session.state,
            policy=self._policy,
            repair_attempted=repair_attempted,
        )

        issued_repair = self._maybe_issue_repair(
            state=state,
            signals=signals,
            repair_attempted=repair_attempted,
        )

        active_repair_directive_id = (
            current_session.active_repair_directive.directive_id
            if repair_attempted and current_session.active_repair_directive is not None
            else None
        )

        audit_events = build_turn_audit_events(
            turn=turn,
            assessment=assessment,
            signals=signals,
            state=state,
            previous_state=current_session.state,
            issued_repair=issued_repair,
            active_repair_directive_id=active_repair_directive_id,
        )

        audit_trail = current_session.audit_trail.append_many(audit_events)
        updated_session = MonitorSession(
            history=current_session.history + (turn,),
            state=state,
            audit_trail=audit_trail,
            active_repair_directive=issued_repair,
        )

        return MonitorResult(
            turn=turn,
            assessment=assessment,
            signals=signals,
            state=state,
            issued_repair=issued_repair,
            audit_events=audit_events,
            session=updated_session,
        )

    def process_turns(
        self,
        turns: Sequence[InteractionTurn],
        *,
        session: MonitorSession | None = None,
    ) -> tuple[MonitorResult, ...]:
        """Process a sequence that does not contain repair-attempt turns.

        This convenience helper is intentionally narrow. Interactive repair flows
        should call process_turn directly so each repair attempt is explicit.
        """

        current_session = session or self.new_session()
        results: list[MonitorResult] = []

        for turn in turns:
            result = self.process_turn(
                turn=turn,
                session=current_session,
            )
            results.append(result)
            current_session = result.session

        return tuple(results)

    def _validate_session_turn(
        self,
        *,
        session: MonitorSession,
        turn: InteractionTurn,
        repair_attempted: bool,
    ) -> None:
        if session.history:
            latest_turn = session.history[-1]
            if turn.sequence_number <= latest_turn.sequence_number:
                raise MonitorProcessingError(
                    "Interaction turns must advance monotonically"
                )
            if any(existing.turn_id == turn.turn_id for existing in session.history):
                raise MonitorProcessingError(
                    f"turn_id '{turn.turn_id}' has already been processed"
                )

        active_repair = session.active_repair_directive
        if active_repair is not None and not repair_attempted:
            raise MonitorProcessingError(
                "A pending role-repair directive requires repair_attempted=True"
            )
        if active_repair is None and repair_attempted:
            raise MonitorProcessingError(
                "repair_attempted=True requires an active role-repair directive"
            )

        if session.state is not None and session.history:
            if session.state.through_turn_id != session.history[-1].turn_id:
                raise MonitorProcessingError(
                    "Session state and interaction history are inconsistent"
                )

    def _validate_semantic_evidence(self, semantic_result) -> None:
        """Require auditable evidence whenever semantic scores affect control state."""

        scores = (
            semantic_result.mission_alignment,
            semantic_result.scope_adherence,
            semantic_result.authority_adherence,
            semantic_result.evidence_discipline,
            semantic_result.behavioral_consistency,
        )
        control_relevant_score = any(
            score < self._policy.warning_score_threshold for score in scores
        )
        if not control_relevant_score:
            return

        deviation_types = {
            SignalType.MISSION_DRIFT,
            SignalType.SCOPE_DRIFT,
            SignalType.AUTHORITY_EXPANSION,
            SignalType.BEHAVIORAL_DRIFT,
            SignalType.EVIDENCE_DEGRADATION,
        }
        has_auditable_deviation = any(
            signal.signal_type in deviation_types
            and signal.severity in {SignalSeverity.MEDIUM, SignalSeverity.HIGH}
            for signal in semantic_result.signals
        )
        if not has_auditable_deviation:
            raise MonitorProcessingError(
                "Semantic assessment lowered a control-relevant score without "
                "auditable MEDIUM/HIGH semantic deviation evidence"
            )

    def _maybe_issue_repair(
        self,
        *,
        state: CoherenceState,
        signals: tuple[CoherenceSignal, ...],
        repair_attempted: bool,
    ) -> RoleRepairDirective | None:
        if state.status is not CoherenceStatus.REALIGN_REQUIRED:
            return None

        # Entering REALIGN_REQUIRED or failing an explicit repair attempt both
        # require a fresh bounded directive. A no-op wait state is prevented by
        # _validate_session_turn when a directive is already active.
        if not signals:
            raise MonitorProcessingError(
                "REALIGN_REQUIRED cannot issue repair without auditable signals"
            )

        try:
            return build_role_repair_directive(
                contract=self._contract,
                state=state,
                triggering_signals=signals,
                next_allowed_action=self._repair_next_action,
            )
        except RepairPolicyError as exc:
            raise MonitorProcessingError(
                "Unable to issue bounded role-repair directive"
            ) from exc
