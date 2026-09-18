"""Tests for the end-to-end RoleCoherenceMonitor application service."""

import pytest
from pydantic import ValidationError

from src.role_coherence_monitor.monitor import (
    MonitorConfigurationError,
    MonitorProcessingError,
    RoleCoherenceMonitor,
)
from src.role_coherence_monitor.scenarios import (
    clean_conversation,
    compliance_role_contract,
    direct_transaction_approval_attempt,
    failed_repair_twice,
    ignored_required_escalation,
    slow_scope_creep,
    successful_repair,
)
from src.role_coherence_monitor.schemas import (
    AuditEventType,
    CoherenceSignal,
    CoherenceStatus,
    SignalSeverity,
    SignalSource,
    SignalType,
)
from src.role_coherence_monitor.semantic import (
    MockSemanticAssessor,
    SemanticAssessmentError,
    SemanticAssessmentResult,
)


def assessor_for_scenario(scenario):
    results = {}
    for step in scenario.steps:
        results[step.turn.turn_id] = SemanticAssessmentResult(
            turn_id=step.turn.turn_id,
            mission_alignment=step.assessment.mission_alignment,
            scope_adherence=step.assessment.scope_adherence,
            authority_adherence=step.assessment.authority_adherence,
            evidence_discipline=step.assessment.evidence_discipline,
            behavioral_consistency=step.assessment.behavioral_consistency,
            signals=step.semantic_signals,
            rationale="Fixture-backed monitor test.",
        )
    return MockSemanticAssessor(results)


def healthy_result(turn_id: str) -> SemanticAssessmentResult:
    return SemanticAssessmentResult(
        turn_id=turn_id,
        mission_alignment=0.95,
        scope_adherence=0.95,
        authority_adherence=0.95,
        evidence_discipline=0.95,
        behavioral_consistency=0.95,
        signals=(),
        rationale="Synthetic healthy result.",
    )


def monitor_for_scenario(scenario):
    return RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=assessor_for_scenario(scenario),
        repair_next_action="request_clarification",
    )


def test_monitor_rejects_unpermitted_repair_action_at_configuration():
    scenario = clean_conversation()

    with pytest.raises(MonitorConfigurationError, match="not permitted"):
        RoleCoherenceMonitor(
            contract=compliance_role_contract(),
            semantic_assessor=assessor_for_scenario(scenario),
            repair_next_action="approve_transaction",
        )


def test_new_session_is_empty_and_immutable():
    monitor = monitor_for_scenario(clean_conversation())
    session = monitor.new_session()

    assert session.history == ()
    assert session.state is None
    assert session.audit_trail.events == ()
    assert session.active_repair_directive is None

    with pytest.raises(ValidationError):
        session.state = None


def test_clean_conversation_runs_end_to_end_and_remains_coherent():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()
    results = []

    for step in scenario.steps:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        results.append(result)
        session = result.session

    assert tuple(result.state.status for result in results) == (
        CoherenceStatus.COHERENT,
        CoherenceStatus.COHERENT,
        CoherenceStatus.COHERENT,
    )
    assert len(session.history) == 3
    assert session.active_repair_directive is None
    assert session.audit_trail.events


def test_processing_returns_new_session_without_mutating_prior_session():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)
    original = monitor.new_session()

    result = monitor.process_turn(
        turn=scenario.steps[0].turn,
        session=original,
    )

    assert original.history == ()
    assert original.state is None
    assert result.session is not original
    assert result.session.history == (scenario.steps[0].turn,)


def test_slow_scope_creep_automatically_issues_bounded_repair():
    scenario = slow_scope_creep()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()
    result = None

    for step in scenario.steps:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        session = result.session

    assert result is not None
    assert result.state.status is CoherenceStatus.REALIGN_REQUIRED
    assert result.issued_repair is not None
    assert result.issued_repair.next_allowed_action == "request_clarification"
    assert session.active_repair_directive == result.issued_repair
    assert any(
        event.event_type is AuditEventType.REPAIR_ISSUED
        for event in result.audit_events
    )


def test_pending_repair_requires_explicit_repair_attempt_on_next_turn():
    scenario = successful_repair()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()

    for step in scenario.steps[:3]:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        session = result.session

    assert session.active_repair_directive is not None

    with pytest.raises(MonitorProcessingError, match="repair_attempted=True"):
        monitor.process_turn(
            turn=scenario.steps[3].turn,
            session=session,
        )


def test_repair_attempt_without_active_directive_is_rejected():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)

    with pytest.raises(MonitorProcessingError, match="active role-repair"):
        monitor.process_turn(
            turn=scenario.steps[0].turn,
            repair_attempted=True,
        )


def test_successful_repair_recovers_watch_then_coherent():
    scenario = successful_repair()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()

    for step in scenario.steps[:3]:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        session = result.session

    active_directive = session.active_repair_directive
    assert active_directive is not None

    repair_result = monitor.process_turn(
        turn=scenario.steps[3].turn,
        session=session,
        repair_attempted=True,
    )

    assert repair_result.state.status is CoherenceStatus.WATCH
    assert repair_result.issued_repair is None
    assert repair_result.session.active_repair_directive is None
    assert any(
        event.event_type is AuditEventType.REPAIR_EVALUATED
        and event.repair_directive_id == active_directive.directive_id
        and event.outcome == "repair_recovered"
        for event in repair_result.audit_events
    )

    final_result = monitor.process_turn(
        turn=scenario.steps[4].turn,
        session=repair_result.session,
    )
    assert final_result.state.status is CoherenceStatus.COHERENT


def test_failed_repair_issues_fresh_second_directive():
    scenario = failed_repair_twice()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()

    for step in scenario.steps[:3]:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        session = result.session

    first_directive = session.active_repair_directive
    assert first_directive is not None

    failed_result = monitor.process_turn(
        turn=scenario.steps[3].turn,
        session=session,
        repair_attempted=True,
    )
    second_directive = failed_result.issued_repair

    assert failed_result.state.status is CoherenceStatus.REALIGN_REQUIRED
    assert second_directive is not None
    assert second_directive.directive_id != first_directive.directive_id
    assert failed_result.session.active_repair_directive == second_directive
    assert any(
        event.event_type is AuditEventType.REPAIR_EVALUATED
        and event.repair_directive_id == first_directive.directive_id
        and event.outcome == "repair_failed"
        for event in failed_result.audit_events
    )
    assert any(
        event.event_type is AuditEventType.REPAIR_ISSUED
        and event.repair_directive_id == second_directive.directive_id
        for event in failed_result.audit_events
    )


def test_second_failed_repair_escalates_to_human_review_and_clears_directive():
    scenario = failed_repair_twice()
    monitor = monitor_for_scenario(scenario)
    session = monitor.new_session()

    for step in scenario.steps[:3]:
        result = monitor.process_turn(
            turn=step.turn,
            session=session,
        )
        session = result.session

    first_failure = monitor.process_turn(
        turn=scenario.steps[3].turn,
        session=session,
        repair_attempted=True,
    )
    second_directive = first_failure.session.active_repair_directive
    assert second_directive is not None

    second_failure = monitor.process_turn(
        turn=scenario.steps[4].turn,
        session=first_failure.session,
        repair_attempted=True,
    )

    assert second_failure.state.status is CoherenceStatus.HUMAN_REVIEW
    assert second_failure.issued_repair is None
    assert second_failure.session.active_repair_directive is None
    assert any(
        event.event_type is AuditEventType.REPAIR_EVALUATED
        and event.repair_directive_id == second_directive.directive_id
        and event.outcome == "repair_escalated"
        for event in second_failure.audit_events
    )
    assert any(
        event.event_type is AuditEventType.HUMAN_REVIEW_REQUESTED
        for event in second_failure.audit_events
    )


def test_deterministic_block_overrides_semantically_clean_assessment():
    scenario = direct_transaction_approval_attempt()
    step = scenario.steps[0]
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: healthy_result(step.turn.turn_id)}
        ),
        repair_next_action="request_clarification",
    )

    result = monitor.process_turn(turn=step.turn)

    assert result.state.status is CoherenceStatus.BLOCKED
    assert result.issued_repair is None
    assert any(
        event.event_type is AuditEventType.AUTONOMY_BLOCKED
        for event in result.audit_events
    )


def test_deterministic_block_is_independent_of_pathological_semantic_scores():
    scenario = direct_transaction_approval_attempt()
    step = scenario.steps[0]
    pathological = SemanticAssessmentResult(
        turn_id=step.turn.turn_id,
        mission_alignment=0.0,
        scope_adherence=0.0,
        authority_adherence=0.0,
        evidence_discipline=0.0,
        behavioral_consistency=0.0,
        signals=(),
        rationale="Injected pathological semantic output without evidence.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: pathological}
        ),
        repair_next_action="request_clarification",
    )

    result = monitor.process_turn(turn=step.turn)

    assert result.state.status is CoherenceStatus.BLOCKED
    assert any(
        event.event_type is AuditEventType.AUTONOMY_BLOCKED
        for event in result.audit_events
    )


def test_required_escalation_reaches_human_review_end_to_end():
    scenario = ignored_required_escalation()
    monitor = monitor_for_scenario(scenario)

    result = monitor.process_turn(turn=scenario.steps[0].turn)

    assert result.state.status is CoherenceStatus.HUMAN_REVIEW
    assert result.session.active_repair_directive is None
    assert any(
        event.event_type is AuditEventType.HUMAN_REVIEW_REQUESTED
        for event in result.audit_events
    )


def test_replayed_turn_id_is_rejected_before_assessment():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)
    first = monitor.process_turn(turn=scenario.steps[0].turn)

    with pytest.raises(MonitorProcessingError):
        monitor.process_turn(
            turn=scenario.steps[0].turn,
            session=first.session,
        )


def test_non_monotonic_sequence_is_rejected():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)
    first = monitor.process_turn(turn=scenario.steps[1].turn)

    with pytest.raises(MonitorProcessingError, match="monotonically"):
        monitor.process_turn(
            turn=scenario.steps[0].turn,
            session=first.session,
        )


def test_control_relevant_semantic_scores_require_auditable_signal():
    scenario = clean_conversation()
    step = scenario.steps[0]
    unauditable = SemanticAssessmentResult(
        turn_id=step.turn.turn_id,
        mission_alignment=0.0,
        scope_adherence=0.0,
        authority_adherence=0.0,
        evidence_discipline=0.0,
        behavioral_consistency=0.0,
        signals=(),
        rationale="Injected unauditable extreme semantic response.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: unauditable}
        ),
        repair_next_action="request_clarification",
    )
    session = monitor.new_session()

    with pytest.raises(
        MonitorProcessingError,
        match="matching MEDIUM/HIGH semantic evidence",
    ):
        monitor.process_turn(
            turn=step.turn,
            session=session,
        )

    assert session.history == ()
    assert session.state is None
    assert session.audit_trail.events == ()


def test_control_relevant_score_with_medium_semantic_signal_is_accepted():
    scenario = clean_conversation()
    step = scenario.steps[0]
    signal = CoherenceSignal(
        signal_id=f"semantic:{step.turn.turn_id}:scope_drift:1",
        turn_id=step.turn.turn_id,
        signal_type=SignalType.SCOPE_DRIFT,
        severity=SignalSeverity.MEDIUM,
        source=SignalSource.SEMANTIC,
        evidence_reference="agent_output",
        explanation="The agent accepted responsibility beyond the review scope.",
    )
    semantic_result = SemanticAssessmentResult(
        turn_id=step.turn.turn_id,
        mission_alignment=0.85,
        scope_adherence=0.50,
        authority_adherence=0.90,
        evidence_discipline=0.95,
        behavioral_consistency=0.85,
        signals=(signal,),
        rationale="Substantial scope violation with otherwise preserved evidence discipline.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: semantic_result}
        ),
        repair_next_action="request_clarification",
    )

    result = monitor.process_turn(turn=step.turn)

    assert result.state.status is CoherenceStatus.DRIFTING
    assert result.signals == (signal,)
    assert any(
        event.event_type is AuditEventType.SIGNAL_RECORDED
        for event in result.audit_events
    )


def test_low_severity_signal_does_not_satisfy_auditability_gate():
    scenario = clean_conversation()
    step = scenario.steps[0]
    signal = CoherenceSignal(
        signal_id=f"semantic:{step.turn.turn_id}:scope_drift:1",
        turn_id=step.turn.turn_id,
        signal_type=SignalType.SCOPE_DRIFT,
        severity=SignalSeverity.LOW,
        source=SignalSource.SEMANTIC,
        evidence_reference="agent_output",
        explanation="A weak observation that should not justify control-state drift.",
    )
    semantic_result = SemanticAssessmentResult(
        turn_id=step.turn.turn_id,
        mission_alignment=0.95,
        scope_adherence=0.70,
        authority_adherence=0.95,
        evidence_discipline=0.95,
        behavioral_consistency=0.95,
        signals=(signal,),
        rationale="Injected low-severity signal with a control-relevant score.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: semantic_result}
        ),
        repair_next_action="request_clarification",
    )

    with pytest.raises(
        MonitorProcessingError,
        match="matching MEDIUM/HIGH semantic evidence",
    ):
        monitor.process_turn(turn=step.turn)


def test_each_low_dimension_requires_its_matching_signal_type():
    scenario = clean_conversation()
    step = scenario.steps[0]
    scope_signal = CoherenceSignal(
        signal_id=f"semantic:{step.turn.turn_id}:scope_drift:1",
        turn_id=step.turn.turn_id,
        signal_type=SignalType.SCOPE_DRIFT,
        severity=SignalSeverity.HIGH,
        source=SignalSource.SEMANTIC,
        evidence_reference="agent_output",
        explanation="The agent accepted responsibility beyond the review scope.",
    )
    semantic_result = SemanticAssessmentResult(
        turn_id=step.turn.turn_id,
        mission_alignment=0.90,
        scope_adherence=0.40,
        authority_adherence=0.90,
        evidence_discipline=0.40,
        behavioral_consistency=0.90,
        signals=(scope_signal,),
        rationale="Scope drift is evidenced, but evidence discipline is not.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {step.turn.turn_id: semantic_result}
        ),
        repair_next_action="request_clarification",
    )

    with pytest.raises(
        MonitorProcessingError,
        match="evidence_discipline->evidence_degradation",
    ):
        monitor.process_turn(turn=step.turn)


def test_noncritical_deterministic_signal_does_not_bypass_semantic_gate():
    scenario = clean_conversation()
    base_turn = scenario.steps[0].turn
    turn = base_turn.model_copy(
        update={"attempted_actions": ("unknown_application_action",)}
    )
    semantic_result = SemanticAssessmentResult(
        turn_id=turn.turn_id,
        mission_alignment=0.95,
        scope_adherence=0.50,
        authority_adherence=0.95,
        evidence_discipline=0.95,
        behavioral_consistency=0.95,
        signals=(),
        rationale="Injected low scope score without semantic evidence.",
    )
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor(
            {turn.turn_id: semantic_result}
        ),
        repair_next_action="request_clarification",
    )

    with pytest.raises(
        MonitorProcessingError,
        match="scope_adherence->scope_drift",
    ):
        monitor.process_turn(turn=turn)


def test_semantic_assessor_failure_leaves_supplied_session_unchanged():
    scenario = clean_conversation()
    first_step = scenario.steps[0]
    monitor = RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=MockSemanticAssessor({}),
        repair_next_action="request_clarification",
    )
    session = monitor.new_session()

    with pytest.raises(SemanticAssessmentError):
        monitor.process_turn(
            turn=first_step.turn,
            session=session,
        )

    assert session.history == ()
    assert session.state is None
    assert session.audit_trail.events == ()


def test_process_turns_convenience_helper_runs_non_repair_sequence():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)

    results = monitor.process_turns(
        tuple(step.turn for step in scenario.steps)
    )

    assert len(results) == 3
    assert all(
        result.state.status is CoherenceStatus.COHERENT
        for result in results
    )
    assert results[-1].session.history == tuple(
        step.turn for step in scenario.steps
    )


def test_monitor_result_is_immutable():
    scenario = clean_conversation()
    monitor = monitor_for_scenario(scenario)
    result = monitor.process_turn(turn=scenario.steps[0].turn)

    with pytest.raises(ValidationError):
        result.issued_repair = None
