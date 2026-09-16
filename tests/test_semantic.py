"""Tests for the model-agnostic semantic assessor boundary and deterministic mock."""

import pytest
from pydantic import ValidationError

from src.role_coherence_monitor.deterministic import check_turn_against_contract
from src.role_coherence_monitor.scenarios import (
    compliance_role_contract,
    slow_scope_creep,
)
from src.role_coherence_monitor.schemas import (
    AssessmentMode,
    CoherenceSignal,
    CoherenceStatus,
    InteractionTurn,
    SignalSeverity,
    SignalSource,
    SignalType,
)
from src.role_coherence_monitor.semantic import (
    MockSemanticAssessor,
    SemanticAssessmentError,
    SemanticAssessmentResult,
    SemanticAssessor,
    to_turn_assessment,
)
from src.role_coherence_monitor.state_machine import advance_coherence_state


def make_turn(turn_id: str = "turn-001") -> InteractionTurn:
    return InteractionTurn(
        turn_id=turn_id,
        sequence_number=1,
        user_input="Review the supplied policy evidence.",
        agent_output="I will review the evidence against policy.",
    )


def make_semantic_signal(
    *,
    turn_id: str = "turn-001",
    source: SignalSource = SignalSource.SEMANTIC,
) -> CoherenceSignal:
    return CoherenceSignal(
        signal_id=f"semantic:{turn_id}:scope_drift",
        turn_id=turn_id,
        signal_type=SignalType.SCOPE_DRIFT,
        severity=SignalSeverity.MEDIUM,
        source=source,
        evidence_reference="agent_output",
        explanation="The response expanded beyond the assigned review scope.",
    )


def make_result(
    *,
    turn_id: str = "turn-001",
    signals: tuple[CoherenceSignal, ...] = (),
) -> SemanticAssessmentResult:
    return SemanticAssessmentResult(
        turn_id=turn_id,
        mission_alignment=0.90,
        scope_adherence=0.75,
        authority_adherence=0.95,
        evidence_discipline=0.95,
        behavioral_consistency=0.78,
        signals=signals,
        rationale="Synthetic semantic assessment result.",
    )


def test_semantic_result_accepts_bounded_scores_and_semantic_signals():
    signal = make_semantic_signal()
    result = make_result(signals=(signal,))

    assert result.turn_id == "turn-001"
    assert result.scope_adherence == 0.75
    assert result.signals == (signal,)


def test_semantic_result_rejects_score_outside_zero_to_one():
    with pytest.raises(ValidationError):
        SemanticAssessmentResult(
            turn_id="turn-001",
            mission_alignment=1.1,
            scope_adherence=0.75,
            authority_adherence=0.95,
            evidence_discipline=0.95,
            behavioral_consistency=0.78,
            rationale="Invalid score should be rejected.",
        )


def test_semantic_result_cannot_set_final_control_status():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SemanticAssessmentResult.model_validate(
            {
                "turn_id": "turn-001",
                "mission_alignment": 0.90,
                "scope_adherence": 0.75,
                "authority_adherence": 0.95,
                "evidence_discipline": 0.95,
                "behavioral_consistency": 0.78,
                "rationale": "Possible scope drift.",
                "status": "BLOCKED",
            }
        )


def test_semantic_result_rejects_deterministic_source_signal():
    deterministic_signal = make_semantic_signal(source=SignalSource.DETERMINISTIC)

    with pytest.raises(ValidationError, match="only semantic signals"):
        make_result(signals=(deterministic_signal,))


def test_semantic_result_rejects_signal_from_other_turn():
    wrong_turn_signal = make_semantic_signal(turn_id="turn-999")

    with pytest.raises(ValidationError, match="assessed turn"):
        make_result(turn_id="turn-001", signals=(wrong_turn_signal,))


def test_result_converts_to_turn_assessment_without_control_authority():
    signal = make_semantic_signal()
    result = make_result(signals=(signal,))

    assessment = to_turn_assessment(result)

    assert assessment.turn_id == result.turn_id
    assert assessment.mode is AssessmentMode.SEMANTIC
    assert assessment.signal_ids == (signal.signal_id,)
    assert not hasattr(assessment, "status")


def test_mock_semantic_assessor_satisfies_protocol():
    assessor = MockSemanticAssessor({"turn-001": make_result()})

    assert isinstance(assessor, SemanticAssessor)


def test_mock_returns_configured_result_for_stable_turn_id():
    result = make_result()
    assessor = MockSemanticAssessor({"turn-001": result})

    returned = assessor.assess(
        contract=compliance_role_contract(),
        turn=make_turn(),
    )

    assert returned is result


def test_mock_fails_closed_when_turn_is_not_configured():
    assessor = MockSemanticAssessor({})

    with pytest.raises(SemanticAssessmentError, match="No mock semantic result"):
        assessor.assess(
            contract=compliance_role_contract(),
            turn=make_turn(),
        )


def test_slow_scope_creep_runs_through_mock_semantic_assessor():
    scenario = slow_scope_creep()
    contract = compliance_role_contract()
    configured_results = {}

    for step in scenario.steps:
        configured_results[step.turn.turn_id] = SemanticAssessmentResult(
            turn_id=step.turn.turn_id,
            mission_alignment=step.assessment.mission_alignment,
            scope_adherence=step.assessment.scope_adherence,
            authority_adherence=step.assessment.authority_adherence,
            evidence_discipline=step.assessment.evidence_discipline,
            behavioral_consistency=step.assessment.behavioral_consistency,
            signals=step.semantic_signals,
            rationale="Fixture-backed mock semantic assessment.",
        )

    assessor = MockSemanticAssessor(configured_results)
    previous_state = None
    statuses = []
    history = []

    for step in scenario.steps:
        semantic_result = assessor.assess(
            contract=contract,
            turn=step.turn,
            history=tuple(history),
        )
        assessment = to_turn_assessment(
            semantic_result,
            assessment_id=step.assessment.assessment_id,
            mode=AssessmentMode.HYBRID,
        )
        deterministic_signals = check_turn_against_contract(contract, step.turn)
        signals = deterministic_signals + semantic_result.signals
        state = advance_coherence_state(
            turn=step.turn,
            assessment=assessment,
            signals=signals,
            previous_state=previous_state,
        )
        statuses.append(state.status)
        previous_state = state
        history.append(step.turn)

    assert tuple(statuses) == (
        CoherenceStatus.COHERENT,
        CoherenceStatus.WATCH,
        CoherenceStatus.DRIFTING,
        CoherenceStatus.REALIGN_REQUIRED,
    )
