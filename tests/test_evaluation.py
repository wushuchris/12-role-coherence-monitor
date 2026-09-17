"""Tests for the formal role-coherence evaluation runner."""

import pytest

from src.role_coherence_monitor.evaluation import (
    evaluate_scenario,
    evaluate_scenarios,
)
from src.role_coherence_monitor.scenarios import (
    EvaluationScenario,
    compliance_role_contract,
    core_evaluation_scenarios,
    direct_transaction_approval_attempt,
    harmless_topic_variation,
    slow_scope_creep,
)
from src.role_coherence_monitor.schemas import CoherenceStatus
from src.role_coherence_monitor.semantic import (
    MockSemanticAssessor,
    SemanticAssessmentResult,
)


def fixture_mock(scenarios):
    """Build a deterministic mock from scenario fixture semantic evidence."""

    results = {}
    for scenario in scenarios:
        for step in scenario.steps:
            results[step.turn.turn_id] = SemanticAssessmentResult(
                turn_id=step.turn.turn_id,
                mission_alignment=step.assessment.mission_alignment,
                scope_adherence=step.assessment.scope_adherence,
                authority_adherence=step.assessment.authority_adherence,
                evidence_discipline=step.assessment.evidence_discipline,
                behavioral_consistency=step.assessment.behavioral_consistency,
                signals=step.semantic_signals,
                rationale="Fixture-backed evaluation mock.",
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
        rationale="Synthetic healthy semantic result.",
    )


def test_core_fixture_report_counts_scenarios_and_turns():
    report = evaluate_scenarios(
        core_evaluation_scenarios(),
        contract=compliance_role_contract(),
    )

    assert report.metrics.scenario_count == 8
    assert report.metrics.turn_count == 22


def test_core_fixture_report_has_perfect_exact_control_accuracy():
    report = evaluate_scenarios(
        core_evaluation_scenarios(),
        contract=compliance_role_contract(),
    )

    assert report.metrics.scenario_pass_rate == 1.0
    assert report.metrics.exact_state_accuracy == 1.0
    assert all(scenario.passed for scenario in report.scenarios)


def test_core_fixture_report_separates_detection_metrics():
    report = evaluate_scenarios(
        core_evaluation_scenarios(),
        contract=compliance_role_contract(),
    )
    metrics = report.metrics

    assert metrics.deviation_precision == 1.0
    assert metrics.deviation_recall == 1.0
    assert metrics.false_positive_rate == 0.0


def test_core_fixture_report_scores_terminal_controls_and_repairs():
    report = evaluate_scenarios(
        core_evaluation_scenarios(),
        contract=compliance_role_contract(),
    )
    metrics = report.metrics

    assert metrics.terminal_control_accuracy == 1.0
    assert metrics.repair_transition_accuracy == 1.0


def test_core_fixture_report_tracks_detection_delay_by_scenario():
    report = evaluate_scenarios(
        core_evaluation_scenarios(),
        contract=compliance_role_contract(),
    )
    metrics = report.metrics

    assert metrics.deviation_scenario_count == 5
    assert metrics.detected_deviation_scenario_count == 5
    assert metrics.missed_deviation_scenario_count == 0
    assert metrics.average_detection_delay_turns == 0.0


def test_fixture_backed_mock_reproduces_fixture_evaluation():
    scenarios = core_evaluation_scenarios()
    report = evaluate_scenarios(
        scenarios,
        contract=compliance_role_contract(),
        assessor=fixture_mock(scenarios),
    )

    assert report.metrics.scenario_pass_rate == 1.0
    assert report.metrics.exact_state_accuracy == 1.0
    assert report.metrics.false_positive_rate == 0.0


def test_semantic_false_positive_is_visible_in_formal_metrics():
    scenario = harmless_topic_variation()
    first, second = scenario.steps
    assessor = MockSemanticAssessor(
        {
            first.turn.turn_id: SemanticAssessmentResult(
                turn_id=first.turn.turn_id,
                mission_alignment=0.95,
                scope_adherence=0.75,
                authority_adherence=0.95,
                evidence_discipline=0.95,
                behavioral_consistency=0.95,
                signals=(),
                rationale="Injected false-positive scope concern.",
            ),
            second.turn.turn_id: healthy_result(second.turn.turn_id),
        }
    )

    report = evaluate_scenarios(
        (scenario,),
        contract=compliance_role_contract(),
        assessor=assessor,
    )

    assert report.metrics.scenario_pass_rate == 0.0
    assert report.metrics.exact_state_accuracy == 0.5
    assert report.metrics.false_positive_rate == 0.5
    assert report.metrics.deviation_precision == 0.0
    assert report.scenarios[0].turn_results[0].actual_status is CoherenceStatus.WATCH


def test_missed_semantic_drift_is_visible_in_recall_and_scenario_metrics():
    scenario = slow_scope_creep()
    assessor = MockSemanticAssessor(
        {
            step.turn.turn_id: healthy_result(step.turn.turn_id)
            for step in scenario.steps
        }
    )

    report = evaluate_scenarios(
        (scenario,),
        contract=compliance_role_contract(),
        assessor=assessor,
    )

    assert report.metrics.deviation_recall == 0.0
    assert report.metrics.deviation_precision is None
    assert report.metrics.missed_deviation_scenario_count == 1
    assert report.metrics.detected_deviation_scenario_count == 0
    assert report.metrics.average_detection_delay_turns is None


def test_deterministic_hard_control_survives_semantically_clean_assessment():
    scenario = direct_transaction_approval_attempt()
    step = scenario.steps[0]
    assessor = MockSemanticAssessor(
        {step.turn.turn_id: healthy_result(step.turn.turn_id)}
    )

    result = evaluate_scenario(
        scenario,
        contract=compliance_role_contract(),
        assessor=assessor,
    )

    assert result.passed is True
    assert result.turn_results[0].actual_status is CoherenceStatus.BLOCKED
    assert result.turn_results[0].deterministic_signal_count >= 1
    assert result.turn_results[0].semantic_signal_count == 0


def test_exact_state_mismatch_is_not_hidden_by_binary_deviation_detection():
    scenario = slow_scope_creep()
    first, second, third, fourth = scenario.steps
    assessor = MockSemanticAssessor(
        {
            first.turn.turn_id: healthy_result(first.turn.turn_id),
            second.turn.turn_id: SemanticAssessmentResult(
                turn_id=second.turn.turn_id,
                mission_alignment=0.95,
                scope_adherence=0.40,
                authority_adherence=0.95,
                evidence_discipline=0.95,
                behavioral_consistency=0.75,
                signals=second.semantic_signals,
                rationale="Injected severe scope drift.",
            ),
            third.turn.turn_id: SemanticAssessmentResult(
                turn_id=third.turn.turn_id,
                mission_alignment=third.assessment.mission_alignment,
                scope_adherence=third.assessment.scope_adherence,
                authority_adherence=third.assessment.authority_adherence,
                evidence_discipline=third.assessment.evidence_discipline,
                behavioral_consistency=third.assessment.behavioral_consistency,
                signals=third.semantic_signals,
                rationale="Fixture-backed third turn.",
            ),
            fourth.turn.turn_id: SemanticAssessmentResult(
                turn_id=fourth.turn.turn_id,
                mission_alignment=fourth.assessment.mission_alignment,
                scope_adherence=fourth.assessment.scope_adherence,
                authority_adherence=fourth.assessment.authority_adherence,
                evidence_discipline=fourth.assessment.evidence_discipline,
                behavioral_consistency=fourth.assessment.behavioral_consistency,
                signals=fourth.semantic_signals,
                rationale="Fixture-backed fourth turn.",
            ),
        }
    )

    report = evaluate_scenarios(
        (scenario,),
        contract=compliance_role_contract(),
        assessor=assessor,
    )

    second_result = report.scenarios[0].turn_results[1]
    assert second_result.expected_status is CoherenceStatus.WATCH
    assert second_result.actual_status is CoherenceStatus.DRIFTING
    assert second_result.expected_deviation is True
    assert second_result.actual_deviation is True
    assert second_result.exact_match is False
    assert report.metrics.exact_state_accuracy < 1.0


def test_evaluator_rejects_empty_scenario_collection():
    with pytest.raises(ValueError, match="At least one evaluation scenario"):
        evaluate_scenarios(
            (),
            contract=compliance_role_contract(),
        )


def test_scenario_result_preserves_stable_turn_order():
    scenario: EvaluationScenario = slow_scope_creep()

    result = evaluate_scenario(
        scenario,
        contract=compliance_role_contract(),
    )

    assert tuple(turn.sequence_number for turn in result.turn_results) == (1, 2, 3, 4)
    assert tuple(turn.turn_id for turn in result.turn_results) == tuple(
        step.turn.turn_id for step in scenario.steps
    )
