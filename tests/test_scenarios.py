"""Integration tests for the public-safe long-horizon evaluation scenarios."""

import pytest

from src.role_coherence_monitor.deterministic import check_turn_against_contract
from src.role_coherence_monitor.repair import build_role_repair_directive
from src.role_coherence_monitor.scenarios import (
    EvaluationScenario,
    clean_conversation,
    core_evaluation_scenarios,
    direct_transaction_approval_attempt,
    failed_repair_twice,
    harmless_topic_variation,
    ignored_required_escalation,
    memory_claims_role_changed,
    slow_scope_creep,
    successful_repair,
    compliance_role_contract,
)
from src.role_coherence_monitor.schemas import CoherenceStatus, SignalSource, SignalType
from src.role_coherence_monitor.state_machine import advance_coherence_state


def run_scenario(scenario: EvaluationScenario):
    """Run one scenario through real deterministic checks and state transitions."""

    contract = compliance_role_contract()
    previous_state = None
    states = []
    all_signals = []

    for step in scenario.steps:
        deterministic_signals = check_turn_against_contract(contract, step.turn)
        signals = deterministic_signals + step.semantic_signals
        state = advance_coherence_state(
            turn=step.turn,
            assessment=step.assessment,
            signals=signals,
            previous_state=previous_state,
            repair_attempted=step.repair_attempted,
        )
        states.append(state)
        all_signals.append(signals)
        previous_state = state

    return tuple(states), tuple(all_signals)


@pytest.mark.parametrize(
    "scenario_factory",
    [
        clean_conversation,
        harmless_topic_variation,
        slow_scope_creep,
        direct_transaction_approval_attempt,
        ignored_required_escalation,
        memory_claims_role_changed,
        successful_repair,
        failed_repair_twice,
    ],
)
def test_scenario_reaches_expected_state_at_every_turn(scenario_factory):
    scenario = scenario_factory()
    states, _ = run_scenario(scenario)

    assert tuple(state.status for state in states) == tuple(
        step.expected_status for step in scenario.steps
    )


def test_core_scenario_registry_has_unique_public_ids():
    scenarios = core_evaluation_scenarios()
    scenario_ids = [scenario.scenario_id for scenario in scenarios]

    assert len(scenarios) == 8
    assert len(set(scenario_ids)) == len(scenario_ids)
    assert all(scenario_id.strip() for scenario_id in scenario_ids)


def test_each_scenario_has_monotonic_turn_numbers_and_unique_turn_ids():
    for scenario in core_evaluation_scenarios():
        sequence_numbers = [step.turn.sequence_number for step in scenario.steps]
        turn_ids = [step.turn.turn_id for step in scenario.steps]

        assert sequence_numbers == list(range(1, len(scenario.steps) + 1))
        assert len(set(turn_ids)) == len(turn_ids)


def test_fixture_assessment_signal_ids_match_pre_labeled_semantic_signals():
    for scenario in core_evaluation_scenarios():
        for step in scenario.steps:
            fixture_signal_ids = {signal.signal_id for signal in step.semantic_signals}
            assessment_signal_ids = set(step.assessment.signal_ids)

            assert assessment_signal_ids == fixture_signal_ids


def test_direct_approval_block_is_derived_by_deterministic_checker():
    scenario = direct_transaction_approval_attempt()
    states, signals_by_turn = run_scenario(scenario)
    signals = signals_by_turn[0]

    assert states[0].status is CoherenceStatus.BLOCKED
    assert any(
        signal.signal_type is SignalType.PROHIBITED_ACTION
        and signal.source is SignalSource.DETERMINISTIC
        for signal in signals
    )
    assert scenario.steps[0].semantic_signals == ()


def test_required_escalation_human_review_is_derived_deterministically():
    scenario = ignored_required_escalation()
    states, signals_by_turn = run_scenario(scenario)
    signals = signals_by_turn[0]

    assert states[0].status is CoherenceStatus.HUMAN_REVIEW
    assert any(
        signal.signal_type is SignalType.MISSING_ESCALATION
        and signal.source is SignalSource.DETERMINISTIC
        for signal in signals
    )


def test_memory_claim_does_not_mutate_authority_or_create_false_positive():
    scenario = memory_claims_role_changed()
    states, signals_by_turn = run_scenario(scenario)

    assert states[0].status is CoherenceStatus.COHERENT
    assert signals_by_turn[0] == ()
    assert compliance_role_contract().role_id == "compliance-reviewer-v1"
    assert compliance_role_contract().version == "1.0.0"


def test_slow_scope_creep_uses_semantic_evidence_for_gradual_drift():
    scenario = slow_scope_creep()
    states, signals_by_turn = run_scenario(scenario)

    assert tuple(state.status for state in states) == (
        CoherenceStatus.COHERENT,
        CoherenceStatus.WATCH,
        CoherenceStatus.DRIFTING,
        CoherenceStatus.REALIGN_REQUIRED,
    )
    assert all(
        signal.source is SignalSource.SEMANTIC
        for signals in signals_by_turn[1:]
        for signal in signals
    )


def test_successful_repair_scenario_builds_directive_from_realign_state():
    scenario = successful_repair()
    states, signals_by_turn = run_scenario(scenario)
    realign_index = 2
    realign_state = states[realign_index]
    triggering_signals = signals_by_turn[realign_index]

    directive = build_role_repair_directive(
        contract=compliance_role_contract(),
        state=realign_state,
        triggering_signals=triggering_signals,
        next_allowed_action="request_clarification",
    )

    assert realign_state.status is CoherenceStatus.REALIGN_REQUIRED
    assert directive.next_allowed_action == "request_clarification"
    assert directive.triggering_signal_ids == tuple(
        signal.signal_id for signal in triggering_signals
    )
    assert "compliance-reviewer-v1" in directive.corrective_instruction


def test_successful_repair_recovers_gradually_after_realignment():
    states, _ = run_scenario(successful_repair())

    assert tuple(state.status for state in states[-3:]) == (
        CoherenceStatus.REALIGN_REQUIRED,
        CoherenceStatus.WATCH,
        CoherenceStatus.COHERENT,
    )


def test_two_failed_repairs_escalate_to_human_review():
    states, _ = run_scenario(failed_repair_twice())

    assert states[-2].status is CoherenceStatus.REALIGN_REQUIRED
    assert states[-2].repair_attempts == 1
    assert states[-1].status is CoherenceStatus.HUMAN_REVIEW
    assert states[-1].repair_attempts == 2
