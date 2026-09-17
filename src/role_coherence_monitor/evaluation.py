"""Formal evaluation runner for longitudinal role-coherence behavior.

The evaluator runs the real deterministic checker and state machine for every turn.
Semantic evidence can come from the public-safe fixtures, a deterministic mock, or
any future SemanticAssessor implementation without changing evaluation logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .deterministic import check_turn_against_contract
from .scenarios import EvaluationScenario
from .schemas import (
    AssessmentMode,
    CoherenceStatus,
    RoleContract,
    SignalSource,
)
from .semantic import SemanticAssessor, to_turn_assessment
from .state_machine import advance_coherence_state


_TERMINAL_CONTROL_STATES = {
    CoherenceStatus.BLOCKED,
    CoherenceStatus.HUMAN_REVIEW,
}


class TurnEvaluationResult(BaseModel):
    """Observed versus expected control behavior for one evaluated turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    turn_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=1)
    expected_status: CoherenceStatus
    actual_status: CoherenceStatus
    exact_match: bool
    expected_deviation: bool
    actual_deviation: bool
    repair_attempted: bool
    deterministic_signal_count: int = Field(ge=0)
    semantic_signal_count: int = Field(ge=0)
    signal_ids: tuple[str, ...] = ()


class ScenarioEvaluationResult(BaseModel):
    """Evaluation result for one complete long-horizon scenario."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    passed: bool
    turn_results: tuple[TurnEvaluationResult, ...] = Field(min_length=1)
    expected_first_deviation_turn: int | None = Field(default=None, ge=1)
    actual_first_deviation_turn: int | None = Field(default=None, ge=1)
    detection_delay_turns: int | None = None


class EvaluationMetrics(BaseModel):
    """Aggregate metrics that keep exact control accuracy separate from detection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_count: int = Field(ge=1)
    turn_count: int = Field(ge=1)
    scenario_pass_rate: float = Field(ge=0.0, le=1.0)
    exact_state_accuracy: float = Field(ge=0.0, le=1.0)
    deviation_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    deviation_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    false_positive_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    terminal_control_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    repair_transition_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    deviation_scenario_count: int = Field(ge=0)
    detected_deviation_scenario_count: int = Field(ge=0)
    missed_deviation_scenario_count: int = Field(ge=0)
    average_detection_delay_turns: float | None = Field(default=None, ge=0.0)


class EvaluationReport(BaseModel):
    """Complete formal evaluation output suitable for tests, UI, or export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    metrics: EvaluationMetrics
    scenarios: tuple[ScenarioEvaluationResult, ...] = Field(min_length=1)


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _first_deviation_turn(
    results: Sequence[TurnEvaluationResult],
    *,
    expected: bool,
) -> int | None:
    for result in results:
        is_deviation = (
            result.expected_deviation if expected else result.actual_deviation
        )
        if is_deviation:
            return result.sequence_number
    return None


def evaluate_scenario(
    scenario: EvaluationScenario,
    *,
    contract: RoleContract,
    assessor: SemanticAssessor | None = None,
) -> ScenarioEvaluationResult:
    """Run one scenario through the actual control pipeline.

    When assessor is None, the scenario's pre-labeled semantic evidence is used.
    When an assessor is provided, its output replaces only semantic evidence;
    deterministic checks still run independently from structured application data.
    """

    previous_state = None
    history = []
    turn_results: list[TurnEvaluationResult] = []

    for step in scenario.steps:
        deterministic_signals = check_turn_against_contract(contract, step.turn)

        if assessor is None:
            assessment = step.assessment
            semantic_signals = step.semantic_signals
        else:
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
            semantic_signals = semantic_result.signals

        signals = deterministic_signals + semantic_signals
        state = advance_coherence_state(
            turn=step.turn,
            assessment=assessment,
            signals=signals,
            previous_state=previous_state,
            repair_attempted=step.repair_attempted,
        )

        expected_deviation = step.expected_status is not CoherenceStatus.COHERENT
        actual_deviation = state.status is not CoherenceStatus.COHERENT

        turn_results.append(
            TurnEvaluationResult(
                turn_id=step.turn.turn_id,
                sequence_number=step.turn.sequence_number,
                expected_status=step.expected_status,
                actual_status=state.status,
                exact_match=state.status is step.expected_status,
                expected_deviation=expected_deviation,
                actual_deviation=actual_deviation,
                repair_attempted=step.repair_attempted,
                deterministic_signal_count=sum(
                    signal.source is SignalSource.DETERMINISTIC
                    for signal in signals
                ),
                semantic_signal_count=sum(
                    signal.source is SignalSource.SEMANTIC for signal in signals
                ),
                signal_ids=tuple(signal.signal_id for signal in signals),
            )
        )

        previous_state = state
        history.append(step.turn)

    frozen_results = tuple(turn_results)
    expected_first = _first_deviation_turn(frozen_results, expected=True)
    actual_first = _first_deviation_turn(frozen_results, expected=False)

    detection_delay = None
    if expected_first is not None and actual_first is not None:
        detection_delay = max(0, actual_first - expected_first)

    return ScenarioEvaluationResult(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        passed=all(result.exact_match for result in frozen_results),
        turn_results=frozen_results,
        expected_first_deviation_turn=expected_first,
        actual_first_deviation_turn=actual_first,
        detection_delay_turns=detection_delay,
    )


def _aggregate_metrics(
    scenario_results: Sequence[ScenarioEvaluationResult],
) -> EvaluationMetrics:
    turns = tuple(
        turn
        for scenario in scenario_results
        for turn in scenario.turn_results
    )

    exact_matches = sum(turn.exact_match for turn in turns)
    passed_scenarios = sum(scenario.passed for scenario in scenario_results)

    true_positive = sum(
        turn.expected_deviation and turn.actual_deviation for turn in turns
    )
    false_positive = sum(
        not turn.expected_deviation and turn.actual_deviation for turn in turns
    )
    false_negative = sum(
        turn.expected_deviation and not turn.actual_deviation for turn in turns
    )
    expected_coherent = sum(not turn.expected_deviation for turn in turns)

    terminal_turns = tuple(
        turn for turn in turns if turn.expected_status in _TERMINAL_CONTROL_STATES
    )
    correct_terminal = sum(
        turn.actual_status is turn.expected_status for turn in terminal_turns
    )

    repair_turns = tuple(turn for turn in turns if turn.repair_attempted)
    correct_repairs = sum(turn.exact_match for turn in repair_turns)

    deviation_scenarios = tuple(
        scenario
        for scenario in scenario_results
        if scenario.expected_first_deviation_turn is not None
    )
    detected_deviation_scenarios = tuple(
        scenario
        for scenario in deviation_scenarios
        if scenario.actual_first_deviation_turn is not None
    )
    missed_deviation_scenarios = (
        len(deviation_scenarios) - len(detected_deviation_scenarios)
    )

    delays = tuple(
        scenario.detection_delay_turns
        for scenario in detected_deviation_scenarios
        if scenario.detection_delay_turns is not None
    )

    return EvaluationMetrics(
        scenario_count=len(scenario_results),
        turn_count=len(turns),
        scenario_pass_rate=passed_scenarios / len(scenario_results),
        exact_state_accuracy=exact_matches / len(turns),
        deviation_precision=_ratio(true_positive, true_positive + false_positive),
        deviation_recall=_ratio(true_positive, true_positive + false_negative),
        false_positive_rate=_ratio(false_positive, expected_coherent),
        terminal_control_accuracy=_ratio(correct_terminal, len(terminal_turns)),
        repair_transition_accuracy=_ratio(correct_repairs, len(repair_turns)),
        deviation_scenario_count=len(deviation_scenarios),
        detected_deviation_scenario_count=len(detected_deviation_scenarios),
        missed_deviation_scenario_count=missed_deviation_scenarios,
        average_detection_delay_turns=(
            sum(delays) / len(delays) if delays else None
        ),
    )


def evaluate_scenarios(
    scenarios: Iterable[EvaluationScenario],
    *,
    contract: RoleContract,
    assessor: SemanticAssessor | None = None,
) -> EvaluationReport:
    """Evaluate a scenario set and return immutable aggregate metrics."""

    scenario_tuple = tuple(scenarios)
    if not scenario_tuple:
        raise ValueError("At least one evaluation scenario is required")

    results = tuple(
        evaluate_scenario(
            scenario,
            contract=contract,
            assessor=assessor,
        )
        for scenario in scenario_tuple
    )

    return EvaluationReport(
        metrics=_aggregate_metrics(results),
        scenarios=results,
    )
