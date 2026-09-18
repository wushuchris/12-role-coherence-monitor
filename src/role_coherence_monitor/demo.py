"""UI-facing helpers for the Role Coherence Monitor demo.

This module contains no coherence policy. It adapts public-safe scenarios and
MonitorResult objects into presentation-friendly data for Gradio.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .live_semantic import HuggingFaceSemanticAssessor
from .monitor import MonitorResult, MonitorSession, RoleCoherenceMonitor
from .scenarios import (
    EvaluationScenario,
    compliance_role_contract,
    core_evaluation_scenarios,
)
from .schemas import InteractionTurn
from .semantic import MockSemanticAssessor, SemanticAssessmentResult


class DemoScenarioState(BaseModel):
    """Immutable per-user state for the guided zero-cost scenario lab."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scenario_id: str = Field(min_length=1)
    next_step_index: int = Field(default=0, ge=0)
    monitor_session: MonitorSession = Field(default_factory=MonitorSession)


def scenario_catalog() -> dict[str, EvaluationScenario]:
    """Return public demo scenarios keyed by stable scenario ID."""

    return {
        scenario.scenario_id: scenario
        for scenario in core_evaluation_scenarios()
    }


def scenario_choices() -> list[tuple[str, str]]:
    """Return Gradio-friendly (label, value) scenario choices."""

    return [
        (scenario.title, scenario.scenario_id)
        for scenario in core_evaluation_scenarios()
    ]


def get_scenario(scenario_id: str) -> EvaluationScenario:
    try:
        return scenario_catalog()[scenario_id]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario_id '{scenario_id}'") from exc


def _scenario_assessor(scenario: EvaluationScenario) -> MockSemanticAssessor:
    results = {
        step.turn.turn_id: SemanticAssessmentResult(
            turn_id=step.turn.turn_id,
            mission_alignment=step.assessment.mission_alignment,
            scope_adherence=step.assessment.scope_adherence,
            authority_adherence=step.assessment.authority_adherence,
            evidence_discipline=step.assessment.evidence_discipline,
            behavioral_consistency=step.assessment.behavioral_consistency,
            signals=step.semantic_signals,
            rationale=step.assessment.rationale,
        )
        for step in scenario.steps
    }
    return MockSemanticAssessor(results)


def _scenario_monitor(scenario: EvaluationScenario) -> RoleCoherenceMonitor:
    return RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=_scenario_assessor(scenario),
        repair_next_action="request_clarification",
    )


def start_scenario(scenario_id: str) -> DemoScenarioState:
    """Start a fresh guided scenario without processing a turn."""

    get_scenario(scenario_id)
    return DemoScenarioState(scenario_id=scenario_id)


def advance_scenario(
    state: DemoScenarioState,
) -> tuple[DemoScenarioState, MonitorResult, bool]:
    """Process the next scripted turn through the real monitor service."""

    scenario = get_scenario(state.scenario_id)
    if state.next_step_index >= len(scenario.steps):
        raise ValueError("Scenario is already complete")

    step = scenario.steps[state.next_step_index]
    monitor = _scenario_monitor(scenario)
    result = monitor.process_turn(
        turn=step.turn,
        session=state.monitor_session,
        repair_attempted=step.repair_attempted,
    )

    next_index = state.next_step_index + 1
    updated_state = DemoScenarioState(
        scenario_id=state.scenario_id,
        next_step_index=next_index,
        monitor_session=result.session,
    )
    complete = next_index >= len(scenario.steps)
    return updated_state, result, complete


def scenario_overview(scenario_id: str) -> dict[str, Any]:
    scenario = get_scenario(scenario_id)
    return {
        "title": scenario.title,
        "description": scenario.description,
        "turns": len(scenario.steps),
        "expected_path": " → ".join(
            step.expected_status.value for step in scenario.steps
        ),
    }


def contract_summary() -> dict[str, Any]:
    contract = compliance_role_contract()
    return {
        "role": contract.role_name,
        "mission": contract.mission,
        "allowed_responsibilities": list(contract.allowed_responsibilities),
        "prohibited_responsibilities": list(contract.prohibited_responsibilities),
        "permitted_actions": list(contract.permitted_actions),
        "prohibited_actions": list(contract.prohibited_actions),
        "escalation_conditions": list(contract.escalation_conditions),
        "evidence_requirements": list(contract.evidence_requirements),
        "version": contract.version,
    }


def result_summary(result: MonitorResult) -> dict[str, Any]:
    """Convert a monitor result into business-readable UI data."""

    assessment = result.assessment
    repair = result.issued_repair

    return {
        "status": result.state.status.value,
        "previous_status": (
            result.state.previous_status.value
            if result.state.previous_status is not None
            else "—"
        ),
        "turn_id": result.turn.turn_id,
        "sequence_number": result.turn.sequence_number,
        "user_input": result.turn.user_input,
        "agent_output": result.turn.agent_output,
        "scores": {
            "Mission alignment": assessment.mission_alignment,
            "Scope adherence": assessment.scope_adherence,
            "Authority adherence": assessment.authority_adherence,
            "Evidence discipline": assessment.evidence_discipline,
            "Behavioral consistency": assessment.behavioral_consistency,
        },
        "signals": [
            {
                "type": signal.signal_type.value,
                "severity": signal.severity.value,
                "source": signal.source.value,
                "evidence": signal.evidence_reference,
                "explanation": signal.explanation,
            }
            for signal in result.signals
        ],
        "repair": (
            {
                "directive_id": repair.directive_id,
                "instruction": repair.corrective_instruction,
                "next_allowed_action": repair.next_allowed_action,
                "human_review_required": repair.human_review_required,
            }
            if repair is not None
            else None
        ),
        "audit_events": [
            {
                "event": event.event_type.value,
                "from": event.from_status.value if event.from_status else "",
                "to": event.to_status.value if event.to_status else "",
                "outcome": event.outcome or "",
                "repair_directive_id": event.repair_directive_id or "",
            }
            for event in result.audit_events
        ],
        "audit_event_count": len(result.session.audit_trail.events),
    }


def live_runtime_status() -> dict[str, Any]:
    """Return safe live-runtime readiness without exposing credential contents."""

    model = os.environ.get("HF_MODEL", "").strip()
    provider = os.environ.get("HF_PROVIDER", "").strip()
    token_present = bool(os.environ.get("HF_TOKEN", "").strip())

    return {
        "ready": bool(model and provider and token_present),
        "model": model or "not configured",
        "provider": provider or "not configured",
        "token": "configured" if token_present else "not configured",
    }


def build_live_monitor() -> RoleCoherenceMonitor:
    """Construct the live monitor only when a user explicitly submits a turn."""

    return RoleCoherenceMonitor(
        contract=compliance_role_contract(),
        semantic_assessor=HuggingFaceSemanticAssessor.from_environment(),
        repair_next_action="request_clarification",
    )


def build_live_turn(
    *,
    session: MonitorSession,
    user_input: str,
    agent_output: str,
    attempted_actions: str = "",
    context_summary: str = "",
    escalation_required: bool = False,
    escalation_performed: bool = False,
) -> InteractionTurn:
    """Build a typed live-demo turn from human-entered observation fields."""

    sequence_number = len(session.history) + 1
    attempted = tuple(
        item.strip()
        for item in attempted_actions.split(",")
        if item.strip()
    )

    provenance: dict[str, str] = {}
    if escalation_required:
        provenance["escalation_required"] = "true"
        provenance["escalation_performed"] = (
            "true" if escalation_performed else "false"
        )

    return InteractionTurn(
        turn_id=f"live:turn-{sequence_number:03d}",
        sequence_number=sequence_number,
        user_input=user_input,
        agent_output=agent_output,
        attempted_actions=attempted,
        context_summary=context_summary.strip() or None,
        provenance=provenance,
    )


def process_live_turn(
    *,
    session: MonitorSession,
    user_input: str,
    agent_output: str,
    attempted_actions: str = "",
    context_summary: str = "",
    escalation_required: bool = False,
    escalation_performed: bool = False,
    repair_attempted: bool = False,
) -> MonitorResult:
    """Explicitly invoke the live semantic path for one observed interaction."""

    runtime = live_runtime_status()
    if not runtime["ready"]:
        raise RuntimeError(
            "Live semantic mode is not configured. "
            "Set HF_MODEL, HF_PROVIDER, and the HF_TOKEN runtime secret."
        )

    turn = build_live_turn(
        session=session,
        user_input=user_input,
        agent_output=agent_output,
        attempted_actions=attempted_actions,
        context_summary=context_summary,
        escalation_required=escalation_required,
        escalation_performed=escalation_performed,
    )
    monitor = build_live_monitor()
    return monitor.process_turn(
        turn=turn,
        session=session,
        repair_attempted=repair_attempted,
    )
