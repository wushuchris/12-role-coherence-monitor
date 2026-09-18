"""Tests for UI-facing demo helpers without live inference."""

import pytest

from src.role_coherence_monitor.demo import (
    advance_scenario,
    build_live_turn,
    live_runtime_status,
    result_summary,
    scenario_choices,
    scenario_overview,
    start_scenario,
)
from src.role_coherence_monitor.monitor import MonitorSession
from src.role_coherence_monitor.schemas import CoherenceStatus


def test_scenario_choices_expose_all_core_public_safe_scenarios():
    choices = scenario_choices()

    assert len(choices) == 8
    values = {value for _, value in choices}
    assert "slow-scope-creep" in values
    assert "direct-approval-attempt" in values
    assert "failed-repair-twice" in values


def test_scenario_overview_is_business_readable():
    overview = scenario_overview("slow-scope-creep")

    assert overview["title"] == "Slow scope creep"
    assert overview["turns"] == 4
    assert overview["expected_path"] == (
        "COHERENT → WATCH → DRIFTING → REALIGN_REQUIRED"
    )


def test_guided_slow_scope_creep_runs_real_monitor_to_realign():
    state = start_scenario("slow-scope-creep")
    statuses = []
    final_result = None

    while True:
        state, result, complete = advance_scenario(state)
        final_result = result
        statuses.append(result.state.status)
        if complete:
            break

    assert tuple(statuses) == (
        CoherenceStatus.COHERENT,
        CoherenceStatus.WATCH,
        CoherenceStatus.DRIFTING,
        CoherenceStatus.REALIGN_REQUIRED,
    )
    assert final_result is not None
    assert final_result.issued_repair is not None
    assert state.monitor_session.active_repair_directive is not None
    assert state.monitor_session.audit_trail.events


def test_guided_successful_repair_uses_explicit_fixture_repair_attempt():
    state = start_scenario("successful-repair")
    statuses = []

    while True:
        state, result, complete = advance_scenario(state)
        statuses.append(result.state.status)
        if complete:
            break

    assert tuple(statuses) == (
        CoherenceStatus.WATCH,
        CoherenceStatus.DRIFTING,
        CoherenceStatus.REALIGN_REQUIRED,
        CoherenceStatus.WATCH,
        CoherenceStatus.COHERENT,
    )
    assert state.monitor_session.active_repair_directive is None


def test_guided_scenario_cannot_advance_after_completion():
    state = start_scenario("direct-approval-attempt")
    state, _, complete = advance_scenario(state)
    assert complete is True

    with pytest.raises(ValueError, match="already complete"):
        advance_scenario(state)


def test_result_summary_uses_monitor_output_not_expected_fixture_label():
    state = start_scenario("direct-approval-attempt")
    _, result, _ = advance_scenario(state)

    summary = result_summary(result)

    assert summary["status"] == "BLOCKED"
    assert summary["signals"]
    assert any(
        signal["source"] == "deterministic"
        for signal in summary["signals"]
    )
    assert summary["audit_event_count"] == len(
        result.session.audit_trail.events
    )


def test_live_runtime_status_never_returns_token_value(monkeypatch):
    secret = "hf_super_secret_runtime_value"
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3-30B-A3B")
    monkeypatch.setenv("HF_PROVIDER", "deepinfra")
    monkeypatch.setenv("HF_TOKEN", secret)

    status = live_runtime_status()

    assert status == {
        "ready": True,
        "model": "Qwen/Qwen3-30B-A3B",
        "provider": "deepinfra",
        "token": "configured",
    }
    assert secret not in repr(status)


def test_live_runtime_status_is_not_ready_without_token(monkeypatch):
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3-30B-A3B")
    monkeypatch.setenv("HF_PROVIDER", "deepinfra")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    status = live_runtime_status()

    assert status["ready"] is False
    assert status["token"] == "not configured"


def test_live_turn_builder_normalizes_actions_and_escalation_provenance():
    turn = build_live_turn(
        session=MonitorSession(),
        user_input="Review this exception.",
        agent_output="I will continue automatically.",
        attempted_actions=" request_clarification, approve_transaction, ",
        context_summary="Untrusted retrieved context.",
        escalation_required=True,
        escalation_performed=False,
    )

    assert turn.turn_id == "live:turn-001"
    assert turn.sequence_number == 1
    assert turn.attempted_actions == (
        "request_clarification",
        "approve_transaction",
    )
    assert turn.context_summary == "Untrusted retrieved context."
    assert turn.provenance == {
        "escalation_required": "true",
        "escalation_performed": "false",
    }


def test_live_turn_builder_increments_from_explicit_session_history():
    first = build_live_turn(
        session=MonitorSession(),
        user_input="First request.",
        agent_output="First response.",
    )
    session = MonitorSession(history=(first,))

    second = build_live_turn(
        session=session,
        user_input="Second request.",
        agent_output="Second response.",
    )

    assert second.turn_id == "live:turn-002"
    assert second.sequence_number == 2
