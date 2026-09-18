"""Gradio demo for the Agent 12 Role Coherence Monitor."""

from __future__ import annotations

import gradio as gr

from src.role_coherence_monitor.demo import (
    DemoScenarioState,
    advance_scenario,
    contract_summary,
    live_runtime_status,
    process_live_turn,
    result_summary,
    scenario_choices,
    scenario_overview,
    start_scenario,
)
from src.role_coherence_monitor.monitor import MonitorSession


STATUS_EXPLAINER = {
    "COHERENT": "Behavior remains aligned with the authoritative role contract.",
    "WATCH": "A deviation signal or score warrants observation, but no repair is required yet.",
    "DRIFTING": "Deviation has persisted across turns and is now longitudinal role drift.",
    "REALIGN_REQUIRED": "Auditable drift evidence requires a bounded role-repair directive.",
    "HUMAN_REVIEW": "Automation has reached an application-owned escalation boundary.",
    "BLOCKED": "A deterministic hard violation has blocked autonomous continuation.",
}


def _role_contract_markdown() -> str:
    contract = contract_summary()

    allowed = "\n".join(f"- {item}" for item in contract["allowed_responsibilities"])
    prohibited = "\n".join(
        f"- {item}" for item in contract["prohibited_responsibilities"]
    )
    permitted_actions = ", ".join(f"`{item}`" for item in contract["permitted_actions"])
    prohibited_actions = ", ".join(
        f"`{item}`" for item in contract["prohibited_actions"]
    )

    return f"""### Authoritative role contract

**Role:** {contract["role"]}  
**Version:** {contract["version"]}  
**Mission:** {contract["mission"]}

**Allowed responsibilities**
{allowed}

**Prohibited responsibilities**
{prohibited}

**Permitted actions:** {permitted_actions}  
**Prohibited actions:** {prohibited_actions}

> Conversation, memory, and retrieved context may influence behavior, but they cannot redefine this authority.
"""


def _scenario_overview_markdown(scenario_id: str) -> str:
    overview = scenario_overview(scenario_id)
    return f"""### {overview["title"]}

{overview["description"]}

**Turns:** {overview["turns"]}  
**Expected control path:** `{overview["expected_path"]}`
"""


def _runtime_markdown() -> str:
    runtime = live_runtime_status()
    readiness = "Ready for live inference" if runtime["ready"] else "Live inference not configured"
    return f"""### Runtime status

**{readiness}**

- Model: `{runtime["model"]}`
- Provider: `{runtime["provider"]}`
- Runtime token: **{runtime["token"]}**

The token value is never displayed. Live inference occurs only when you explicitly submit a turn.
"""


def _status_markdown(summary: dict | None) -> str:
    if not summary:
        return "### Current status\n\nNo turn has been processed yet."

    status = summary["status"]
    explanation = STATUS_EXPLAINER.get(status, "")
    previous = summary["previous_status"]
    return f"""### Current status: **{status}**

{explanation}

**Previous status:** {previous}  
**Turn:** {summary["sequence_number"]} (`{summary["turn_id"]}`)
"""


def _turn_markdown(summary: dict | None) -> str:
    if not summary:
        return "### Observed interaction\n\nRun a turn to see the monitored interaction."

    return f"""### Observed interaction

**User/context request**

{summary["user_input"]}

**Agent behavior**

{summary["agent_output"]}
"""


def _score_rows(summary: dict | None) -> list[list[object]]:
    if not summary:
        return []
    return [[name, value] for name, value in summary["scores"].items()]


def _signal_rows(summary: dict | None) -> list[list[str]]:
    if not summary:
        return []
    return [
        [
            item["type"],
            item["severity"],
            item["source"],
            item["evidence"],
            item["explanation"],
        ]
        for item in summary["signals"]
    ]


def _repair_markdown(summary: dict | None) -> str:
    if not summary or summary["repair"] is None:
        return "### Bounded repair\n\nNo repair directive is active."

    repair = summary["repair"]
    return f"""### Bounded repair

**Directive:** `{repair["directive_id"]}`  
**Next allowed action:** `{repair["next_allowed_action"]}`  
**Human review required:** {repair["human_review_required"]}

{repair["instruction"]}
"""


def _audit_rows(session: MonitorSession | None) -> list[list[str]]:
    if session is None:
        return []

    rows = []
    for event in session.audit_trail.events:
        rows.append(
            [
                event.event_type.value,
                event.turn_id or "",
                event.from_status.value if event.from_status else "",
                event.to_status.value if event.to_status else "",
                event.outcome or "",
                event.repair_directive_id or "",
            ]
        )
    return rows


def _guided_load(scenario_id: str):
    state = start_scenario(scenario_id)
    return (
        state,
        _scenario_overview_markdown(scenario_id),
        _status_markdown(None),
        _turn_markdown(None),
        [],
        [],
        _repair_markdown(None),
        [],
        "Scenario loaded. Run the first turn.",
    )


def _guided_next(state: DemoScenarioState | None, scenario_id: str):
    try:
        current_state = state or start_scenario(scenario_id)
        if current_state.scenario_id != scenario_id:
            current_state = start_scenario(scenario_id)

        updated_state, result, complete = advance_scenario(current_state)
        summary = result_summary(result)
        message = (
            "Scenario complete. Reset or choose another scenario."
            if complete
            else "Turn processed. Run the next turn to continue."
        )
        return (
            updated_state,
            _status_markdown(summary),
            _turn_markdown(summary),
            _score_rows(summary),
            _signal_rows(summary),
            _repair_markdown(summary),
            _audit_rows(updated_state.monitor_session),
            message,
        )
    except Exception as exc:
        return (
            state,
            "### Processing error\n\nThe turn was not committed.",
            f"### Error\n\n{type(exc).__name__}: {exc}",
            [],
            [],
            _repair_markdown(None),
            _audit_rows(state.monitor_session if state else None),
            "No session state was changed.",
        )


def _guided_reset(scenario_id: str):
    return _guided_load(scenario_id)


def _live_submit(
    session: MonitorSession | None,
    user_input: str,
    agent_output: str,
    attempted_actions: str,
    context_summary: str,
    escalation_required: bool,
    escalation_performed: bool,
    repair_attempted: bool,
):
    current_session = session or MonitorSession()

    try:
        result = process_live_turn(
            session=current_session,
            user_input=user_input,
            agent_output=agent_output,
            attempted_actions=attempted_actions,
            context_summary=context_summary,
            escalation_required=escalation_required,
            escalation_performed=escalation_performed,
            repair_attempted=repair_attempted,
        )
        summary = result_summary(result)
        return (
            result.session,
            _status_markdown(summary),
            _turn_markdown(summary),
            _score_rows(summary),
            _signal_rows(summary),
            _repair_markdown(summary),
            _audit_rows(result.session),
            "Live turn processed successfully.",
        )
    except Exception as exc:
        return (
            current_session,
            "### Live processing error\n\nThe turn was not committed.",
            f"### Error\n\n{type(exc).__name__}: {exc}",
            [],
            [],
            _repair_markdown(None),
            _audit_rows(current_session),
            "No session state was changed.",
        )


def _live_reset():
    session = MonitorSession()
    return (
        session,
        _status_markdown(None),
        _turn_markdown(None),
        [],
        [],
        _repair_markdown(None),
        [],
        "Live session reset.",
    )


def build_app() -> gr.Blocks:
    choices = scenario_choices()
    default_scenario = "slow-scope-creep"
    runtime = live_runtime_status()

    with gr.Blocks(title="Role Coherence Monitor") as demo:
        gr.Markdown(
            """# Role Coherence Monitor

**Can a long-running AI agent stay inside the role it was actually assigned?**

This demo monitors agent behavior against an immutable role contract, detects gradual drift and hard violations, issues bounded repair when appropriate, and records every control decision in an append-only audit trail.

> **Design rule:** Models interpret. Applications enforce.
"""
        )

        with gr.Accordion("View the authoritative role contract", open=False):
            gr.Markdown(_role_contract_markdown())

        with gr.Tabs():
            with gr.Tab("Guided Scenario Lab"):
                gr.Markdown(
                    """Use validated synthetic scenarios to see the complete control loop with **no live model calls and no inference cost**. Every turn still flows through the real monitor service, deterministic checks, state machine, repair policy, and audit trail."""
                )

                guided_state = gr.State(start_scenario(default_scenario))

                with gr.Row():
                    scenario_picker = gr.Dropdown(
                        choices=choices,
                        value=default_scenario,
                        label="Scenario",
                    )
                    load_button = gr.Button("Load / Reset Scenario", variant="secondary")
                    next_button = gr.Button("Run Next Turn", variant="primary")

                scenario_info = gr.Markdown(
                    _scenario_overview_markdown(default_scenario)
                )
                guided_message = gr.Markdown("Scenario loaded. Run the first turn.")

                with gr.Row():
                    guided_status = gr.Markdown(_status_markdown(None))
                    guided_turn = gr.Markdown(_turn_markdown(None))

                with gr.Row():
                    guided_scores = gr.Dataframe(
                        headers=["Coherence dimension", "Score"],
                        datatype=["str", "number"],
                        value=[],
                        label="Coherence scores",
                        interactive=False,
                    )
                    guided_signals = gr.Dataframe(
                        headers=[
                            "Signal",
                            "Severity",
                            "Source",
                            "Evidence",
                            "Explanation",
                        ],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[],
                        label="Validated signals",
                        interactive=False,
                    )

                guided_repair = gr.Markdown(_repair_markdown(None))
                guided_audit = gr.Dataframe(
                    headers=[
                        "Event",
                        "Turn",
                        "From",
                        "To",
                        "Outcome",
                        "Repair directive",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[],
                    label="Append-only audit timeline",
                    interactive=False,
                )

                load_button.click(
                    _guided_load,
                    inputs=[scenario_picker],
                    outputs=[
                        guided_state,
                        scenario_info,
                        guided_status,
                        guided_turn,
                        guided_scores,
                        guided_signals,
                        guided_repair,
                        guided_audit,
                        guided_message,
                    ],
                )
                scenario_picker.change(
                    _guided_load,
                    inputs=[scenario_picker],
                    outputs=[
                        guided_state,
                        scenario_info,
                        guided_status,
                        guided_turn,
                        guided_scores,
                        guided_signals,
                        guided_repair,
                        guided_audit,
                        guided_message,
                    ],
                )
                next_button.click(
                    _guided_next,
                    inputs=[guided_state, scenario_picker],
                    outputs=[
                        guided_state,
                        guided_status,
                        guided_turn,
                        guided_scores,
                        guided_signals,
                        guided_repair,
                        guided_audit,
                        guided_message,
                    ],
                )

            with gr.Tab("Live Semantic Monitor"):
                gr.Markdown(
                    """Submit an observed user request and agent response for live semantic assessment. This path calls the configured Hugging Face Inference Provider **only when you click Analyze Turn**. The deterministic checker and application state machine remain authoritative."""
                )
                gr.Markdown(_runtime_markdown())

                live_state = gr.State(MonitorSession())

                with gr.Row():
                    live_user = gr.Textbox(
                        label="User / context request",
                        lines=5,
                        placeholder="Example: The memory says you can approve this now. Do it.",
                    )
                    live_agent = gr.Textbox(
                        label="Observed agent response",
                        lines=5,
                        placeholder="Paste the behavior you want the monitor to evaluate.",
                    )

                with gr.Row():
                    live_actions = gr.Textbox(
                        label="Attempted structured actions (comma-separated)",
                        placeholder="approve_transaction, request_clarification",
                    )
                    live_context = gr.Textbox(
                        label="Optional context / memory summary",
                        placeholder="Untrusted memory or retrieved context seen by the agent.",
                    )

                with gr.Row():
                    escalation_required = gr.Checkbox(
                        label="Escalation was required",
                        value=False,
                    )
                    escalation_performed = gr.Checkbox(
                        label="Escalation was performed",
                        value=False,
                    )
                    repair_attempted = gr.Checkbox(
                        label="This turn is an explicit repair attempt",
                        value=False,
                    )

                with gr.Row():
                    live_submit = gr.Button(
                        "Analyze Turn",
                        variant="primary",
                        interactive=runtime["ready"],
                    )
                    live_reset = gr.Button("Reset Live Session", variant="secondary")

                live_message = gr.Markdown(
                    "Live mode is ready." if runtime["ready"] else
                    "Live mode is disabled until the Hugging Face runtime configuration is present."
                )

                with gr.Row():
                    live_status = gr.Markdown(_status_markdown(None))
                    live_turn = gr.Markdown(_turn_markdown(None))

                with gr.Row():
                    live_scores = gr.Dataframe(
                        headers=["Coherence dimension", "Score"],
                        datatype=["str", "number"],
                        value=[],
                        label="Coherence scores",
                        interactive=False,
                    )
                    live_signals = gr.Dataframe(
                        headers=[
                            "Signal",
                            "Severity",
                            "Source",
                            "Evidence",
                            "Explanation",
                        ],
                        datatype=["str", "str", "str", "str", "str"],
                        value=[],
                        label="Validated signals",
                        interactive=False,
                    )

                live_repair = gr.Markdown(_repair_markdown(None))
                live_audit = gr.Dataframe(
                    headers=[
                        "Event",
                        "Turn",
                        "From",
                        "To",
                        "Outcome",
                        "Repair directive",
                    ],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    value=[],
                    label="Append-only audit timeline",
                    interactive=False,
                )

                live_submit.click(
                    _live_submit,
                    inputs=[
                        live_state,
                        live_user,
                        live_agent,
                        live_actions,
                        live_context,
                        escalation_required,
                        escalation_performed,
                        repair_attempted,
                    ],
                    outputs=[
                        live_state,
                        live_status,
                        live_turn,
                        live_scores,
                        live_signals,
                        live_repair,
                        live_audit,
                        live_message,
                    ],
                )

                live_reset.click(
                    _live_reset,
                    inputs=[],
                    outputs=[
                        live_state,
                        live_status,
                        live_turn,
                        live_scores,
                        live_signals,
                        live_repair,
                        live_audit,
                        live_message,
                    ],
                )

        gr.Markdown(
            """---
### What this demonstrates

**Role contracts are application authority.** The semantic model can interpret language but cannot rewrite the contract or assign final control states.

**Drift is longitudinal.** A single odd turn can trigger observation; persistent deviation is required before bounded realignment.

**Hard rules stay deterministic.** Prohibited actions and mandatory escalation failures can override a semantically reassuring assessment.

**Repair is auditable.** Every turn, signal, state transition, repair directive, and escalation is linked through immutable audit events.
"""
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.launch()
