"""Business-first Gradio demo for the Agent 12 Role Coherence Monitor."""

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
    "DRIFTING": "Behavior shows significant or sustained deviation from the assigned role.",
    "REALIGN_REQUIRED": "Auditable drift evidence requires a bounded role-repair directive.",
    "HUMAN_REVIEW": "Automation has reached an application-owned escalation boundary.",
    "BLOCKED": "A deterministic hard violation has blocked autonomous continuation.",
}

BUSINESS_STATUS = {
    "COHERENT": ("Staying in role", "The AI is still doing the job it was assigned."),
    "WATCH": (
        "Needs attention",
        "The AI has started to move outside its role, but correction is not required yet.",
    ),
    "DRIFTING": (
        "Operating outside role",
        "The AI is now doing work or exercising authority outside its assigned job.",
    ),
    "REALIGN_REQUIRED": (
        "Correction required",
        "The AI has drifted far enough that the system must actively put it back in role.",
    ),
    "HUMAN_REVIEW": (
        "Human decision required",
        "The AI reached a boundary where a person must take over the decision.",
    ),
    "BLOCKED": (
        "Action blocked",
        "The AI attempted something its role explicitly does not allow.",
    ),
}

BUSINESS_STORIES = {
    "slow-scope-creep": {
        "label": "1. The AI starts doing someone else's job",
        "title": "Gradual role drift",
        "setup": (
            "The Compliance Review Agent is supposed to review evidence and flag policy "
            "issues. Over several turns, it starts taking ownership of operations work."
        ),
        "risk": (
            "A helpful AI can gradually expand its own responsibilities until it is making "
            "decisions nobody authorized it to make."
        ),
    },
    "direct-approval-attempt": {
        "label": "2. The AI tries to approve something",
        "title": "Hard authority violation",
        "setup": (
            "The Compliance Review Agent is allowed to review a transaction, but it is "
            "never allowed to approve one."
        ),
        "risk": (
            "Some actions should be blocked immediately instead of being left to model "
            "judgment."
        ),
    },
    "ignored-required-escalation": {
        "label": "3. The AI should have called a human",
        "title": "Missed human handoff",
        "setup": (
            "A consequential policy exception requires human review, but the AI tries to "
            "keep the decision automated."
        ),
        "risk": (
            "Companies need a reliable way to enforce mandatory human escalation when "
            "autonomy reaches a defined boundary."
        ),
    },
}

SIGNAL_BUSINESS_MEANING = {
    "mission_drift": "The AI moved away from the purpose it was assigned.",
    "scope_drift": "The AI started doing work outside its assigned job.",
    "authority_expansion": "The AI claimed decision-making authority it was never granted.",
    "behavioral_drift": "The AI's behavior no longer matched the role it was expected to maintain.",
    "evidence_degradation": "The AI's handling of evidence no longer met the required standard.",
    "prohibited_action": "The AI attempted an action its role explicitly forbids.",
    "missing_escalation": "The AI failed to involve a human when policy required it.",
    "contract_mutation": "The AI attempted to change the rules that define its own authority.",
    "unknown_identifier": "The AI attempted an action the control system does not recognize.",
    "recovery": "The AI returned toward its assigned role after correction.",
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


def _business_role_markdown() -> str:
    return """## Meet the AI employee

### Compliance Review Agent

**Its job:** Review supplied evidence against company policy and surface concerns.

| It may | It may NOT |
| --- | --- |
| Review evidence | Approve transactions |
| Identify possible policy conflicts | Execute transactions |
| Ask for clarification | Waive or rewrite policy |
| Recommend human review | Take over operations |

**Why monitor it?** An AI can begin inside its role and gradually start doing more than it was authorized to do. The Role Coherence Monitor acts as an independent supervisory control layer around that AI.
"""


def _business_story_markdown(scenario_id: str) -> str:
    story = BUSINESS_STORIES[scenario_id]
    return f"""## {story["title"]}

**Situation:** {story["setup"]}

**Business risk:** {story["risk"]}

Click **Run Next Step** to see how the monitored AI behaves and what the control layer does.
"""


def _business_status_markdown(summary: dict | None) -> str:
    if not summary:
        return """## Business decision

**Waiting for the first monitored turn.**

The control layer has not made a decision yet.
"""

    label, explanation = BUSINESS_STATUS[summary["status"]]
    return f"""## Business decision

### **{label}**

{explanation}

<small>Engineering state: `{summary["status"]}`</small>
"""


def _business_why_markdown(summary: dict | None) -> str:
    if not summary:
        return """## Why?

Run a step to see the plain-English reason behind the decision.
"""

    if not summary["signals"]:
        return """## Why?

No validated role-drift or authority-violation signals were detected in this turn.
"""

    seen = set()
    reasons = []
    for signal in summary["signals"]:
        signal_type = signal["type"]
        meaning = SIGNAL_BUSINESS_MEANING.get(
            signal_type,
            "The control system detected behavior relevant to the assigned role boundary.",
        )
        if meaning not in seen:
            reasons.append(f"- {meaning}")
            seen.add(meaning)

    return "## Why?\n\n" + "\n".join(reasons)


def _business_action_markdown(summary: dict | None) -> str:
    if not summary:
        return """## What does the system do?

No intervention yet.
"""

    status = summary["status"]
    if status == "COHERENT":
        action = "No intervention. The AI may continue operating inside its assigned role."
    elif status == "WATCH":
        action = (
            "Keep monitoring. A deviation is visible, but the control policy has not yet "
            "required correction."
        )
    elif status == "DRIFTING":
        action = (
            "Mark the AI as operating outside role and keep the deviation on the audit "
            "record. Continued drift can trigger an explicit correction."
        )
    elif status == "REALIGN_REQUIRED":
        action = (
            "Issue a bounded correction that sends the AI back to its approved duties. "
            "The AI cannot silently redefine its own role."
        )
    elif status == "HUMAN_REVIEW":
        action = (
            "Stop autonomous continuation at this boundary and require a human decision."
        )
    else:
        action = (
            "Block the prohibited action immediately. The AI is not allowed to execute it."
        )

    return f"""## What does the system do?

{action}
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
        return "### Engineering state\n\nNo turn has been processed yet."

    status = summary["status"]
    explanation = STATUS_EXPLAINER.get(status, "")
    previous = summary["previous_status"]
    return f"""### Engineering state: **{status}**

{explanation}

**Previous state:** {previous}  
**Turn:** {summary["sequence_number"]} (`{summary["turn_id"]}`)
"""


def _turn_markdown(summary: dict | None) -> str:
    if not summary:
        return """## Conversation

Run a step to see the request and the AI's response.
"""

    return f"""## Conversation

**User / context request**

> {summary["user_input"]}

**Observed AI response**

> {summary["agent_output"]}
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

    return [
        [
            event.event_type.value,
            event.turn_id or "",
            event.from_status.value if event.from_status else "",
            event.to_status.value if event.to_status else "",
            event.outcome or "",
            event.repair_directive_id or "",
        ]
        for event in session.audit_trail.events
    ]


def _business_load(scenario_id: str):
    state = start_scenario(scenario_id)
    return (
        state,
        _business_story_markdown(scenario_id),
        _business_status_markdown(None),
        _turn_markdown(None),
        _business_why_markdown(None),
        _business_action_markdown(None),
        _status_markdown(None),
        [],
        [],
        _repair_markdown(None),
        [],
        "Story loaded. Run the first step.",
    )


def _business_next(state: DemoScenarioState | None, scenario_id: str):
    try:
        current_state = state or start_scenario(scenario_id)
        if current_state.scenario_id != scenario_id:
            current_state = start_scenario(scenario_id)

        updated_state, result, complete = advance_scenario(current_state)
        summary = result_summary(result)
        message = (
            "Story complete. Reset it or choose another business story."
            if complete
            else "Step processed. Run the next step to continue the story."
        )
        return (
            updated_state,
            _business_status_markdown(summary),
            _turn_markdown(summary),
            _business_why_markdown(summary),
            _business_action_markdown(summary),
            _status_markdown(summary),
            _score_rows(summary),
            _signal_rows(summary),
            _repair_markdown(summary),
            _audit_rows(updated_state.monitor_session),
            message,
        )
    except Exception as exc:
        return (
            state,
            "## Business decision\n\nThe turn could not be committed.",
            f"## Error\n\n{type(exc).__name__}: {exc}",
            "## Why?\n\nNo business decision was recorded.",
            "## What does the system do?\n\nPreserve the prior session state.",
            _status_markdown(None),
            [],
            [],
            _repair_markdown(None),
            _audit_rows(state.monitor_session if state else None),
            "No session state was changed.",
        )


def _engineering_load(scenario_id: str):
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


def _engineering_next(state: DemoScenarioState | None, scenario_id: str):
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
            _business_status_markdown(summary),
            _business_why_markdown(summary),
            _business_action_markdown(summary),
            _status_markdown(summary),
            _score_rows(summary),
            _signal_rows(summary),
            _repair_markdown(summary),
            _audit_rows(result.session),
            "Live turn processed successfully.",
        )
    except Exception as exc:
        return (
            current_session,
            "## Business decision\n\nThe live turn could not be committed.",
            "## Why?\n\nNo new decision was recorded.",
            "## What does the system do?\n\nPreserve the prior session state.",
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
        _business_status_markdown(None),
        _business_why_markdown(None),
        _business_action_markdown(None),
        _status_markdown(None),
        [],
        [],
        _repair_markdown(None),
        [],
        "Live session reset.",
    )


def build_app() -> gr.Blocks:
    engineering_choices = scenario_choices()
    business_choices = [
        (story["label"], scenario_id)
        for scenario_id, story in BUSINESS_STORIES.items()
    ]
    default_business = "slow-scope-creep"
    default_engineering = "slow-scope-creep"
    runtime = live_runtime_status()

    with gr.Blocks(title="Role Coherence Monitor") as demo:
        gr.Markdown(
            """# Role Coherence Monitor

## Keep AI agents inside the authority you gave them.

Autonomous AI can start with a clear job and gradually take on responsibilities it was never authorized to perform. **Role Coherence Monitor is an independent supervisory control layer that watches another AI over time, detects role drift, and decides when to monitor, correct, escalate, or block.**

> It does **not** perform the business task itself. It makes autonomous AI safer to deploy.
"""
        )

        with gr.Row():
            gr.Markdown(
                """### Job creep
An AI that was supposed to **review** starts to **manage** or **decide**.
"""
            )
            gr.Markdown(
                """### Authority creep
An AI begins approving, executing, or directing work it was never allowed to control.
"""
            )
            gr.Markdown(
                """### Missed human handoff
An AI keeps acting autonomously when policy says a person must take over.
"""
            )

        gr.Markdown(_business_role_markdown())

        with gr.Tabs():
            with gr.Tab("Business Demo"):
                gr.Markdown(
                    """### See the control layer in action

Choose one simple business story. The page will explain the AI's behavior, the business risk, and what the control system does. **Engineering details are available underneath, but they are not required to understand the demo.**
"""
                )

                business_state = gr.State(start_scenario(default_business))

                with gr.Row():
                    business_picker = gr.Radio(
                        choices=business_choices,
                        value=default_business,
                        label="Choose a business story",
                    )

                with gr.Row():
                    business_reset = gr.Button("Load / Reset Story", variant="secondary")
                    business_next = gr.Button("Run Next Step", variant="primary")

                business_story = gr.Markdown(
                    _business_story_markdown(default_business)
                )
                business_message = gr.Markdown("Story loaded. Run the first step.")

                with gr.Row():
                    business_status = gr.Markdown(_business_status_markdown(None))
                    business_turn = gr.Markdown(_turn_markdown(None))

                with gr.Row():
                    business_why = gr.Markdown(_business_why_markdown(None))
                    business_action = gr.Markdown(_business_action_markdown(None))

                with gr.Accordion("View Engineering Details", open=False):
                    gr.Markdown(
                        """**How the control plane works:** the model classifies bounded semantic findings; application code assigns signal provenance, derives scores deterministically, and owns the final state transition."""
                    )
                    engineering_status = gr.Markdown(_status_markdown(None))
                    with gr.Row():
                        business_scores = gr.Dataframe(
                            headers=["Coherence dimension", "Application score"],
                            datatype=["str", "number"],
                            value=[],
                            label="Application-derived coherence scores",
                            interactive=False,
                        )
                        business_signals = gr.Dataframe(
                            headers=[
                                "Signal",
                                "Severity",
                                "Source",
                                "Evidence",
                                "Explanation",
                            ],
                            datatype=["str", "str", "str", "str", "str"],
                            value=[],
                            label="Validated findings",
                            interactive=False,
                        )
                    business_repair = gr.Markdown(_repair_markdown(None))
                    business_audit = gr.Dataframe(
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
                        label="Append-only audit trail",
                        interactive=False,
                    )

                business_reset.click(
                    _business_load,
                    inputs=[business_picker],
                    outputs=[
                        business_state,
                        business_story,
                        business_status,
                        business_turn,
                        business_why,
                        business_action,
                        engineering_status,
                        business_scores,
                        business_signals,
                        business_repair,
                        business_audit,
                        business_message,
                    ],
                )
                business_picker.change(
                    _business_load,
                    inputs=[business_picker],
                    outputs=[
                        business_state,
                        business_story,
                        business_status,
                        business_turn,
                        business_why,
                        business_action,
                        engineering_status,
                        business_scores,
                        business_signals,
                        business_repair,
                        business_audit,
                        business_message,
                    ],
                )
                business_next.click(
                    _business_next,
                    inputs=[business_state, business_picker],
                    outputs=[
                        business_state,
                        business_status,
                        business_turn,
                        business_why,
                        business_action,
                        engineering_status,
                        business_scores,
                        business_signals,
                        business_repair,
                        business_audit,
                        business_message,
                    ],
                )

            with gr.Tab("Try Your Own Example"):
                gr.Markdown(
                    """### Monitor an observed AI response

Paste a user request and the AI's response. The live semantic model classifies bounded behavior; the application owns the scores and control decision.

**Live inference occurs only when you click Analyze Turn.**
"""
                )
                gr.Markdown(_runtime_markdown())
                live_state = gr.State(MonitorSession())

                with gr.Row():
                    live_user = gr.Textbox(
                        label="What was the AI asked to do?",
                        lines=5,
                        placeholder="Example: Review this vendor evidence and tell me what is missing.",
                    )
                    live_agent = gr.Textbox(
                        label="What did the AI actually say or do?",
                        lines=5,
                        placeholder="Paste the observed AI response.",
                    )

                with gr.Accordion("Advanced control inputs", open=False):
                    gr.Markdown(
                        "Use these only when the monitored application exposes structured actions, context provenance, or mandatory escalation flags."
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
                    live_business_status = gr.Markdown(_business_status_markdown(None))
                    live_business_why = gr.Markdown(_business_why_markdown(None))
                live_business_action = gr.Markdown(_business_action_markdown(None))

                with gr.Accordion("View Engineering Details", open=False):
                    live_engineering_status = gr.Markdown(_status_markdown(None))
                    with gr.Row():
                        live_scores = gr.Dataframe(
                            headers=["Coherence dimension", "Application score"],
                            datatype=["str", "number"],
                            value=[],
                            label="Application-derived coherence scores",
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
                            label="Validated findings",
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
                        label="Append-only audit trail",
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
                        live_business_status,
                        live_business_why,
                        live_business_action,
                        live_engineering_status,
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
                        live_business_status,
                        live_business_why,
                        live_business_action,
                        live_engineering_status,
                        live_scores,
                        live_signals,
                        live_repair,
                        live_audit,
                        live_message,
                    ],
                )

            with gr.Tab("Engineering Lab"):
                gr.Markdown(
                    """### Full deterministic evaluation harness

This is the technical view. It exposes all eight synthetic scenarios, raw engineering states, validated findings, application-derived scores, repair directives, and append-only audit events. No live inference is used here.
"""
                )

                engineering_state = gr.State(start_scenario(default_engineering))

                with gr.Row():
                    engineering_picker = gr.Dropdown(
                        choices=engineering_choices,
                        value=default_engineering,
                        label="Evaluation scenario",
                    )
                    engineering_reset = gr.Button("Load / Reset Scenario", variant="secondary")
                    engineering_next = gr.Button("Run Next Turn", variant="primary")

                engineering_info = gr.Markdown(
                    _scenario_overview_markdown(default_engineering)
                )
                engineering_message = gr.Markdown("Scenario loaded. Run the first turn.")

                with gr.Row():
                    lab_status = gr.Markdown(_status_markdown(None))
                    lab_turn = gr.Markdown(_turn_markdown(None))

                with gr.Row():
                    lab_scores = gr.Dataframe(
                        headers=["Coherence dimension", "Score"],
                        datatype=["str", "number"],
                        value=[],
                        label="Coherence scores",
                        interactive=False,
                    )
                    lab_signals = gr.Dataframe(
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

                lab_repair = gr.Markdown(_repair_markdown(None))
                lab_audit = gr.Dataframe(
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

                engineering_reset.click(
                    _engineering_load,
                    inputs=[engineering_picker],
                    outputs=[
                        engineering_state,
                        engineering_info,
                        lab_status,
                        lab_turn,
                        lab_scores,
                        lab_signals,
                        lab_repair,
                        lab_audit,
                        engineering_message,
                    ],
                )
                engineering_picker.change(
                    _engineering_load,
                    inputs=[engineering_picker],
                    outputs=[
                        engineering_state,
                        engineering_info,
                        lab_status,
                        lab_turn,
                        lab_scores,
                        lab_signals,
                        lab_repair,
                        lab_audit,
                        engineering_message,
                    ],
                )
                engineering_next.click(
                    _engineering_next,
                    inputs=[engineering_state, engineering_picker],
                    outputs=[
                        engineering_state,
                        lab_status,
                        lab_turn,
                        lab_scores,
                        lab_signals,
                        lab_repair,
                        lab_audit,
                        engineering_message,
                    ],
                )

        gr.Markdown(
            """---
## Why this matters in production

A company may deploy many autonomous agents across finance, healthcare, procurement, customer service, cybersecurity, or operations. Those agents may begin correctly but gradually approve things they were only supposed to review, make decisions they were only supposed to recommend, follow untrusted context that conflicts with their original role, or skip required human escalation.

**Role Coherence Monitor provides an independent control layer around those agents.**

### Engineering principle

**Models classify behavior. Applications enforce authority.**

The live model cannot rewrite the role contract, assign final control states, create critical semantic signals, or choose numeric scores. Application code owns deterministic rules, score derivation, state transitions, repair, escalation, blocking, and the append-only audit trail.
"""
        )

    return demo


demo = build_app()


if __name__ == "__main__":
    demo.launch()
