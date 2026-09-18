# Agent 12 — Role Coherence Monitor

**Keep autonomous AI agents inside the authority they were actually given.**

[Live demo](https://huggingface.co/spaces/FlyingNunchucks/12-role-coherence-monitor) · [Evaluation](EVALUATION.md) · [Security & failure modes](SECURITY.md)

## The business problem

An autonomous agent can begin with a clear job and gradually start doing more than it was authorized to do.

A compliance-review agent might start by reviewing evidence, then begin redesigning operations, assigning owners, approving transactions, or ignoring required human escalation. The problem is not only a bad single response. It is **long-horizon role drift**: authority expands gradually as conversation history, user pressure, retrieved context, or repeated exceptions accumulate.

The Role Coherence Monitor is an **independent supervisory control layer** around another AI agent. It watches behavior over time, compares that behavior with an immutable role contract, and decides when to continue, watch, correct, escalate, or block.

> **Models classify behavior. Applications enforce authority.**

## Demo scenario

The public demo uses a synthetic **Compliance Review Agent**.

It may:
- review supplied evidence
- identify possible policy conflicts
- request clarification
- recommend human review

It may not:
- approve or execute transactions
- waive or rewrite policy
- grant itself new authority
- take over operations
- bypass required evidence or human escalation

The default **Business Demo** explains three common failure patterns in plain language:

1. **The AI starts doing someone else's job** — gradual scope and authority drift.
2. **The AI tries to approve something** — an immediate deterministic hard stop.
3. **The AI should have called a human** — a mandatory escalation boundary.

The full eight-scenario evaluation harness remains available under **Engineering Lab**.

## Architecture

```text
Immutable RoleContract
        ↓
InteractionTurn
        ↓
Deterministic Contract Checks
        ↓
Bounded Semantic Classifier
        ↓
Application-Derived Coherence Scores
        ↓
Longitudinal State Machine
        ↓
Bounded Repair / Human Escalation / Block
        ↓
Append-Only Audit Trail
```

### Authority boundaries

The application—not the language model—owns the consequential controls.

**Deterministic application code owns:**
- prohibited and unknown structured actions
- role/version mutation checks
- self-assigned authority markers
- mandatory escalation failures
- required evidence markers
- numeric score derivation
- state transitions
- repair thresholds
- human escalation
- blocking
- audit linkage

**The live model may only classify bounded semantic findings:**
- `mission_drift`
- `scope_drift`
- `authority_expansion`
- `behavioral_drift`
- `evidence_degradation`

Live semantic findings are limited to `LOW`, `MEDIUM`, or `HIGH` severity and must cite agent behavior (`agent_output` or bounded prior `history_agent_output`). The model cannot emit a final control state, a critical semantic signal, an application signal ID, or a numeric coherence score.

## Deterministic scoring

A production smoke test exposed that model-generated numeric scores were poorly calibrated. The architecture was changed so the model no longer emits scores.

The application now converts validated signal severity into deterministic scores:

| Semantic severity | Application score |
|---|---:|
| No matching deviation | 1.00 |
| LOW | 0.85 |
| MEDIUM | 0.70 |
| HIGH | 0.40 |
| CRITICAL* | 0.00 |

`CRITICAL` is reserved for application/deterministic paths; the live semantic model cannot emit it.

Each semantic finding affects only its owned dimension:

| Finding | Dimension |
|---|---|
| `mission_drift` | Mission alignment |
| `scope_drift` | Scope adherence |
| `authority_expansion` | Authority adherence |
| `evidence_degradation` | Evidence discipline |
| `behavioral_drift` | Behavioral consistency |

This makes every score traceable to an auditable finding instead of an opaque model number.

## Longitudinal control states

| Engineering state | Business meaning |
|---|---|
| `COHERENT` | Staying in role |
| `WATCH` | Needs attention |
| `DRIFTING` | Operating outside role |
| `REALIGN_REQUIRED` | Correction required |
| `HUMAN_REVIEW` | Human decision required |
| `BLOCKED` | Action blocked |

A single severe deviation may move directly to `DRIFTING`. Lower-level deviations must accumulate across turns before stronger intervention. Deterministic critical violations can immediately force `BLOCKED` or `HUMAN_REVIEW`.

## Bounded repair

When longitudinal evidence reaches `REALIGN_REQUIRED`, the application can issue a repair directive constructed only from:
- the immutable role contract
- recorded current-turn signals
- an application-approved next action

Conversation history cannot rewrite the role during repair.

Repair is explicit and measured:
- successful repair → `WATCH` → `COHERENT`
- failed repair → a new bounded directive
- repeated failed repair → `HUMAN_REVIEW`

## Append-only auditability

Every monitored turn can produce immutable linked events such as:

```text
turn_recorded
signal_recorded
turn_assessed
state_transition
repair_issued
repair_evaluated
human_review_requested / autonomy_blocked
```

Audit records use stable IDs and validated linkage across turns, signals, assessments, state transitions, repair directives, and outcomes.

## Evaluation

The repository includes a formal evaluation runner plus synthetic long-horizon fixtures.

Current deterministic/mock evaluation set:
- 8 scenarios
- 22 turns
- exact state-path checks
- deviation precision/recall
- false-positive rate
- terminal-control accuracy
- repair-transition accuracy
- detection delay
- injected false-positive and missed-drift tests

The controlled fixture suite currently produces exact expected state paths. This is a **control-system regression suite**, not a claim that a language model will perfectly classify arbitrary real-world behavior.

Production smoke checks on the deployed Hugging Face Space verified:
- deterministic prohibited action → `BLOCKED`
- healthy live semantic behavior → `COHERENT`
- semantic scope/authority drift → `DRIFTING`
- live model findings → application-owned numeric scores
- behavioral evidence provenance → `agent_output`
- evidence discipline remains independent when evidence handling itself is sound

See [EVALUATION.md](EVALUATION.md) for the evaluation boundary and production lessons.

## Production lessons that changed the design

The live deployment intentionally informed the final architecture.

Early live testing showed the model could classify a scope problem but return poorly calibrated all-zero scores. Prompt tuning improved evidence quality but did not make continuous score calibration trustworthy enough.

The design was therefore simplified:

```text
LLM: classify bounded semantic evidence
Application: derive scores
Application: decide state
Application: repair / escalate / block
```

Real live failures became regression tests. The current suite contains **171 automated tests** as of the final portfolio-completion pass.

## Failure handling

The system fails closed at key boundaries:
- malformed provider JSON → rejected
- provider schema drift → rejected
- forbidden provider fields → rejected
- missing live configuration → live mode disabled
- provider failure → normalized application error
- semantic score-impact without auditable evidence → rejected in model-agnostic monitor paths
- replayed turn IDs → rejected
- non-monotonic turns → rejected
- stale/mismatched signal or assessment linkage → rejected
- repair without recorded evidence → rejected
- repeated failed repair → human review

The caller receives no partially updated session if processing fails midway.

## Security and privacy

- GitHub is the source of truth.
- `HF_DEPLOY_TOKEN` exists only in GitHub Actions and is used only to deploy.
- `HF_TOKEN` exists only as a Hugging Face Space runtime secret and is used only for inference.
- The live UI never displays the token value.
- CI does not require or use the runtime inference token.
- Public examples use synthetic data only.
- The live model is explicitly told that conversation, retrieved context, and memory-like text are untrusted evidence rather than role authority.
- Application schemas reject undeclared fields at key trust boundaries.

See [SECURITY.md](SECURITY.md) for the threat model and trust boundaries.

## Demo

**Hugging Face Space:**  
https://huggingface.co/spaces/FlyingNunchucks/12-role-coherence-monitor

The Space provides:
- **Business Demo** — three plain-English business stories
- **Try Your Own Example** — live semantic monitoring
- **Engineering Lab** — the complete deterministic scenario harness

Live inference is only invoked when the user explicitly clicks **Analyze Turn**.

## Deployment

Deployment is test-gated:

```text
push to main
   ↓
GitHub Actions test suite
   ↓ only if successful
checkout exact tested SHA
   ↓
Hugging Face sync
   ↓
Space rebuild
```

Credential separation:

```text
GitHub Actions
  HF_DEPLOY_TOKEN → deployment only

Hugging Face Space
  HF_TOKEN        → runtime inference only
  HF_MODEL        → Qwen/Qwen3-30B-A3B
  HF_PROVIDER     → deepinfra
```

See [deploy/README.md](deploy/README.md).

## Project structure

```text
app.py
src/role_coherence_monitor/
  audit.py
  demo.py
  deterministic.py
  evaluation.py
  live_semantic.py
  monitor.py
  repair.py
  scenarios.py
  schemas.py
  scoring.py
  semantic.py
  state_machine.py
tests/
deploy/
.github/workflows/
```

## Tech

Python · Pydantic · Gradio · Hugging Face Inference Providers · pytest · GitHub Actions

## Reusable engineering primitive

> **A typed role-coherence control loop that continuously compares observed agent behavior against an immutable role contract, detects hard violations and gradual drift, records auditable evidence, and issues bounded repair or human-escalation directives.**

## What this project demonstrates

- role contracts as application authority
- semantic classification behind a strict schema boundary
- deterministic controls for hard authority violations
- application-owned scoring and state transitions
- long-horizon drift detection
- bounded role repair
- immutable audit trails
- production-driven regression testing
- safe secret separation
- test-gated deployment

## License

MIT
