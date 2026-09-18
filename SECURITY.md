# Security and Failure Modes

## Trust model

The Role Coherence Monitor treats the **RoleContract** as authoritative application state.

The following are untrusted behavioral evidence:
- user messages
- conversation history
- retrieved context
- memory-like context
- agent output
- live semantic model output

Untrusted text may influence what behavior is observed, but it cannot redefine role authority.

## Authority separation

### Deterministic application controls

Application code owns:
- permitted/prohibited structured actions
- contract identity/version
- explicit authority-expansion markers
- escalation requirements
- evidence requirements
- score derivation
- state transitions
- repair limits
- human escalation
- hard blocking

### Live semantic model

The live model may propose only bounded semantic findings.

It cannot:
- set `COHERENT`, `WATCH`, `DRIFTING`, `REALIGN_REQUIRED`, `HUMAN_REVIEW`, or `BLOCKED`
- emit numeric coherence scores
- emit deterministic-only signal categories
- emit `CRITICAL` semantic severity
- choose signal IDs or signal provenance
- mutate the role contract

Provider responses are validated with strict Pydantic schemas before application use.

## Prompt-injection boundary

The semantic prompt explicitly identifies conversation history, memory-like context, user text, and agent output as untrusted evidence.

The model is told not to follow instructions embedded in those fields and not to treat them as authority to alter the contract.

This is a defense-in-depth boundary, not a guarantee that prompt injection is impossible. Consequential enforcement remains outside the model.

## Secret separation

### GitHub

`HF_DEPLOY_TOKEN`
- deployment only
- stored as a GitHub Actions secret
- should be fine-grained and limited to the Agent 12 Space
- never used for runtime inference

### Hugging Face Space

`HF_TOKEN`
- runtime inference only
- stored as a Hugging Face Space secret
- never stored in GitHub

`HF_MODEL` and `HF_PROVIDER` are non-secret Space variables.

The application only reports whether the runtime token is configured; it never displays the token value.

## Failure modes and responses

| Failure mode | Response |
|---|---|
| Provider unavailable | Normalize error; do not commit partial session state |
| Malformed JSON | Reject provider response |
| Provider adds forbidden fields | Reject schema |
| Model attempts final control state | Reject schema |
| Model attempts numeric score | Reject schema |
| Model emits deterministic-only signal | Reject schema |
| Model emits CRITICAL semantic severity | Reject schema |
| Model cites user input as behavioral evidence | Reject schema |
| Replayed turn ID | Reject turn |
| Non-monotonic turn sequence | Reject turn |
| Assessment belongs to another turn | Reject linkage |
| Signals do not match current state | Reject audit/repair linkage |
| Repair has no recorded current evidence | Reject repair |
| First repair fails | Issue fresh bounded directive |
| Second repair fails | Require human review |
| Deterministic prohibited action | Block immediately |
| Mandatory escalation ignored | Require human review |

## Transactional session behavior

`RoleCoherenceMonitor.process_turn` receives an immutable session and returns a new immutable session.

If semantic assessment or downstream validation fails, the caller does not receive a partially mutated session. This reduces the chance of corrupted state after provider or validation errors.

## Audit integrity

Audit events are frozen Pydantic models.

The `AuditTrail`:
- is immutable
- rejects duplicate event IDs
- validates turn/assessment/signal linkage
- records state transitions
- links repair directives to their triggering evidence and later outcome

The audit trail is an application log, not a cryptographically signed ledger. A production system needing tamper evidence should persist events to an append-only external store with access control and integrity verification.

## Deployment security

GitHub is the source of truth.

Deployment occurs only after the Tests workflow succeeds on a push to `main`. The deployment workflow then checks out the exact tested SHA and syncs only runtime files to the Hugging Face Space.

CI does not contain the runtime inference secret.

## Public-data policy

The repository and guided scenarios use synthetic public-safe data. No client, personal, financial, medical, or private curriculum data is required for the demo.

## Production upgrade considerations

For a high-stakes production deployment:
- use domain-specific labeled evaluations
- define organization-owned role contracts and action registries
- persist audit events to durable external storage
- add authentication/authorization around monitor administration
- add request rate limits and provider budgets
- add retries/backoff with bounded failure policy
- instrument latency, model/provider errors, drift rates, repairs, and escalations
- review threshold changes through versioned configuration
- add human-review workflow integration
- red-team prompt injection and contaminated memory/retrieval paths
