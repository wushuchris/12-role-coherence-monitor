# Hugging Face Deployment

GitHub remains the source of truth for Agent 12.

## Target Space

`FlyingNunchucks/12-role-coherence-monitor`

## Credential separation

### GitHub Actions

Secret:

- `HF_DEPLOY_TOKEN` — deployment only; use a fine-grained Hugging Face token with write access limited to the Agent 12 Space.

Variable:

- `HF_DEPLOY_ENABLED=true` — explicitly enables deployment after the Space and token are configured.

The deployment workflow never uses the runtime inference token.

### Hugging Face Space

Variables:

- `HF_MODEL=Qwen/Qwen3-30B-A3B`
- `HF_PROVIDER=deepinfra`

Secret:

- `HF_TOKEN` — runtime inference only.

The Space must not store `HF_DEPLOY_TOKEN`.

## Deployment gate

1. A push to `main` runs the complete GitHub test suite.
2. The deployment workflow receives the completed Tests workflow event.
3. Deployment proceeds only when:
   - Tests concluded successfully,
   - the tested branch was `main`,
   - the Tests workflow was triggered by a push, and
   - `HF_DEPLOY_ENABLED` is exactly `true`.
4. The workflow checks out the exact tested commit SHA.
5. A Space-specific README replaces the GitHub README only inside the disposable runner checkout.
6. Only runtime files are mirrored to Hugging Face:
   - `README.md`
   - `LICENSE`
   - `requirements.txt`
   - `app.py`
   - `src/*`

Tests, GitHub workflow files, and deployment documentation are not copied to the Space.
