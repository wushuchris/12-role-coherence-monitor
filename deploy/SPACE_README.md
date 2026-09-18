---
title: Role Coherence Monitor
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: "3.11"
app_file: app.py
short_description: Monitor AI agents for role drift and bounded repair.
models:
  - Qwen/Qwen3-30B-A3B
---

# Role Coherence Monitor

A portfolio demonstration of a bounded control loop for monitoring long-running AI agents for role drift, mission misalignment, authority expansion, evidence degradation, and repair outcomes.

The default **Guided Scenario Lab** runs entirely from validated synthetic fixtures and does not require live inference. The **Live Semantic Monitor** activates only when the Space runtime is configured with `HF_MODEL`, `HF_PROVIDER`, and the private `HF_TOKEN` secret.

## Architecture

`RoleContract → deterministic checks → semantic assessment → longitudinal state → bounded repair → append-only audit`

**Design rule:** Models interpret. Applications enforce.

Source code: https://github.com/wushuchris/12-role-coherence-monitor
