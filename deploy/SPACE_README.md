---
title: Role Coherence Monitor
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.27.0
python_version: "3.11"
app_file: app.py
short_description: Keep autonomous AI agents inside their assigned authority.
models:
  - Qwen/Qwen3-30B-A3B
---

# Role Coherence Monitor

**Keep autonomous AI agents inside the authority they were actually given.**

This portfolio project demonstrates an independent supervisory control layer that monitors another AI for gradual role drift, hard authority violations, missed human escalation, and bounded repair.

The Space provides:

- **Business Demo** — three plain-English business stories
- **Try Your Own Example** — live semantic monitoring
- **Engineering Lab** — the full deterministic evaluation harness

## Architecture

`RoleContract → deterministic checks → semantic findings → application-derived scores → longitudinal state → bounded repair / escalation / block → append-only audit`

**Design rule:** Models classify behavior. Applications enforce authority.

The live model cannot choose numeric scores or final control states.

Source code: https://github.com/wushuchris/12-role-coherence-monitor
