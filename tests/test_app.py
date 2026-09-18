"""Smoke tests for the Gradio portfolio application."""

import importlib

import gradio as gr


def test_app_builds_without_runtime_inference_configuration(monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.delenv("HF_PROVIDER", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    app_module = importlib.import_module("app")
    demo = app_module.build_app()

    assert isinstance(demo, gr.Blocks)


def test_app_import_does_not_require_hugging_face_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    app_module = importlib.import_module("app")

    assert isinstance(app_module.demo, gr.Blocks)


def test_drifting_status_explanation_covers_severe_or_sustained_drift():
    app_module = importlib.import_module("app")

    explanation = app_module.STATUS_EXPLAINER["DRIFTING"]

    assert "significant or sustained deviation" in explanation
    assert "persisted across turns" not in explanation


def test_business_demo_exposes_three_plain_english_stories():
    app_module = importlib.import_module("app")

    assert tuple(app_module.BUSINESS_STORIES) == (
        "slow-scope-creep",
        "direct-approval-attempt",
        "ignored-required-escalation",
    )
    labels = [
        story["label"]
        for story in app_module.BUSINESS_STORIES.values()
    ]
    assert any("someone else's job" in label for label in labels)
    assert any("tries to approve" in label for label in labels)
    assert any("called a human" in label for label in labels)


def test_business_status_translates_engineering_state_without_hiding_it():
    app_module = importlib.import_module("app")

    rendered = app_module._business_status_markdown(
        {"status": "DRIFTING"}
    )

    assert "Operating outside role" in rendered
    assert "Engineering state: `DRIFTING`" in rendered


def test_business_reason_translates_scope_and_authority_signals():
    app_module = importlib.import_module("app")

    rendered = app_module._business_why_markdown(
        {
            "signals": [
                {"type": "scope_drift"},
                {"type": "authority_expansion"},
            ]
        }
    )

    assert "outside its assigned job" in rendered
    assert "authority it was never granted" in rendered
    assert "scope_drift" not in rendered
    assert "authority_expansion" not in rendered


def test_business_action_explains_hard_block_in_plain_english():
    app_module = importlib.import_module("app")

    rendered = app_module._business_action_markdown(
        {"status": "BLOCKED"}
    )

    assert "Block the prohibited action immediately" in rendered
    assert "BLOCKED" not in rendered


def test_business_role_copy_makes_supervisory_case_explicit():
    app_module = importlib.import_module("app")

    rendered = app_module._business_role_markdown()

    assert "Compliance Review Agent" in rendered
    assert "It may NOT" in rendered
    assert "independent supervisory control layer" in rendered
