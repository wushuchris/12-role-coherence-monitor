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
