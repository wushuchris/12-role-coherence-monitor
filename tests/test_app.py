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
