"""Smoke test: the Streamlit app runs end to end on the committed synthetic DB."""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "src" / "insta_analytics" / "dashboard" / "app.py"


def test_app_renders_synthetic_demo_without_exceptions(monkeypatch):
    monkeypatch.setenv("INSTA_DATA_SOURCE", "synthetic")
    monkeypatch.delenv("INSTA_DUCKDB_PATH", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=600).run()
    assert not at.exception
    assert any("synthetic data" in info.value for info in at.info)
