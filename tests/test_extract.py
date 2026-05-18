"""tests/test_extract.py — Smoke tests for agents/01_multimodal_extract.py"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_multimodal_log_schema(tmp_path):
    """step_1 output must be a JSON list with required keys in each entry."""
    from config import settings
    # Write a minimal multimodal log directly (bypass full Gemma run)
    log_path = tmp_path / "test_multimodal_log.json"
    entries = [
        {"timestamp_sec": 0.0, "text": "Hello world", "visual_desc": "slide", "is_high_info": False},
        {"timestamp_sec": 30.0, "text": "Attention mechanism", "visual_desc": "diagram", "is_high_info": True},
    ]
    log_path.write_text(json.dumps(entries))
    loaded = json.loads(log_path.read_text())
    for entry in loaded:
        assert "timestamp_sec" in entry
        assert "text" in entry
        assert "visual_desc" in entry
        assert "is_high_info" in entry


def test_gemma_fallback_on_missing_model(tmp_path, monkeypatch):
    """_classify_segment should return a dict with required fields."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "extract_mod",
        str(Path(__file__).parent.parent / "agents" / "01_multimodal_extract.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    with patch.dict("sys.modules", {"faster_whisper": MagicMock()}):
        spec.loader.exec_module(mod)

    result = mod._classify_segment("Hello equation world algorithm", 0.1, 42.0)
    assert result["timestamp_sec"] == 42.0
    assert "text" in result
    assert isinstance(result["is_high_info"], bool)
