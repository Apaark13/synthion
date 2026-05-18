"""tests/test_fetch.py — Smoke tests for agents/00_fetch.py (v2 two-pass fetch)"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

# Patch heavy subprocess calls before importing the agent
@pytest.fixture(autouse=True)
def mock_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr("subprocess.run", MagicMock(return_value=MagicMock(returncode=0)))
    monkeypatch.setattr("subprocess.check_output", MagicMock(return_value=b"120.0\n"))


def test_step0_returns_dict(tmp_path, monkeypatch):
    """step_0 should return a dict with the required keys."""
    from config import settings
    monkeypatch.setattr(settings, "VIDEO_LOW_RES_DIR", tmp_path / "low_res")

    # Create a fake .mp4 so _ydlp_download thinks it succeeded
    (tmp_path / "low_res").mkdir(parents=True)
    fake_mp4 = tmp_path / "low_res" / "lecture.mp4"
    fake_mp4.write_bytes(b"\x00" * 16)

    with patch("agents.fetch_agent._ydlp_download", return_value=fake_mp4):
        import importlib, sys
        if "agents.fetch_agent" in sys.modules:
            del sys.modules["agents.fetch_agent"]
        spec = importlib.util.spec_from_file_location(
            "agents.fetch_agent",
            str(Path(__file__).parent.parent / "agents" / "00_fetch.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(mod, "_update_checkpoint"), patch.object(mod, "_append_done"):
            result = mod.step_0("https://youtube.com/watch?v=test")

    assert isinstance(result, dict)
    assert "low_res_path" in result
    assert "hq_segments" in result
    assert "flagged_timestamps" in result
    assert result["hq_segments"] == []  # v3: no HQ downloads


def test_video_url_saved(tmp_path, monkeypatch):
    """step_0 should save the YouTube URL alongside the video."""
    from config import settings
    monkeypatch.setattr(settings, "VIDEO_LOW_RES_DIR", tmp_path / "low_res")

    (tmp_path / "low_res").mkdir(parents=True)
    fake_mp4 = tmp_path / "low_res" / "lecture.mp4"
    fake_mp4.write_bytes(b"\x00" * 16)

    with patch("agents.fetch_agent._ydlp_download", return_value=fake_mp4):
        import importlib, sys
        if "agents.fetch_agent" in sys.modules:
            del sys.modules["agents.fetch_agent"]
        spec = importlib.util.spec_from_file_location(
            "agents.fetch_agent",
            str(Path(__file__).parent.parent / "agents" / "00_fetch.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(mod, "_update_checkpoint"), patch.object(mod, "_append_done"):
            mod.step_0("https://youtube.com/watch?v=test123")

    url_file = tmp_path / "low_res" / ".lecture.url"
    assert url_file.exists()
    assert url_file.read_text().strip() == "https://youtube.com/watch?v=test123"
