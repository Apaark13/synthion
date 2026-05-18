"""tests/test_publish.py — Smoke tests for agents/07_image_upscale_agent.py
                           and agents/08_publish_agent.py"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


# ── Image upscale tests ───────────────────────────────────────────────────────

def test_find_best_segment_closest():
    """_find_best_segment should return the segment with the closest start time."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "upscale_mod",
        str(Path(__file__).parent.parent / "agents" / "07_image_upscale_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    segments = [
        Path("lecture_000010.mp4"),
        Path("lecture_000060.mp4"),
        Path("lecture_000120.mp4"),
    ]
    result = mod._find_best_segment(65.0, segments)
    assert result is not None
    assert "000060" in result.name


def test_step7_empty_timestamps(tmp_path, monkeypatch):
    """step_7 with no timestamps should return an empty list without error."""
    from config import settings
    monkeypatch.setattr(settings, "IMAGES_DIR", tmp_path)
    monkeypatch.setattr(settings, "VIDEO_HQ_DIR", tmp_path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "upscale_mod2",
        str(Path(__file__).parent.parent / "agents" / "07_image_upscale_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.IMAGES_DIR = tmp_path

    with patch.object(mod, "_update_checkpoint"), patch.object(mod, "_append_done"):
        result = mod.step_7(flagged_timestamps=[], hq_segments=[], multimodal_log=[])

    assert result == []


# ── Publish tests ─────────────────────────────────────────────────────────────

def test_resolve_figure_anchors(tmp_path):
    """_resolve_figure_anchors should replace [Figure:...] with image tags."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pub_mod",
        str(Path(__file__).parent.parent / "agents" / "08_publish_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Create a fake upscaled PNG so the path exists
    fake_png = tmp_path / "fig_000_upscaled.png"
    fake_png.write_bytes(b"\x89PNG\r\n")

    manifest = [{"index": 0, "timestamp_sec": 42.0, "caption": "Matrix multiply", "path": str(fake_png)}]
    text = "See [Figure: timestamp=00:00:42, desc=Matrix multiply diagram] for details."
    result = mod._resolve_figure_anchors(text, manifest)

    assert "![Figure" in result or "Matrix multiply" in result


def test_assemble_textbook_creates_md(tmp_path):
    """_assemble_textbook should write a non-empty textbook.md."""
    from config import settings
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pub_mod2",
        str(Path(__file__).parent.parent / "agents" / "08_publish_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.BOOK_DIR = tmp_path

    chapters = {
        "ch_001": "## Introduction\n\nHello world.",
        "ch_002": "## Attention\n\nAttention mechanisms matter.",
    }
    toc = {"sections": [{"id": "ch_001"}, {"id": "ch_002"}]}
    manifest: list = []

    out_path = mod._assemble_textbook(chapters, toc, manifest)
    assert out_path.exists()
    content = out_path.read_text()
    assert "Hello world" in content
    assert "Attention" in content
    assert len(content) > 50
