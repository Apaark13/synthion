"""Tests for engine_v4.ingest (Layer 1)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Import directly from the submodule to avoid pulling engine_v4.runner
# (not yet implemented).
from engine_v4 import ingest as ingest_mod
from engine_v4.ingest import ingest, _build_windows
from engine_v4.types import Frame, SourceSpec, TranscriptSegment
from engine_v4.workspace import stable_source_id


def test_stable_source_id_consistent():
    a = SourceSpec(location="https://www.youtube.com/watch?v=ABCDEFG&list=PL1&index=2", kind="url")
    b = SourceSpec(location="https://www.youtube.com/watch?v=ABCDEFG", kind="url")
    # Volatile params are stripped → same id.
    assert stable_source_id(a) == stable_source_id(b)
    # Same spec is deterministic.
    assert stable_source_id(a) == stable_source_id(a)
    # Different kind → different id.
    c = SourceSpec(location=a.location, kind="video")
    assert stable_source_id(c) != stable_source_id(a)


def test_ingest_text_file(tmp_path: Path):
    txt = tmp_path / "lesson.txt"
    txt.write_text("alpha beta gamma delta " * 100)  # 400 words → 3 segments of 150
    run_ws = tmp_path / "run"
    run_ws.mkdir()

    src = ingest(SourceSpec(location=str(txt), kind="text"), run_ws)

    assert src.media_path is not None and src.media_path.exists()
    assert len(src.segments) >= 2
    # Timestamps spaced ~30s apart, monotonically increasing.
    starts = [s.start_sec for s in src.segments]
    assert starts == sorted(starts)
    assert starts[1] - starts[0] == pytest.approx(30.0)
    # Persistence.
    seg_json = src.workspace / "transcript" / "segments.json"
    win_json = src.workspace / "transcript" / "windows.json"
    meta_json = src.workspace / "meta.json"
    assert seg_json.exists() and win_json.exists() and meta_json.exists()
    meta = json.loads(meta_json.read_text())
    assert meta["kind"] == "text"
    assert meta["n_segments"] == len(src.segments)
    # No frames for plain text.
    assert src.frames == []


def test_ingest_pdf_skipped_gracefully_when_no_libs(tmp_path: Path, monkeypatch):
    # Create a fake .pdf file (content doesn't matter — we mock the libs).
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake\n")
    run_ws = tmp_path / "run"
    run_ws.mkdir()

    # Simulate every PDF backend being unimportable.
    blocked = {"fitz", "pdfplumber", "pdfminer", "pdfminer.high_level"}

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name in blocked or name.split(".")[0] in blocked:
            raise ImportError(f"blocked: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    # Also force pdftoppm to look unavailable so no rendering happens.
    monkeypatch.setattr(ingest_mod.shutil, "which", lambda _: None)

    src = ingest(SourceSpec(location=str(pdf), kind="pdf"), run_ws)

    # Should not raise. Should still return a Source with empty frames.
    assert src.frames == []
    assert src.media_path is not None and src.media_path.exists()
    meta = json.loads((src.workspace / "meta.json").read_text())
    assert meta["kind"] == "pdf"
    assert meta["n_frames"] == 0


def test_window_aggregation():
    # 12 segments × 10s = 120s total. With min=60, max=120, target=75
    # we expect each window to be ≥60s and ≤120s.
    segments = [
        TranscriptSegment(start_sec=i * 10.0, end_sec=(i + 1) * 10.0, text=f"seg{i} matrix vector")
        for i in range(12)
    ]
    windows = _build_windows(segments, frames=[])
    assert windows, "should produce at least one window"
    for w in windows:
        span = w.end_sec - w.start_sec
        assert 60.0 <= span <= 120.0, f"window {w.start_sec}-{w.end_sec} span={span}"
    # Total coverage should match the segment span.
    assert windows[0].start_sec == 0.0
    assert windows[-1].end_sec == 120.0
    # 'matrix' + 'vector' twice per segment → keyword_hits >= 2 → high info.
    assert all(w.is_high_info for w in windows)


def test_window_aggregation_with_frame_marks_high_info():
    segs = [
        TranscriptSegment(start_sec=i * 10.0, end_sec=(i + 1) * 10.0, text="hello world plain prose")
        for i in range(8)
    ]
    # No keywords, but a frame inside the window → still high info.
    frames = [Frame(timestamp_sec=15.0, path=Path("/tmp/x.png"))]
    windows = _build_windows(segs, frames=frames)
    assert windows
    assert any(w.is_high_info for w in windows)
