"""Unit tests for modules.visual_sync. No real video needed."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modules.visual_sync import (
    AlignedFigure,
    TranscriptSegment,
    VisualSyncConfig,
    extract_aligned_figures,
)
from modules.visual_sync.align import align_section, score_pair
from modules.visual_sync.dedupe import dedupe_by_phash, hamming, is_solid_color, phash, slide_likeness
from modules.visual_sync.ocr import configure as ocr_configure
from modules.visual_sync.types import CandidateFrame


# --- helpers ---------------------------------------------------------------

def _checker_img(path: Path, color_a=(255, 255, 255), color_b=(0, 0, 0), block=8, size=128):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, block):
        for x in range(0, size, block):
            arr[y:y + block, x:x + block] = color_a if ((x // block + y // block) % 2 == 0) else color_b
    Image.fromarray(arr).save(path)


def _solid_img(path: Path, rgb=(50, 50, 50), size=128):
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    Image.fromarray(arr).save(path)


# --- dedupe ----------------------------------------------------------------

def test_phash_and_hamming_basic(tmp_path):
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _checker_img(a)
    _checker_img(b)  # identical
    ha, hb = phash(a), phash(b)
    assert ha is not None and hb is not None
    assert hamming(ha, hb) <= 2  # near-zero distance


def test_is_solid_color_detects_uniform(tmp_path):
    p = tmp_path / "blank.png"
    _solid_img(p)
    assert is_solid_color(p, std_threshold=5.0)
    q = tmp_path / "patterned.png"
    _checker_img(q)
    assert not is_solid_color(q, std_threshold=5.0)


def test_slide_likeness_higher_for_patterned(tmp_path):
    blank = tmp_path / "b.png"; _solid_img(blank)
    pat = tmp_path / "p.png"; _checker_img(pat, block=4)
    assert slide_likeness(pat) > slide_likeness(blank)


def test_dedupe_by_phash_drops_duplicates(tmp_path):
    a = tmp_path / "a.png"; _checker_img(a)
    b = tmp_path / "b.png"; _checker_img(b)  # dup of a
    c = tmp_path / "c.png"; _solid_img(c, rgb=(200, 50, 50))  # different
    items = [(0.0, str(a), 1.0), (1.0, str(b), 1.0), (2.0, str(c), 1.0)]
    kept = dedupe_by_phash(items, threshold=10)
    paths = [p for _, p, _ in kept]
    assert str(a) in paths and str(c) in paths and str(b) not in paths


# --- ocr -------------------------------------------------------------------

def test_ocr_graceful_when_missing():
    name = ocr_configure("none")
    assert name == "none"
    from modules.visual_sync import ocr as ocr_mod
    assert ocr_mod.extract_text("/nonexistent.png") == ""


# --- align -----------------------------------------------------------------

def test_score_pair_components_in_unit_range():
    cfg = VisualSyncConfig()
    f = CandidateFrame(timestamp_sec=10.0, path="x", visual_score=0.9, ocr_text="neural network gradient")
    s = TranscriptSegment(start=8.0, end=12.0, text="we compute the gradient of the neural network here")
    comp, vis, txt, ts = score_pair(f, s, cfg)
    assert 0.0 <= vis <= 1.0
    assert 0.0 <= txt <= 1.0
    assert 0.0 <= ts <= 1.0
    assert 0.0 <= comp <= 1.0
    assert txt > 0  # there is real overlap


def test_align_section_picks_best_candidate():
    cfg = VisualSyncConfig(min_match_score=0.1, min_margin=0.0, max_figures_per_section=2)
    frames = [
        CandidateFrame(timestamp_sec=5.0, path="a", visual_score=1.0, ocr_text="backpropagation chain rule"),
        CandidateFrame(timestamp_sec=50.0, path="b", visual_score=1.0, ocr_text="convolution kernel stride"),
    ]
    segs = [
        TranscriptSegment(start=0.0, end=10.0, text="today we cover backpropagation and the chain rule"),
        TranscriptSegment(start=40.0, end=60.0, text="convolution layers with kernel and stride parameters"),
    ]
    out = align_section(frames, segs, section_id="ch1", cfg=cfg)
    assert len(out) == 2
    assert out[0].matched_segment_idx == 0
    assert out[1].matched_segment_idx == 1
    assert out[0].path == "a"
    assert out[1].path == "b"


def test_align_section_respects_time_gap():
    cfg = VisualSyncConfig(max_time_gap_sec=10.0, min_match_score=0.0, min_margin=0.0)
    frames = [CandidateFrame(timestamp_sec=1000.0, path="x", visual_score=1.0)]
    segs = [TranscriptSegment(start=0.0, end=5.0, text="hello world")]
    out = align_section(frames, segs, cfg=cfg)
    assert out == []


def test_align_section_monotonic_constraint():
    cfg = VisualSyncConfig(min_match_score=0.0, min_margin=0.0, enforce_monotonic=True)
    frames = [
        CandidateFrame(timestamp_sec=1.0, path="a", visual_score=1.0, ocr_text="alpha"),
        CandidateFrame(timestamp_sec=2.0, path="b", visual_score=1.0, ocr_text="beta"),
    ]
    segs = [
        TranscriptSegment(start=0.0, end=2.0, text="alpha discussion"),
        TranscriptSegment(start=3.0, end=5.0, text="beta discussion"),
    ]
    out = align_section(frames, segs, cfg=cfg)
    # frame 'a' should match segment 0, 'b' should match segment 1 (monotonic)
    assert [a.path for a in out] == ["a", "b"]
    assert [a.matched_segment_idx for a in out] == [0, 1]


def test_align_section_returns_empty_on_no_inputs():
    cfg = VisualSyncConfig()
    assert align_section([], [TranscriptSegment(0, 1, "x")], cfg=cfg) == []
    assert align_section([CandidateFrame(0, "p")], [], cfg=cfg) == []


# --- pipeline (smoke, monkey-patched scene+extract) ------------------------

def test_pipeline_end_to_end_with_stubbed_extraction(tmp_path, monkeypatch):
    # Stub scene detection to return two timestamps.
    from modules.visual_sync import scene as scene_mod
    from modules.visual_sync import extract as extract_mod

    monkeypatch.setattr(scene_mod, "detect_scenes",
                        lambda *_a, **_kw: [(2.0, 0.9), (20.0, 0.95)])

    # Stub frame extract: actually write 2 small images.
    def _fake_extract(_video, timestamps, output_dir, **_kw):
        out = []
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for ts in timestamps:
            p = Path(output_dir) / f"f_{ts:.3f}.png"
            _checker_img(p, block=4 if ts < 10 else 8)
            out.append((ts, str(p)))
        return out

    monkeypatch.setattr(extract_mod, "extract_frames", _fake_extract)

    segs = [
        TranscriptSegment(0.0, 5.0, "we introduce the loss function"),
        TranscriptSegment(15.0, 25.0, "softmax produces a probability distribution"),
    ]
    cfg = VisualSyncConfig(min_match_score=0.0, min_margin=0.0, ocr_backend="none")
    result = extract_aligned_figures(
        "fake.mp4", segs, output_dir=tmp_path, config=cfg,
    )
    assert "all" in result
    figs = result["all"]
    assert len(figs) >= 1
    assert all(isinstance(f, AlignedFigure) for f in figs)
    assert all(Path(f.path).exists() for f in figs)
