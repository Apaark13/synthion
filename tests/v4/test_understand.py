"""Tests for engine_v4.understand and engine_v4.index (L2 UNDERSTAND)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

# engine_v4/__init__.py imports .runner which is a future layer; stub it so
# importing the package doesn't fail.
if "engine_v4.runner" not in sys.modules:
    stub = types.ModuleType("engine_v4.runner")
    stub.run = lambda *a, **k: None
    stub.run_many = lambda *a, **k: None
    sys.modules["engine_v4.runner"] = stub

import json

import pytest

from engine_v4 import understand
from engine_v4.index import Context
from engine_v4.types import Source, SourceSpec, TranscriptWindow


# ── Helpers ─────────────────────────────────────────────────────────────────

def _mk_windows(n: int = 12, dt: float = 60.0, hi_every: int = 2):
    topics = [
        "Introduction to attention mechanisms in transformers",
        "Query key and value matrices explained step by step",
        "Softmax normalises attention scores across positions",
        "Multi head attention runs many parallel attention heads",
        "Positional encodings inject order into token embeddings",
        "Residual connections stabilise deep transformer training",
        "Layer normalisation keeps activations well scaled",
        "Encoder decoder architecture for sequence to sequence tasks",
        "Masked self attention enables autoregressive decoding",
        "Cross attention links decoder queries to encoder keys",
        "Training transformers with the Adam optimiser",
        "Inference and beam search for high quality generation",
    ]
    out = []
    for i in range(n):
        out.append(TranscriptWindow(
            start_sec=i * dt,
            end_sec=(i + 1) * dt,
            text=topics[i % len(topics)],
            is_high_info=(i % hi_every == 0),
        ))
    return out


def _mk_source(tmp_path: Path, windows=None, duration=720.0) -> Source:
    spec = SourceSpec(location="https://example.com/x", kind="video", title="x")
    ws = tmp_path / "src"
    ws.mkdir(parents=True, exist_ok=True)
    return Source(
        source_id="x_0000000000",
        spec=spec,
        workspace=ws,
        media_path=None,
        duration_sec=duration,
        windows=windows or _mk_windows(),
    )


# ── Tests ──────────────────────────────────────────────────────────────────

def test_target_chapter_count_v4_1_values():
    """v4.1: minimum 3 chapters whenever video has any meaningful length;
    short videos get a chapter every ~5 minutes, longer ones scale by
    density.
    """
    f = understand._target_chapter_count
    # Short video (<30 min): 1 chapter / 5 min, min 3.
    assert f(15 * 60, 0.30) == 3
    assert f(25 * 60, 0.30) == 5
    # 30 min average-density crosses the short/long boundary → min 3.
    assert f(30 * 60, 0.30) >= 3
    # 90-min average-density: round(90/30)=3, min 3.
    assert f(90 * 60, 0.30) == 3
    # 90-min dense: round(90/20)=4 (banker's rounding to even).
    assert f(90 * 60, 0.55) == 4
    # 90-min light: round(90/45)=2, raised to min 3.
    assert f(90 * 60, 0.10) == 3
    # 120-min dense: round(120/20)=6.
    assert f(120 * 60, 0.50) == 6
    # Always >= 1 even on tiny inputs.
    assert f(60.0, 0.0) >= 1


def test_heuristic_outline_titles_are_title_case():
    windows = _mk_windows()
    sections = understand._heuristic_outline(windows, target_n=4)
    assert len(sections) == 4
    for sec in sections:
        title = sec["title"]
        assert title, "title must be non-empty"
        assert title == title.title(), f"not title-case: {title!r}"
        # Noun-phrase heuristic strips stopwords like 'the', 'in', 'to'.
        lowered = {w.lower() for w in title.split()}
        assert not (lowered & {"the", "in", "to", "of"})
        assert sec["id"].startswith("ch_")
        assert sec["bloom_level"] in understand.BLOOM_LEVELS


def test_context_store_retrieve_round_trip(tmp_path, monkeypatch):
    # Force the deterministic char-hash fallback (no MiniLM dependency).
    import engine_v4.index as idx
    monkeypatch.setattr(idx, "_get_st_model", lambda: None)
    monkeypatch.setattr(idx, "_st_model", None, raising=False)
    monkeypatch.setattr(idx, "_st_tried", True, raising=False)

    windows = _mk_windows(n=8, hi_every=1)
    ctx = Context()
    ctx.store(windows)

    out = ctx.retrieve("attention softmax", top_k=3)
    assert out and isinstance(out, str)
    # Top-k results should each be one of the original window texts.
    parts = [w.text for w in windows]
    assert any(p in out for p in parts)

    # Save / load round trip.
    p = tmp_path / "index.npz"
    ctx.save(p)
    assert p.exists() and p.stat().st_size > 0
    loaded = Context.load(p)
    out2 = loaded.retrieve("attention softmax", top_k=3)
    assert out2 == out


def test_outline_persisted_to_workspace(tmp_path, monkeypatch):
    # Force LLM call to return nothing → exercises heuristic path
    # without needing the real Qwen MLX model.
    monkeypatch.setattr(understand, "call_mlx", lambda *a, **k: "")

    src = _mk_source(tmp_path, duration=90 * 60)
    outline = understand.plan_outline(src)

    assert outline.target_count >= 1
    assert outline.sections, "expected at least one section"
    p = src.workspace / "outline.json"
    assert p.exists()
    payload = json.loads(p.read_text())
    assert payload["target_count"] == outline.target_count
    assert len(payload["sections"]) == len(outline.sections)
    for s in payload["sections"]:
        assert s["id"].startswith("ch_")
        assert s["bloom_level"] in understand.BLOOM_LEVELS
