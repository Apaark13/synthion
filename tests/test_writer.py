"""tests/test_writer.py — Smoke tests for agents/04_writer_agent.py
                          and agents/03_kv_cache_agent.py"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import pytest


# ── KV-cache tests ────────────────────────────────────────────────────────────

def test_kv_cache_store_retrieve(tmp_path, monkeypatch):
    """store_context + retrieve_context round-trip should return related content."""
    from config import settings
    monkeypatch.setattr(settings, "KV_CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "KV_QUANTIZE_BITS", 4)

    import agents
    kv = agents.kv_cache_agent
    kv.KV_CACHE_DIR = tmp_path  # patch module-level constant

    test_text = (
        "Attention mechanisms allow transformers to weigh token importance. "
        "The query and key matrices are multiplied to form attention scores. "
        "Softmax normalises these scores across all positions."
    )
    kv.store_context(test_text, cache_id="test")
    result = kv.retrieve_context("how does attention work", cache_id="test", top_k=2)
    assert len(result) > 0
    # At least one attention-related word should survive the round-trip
    assert any(word in result.lower() for word in ["attention", "query", "softmax", "token"])


def test_kv_cache_empty_retrieve(tmp_path, monkeypatch):
    """retrieve_context returns empty string if no cache exists."""
    from config import settings
    monkeypatch.setattr(settings, "KV_CACHE_DIR", tmp_path)
    import agents
    kv = agents.kv_cache_agent
    kv.KV_CACHE_DIR = tmp_path
    result = kv.retrieve_context("anything", cache_id="nonexistent")
    assert result == ""


# ── Writer tests ──────────────────────────────────────────────────────────────

def test_seconds_to_hhmmss():
    """_seconds_to_hhmmss should format correctly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "writer_mod",
        str(Path(__file__).parent.parent / "agents" / "04_writer_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._seconds_to_hhmmss(0) == "00:00:00"
    assert mod._seconds_to_hhmmss(3661) == "01:01:01"
    assert mod._seconds_to_hhmmss(7322) == "02:02:02"


def test_batched_writer_fallback_when_model_unavailable(tmp_path, monkeypatch):
    """step_4 should produce non-empty output even when Gemma is unavailable."""
    from config import settings
    monkeypatch.setattr(settings, "KV_CACHE_DIR", tmp_path)
    monkeypatch.setattr(settings, "BOOK_DIR", tmp_path / "book")
    (tmp_path / "book").mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "writer_mod",
        str(Path(__file__).parent.parent / "agents" / "04_writer_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)

    with patch("agents.kv_cache_agent.retrieve_context", return_value="Test context."):
        spec.loader.exec_module(mod)

    toc = {"sections": [
        {"id": "ch_001", "title": "Test Section", "bloom_level": "understand",
         "key_terms": ["attention"], "timestamps": []},
    ]}

    # mock the lazy-loaded kv retrieve
    import agents
    agents.kv_cache_agent.retrieve_context = lambda *a, **kw: "Test context about attention."

    with patch.object(mod, "_call_gemma", return_value=""):
        with patch.object(mod, "_update_checkpoint"), patch.object(mod, "_append_done"):
            result = mod.step_4(toc=toc, multimodal_log=[])

    assert "ch_001" in result
    assert len(result["ch_001"]) > 10


def test_sanitise_chapter_strips_meta():
    """_sanitise_chapter removes self-correction notes and section markers."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "writer_mod_san",
        str(Path(__file__).parent.parent / "agents" / "04_writer_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    raw = (
        "## Title\n\nSome prose.\n\n=== SECTION ch_001 ===\n"
        "**Self-Correction/Refinement during drafting:** notes here\n"
        "*(Self-Correction/Review: meta)*\n"
        "**Final Output Generation.** more meta\n"
        "Real final line.\n"
    )
    cleaned = mod._sanitise_chapter(raw, "ch_001", title="Title")
    assert "Self-Correction" not in cleaned
    assert "=== SECTION" not in cleaned
    assert "Final Output" not in cleaned
    assert "Real final line." in cleaned
    assert cleaned.startswith("## Title")


def test_split_batched_with_chapter_tags():
    """_split_batched_output splits on <<<CHAPTER_BEGIN id>>> tags."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "writer_mod_split",
        str(Path(__file__).parent.parent / "agents" / "04_writer_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    raw = (
        "preamble\n"
        "<<<CHAPTER_BEGIN ch_001>>>\n## A\nLong prose body for chapter A here.\n<<<CHAPTER_END>>>\n"
        "<<<CHAPTER_BEGIN ch_002>>>\n## B\nLong prose body for chapter B here.\n<<<CHAPTER_END>>>\n"
    )
    out = mod._split_batched_output(raw, ["ch_001", "ch_002"], section_titles={"ch_001":"A","ch_002":"B"})
    assert "prose body for chapter A" in out["ch_001"]
    assert "prose body for chapter B" in out["ch_002"]
    assert "preamble" not in out["ch_001"]
