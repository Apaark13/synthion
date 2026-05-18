"""Tests for engine_v4.writer (v4.1 per-section rewrite)."""
from engine_v4.writer import (
    _BODY_OPEN, _BODY_CLOSE, _LEGACY_BEGIN_TAG, _LEGACY_END_TAG,
    _sanitise, _section_query, draft_all,
)
from engine_v4.types import Outline, OutlineSection, Source, SourceSpec


def _mk_source() -> Source:
    spec = SourceSpec(location="x.mp4", kind="video")
    return Source(
        source_id="s", spec=spec, workspace=None,  # type: ignore[arg-type]
        media_path=None, duration_sec=600.0,
    )


def test_sanitise_strips_body_markers_and_meta():
    raw = (
        f"{_BODY_OPEN}\n"
        "## Real Title\n"
        "Body line 1.\n"
        "*(Self-Correction: I should rephrase)*\n"
        "Body line 2.\n"
        "=== INPUT ===\n"
        "Title: leaking\n"
        "Body line 3.\n"
        f"{_BODY_CLOSE}\n"
    )
    cleaned = _sanitise(raw, "Real Title")
    assert _BODY_OPEN not in cleaned
    assert _BODY_CLOSE not in cleaned
    assert "Self-Correction" not in cleaned
    assert "=== INPUT" not in cleaned
    assert "Title: leaking" not in cleaned
    assert cleaned.startswith("## ")
    assert "Body line 1." in cleaned
    assert "Body line 3." in cleaned


def test_sanitise_handles_legacy_tags():
    raw = (
        f"{_LEGACY_BEGIN_TAG} ch_001>>>\n"
        "## A title\n"
        "Body.\n"
        f"{_LEGACY_END_TAG}"
    )
    cleaned = _sanitise(raw, "A title")
    assert _LEGACY_BEGIN_TAG not in cleaned
    assert _LEGACY_END_TAG not in cleaned
    assert "## A title" in cleaned


def test_sanitise_drops_garbage_preamble_before_first_heading():
    # Long, multi-line preamble that smells like spec echo should be dropped.
    raw = (
        "Title: Echoed\n"
        "Bloom level: understand\n"
        "Key terms: a, b, c\n"
        "Some rambling pre-amble that the model added.\n"
        "More rambling.\n"
        "## The Actual Heading\n"
        "Real body here.\n"
    )
    cleaned = _sanitise(raw, "The Actual Heading")
    assert "Title: Echoed" not in cleaned
    assert "Bloom level" not in cleaned
    assert "Key terms" not in cleaned
    assert cleaned.startswith("## ")


def test_sanitise_adds_heading_when_missing():
    cleaned = _sanitise("Just prose, no heading.", "My Title")
    assert cleaned.startswith("## My Title")


def test_section_query_combines_title_keyterms_bloom():
    sec = OutlineSection(
        id="ch_001", title="Eigenvalues", bloom_level="apply",
        key_terms=["matrix", "spectrum"],
    )
    q = _section_query(sec)
    assert "Eigenvalues" in q
    assert "matrix" in q
    assert "apply" in q


def test_draft_all_returns_placeholder_when_llm_unavailable(monkeypatch):
    # Force the writer's LLM call to return empty (simulating no model).
    import engine_v4.writer as w
    monkeypatch.setattr(w, "call_gemma", lambda *a, **k: "")

    outline = Outline(sections=[
        OutlineSection(id="ch_001", title="Alpha", timestamps=[0.0, 100.0]),
        OutlineSection(id="ch_002", title="Beta",  timestamps=[200.0, 300.0]),
    ], target_count=2)
    out = draft_all(outline, lambda q, k: "context", _mk_source())
    assert set(out.keys()) == {"ch_001", "ch_002"}
    assert all("Content pending" in v for v in out.values())


def test_draft_all_only_section_ids_filters(monkeypatch):
    import engine_v4.writer as w
    monkeypatch.setattr(w, "call_gemma",
                        lambda *a, **k: f"{_BODY_OPEN}\n## X\nBody.\n{_BODY_CLOSE}")

    outline = Outline(sections=[
        OutlineSection(id="ch_001", title="Alpha"),
        OutlineSection(id="ch_002", title="Beta"),
    ], target_count=2)
    out = draft_all(outline, lambda q, k: "ctx", _mk_source(),
                    only_section_ids=["ch_002"])
    assert set(out.keys()) == {"ch_002"}
