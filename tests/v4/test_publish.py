"""Tests for engine_v4.publish (no LLM, no network)."""
from __future__ import annotations

from pathlib import Path

from engine_v4.publish import publish
from engine_v4.types import (
    Chapter, Outline, OutlineSection, PipelineState, Source, SourceSpec,
)


def _make_state(tmp_path: Path) -> PipelineState:
    spec = SourceSpec(location="memory://demo", kind="text", title="Demo Lecture")
    src_ws = tmp_path / "sources" / "demo"
    src_ws.mkdir(parents=True, exist_ok=True)
    source = Source(
        source_id="demo_0000000000",
        spec=spec,
        workspace=src_ws,
        media_path=None,
        duration_sec=0.0,
        meta={"url": "https://example.com/demo"},
    )
    outline = Outline(
        sections=[
            OutlineSection(id="ch1", title="Intro"),
            OutlineSection(id="ch2", title="Conclusion"),
        ],
        target_count=2,
    )
    chapters = [
        Chapter(id="ch2", title="Conclusion",
                body_md="## Conclusion\n\nWrap-up text.\n", approved=True),
        Chapter(id="ch1", title="Intro",
                body_md="## Intro\n\nHello world.\n", approved=True),
    ]
    return PipelineState(spec=spec, run_workspace=tmp_path,
                         source=source, outline=outline, chapters=chapters)


def test_publish_writes_textbook_md(tmp_path: Path):
    state = _make_state(tmp_path)
    textbook_md, epub, html, pdf = publish([state], tmp_path)

    assert textbook_md.exists()
    text = textbook_md.read_text(encoding="utf-8")
    assert "title: 'Demo Lecture'" in text
    assert "Multimodal Textbook Engine v4" in text
    assert "# Source: Demo Lecture" in text
    assert "demo_0000000000" in text
    # Outline order: ch1 before ch2 even though chapters list was reversed.
    assert text.index("Hello world") < text.index("Wrap-up text")

    # At least one of epub/html should succeed (pandoc is installed in dev env).
    # If pandoc is missing entirely, both will be None — accept that gracefully.
    import shutil
    if shutil.which("pandoc"):
        assert (epub is not None) or (html is not None)


def test_publish_handles_empty_state(tmp_path: Path):
    spec = SourceSpec(location="memory://empty", kind="text", title="Empty")
    state = PipelineState(spec=spec, run_workspace=tmp_path)
    textbook_md, _, _, _ = publish([state], tmp_path)
    assert textbook_md.exists()
    txt = textbook_md.read_text(encoding="utf-8")
    assert "no chapters generated" in txt
