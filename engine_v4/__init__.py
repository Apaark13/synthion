"""engine_v4 — simplified, parallel multimodal textbook pipeline.

Public entry: ``engine_v4.run(spec)`` for one Source, ``engine_v4.run_many``
for a playlist / multi-source build.

Architecture (v4):

    L1 INGEST     -> Source(video, transcript, frames, meta)
    L2 UNDERSTAND -> Outline + Context
    L3 COMPOSE    -> Chapters[] (writer + citation + evaluator + figure resolver)
    L4 PUBLISH    -> textbook.md + epub/html (PDF when xelatex present)

Coexists with the v3 ``agents/`` package; nothing in ``agents/`` is touched.
"""
from __future__ import annotations

from .types import (
    SourceSpec,
    Source,
    Frame,
    Outline,
    OutlineSection,
    Chapter,
    BuildResult,
    PipelineState,
)
from .runner import run, run_many, run_playlist_as_book, append_chapter_to_run

__all__ = [
    "SourceSpec", "Source", "Frame", "Outline", "OutlineSection",
    "Chapter", "BuildResult", "PipelineState",
    "run", "run_many", "run_playlist_as_book", "append_chapter_to_run",
]
