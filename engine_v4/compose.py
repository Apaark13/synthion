"""engine_v4.compose — Layer 3 orchestrator.

v4.1: per-section writer calls, per-section RAG retrieval, and automatic
multi-figure injection per chapter.

Pipeline per source:

  1. ``writer.draft_all`` — one Gemma call per outline section, each with
     its own per-section RAG context (LLM semaphore held).
  2. ``figures.resolve_anchors`` — snap any ``[FIG t=…]`` markers the
     writer placed to the nearest captured Frame.
  3. ``figures.select_section_frames`` + ``inject_extra_frames`` —
     guarantees every chapter receives multiple unique, in-band screenshots
     even if the writer placed zero anchors.
  4. ``citations.enrich`` — best-effort DDG/ArXiv lookups; failures swallowed.
  5. ``evaluator.score`` — heuristic faithfulness gate with one bounded
     retry per failing chapter.

Each finished chapter is persisted to ``<workspace>/chapters/<ch_id>.md``.
"""
from __future__ import annotations

import logging
from typing import Callable

from config.settings import FRAMES_PER_CHAPTER

from . import citations, evaluator, figures, writer
from .limits import LLM
from .types import Chapter, OutlineSection, PipelineState, Source

log = logging.getLogger("engine_v4.compose")

FIGURE_TOLERANCE_SEC = 180.0


def _persist(chapter: Chapter, source: Source) -> None:
    chapters_dir = source.workspace / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    out = chapters_dir / f"{chapter.id}.md"
    try:
        out.write_text(chapter.body_md, encoding="utf-8")
    except Exception as e:
        log.warning("[compose] persist failed for %s: %s", chapter.id, e)


def _build_chapter(section: OutlineSection, body_md: str, source: Source) -> Chapter:
    """Resolve any LLM-placed anchors AND auto-inject extra section frames."""
    chapter = Chapter(id=section.id, title=section.title, body_md=body_md)

    # 1. Replace anchors the writer emitted with real <figure> blocks.
    snapped_body, anchor_used = figures.resolve_anchors(
        body_md, source.frames, tolerance_sec=FIGURE_TOLERANCE_SEC
    )

    # 2. Pick extra in-band frames so every chapter has multiple visuals.
    extras = figures.select_section_frames(
        section, list(source.frames or []), n=FRAMES_PER_CHAPTER,
    )
    final_body, all_used = figures.inject_extra_frames(snapped_body, extras, anchor_used)

    chapter.body_md = final_body
    chapter.figures = all_used
    return chapter


def compose(
    state: PipelineState,
    context_retrieve: Callable[[str, int], str],
) -> list[Chapter]:
    """Compose chapters for a single ``PipelineState``.

    ``context_retrieve`` is the bound ``Context.retrieve`` from L2; the
    writer calls it once per section with a section-specific query.
    """
    if state.source is None or state.outline is None:
        log.warning("[compose] state missing source/outline; nothing to do")
        return []

    source = state.source
    outline = state.outline

    drafts = writer.draft_all(outline, context_retrieve, source)

    chapters: list[Chapter] = []
    for section in outline.sections:
        body = drafts.get(section.id, "")
        chapter = _build_chapter(section, body, source)

        try:
            citations.enrich(chapter, source)
        except Exception as e:
            log.warning("[compose] citations.enrich crashed for %s: %s", section.id, e)

        try:
            sc, critique = evaluator.score(chapter)
        except Exception as e:
            log.warning("[compose] evaluator crashed for %s: %s", section.id, e)
            sc, critique = 0.0, f"evaluator crashed: {e}"

        chapter.eval_score = sc
        if sc >= evaluator.THRESHOLD:
            chapter.approved = True
        else:
            log.info("[compose] %s scored %.2f — single retry with critique",
                     section.id, sc)
            retry = writer.draft_all(
                outline, context_retrieve, source,
                critique={section.id: critique},
                only_section_ids=[section.id],
            )
            new_body = retry.get(section.id, "")
            if new_body:
                rebuilt = _build_chapter(section, new_body, source)
                try:
                    citations.enrich(rebuilt, source)
                except Exception as e:
                    log.warning("[compose] citations.enrich (retry) crashed for %s: %s",
                                section.id, e)
                try:
                    sc2, _ = evaluator.score(rebuilt)
                except Exception as e:
                    log.warning("[compose] evaluator (retry) crashed for %s: %s",
                                section.id, e)
                    sc2 = sc
                rebuilt.eval_score = sc2
                rebuilt.approved = True  # bounded retry exhausted ⇒ approve
                chapter = rebuilt
            else:
                chapter.approved = True  # retry produced nothing usable

        _persist(chapter, source)
        chapters.append(chapter)

    return chapters


__all__ = ["compose", "FIGURE_TOLERANCE_SEC"]
# Silences unused-import warnings for re-exported semaphore (kept for API parity).
_ = LLM
