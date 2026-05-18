"""engine_v4.runner — top-level orchestration of the v4 pipeline.

Uses ThreadPoolExecutor (not asyncio) so we can keep llama-cpp happy and reuse
the v3 ffmpeg subprocess code unchanged. The LLM is single-slot (see
``engine_v4.limits.LLM`` and ``engine_v4.llm.LLM_LOCK``); even with
``max_concurrency > 1`` writer/eval calls serialise. Per-source ingest +
embedding work however can overlap, and ``plan_outline`` and ``build_context``
run concurrently for the same source via a small inner pool.
"""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import BuildResult, PipelineState, SourceSpec
from .workspace import REPO_ROOT, new_run_id, run_workspace as _make_run_workspace

ERRORS_LOG = REPO_ROOT / "ERRORS.log"


def _log_error(step: str, exc: BaseException) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    msg = f"[{ts}] [v4 runner] {step}: {exc!r}\n{traceback.format_exc()}\n"
    try:
        with open(ERRORS_LOG, "a", encoding="utf-8") as f:
            f.write(msg)
    except Exception:
        pass


def _validate_spec(spec: SourceSpec) -> None:
    """Cheap upfront validation — surfaces obvious user mistakes cleanly."""
    if spec is None or not isinstance(spec, SourceSpec):
        raise TypeError(f"expected SourceSpec, got {type(spec).__name__}")
    loc = (spec.location or "").strip()
    if not loc:
        raise ValueError("SourceSpec.location is empty")
    is_url = loc.startswith(("http://", "https://"))
    if is_url:
        return
    # Local path: file must exist (kind=auto/text/pdf/audio/video).
    p = Path(loc)
    if not p.exists():
        raise FileNotFoundError(f"source not found: {loc}")


def _process_one(spec: SourceSpec, run_ws: Path) -> PipelineState:
    """Execute layers 1–3 for a single source. Errors are absorbed and
    surfaced as an empty PipelineState placeholder.
    """
    state = PipelineState(spec=spec, run_workspace=run_ws)
    try:
        # Lazy imports — sibling modules may not exist at import time and we
        # want a pristine ImportError -> ERRORS.log path rather than crashing
        # the whole runner module.
        from . import ingest, understand, compose  # type: ignore
    except Exception as e:
        _log_error("import sibling layers", e)
        return state

    try:
        state.source = ingest.ingest(spec, run_ws)
    except Exception as e:
        _log_error(f"ingest({spec.location})", e)
        return state

    try:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="v4-understand") as pool:
            outline_future = pool.submit(understand.plan_outline, state.source)
            context_future = pool.submit(understand.build_context, state.source)
            try:
                state.outline = outline_future.result()
            except Exception as e:
                _log_error(f"plan_outline({spec.location})", e)
                state.outline = None
            try:
                ctx = context_future.result()
            except Exception as e:
                _log_error(f"build_context({spec.location})", e)
                ctx = None
    except Exception as e:
        _log_error(f"understand pool({spec.location})", e)
        return state

    if state.outline is None or ctx is None:
        return state

    try:
        state.chapters = compose.compose(state, ctx.retrieve)
    except Exception as e:
        _log_error(f"compose({spec.location})", e)
        state.chapters = []
    return state


def _source_title_from_state(state: PipelineState) -> Optional[str]:
    """Extract the best human-readable title for a source state."""
    if state.spec and state.spec.title:
        return state.spec.title
    if state.source and isinstance(state.source.meta, dict):
        t = state.source.meta.get("title")
        if t:
            return t
    return None


def _extract_textbook_title(
    states: list[PipelineState],
    book_title: Optional[str] = None,
) -> str:
    """Return the best title for the assembled textbook."""
    if book_title:
        return book_title
    for s in states:
        t = _source_title_from_state(s)
        if t:
            return t
    return "Multimodal Textbook"


def _build_result(
    run_id: str,
    run_ws: Path,
    states: list[PipelineState],
    started_at: float,
    *,
    chapter_groups: Optional[list[dict[str, Any]]] = None,
    book_title: Optional[str] = None,
) -> BuildResult:
    from . import publish as publish_mod

    textbook_md: Optional[Path] = None
    epub = html = pdf = None
    publish_error: Optional[str] = None
    try:
        textbook_md, epub, html, pdf = publish_mod.publish(
            states, run_ws, chapter_groups=chapter_groups
        )
    except Exception as e:
        _log_error("publish", e)
        publish_error = repr(e)

    elapsed = time.monotonic() - started_at
    textbook_title = _extract_textbook_title(states, book_title)

    summary: dict[str, Any] = {
        "run_id": run_id,
        "workspace": str(run_ws),
        "textbook_title": textbook_title,
        "started_at_utc": datetime.fromtimestamp(
            time.time() - elapsed, tz=timezone.utc
        ).isoformat(),
        "elapsed_sec": round(elapsed, 3),
        "sources_processed": len(states),
        "chapters_total": sum(len(s.chapters) for s in states),
        "sources": [
            {
                "location": s.spec.location,
                "kind": s.spec.kind,
                "title": _source_title_from_state(s),
                "source_id": s.source.source_id if s.source else None,
                "chapters": len(s.chapters),
                "outline_sections": (len(s.outline.sections) if s.outline else 0),
            }
            for s in states
        ],
        "outputs": {
            "textbook_md": str(textbook_md) if textbook_md else None,
            "epub": str(epub) if epub else None,
            "html": str(html) if html else None,
            "pdf": str(pdf) if pdf else None,
        },
    }
    if chapter_groups:
        summary["chapter_groups"] = chapter_groups
    if publish_error:
        summary["publish_error"] = publish_error

    try:
        (run_ws / "run.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    except Exception as e:
        _log_error("write run.json", e)

    return BuildResult(
        run_id=run_id,
        workspace=run_ws,
        states=states,
        textbook_md=textbook_md,
        epub=epub,
        html=html,
        pdf=pdf,
        chapter_groups=chapter_groups or [],
    )


# ── Public API ───────────────────────────────────────────────────────────────

def run(spec: SourceSpec, *, run_id: Optional[str] = None) -> BuildResult:
    """Build a single-source textbook."""
    _validate_spec(spec)
    rid = run_id or new_run_id()
    run_ws = _make_run_workspace(rid)
    started = time.monotonic()
    state = _process_one(spec, run_ws)
    return _build_result(rid, run_ws, [state], started)


def run_many(
    specs: list[SourceSpec],
    *,
    run_id: Optional[str] = None,
    max_concurrency: int = 2,
) -> BuildResult:
    """Build a textbook from many sources. One source's failure does not kill
    the others — its slot becomes a placeholder PipelineState with no
    chapters.
    """
    if not isinstance(specs, list):
        raise TypeError("specs must be a list of SourceSpec")
    # Validate eagerly but don't abort the whole batch on a single bad spec —
    # however, type errors *should* surface immediately.
    for s in specs:
        if not isinstance(s, SourceSpec):
            raise TypeError(f"specs contains non-SourceSpec: {type(s).__name__}")

    rid = run_id or new_run_id()
    run_ws = _make_run_workspace(rid)
    started = time.monotonic()

    states: list[PipelineState] = [None] * len(specs)  # type: ignore[list-item]
    workers = max(1, int(max_concurrency))

    def _safe_process(idx: int, spec: SourceSpec) -> tuple[int, PipelineState]:
        try:
            _validate_spec(spec)
            return idx, _process_one(spec, run_ws)
        except Exception as e:
            _log_error(f"run_many spec[{idx}] {spec.location!r}", e)
            return idx, PipelineState(spec=spec, run_workspace=run_ws)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v4-source") as pool:
        for idx, st in pool.map(lambda p: _safe_process(*p), list(enumerate(specs))):
            states[idx] = st

    return _build_result(rid, run_ws, states, started)


def run_playlist_as_book(
    groups: list[tuple[str, list[SourceSpec]]],
    *,
    run_id: Optional[str] = None,
    max_concurrency: int = 2,
    book_title: Optional[str] = None,
) -> BuildResult:
    """Build a single textbook from playlist chapter groups.

    groups: [(chapter_title, [SourceSpec, ...]), ...]

    Each group maps to a top-level chapter heading in the textbook.
    Sources within a group appear as sub-sections of that chapter.
    """
    # Flatten specs in order, tracking which group each belongs to.
    all_specs: list[SourceSpec] = []
    chapter_groups_meta: list[dict[str, Any]] = []

    for chapter_title, specs in groups:
        if not specs:
            continue
        start_idx = len(all_specs)
        all_specs.extend(specs)
        chapter_groups_meta.append({
            "chapter_title": chapter_title,
            "state_indices": list(range(start_idx, len(all_specs))),
        })

    if not all_specs:
        raise ValueError("No SourceSpecs provided in any group")

    rid = run_id or new_run_id()
    run_ws = _make_run_workspace(rid)
    started = time.monotonic()

    states: list[PipelineState] = [None] * len(all_specs)  # type: ignore[list-item]
    workers = max(1, int(max_concurrency))

    def _safe_process(idx: int, spec: SourceSpec) -> tuple[int, PipelineState]:
        try:
            _validate_spec(spec)
            return idx, _process_one(spec, run_ws)
        except Exception as e:
            _log_error(f"run_playlist_as_book spec[{idx}] {spec.location!r}", e)
            return idx, PipelineState(spec=spec, run_workspace=run_ws)

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="v4-playlist") as pool:
        for idx, st in pool.map(
            lambda p: _safe_process(*p), list(enumerate(all_specs))
        ):
            states[idx] = st

    return _build_result(
        rid, run_ws, states, started,
        chapter_groups=chapter_groups_meta,
        book_title=book_title,
    )


def append_chapter_to_run(
    run_ws_path: Path,
    new_specs: list[SourceSpec],
    chapter_title: Optional[str] = None,
    *,
    max_concurrency: int = 2,
) -> BuildResult:
    """Process new source(s) and append them as a new chapter to an existing
    textbook. The existing textbook.md is preserved; new content is appended
    and all formats are re-rendered.

    Returns a BuildResult representing the updated textbook.
    """
    run_json_path = run_ws_path / "run.json"
    if not run_json_path.exists():
        raise FileNotFoundError(f"run.json not found in {run_ws_path}")

    existing = json.loads(run_json_path.read_text(encoding="utf-8"))
    run_id = existing.get("run_id", run_ws_path.name)

    started = time.monotonic()
    workers = max(1, int(max_concurrency))
    new_states: list[PipelineState] = []

    for spec in new_specs:
        try:
            _validate_spec(spec)
            st = _process_one(spec, run_ws_path)
        except Exception as e:
            _log_error(f"append_chapter spec {spec.location!r}", e)
            st = PipelineState(spec=spec, run_workspace=run_ws_path)
        new_states.append(st)

    from . import publish as publish_mod
    textbook_md, epub, html, pdf = publish_mod.append_chapters(
        run_ws_path, new_states, chapter_title
    )

    elapsed = time.monotonic() - started

    # Update run.json in-place.
    existing["chapters_total"] = (
        existing.get("chapters_total", 0) + sum(len(s.chapters) for s in new_states)
    )
    existing["sources_processed"] = (
        existing.get("sources_processed", 0) + len(new_states)
    )
    for s in new_states:
        existing.setdefault("sources", []).append({
            "location": s.spec.location,
            "kind": s.spec.kind,
            "title": _source_title_from_state(s),
            "source_id": s.source.source_id if s.source else None,
            "chapters": len(s.chapters),
            "outline_sections": (len(s.outline.sections) if s.outline else 0),
            "appended": True,
        })
    if chapter_title:
        existing.setdefault("chapter_groups", []).append({
            "chapter_title": chapter_title,
            "appended": True,
        })
    existing["outputs"] = {
        "textbook_md": str(textbook_md) if textbook_md else None,
        "epub": str(epub) if epub else None,
        "html": str(html) if html else None,
        "pdf": str(pdf) if pdf else None,
    }
    existing["last_appended_utc"] = datetime.now(timezone.utc).isoformat()
    run_json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return BuildResult(
        run_id=run_id,
        workspace=run_ws_path,
        states=new_states,
        textbook_md=textbook_md,
        epub=epub,
        html=html,
        pdf=pdf,
    )
