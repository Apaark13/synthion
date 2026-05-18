"""engine_v4.publish — Layer 4: assemble chapters and render textbook outputs.

v4.1: HTML is the **primary, default** output. The bundle still produces
EPUB and (best-effort) PDF, but the HTML build:

* runs first, so it never fails because of pandoc-EPUB quirks;
* uses a standalone-with-CSS render so the textbook is readable in a
  browser without any extra files;
* preserves the inline ``<figure>`` blocks the writer injected via
  ``engine_v4.figures`` so screenshots actually render.

We still emit a ``textbook.md`` for archival / debugging.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .types import Chapter, Frame, PipelineState


# ── Figure helpers (legacy fallback path for any leftover anchors) ───────────

def _figure_md(fig: dict) -> str:
    path = fig.get("path", "")
    caption = (fig.get("caption", "") or "").strip()
    ts = fig.get("timestamp_sec", 0)
    if path and Path(path).exists():
        cap = caption or "Figure"
        return (
            f'\n\n<figure class="mte-figure">\n'
            f'  <img src="{path}" alt="{cap}" />\n'
            f'  <figcaption><strong>t={int(ts)}s.</strong> {cap}</figcaption>\n'
            f'</figure>\n'
        )
    return ""


def _ts_to_sec(ts_str: str) -> int:
    try:
        parts = ts_str.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def _resolve_legacy_anchors(text: str, manifest: list[dict]) -> str:
    """Catch any [FIG t=…] anchor that escaped the compose-time resolver."""
    def _replace(ts_str: str, original: str) -> str:
        ts_sec = _ts_to_sec(ts_str)
        if manifest:
            closest = min(manifest, key=lambda f: abs(f.get("timestamp_sec", 0) - ts_sec))
            if abs(closest.get("timestamp_sec", 0) - ts_sec) < 300:
                return _figure_md(closest)
        return ""

    text = re.sub(
        r"\[FIG\s+t=([0-9:]+)\]",
        lambda m: _replace(m.group(1), m.group(0)),
        text,
    )
    text = re.sub(
        r"\[Figure:\s*(?:tim[a-z]*timestamp|timestamp)=([0-9:]+)[^\]]*\]",
        lambda m: _replace(m.group(1), m.group(0)),
        text,
    )
    return text


def _frame_to_dict(frame: Frame, index: int) -> dict:
    return {
        "path": str(frame.path),
        "caption": frame.caption or "",
        "timestamp_sec": float(frame.timestamp_sec or 0),
        "index": index,
    }


# ── Chapter ordering ─────────────────────────────────────────────────────────

def _order_chapters(state: PipelineState) -> list[Chapter]:
    chapters = list(state.chapters)
    if not chapters:
        return []
    if state.outline and state.outline.sections:
        order = [s.id for s in state.outline.sections]
        by_id: dict[str, Chapter] = {ch.id: ch for ch in chapters}
        ordered: list[Chapter] = []
        used: set[str] = set()
        for sid in order:
            ch = by_id.get(sid)
            if ch is not None:
                ordered.append(ch)
                used.add(ch.id)
        for ch in chapters:
            if ch.id not in used:
                ordered.append(ch)
        return ordered
    return chapters


def _source_heading(state: PipelineState) -> str:
    src = state.source
    spec = state.spec
    title = (spec.title if spec and spec.title else None)
    if not title and src is not None:
        title = src.meta.get("title") if isinstance(src.meta, dict) else None
    if not title:
        title = (spec.location if spec else "Untitled")
    sid = src.source_id if src else "(no source)"
    url = ""
    if src and isinstance(src.meta, dict):
        url = src.meta.get("url") or src.meta.get("source_url") or ""
    if not url and spec and (spec.location.startswith("http://") or spec.location.startswith("https://")):
        url = spec.location

    lines = [f"# Source: {title}", "", f"*source_id: `{sid}`*"]
    if url:
        lines.append("  ")
        lines.append(f"*url: <{url}>*")
    lines.append("")
    return "\n".join(lines)


def _source_subheading(state: PipelineState) -> str:
    """Return a ## sub-heading line for a source within a chapter group."""
    src = state.source
    spec = state.spec
    title = (spec.title if spec and spec.title else None)
    if not title and src is not None:
        title = src.meta.get("title") if isinstance(src.meta, dict) else None
    if not title:
        title = (spec.location if spec else "Untitled")
    url = ""
    if src and isinstance(src.meta, dict):
        url = src.meta.get("source_url") or src.meta.get("url") or ""
    if not url and spec and (
        spec.location.startswith("http://") or spec.location.startswith("https://")
    ):
        url = spec.location

    lines = [f"## {title}"]
    if url:
        lines.append(f"*<{url}>*")
    lines.append("")
    return "\n".join(lines)


# ── Assembly ─────────────────────────────────────────────────────────────────

def _assemble_markdown(
    states: list[PipelineState],
    book_dir: Path,
    chapter_groups: Optional[list[dict[str, Any]]] = None,
) -> Path:
    book_dir.mkdir(parents=True, exist_ok=True)
    out_path = book_dir / "textbook.md"

    title = "Multimodal Textbook"
    for st in states:
        t = (st.spec.title if st.spec else None) or (
            st.source.meta.get("title") if st.source and isinstance(st.source.meta, dict) else None
        )
        if t:
            title = t
            break
    # YAML-safe quoting: single-quote and escape any embedded single quotes.
    safe_title = "'" + title.replace("'", "''") + "'"

    parts: list[str] = [
        "---",
        f"title: {safe_title}",
        "author: Multimodal Textbook Engine v4",
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "---",
        "",
    ]

    def _emit_state(
        state: PipelineState,
        use_subheading: bool = False,
        skip_heading: bool = False,
    ) -> None:
        """Append one source's chapters to ``parts``.

        use_subheading: emit a ## subheading instead of a # Source heading.
        skip_heading:   emit no heading at all (for single-source chapter groups).
        """
        ordered = _order_chapters(state)
        manifest: list[dict] = []
        if state.source and state.source.frames:
            manifest = [_frame_to_dict(f, i) for i, f in enumerate(state.source.frames)]
        chapter_fig_offset = len(manifest)
        for ch in ordered:
            for f in ch.figures or []:
                manifest.append(_frame_to_dict(f, chapter_fig_offset))
                chapter_fig_offset += 1

        if not skip_heading:
            if use_subheading:
                parts.append(_source_subheading(state))
            else:
                parts.append(_source_heading(state))
            parts.append("")

        if not ordered:
            parts.append("*(no chapters generated for this source)*")
            parts.append("")
        for ch in ordered:
            body = _resolve_legacy_anchors(ch.body_md or "", manifest)
            parts.append(body)
            parts.append("")
            parts.append("---")
            parts.append("")

    if chapter_groups:
        # Grouped mode: each group gets a # Chapter heading.
        # Multiple sources within a group → ## subheadings per source.
        # Single source within a group → no source heading (chapter H1 suffices).
        states_by_idx = {i: s for i, s in enumerate(states)}
        for group_num, group in enumerate(chapter_groups, 1):
            chapter_title = group.get("chapter_title", f"Chapter {group_num}")
            indices = group.get("state_indices", [])
            parts.append(f"# {chapter_title}")
            parts.append("")
            multi_source = len(indices) > 1
            for idx in indices:
                state = states_by_idx.get(idx)
                if state is None:
                    continue
                _emit_state(
                    state,
                    use_subheading=multi_source,
                    skip_heading=(not multi_source),
                )
    else:
        # Default mode: each source gets a # Source heading.
        for state in states:
            _emit_state(state, use_subheading=False)

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


# ── HTML styling ─────────────────────────────────────────────────────────────

_HTML_CSS = """
:root { --fg:#1c1c1c; --muted:#666; --accent:#0b5fae; --bg:#fdfdfb; --rule:#e6e2d6; }
html, body { background: var(--bg); color: var(--fg); }
body {
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", "Segoe UI", Roboto, sans-serif;
  font-size: 16.5px; line-height: 1.65;
  max-width: 820px; margin: 2.2rem auto; padding: 0 1.25rem;
}
h1, h2, h3, h4 { font-family: "Georgia", "Iowan Old Style", serif; color: var(--fg); line-height: 1.25; }
h1 { font-size: 2.1rem; border-bottom: 2px solid var(--accent); padding-bottom: .25rem; margin-top: 2.4rem; }
h2 { font-size: 1.55rem; margin-top: 2.2rem; color: var(--accent); }
h3 { font-size: 1.18rem; margin-top: 1.6rem; }
p { margin: .65rem 0; }
hr { border: 0; border-top: 1px dashed var(--rule); margin: 2.5rem 0; }
blockquote { border-left: 4px solid var(--accent); padding: .25rem 1rem; color: var(--muted); margin: 1rem 0; background: #f5f3ec; }
code { background: #f1efe7; padding: .12em .35em; border-radius: 3px; font-size: 95%; }
pre code { display: block; padding: .8rem 1rem; overflow-x: auto; }
a { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(11,95,174,.3); }
a:hover { border-bottom-color: var(--accent); }
nav#TOC { background: #f5f3ec; border: 1px solid var(--rule); border-radius: 6px; padding: 1rem 1.25rem; margin-bottom: 2rem; }
nav#TOC > ul { padding-left: 1.1rem; margin: 0; }
figure.mte-figure { margin: 1.6rem 0; text-align: center; }
figure.mte-figure img { max-width: 100%; height: auto; border: 1px solid var(--rule); border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
figure.mte-figure figcaption { color: var(--muted); font-size: 0.92em; margin-top: .35rem; font-style: italic; }
table { border-collapse: collapse; margin: 1.2rem 0; }
th, td { border: 1px solid var(--rule); padding: .35rem .6rem; }
em em, strong em { font-style: italic; }
""".strip()


def _write_html_assets(out_dir: Path) -> Path:
    css_path = out_dir / "textbook.css"
    css_path.write_text(_HTML_CSS, encoding="utf-8")
    return css_path


# ── Pandoc rendering ─────────────────────────────────────────────────────────

def _run_pandoc(cmd: list[str], timeout: int = 180) -> bool:
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def _render_html(textbook_md: Path, out_dir: Path) -> Optional[Path]:
    out = out_dir / "textbook.html"
    css = _write_html_assets(out_dir)
    ok = _run_pandoc([
        "pandoc", str(textbook_md),
        "--standalone", "--toc", "--toc-depth=3",
        "--number-sections",
        "--highlight-style=tango",
        "--metadata", "lang=en",
        "--mathjax",
        "--css", css.name,
        "--embed-resources" if _pandoc_supports("--embed-resources") else "--self-contained",
        "-o", str(out),
    ])
    if ok and out.exists():
        return out
    # Retry without embed if pandoc choked on it (very old versions).
    ok = _run_pandoc([
        "pandoc", str(textbook_md),
        "--standalone", "--toc", "--toc-depth=3",
        "--number-sections", "--highlight-style=tango",
        "--metadata", "lang=en", "--mathjax",
        "--css", css.name,
        "-o", str(out),
    ])
    return out if (ok and out.exists()) else None


def _pandoc_supports(flag: str) -> bool:
    try:
        out = subprocess.run(
            ["pandoc", "--help"], capture_output=True, text=True, timeout=5,
        )
        return flag in (out.stdout or "")
    except Exception:
        return False


def _render_epub(textbook_md: Path, out_dir: Path) -> Optional[Path]:
    out = out_dir / "textbook.epub"
    ok = _run_pandoc([
        "pandoc", str(textbook_md),
        "--toc", "--epub-chapter-level=2",
        "-o", str(out),
    ])
    return out if (ok and out.exists()) else None


def _render_pdf(textbook_md: Path, out_dir: Path) -> Optional[Path]:
    out = out_dir / "textbook.pdf"
    ok = _run_pandoc([
        "pandoc", str(textbook_md),
        "--pdf-engine=xelatex",
        "--toc", "--number-sections",
        "--highlight-style=tango",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=DejaVu Serif",
        "-V", "monofont=DejaVu Sans Mono",
        "-o", str(out),
    ], timeout=300)
    return out if (ok and out.exists()) else None


# ── Public API ───────────────────────────────────────────────────────────────

def publish(
    states: list[PipelineState],
    run_workspace: Path,
    *,
    chapter_groups: Optional[list[dict[str, Any]]] = None,
) -> tuple[Path, Optional[Path], Optional[Path], Optional[Path]]:
    """Assemble + render. Returns (textbook_md, epub, html, pdf).

    HTML is the **default/primary** output and is generated first.
    When ``chapter_groups`` is provided the textbook is structured into
    top-level chapter headings instead of per-source headings.
    """
    book_dir = run_workspace / "book"
    out_dir = run_workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    textbook_md = _assemble_markdown(states, book_dir, chapter_groups=chapter_groups)
    html = _render_html(textbook_md, out_dir)        # primary
    epub = _render_epub(textbook_md, out_dir)        # secondary
    pdf = _render_pdf(textbook_md, out_dir)          # best-effort
    return textbook_md, epub, html, pdf


def append_chapters(
    run_workspace: Path,
    new_states: list[PipelineState],
    chapter_title: Optional[str] = None,
) -> tuple[Path, Optional[Path], Optional[Path], Optional[Path]]:
    """Append new chapter(s) to an existing textbook and re-render all formats.

    If no textbook.md exists yet, delegates to a fresh ``publish`` call.
    The TOC is automatically regenerated by pandoc from the full heading tree.
    """
    book_dir = run_workspace / "book"
    out_dir = run_workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_md = book_dir / "textbook.md"
    if not existing_md.exists():
        return publish(new_states, run_workspace)

    existing_text = existing_md.read_text(encoding="utf-8")

    # Build the new content to append.
    new_parts: list[str] = [""]  # blank line before new content
    if chapter_title:
        new_parts.append(f"# {chapter_title}")
        new_parts.append("")

    for state in new_states:
        ordered = _order_chapters(state)
        manifest: list[dict] = []
        if state.source and state.source.frames:
            manifest = [_frame_to_dict(f, i) for i, f in enumerate(state.source.frames)]
        chapter_fig_offset = len(manifest)
        for ch in ordered:
            for f in ch.figures or []:
                manifest.append(_frame_to_dict(f, chapter_fig_offset))
                chapter_fig_offset += 1

        # Use sub-heading when inside a named chapter, source heading otherwise.
        if chapter_title:
            new_parts.append(_source_subheading(state))
        else:
            new_parts.append(_source_heading(state))
        new_parts.append("")

        if not ordered:
            new_parts.append("*(no chapters generated for this source)*")
            new_parts.append("")
        else:
            for ch in ordered:
                body = _resolve_legacy_anchors(ch.body_md or "", manifest)
                new_parts.append(body)
                new_parts.append("")
                new_parts.append("---")
                new_parts.append("")

    combined = existing_text.rstrip() + "\n" + "\n".join(new_parts)
    existing_md.write_text(combined, encoding="utf-8")

    html = _render_html(existing_md, out_dir)
    epub = _render_epub(existing_md, out_dir)
    pdf = _render_pdf(existing_md, out_dir)
    return existing_md, epub, html, pdf
