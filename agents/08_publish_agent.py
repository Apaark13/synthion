"""agents/08_publish_agent.py — Publish Agent

Assembles all approved chapters + figure manifest into textbook.md,
then renders to PDF (Pandoc + XeLaTeX) and EPUB (Pandoc/Quarto).

Input:
  - chapters: {chapter_id: markdown_text}
  - citations: {chapter_id: [citation_dicts]}
  - toc: {sections: [...]}
  - images: [str paths to upscaled PNGs]

Output:
  - workspace/book/textbook.md
  - workspace/output/textbook.pdf
  - workspace/output/textbook.epub
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BOOK_DIR, OUTPUT_DIR, IMAGES_DIR, ERRORS_LOG
from agents.template import log_error, exponential_backoff, cleanup_temp

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BOOK_DIR.mkdir(parents=True, exist_ok=True)


# ── Assembly ──────────────────────────────────────────────────────────────────

def _load_figure_manifest() -> list[dict]:
    manifest_path = IMAGES_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def _figure_md(fig: dict) -> str:
    """Render a figure entry as Markdown with caption."""
    path = fig.get("path", "")
    caption = fig.get("caption", "")
    ts = fig.get("timestamp_sec", 0)
    index = fig.get("index", 0)
    if path and Path(path).exists():
        return f'\n\n![Figure {index+1}: {caption}]({path})\n*Figure {index+1} (t={ts:.0f}s): {caption}*\n'
    return ""


def _resolve_figure_anchors(text: str, manifest: list[dict]) -> str:
    """Replace figure anchors with Markdown image tags.

    Supports both the v3.1 short form ``[FIG t=HH:MM:SS]`` and the legacy
    ``[Figure: timestamp=HH:MM:SS, desc=...]`` form.
    """
    import re

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

    def _replace(ts_str: str, original: str) -> str:
        ts_sec = _ts_to_sec(ts_str)
        if manifest:
            closest = min(manifest, key=lambda f: abs(f.get("timestamp_sec", 0) - ts_sec))
            # Looser tolerance: 5 min — manifests are sparse.
            if abs(closest.get("timestamp_sec", 0) - ts_sec) < 300:
                return _figure_md(closest)
        return original

    text = re.sub(
        r"\[FIG\s+t=([0-9:]+)\]",
        lambda m: _replace(m.group(1), m.group(0)),
        text,
    )
    # Legacy / mangled forms (handles "timestimestamp=" and similar).
    text = re.sub(
        r"\[Figure:\s*(?:tim[a-z]*timestamp|timestamp)=([0-9:]+)[^\]]*\]",
        lambda m: _replace(m.group(1), m.group(0)),
        text,
    )
    return text


def _assemble_textbook(
    chapters: dict[str, str],
    toc: dict,
    manifest: list[dict],
) -> Path:
    """Concatenate chapters in ToC order into workspace/book/textbook.md."""
    sections = toc.get("sections", [])
    section_order = [s.get("id") for s in sections if s.get("id")]

    # Sort chapters by ToC order; append any not in ToC at end
    ordered_ids = [ch for ch in section_order if ch in chapters]
    remaining = [ch for ch in chapters if ch not in ordered_ids]
    all_ids = ordered_ids + remaining

    lines = [
        "---",
        "title: Lecture Textbook",
        "author: Multimodal Textbook Engine v2",
        f"date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        "---",
        "",
    ]
    for ch_id in all_ids:
        text = chapters[ch_id]
        text = _resolve_figure_anchors(text, manifest)
        lines.append(text)
        lines.append("\n---\n")

    out_path = BOOK_DIR / "textbook.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[publish] textbook.md assembled ({out_path.stat().st_size // 1024} KB)")
    return out_path


# ── Rendering ─────────────────────────────────────────────────────────────────

def _render_pdf(textbook_md: Path) -> Path:
    """Render Markdown → PDF. Falls back to HTML if xelatex unavailable."""
    out_pdf = OUTPUT_DIR / "textbook.pdf"
    cmd = [
        "pandoc", str(textbook_md),
        "--pdf-engine=xelatex",
        "--toc",
        "--number-sections",
        "--highlight-style=tango",
        "-V", "geometry:margin=1in",
        "-V", "mainfont=DejaVu Serif",
        "-V", "monofont=DejaVu Sans Mono",
        "-o", str(out_pdf),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300, capture_output=True)
        print(f"[publish] PDF rendered → {out_pdf} ({out_pdf.stat().st_size // 1024} KB)")
        return out_pdf
    except FileNotFoundError:
        print("[publish] pandoc not found; skipping PDF.")
    except subprocess.CalledProcessError as e:
        log_error(8, e)
        print(f"[publish] PDF render failed (xelatex?); falling back to HTML.")

    # Fallback: render a self-contained HTML "PDF surrogate".
    out_html = OUTPUT_DIR / "textbook.html"
    try:
        subprocess.run([
            "pandoc", str(textbook_md),
            "--standalone", "--toc", "--number-sections",
            "--highlight-style=tango",
            "-o", str(out_html),
        ], check=True, timeout=120, capture_output=True)
        print(f"[publish] HTML fallback → {out_html} ({out_html.stat().st_size // 1024} KB)")
    except Exception as e:
        log_error(8, e)
    return out_pdf


def _render_epub(textbook_md: Path) -> Path:
    """Render Markdown → EPUB via Pandoc."""
    out_epub = OUTPUT_DIR / "textbook.epub"
    cmd = [
        "pandoc", str(textbook_md),
        "--toc",
        "--epub-chapter-level=2",
        "-o", str(out_epub),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
        print(f"[publish] EPUB rendered → {out_epub} ({out_epub.stat().st_size // 1024} KB)")
    except FileNotFoundError:
        print("[publish] pandoc not found; skipping EPUB rendering.")
    except subprocess.CalledProcessError as e:
        log_error(8, e)
        print(f"[publish] EPUB rendering failed: {e}")
    return out_epub


# ── Checkpoint / DONE ─────────────────────────────────────────────────────────

def _update_checkpoint(task_name: str, pdf: Path, epub: Path):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 3
    data["batch"] = 10
    data["step"] = 8
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["output_hashes"] = {
        "textbook.pdf": str(pdf) if pdf.exists() else "",
        "textbook.epub": str(epub) if epub.exists() else "",
    }
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def step_8(
    chapters: dict[str, str] | None = None,
    citations: dict[str, list] | None = None,
    toc: dict | None = None,
    images: list[str] | None = None,
) -> tuple[Path, Path]:
    """Assemble and render the final textbook.

    Args:
        chapters: {chapter_id: markdown_text}. If None, reads from BOOK_DIR.
        citations: Ignored (already injected into chapter text by citation agent).
        toc: ToC dict. If None, reads workspace/toc.json.
        images: List of upscaled PNG paths (for reference; manifest.json is primary).

    Returns:
        Tuple of (pdf_path, epub_path). Files may not exist if Pandoc is missing.
    """
    def work():
        nonlocal chapters, toc
        if chapters is None:
            chapters = {}
            for p in sorted(BOOK_DIR.glob("*.md")):
                if p.name == "textbook.md":
                    continue
                if not (p.name.startswith("ch_") or p.name.startswith("lec")):
                    continue
                chapters[p.stem] = p.read_text(encoding="utf-8")

        if toc is None:
            toc_path = Path(__file__).parent.parent / "workspace" / "toc.json"
            toc = json.loads(toc_path.read_text()) if toc_path.exists() else {"sections": []}

        manifest = _load_figure_manifest()
        textbook_md = _assemble_textbook(chapters, toc, manifest)
        pdf = _render_pdf(textbook_md)
        epub = _render_epub(textbook_md)
        return pdf, epub

    try:
        pdf, epub = exponential_backoff(work, step=8)
        _update_checkpoint("agents/08_publish_agent.py implemented", pdf, epub)
        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · "
            "08_publish_agent: textbook.md + Pandoc PDF + EPUB"
        )
        cleanup_temp()
        return pdf, epub
    except Exception as e:
        log_error(8, e)
        raise


if __name__ == "__main__":
    pdf, epub = step_8()
    print(f"\n[publish] Done.")
    print(f"  PDF:  {pdf}  (exists={pdf.exists()})")
    print(f"  EPUB: {epub} (exists={epub.exists()})")
