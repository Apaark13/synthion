"""mte: Multimodal Textbook Engine command-line interface.

Subcommands:
  build       Build a textbook from a single source (video/audio/pdf/text/url).
  build-many  Build from multiple sources at once (file with one per line, or args).
  sync        Run only the visual_sync module on a video + transcript.
  info        Print environment + model status.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import click


_VIDEO_EXT = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
_AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
_PDF_EXT = {".pdf"}
_TEXT_EXT = {".txt", ".md", ".rst"}


def _sniff_kind(location: str) -> str:
    low = location.lower()
    if low.startswith(("http://", "https://")):
        if "youtube.com" in low or "youtu.be" in low:
            return "youtube"
        return "url"
    suf = Path(location).suffix.lower()
    if suf in _VIDEO_EXT:
        return "video"
    if suf in _AUDIO_EXT:
        return "audio"
    if suf in _PDF_EXT:
        return "pdf"
    if suf in _TEXT_EXT:
        return "text"
    return "custom"


def _load_sources_file(path: Path) -> List[str]:
    items: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        items.append(line)
    return items


def _emit_result(result, fmt: str) -> None:
    if fmt == "json":
        payload = {
            "run_id": getattr(result, "run_id", None),
            "workspace": getattr(result, "workspace", None),
            "html": getattr(result, "html", None),
            "textbook_md": getattr(result, "textbook_md", None),
            "epub": getattr(result, "epub", None),
            "pdf": getattr(result, "pdf", None),
            "errors": list(getattr(result, "errors", []) or []),
        }
        click.echo(json.dumps(payload, indent=2, default=str))
    else:
        click.echo(f"run_id:    {getattr(result, 'run_id', '?')}")
        click.echo(f"workspace: {getattr(result, 'workspace', '?')}")
        html = getattr(result, "html", None)
        if html:
            click.echo(f"\n📖  HTML textbook (default output):\n    {html}\n")
        for key in ("textbook_md", "epub", "pdf"):
            val = getattr(result, key, None)
            if val:
                click.echo(f"{key:10s} {val}")
        errs = list(getattr(result, "errors", []) or [])
        if errs:
            click.echo("errors:")
            for e in errs:
                click.echo(f"  - {e}")


@click.group(help="Multimodal Textbook Engine CLI.")
@click.version_option("0.4.0", prog_name="mte")
def cli() -> None:
    pass


@cli.command(help="Build a textbook from a single source.")
@click.argument("source", type=str)
@click.option("--kind", type=click.Choice(
    ["auto", "video", "youtube", "url", "audio", "pdf", "text", "custom"]),
    default="auto", show_default=True)
@click.option("--title", type=str, default=None)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", show_default=True)
@click.option("--open", "open_html", is_flag=True, default=False,
              help="Open the produced HTML textbook in the default browser.")
def build(source: str, kind: str, title: Optional[str], out_dir: Optional[Path],
          fmt: str, open_html: bool) -> None:
    from engine_v4 import SourceSpec, run

    if kind == "auto":
        kind = _sniff_kind(source)
    if out_dir:
        os.environ["MTE_WORKSPACE_ROOT"] = str(out_dir)

    spec = SourceSpec(location=source, kind=kind, title=title)
    result = run(spec)
    _emit_result(result, fmt)

    html = getattr(result, "html", None)
    if open_html and html:
        import webbrowser
        webbrowser.open(Path(html).resolve().as_uri())


@cli.command("build-many", help="Build from multiple sources (positional args or --from-file).")
@click.argument("sources", nargs=-1)
@click.option("--from-file", "from_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None)
@click.option("--kind", type=click.Choice(
    ["auto", "video", "youtube", "url", "audio", "pdf", "text", "custom"]),
    default="auto", show_default=True)
@click.option("--max-concurrency", type=int, default=2, show_default=True)
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text", show_default=True)
def build_many(sources, from_file, kind, max_concurrency, out_dir, fmt):
    from engine_v4 import SourceSpec, run_many

    items: List[str] = list(sources)
    if from_file:
        items.extend(_load_sources_file(from_file))
    if not items:
        raise click.UsageError("Provide at least one SOURCE arg or --from-file PATH.")
    if out_dir:
        os.environ["MTE_WORKSPACE_ROOT"] = str(out_dir)

    specs = [
        SourceSpec(location=loc, kind=(kind if kind != "auto" else _sniff_kind(loc)))
        for loc in items
    ]
    result = run_many(specs, max_concurrency=max_concurrency)
    _emit_result(result, fmt)


@cli.command(help="Run visual_sync to align video frames with a transcript JSON.")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("transcript", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--ocr-backend", type=click.Choice(["auto", "vision", "tesseract", "none"]),
              default="auto", show_default=True)
def sync(video: Path, transcript: Path, out_dir: Path, ocr_backend: str) -> None:
    from modules.visual_sync import (
        TranscriptSegment,
        VisualSyncConfig,
        extract_aligned_figures,
    )

    raw = json.loads(transcript.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "segments" in raw:
        raw = raw["segments"]
    segments = [
        TranscriptSegment(
            start=float(s.get("start", 0.0)),
            end=float(s.get("end", s.get("start", 0.0))),
            text=str(s.get("text", "")),
        )
        for s in raw
    ]
    cfg = VisualSyncConfig(ocr_backend=ocr_backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = extract_aligned_figures(str(video), segments, output_dir=out_dir, config=cfg)
    figs_path = out_dir / "figures.json"
    figs_path.write_text(json.dumps(
        {sec: [vars(f) for f in figs] for sec, figs in result.items()},
        indent=2, default=str,
    ), encoding="utf-8")
    total = sum(len(v) for v in result.values())
    click.echo(f"Wrote {total} aligned figures across {len(result)} sections -> {figs_path}")


@cli.command(help="Print environment + model status.")
def info() -> None:
    import shutil

    click.echo(f"python:    {sys.version.split()[0]}")
    click.echo(f"cwd:       {os.getcwd()}")
    click.echo(f"ffmpeg:    {shutil.which('ffmpeg') or 'NOT FOUND'}")
    click.echo(f"ffprobe:   {shutil.which('ffprobe') or 'NOT FOUND'}")
    click.echo(f"pandoc:    {shutil.which('pandoc') or 'NOT FOUND'}")
    click.echo(f"xelatex:   {shutil.which('xelatex') or 'NOT FOUND (PDF will fall back to HTML)'}")
    click.echo(f"tesseract: {shutil.which('tesseract') or 'NOT FOUND'}")

    for mod in ("imagehash", "scenedetect", "PIL", "click",
                "engine_v4", "modules.visual_sync"):
        try:
            __import__(mod)
            click.echo(f"  [ok] {mod}")
        except Exception as exc:
            click.echo(f"  [--] {mod}: {exc}")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
