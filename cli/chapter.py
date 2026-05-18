"""cli/chapter.py — Chapter: the interactive aesthetic interface to MTE.

Usage
-----
  chapter              → interactive menu
  chapter build <url>  → build non-interactively with live output
  chapter library      → browse past runs
  chapter open [id]    → open a textbook in the browser
  chapter info         → system / model status
"""
from __future__ import annotations

import io
import json
import os
import queue
import sys
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Console ───────────────────────────────────────────────────────────────────

console = Console(highlight=False)

# ── Palette ───────────────────────────────────────────────────────────────────

C = {
    "violet_bright": "#e9d5ff",
    "violet_light":  "#c084fc",
    "violet":        "#a855f7",
    "violet_mid":    "#9333ea",
    "violet_deep":   "#7c3aed",
    "violet_dark":   "#6d28d9",
    "indigo":        "#4f46e5",
    "gold":          "#f59e0b",
    "gold_light":    "#fcd34d",
    "sky":           "#38bdf8",
    "green":         "#4ade80",
    "red":           "#f87171",
    "muted":         "#64748b",
    "text":          "#e2e8f0",
    "border":        "#6d28d9",
}

# ── ASCII art — CHAPTER (block font, 6 rows) ──────────────────────────────────

_TITLE_ROWS = [
    " ██████╗██╗  ██╗ █████╗ ██████╗ ████████╗███████╗██████╗ ",
    "██╔════╝██║  ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██╔══██╗",
    "██║     ███████║███████║██████╔╝   ██║   █████╗  ██████╔╝",
    "██║     ██╔══██║██╔══██║██╔═══╝    ██║   ██╔══╝  ██╔══██╗",
    "╚██████╗██║  ██║██║  ██║██║        ██║   ███████╗██║  ██║",
    " ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚══════╝╚═╝  ╚═╝",
]

# Per-row gradient: top bright → bottom deep
_TITLE_GRADIENT = [
    C["violet_bright"],
    C["violet_light"],
    C["violet"],
    C["violet_mid"],
    C["violet_deep"],
    C["violet_dark"],
]


def _title_text() -> Text:
    t = Text(justify="center")
    for i, row in enumerate(_TITLE_ROWS):
        t.append(row, style=f"bold {_TITLE_GRADIENT[i]}")
        if i < len(_TITLE_ROWS) - 1:
            t.append("\n")
    return t


def _header() -> Panel:
    content = Text(justify="center")
    content.append_text(_title_text())
    content.append(f"\n\n  Multimodal Textbook Engine  ·  v4.1  ", style=f"dim {C['violet_light']}")
    content.append(f"\n  Transform any lecture into a structured textbook  \n",
                   style=C["muted"])
    return Panel(
        Align.center(content),
        border_style=C["border"],
        box=box.DOUBLE_EDGE,
        padding=(0, 4),
    )


# ── Workspace helpers ─────────────────────────────────────────────────────────

def _workspace_root() -> Path:
    override = os.environ.get("MTE_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve().parent.parent
    return here / "workspace_v4"


def _runs_dir() -> Path:
    return _workspace_root() / "runs"


def _load_run(run_path: Path) -> Optional[dict]:
    j = run_path / "run.json"
    if not j.exists():
        return None
    try:
        return json.loads(j.read_text(encoding="utf-8"))
    except Exception:
        return None


def _list_runs() -> list[dict]:
    rd = _runs_dir()
    if not rd.exists():
        return []
    runs = []
    for p in sorted(rd.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        data = _load_run(p)
        if data:
            runs.append(data)
    return runs


def _fmt_elapsed(secs: float) -> str:
    secs = int(secs)
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}" if m else f"{s}s"


def _source_title(run: dict) -> str:
    """Best display title for a run: prefer stored textbook_title, then source titles."""
    t = run.get("textbook_title")
    if t:
        return t[:60]
    srcs = run.get("sources") or []
    if srcs:
        # Try stored title first
        st = srcs[0].get("title")
        if st:
            return st[:60]
        loc = srcs[0].get("location", "")
        if "youtube.com/watch" in loc or "youtu.be" in loc:
            vid = loc.split("v=")[-1].split("&")[0] if "v=" in loc else loc.split("/")[-1]
            return f"youtube/{vid}"
        if loc.startswith("http"):
            from urllib.parse import urlparse
            return urlparse(loc).netloc + urlparse(loc).path[:30]
        return Path(loc).name[:40]
    return run.get("run_id", "?")


def _fmt_datetime(iso: Optional[str]) -> str:
    """Format ISO datetime string for library display."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d  %H:%M")
    except Exception:
        return iso[:16]


# ── Build with live progress ──────────────────────────────────────────────────

class _Capture(io.StringIO):
    """Redirect stdout to a thread-safe queue during build."""
    def __init__(self, q: "queue.Queue[Optional[str]]") -> None:
        super().__init__()
        self._q = q

    def write(self, s: str) -> int:  # type: ignore[override]
        stripped = s.rstrip()
        if stripped:
            self._q.put(stripped)
        return len(s)

    def flush(self) -> None:
        pass


def _progress_panel(logs: list[str], elapsed: float, title: str) -> Panel:
    content = Text()
    spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    frame = spinner_frames[int(elapsed * 8) % len(spinner_frames)]
    content.append(f"  {frame} ", style=f"bold {C['gold']}")
    content.append(f"elapsed {_fmt_elapsed(elapsed)}", style=C["muted"])
    content.append("\n\n")
    # Show last 7 log lines
    recent = logs[-7:] if len(logs) > 7 else logs
    for line in recent:
        if line.startswith("[ingest]"):
            tag, rest = "[ingest]", line[8:]
            content.append(f"  {tag}", style=f"dim {C['violet_light']}")
            content.append(rest + "\n", style=C["muted"])
        else:
            content.append(f"  {line}\n", style=C["muted"])
    if not recent:
        content.append(f"  Starting…\n", style=C["muted"])
    return Panel(
        content,
        title=f"[bold {C['gold']}]◆ {title}[/]",
        border_style=C["violet_deep"],
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _result_panel(result: object, elapsed: float) -> Panel:
    from engine_v4.types import BuildResult  # type: ignore
    r: BuildResult = result  # type: ignore

    content = Text()
    total_ch = sum(len(s.chapters) for s in r.states)
    total_figs = sum(
        sum(len(ch.figures) for ch in s.chapters) for s in r.states
    )

    content.append(f"  ✓  ", style=f"bold {C['green']}")
    content.append(
        f"{total_ch} chapter{'s' if total_ch != 1 else ''}  ·  "
        f"{total_figs} figure{'s' if total_figs != 1 else ''}  ·  "
        f"completed in {_fmt_elapsed(elapsed)}",
        style=C["text"],
    )
    content.append("\n\n")

    for state in r.states:
        for i, ch in enumerate(state.chapters, 1):
            nfigs = len(ch.figures or [])
            score = ch.eval_score
            score_col = C["green"] if score >= 0.70 else C["gold"] if score >= 0.50 else C["red"]
            content.append(f"  {i}  ", style=f"bold {C['gold']}")
            content.append(f"{ch.title[:52]:<52}", style=C["text"])
            content.append(f"  {nfigs} ", style=f"dim {C['violet_light']}")
            content.append("◆", style=C["violet_light"])
            if score:
                content.append(f"  {score:.0%}", style=f"dim {score_col}")
            content.append("\n")

    content.append("\n")
    if r.html and Path(str(r.html)).exists():
        content.append(f"  📖  HTML  ", style=f"bold {C['sky']}")
        content.append(str(r.html) + "\n", style=C["muted"])
    if r.epub and Path(str(r.epub)).exists():
        content.append(f"  📦  EPUB  ", style=f"dim {C['violet_light']}")
        content.append(str(r.epub) + "\n", style=C["muted"])

    return Panel(
        content,
        title=f"[bold {C['green']}]✦ Build Complete[/]",
        border_style=C["green"],
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _run_build(spec: object, title: str = "Building Textbook") -> object:
    """Execute engine_v4.runner.run() with live progress display."""
    from engine_v4 import runner  # type: ignore
    from engine_v4.workspace import new_run_id  # type: ignore

    run_id = new_run_id()
    log_q: "queue.Queue[Optional[str]]" = queue.Queue()
    result_holder: list = [None]
    error_holder: list = [None]
    started = time.monotonic()

    def _worker() -> None:
        old_stdout = sys.stdout
        sys.stdout = _Capture(log_q)
        try:
            result_holder[0] = runner.run(spec, run_id=run_id)  # type: ignore
        except Exception as exc:
            error_holder[0] = exc
        finally:
            sys.stdout = old_stdout
            log_q.put(None)  # sentinel

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    logs: list[str] = []
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        while True:
            try:
                msg = log_q.get(timeout=0.08)
            except queue.Empty:
                live.update(_progress_panel(logs, time.monotonic() - started, title))
                continue
            if msg is None:
                break
            logs.append(msg)
            live.update(_progress_panel(logs, time.monotonic() - started, title))

    thread.join()
    elapsed = time.monotonic() - started

    if error_holder[0] is not None:
        raise error_holder[0]

    result = result_holder[0]
    console.print(_result_panel(result, elapsed))
    return result


# ── Playlist support ──────────────────────────────────────────────────────────

def _is_playlist_url(url: str) -> bool:
    """Return True for youtube.com/playlist?list=… URLs (not single-video-in-playlist)."""
    u = url.lower()
    if "youtube.com/playlist" in u:
        return True
    # list= present but no watch?v= or youtu.be short-link → treat as playlist
    if "list=" in u and "youtu" in u and "watch?v=" not in u and "youtu.be/" not in u:
        return True
    return False


def _fmt_duration(secs: "float | None") -> str:
    if not secs:
        return "—"
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fetch_playlist_entries(url: str) -> list[dict]:
    """Flat-fetch playlist metadata via yt-dlp (no download). Returns list of video dicts."""
    import yt_dlp  # type: ignore

    entries: list[dict] = []
    with console.status(
        f"[{C['violet_light']}]Fetching playlist…[/]",
        spinner="dots",
        spinner_style=f"bold {C['gold']}",
    ):
        opts = {"quiet": True, "extract_flat": True, "no_warnings": True, "ignoreerrors": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)

    if not info:
        return entries

    playlist_title = info.get("title", "")
    if playlist_title:
        console.print(f"\n  [{C['violet_light']}]◆ Playlist:[/] [{C['text']}]{playlist_title}[/]")

    for i, e in enumerate(info.get("entries") or [], 1):
        if not e:
            continue
        vid_id  = e.get("id", "")
        vid_url = e.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
        entries.append({
            "index":          i,
            "id":             vid_id,
            "title":          e.get("title", "Untitled"),
            "duration":       e.get("duration"),
            "url":            vid_url,
            "playlist_title": playlist_title,
        })
    return entries


def _playlist_table(entries: list[dict], selected: "set[int] | None" = None) -> Table:
    """Rich table of playlist entries; ● marks selected rows (None = all selected)."""
    t = Table(
        box=box.ROUNDED,
        border_style=C["border"],
        header_style=f"bold {C['violet_light']}",
        show_lines=False,
        pad_edge=True,
        padding=(0, 1),
    )
    t.add_column("#",     width=3,  justify="right")
    t.add_column("Title", max_width=56, no_wrap=True)
    t.add_column("Len",   width=8,  justify="right")
    t.add_column(" ",     width=2,  justify="center")

    for e in entries:
        i       = e["index"]
        is_sel  = (selected is None) or (i in selected)
        dot     = Text("●", style=f"bold {C['green']}") if is_sel else Text("○", style=C["muted"])
        t.add_row(
            Text(str(i),                       style=f"bold {C['gold']}"),
            Text(e["title"],                   style=C["text"]),
            Text(_fmt_duration(e.get("duration")), style=C["muted"]),
            dot,
        )
    return t


def _parse_selection(raw: str, n: int) -> "set[int]":
    """Parse 'a'/'all', '1,3,5', '1-5', '2,4-7' → set of valid 1-based indices."""
    raw = raw.strip().lower()
    if not raw or raw in ("a", "all"):
        return set(range(1, n + 1))
    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            try:
                lo_s, _, hi_s = part.partition("-")
                result.update(range(int(lo_s), int(hi_s) + 1))
            except ValueError:
                pass
        else:
            try:
                result.add(int(part))
            except ValueError:
                pass
    return {i for i in result if 1 <= i <= n}


def _build_playlist_selection(selected: list[dict]) -> None:
    """Sequentially build individual textbooks for each entry in *selected*."""
    from engine_v4 import SourceSpec  # type: ignore

    total = len(selected)
    for idx, entry in enumerate(selected, 1):
        console.print()
        console.print(Rule(
            title=f"[bold {C['gold']}]  {idx}/{total} · {entry['title'][:52]}  [/]",
            style=C["border"],
        ))
        try:
            spec   = SourceSpec(location=entry["url"], kind="auto", title=entry["title"])
            result = _run_build(spec, title=f"[{idx}/{total}] {entry['title'][:35]}")
            html   = getattr(result, "html", None)
            if html and Path(str(html)).exists():
                console.print(f"  [{C['sky']}]HTML →[/] [{C['muted']}]{html}[/]")
        except Exception as exc:
            console.print(f"  [{C['red']}]Failed:[/] {exc}")
            if not Confirm.ask(
                f"  [{C['gold']}]Continue with next video?[/]",
                console=console, default=True,
            ):
                break


# ── Playlist chapter grouping ─────────────────────────────────────────────────

def _chapter_grouping_ui(selected: list[dict]) -> list[tuple[str, list[dict]]]:
    """Interactively define chapter groups from a list of selected video entries.

    Returns a list of (chapter_title, [entry, ...]) tuples.
    """
    n = len(selected)
    idx_to_entry = {e["index"]: e for e in selected}
    valid_indices = {e["index"] for e in selected}

    console.print(f"\n  [{C['violet_light']}]Define chapters.[/] [{C['muted']}]Assign videos to chapters "
                  f"by range/list (same syntax as selection).[/]")
    console.print(f"  [{C['muted']}]Enter a blank line when done.[/]")
    console.print(f"  [{C['muted']}]Example:  Chapter 1 → '1-3'   Chapter 2 → '4-7'[/]\n")

    groups: list[tuple[str, list[dict]]] = []
    assigned: set[int] = set()
    chapter_num = 1

    while True:
        remaining = sorted(valid_indices - assigned)
        if not remaining:
            break

        console.print(f"  [{C['muted']}]Unassigned: {remaining}[/]")
        raw_range = Prompt.ask(
            f"  [{C['gold']}]Chapter {chapter_num} videos[/] [{C['muted']}](blank = done)[/]",
            console=console,
            default="",
        ).strip()
        if not raw_range:
            break

        indices = _parse_selection(raw_range, max(valid_indices)) & valid_indices
        if not indices:
            console.print(f"  [{C['red']}]No valid indices — try again.[/]")
            continue

        already = indices & assigned
        if already:
            console.print(f"  [{C['gold']}]Warning:[/] [{C['muted']}]{sorted(already)} already assigned, skipping.[/]")
            indices -= already
            if not indices:
                continue

        default_name = f"Chapter {chapter_num}"
        name = Prompt.ask(
            f"  [{C['gold']}]Chapter {chapter_num} name[/]",
            console=console,
            default=default_name,
        ).strip() or default_name

        group_entries = [idx_to_entry[i] for i in sorted(indices)]
        groups.append((name, group_entries))
        assigned |= indices
        chapter_num += 1

    # Any unassigned videos → extra chapter
    leftover = sorted(valid_indices - assigned)
    if leftover:
        name = f"Chapter {chapter_num}"
        console.print(
            f"\n  [{C['muted']}]Remaining videos {leftover} → [{C['text']}]{name}[/][/]"
        )
        groups.append((name, [idx_to_entry[i] for i in leftover]))

    return groups


def _run_playlist_as_single_book(selected: list[dict], playlist_title: str = "") -> None:
    """Build all selected videos into ONE textbook, grouped into chapters."""
    from engine_v4 import SourceSpec, run_playlist_as_book  # type: ignore

    n = len(selected)
    console.print(f"\n  [{C['violet_light']}]◆ Single-textbook mode — {n} video(s)[/]")

    # Ask for chapter grouping strategy
    console.print(f"\n  [{C['muted']}]How should videos be grouped into chapters?[/]")
    console.print(f"  [{C['gold']}][a][/] One video per chapter (auto)")
    console.print(f"  [{C['gold']}][g][/] Define chapter groups manually")
    strategy = Prompt.ask(
        f"  [{C['gold']}]Strategy[/]",
        console=console,
        default="a",
    ).strip().lower()

    if strategy == "g":
        groups_data = _chapter_grouping_ui(selected)
    else:
        groups_data = [(e["title"][:60], [e]) for e in selected]

    if not groups_data:
        console.print(f"  [{C['red']}]No groups defined.[/]")
        return

    # Show summary
    console.print(f"\n  [{C['violet_light']}]◆ Chapter plan:[/]")
    for i, (chapter_title, entries) in enumerate(groups_data, 1):
        console.print(f"  [{C['gold']}]{i}.[/] [{C['text']}]{chapter_title}[/]")
        for e in entries:
            console.print(f"      [{C['muted']}]· {e['title'][:55]}[/]")

    book_title = Prompt.ask(
        f"\n  [{C['gold']}]Book title[/]",
        console=console,
        default=playlist_title or "Playlist Textbook",
    ).strip() or playlist_title or "Playlist Textbook"

    if not Confirm.ask(f"\n  [{C['gold']}]Build single textbook?[/]", console=console, default=True):
        return

    groups: list[tuple[str, list[SourceSpec]]] = [
        (
            chapter_title,
            [SourceSpec(location=e["url"], kind="auto", title=e["title"]) for e in entries],
        )
        for chapter_title, entries in groups_data
    ]

    console.print()
    log_q: "queue.Queue[Optional[str]]" = queue.Queue()
    result_holder: list = [None]
    error_holder: list = [None]
    started = time.monotonic()

    def _worker() -> None:
        old_stdout = sys.stdout
        sys.stdout = _Capture(log_q)
        try:
            result_holder[0] = run_playlist_as_book(groups, book_title=book_title)
        except Exception as exc:
            error_holder[0] = exc
        finally:
            sys.stdout = old_stdout
            log_q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    logs: list[str] = []
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        while True:
            try:
                msg = log_q.get(timeout=0.08)
            except queue.Empty:
                live.update(_progress_panel(logs, time.monotonic() - started, f"Building — {book_title[:35]}"))
                continue
            if msg is None:
                break
            logs.append(msg)
            live.update(_progress_panel(logs, time.monotonic() - started, f"Building — {book_title[:35]}"))

    thread.join()
    elapsed = time.monotonic() - started

    if error_holder[0] is not None:
        console.print(f"  [{C['red']}]Build failed:[/] {error_holder[0]}")
        return

    result = result_holder[0]
    console.print(_result_panel(result, elapsed))

    html = getattr(result, "html", None)
    if html and Path(str(html)).exists():
        if Confirm.ask(f"\n  [{C['gold']}]Open HTML in browser?[/]", console=console, default=True):
            import webbrowser as _wb
            _wb.open(Path(str(html)).resolve().as_uri())


def _run_playlist_interactive(entries: list[dict]) -> None:
    """Show playlist picker, ask selection, confirm, then build."""
    if not entries:
        console.print(f"  [{C['red']}]No videos found in playlist.[/]")
        return

    playlist_title = entries[0].get("playlist_title", "") if entries else ""

    console.print(f"\n  [{C['muted']}]{len(entries)} video(s) in playlist.[/]")
    console.print(_playlist_table(entries))
    console.print(
        f"\n  [{C['muted']}]Select videos to build.  "
        f"[{C['gold']}]Examples:[/] "
        f"[{C['text']}]a[/] = all  ·  "
        f"[{C['text']}]1,3,5[/] = specific  ·  "
        f"[{C['text']}]1-5[/] = range  ·  "
        f"[{C['text']}]2,4-7[/] = mixed"
    )
    raw     = Prompt.ask(f"  [{C['gold']}]Selection[/]", console=console, default="a").strip()
    indices = _parse_selection(raw, len(entries))

    if not indices:
        console.print(f"  [{C['red']}]No valid videos selected.[/]")
        return

    selected = [e for e in entries if e["index"] in indices]
    console.print()
    console.print(_playlist_table(selected, selected=indices))
    console.print(f"\n  [{C['violet_light']}]{len(selected)} video(s) will be built.[/]")

    # Ask: single textbook or individual textbooks?
    console.print(f"\n  [{C['muted']}]Output format:[/]")
    console.print(f"  [{C['gold']}][s][/] Single textbook (all videos combined in one book)")
    console.print(f"  [{C['gold']}][i][/] Individual textbooks (one per video)")
    mode = Prompt.ask(f"  [{C['gold']}]Mode[/]", console=console, default="s").strip().lower()

    if mode == "i":
        if not Confirm.ask(f"\n  [{C['gold']}]Start building?[/]", console=console, default=True):
            return
        _build_playlist_selection(selected)
    else:
        _run_playlist_as_single_book(selected, playlist_title=playlist_title)

def _library_table(runs: list[dict]) -> Table:
    t = Table(
        box=box.ROUNDED,
        border_style=C["border"],
        header_style=f"bold {C['violet_light']}",
        show_lines=False,
        pad_edge=True,
        padding=(0, 1),
    )
    t.add_column("#",       width=3,  justify="right")
    t.add_column("Title",   max_width=46, no_wrap=True)
    t.add_column("Date",    width=17)
    t.add_column("Ch",      width=3,  justify="right")
    t.add_column("Formats", width=9)

    for i, run in enumerate(runs, 1):
        title    = _source_title(run)
        dt       = _fmt_datetime(run.get("started_at_utc"))
        chapters = run.get("chapters_total", 0)
        outputs  = run.get("outputs", {})

        fmts = Text()
        if outputs.get("html"):
            fmts.append("H", style=f"bold {C['sky']}")
        if outputs.get("epub"):
            fmts.append(" E", style=f"bold {C['violet_light']}")
        if outputs.get("pdf"):
            fmts.append(" P", style=f"bold {C['green']}")

        t.add_row(
            Text(str(i),        style=f"bold {C['gold']}"),
            Text(title,         style=C["text"]),
            Text(dt,            style=C["muted"]),
            Text(str(chapters), style=f"bold {C['violet_light']}"),
            fmts,
        )

    return t


# ── System info ───────────────────────────────────────────────────────────────

def _sysinfo_table() -> Table:
    import shutil

    t = Table(box=box.SIMPLE, pad_edge=True, show_header=False, padding=(0, 2))
    t.add_column("item", style=f"bold {C['violet_light']}", width=22, no_wrap=True)
    t.add_column("value", style=C["text"])

    def _row(label: str, val: str, ok: bool) -> None:
        icon = f"[{C['green']}]✓[/]" if ok else f"[{C['red']}]✗[/]"
        t.add_row(f"{icon} {label}", val)

    _row("python",    sys.version.split()[0], True)
    for tool in ("ffmpeg", "ffprobe", "pandoc", "yt-dlp", "tesseract", "xelatex"):
        path = shutil.which(tool)
        _row(tool, path or "not found", bool(path))

    _MOD_LABELS = {
        "engine_v4":             "engine_v4",
        "llama_cpp":             "llama_cpp",
        "mlx_lm":                "mlx_lm",
        "sentence_transformers": "sentence_tf",
        "imagehash":             "imagehash",
        "scenedetect":           "scenedetect",
        "PIL":                   "PIL",
        "faster_whisper":        "faster_whisper",
    }
    t.add_row("", "")
    for mod, label in _MOD_LABELS.items():
        try:
            __import__(mod)
            _row(label, "installed", True)
        except Exception as e:
            _row(label, str(e)[:50], False)

    # Model files
    try:
        from config.settings import GEMMA_GGUF, QWEN_MLX_MODEL, MINILM_LOCAL  # type: ignore
        t.add_row("", "")
        _row("Gemma GGUF",   str(GEMMA_GGUF)[-50:],     GEMMA_GGUF.exists())
        _row("Qwen MLX",     str(QWEN_MLX_MODEL)[-50:], QWEN_MLX_MODEL.exists())
        _row("MiniLM",       str(MINILM_LOCAL)[-50:],   MINILM_LOCAL.exists())
    except Exception:
        pass

    return t


# ── Interactive menu loop ─────────────────────────────────────────────────────

_MENU_ITEMS = [
    ("1", "📖", "Build Textbook",    "Convert any URL or local file"),
    ("2", "▶ ", "Build Playlist",    "Select videos from a YouTube playlist"),
    ("3", "📚", "My Library",        "Browse past textbooks"),
    ("4", "🔍", "Open Textbook",     "Launch HTML in browser"),
    ("5", "➕", "Add Chapter",       "Append chapter(s) to existing textbook"),
    ("6", "🔎", "Textbook Details",  "Inspect a textbook's metadata"),
    ("7", "ℹ️ ", "System Info",       "Check tools & models"),
    ("q", "×",  "Quit",              ""),
]


def _menu_panel(run_count: int) -> Panel:
    content = Text()
    content.append("\n")
    for key, icon, label, desc in _MENU_ITEMS:
        content.append(f"   [{key}]  ", style=f"bold {C['gold']}")
        content.append(f"{icon}  ", style="")
        content.append(f"{label:<22}", style=f"bold {C['text']}")
        if desc:
            if label == "My Library":
                desc = f"Browse {run_count} past textbook{'s' if run_count != 1 else ''}"
            content.append(f"·  {desc}", style=C["muted"])
        content.append("\n")
    content.append("\n")
    return Panel(
        content,
        title=f"[bold {C['violet_light']}]◆  M E N U[/]",
        border_style=C["border"],
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _ask_source() -> tuple[str, Optional[str]]:
    """Prompt for URL/path and optional title. Returns (location, title)."""
    console.print(Rule(style=C["border"]))
    console.print(f"[{C['violet_light']}]  Enter a YouTube URL, video path, PDF, or text file.[/]")
    loc = Prompt.ask(
        f"  [{C['gold']}]Source[/]",
        console=console,
    ).strip()
    console.print(f"  [{C['muted']}]Optional title (press Enter to skip)[/]")
    title = Prompt.ask(
        f"  [{C['gold']}]Title[/]",
        console=console,
        default="",
    ).strip() or None
    return loc, title


def _pick_run(runs: list[dict], prompt: str = "Select run #") -> Optional[dict]:
    if not runs:
        console.print(f"  [{C['red']}]No runs found.[/]")
        return None
    console.print(_library_table(runs))
    choice = Prompt.ask(
        f"  [{C['gold']}]{prompt}[/]",
        console=console,
        default="1",
    ).strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(runs):
            return runs[idx]
    except ValueError:
        pass
    console.print(f"  [{C['red']}]Invalid selection.[/]")
    return None


def _do_open(run: Optional[dict] = None) -> None:
    runs = _list_runs()
    if run is None:
        run = _pick_run(runs, "Open which run?")
    if run is None:
        return
    html = (run.get("outputs") or {}).get("html")
    if html and Path(html).exists():
        console.print(f"\n  [{C['sky']}]Opening:[/] [{C['muted']}]{html}[/]")
        webbrowser.open(Path(html).resolve().as_uri())
    else:
        console.print(f"  [{C['red']}]HTML output not found for this run.[/]")


def _details_panel(run: dict) -> Panel:
    """Build a rich panel showing detailed metadata for a textbook run."""
    content = Text()

    title = _source_title(run)
    dt    = _fmt_datetime(run.get("started_at_utc"))

    content.append(f"\n  Title:    ", style=f"bold {C['violet_light']}")
    content.append(f"{title}\n",       style=C["text"])
    content.append(f"  Date:     ", style=f"bold {C['violet_light']}")
    content.append(f"{dt}\n",          style=C["muted"])
    content.append(f"  Run ID:   ", style=f"bold {C['violet_light']}")
    content.append(f"{run.get('run_id', '?')}\n", style=C["muted"])
    content.append(f"  Chapters: ", style=f"bold {C['violet_light']}")
    content.append(f"{run.get('chapters_total', 0)}\n", style=C["text"])

    elapsed = run.get("elapsed_sec")
    if elapsed is not None:
        content.append(f"  Built in: ", style=f"bold {C['violet_light']}")
        content.append(f"{_fmt_elapsed(elapsed)}\n", style=C["muted"])

    # Chapter groups
    groups = run.get("chapter_groups") or []
    if groups:
        content.append(f"\n  [{C['gold']}]◆ Chapters[/]\n")
        for g in groups:
            ct = g.get("chapter_title", "Untitled")
            idxs = g.get("state_indices", [])
            suffix = f"  ({len(idxs)} source{'s' if len(idxs) != 1 else ''})" if idxs else ""
            appended = "  [appended]" if g.get("appended") else ""
            content.append(f"    · {ct}{suffix}{appended}\n", style=C["text"])

    # Sources
    srcs = run.get("sources") or []
    if srcs:
        content.append(f"\n  [{C['gold']}]◆ Sources[/]\n")
        for s in srcs:
            t = s.get("title") or Path(s.get("location", "?")).name
            ch = s.get("chapters", 0)
            appended = "  [appended]" if s.get("appended") else ""
            content.append(f"    · {t[:60]}  ", style=C["text"])
            content.append(f"{ch} ch{appended}\n",  style=C["muted"])

    # Outputs
    outputs = run.get("outputs") or {}
    content.append(f"\n  [{C['gold']}]◆ Outputs[/]\n")
    for key, icon in [("html", "📖 HTML"), ("epub", "📦 EPUB"), ("pdf", "📄 PDF"), ("textbook_md", "📝 MD")]:
        val = outputs.get(key)
        if val:
            exists = Path(val).exists()
            status = f"[{C['green']}]✓[/]" if exists else f"[{C['red']}]✗ (missing)[/]"
            content.append(f"    {status} {icon}  ", style="")
            content.append(f"{val}\n", style=C["muted"])

    content.append("")
    return Panel(
        content,
        title=f"[bold {C['violet_light']}]◆ Textbook Details[/]",
        border_style=C["border"],
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _do_details(run: Optional[dict] = None) -> None:
    runs = _list_runs()
    if run is None:
        run = _pick_run(runs, "Details for run #")
    if run is None:
        return
    console.print(_details_panel(run))


def _do_add_chapter(
    source: str,
    run: Optional[dict],
    chapter_name: Optional[str],
) -> Optional[object]:
    """Process *source* and append it as a chapter to *run*."""
    from engine_v4 import SourceSpec, append_chapter_to_run  # type: ignore

    if run is None:
        console.print(f"  [{C['red']}]No target run selected.[/]")
        return None

    run_ws = Path(run.get("workspace") or "")
    if not run_ws.is_dir():
        console.print(f"  [{C['red']}]Workspace not found: {run_ws}[/]")
        return None

    spec = SourceSpec(location=source, kind="auto")

    log_q: "queue.Queue[Optional[str]]" = queue.Queue()
    result_holder: list = [None]
    error_holder: list = [None]
    started = time.monotonic()
    title_lbl = chapter_name or source[:40]

    def _worker() -> None:
        old_stdout = sys.stdout
        sys.stdout = _Capture(log_q)
        try:
            result_holder[0] = append_chapter_to_run(
                run_ws, [spec], chapter_title=chapter_name
            )
        except Exception as exc:
            error_holder[0] = exc
        finally:
            sys.stdout = old_stdout
            log_q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    logs: list[str] = []
    with Live(console=console, refresh_per_second=10, transient=True) as live:
        while True:
            try:
                msg = log_q.get(timeout=0.08)
            except queue.Empty:
                live.update(_progress_panel(logs, time.monotonic() - started, f"Adding chapter — {title_lbl}"))
                continue
            if msg is None:
                break
            logs.append(msg)
            live.update(_progress_panel(logs, time.monotonic() - started, f"Adding chapter — {title_lbl}"))

    thread.join()
    elapsed = time.monotonic() - started

    if error_holder[0] is not None:
        raise error_holder[0]

    result = result_holder[0]
    console.print(_result_panel(result, elapsed))
    return result


def _interactive_loop() -> None:
    while True:
        console.clear()
        console.print(_header())

        runs = _list_runs()
        console.print(_menu_panel(len(runs)))

        choice = Prompt.ask(
            f"  [{C['gold']}]Enter choice[/]",
            console=console,
            default="q",
        ).strip().lower()

        if choice == "q":
            console.print(f"\n  [{C['violet_light']}]Goodbye. ◆[/]\n")
            break

        elif choice == "1":
            # ── Build single video / file ──────────────────────────────────
            try:
                console.print(Rule(style=C["border"]))
                console.print(f"  [{C['violet_light']}]Enter a YouTube URL, video path, PDF, or text file.[/]")
                loc = Prompt.ask(f"  [{C['gold']}]Source[/]", console=console).strip()
                if not loc:
                    continue
                out_dir = Prompt.ask(
                    f"  [{C['gold']}]Output workspace[/]",
                    console=console,
                    default=str(_workspace_root()),
                ).strip()
                os.environ["MTE_WORKSPACE_ROOT"] = out_dir
                title = Prompt.ask(
                    f"  [{C['gold']}]Title (optional)[/]",
                    console=console, default="",
                ).strip() or None
                from engine_v4 import SourceSpec  # type: ignore
                spec   = SourceSpec(location=loc, kind="auto", title=title)
                result = _run_build(spec, title=f"Building — {loc[:40]}")
                html   = getattr(result, "html", None)
                if html and Path(str(html)).exists():
                    if Confirm.ask(
                        f"\n  [{C['gold']}]Open HTML in browser?[/]",
                        console=console, default=True,
                    ):
                        webbrowser.open(Path(str(html)).resolve().as_uri())
            except Exception as exc:
                console.print(f"\n  [{C['red']}]Build failed:[/] {exc}")
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "2":
            # ── Build playlist ─────────────────────────────────────────────
            try:
                console.print(Rule(style=C["border"]))
                console.print(f"  [{C['violet_light']}]Enter a YouTube playlist URL.[/]")
                loc = Prompt.ask(f"  [{C['gold']}]Playlist URL[/]", console=console).strip()
                if not loc:
                    continue
                out_dir = Prompt.ask(
                    f"  [{C['gold']}]Output workspace[/]",
                    console=console,
                    default=str(_workspace_root()),
                ).strip()
                os.environ["MTE_WORKSPACE_ROOT"] = out_dir
                entries = _fetch_playlist_entries(loc)
                _run_playlist_interactive(entries)
            except Exception as exc:
                console.print(f"\n  [{C['red']}]Playlist build failed:[/] {exc}")
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "3":
            # ── Library ────────────────────────────────────────────────────
            console.clear()
            console.print(_header())
            if not runs:
                console.print(f"\n  [{C['muted']}]No textbooks found in {_runs_dir()}[/]\n")
            else:
                console.print(f"\n  [{C['violet_light']}]◆ Library — {len(runs)} textbook(s)[/]\n")
                console.print(_library_table(runs))
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "4":
            _do_open()
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "5":
            # ── Add Chapter ────────────────────────────────────────────────
            try:
                console.print(Rule(style=C["border"]))
                console.print(f"  [{C['violet_light']}]Select a textbook to append to.[/]")
                run = _pick_run(runs, "Append to which textbook?")
                if not run:
                    continue
                console.print(f"\n  [{C['muted']}]Enter the source to add as a new chapter.[/]")
                loc = Prompt.ask(f"  [{C['gold']}]Source URL or path[/]", console=console).strip()
                if not loc:
                    continue
                chapter_name = Prompt.ask(
                    f"  [{C['gold']}]Chapter name (optional)[/]",
                    console=console, default="",
                ).strip() or None
                result = _do_add_chapter(loc, run, chapter_name)
                if result:
                    html = getattr(result, "html", None)
                    if html and Path(str(html)).exists():
                        if Confirm.ask(
                            f"\n  [{C['gold']}]Open updated textbook in browser?[/]",
                            console=console, default=True,
                        ):
                            webbrowser.open(Path(str(html)).resolve().as_uri())
            except Exception as exc:
                console.print(f"\n  [{C['red']}]Add chapter failed:[/] {exc}")
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "6":
            # ── Textbook Details ───────────────────────────────────────────
            console.clear()
            console.print(_header())
            _do_details()
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        elif choice == "7":
            console.clear()
            console.print(_header())
            console.print(Panel(
                _sysinfo_table(),
                title=f"[bold {C['violet_light']}]◆ System Info[/]",
                border_style=C["border"],
                box=box.ROUNDED,
                padding=(0, 2),
            ))
            Prompt.ask(f"\n  [{C['muted']}]Press Enter to continue[/]",
                       console=console, default="")

        else:
            console.print(f"  [{C['red']}]Unknown option '{choice}'.[/]")
            time.sleep(0.8)



# ── Click subcommands ─────────────────────────────────────────────────────────

@click.group(
    invoke_without_command=True,
    help="Chapter — interactive MTE interface. Run without args for the menu.",
)
@click.pass_context
def cli(ctx: click.Context) -> None:
    if ctx.invoked_subcommand is None:
        _interactive_loop()


@cli.command(help="Build a textbook from a URL or file.")
@click.argument("source", type=str)
@click.option("--title", "-t", default=None, help="Override textbook title.")
@click.option("--out", "-o", "out_dir",
              type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Output workspace directory.")
@click.option("--open", "open_html", is_flag=True, default=False,
              help="Open HTML textbook in browser after build.")
def build(source: str, title: Optional[str], out_dir: Optional[Path],
          open_html: bool) -> None:
    console.print(_header())
    if out_dir:
        os.environ["MTE_WORKSPACE_ROOT"] = str(out_dir)
    from engine_v4 import SourceSpec  # type: ignore
    spec = SourceSpec(location=source, kind="auto", title=title)
    try:
        result = _run_build(spec, title=f"Building — {source[:45]}")
        html = getattr(result, "html", None)
        if open_html and html and Path(str(html)).exists():
            webbrowser.open(Path(str(html)).resolve().as_uri())
    except Exception as exc:
        console.print(f"\n  [{C['red']}]Build failed:[/] {exc}")
        sys.exit(1)


@cli.command(help="Browse past textbooks.")
@click.option("--open", "open_html", is_flag=True, default=False,
              help="Interactively open a textbook in the browser.")
def library(open_html: bool) -> None:
    console.print(_header())
    runs = _list_runs()
    if not runs:
        console.print(f"\n  [{C['muted']}]No textbooks found in {_runs_dir()}[/]\n")
        return
    console.print(f"\n  [{C['violet_light']}]◆ Library — {len(runs)} textbook(s) in {_runs_dir()}[/]\n")
    console.print(_library_table(runs))
    if open_html:
        run = _pick_run(runs, "Open which textbook?")
        if run:
            _do_open(run)


@cli.command(name="open", help="Open a textbook HTML in the browser.")
@click.argument("run_id", required=False, default=None)
def open_cmd(run_id: Optional[str]) -> None:
    runs = _list_runs()
    if run_id:
        matched = [r for r in runs if r.get("run_id") == run_id]
        _do_open(matched[0] if matched else None)
    else:
        _do_open()


@cli.command(help="Show details of a textbook (title, date, sources, chapters, outputs).")
@click.argument("run_id", required=False, default=None)
def details(run_id: Optional[str]) -> None:
    console.print(_header())
    runs = _list_runs()
    run: Optional[dict] = None
    if run_id:
        matched = [r for r in runs if r.get("run_id") == run_id]
        run = matched[0] if matched else None
        if run is None:
            console.print(f"  [{C['red']}]Run '{run_id}' not found.[/]")
            sys.exit(1)
    else:
        run = _pick_run(runs, "Details for which textbook?")
    if run:
        console.print(_details_panel(run))


@cli.command("add-chapter", help="Add chapter(s) from a source to an existing textbook.")
@click.argument("source", type=str)
@click.option("--to", "run_id", default=None,
              help="Run ID of the textbook to append to. Defaults to the most recent.")
@click.option("--chapter-name", "chapter_name", default=None,
              help="Chapter heading title (e.g. 'Chapter 2: Advanced Topics').")
@click.option("--out", "-o", "out_dir",
              type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Workspace root (only needed if different from default).")
@click.option("--open", "open_html", is_flag=True, default=False,
              help="Open updated HTML textbook in browser after appending.")
def add_chapter(
    source: str,
    run_id: Optional[str],
    chapter_name: Optional[str],
    out_dir: Optional[Path],
    open_html: bool,
) -> None:
    console.print(_header())
    if out_dir:
        os.environ["MTE_WORKSPACE_ROOT"] = str(out_dir)

    runs = _list_runs()
    if not runs:
        console.print(f"  [{C['red']}]No textbooks found. Build one first with `chapter build`.[/]")
        sys.exit(1)

    run: Optional[dict] = None
    if run_id:
        matched = [r for r in runs if r.get("run_id") == run_id]
        run = matched[0] if matched else None
        if run is None:
            console.print(f"  [{C['red']}]Run '{run_id}' not found.[/]")
            sys.exit(1)
    else:
        # Pick most recent or show picker
        if len(runs) == 1:
            run = runs[0]
        else:
            run = _pick_run(runs, "Append to which textbook?")

    if run is None:
        sys.exit(0)

    try:
        result = _do_add_chapter(source, run, chapter_name)
        html = getattr(result, "html", None)
        if open_html and html and Path(str(html)).exists():
            webbrowser.open(Path(str(html)).resolve().as_uri())
    except Exception as exc:
        console.print(f"\n  [{C['red']}]Add chapter failed:[/] {exc}")
        sys.exit(1)


@cli.command(help="Show system and model status.")
def info() -> None:
    console.print(_header())
    console.print(Panel(
        _sysinfo_table(),
        title=f"[bold {C['violet_light']}]◆ System Info[/]",
        border_style=C["border"],
        box=box.ROUNDED,
        padding=(0, 2),
    ))


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> None:
    cli()


if __name__ == "__main__":
    main()
