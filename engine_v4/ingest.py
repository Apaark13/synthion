"""engine_v4.ingest — Layer 1 (INGEST).

Turns a ``SourceSpec`` into a fully populated ``Source`` artefact:
media on disk, transcript segments, coarse semantic windows, and
deduplicated scene frames. All work is scoped to a per-source workspace
created via :func:`engine_v4.workspace.source_workspace`.

This is a v4 port of the good parts of ``agents/00_fetch.py`` and
``agents/01_multimodal_extract.py`` — minus the v3 debug ``visual_desc``
string and the legacy 4K-segment download path.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from config.settings import (
    SCENE_THRESHOLD,
    MAX_SCENE_FRAMES,
    PHASH_DEDUP_DIST,
    FRAME_DIFF_THRESHOLD,
)

from .limits import DOWNLOAD, MEDIA
from .types import Frame, Source, SourceSpec, TranscriptSegment, TranscriptWindow
from .workspace import source_workspace, stable_source_id


# ── Constants ─────────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
TEXT_EXTS = {".md", ".txt", ".rst"}

YOUTUBE_RE = re.compile(r"^https?://(www\.)?(youtube\.com|youtu\.be)/", re.I)

LOW_RES_FORMAT = "bestvideo[height<=360]+bestaudio/best[height<=360]"

WINDOW_TARGET_SEC = 75.0
WINDOW_MIN_SEC = 60.0
WINDOW_MAX_SEC = 120.0

HIGH_INFO_KEYWORDS = frozenset({
    "equation", "equations", "formula", "proof", "theorem", "definition",
    "algorithm", "matrix", "determinant", "eigenvalue", "vector", "inverse",
    "elimination", "pivot", "rank", "null", "linear", "independent",
    "dependent", "basis", "dimension", "projection", "orthogonal",
    "transpose", "symmetric", "singular", "factorization", "decomposition",
    "derivative", "integral", "gradient", "function", "limit", "convergence",
    "polynomial", "series", "probability", "distribution", "variance",
})


def _log(msg: str) -> None:
    print(f"[ingest] {msg}")


# ── Kind sniffing ────────────────────────────────────────────────────────────

def _sniff_kind(spec: SourceSpec) -> str:
    if spec.kind != "auto":
        # Normalise CLI-only aliases.
        if spec.kind in ("youtube",):
            return "url"
        return spec.kind
    loc = spec.location.strip()
    if YOUTUBE_RE.match(loc):
        return "url"
    parsed = urlparse(loc)
    if parsed.scheme in ("http", "https"):
        # Generic URL: treat as video URL — yt-dlp handles many sites.
        return "url"
    ext = Path(loc).suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext == ".pdf":
        return "pdf"
    if ext in TEXT_EXTS:
        return "text"
    raise ValueError(f"cannot infer kind for location: {spec.location}")


# ── Video title fetch ────────────────────────────────────────────────────────

def _fetch_yt_title(url: str) -> Optional[str]:
    """Quick yt-dlp metadata fetch to get video title. No download, ~1s."""
    try:
        r = subprocess.run(
            [
                "yt-dlp", "--no-check-certificate",
                "--print", "title", "--no-download", "--no-playlist",
                "--quiet", url,
            ],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            title = r.stdout.strip().splitlines()[0].strip()
            return title or None
    except Exception:
        pass
    return None


# ── Media acquisition (download / copy) ──────────────────────────────────────

def _ydlp_download(url: str, out_dir: Path) -> Path:
    """Download a single video at <=360p into out_dir/source.mp4."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "source.mp4"
    cmd = [
        "yt-dlp", "--no-check-certificate",
        "--format", LOW_RES_FORMAT,
        "--merge-output-format", "mp4",
        "--output", str(target),
        url,
    ]
    with DOWNLOAD:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not target.exists():
        # yt-dlp sometimes appends an extension differently — fall back to glob.
        cands = sorted(out_dir.glob("source.*"))
        if not cands:
            raise FileNotFoundError(f"yt-dlp produced no file in {out_dir}")
        return cands[0]
    return target


def _acquire_media(kind: str, location: str, media_dir: Path) -> tuple[Path, Optional[str]]:
    """Return (local media path, remote URL if any)."""
    p = Path(location)
    if p.exists():
        ext = p.suffix.lower() or ".mp4"
        target = media_dir / f"source{ext}"
        if not target.exists():
            try:
                os.symlink(p.resolve(), target)
            except OSError:
                shutil.copy2(p, target)
        return target, None

    if kind in ("url", "video"):
        _log(f"yt-dlp <=360p {location}")
        path = _ydlp_download(location, media_dir)
        # Cache URL for subtitle fetch.
        (media_dir / "source.url").write_text(location)
        return path, location

    raise FileNotFoundError(f"no local file and not a URL: {location}")


# ── ffprobe ──────────────────────────────────────────────────────────────────

def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output([
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ], stderr=subprocess.DEVNULL, timeout=15)
        return float(out.decode().strip())
    except Exception:
        return 0.0


# ── Subtitle / transcript fetch ──────────────────────────────────────────────

def _download_youtube_subs(url: str) -> list[TranscriptSegment]:
    """Fetch YouTube auto-subs in json3; return [] on any failure."""
    tmp_base = Path(tempfile.mkdtemp()) / "subs"
    try:
        with open(os.devnull, "r") as din, open(os.devnull, "w") as derr:
            subprocess.run([
                "yt-dlp", "--no-check-certificate",
                "--write-auto-sub", "--sub-lang", "en",
                "--sub-format", "json3", "--skip-download",
                "-o", str(tmp_base), url,
            ], stdin=din, stdout=subprocess.DEVNULL, stderr=derr, timeout=60)
        sub_file = Path(f"{tmp_base}.en.json3")
        if not sub_file.exists():
            return []
        data = json.loads(sub_file.read_text())
        out: list[TranscriptSegment] = []
        for ev in data.get("events", []):
            segs = ev.get("segs") or []
            if not segs:
                continue
            start_ms = ev.get("tStartMs", 0)
            dur_ms = ev.get("dDurationMs", 0)
            text = "".join(s.get("utf8", "") for s in segs).strip()
            if text and text != "\n":
                out.append(TranscriptSegment(
                    start_sec=start_ms / 1000.0,
                    end_sec=(start_ms + dur_ms) / 1000.0,
                    text=text.replace("\n", " "),
                ))
        return out
    except Exception:
        return []
    finally:
        try:
            shutil.rmtree(tmp_base.parent, ignore_errors=True)
        except Exception:
            pass


_whisper_model = None


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel  # type: ignore
        local = Path("/Users/i45367/models/whisper-small")
        path = str(local) if local.exists() else "small"
        _whisper_model = WhisperModel(path, device="cpu", compute_type="int8")
    except Exception as e:
        _log(f"whisper unavailable: {e}")
        _whisper_model = None
    return _whisper_model


def _whisper_transcribe(media_path: Path) -> list[TranscriptSegment]:
    model = _load_whisper()
    if model is None:
        return []
    try:
        with MEDIA:
            segs, _ = model.transcribe(str(media_path), beam_size=3, language="en")
        return [
            TranscriptSegment(start_sec=s.start, end_sec=s.end, text=s.text.strip())
            for s in segs
        ]
    except Exception as e:
        _log(f"whisper failed: {e}")
        return []


def _fetch_transcript(media_path: Path, source_url: Optional[str]) -> list[TranscriptSegment]:
    if source_url and YOUTUBE_RE.match(source_url):
        _log("fetching YouTube auto-subs (json3)")
        subs = _download_youtube_subs(source_url)
        if subs:
            _log(f"  got {len(subs)} subtitle segments")
            return subs
        _log("  no subs; falling back to whisper")
    return _whisper_transcribe(media_path)


# ── Frames: scene detection + pHash dedup ────────────────────────────────────

def _phash_image(path: Path) -> int:
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(path),
            "-vf", "scale=8:8,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
        ], capture_output=True, timeout=5)
        pixels = list(result.stdout[:64])
        if len(pixels) < 64:
            return hash(result.stdout)
        avg = sum(pixels) / len(pixels)
        bits = 0
        for i, p in enumerate(pixels):
            if p > avg:
                bits |= (1 << i)
        return bits
    except Exception:
        return hash(path.name)


def _hamming(a: int, b: int) -> int:
    x, n = a ^ b, 0
    while x:
        n += x & 1
        x >>= 1
    return n


def _extract_scene_frames(video_path: Path, frames_dir: Path) -> list[Frame]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = frames_dir / "_raw"
    raw_dir.mkdir(exist_ok=True)
    showinfo_log = ""
    try:
        with MEDIA:
            res = subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"select='gt(scene\\,{SCENE_THRESHOLD})',showinfo",
                "-vsync", "vfr",
                str(raw_dir / "scene_%04d.png"),
            ], capture_output=True, text=True, timeout=600)
        showinfo_log = res.stderr or ""
    except Exception as e:
        _log(f"ffmpeg scene extract failed: {e}")
        return []

    raw = sorted(raw_dir.glob("scene_*.png"))
    pts_times = [float(t) for t in re.findall(r"pts_time:([\d.]+)", showinfo_log)]

    if not raw:
        # Regular-interval fallback.
        duration = _probe_duration(video_path) or 1.0
        interval = max(30.0, duration / max(MAX_SCENE_FRAMES, 1))
        try:
            with MEDIA:
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(video_path),
                    "-vf", f"fps=1/{interval}",
                    "-vsync", "vfr",
                    str(raw_dir / "scene_%04d.png"),
                ], capture_output=True, timeout=300)
            raw = sorted(raw_dir.glob("scene_*.png"))
            pts_times = [i * interval for i in range(len(raw))]
        except Exception:
            pass

    duration = _probe_duration(video_path) or 1.0
    out: list[Frame] = []
    seen: list[int] = []
    for i, fp in enumerate(raw):
        ph = _phash_image(fp)
        if any(_hamming(ph, h) < PHASH_DEDUP_DIST for h in seen):
            fp.unlink(missing_ok=True)
            continue
        seen.append(ph)
        ts = pts_times[i] if i < len(pts_times) else (i / max(len(raw), 1)) * duration
        idx = len(out)
        final = frames_dir / f"fig_{idx:03d}.png"
        fp.rename(final)
        out.append(Frame(timestamp_sec=ts, path=final, phash=ph))
        if len(out) >= MAX_SCENE_FRAMES:
            break

    for leftover in raw_dir.glob("scene_*.png"):
        leftover.unlink(missing_ok=True)
    try:
        raw_dir.rmdir()
    except OSError:
        pass

    _log(f"scene frames: {len(raw)} raw → {len(out)} unique")
    return out


def _write_frame_manifest(frames: list[Frame], segments: list[TranscriptSegment], frames_dir: Path) -> None:
    manifest = []
    for i, f in enumerate(frames):
        caption = ""
        if segments:
            closest = min(segments, key=lambda s: abs(s.start_sec - f.timestamp_sec))
            caption = (closest.text or "").strip()[:200]
        # Strip debug crumbs that occasionally bleed into v3-era captions.
        if caption.startswith("keyword_hits=") or "visual_score=" in caption:
            caption = ""
        # Attach the cleaned caption back to the Frame so downstream layers
        # don't have to re-derive it.
        f.caption = caption
        manifest.append({
            "index": i,
            "timestamp_sec": f.timestamp_sec,
            "phash": f.phash,
            "path": str(f.path),
            "caption": caption,
        })
    (frames_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── PDF ingest ───────────────────────────────────────────────────────────────

def _pdf_ingest(pdf_path: Path, ws: Path) -> tuple[list[TranscriptSegment], list[Frame]]:
    frames_dir = ws / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    segments: list[TranscriptSegment] = []
    frames: list[Frame] = []

    # Try PyMuPDF first.
    try:
        import fitz  # type: ignore
    except Exception:
        fitz = None  # type: ignore

    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                text = (page.get_text() or "").strip()
                segments.append(TranscriptSegment(start_sec=float(i), end_sec=float(i + 1), text=text))
                pix = page.get_pixmap(dpi=120)
                out = frames_dir / f"fig_{i:03d}.png"
                pix.save(str(out))
                frames.append(Frame(timestamp_sec=float(i), path=out))
            doc.close()
            return segments, frames
        except Exception as e:
            _log(f"PyMuPDF failed: {e}; trying fallbacks")

    # Text fallback: pdfplumber → pdfminer.
    text_pages: list[str] = []
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf_path)) as pdf:
            text_pages = [(p.extract_text() or "").strip() for p in pdf.pages]
    except Exception:
        try:
            from pdfminer.high_level import extract_text  # type: ignore
            full = extract_text(str(pdf_path)) or ""
            text_pages = [full]
        except Exception as e:
            _log(f"no PDF text backend available: {e}")

    for i, text in enumerate(text_pages):
        segments.append(TranscriptSegment(start_sec=float(i), end_sec=float(i + 1), text=text))

    # Render pages with pdftoppm if available.
    if shutil.which("pdftoppm"):
        try:
            with MEDIA:
                subprocess.run([
                    "pdftoppm", "-r", "120", "-png",
                    str(pdf_path), str(frames_dir / "fig"),
                ], check=True, capture_output=True, timeout=300)
            for i, fp in enumerate(sorted(frames_dir.glob("fig-*.png"))):
                final = frames_dir / f"fig_{i:03d}.png"
                fp.rename(final)
                frames.append(Frame(timestamp_sec=float(i), path=final))
        except Exception as e:
            _log(f"pdftoppm failed: {e}")

    return segments, frames


# ── Text ingest ──────────────────────────────────────────────────────────────

def _text_ingest(text_path: Path) -> list[TranscriptSegment]:
    raw = text_path.read_text(encoding="utf-8", errors="replace")
    words = raw.split()
    segments: list[TranscriptSegment] = []
    chunk = 150
    spacing = 30.0
    for i in range(0, len(words), chunk):
        chunk_words = words[i:i + chunk]
        if not chunk_words:
            continue
        idx = i // chunk
        segments.append(TranscriptSegment(
            start_sec=idx * spacing,
            end_sec=(idx + 1) * spacing,
            text=" ".join(chunk_words),
        ))
    if not segments:
        segments.append(TranscriptSegment(start_sec=0.0, end_sec=spacing, text=raw.strip()))
    return segments


# ── Window aggregation ──────────────────────────────────────────────────────

def _build_windows(segments: list[TranscriptSegment], frames: list[Frame]) -> list[TranscriptWindow]:
    windows: list[TranscriptWindow] = []
    if not segments:
        return windows
    frame_ts = [f.timestamp_sec for f in frames]

    cur_start: Optional[float] = None
    cur_end: float = 0.0
    cur_texts: list[str] = []

    def flush() -> None:
        if cur_start is None or not cur_texts:
            return
        text = " ".join(cur_texts).strip()
        words = set(re.findall(r"[a-zA-Z]+", text.lower()))
        kw_hits = len(words & HIGH_INFO_KEYWORDS)
        has_frame = any(cur_start <= ts < cur_end for ts in frame_ts)
        windows.append(TranscriptWindow(
            start_sec=cur_start,
            end_sec=cur_end,
            text=text,
            is_high_info=(kw_hits >= 2 or has_frame),
            keyword_hits=kw_hits,
        ))

    for seg in segments:
        if cur_start is None:
            cur_start = seg.start_sec
        prospective_end = seg.end_sec
        span = prospective_end - cur_start
        # If adding this segment would exceed the hard cap, flush first.
        if cur_texts and span > WINDOW_MAX_SEC:
            flush()
            cur_start = seg.start_sec
            cur_texts = []
        cur_end = seg.end_sec
        cur_texts.append(seg.text)
        if (cur_end - cur_start) >= WINDOW_MIN_SEC and (cur_end - cur_start) >= WINDOW_TARGET_SEC * 0.95:
            # We are inside the [60, 120] band and at/over target; close the window.
            flush()
            cur_start = None
            cur_end = 0.0
            cur_texts = []

    if cur_texts:
        flush()

    # Merge a trailing too-small window into its predecessor when the combined
    # span still fits inside the hard cap. Avoids dribble windows like 40s after
    # an 80s window.
    if len(windows) >= 2:
        last = windows[-1]
        prev = windows[-2]
        if (last.end_sec - last.start_sec) < WINDOW_MIN_SEC and (last.end_sec - prev.start_sec) <= WINDOW_MAX_SEC:
            merged_text = (prev.text + " " + last.text).strip()
            words = set(re.findall(r"[a-zA-Z]+", merged_text.lower()))
            kw_hits = len(words & HIGH_INFO_KEYWORDS)
            has_frame = any(prev.start_sec <= ts < last.end_sec for ts in frame_ts)
            windows[-2:] = [TranscriptWindow(
                start_sec=prev.start_sec,
                end_sec=last.end_sec,
                text=merged_text,
                is_high_info=(kw_hits >= 2 or has_frame),
                keyword_hits=kw_hits,
            )]

    return windows


# ── Persistence ──────────────────────────────────────────────────────────────

def _serialize_segment(s: TranscriptSegment) -> dict:
    return asdict(s)


def _serialize_window(w: TranscriptWindow) -> dict:
    return asdict(w)


def _persist(source: Source) -> None:
    ws = source.workspace
    (ws / "transcript").mkdir(parents=True, exist_ok=True)
    (ws / "transcript" / "segments.json").write_text(
        json.dumps([_serialize_segment(s) for s in source.segments], indent=2, ensure_ascii=False)
    )
    (ws / "transcript" / "windows.json").write_text(
        json.dumps([_serialize_window(w) for w in source.windows], indent=2, ensure_ascii=False)
    )
    meta = {
        "source_id": source.source_id,
        "kind": source.spec.kind,
        "title": source.meta.get("title"),
        "source_url": source.meta.get("source_url"),
        "duration_sec": source.duration_sec,
        "n_segments": len(source.segments),
        "n_windows": len(source.windows),
        "n_frames": len(source.frames),
        "media_path": str(source.media_path) if source.media_path else None,
        "ingest_started_at": source.meta.get("ingest_started_at"),
        "ingest_finished_at": source.meta.get("ingest_finished_at"),
        "ingest_seconds": source.meta.get("ingest_seconds"),
    }
    (ws / "meta.json").write_text(json.dumps(meta, indent=2))


# ── Public API ───────────────────────────────────────────────────────────────

def ingest(spec: SourceSpec, run_workspace: Path) -> Source:
    """Run Layer 1 (INGEST) for a single ``SourceSpec``.

    Side effects: writes media, frames, transcript JSON, and meta.json into
    a per-source workspace under ``run_workspace/sources/<source_id>``.

    Raises only on critical failures (no usable media at all). Sub-task
    failures (subtitles, scene extract) degrade gracefully.
    """
    started = time.time()
    sid = stable_source_id(spec)
    ws = source_workspace(run_workspace, sid)
    kind = _sniff_kind(spec)
    _log(f"start sid={sid} kind={kind} loc={spec.location[:80]}")

    media_path: Optional[Path] = None
    source_url: Optional[str] = None
    segments: list[TranscriptSegment] = []
    frames: list[Frame] = []
    video_title: Optional[str] = spec.title  # may be enriched below

    try:
        if kind in ("url", "video"):
            media_path, source_url = _acquire_media(kind, spec.location, ws / "media")
            if not media_path or not media_path.exists():
                raise RuntimeError(f"media not on disk after acquire: {media_path}")

            # Fetch video title if not already provided (fast metadata-only call).
            if not video_title and source_url:
                video_title = _fetch_yt_title(source_url)

            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_subs = pool.submit(_fetch_transcript, media_path, source_url)
                fut_frames = pool.submit(_extract_scene_frames, media_path, ws / "frames")
                try:
                    segments = fut_subs.result()
                except Exception as e:
                    _log(f"transcript task failed: {e}")
                    segments = []
                try:
                    frames = fut_frames.result()
                except Exception as e:
                    _log(f"frame task failed: {e}")
                    frames = []

        elif kind == "audio":
            media_path, _ = _acquire_media("audio", spec.location, ws / "media")
            segments = _whisper_transcribe(media_path)

        elif kind == "pdf":
            p = Path(spec.location)
            if not p.exists():
                raise FileNotFoundError(f"pdf not found: {spec.location}")
            target = ws / "media" / "source.pdf"
            if not target.exists():
                try:
                    os.symlink(p.resolve(), target)
                except OSError:
                    shutil.copy2(p, target)
            media_path = target
            segments, frames = _pdf_ingest(target, ws)

        elif kind == "text":
            p = Path(spec.location)
            if not p.exists():
                raise FileNotFoundError(f"text not found: {spec.location}")
            target = ws / "media" / f"source{p.suffix}"
            if not target.exists():
                try:
                    os.symlink(p.resolve(), target)
                except OSError:
                    shutil.copy2(p, target)
            media_path = target
            segments = _text_ingest(target)

        else:  # pragma: no cover — _sniff_kind already validated
            raise ValueError(f"unsupported kind: {kind}")

    except Exception as e:
        # Truly critical: no segments AND no media. Re-raise.
        if media_path is None:
            _log(f"FATAL ingest failure: {e}")
            raise

    duration = _probe_duration(media_path) if media_path and kind in ("url", "video", "audio") else (
        float(len(segments)) if kind in ("pdf", "text") else 0.0
    )

    windows = _build_windows(segments, frames)

    # Frame manifest (also written for pdf where frames are pages).
    if frames:
        try:
            _write_frame_manifest(frames, segments, ws / "frames")
        except Exception as e:
            _log(f"manifest write failed: {e}")

    finished = time.time()
    source = Source(
        source_id=sid,
        spec=spec,
        workspace=ws,
        media_path=media_path,
        duration_sec=float(duration),
        segments=segments,
        windows=windows,
        frames=frames,
        meta={
            "title": video_title,
            "source_url": source_url,
            "ingest_started_at": started,
            "ingest_finished_at": finished,
            "ingest_seconds": round(finished - started, 3),
            "kind_resolved": kind,
        },
    )
    _persist(source)
    _log(
        f"done sid={sid} segs={len(segments)} windows={len(windows)} "
        f"frames={len(frames)} dur={duration:.1f}s in {finished - started:.1f}s"
    )
    return source
