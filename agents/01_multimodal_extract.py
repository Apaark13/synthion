"""agents/01_multimodal_extract.py — v3 Optimized Multimodal Extraction

Optimized pipeline:
  Phase A: Download YouTube subtitles via yt-dlp (2 seconds).
           Falls back to faster-whisper only if no subtitles available.
  Phase B: Smart scene-change frame detection via ffmpeg scene filter.
           pHash deduplication to eliminate repeated/similar frames.
  Phase C: Heuristic is_high_info classification using keyword + visual scoring.
           (No per-segment LLM call — massive speedup.)

Input:  workspace/video/low_res/<title>.mp4
Output: workspace/transcripts/<stem>_multimodal_log.json
        schema: [{timestamp_sec, text, visual_desc, is_high_info}]
        workspace/images/scene_*.png (deduplicated scene-change frames)
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import (
    VIDEO_LOW_RES_DIR,
    TRANSCRIPT_DIR,
    IMAGES_DIR,
    FRAME_DIFF_THRESHOLD,
    ERRORS_LOG,
)
from agents.template import exponential_backoff, log_error, cleanup_temp


# ── Phase A: YouTube Subtitle Download ────────────────────────────────────────

def _download_youtube_subs(video_path: Path) -> list[dict] | None:
    """Try to download YouTube auto-subs via yt-dlp. Returns segments or None.

    Uses json3 format for word-level timing. ~2 seconds total.
    """
    # Attempt to find the YouTube URL from the video filename or metadata
    # First check if we have a cached URL
    url_file = video_path.parent / f".{video_path.stem}.url"
    if not url_file.exists():
        return None

    url = url_file.read_text().strip()
    if not url:
        return None

    tmp_base = Path(tempfile.mktemp(suffix=""))
    try:
        with open(os.devnull, "r") as devnull_in, \
             open(os.devnull, "w") as devnull_err:
            result = subprocess.run([
                "yt-dlp", "--no-check-certificate",
                "--write-auto-sub", "--sub-lang", "en",
                "--sub-format", "json3", "--skip-download",
                "-o", str(tmp_base),
                url,
            ], stdin=devnull_in, stdout=subprocess.DEVNULL,
               stderr=devnull_err, timeout=30)

        sub_file = Path(f"{tmp_base}.en.json3")
        if not sub_file.exists():
            return None

        data = json.loads(sub_file.read_text())
        events = data.get("events", [])

        segments = []
        for event in events:
            segs = event.get("segs", [])
            if not segs:
                continue
            start_ms = event.get("tStartMs", 0)
            dur_ms = event.get("dDurationMs", 0)
            text = "".join(s.get("utf8", "") for s in segs).strip()
            if text and text != "\n":
                segments.append({
                    "start": start_ms / 1000.0,
                    "end": (start_ms + dur_ms) / 1000.0,
                    "text": text.replace("\n", " "),
                })

        print(f"[extract] YouTube subs: {len(segments)} segments downloaded in ~2s")
        return segments if segments else None

    except Exception as e:
        log_error(1, e)
        return None
    finally:
        for ext in [".en.json3", ".en.vtt", ".en.srt"]:
            Path(f"{tmp_base}{ext}").unlink(missing_ok=True)


# ── Whisper fallback ──────────────────────────────────────────────────────────
_whisper_model = None
_WHISPER_LOCAL = Path("/Users/i45367/models/whisper-small")


def _load_whisper():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model
    try:
        from faster_whisper import WhisperModel
        model_path = str(_WHISPER_LOCAL) if _WHISPER_LOCAL.exists() else "small"
        _whisper_model = WhisperModel(model_path, device="cpu", compute_type="int8")
        print(f"[extract] faster-whisper loaded from {model_path}")
    except Exception as e:
        log_error(1, e)
        _whisper_model = None
    return _whisper_model


def _transcribe_video(video_path: Path) -> list[dict]:
    """Return list of {start, end, text} dicts from Whisper."""
    model = _load_whisper()
    if model is None:
        return []
    segments, _ = model.transcribe(str(video_path), beam_size=3, language="en")
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


# ── Phase B: Smart Scene-Change Frame Extraction ─────────────────────────────

def _phash_image(path: Path) -> int:
    """Compute perceptual hash of an image using ffmpeg resize + raw bytes.
    Returns a 64-bit integer hash.
    """
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(path),
            "-vf", "scale=8:8,format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "pipe:1",
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


def _hamming_distance(h1: int, h2: int) -> int:
    """Count differing bits between two hashes."""
    x = h1 ^ h2
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def _extract_scene_frames(
    video_path: Path,
    scene_threshold: float = 0.25,
    max_frames: int = 15,
    dedup_threshold: int = 10,
    out_dir: Path | None = None,
) -> list[dict]:
    """Extract frames at scene changes, deduplicate by perceptual hash.

    Returns list of {timestamp_sec, path, phash} for unique frames.
    Real PTS values are parsed from ffmpeg's ``showinfo`` filter output rather
    than estimated by index — this is what makes figure→text alignment honest.
    """
    target_dir = out_dir if out_dir is not None else IMAGES_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = target_dir / "_scene_raw"
    frames_dir.mkdir(exist_ok=True)

    # Step 1: Extract scene-change frames with ffmpeg + showinfo (PTS comes via stderr)
    showinfo_log = ""
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-vf", f"select='gt(scene\\,{scene_threshold})',showinfo",
            "-vsync", "vfr",
            str(frames_dir / "scene_%04d.png"),
        ], capture_output=True, text=True, timeout=300)
        showinfo_log = result.stderr or ""
    except Exception as e:
        log_error(1, e)
        return []

    raw_frames = sorted(frames_dir.glob("scene_*.png"))

    # Parse "pts_time:NNN.NNN" tokens from showinfo log — one per emitted frame.
    import re as _re
    pts_times = [float(t) for t in _re.findall(r"pts_time:([\d.]+)", showinfo_log)]

    if not raw_frames:
        # Fallback: extract at regular intervals.
        try:
            duration = _video_duration(video_path)
            interval = max(30, duration / max_frames)
            subprocess.run([
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"fps=1/{interval}",
                "-vsync", "vfr",
                str(frames_dir / "scene_%04d.png"),
            ], capture_output=True, timeout=120)
            raw_frames = sorted(frames_dir.glob("scene_*.png"))
            pts_times = [i * interval for i in range(len(raw_frames))]
        except Exception:
            pass

    duration = _video_duration(video_path)
    n_raw = len(raw_frames)

    # Step 2: Deduplicate using perceptual hash
    unique_frames: list[dict] = []
    seen_hashes: list[int] = []

    for i, frame_path in enumerate(raw_frames):
        ph = _phash_image(frame_path)

        # Check against all seen hashes for duplicates
        is_dup = False
        for seen_h in seen_hashes:
            if _hamming_distance(ph, seen_h) < dedup_threshold:
                is_dup = True
                break

        if is_dup:
            frame_path.unlink(missing_ok=True)
            continue

        seen_hashes.append(ph)
        # Real PTS when we have it; fall back to interpolation only if showinfo
        # parsing failed.
        ts = pts_times[i] if i < len(pts_times) else (i / max(n_raw, 1)) * duration

        # Rename to final location
        out_name = f"fig_{len(unique_frames):03d}_upscaled.png"
        out_path = target_dir / out_name
        frame_path.rename(out_path)

        unique_frames.append({
            "timestamp_sec": ts,
            "path": str(out_path),
            "phash": ph,
        })

        if len(unique_frames) >= max_frames:
            break

    # Cleanup remaining raw frames
    for f in frames_dir.glob("scene_*.png"):
        f.unlink(missing_ok=True)
    try:
        frames_dir.rmdir()
    except OSError:
        pass

    print(f"[extract] Scene frames: {n_raw} raw → {len(unique_frames)} unique (dedup, real PTS)")
    return unique_frames


# ── Phase C: Heuristic Classification (no LLM) ──────────────────────────────

_HIGH_INFO_KEYWORDS = frozenset({
    "equation", "equations", "formula", "proof", "theorem", "definition",
    "algorithm", "therefore", "key", "important", "matrix", "determinant",
    "eigenvalue", "vector", "inverse", "solution", "solve", "multiply",
    "elimination", "pivot", "column", "row", "rank", "null", "space",
    "linear", "combination", "independent", "dependent", "basis",
    "dimension", "projection", "orthogonal", "transpose", "symmetric",
    "positive", "definite", "singular", "factorization", "decomposition",
})


def _classify_segment(text: str, visual_score: float, timestamp: float) -> dict:
    """Fast heuristic classification — no LLM call needed.

    visual_desc carries the first salient sentence of the text so it stays
    useful for downstream prompts and figure captions (the v3 debug string
    "keyword_hits=N" was unhelpful for both humans and the writer LLM).
    """
    words = set(text.lower().split())
    keyword_hits = len(words & _HIGH_INFO_KEYWORDS)
    is_hi = keyword_hits >= 2 or visual_score > FRAME_DIFF_THRESHOLD

    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 8]
    snippet = (sentences[0] if sentences else text.strip())[:140]

    return {
        "timestamp_sec": timestamp,
        "text": text,
        "visual_desc": snippet,
        "is_high_info": is_hi,
        "_signals": {"keyword_hits": keyword_hits, "visual_score": round(visual_score, 2)},
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _video_duration(video_path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ], stderr=subprocess.DEVNULL)
    return float(out.decode().strip())


def _save_video_url(video_path: Path, url: str):
    """Cache the YouTube URL alongside the video for subtitle download."""
    url_file = video_path.parent / f".{video_path.stem}.url"
    url_file.write_text(url)


def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 1
    data["batch"] = 2
    data["step"] = 1
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def step_1(video_path: Path | None = None, segment_secs: float = 30.0) -> Path:
    """Run optimized multimodal extraction.

    Phase A: YouTube subtitle download (2s) or Whisper fallback (4 min).
    Phase B: ffmpeg scene-change detection + pHash dedup (~5s).
    Phase C: Heuristic is_high_info classification (instant).

    Args:
        video_path: Path to 360p .mp4. If None, uses newest in VIDEO_LOW_RES_DIR.
        segment_secs: Merging window for grouping transcript segments.

    Returns:
        Path to the written *_multimodal_log.json.
    """
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    def work():
        # Resolve video path
        if video_path is None:
            candidates = sorted(
                VIDEO_LOW_RES_DIR.glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                raise FileNotFoundError(f"No .mp4 found in {VIDEO_LOW_RES_DIR}")
            vid = candidates[0]
        else:
            vid = Path(video_path)

        duration = _video_duration(vid)
        print(f"[extract] Video: {vid.name} ({duration:.0f}s / {duration/60:.1f}min)")

        # Phase A: Get transcript (YouTube subs → Whisper fallback)
        print("[extract] Phase A — fetching transcript...")
        whisper_segs = _download_youtube_subs(vid)
        if whisper_segs is None:
            print("[extract]   YouTube subs not available; falling back to Whisper...")
            whisper_segs = _transcribe_video(vid)
        if not whisper_segs:
            whisper_segs = [
                {"start": i * segment_secs, "end": min((i + 1) * segment_secs, duration),
                 "text": f"[segment {i+1}]"}
                for i in range(max(1, int(duration / segment_secs)))
            ]

        # Phase B: Smart scene-change frame extraction
        print("[extract] Phase B — extracting scene-change frames...")
        scene_frames = _extract_scene_frames(vid, max_frames=12)

        # Build a set of frame timestamps for visual_score
        frame_timestamps = {f["timestamp_sec"] for f in scene_frames}

        # Phase C: Group transcript + classify
        print(f"[extract] Phase C — classifying {len(whisper_segs)} transcript segments")
        groups: list[dict] = []
        current_group: dict = {"start": None, "end": None, "texts": []}
        for seg in whisper_segs:
            if current_group["start"] is None:
                current_group["start"] = seg["start"]
            current_group["end"] = seg["end"]
            current_group["texts"].append(seg["text"])
            if seg["end"] - current_group["start"] >= segment_secs:
                groups.append(current_group)
                current_group = {"start": None, "end": None, "texts": []}
        if current_group["start"] is not None:
            groups.append(current_group)

        log_entries = []
        for grp in groups:
            ts = grp["start"]
            combined_text = " ".join(grp["texts"])

            # Visual score: 1.0 if near a scene-change frame, else 0.0
            visual_score = 0.0
            for ft in frame_timestamps:
                if abs(ft - ts) < segment_secs:
                    visual_score = 1.0
                    break

            entry = _classify_segment(combined_text, visual_score, ts)
            log_entries.append(entry)
            hi = "★" if entry["is_high_info"] else " "
            print(f"[extract] [{hi}] {ts:6.0f}s  {combined_text[:60]}")

        # Write manifest for scene frames — caption from nearest transcript line.
        manifest = []
        for f in scene_frames:
            ts = f["timestamp_sec"]
            caption = ""
            if log_entries:
                closest = min(log_entries, key=lambda e: abs(e.get("timestamp_sec", 0) - ts))
                caption = (closest.get("text") or "").strip()[:160]
            manifest.append({
                "index": len(manifest),
                "timestamp_sec": ts,
                "caption": caption,
                "path": f["path"],
            })
        manifest_path = IMAGES_DIR / "manifest.json"
        with open(manifest_path, "w") as mf:
            json.dump(manifest, mf, indent=2)

        out_path = TRANSCRIPT_DIR / f"{vid.stem}_multimodal_log.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(log_entries, f, indent=2, ensure_ascii=False)

        print(f"[extract] ✓ {len(log_entries)} log entries, {len(scene_frames)} scene frames → {out_path}")
        return out_path

    try:
        out = exponential_backoff(work, step=1)
        _update_checkpoint("agents/01_multimodal_extract.py v3 optimized")
        _append_done(
            f"[v6] {datetime.now(timezone.utc).date()} · Copilot · "
            "01_multimodal_extract: YouTube subs + scene-change + pHash dedup"
        )
        cleanup_temp()
        return out
    except Exception as e:
        log_error(1, e)
        raise


if __name__ == "__main__":
    import sys
    path_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out = step_1(path_arg)
    print(f"\n[extract] Done → {out}")
