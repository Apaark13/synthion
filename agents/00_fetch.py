"""agents/00_fetch.py — v3 Optimized Fetch

Single-pass: Download 360p stream only. No 4K segment downloads.
Scene-change frame extraction handled by 01_multimodal_extract.py.
Saves YouTube URL alongside video for subtitle download.

Outputs:
  workspace/video/low_res/<title>.mp4        (360p full video)
  workspace/video/low_res/.<title>.url       (YouTube URL for sub download)
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from config.settings import (
    VIDEO_LOW_RES_DIR,
    LOW_RES_FORMAT,
    ERRORS_LOG,
)
from agents.template import exponential_backoff, log_error, cleanup_temp


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ydlp_download(url: str, fmt: str, out_dir: Path, extra_opts: list | None = None) -> Path:
    """Run yt-dlp and return the newest file produced in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "--format", fmt,
        "--output", str(out_dir / "%(title)s.%(ext)s"),
        "--merge-output-format", "mp4",
    ] + (extra_opts or []) + [url]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    files = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"yt-dlp produced no .mp4 in {out_dir}")
    return files[0]


def _video_duration(video_path: Path) -> float:
    """Return duration in seconds via ffprobe."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ], stderr=subprocess.DEVNULL)
    return float(out.decode().strip())


def _update_checkpoint(task_name: str, extra: dict | None = None):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 1
    data["batch"] = 1
    data["step"] = 0
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    if extra:
        data.update(extra)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def step_0(url: str) -> dict:
    """Single-pass fetch: 360p only. Fast.

    Returns:
        {
            "low_res_path": Path,
            "hq_segments": [],          # v3: no HQ downloads
            "flagged_timestamps": [],   # v3: handled by extract step
            "video_duration_sec": float,
        }
    """
    def work():
        print(f"[fetch] Downloading 360p stream...")
        low_res_path = _ydlp_download(url, LOW_RES_FORMAT, VIDEO_LOW_RES_DIR)
        print(f"[fetch] ✓ Downloaded: {low_res_path.name}")

        # Save URL for YouTube subtitle download in step 1
        url_file = low_res_path.parent / f".{low_res_path.stem}.url"
        url_file.write_text(url)

        duration_sec = 0.0
        try:
            duration_sec = _video_duration(low_res_path)
            print(f"[fetch] Duration: {duration_sec/60:.1f} min")
        except Exception:
            pass

        return {
            "low_res_path": low_res_path,
            "hq_segments": [],
            "flagged_timestamps": [],
            "video_duration_sec": duration_sec,
        }

    try:
        result = exponential_backoff(work, step=0)
        _update_checkpoint(
            "agents/00_fetch.py v3 optimized",
            extra={"video_path": str(result["low_res_path"])}
        )
        _append_done(
            f"[v6] {datetime.now(timezone.utc).date()} · Copilot · "
            "00_fetch: v3 single-pass 360p (no HQ segments)"
        )
        cleanup_temp()
        return result
    except Exception as e:
        log_error(0, e)
        raise


def get_playlist_videos(playlist_url: str) -> list[dict]:
    """Return list of {url, title, duration_sec} for a YouTube playlist.

    Uses yt-dlp --flat-playlist (no download). Fast.
    """
    import json as _json
    import tempfile, os
    cmd = [
        "yt-dlp",
        "--no-check-certificate",
        "--flat-playlist",
        "--dump-single-json",
        playlist_url,
    ]
    # Write to temp file to avoid FD issues in nohup/detached contexts
    tmp_path = Path(tempfile.mktemp(suffix=".json"))
    try:
        with open(tmp_path, "w") as out_fh, open(os.devnull, "r") as devnull_in, \
             open(os.devnull, "w") as devnull_err:
            result = subprocess.run(
                cmd, stdout=out_fh, stderr=devnull_err,
                stdin=devnull_in, timeout=90,
            )
        if result.returncode != 0:
            import time; time.sleep(3)
            with open(tmp_path, "w") as out_fh, open(os.devnull, "r") as devnull_in, \
                 open(os.devnull, "w") as devnull_err:
                result = subprocess.run(
                    cmd, stdout=out_fh, stderr=devnull_err,
                    stdin=devnull_in, timeout=90,
                )
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp --flat-playlist returned code {result.returncode}")
        data = _json.loads(tmp_path.read_text())
        entries = data.get("entries", [])
        videos = []
        for e in entries:
            if not e:
                continue
            video_id = e.get("id") or e.get("url", "")
            url = (
                f"https://www.youtube.com/watch?v={video_id}"
                if not str(video_id).startswith("http")
                else str(video_id)
            )
            videos.append({
                "url": url,
                "title": e.get("title", str(video_id)),
                "duration_sec": float(e.get("duration") or 0),
            })
        return videos
    except Exception as ex:
        log_error(0, ex)
        raise
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agents.00_fetch <youtube-url>")
        sys.exit(1)
    out = step_0(sys.argv[1])
    print(f"\n[fetch] Done. low_res={out['low_res_path']}, "
          f"hq_segments={len(out['hq_segments'])}, "
          f"timestamps={len(out['flagged_timestamps'])}, "
          f"duration={out.get('video_duration_sec', 0)/60:.1f}min")

