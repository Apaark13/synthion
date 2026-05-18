"""Precise frame extraction at known timestamps.

Uses output-side `-ss` for accuracy (decodes up to target). Slower than
input-side seek but exact, which matters when we already know the PTS.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

log = logging.getLogger(__name__)


def extract_frames(
    video_path: str | Path,
    timestamps: Iterable[float],
    output_dir: str | Path,
    *,
    prefix: str = "frame",
    fmt: str = "jpg",
) -> List[Tuple[float, str]]:
    """Extract one frame per timestamp; returns (ts, path) for successful ones."""
    video_path = str(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[float, str]] = []
    for ts in timestamps:
        out_path = output_dir / f"{prefix}_{ts:09.3f}.{fmt}"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", video_path, "-ss", f"{ts:.3f}",
            "-frames:v", "1", str(out_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            results.append((ts, str(out_path)))
        else:
            log.warning("[visual_sync.extract] ffmpeg failed at t=%.3f: %s", ts, proc.stderr.strip()[:200])
    return results
