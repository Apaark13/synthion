"""Scene boundary detection.

Strategy:
  1) Try PySceneDetect ContentDetector on a downscaled copy of the video.
  2) Fall back to ffmpeg `select=gt(scene,T),showinfo` and parse pts_time
     from stderr.

Returns a list of (timestamp_sec, score) tuples in source-time coordinates.
"""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger(__name__)

_PTS_RE = re.compile(r"pts_time:([\d.]+)")
_SCENE_SCORE_RE = re.compile(r"scene_score:([\d.]+)")


def detect_scenes(
    video_path: str | Path,
    *,
    threshold: float = 27.0,
    min_scene_len_sec: float = 1.5,
    ffmpeg_threshold: float = 0.20,
    downscale_height: int = 360,
) -> List[Tuple[float, float]]:
    """Return [(start_time_sec, score), ...] for detected scenes.

    score is in [0,1]: higher = stronger change.
    """
    video_path = str(video_path)
    try:
        return _scenedetect(video_path, threshold, min_scene_len_sec, downscale_height)
    except Exception as exc:  # pragma: no cover (fallback path)
        log.warning("[visual_sync.scene] PySceneDetect failed (%s); falling back to ffmpeg", exc)
        return _ffmpeg_scenes(video_path, ffmpeg_threshold)


def _scenedetect(video_path: str, threshold: float, min_len: float, down_h: int) -> List[Tuple[float, float]]:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    fps = video.frame_rate or 30.0
    min_len_frames = max(1, int(min_len * fps))

    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len_frames))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()

    out: List[Tuple[float, float]] = []
    for start, _end in scenes:
        # ContentDetector doesn't expose per-cut score in get_scene_list; use threshold as nominal.
        out.append((float(start.get_seconds()), min(1.0, threshold / 100.0 + 0.5)))
    return out


def _ffmpeg_scenes(video_path: str, threshold: float) -> List[Tuple[float, float]]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr", "-f", "null", "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    out: List[Tuple[float, float]] = []
    for line in proc.stderr.splitlines():
        if "showinfo" not in line and "Parsed_showinfo" not in line:
            continue
        m = _PTS_RE.search(line)
        if not m:
            continue
        ts = float(m.group(1))
        score_m = _SCENE_SCORE_RE.search(line)
        score = float(score_m.group(1)) if score_m else 1.0
        out.append((ts, min(1.0, score)))
    return out
