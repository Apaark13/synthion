"""Top-level orchestration for visual_sync.

extract_aligned_figures(video_path, segments, *, output_dir, sections=None, config=None)
  -> dict[section_id, list[AlignedFigure]]

If `sections` is None, the whole transcript is treated as one section "all".
Each section is a (section_id, [TranscriptSegment, ...]) pair.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import align, dedupe, extract, ocr, scene
from .types import AlignedFigure, CandidateFrame, TranscriptSegment, VisualSyncConfig

log = logging.getLogger(__name__)


def _build_candidates(
    video_path: str,
    output_dir: Path,
    cfg: VisualSyncConfig,
) -> List[CandidateFrame]:
    scenes = scene.detect_scenes(
        video_path,
        threshold=cfg.scene_threshold,
        min_scene_len_sec=cfg.min_scene_len_sec,
        ffmpeg_threshold=cfg.ffmpeg_scene_threshold,
        downscale_height=cfg.downscale_height,
    )
    if not scenes:
        log.warning("[visual_sync] no scenes detected in %s", video_path)
        return []

    pairs = extract.extract_frames(video_path, [t for t, _ in scenes], output_dir)
    score_by_ts = {round(t, 3): s for t, s in scenes}

    items: List[Tuple[float, str, float]] = []
    for ts, path in pairs:
        if dedupe.is_solid_color(path, std_threshold=cfg.drop_solid_color_std):
            continue
        items.append((ts, path, score_by_ts.get(round(ts, 3), 1.0)))

    items = dedupe.dedupe_by_phash(items, hash_size=cfg.phash_size, threshold=cfg.phash_dup_distance)

    ocr.configure(cfg.ocr_backend)
    candidates: List[CandidateFrame] = []
    for ts, path, vscore in items:
        sl = dedupe.slide_likeness(path)
        text = ocr.extract_text(path) if sl >= cfg.skip_ocr_below_slide_likeness else ""
        candidates.append(CandidateFrame(
            timestamp_sec=ts,
            path=path,
            visual_score=vscore,
            slide_likeness=sl,
            ocr_text=text,
        ))
    log.info("[visual_sync] %d candidate frames (ocr_backend=%s)", len(candidates), ocr.backend_name())
    return candidates


def extract_aligned_figures(
    video_path: str | Path,
    segments: Sequence[TranscriptSegment],
    *,
    output_dir: str | Path,
    sections: Optional[Sequence[Tuple[str, Sequence[TranscriptSegment]]]] = None,
    config: Optional[VisualSyncConfig] = None,
) -> Dict[str, List[AlignedFigure]]:
    cfg = config or VisualSyncConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    candidates = _build_candidates(str(video_path), frames_dir, cfg)
    if not candidates:
        return {}

    if sections is None:
        sections = [("all", list(segments))]

    out: Dict[str, List[AlignedFigure]] = {}
    for sec_id, sec_segments in sections:
        if not sec_segments:
            out[sec_id] = []
            continue
        sec_start = min(s.start for s in sec_segments)
        sec_end = max(s.end for s in sec_segments)
        # Filter frames to within the section's time window (with tolerance).
        local = [
            f for f in candidates
            if sec_start - cfg.max_time_gap_sec <= f.timestamp_sec <= sec_end + cfg.max_time_gap_sec
        ]
        out[sec_id] = align.align_section(local, sec_segments, section_id=sec_id, cfg=cfg)
    return out
