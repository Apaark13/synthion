"""Public dataclasses for the visual_sync module."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TranscriptSegment:
    """A single transcript line / caption block with timing in seconds."""
    start: float
    end: float
    text: str

    @property
    def mid(self) -> float:
        return (self.start + self.end) / 2.0


@dataclass
class CandidateFrame:
    """A frame extracted from the video before alignment."""
    timestamp_sec: float
    path: str
    visual_score: float = 1.0  # scene-change strength, 0-1
    slide_likeness: float = 0.0  # 0=talking head, 1=clean slide
    ocr_text: str = ""
    phash: Optional[int] = None


@dataclass
class AlignedFigure:
    """A frame matched to a transcript window, ready to embed in prose."""
    timestamp_sec: float
    path: str
    caption: str           # short auto-caption (OCR title or transcript snippet)
    section_id: str = ""   # caller-supplied chapter id (free-form)
    match_score: float = 0.0
    text_score: float = 0.0
    visual_score: float = 0.0
    time_score: float = 0.0
    ocr_text: str = ""
    matched_segment_idx: int = -1  # index into the segments list for that section


@dataclass
class VisualSyncConfig:
    """All knobs in one place. Defaults follow the research briefs."""
    # scene detection
    scene_threshold: float = 27.0          # PySceneDetect ContentDetector default-ish
    ffmpeg_scene_threshold: float = 0.20   # used only on ffmpeg fallback
    min_scene_len_sec: float = 1.5
    downscale_height: int = 360            # detect at low res, extract at source res

    # dedup
    phash_size: int = 8                    # 8x8 = 64 bit
    phash_dup_distance: int = 10
    drop_solid_color_std: float = 5.0      # ignore near-uniform frames (e.g. blackboard between slides)

    # slide-likeness gate (Canny edge density on resized 256-wide grayscale)
    slide_min_edge_density: float = 0.025
    skip_ocr_below_slide_likeness: float = 0.10

    # OCR
    ocr_backend: str = "auto"              # auto|vision|tesseract|none

    # alignment scoring weights (must sum ~1)
    weight_visual: float = 0.45
    weight_text: float = 0.35
    weight_time: float = 0.20
    time_tau_sec: float = 60.0             # exp(-|dt|/tau)
    max_time_gap_sec: float = 180.0
    min_match_score: float = 0.35          # below this => leave unplaced
    min_margin: float = 0.05               # 1st-best must beat 2nd by >= this

    # output
    max_figures_per_section: int = 6
    enforce_monotonic: bool = True
