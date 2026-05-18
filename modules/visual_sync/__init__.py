"""
visual_sync: video <-> transcript figure alignment.

High-level entry: extract_aligned_figures(video, segments, *, output_dir, ...)

Pipeline:
  1) scene detection (PySceneDetect ContentDetector w/ ffmpeg fallback)
  2) precise frame extraction at scene timestamps
  3) pHash dedup + slide-likeness gate
  4) optional OCR (tesseract / Apple Vision) for text relevance
  5) per-chapter global assignment of frames to transcript windows

See modules.visual_sync.types.AlignedFigure for the output shape.
"""

from .types import AlignedFigure, TranscriptSegment, VisualSyncConfig
from .pipeline import extract_aligned_figures

__all__ = [
    "AlignedFigure",
    "TranscriptSegment",
    "VisualSyncConfig",
    "extract_aligned_figures",
]
