"""Frame <-> transcript-segment alignment.

Composite score:
    match = w_v * visual + w_t * text + w_time * time_score
where
    visual: scene-change strength in [0,1]
    text:   token-overlap (Jaccard) of OCR vs transcript window in [0,1]
    time:   exp(-|dt| / tau) in [0,1]

Per section, frames are assigned greedily in descending match score with:
  * monotonic-time constraint (later frames -> later segments) optional,
  * margin filter (best must beat 2nd-best by min_margin),
  * absolute thresholds (min_match_score, max_time_gap_sec).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence, Tuple

from .types import AlignedFigure, CandidateFrame, TranscriptSegment, VisualSyncConfig

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")
_STOP = {
    "the", "and", "for", "this", "that", "with", "from", "are", "was", "were",
    "have", "has", "had", "you", "your", "but", "not", "any", "all", "can",
    "will", "would", "should", "could", "into", "out", "about", "what", "when",
    "where", "which", "who", "how", "they", "them", "their", "there", "than",
    "then", "also", "such", "some", "more", "most", "very", "just", "like",
    "one", "two", "three", "now", "see", "say", "said", "use", "used", "uses",
    "going", "really", "okay", "right", "well", "yeah", "know", "kind", "sort",
}


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if t.lower() not in _STOP}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _time_score(dt_sec: float, tau: float) -> float:
    return math.exp(-abs(dt_sec) / max(tau, 1e-6))


def score_pair(frame: CandidateFrame, seg: TranscriptSegment, cfg: VisualSyncConfig) -> Tuple[float, float, float, float]:
    """Returns (composite, visual, text, time)."""
    vis = max(0.0, min(1.0, frame.visual_score))
    text = _jaccard(_tokens(frame.ocr_text), _tokens(seg.text))
    tscore = _time_score(frame.timestamp_sec - seg.mid, cfg.time_tau_sec)
    composite = (
        cfg.weight_visual * vis
        + cfg.weight_text * text
        + cfg.weight_time * tscore
    )
    return composite, vis, text, tscore


def align_section(
    frames: Sequence[CandidateFrame],
    segments: Sequence[TranscriptSegment],
    *,
    section_id: str = "",
    cfg: Optional[VisualSyncConfig] = None,
) -> List[AlignedFigure]:
    """Greedy global assignment of frames to segments within one section."""
    cfg = cfg or VisualSyncConfig()
    if not frames or not segments:
        return []

    # Build candidate pairs within max_time_gap.
    candidates: List[Tuple[float, int, int, float, float, float]] = []
    for fi, f in enumerate(frames):
        # find segments whose midpoint is within max_time_gap
        for si, s in enumerate(segments):
            if abs(f.timestamp_sec - s.mid) > cfg.max_time_gap_sec:
                continue
            comp, vis, txt, ts = score_pair(f, s, cfg)
            if comp < cfg.min_match_score:
                continue
            candidates.append((comp, fi, si, vis, txt, ts))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # Per-frame margin filter: only keep frames whose best beats their 2nd-best by margin.
    best_per_frame: dict[int, list[float]] = {}
    for comp, fi, _si, *_ in candidates:
        best_per_frame.setdefault(fi, []).append(comp)
    eligible_frames = {
        fi for fi, scores in best_per_frame.items()
        if scores[0] - (scores[1] if len(scores) > 1 else 0.0) >= cfg.min_margin
    }

    used_frames: set[int] = set()
    used_segs: set[int] = set()
    last_seg_used = -1
    out: List[AlignedFigure] = []

    for comp, fi, si, vis, txt, ts in candidates:
        if fi in used_frames or si in used_segs:
            continue
        if fi not in eligible_frames:
            continue
        if cfg.enforce_monotonic and si < last_seg_used:
            continue
        if len(out) >= cfg.max_figures_per_section:
            break

        f = frames[fi]
        s = segments[si]
        cap = (f.ocr_text.strip().splitlines()[0] if f.ocr_text.strip() else s.text.strip())[:140]
        out.append(AlignedFigure(
            timestamp_sec=f.timestamp_sec,
            path=f.path,
            caption=cap,
            section_id=section_id,
            match_score=comp,
            visual_score=vis,
            text_score=txt,
            time_score=ts,
            ocr_text=f.ocr_text,
            matched_segment_idx=si,
        ))
        used_frames.add(fi)
        used_segs.add(si)
        if cfg.enforce_monotonic:
            last_seg_used = si

    out.sort(key=lambda a: a.timestamp_sec)
    return out
