"""engine_v4.figures — snap [FIG t=…] anchors AND auto-inject section frames.

v4.1 rewrite:

* ``resolve_anchors`` now (a) keeps the legacy snap-to-nearest behaviour for
  any anchor the writer placed, but (b) also returns the set of frame
  timestamps already covered so that callers can fill the rest of the
  chapter with unique additional frames.
* ``select_section_frames`` chooses N evenly-spaced unique frames whose
  timestamps lie within the section's time band — used by ``compose`` to
  guarantee every chapter has multiple relevant screenshots, even if the
  LLM forgot the anchors.
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

from .types import Frame, OutlineSection

log = logging.getLogger("engine_v4.figures")

_ANCHOR_RE = re.compile(
    r"\[FIG\s+t\s*=\s*(\d{1,2}):(\d{2}):(\d{2})\]"
    r"|"
    r"\[Figure:\s*tim[a-z]*timestamp\s*=\s*(\d{1,2}):(\d{2}):(\d{2})[^\]]*\]"
    r"|"
    r"\[Figure:\s*timestamp\s*=\s*(\d{1,2}):(\d{2}):(\d{2})[^\]]*\]",
    re.IGNORECASE,
)


def _hhmmss_to_sec(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s)


def _seconds_to_hhmmss(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _clean_caption(text: str) -> str:
    t = (text or "").strip().replace("\n", " ")
    if not t or t.startswith("keyword_hits=") or "visual_score=" in t:
        return ""
    # Collapse whitespace.
    return re.sub(r"\s+", " ", t)[:200]


def _format_image(frame: Frame, caption: Optional[str] = None) -> str:
    cap = _clean_caption(caption if caption is not None else frame.caption) or "Figure"
    hhmmss = _seconds_to_hhmmss(frame.timestamp_sec)
    path = str(frame.path)
    return (
        f'\n\n<figure class="mte-figure">\n'
        f'  <img src="{path}" alt="{cap}" />\n'
        f'  <figcaption><strong>t={hhmmss}.</strong> {cap}</figcaption>\n'
        f'</figure>\n'
    )


# ── Anchor snapping ──────────────────────────────────────────────────────────

def resolve_anchors(
    body_md: str,
    frames: Iterable[Frame],
    tolerance_sec: float = 180.0,
) -> tuple[str, list[Frame]]:
    """Replace ``[FIG t=…]`` anchors with HTML <figure> blocks pointing at
    the nearest captured ``Frame`` within ``tolerance_sec``. Anchors with
    no nearby frame are silently removed.

    Returns ``(new_body, used_frames)`` — ``used_frames`` is the
    deduplicated ordered list of frames that were inlined.
    """
    frames_list = list(frames)
    used: list[Frame] = []
    used_ids: set[int] = set()

    def _sub(m: re.Match) -> str:
        groups = [g for g in m.groups() if g is not None]
        if len(groups) < 3:
            return ""
        h, mm, s = groups[0], groups[1], groups[2]
        try:
            ts = _hhmmss_to_sec(h, mm, s)
        except ValueError:
            return ""
        if not frames_list:
            return ""
        nearest = min(frames_list, key=lambda f: abs(f.timestamp_sec - ts))
        if abs(nearest.timestamp_sec - ts) > tolerance_sec:
            return ""
        if id(nearest) not in used_ids:
            used.append(nearest)
            used_ids.add(id(nearest))
        return _format_image(nearest)

    return _ANCHOR_RE.sub(_sub, body_md), used


# ── Auto-injection helpers ───────────────────────────────────────────────────

def select_section_frames(
    section: OutlineSection,
    all_frames: list[Frame],
    n: int = 4,
    pad_sec: float = 30.0,
) -> list[Frame]:
    """Pick up to ``n`` unique frames whose timestamps fall inside this
    section's time band. Frames are returned in chronological order and
    evenly spread across the band.
    """
    if not all_frames or n <= 0:
        return []
    if not section.timestamps:
        return []
    lo, hi = min(section.timestamps), max(section.timestamps)
    pad = max(pad_sec, (hi - lo) * 0.10)
    band = [f for f in all_frames if (lo - pad) <= f.timestamp_sec <= (hi + pad)]
    if not band:
        # Fall back to the nearest frames around the section's midpoint.
        mid = (lo + hi) / 2.0
        band = sorted(all_frames, key=lambda f: abs(f.timestamp_sec - mid))[: n * 2]
    band = sorted(band, key=lambda f: f.timestamp_sec)
    if len(band) <= n:
        return band
    step = len(band) / n
    return [band[int(i * step)] for i in range(n)]


def inject_extra_frames(
    body_md: str,
    extras: list[Frame],
    already_used: list[Frame],
) -> tuple[str, list[Frame]]:
    """Append the extra frames not already inlined to the end of the body,
    in a small gallery. Returns ``(new_body, all_used_frames)``.

    Skips any frame already inlined (by id). The first extra is appended
    just after the chapter intro paragraphs (after the first sub-heading
    if present) so figures are spread throughout the chapter, not just at
    the bottom.
    """
    already_ids = {id(f) for f in already_used}
    new_frames = [f for f in extras if id(f) not in already_ids]
    if not new_frames:
        return body_md, list(already_used)

    # Drop perceptual duplicates of frames we already injected.
    used_phashes = {f.phash for f in already_used if f.phash}
    deduped: list[Frame] = []
    for f in new_frames:
        if f.phash and f.phash in used_phashes:
            continue
        deduped.append(f)
        if f.phash:
            used_phashes.add(f.phash)
    new_frames = deduped
    if not new_frames:
        return body_md, list(already_used)

    # Split: keep one figure inline after intro, rest in trailing gallery.
    lines = body_md.splitlines()
    insert_at = None
    seen_h2 = False
    for i, ln in enumerate(lines):
        if ln.startswith("## "):
            seen_h2 = True
            continue
        if seen_h2 and ln.startswith("### "):
            insert_at = i
            break

    inline = new_frames[0]
    gallery = new_frames[1:]

    if insert_at is not None:
        head_block = _format_image(inline)
        lines.insert(insert_at, head_block)
        body_md = "\n".join(lines)
    else:
        body_md = body_md + _format_image(inline)

    if gallery:
        body_md += "\n\n### Additional figures\n"
        for f in gallery:
            body_md += _format_image(f)

    all_used = list(already_used) + new_frames
    return body_md, all_used
