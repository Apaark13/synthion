"""pHash dedup + slide-likeness scoring."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import imagehash
    from PIL import Image
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False
    imagehash = None  # type: ignore
    Image = None  # type: ignore


def phash(path: str | Path, hash_size: int = 8) -> Optional[int]:
    if not _PIL_OK:
        return None
    try:
        with Image.open(path) as img:
            h = imagehash.phash(img, hash_size=hash_size)
        return int(str(h), 16)
    except Exception as exc:
        log.warning("[visual_sync.dedupe] phash failed for %s: %s", path, exc)
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def is_solid_color(path: str | Path, std_threshold: float = 5.0) -> bool:
    """True if the frame is essentially uniform (near-blank slide / blackboard)."""
    if not _PIL_OK:
        return False
    try:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("L").resize((64, 64)), dtype=np.float32)
        return float(arr.std()) < std_threshold
    except Exception:
        return False


def slide_likeness(path: str | Path) -> float:
    """Cheap heuristic: edge density in [0,1]; >0.05 ~ slide-like, <0.02 ~ talking head."""
    if not _PIL_OK:
        return 0.0
    try:
        with Image.open(path) as img:
            g = np.asarray(img.convert("L").resize((256, 144)), dtype=np.float32)
        # finite-difference Sobel-ish gradient
        gx = np.abs(np.diff(g, axis=1))
        gy = np.abs(np.diff(g, axis=0))
        edges = (gx[:-1, :] > 25).astype(np.float32) + (gy[:, :-1] > 25).astype(np.float32)
        density = float((edges > 0).mean())
        # Map [0, 0.15] -> [0, 1]; clamp.
        return max(0.0, min(1.0, density / 0.15))
    except Exception:
        return 0.0


def dedupe_by_phash(
    items: List[tuple],
    *,
    hash_size: int = 8,
    threshold: int = 10,
) -> List[tuple]:
    """items: list of (timestamp, path, ...). Keeps first of each near-duplicate cluster."""
    kept: List[tuple] = []
    kept_hashes: List[int] = []
    for it in items:
        path = it[1]
        h = phash(path, hash_size=hash_size)
        if h is None:
            kept.append(it)
            continue
        if any(hamming(h, kh) <= threshold for kh in kept_hashes):
            continue
        kept.append(it)
        kept_hashes.append(h)
    return kept
