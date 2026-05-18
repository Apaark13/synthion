"""engine_v4.index — KV / embedding context store for L2 UNDERSTAND.

Ports the algorithm from agents/03_kv_cache_agent.py but exposes it as a
``Context`` class so multiple sources can each own an independent index
without colliding on a single shared cache file.

Storage layout (per ``Context.save(path)``):

    path                # an .npz produced by numpy.savez_compressed with:
        keys            # uint8, shape (N, D)  — 4-bit min-max quantised
        vmin, scale     # float32 (1,)
        sentences       # object array of original window texts
        order           # int32  (N,)  — original temporal index
        bits            # int8   (1,)
        dim             # int32  (1,)
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np

from config.settings import KV_QUANTIZE_BITS, MINILM_LOCAL
from .types import TranscriptWindow


# ── Embedding backend ────────────────────────────────────────────────────────

_st_model = None
_st_tried = False


def _get_st_model():
    """Load local MiniLM with offline mode; return None if unavailable."""
    global _st_model, _st_tried
    if _st_tried:
        return _st_model
    _st_tried = True
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        from sentence_transformers import SentenceTransformer
        model_path = str(MINILM_LOCAL) if MINILM_LOCAL.exists() else "all-MiniLM-L6-v2"
        _st_model = SentenceTransformer(model_path)
    except Exception:
        _st_model = None
    return _st_model


def _embed(sentences: list[str]) -> np.ndarray:
    """(N, D) float32, L2-normalised embedding matrix."""
    model = _get_st_model()
    if model is not None:
        try:
            vecs = model.encode(sentences, convert_to_numpy=True,
                                normalize_embeddings=True)
            return vecs.astype(np.float32)
        except Exception:
            pass

    # Deterministic char-hash fallback (128-dim) — ported from v3.
    D = 128
    out = np.zeros((len(sentences), D), dtype=np.float32)
    for i, s in enumerate(sentences):
        h = hashlib.md5(s.encode("utf-8")).digest()
        for j in range(min(D, len(h))):
            out[i, j] = (h[j] - 128) / 128.0
    norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out / norms


# ── Quantisation helpers (4-bit min-max) ─────────────────────────────────────

def _quantize(arr: np.ndarray, bits: int = KV_QUANTIZE_BITS):
    vmin = float(arr.min())
    vmax = float(arr.max())
    scale = (vmax - vmin) / (2 ** bits - 1) if vmax != vmin else 1.0
    q = np.round((arr - vmin) / scale).astype(np.uint8 if bits <= 8 else np.uint16)
    return q, vmin, scale


def _dequantize(q: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale + vmin


# ── Context class ────────────────────────────────────────────────────────────

class Context:
    """In-memory + persistable embedding store over TranscriptWindows."""

    def __init__(self) -> None:
        self._keys_q: Optional[np.ndarray] = None      # (N, D) uint8
        self._vmin: float = 0.0
        self._scale: float = 1.0
        self._texts: list[str] = []
        self._order: list[int] = []
        self._bits: int = KV_QUANTIZE_BITS

    # ---------- write ----------
    def store(self, windows: list[TranscriptWindow]) -> None:
        texts: list[str] = []
        order: list[int] = []
        for i, w in enumerate(windows):
            t = (w.text or "").strip()
            if not t:
                continue
            texts.append(t)
            order.append(i)
        if not texts:
            self._keys_q = np.zeros((0, 0), dtype=np.uint8)
            self._texts = []
            self._order = []
            return

        vecs = _embed(texts)                       # (N, D) float32
        q, vmin, scale = _quantize(vecs, self._bits)
        self._keys_q = q
        self._vmin = vmin
        self._scale = scale
        self._texts = texts
        self._order = order

    # ---------- read ----------
    def retrieve(self, query: str, top_k: int = 8) -> str:
        if self._keys_q is None or len(self._texts) == 0:
            return ""
        keys = _dequantize(self._keys_q, self._vmin, self._scale)
        q_vec = _embed([query])[0]
        scores = keys @ q_vec
        k = min(top_k, len(self._texts))
        top_idx = np.argpartition(-scores, k - 1)[:k] if k < len(scores) \
            else np.arange(len(scores))
        # Re-order chronologically (original window index) so retrieved
        # context reads forward in time, not by score.
        chrono = sorted(top_idx, key=lambda i: self._order[i])
        return " ".join(self._texts[i] for i in chrono)

    # ---------- persistence ----------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._keys_q is None:
            self._keys_q = np.zeros((0, 0), dtype=np.uint8)
        np.savez_compressed(
            path,
            keys=self._keys_q,
            vmin=np.array([self._vmin], dtype=np.float32),
            scale=np.array([self._scale], dtype=np.float32),
            sentences=np.array(self._texts, dtype=object),
            order=np.array(self._order, dtype=np.int32),
            bits=np.array([self._bits], dtype=np.int8),
            dim=np.array([self._keys_q.shape[1] if self._keys_q.ndim == 2 else 0],
                         dtype=np.int32),
        )

    @classmethod
    def load(cls, path: Path) -> "Context":
        data = np.load(Path(path), allow_pickle=True)
        ctx = cls()
        ctx._keys_q = np.asarray(data["keys"])
        ctx._vmin = float(data["vmin"][0])
        ctx._scale = float(data["scale"][0])
        ctx._texts = [str(s) for s in data["sentences"]]
        ctx._order = [int(i) for i in data["order"]] if "order" in data.files \
            else list(range(len(ctx._texts)))
        ctx._bits = int(data["bits"][0]) if "bits" in data.files else KV_QUANTIZE_BITS
        return ctx
