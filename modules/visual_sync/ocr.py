"""OCR backends: Apple Vision -> Tesseract -> no-op.

`extract_text(path)` always returns a string (possibly empty); never raises.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger(__name__)

_BACKEND: Optional[Callable[[str], str]] = None
_BACKEND_NAME: str = "none"


def _try_vision() -> Optional[Callable[[str], str]]:
    try:
        import objc  # noqa: F401
        import Vision  # type: ignore
        from Foundation import NSURL  # type: ignore
    except Exception:
        return None

    def _run(path: str) -> str:
        try:
            url = NSURL.fileURLWithPath_(path)
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(1)  # accurate
            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
            ok = handler.performRequests_error_([req], None)
            if not ok:
                return ""
            out: list[str] = []
            for obs in (req.results() or []):
                cand = obs.topCandidates_(1)
                if cand:
                    out.append(str(cand[0].string()))
            return "\n".join(out)
        except Exception as exc:
            log.warning("[visual_sync.ocr] vision failed: %s", exc)
            return ""

    return _run


def _try_tesseract() -> Optional[Callable[[str], str]]:
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:
        return None
    # ensure binary exists
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
    except Exception:
        return None

    def _run(path: str) -> str:
        try:
            return pytesseract.image_to_string(Image.open(path)) or ""
        except Exception as exc:
            log.warning("[visual_sync.ocr] tesseract failed: %s", exc)
            return ""

    return _run


def _resolve(backend: str) -> tuple[Callable[[str], str], str]:
    if backend == "none":
        return (lambda _p: ""), "none"
    if backend in ("vision", "auto"):
        v = _try_vision()
        if v is not None:
            return v, "vision"
        if backend == "vision":
            return (lambda _p: ""), "none"
    if backend in ("tesseract", "auto"):
        t = _try_tesseract()
        if t is not None:
            return t, "tesseract"
    return (lambda _p: ""), "none"


def configure(backend: str = "auto") -> str:
    """Pick a backend; returns the chosen name."""
    global _BACKEND, _BACKEND_NAME
    _BACKEND, _BACKEND_NAME = _resolve(backend)
    return _BACKEND_NAME


def extract_text(path: str | Path) -> str:
    if _BACKEND is None:
        configure("auto")
    return _BACKEND(str(path))  # type: ignore[misc]


def backend_name() -> str:
    if _BACKEND is None:
        configure("auto")
    return _BACKEND_NAME
