"""engine_v4.workspace — run-level + per-source workspace management."""
from __future__ import annotations

import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .types import SourceSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = REPO_ROOT / "workspace_v4"


def _workspace_root() -> Path:
    """Resolve workspace root, allowing CLI override via $MTE_WORKSPACE_ROOT."""
    override = os.environ.get("MTE_WORKSPACE_ROOT")
    return Path(override).resolve() if override else WORKSPACE_ROOT


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def run_workspace(run_id: str | None = None) -> Path:
    rid = run_id or new_run_id()
    p = _workspace_root() / "runs" / rid
    p.mkdir(parents=True, exist_ok=True)
    return p


def stable_source_id(spec: SourceSpec) -> str:
    """Stable, sanitized id from canonical location + kind.

    Strips volatile YouTube list/index/timestamp params so playlist reruns
    or shared links collapse to the same source.
    """
    loc = spec.location.strip()
    loc = re.sub(r"[?&](list|index|t|si|pp)=[^&]*", "", loc)
    digest = hashlib.sha1(f"{spec.kind}|{loc}".encode()).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9]+", "-", loc.split("/")[-1])[:40].strip("-").lower()
    return f"{base or 'src'}_{digest}"


def source_workspace(run_ws: Path, source_id: str) -> Path:
    p = run_ws / "sources" / source_id
    for sub in ("media", "transcript", "frames", "chapters", "citations"):
        (p / sub).mkdir(parents=True, exist_ok=True)
    return p
