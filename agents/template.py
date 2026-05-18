"""
agents/template.py – HARDENED TEMPLATE (v2)
────────────────────────────────────────────
Patterns: try/except all IO/network, exponential backoff (max 3),
context managers for files, log exceptions to ERRORS_LOG with UTC timestamp
+ step name. All step stubs have been moved to individual agent files.
"""
import time
import hashlib
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from config.settings import ERRORS_LOG

# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(filename=str(ERRORS_LOG), level=logging.ERROR)
logger = logging.getLogger(__name__)


def log_error(step: int, exc: Exception):
    """Log exception with UTC timestamp and step name."""
    utc_now = datetime.now(timezone.utc).isoformat()
    logger.error(f"[{utc_now}] step-{step}: {exc.__class__.__name__}: {exc}")


def exponential_backoff(fn, step: int, max_retries: int = 3):
    """Retry with backoff on OOM/network errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except (MemoryError, ConnectionError, TimeoutError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"Retry step-{step} after {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                log_error(step, e)
                raise


def file_hash(path: Path) -> str:
    """SHA256 hash of file for rollback."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def cleanup_temp():
    """Delete *.tmp, *.cache, __pycache__."""
    import shutil
    for pattern in ["**/*.tmp", "**/*.cache", "**/__pycache__"]:
        for p in Path(".").glob(pattern):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations live in individual agent files (agents/00_fetch.py, etc.)
# This template provides only the shared utility functions above.

