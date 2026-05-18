"""engine_v4.limits — global resource semaphores."""
from __future__ import annotations

import threading

DOWNLOAD = threading.Semaphore(2)
MEDIA    = threading.Semaphore(2)
LLM      = threading.Semaphore(1)
CITATION = threading.Semaphore(3)
