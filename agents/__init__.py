"""agents/__init__.py — register numbered agent files as importable submodules."""
import importlib.util
import sys
from pathlib import Path

_AGENTS_DIR = Path(__file__).parent

_MODULE_MAP = {
    "fetch_agent":           "00_fetch.py",
    "multimodal_extract":    "01_multimodal_extract.py",
    "pedagogical_agent":     "02_pedagogical_agent.py",
    "kv_cache_agent":        "03_kv_cache_agent.py",
    "writer_agent":          "04_writer_agent.py",
    "citation_agent":        "05_citation_agent.py",
    "evaluation_agent":      "06_evaluation_agent.py",
    "image_upscale_agent":   "07_image_upscale_agent.py",
    "publish_agent":         "08_publish_agent.py",
}


def __getattr__(name: str):
    """Lazy-load numbered agent files as `agents.<alias>`."""
    if name not in _MODULE_MAP:
        raise AttributeError(f"module 'agents' has no attribute {name!r}")
    full_name = f"agents.{name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    file_path = _AGENTS_DIR / _MODULE_MAP[name]
    spec = importlib.util.spec_from_file_location(full_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod
