"""agents/04_assemble.py

Hardened assemble agent (Batch 5).
- Concatenates `BOOK_DIR/chapter_*.md` into `workspace/textbook.md`.
- Ensures dirs exist, writes checkpoint, appends DONE.md, cleans temp files.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

from config.settings import BOOK_DIR, ERRORS_LOG
from agents.template import exponential_backoff, log_error, cleanup_temp


def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["step"] = 4
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


def step_4(book_dir: Path = BOOK_DIR) -> Path:
    """Assemble chapter_*.md files into `workspace/textbook.md` and return its path."""
    BOOK_DIR.mkdir(parents=True, exist_ok=True)
    out_file = Path(book_dir) / "textbook.md"

    def work():
        chapters = sorted(Path(book_dir).glob("chapter_*.md"))
        with open(out_file, "w", encoding="utf-8") as w:
            for ch in chapters:
                with open(ch, "r", encoding="utf-8") as r:
                    w.write(r.read())
                    w.write("\n\n")
        # attempt GPU cleanup if any model used earlier
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return out_file

    try:
        out = exponential_backoff(lambda: work(), step=4)
        _update_checkpoint("agents/04_assemble.py implemented")
        _append_done(f"[v8] {datetime.now(timezone.utc).date()} · Copilot · agents/04_assemble.py implemented")
        cleanup_temp()
        return out
    except Exception as e:
        log_error(4, e)
        raise
