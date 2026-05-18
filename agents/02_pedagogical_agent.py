"""agents/02_pedagogical_agent.py — v3 Pedagogical Agent (MLX Qwen)

Reads the multimodal_log.json and applies Bloom's Taxonomy to produce a
structured Table of Contents with prerequisite mapping and concept sequencing.

v3: Uses Qwen3.5-0.8B MLX (83 tok/s) instead of Gemma 4 for speed.

Input:  list[dict] — multimodal_log entries
Output: workspace/toc.json
        { sections: [{id, title, bloom_level, prereqs, key_terms, timestamps}] }
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import TRANSCRIPT_DIR, QWEN_MLX_MODEL, TOKEN_BUDGET, ERRORS_LOG
from agents.template import exponential_backoff, log_error, cleanup_temp

BLOOM_LEVELS = ["remember", "understand", "apply", "analyse", "evaluate", "create"]

# ── Qwen3.5-0.8B MLX (fast model for structural tasks) ───────────────────────
_mlx_model = None
_mlx_tokenizer = None


def _get_mlx():
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer
    if not QWEN_MLX_MODEL.exists():
        print(f"[pedagogical] WARN: Qwen MLX not found at {QWEN_MLX_MODEL}")
        return None, None
    try:
        import mlx_lm
        _mlx_model, _mlx_tokenizer = mlx_lm.load(str(QWEN_MLX_MODEL))
        print(f"[pedagogical] Qwen3.5-0.8B MLX loaded (83 tok/s)")
    except Exception as e:
        log_error(2, e)
        _mlx_model, _mlx_tokenizer = None, None
    return _mlx_model, _mlx_tokenizer


def _call_qwen(prompt: str, max_tokens: int = 600) -> str:
    model, tokenizer = _get_mlx()
    if model is None:
        return ""
    try:
        import mlx_lm
        # Use /no_think to disable verbose reasoning for speed
        result = mlx_lm.generate(
            model, tokenizer,
            prompt=f"/no_think\n{prompt}",
            max_tokens=max_tokens,
            verbose=False,
        )
        # Strip any <think>...</think> blocks that may slip through
        result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
        return result
    except Exception as e:
        log_error(2, e)
        return ""


def _extract_json(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Try array
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _target_chapter_count(duration_sec: float, high_info_ratio: float) -> int:
    """Compute content-aware target chapter count.

    Rules:
      - Base: 1 chapter per 45 minutes of video.
      - Dense content (high_info_ratio > 0.40): scale up by 1.5×
      - Light content  (high_info_ratio < 0.20): scale down by 0.75×
      - Always at least 1 chapter, at most ceil(duration_min/15).
    """
    duration_min = duration_sec / 60.0
    base = max(1, round(duration_min / 45.0))
    if high_info_ratio > 0.40:
        base = max(base, round(duration_min / 30.0))
    elif high_info_ratio < 0.20:
        base = max(1, round(duration_min / 60.0))
    cap = max(1, int(duration_min / 15))
    return min(base, cap)


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "those",
    "these", "it", "its", "as", "by", "with", "from", "into", "about", "so",
    "what", "what's", "which", "who", "whose", "how", "i", "you", "we", "they",
    "he", "she", "do", "does", "did", "have", "has", "had", "will", "would",
    "can", "could", "should", "shall", "may", "might", "just", "now", "then",
    "okay", "ok", "uh", "um", "um,", "yeah", "right", "well", "going", "go",
    "see", "say", "said", "got", "get", "really", "actually", "kind", "sort",
    "thing", "things", "way", "want", "let", "like", "think", "know", "some",
    "any", "all", "more", "less", "very", "much", "many", "first", "second",
})


def _heuristic_title(snippet: str) -> str:
    """Convert a transcript snippet into a Title-Cased noun-phrase chapter title."""
    if not snippet:
        return "Section"
    # Take first ~12 content words.
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", snippet)
    content = [t for t in tokens if t.lower() not in _STOPWORDS]
    if not content:
        content = tokens[:6] or ["Section"]
    title = " ".join(content[:8]).title()
    if not title:
        title = "Section"
    return title


def _heuristic_toc(log_entries: list[dict], target_n: int | None = None) -> dict:
    """Fallback: cluster high-info entries into sections without LLM.

    Args:
        log_entries: Full multimodal log.
        target_n: Number of sections to produce (content-aware). If None, one
                  section per high-info entry (old behaviour — not recommended).
    """
    high_info = [e for e in log_entries if e.get("is_high_info")]
    if not high_info:
        high_info = log_entries[:max(1, target_n or 1)]
    if not high_info:
        high_info = [{"text": "Introduction", "timestamp_sec": 0}]

    n = target_n or len(high_info)
    n = max(1, n)

    # Cluster high_info into n equal-time buckets
    clusters: list[list[dict]] = [[] for _ in range(n)]
    total_ts = max(e.get("timestamp_sec", 0) for e in high_info) or 1.0
    for entry in high_info:
        ts = entry.get("timestamp_sec", 0)
        idx = min(int(ts / total_ts * n), n - 1)
        clusters[idx].append(entry)

    # Fill empty clusters by borrowing from neighbours
    all_entries = list(high_info)
    for i, cluster in enumerate(clusters):
        if not cluster:
            clusters[i] = [all_entries[int(len(all_entries) * i / n)]]

    sections = []
    for i, cluster in enumerate(clusters):
        # Pick the middle entry as the section representative
        rep = cluster[len(cluster) // 2]
        text_snippet = rep.get("text", "")[:160].strip()
        title = _heuristic_title(text_snippet) or f"Section {i+1}"
        timestamps = sorted(set(e.get("timestamp_sec", 0) for e in cluster))
        sections.append({
            "id": f"ch_{i+1:03d}",
            "title": title,
            "bloom_level": BLOOM_LEVELS[min(i, len(BLOOM_LEVELS) - 1)],
            "prereqs": [f"ch_{i:03d}"] if i > 0 else [],
            "key_terms": [],
            "timestamps": timestamps,
        })
    return {"sections": sections}


def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 2
    data["batch"] = 4
    data["step"] = 2
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


# ── Public API ────────────────────────────────────────────────────────────────

def step_2(multimodal_log: list[dict] | None = None,
           video_duration_sec: float | None = None) -> dict:
    """Build a ToC from the multimodal log using Bloom's Taxonomy.

    Args:
        multimodal_log: Parsed list of multimodal log entries. If None,
                        reads the newest *_multimodal_log.json from TRANSCRIPT_DIR.
        video_duration_sec: Total video length in seconds. Used to compute a
                            content-aware chapter count. If None, estimated from
                            the maximum timestamp in the log.

    Returns:
        toc dict written to workspace/toc.json.
    """
    def work():
        log = multimodal_log
        if log is None:
            files = sorted(TRANSCRIPT_DIR.glob("*_multimodal_log.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                raise FileNotFoundError(f"No multimodal_log.json in {TRANSCRIPT_DIR}")
            with open(files[0], encoding="utf-8") as f:
                log = json.load(f)

        # ── Compute content-aware target chapter count ────────────────────
        duration = video_duration_sec
        if not duration:
            timestamps = [e.get("timestamp_sec", 0) for e in log]
            duration = max(timestamps) if timestamps else 1800.0

        high_info_count = sum(1 for e in log if e.get("is_high_info"))
        hi_ratio = high_info_count / max(len(log), 1)
        target_chapters = _target_chapter_count(duration, hi_ratio)
        print(f"[pedagogical] duration={duration/60:.1f}min, hi_ratio={hi_ratio:.2f} "
              f"→ target {target_chapters} chapter(s)")

        # Summarise log for prompt (keep within token budget)
        max_chars = TOKEN_BUDGET["step2"]["max_in"] * 4
        summary_lines = []
        for e in log:
            ts = e.get("timestamp_sec", 0)
            text = e.get("text", "")[:100]
            vis = e.get("visual_desc", "")[:60]
            hi = "★" if e.get("is_high_info") else " "
            summary_lines.append(f"[{ts:6.0f}s {hi}] {text} | visual: {vis}")
        summary = "\n".join(summary_lines)[:max_chars]

        prompt = (
            "You are an expert curriculum designer.\n"
            f"Lecture transcript log (video duration: {duration/60:.0f} min):\n{summary}\n\n"
            f"Produce EXACTLY {target_chapters} sections in the Table of Contents. "
            "Each section must cover a major topic and span multiple timestamps. "
            "Each section must have: id (ch_NNN), title (descriptive, 5-15 words), "
            "bloom_level (one of: remember/understand/apply/analyse/evaluate/create), "
            "prereqs (list of section ids), key_terms (list), timestamps (list of "
            "relevant seconds).\n"
            f"Return ONLY valid JSON with exactly {target_chapters} sections: "
            "{\"sections\": [...]}"
        )

        raw = _call_qwen(prompt, max_tokens=TOKEN_BUDGET["step2"]["max_out"])
        toc = _extract_json(raw)

        if not toc or not isinstance(toc.get("sections"), list) or len(toc["sections"]) < 1:
            print("[pedagogical] LLM output insufficient; using heuristic ToC")
            toc = _heuristic_toc(log, target_n=target_chapters)
        elif abs(len(toc["sections"]) - target_chapters) > target_chapters * 0.5:
            # LLM produced wildly wrong count — use heuristic
            print(f"[pedagogical] LLM produced {len(toc['sections'])} sections "
                  f"(expected {target_chapters}); using heuristic")
            toc = _heuristic_toc(log, target_n=target_chapters)

        # Validate / normalise
        for sec in toc["sections"]:
            sec.setdefault("id", f"ch_{len(toc['sections']):03d}")
            sec.setdefault("bloom_level", "understand")
            sec.setdefault("prereqs", [])
            sec.setdefault("key_terms", [])
            sec.setdefault("timestamps", [])
            if sec["bloom_level"] not in BLOOM_LEVELS:
                sec["bloom_level"] = "understand"

        out_path = Path(__file__).parent.parent / "workspace" / "toc.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(toc, f, indent=2, ensure_ascii=False)

        print(f"[pedagogical] Written ToC with {len(toc['sections'])} sections → {out_path}")
        return toc

    try:
        toc = exponential_backoff(work, step=2)
        _update_checkpoint("agents/02_pedagogical_agent.py implemented")
        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · "
            "02_pedagogical_agent: Bloom Taxonomy → toc.json"
        )
        cleanup_temp()
        return toc
    except Exception as e:
        log_error(2, e)
        raise


if __name__ == "__main__":
    toc = step_2()
    print(json.dumps(toc, indent=2))
