"""engine_v4.understand — L2 UNDERSTAND layer.

Two pure functions (independent; runner gathers them concurrently):

    plan_outline(source)  -> Outline
    build_context(source) -> Context

LLM calls go through ``engine_v4.llm.call_mlx`` (which holds ``LLM_LOCK``);
we additionally take the global ``limits.LLM`` semaphore as a guard against
future fan-out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from config.settings import TOKEN_BUDGET
from . import limits
from .index import Context
from .llm import call_mlx
from .types import Outline, OutlineSection, Source, TranscriptWindow


BLOOM_LEVELS = ["remember", "understand", "apply", "analyse", "evaluate", "create"]

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


# ── Heuristic helpers (ported from agents/02_pedagogical_agent.py) ──────────

def _target_chapter_count(duration_sec: float, high_info_ratio: float) -> int:
    """Content-aware chapter count.

    For short videos (<30 min) we want at least 3 chapters so the textbook
    has structure; for long lectures we scale roughly one chapter per 20
    minutes of dense content (or per 60 min of sparse content).
    """
    duration_min = duration_sec / 60.0
    if duration_min <= 0:
        return 1
    if duration_min < 30:
        # Short video: aim for one chapter every ~5 minutes, min 3.
        base = max(3, round(duration_min / 5.0))
    elif high_info_ratio > 0.40:
        base = max(3, round(duration_min / 20.0))
    elif high_info_ratio < 0.20:
        base = max(3, round(duration_min / 45.0))
    else:
        base = max(3, round(duration_min / 30.0))
    cap = max(3, int(duration_min / 3))
    return min(base, cap)


def _heuristic_title(snippet: str) -> str:
    """Build a short, clean noun-phrase-ish title from a transcript snippet.

    Strategy: keep the first ~6 *content* words, skip leading filler, and
    title-case. If we still end up with a noisy fragment we fall back to a
    generic label so the chapter heading reads cleanly in the textbook.
    """
    if not snippet:
        return "Section"
    # Take only the first sentence-ish fragment to avoid mid-sentence noise.
    chunk = re.split(r"[.!?;\n]", snippet, maxsplit=1)[0]
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]+", chunk)
    content = [t for t in tokens if t.lower() not in _STOPWORDS]
    if len(content) < 3:
        return "Section"
    # Drop leading verb-y / pronoun-y fillers that often start spoken lines.
    while content and content[0].lower() in {
        "going", "now", "today", "lets", "let", "okay", "alright",
        "remember", "consider", "imagine", "suppose", "think",
    }:
        content.pop(0)
    if len(content) < 2:
        return "Section"
    title = " ".join(content[:6]).title().replace("'S", "'s").replace("'Ll", "'ll").replace("'T", "'t")
    return title or "Section"


def _extract_json(text: str) -> Any:
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer) + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return None


def _heuristic_outline(windows: list[TranscriptWindow], target_n: int) -> list[dict]:
    high = [w for w in windows if w.is_high_info] or list(windows)
    if not high:
        high = [TranscriptWindow(0.0, 0.0, "Introduction")]

    n = max(1, target_n)
    clusters: list[list[TranscriptWindow]] = [[] for _ in range(n)]
    total = max((w.start_sec for w in high), default=1.0) or 1.0
    for w in high:
        idx = min(int(w.start_sec / total * n), n - 1)
        clusters[idx].append(w)
    for i, c in enumerate(clusters):
        if not c:
            clusters[i] = [high[int(len(high) * i / n)]]

    sections: list[dict] = []
    for i, cluster in enumerate(clusters):
        rep = cluster[len(cluster) // 2]
        snippet = (rep.text or "")[:160].strip()
        title = _heuristic_title(snippet) or f"Section {i + 1}"
        timestamps = sorted({float(w.start_sec) for w in cluster})
        sections.append({
            "id": f"ch_{i + 1:03d}",
            "title": title,
            "bloom_level": BLOOM_LEVELS[min(i, len(BLOOM_LEVELS) - 1)],
            "key_terms": [],
            "timestamps": timestamps,
            "prereqs": [f"ch_{i:03d}"] if i > 0 else [],
        })
    return sections


# ── Public API ──────────────────────────────────────────────────────────────

def plan_outline(source: Source, max_sections: Optional[int] = None) -> Outline:
    """Build an Outline (Bloom-tagged ToC) for the given Source."""
    windows = source.windows or []
    duration = source.duration_sec or (
        max((w.end_sec for w in windows), default=0.0) or 1800.0
    )
    hi_count = sum(1 for w in windows if w.is_high_info)
    hi_ratio = hi_count / max(len(windows), 1)
    target = _target_chapter_count(duration, hi_ratio)
    if max_sections is not None:
        target = max(1, min(target, max_sections))

    # Build prompt summary within the step-2 input budget.
    max_chars = TOKEN_BUDGET["step2"]["max_in"] * 4
    summary_lines: list[str] = []
    for w in windows:
        hi = "★" if w.is_high_info else " "
        summary_lines.append(
            f"[{w.start_sec:6.0f}s {hi}] {(w.text or '')[:100]}"
        )
    summary = "\n".join(summary_lines)[:max_chars]

    prompt = (
        "You are an expert curriculum designer.\n"
        f"Lecture transcript log (video duration: {duration / 60:.0f} min):\n"
        f"{summary}\n\n"
        f"Produce EXACTLY {target} sections in the Table of Contents. "
        "Each section must cover a major topic and span multiple timestamps. "
        "Each section must have: id (ch_NNN), title (descriptive 5-15 words), "
        "bloom_level (one of: remember/understand/apply/analyse/evaluate/create), "
        "prereqs (list of section ids), key_terms (list), timestamps (list of "
        "relevant seconds).\n"
        f"Return ONLY valid JSON with exactly {target} sections: "
        "{\"sections\": [...]}"
    )

    sections_raw: list[dict] = []
    with limits.LLM:
        raw = call_mlx(prompt, max_tokens=TOKEN_BUDGET["step2"]["max_out"])
    parsed = _extract_json(raw) if raw else None
    if isinstance(parsed, dict) and isinstance(parsed.get("sections"), list):
        sections_raw = parsed["sections"]

    # Reject wildly off counts; fall back to heuristic.
    if not sections_raw or abs(len(sections_raw) - target) > target * 0.5:
        sections_raw = _heuristic_outline(windows, target)

    # Map nearest-window timestamps if model didn't give any.
    sections: list[OutlineSection] = []
    for i, sec in enumerate(sections_raw):
        sid = str(sec.get("id") or f"ch_{i + 1:03d}")
        title = str(sec.get("title") or "").strip() or _heuristic_title(
            (windows[0].text if windows else "Section")
        )
        bloom = str(sec.get("bloom_level") or "understand").lower()
        if bloom not in BLOOM_LEVELS:
            bloom = "understand"
        key_terms = [str(t) for t in (sec.get("key_terms") or []) if str(t).strip()]
        timestamps = [float(t) for t in (sec.get("timestamps") or [])
                      if isinstance(t, (int, float))]
        if not timestamps and windows:
            # Assign the nearest temporal cluster of windows for this section.
            n = max(1, len(sections_raw))
            lo = int(len(windows) * i / n)
            hi = max(lo + 1, int(len(windows) * (i + 1) / n))
            timestamps = sorted({float(w.start_sec) for w in windows[lo:hi]})
        prereqs = [str(p) for p in (sec.get("prereqs") or [])]
        sections.append(OutlineSection(
            id=sid, title=title, bloom_level=bloom,
            key_terms=key_terms, timestamps=timestamps, prereqs=prereqs,
        ))

    outline = Outline(sections=sections, target_count=target)

    # Persist outline.json into the source workspace.
    try:
        ws = Path(source.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        payload = {
            "target_count": target,
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "bloom_level": s.bloom_level,
                    "key_terms": s.key_terms,
                    "timestamps": s.timestamps,
                    "prereqs": s.prereqs,
                }
                for s in sections
            ],
        }
        (ws / "outline.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception:
        pass

    return outline


def build_context(source: Source) -> Context:
    """Build (and persist) a Context embedding store for the Source."""
    ctx = Context()
    ctx.store(source.windows or [])
    try:
        ws = Path(source.workspace)
        ws.mkdir(parents=True, exist_ok=True)
        ctx.save(ws / "index.npz")
    except Exception:
        pass
    return ctx
