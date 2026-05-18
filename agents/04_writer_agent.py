"""agents/04_writer_agent.py — v3 Writer Agent (Batched Gemma 4)

v3: One batched LLM call per video instead of N separate calls.
    Gemma 4 kept for prose quality (54 tok/s, excellent output).

For all ToC sections at once, retrieves full-lecture context from the KV-cache
and drafts all chapters in one prompt with section delimiters.

Supports a critique loop: if the state contains a critique for a chapter,
only re-drafts the critiqued chapters.

Input:  toc dict, critique dict (may be empty)
Output: workspace/book/chapter_NNN.md (one per ToC section)
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BOOK_DIR, GEMMA_GGUF, TOKEN_BUDGET, ERRORS_LOG
from agents.template import exponential_backoff, log_error, cleanup_temp


def _get_kv_retrieve():
    """Lazy import to avoid circular dependencies at module load time."""
    import agents
    kv = agents.kv_cache_agent
    return kv.retrieve_context

BOOK_DIR.mkdir(parents=True, exist_ok=True)

# ── Gemma 4 GGUF model (loaded once, kept for quality) ───────────────────────
_llm = None

def _get_llm():
    global _llm
    if _llm is not None:
        return _llm
    if not GEMMA_GGUF.exists():
        return None
    try:
        from llama_cpp import Llama
        _llm = Llama(
            model_path=str(GEMMA_GGUF),
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )
        print(f"[writer] Gemma 4 loaded (Metal, 8192 ctx)")
    except Exception as e:
        log_error(4, e)
        _llm = None
    return _llm


def _call_gemma(prompt: str, max_tokens: int = 2000) -> str:
    llm = _get_llm()
    if llm is None:
        return ""
    try:
        resp = llm(prompt, max_tokens=max_tokens, temperature=0.3, stop=["---END---"])
        return resp["choices"][0]["text"].strip()
    except Exception as e:
        log_error(4, e)
        return ""


def _seconds_to_hhmmss(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Batched chapter drafting ──────────────────────────────────────────────────

_BEGIN_TAG = "<<<CHAPTER_BEGIN"   # rare 5-token sequence Gemma rarely emits unprompted
_END_TAG   = "<<<CHAPTER_END>>>"


def _build_batched_prompt(
    sections: list[dict],
    multimodal_log: list[dict] | None,
    critique: dict[str, str] | None,
) -> str:
    """Build a single prompt that generates all chapters at once."""
    retrieve_context = _get_kv_retrieve()
    context = retrieve_context(" ".join(s.get("title", "") for s in sections), top_k=10)

    section_specs = []
    for i, section in enumerate(sections):
        title = section.get("title", f"Section {i+1}")
        bloom = section.get("bloom_level", "understand")
        key_terms = ", ".join(section.get("key_terms", []))
        ch_id = section.get("id", f"ch_{i+1:03d}")

        # Figure anchors — short, single-token form to avoid LLM mangling.
        timestamps = section.get("timestamps", [])
        figure_hints = ""
        if multimodal_log and timestamps:
            for ts in timestamps[:3]:
                closest = min(multimodal_log, key=lambda e: abs(e.get("timestamp_sec", 0) - ts))
                desc = (closest.get("visual_desc") or "").strip()
                # Skip debug-only descriptions (keyword_hits=… visual_score=…)
                if not desc or desc.startswith("keyword_hits=") or "visual_score=" in desc:
                    desc = (closest.get("text") or "")[:80]
                if desc:
                    hhmmss = _seconds_to_hhmmss(closest["timestamp_sec"])
                    figure_hints += f"  Figure cue at {hhmmss}: {desc[:120]}\n"

        critique_note = ""
        if (critique or {}).get(ch_id):
            critique_note = f"  CRITIQUE to address: {critique[ch_id]}\n"

        section_specs.append(
            f"{_BEGIN_TAG} {ch_id}>>>\n"
            f"  Title: {title}\n"
            f"  Bloom level: {bloom}\n"
            f"  Key terms: {key_terms}\n"
            f"{critique_note}{figure_hints}"
            f"{_END_TAG}\n"
        )

    all_specs = "\n".join(section_specs)

    prompt = (
        "You are an expert textbook author. Write Markdown lecture notes for the\n"
        "sections listed below. For EACH input section, emit exactly one chapter\n"
        "block in this format and NOTHING ELSE between blocks:\n\n"
        f"{_BEGIN_TAG} <chapter_id>>>>\n"
        "## <Chapter title>\n"
        "<chapter prose with sub-headings, bullets, and inline figure anchors>\n"
        f"{_END_TAG}\n\n"
        "Rules:\n"
        f"- Use the exact tags {_BEGIN_TAG} <id>>>> and {_END_TAG}.\n"
        "- Do NOT echo the input spec, key-terms list, or your own commentary.\n"
        "- Do NOT include the words 'Self-Correction' or 'Final Output'.\n"
        "- Inline figure anchors use this short syntax: [FIG t=HH:MM:SS]\n"
        "- Write precise, hallucination-free prose grounded in the lecture context.\n\n"
        f"Lecture context:\n'''{context[:3000]}'''\n\n"
        f"Sections to write:\n{all_specs}\n\n"
        "Now produce all chapter blocks:\n"
    )
    return prompt


_META_LINE = re.compile(
    r"^\s*(?:"
    r"\*{0,2}\(?(?:Self-?Correction|Final Output|Review|Note to self)[^)\n]*\)?\*{0,2}.*"
    r"|=== SECTION.*"
    r"|\*\*\*\s*$"
    r"|---END---\s*$"
    r")",
    re.IGNORECASE,
)


def _sanitise_chapter(text: str, ch_id: str, title: str | None = None) -> str:
    """Strip LLM self-talk, leftover delimiters, and meta-commentary."""
    if not text:
        return text
    # Remove any trailing END tag that survived the split.
    text = text.replace(_END_TAG, "")
    # Remove stray BEGIN tags inside body.
    text = re.sub(rf"{re.escape(_BEGIN_TAG)}\s*\S*?>+", "", text)
    # Remove meta lines.
    cleaned_lines = []
    for line in text.splitlines():
        if _META_LINE.match(line):
            continue
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines).strip()
    # Collapse 3+ blank lines into 2.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Ensure chapter starts with an H2 heading.
    if not re.match(r"^\s*##\s+\S", cleaned):
        heading = (title or ch_id).strip()
        cleaned = f"## {heading}\n\n{cleaned}"
    return cleaned


def _split_batched_output(raw: str, section_ids: list[str],
                          section_titles: dict[str, str] | None = None) -> dict[str, str]:
    """Split batched LLM output into per-chapter text."""
    chapters: dict[str, str] = {}
    section_titles = section_titles or {}

    # Primary delimiter: <<<CHAPTER_BEGIN <id>>>>
    pattern = re.compile(
        rf"{re.escape(_BEGIN_TAG)}\s*([A-Za-z0-9_]+)>+",
    )
    matches = list(pattern.finditer(raw))
    if matches:
        for i, m in enumerate(matches):
            ch_id = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
            body = raw[start:end]
            # Trim at END tag if present.
            if _END_TAG in body:
                body = body.split(_END_TAG, 1)[0]
            if ch_id in section_ids:
                chapters[ch_id] = _sanitise_chapter(body, ch_id, section_titles.get(ch_id))
    else:
        # Fallback: split by H2 headings, one per section in order.
        h2_parts = [p.strip() for p in re.split(r"(?=^##\s)", raw, flags=re.MULTILINE) if p.strip()]
        for i, part in enumerate(h2_parts):
            if i >= len(section_ids):
                break
            sid = section_ids[i]
            chapters[sid] = _sanitise_chapter(part, sid, section_titles.get(sid))

    # Fill any missing sections with placeholders.
    for sid in section_ids:
        if sid not in chapters or len(chapters[sid].strip()) < 20:
            title = section_titles.get(sid, sid)
            chapters[sid] = (
                f"## {title}\n\n"
                "*Content pending — LLM output did not cover this section.*\n"
            )

    return chapters


# ── Checkpoint / DONE helpers ─────────────────────────────────────────────────

def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 2
    data["batch"] = 6
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


# ── Public API ────────────────────────────────────────────────────────────────

def step_4(
    toc: dict | None = None,
    critique: dict[str, str] | None = None,
    multimodal_log: list[dict] | None = None,
) -> dict[str, str]:
    """Draft all chapters in one batched LLM call.

    Args:
        toc: ToC dict with 'sections' list. If None, reads workspace/toc.json.
        critique: {chapter_id: critique_text} from the evaluation loop.
        multimodal_log: Optional log for figure anchor extraction.

    Returns:
        {chapter_id: markdown_text} — also written to BOOK_DIR.
    """
    max_out = TOKEN_BUDGET["step3"]["max_out"]

    def work():
        nonlocal toc, multimodal_log
        if toc is None:
            toc_path = Path(__file__).parent.parent / "workspace" / "toc.json"
            with open(toc_path, encoding="utf-8") as f:
                toc = json.load(f)

        if multimodal_log is None:
            from config.settings import TRANSCRIPT_DIR
            files = sorted(TRANSCRIPT_DIR.glob("*_multimodal_log.json"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                with open(files[0], encoding="utf-8") as f:
                    multimodal_log = json.load(f)

        sections = toc.get("sections", [])
        section_ids = [s.get("id", f"ch_{i+1:03d}") for i, s in enumerate(sections)]

        # Determine which sections need writing
        if critique:
            # Only re-draft critiqued sections
            to_write = [s for s in sections if s.get("id") in critique]
            if not to_write:
                to_write = sections
        else:
            to_write = sections

        print(f"[writer] Batched write: {len(to_write)} section(s) in ONE call")
        prompt = _build_batched_prompt(to_write, multimodal_log, critique)
        raw = _call_gemma(prompt, max_tokens=max_out)

        write_ids = [s.get("id", f"ch_{i+1:03d}") for i, s in enumerate(to_write)]
        title_map = {s.get("id", f"ch_{i+1:03d}"): s.get("title", "")
                     for i, s in enumerate(to_write)}
        chapters = _split_batched_output(raw, write_ids, section_titles=title_map)

        # Write to disk
        for ch_id, text in chapters.items():
            out_file = BOOK_DIR / f"{ch_id}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(text)
            fig_count = text.count("[Figure:")
            print(f"[writer] ✓ {ch_id} ({len(text)} chars, {fig_count} figs)")

        return chapters

    try:
        chapters = exponential_backoff(work, step=4)
        _update_checkpoint("agents/04_writer_agent.py v3 batched")
        _append_done(
            f"[v6] {datetime.now(timezone.utc).date()} · Copilot · "
            "04_writer_agent: v3 batched Gemma 4 (one call per video)"
        )
        cleanup_temp()
        return chapters
    except Exception as e:
        log_error(4, e)
        raise


if __name__ == "__main__":
    chapters = step_4()
    for ch_id, text in chapters.items():
        print(f"{ch_id}: {len(text)} chars")
