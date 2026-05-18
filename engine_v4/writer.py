"""engine_v4.writer — per-section Gemma calls with per-section RAG.

v4.1 rewrite (2026-05): the v4.0 writer issued a single batched call for every
section, so chapters had to share both context and token budget. The first
chapter of each batch tended to absorb LLM "spec echo" and the later
chapters were truncated. The new writer:

* issues one Gemma call per section (LLM lock serialises them anyway, so
  total wall-time is unchanged but each chapter gets the full 1500-2000
  token budget);
* receives a per-section, freshly-retrieved RAG context built from
  ``title + key_terms + bloom_level + closest transcript snippet``;
* uses a much cleaner prompt with explicit ``<<<BODY>>> / <<<END>>>``
  markers (input spec lives outside the body markers so the LLM can't echo
  it);
* strips any echoed spec / meta-commentary from the head of the response.

Public API (``draft_all``) is unchanged so ``compose.compose`` keeps
working. ``compose`` now also has the option of providing the retrieval
callable so the writer can do per-section retrieval; the legacy single-
context callable is still accepted for backwards compatibility.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from .limits import LLM
from .llm import call_gemma
from .types import Frame, Outline, OutlineSection, Source

log = logging.getLogger("engine_v4.writer")

_BODY_OPEN = "<<<BODY>>>"
_BODY_CLOSE = "<<<END>>>"

# Legacy markers — still recognised so external callers / cached outputs
# from v4.0 keep flowing through the same downstream code.
_LEGACY_BEGIN_TAG = "<<<CHAPTER_BEGIN"
_LEGACY_END_TAG = "<<<CHAPTER_END>>>"


def _seconds_to_hhmmss(secs: float) -> str:
    secs = int(secs)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ── Frame / figure cues ──────────────────────────────────────────────────────

def _frames_for_section(section: OutlineSection, source: Source,
                        max_frames: int = 4) -> list[Frame]:
    """Return up to ``max_frames`` frames whose timestamps fall inside the
    section's time band. Falls back to the nearest-frame heuristic if no
    frames lie inside the band.
    """
    frames = source.frames or []
    if not frames or not section.timestamps:
        return []
    lo, hi = min(section.timestamps), max(section.timestamps)
    pad = max(30.0, (hi - lo) * 0.10)
    band = [f for f in frames if (lo - pad) <= f.timestamp_sec <= (hi + pad)]
    if not band:
        # Nearest-by-midpoint fallback so we never come back empty when frames exist.
        mid = (lo + hi) / 2.0
        band = sorted(frames, key=lambda f: abs(f.timestamp_sec - mid))[:max_frames]
    # Even spacing across the band — keeps figures visually distinct.
    band = sorted(band, key=lambda f: f.timestamp_sec)
    if len(band) <= max_frames:
        return band
    step = len(band) / max_frames
    return [band[int(i * step)] for i in range(max_frames)]


def _figure_hints(section: OutlineSection, source: Source) -> str:
    """Cue the writer towards the section's visual timestamps. The actual
    figure injection still happens downstream in ``compose._build_chapter``
    — these hints just give the writer somewhere natural to refer to them.
    """
    cues = _frames_for_section(section, source, max_frames=4)
    if not cues:
        return ""
    lines = []
    for f in cues:
        caption = (f.caption or "").strip()
        # Strip debug crumbs that occasionally appear in v3 captions.
        if not caption or caption.startswith("keyword_hits=") or "visual_score=" in caption:
            caption = ""
        hhmmss = _seconds_to_hhmmss(f.timestamp_sec)
        lines.append(f"  - figure at {hhmmss}: {caption[:120]}")
    return "Visual cues you may reference with [FIG t=HH:MM:SS]:\n" + "\n".join(lines) + "\n"


# ── Prompt assembly ──────────────────────────────────────────────────────────

def _build_section_prompt(
    section: OutlineSection,
    source: Source,
    context: str,
    critique: Optional[str] = None,
) -> str:
    """Build the per-section prompt.

    The input spec (title, bloom, key terms, critique) is wrapped in
    ``=== INPUT ===`` markers and never repeated inside the answer
    structure, which makes the spec-echo problem trivial to detect and
    strip.
    """
    key_terms = ", ".join(section.key_terms) or "(none)"
    critique_block = ""
    if critique:
        critique_block = (
            "\n=== PREVIOUS DRAFT FAILED REVIEW ===\n"
            f"{critique.strip()}\n"
            "Address every point above in this rewrite.\n"
        )

    figure_hints = _figure_hints(section, source)

    return (
        "You are an expert STEM textbook author. Produce ONE polished\n"
        "Markdown chapter for the requested section. Write in clear,\n"
        "explanatory prose grounded ONLY in the lecture context provided —\n"
        "do not invent facts, names, citations, or numerical results.\n"
        "\n"
        "OUTPUT FORMAT (STRICT):\n"
        f"  {_BODY_OPEN}\n"
        "  ## <chapter title — same as the input title>\n"
        "  <2–4 paragraph introduction explaining motivation>\n"
        "  ### <sub-heading>\n"
        "  <prose, bullets, equations as appropriate>\n"
        "  ### <sub-heading>\n"
        "  <prose>\n"
        "  ### Summary\n"
        "  <3–6 bullet recap>\n"
        f"  {_BODY_CLOSE}\n"
        "\n"
        "Rules:\n"
        f"- Begin output with exactly {_BODY_OPEN} on its own line.\n"
        f"- End output with exactly {_BODY_CLOSE} on its own line.\n"
        "- Do NOT echo the INPUT block, the key-term list, the bloom level,\n"
        "  the words 'CRITIQUE'/'Self-Correction'/'Final Output', or any\n"
        "  notes-to-self.\n"
        "- Reference visuals inline with the short anchor [FIG t=HH:MM:SS]\n"
        "  using only the timestamps cued below. Use 2–4 such anchors,\n"
        "  spread through the chapter — they will be replaced with images.\n"
        "- Use LaTeX (`$...$` / `$$...$$`) for any math.\n"
        "- Aim for 500–900 words of body text.\n"
        "\n"
        "=== LECTURE CONTEXT (verbatim transcript excerpts; the only ground truth) ===\n"
        f"{(context or '').strip()[:3500]}\n"
        "=== END CONTEXT ===\n"
        "\n"
        "=== INPUT (describes what to write — do not echo) ===\n"
        f"Title: {section.title}\n"
        f"Bloom level: {section.bloom_level}\n"
        f"Key terms: {key_terms}\n"
        f"{figure_hints}"
        f"{critique_block}"
        "=== END INPUT ===\n"
        "\n"
        f"Now produce the chapter, starting with {_BODY_OPEN}:\n"
    )


# ── Output sanitiser ─────────────────────────────────────────────────────────

# Lines that strongly look like spec/meta echo from the model.
_ECHO_LINE = re.compile(
    r"^\s*(?:"
    r"=+\s*INPUT.*|=+\s*END.*|=+\s*LECTURE.*|=+\s*CONTEXT.*|"
    r"[\*\(]{0,3}\s*(?:Title|Bloom level|Key terms|Visual cues|CRITIQUE|"
    r"Self-?Correction|Final Output|Review|Note to self|Output|Body|Context)"
    r"\s*[:\-].*"
    r"|---END---\s*$"
    r"|\*\*\*\s*$"
    r"|<<<BODY>>>\s*$|<<<END>>>\s*$"
    r")",
    re.IGNORECASE,
)

_FENCE_NEAR_TOP = re.compile(r"^\s*```")


def _strip_to_first_heading(text: str) -> str:
    """If the model rambles before the first ``## …`` heading, drop the
    preamble. This is the most reliable cure for "first part garbage"."""
    m = re.search(r"^\s*##\s+\S", text, flags=re.MULTILINE)
    if not m:
        return text
    # Keep at most one short paragraph above the heading if it's already
    # legitimate prose (no echoey patterns).
    pre = text[: m.start()].strip()
    if not pre or len(pre.splitlines()) > 3 or _ECHO_LINE.search(pre) or "Title:" in pre:
        return text[m.start():]
    return text


def _sanitise(text: str, title: str) -> str:
    if not text:
        return text

    # Trim everything outside body markers if present.
    if _BODY_OPEN in text:
        text = text.split(_BODY_OPEN, 1)[1]
    if _BODY_CLOSE in text:
        text = text.split(_BODY_CLOSE, 1)[0]

    # Legacy delimiters (v4.0 cached outputs).
    text = text.replace(_LEGACY_END_TAG, "")
    text = re.sub(rf"{re.escape(_LEGACY_BEGIN_TAG)}\s*\S*?>+", "", text)

    # Drop fenced code blocks the model wraps the whole answer in.
    lines = text.splitlines()
    if lines and _FENCE_NEAR_TOP.match(lines[0]):
        lines = lines[1:]
        if lines and _FENCE_NEAR_TOP.match(lines[-1]):
            lines = lines[:-1]
    text = "\n".join(lines)

    # Drop spec-echo / meta lines wherever they appear.
    cleaned = [ln for ln in text.splitlines() if not _ECHO_LINE.match(ln)]
    text = "\n".join(cleaned).strip()

    # Drop any "garbage preamble" before the first heading.
    text = _strip_to_first_heading(text)

    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Force-prepend the H2 if the model forgot.
    if not re.match(r"^\s*##\s+\S", text):
        text = f"## {title.strip() or 'Chapter'}\n\n{text}"
    return text


# ── Public API ───────────────────────────────────────────────────────────────

ContextRetrieve = Callable[[str, int], str]


def _section_query(section: OutlineSection) -> str:
    """Build the per-section retrieval query."""
    parts = [section.title.strip()]
    if section.key_terms:
        parts.append(" ".join(section.key_terms[:10]))
    parts.append(section.bloom_level)
    return " ".join(p for p in parts if p)


def _draft_one(
    section: OutlineSection,
    source: Source,
    context_retrieve: ContextRetrieve,
    critique: Optional[str],
    max_tokens: int,
) -> str:
    query = _section_query(section)
    context = ""
    try:
        context = context_retrieve(query, 10) or ""
    except Exception as e:
        log.warning("[writer] retrieve failed for %s: %s", section.id, e)

    # Anchor with closest transcript snippets when retrieval came back thin.
    if (not context or len(context) < 200) and source.windows and section.timestamps:
        lo, hi = min(section.timestamps), max(section.timestamps)
        nearby = [w.text for w in source.windows
                  if (lo - 60) <= w.start_sec <= (hi + 60) and w.text]
        if nearby:
            context = (context + " " + " ".join(nearby)).strip()

    prompt = _build_section_prompt(section, source, context, critique=critique)

    with LLM:
        # NB: we intentionally do NOT include _BODY_CLOSE in the stop list —
        # the marker appears inside our prompt instructions and Gemma will
        # sometimes echo it early, terminating mid-thought. We strip the
        # marker (and anything after it) in _sanitise.
        raw = call_gemma(prompt, max_tokens=max_tokens,
                         stop=[_LEGACY_END_TAG, "---END---"])
    cleaned = _sanitise(raw, section.title)
    # Fallback: if cleaning ate everything, keep the longest raw paragraph.
    if (not cleaned or len(cleaned) < 80) and raw:
        # Heuristic salvage — drop obvious echo lines but keep prose.
        salvage_lines = [ln for ln in raw.splitlines()
                         if ln.strip() and not _ECHO_LINE.match(ln)
                         and _BODY_OPEN not in ln and _BODY_CLOSE not in ln]
        if salvage_lines:
            cleaned = "## " + section.title + "\n\n" + "\n".join(salvage_lines)
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def draft_all(
    outline: Outline,
    context_retrieve: ContextRetrieve,
    source: Source,
    critique: Optional[dict[str, str]] = None,
    only_section_ids: Optional[list[str]] = None,
    max_tokens: int = 2000,
) -> dict[str, str]:
    """Draft every section as its own Gemma call.

    Args:
        outline: Bloom-tagged ToC for the source.
        context_retrieve: ``(query, top_k) -> str`` (typically
            ``Context.retrieve``).
        source: the Source the chapters belong to.
        critique: optional ``{chapter_id: critique_text}`` from the
            evaluator's bounded retry.
        only_section_ids: when provided, only draft these sections.
        max_tokens: per-section LLM token budget.
    """
    sections = list(outline.sections)
    if only_section_ids is not None:
        wanted = set(only_section_ids)
        sections = [s for s in sections if s.id in wanted]
    if not sections:
        return {}

    out: dict[str, str] = {}
    for section in sections:
        crit = (critique or {}).get(section.id)
        try:
            body = _draft_one(section, source, context_retrieve, crit, max_tokens)
        except Exception as e:
            log.warning("[writer] draft crashed for %s: %s", section.id, e)
            body = ""
        if not body or len(body.strip()) < 80:
            log.warning("[writer] empty draft for %s — retrying with simpler prompt",
                        section.id)
            try:
                body = _draft_one_simple(section, source, context_retrieve, max_tokens)
            except Exception as e:
                log.warning("[writer] simple retry crashed for %s: %s", section.id, e)
                body = ""
        if not body or len(body.strip()) < 40:
            body = (
                f"## {section.title}\n\n"
                "*Content pending — the writer produced no usable draft for "
                "this section.*\n"
            )
        out[section.id] = body
    return out


def _draft_one_simple(
    section: OutlineSection,
    source: Source,
    context_retrieve: ContextRetrieve,
    max_tokens: int,
) -> str:
    """Fallback path: minimal prompt, plain stop tokens, no markers."""
    query = _section_query(section)
    context = ""
    try:
        context = context_retrieve(query, 8) or ""
    except Exception:
        pass
    if (not context or len(context) < 200) and source.windows and section.timestamps:
        lo, hi = min(section.timestamps), max(section.timestamps)
        nearby = [w.text for w in source.windows
                  if (lo - 60) <= w.start_sec <= (hi + 60) and w.text]
        if nearby:
            context = (context + " " + " ".join(nearby)).strip()
    context = (context or "").strip()[:4000]

    key_terms = ", ".join(section.key_terms[:6]) if section.key_terms else ""
    prompt = (
        f"Write a textbook section titled \"{section.title}\".\n"
        f"Key terms to cover: {key_terms}\n"
        f"Bloom level: {section.bloom_level}\n\n"
        f"Use the transcript context below as the source of truth. "
        f"Write 4-8 clear paragraphs in your own words; do not echo this "
        f"prompt or any 'Title:' / 'Bloom:' lines.\n\n"
        f"Begin with a Markdown H2 heading: ## {section.title}\n\n"
        f"--- Transcript context ---\n{context}\n--- End context ---\n\n"
        f"Now write the section:\n## {section.title}\n\n"
    )
    with LLM:
        raw = call_gemma(prompt, max_tokens=max_tokens, stop=["---END---"])
    cleaned = _sanitise(raw, section.title)
    if cleaned and len(cleaned) >= 80:
        return cleaned
    # Last resort: prepend H2 to raw if any prose at all.
    raw = (raw or "").strip()
    if raw:
        if not raw.lstrip().startswith("##"):
            raw = f"## {section.title}\n\n{raw}"
        return raw
    return ""
