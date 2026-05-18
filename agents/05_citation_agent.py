"""agents/05_citation_agent.py — Citation Agent

Parses chapter Markdown for inline paper/result mentions and inserts
properly formatted footnotes by calling DuckDuckGo and ArXiv APIs.

Input:  {chapter_id: markdown_text}
Output: Updated {chapter_id: markdown_text} with [^N]: footnotes appended,
        plus {chapter_id: [citation_dicts]}
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BOOK_DIR, TOKEN_BUDGET, ERRORS_LOG
from agents.template import log_error, exponential_backoff, cleanup_temp

# ── Citation detection ────────────────────────────────────────────────────────

_PAPER_PATTERNS = [
    r"\b([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s+et\s+al\.?,?\s+\d{4})\b",
    r"\b([A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+,?\s+\d{4})\b",
    r'"([^"]{10,80})"\s+\(\d{4}\)',
    r"\(([A-Z][a-zA-Z\s]{5,50},\s+\d{4})\)",
]

def _find_references(text: str) -> list[str]:
    """Return deduplicated list of potential citation strings."""
    found = []
    for pat in _PAPER_PATTERNS:
        found.extend(re.findall(pat, text))
    return list(dict.fromkeys(found))  # preserve order, deduplicate


# ── Lookup functions ──────────────────────────────────────────────────────────

def _duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    """Search DuckDuckGo for academic references. Returns list of result dicts."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(f"{query} site:arxiv.org OR site:doi.org", max_results=max_results):
                results.append({"title": r.get("title",""), "url": r.get("href",""), "body": r.get("body","")})
        return results
    except Exception as e:
        log_error(5, e)
        return []


def _arxiv_search(query: str) -> dict | None:
    """Search ArXiv for the best matching paper."""
    try:
        import urllib.request, urllib.parse
        q = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode()
        # Parse minimal fields from Atom XML
        title_m = re.search(r"<entry>.*?<title>(.*?)</title>", raw, re.DOTALL)
        author_m = re.search(r"<author><name>(.*?)</name>", raw, re.DOTALL)
        id_m = re.search(r"<id>(http://arxiv\.org/abs/[\d.v]+)</id>", raw)
        year_m = re.search(r"<published>(\d{4})", raw)
        if title_m:
            return {
                "title": title_m.group(1).strip(),
                "authors": author_m.group(1).strip() if author_m else "Unknown",
                "year": year_m.group(1) if year_m else "",
                "url": id_m.group(1) if id_m else "",
            }
    except Exception as e:
        log_error(5, e)
    return None


def _resolve_citation(reference_str: str) -> dict | None:
    """Try ArXiv first, fall back to DuckDuckGo."""
    time.sleep(0.5)  # rate-limit courtesy
    paper = _arxiv_search(reference_str)
    if paper and paper.get("title"):
        return paper
    results = _duckduckgo_search(reference_str)
    if results:
        r = results[0]
        year_m = re.search(r"\b(19|20)\d{2}\b", r.get("body", ""))
        return {
            "title": r["title"],
            "authors": "",
            "year": year_m.group(0) if year_m else "",
            "url": r["url"],
        }
    return None


# ── Injection ─────────────────────────────────────────────────────────────────

def _inject_footnotes(text: str, citations: list[dict], start_n: int = 1) -> str:
    """Append footnote definitions at end of markdown and insert inline markers."""
    if not citations:
        return text
    footnotes = []
    for i, c in enumerate(citations, start=start_n):
        marker = f"[^{i}]"
        ref_str = c.get("ref_str", "")
        # Replace first occurrence of the reference string with an inline marker
        if ref_str and ref_str in text:
            text = text.replace(ref_str, f"{ref_str}{marker}", 1)
        url = c.get("url", "")
        authors = c.get("authors", "")
        year = c.get("year", "")
        title = c.get("title", ref_str)
        footnote = f"{marker}: {authors + ' ' if authors else ''}{title}"
        if year:
            footnote += f" ({year})"
        if url:
            footnote += f". <{url}>"
        footnotes.append(footnote)

    return text + "\n\n---\n\n" + "\n\n".join(footnotes)


# ── Checkpoint / DONE ─────────────────────────────────────────────────────────

def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 2
    data["batch"] = 7
    data["step"] = 5
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

def step_5(
    chapters: dict[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, list]]:
    """Inject citations into all chapters.

    Args:
        chapters: {chapter_id: markdown_text}. If None, reads from BOOK_DIR.

    Returns:
        Tuple of (updated_chapters, citations_by_chapter).
    """
    def work():
        nonlocal chapters
        if chapters is None:
            chapters = {}
            for p in sorted(BOOK_DIR.glob("ch_*.md")):
                chapters[p.stem] = p.read_text(encoding="utf-8")

        updated: dict[str, str] = {}
        all_citations: dict[str, list] = {}

        for ch_id, text in chapters.items():
            refs = _find_references(text)
            if not refs:
                updated[ch_id] = text
                all_citations[ch_id] = []
                continue

            print(f"[citation] {ch_id}: found {len(refs)} references")
            resolved = []
            for ref in refs[:5]:  # cap at 5 lookups per chapter
                result = _resolve_citation(ref)
                if result:
                    result["ref_str"] = ref
                    resolved.append(result)
                    print(f"[citation]   ✓ {result.get('title','')[:60]}")

            new_text = _inject_footnotes(text, resolved)
            updated[ch_id] = new_text
            all_citations[ch_id] = resolved

            out_file = BOOK_DIR / f"{ch_id}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(new_text)

        return updated, all_citations

    try:
        chapters_out, citations_out = exponential_backoff(work, step=5)
        _update_checkpoint("agents/05_citation_agent.py implemented")
        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · "
            "05_citation_agent: DuckDuckGo + ArXiv footnote injection"
        )
        cleanup_temp()
        return chapters_out, citations_out
    except Exception as e:
        log_error(5, e)
        raise


if __name__ == "__main__":
    ch, cit = step_5()
    for ch_id, c_list in cit.items():
        print(f"{ch_id}: {len(c_list)} citations")
