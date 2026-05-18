"""engine_v4.citations — DDG/ArXiv lookup, file-cached, best-effort.

Ports ``agents/05_citation_agent.py`` into the v4 type system. All network
work is best-effort: any exception is swallowed and the chapter is returned
unchanged. Per-chapter results are cached at
``<source.workspace>/citations/<ch_id>.json`` so reruns skip the network.
"""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from .limits import CITATION
from .types import Chapter, Citation, Source

log = logging.getLogger("engine_v4.citations")

_PAPER_PATTERNS = [
    r"\b([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?\s+et\s+al\.?,?\s+\d{4})\b",
    r"\b([A-Z][a-z]+\s+(?:and|&)\s+[A-Z][a-z]+,?\s+\d{4})\b",
    r'"([^"]{10,80})"\s+\(\d{4}\)',
    r"\(([A-Z][a-zA-Z\s]{5,50},\s+\d{4})\)",
]


def _find_references(text: str) -> list[str]:
    found: list[str] = []
    for pat in _PAPER_PATTERNS:
        found.extend(re.findall(pat, text))
    return list(dict.fromkeys(found))


def _arxiv_search(query: str) -> Optional[dict]:
    try:
        import urllib.parse
        import urllib.request
        q = urllib.parse.quote(query)
        url = f"http://export.arxiv.org/api/query?search_query=all:{q}&max_results=1"
        with urllib.request.urlopen(url, timeout=10) as resp:
            raw = resp.read().decode()
        title_m = re.search(r"<entry>.*?<title>(.*?)</title>", raw, re.DOTALL)
        author_m = re.search(r"<author><name>(.*?)</name>", raw, re.DOTALL)
        id_m = re.search(r"<id>(http://arxiv\.org/abs/[\d.v]+)</id>", raw)
        year_m = re.search(r"<published>(\d{4})", raw)
        if title_m:
            return {
                "title": title_m.group(1).strip(),
                "authors": author_m.group(1).strip() if author_m else "",
                "year": year_m.group(1) if year_m else "",
                "url": id_m.group(1) if id_m else "",
            }
    except Exception as e:
        log.warning("[citations] arxiv failed: %s", e)
    return None


def _duckduckgo_search(query: str, max_results: int = 3) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        results: list[dict] = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} site:arxiv.org OR site:doi.org",
                max_results=max_results,
            ):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "body": r.get("body", ""),
                })
        return results
    except Exception as e:
        log.warning("[citations] ddg failed: %s", e)
        return []


def _resolve_citation(reference_str: str) -> Optional[dict]:
    """Resolve one reference string. Holds the CITATION semaphore + 0.5s sleep."""
    with CITATION:
        time.sleep(0.5)
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


def _inject_footnotes(text: str, citations: list[dict], start_n: int = 1) -> str:
    """Append footnote definitions and insert inline markers."""
    if not citations:
        return text
    footnotes: list[str] = []
    for i, c in enumerate(citations, start=start_n):
        marker = f"[^{i}]"
        ref_str = c.get("ref_str", "")
        if ref_str and ref_str in text:
            text = text.replace(ref_str, f"{ref_str}{marker}", 1)
        url = c.get("url", "")
        authors = c.get("authors", "")
        year = c.get("year", "")
        title = c.get("title", ref_str)
        line = f"{marker}: {authors + ' ' if authors else ''}{title}"
        if year:
            line += f" ({year})"
        if url:
            line += f". <{url}>"
        footnotes.append(line)
    return text + "\n\n---\n\n" + "\n\n".join(footnotes)


def _cache_path(source: Source, ch_id: str) -> Path:
    p = source.workspace / "citations"
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{ch_id}.json"


def enrich(chapter: Chapter, source: Source, max_refs: int = 5) -> Chapter:
    """Add footnotes + Citation objects to ``chapter``. Best-effort."""
    cache = _cache_path(source, chapter.id)
    resolved: list[dict] = []

    if cache.exists():
        try:
            resolved = json.loads(cache.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("[citations] cache read failed for %s: %s", chapter.id, e)
            resolved = []
    else:
        try:
            refs = _find_references(chapter.body_md)
            for ref in refs[:max_refs]:
                try:
                    res = _resolve_citation(ref)
                except Exception as e:
                    log.warning("[citations] resolve failed for %r: %s", ref, e)
                    res = None
                if res:
                    res["ref_str"] = ref
                    resolved.append(res)
            try:
                cache.write_text(json.dumps(resolved, indent=2), encoding="utf-8")
            except Exception as e:
                log.warning("[citations] cache write failed for %s: %s", chapter.id, e)
        except Exception as e:
            log.warning("[citations] enrich failed for %s: %s", chapter.id, e)
            resolved = []

    if resolved:
        chapter.body_md = _inject_footnotes(chapter.body_md, resolved)
        chapter.citations = [
            Citation(
                ref_str=c.get("ref_str", ""),
                title=c.get("title", ""),
                authors=c.get("authors", ""),
                year=c.get("year", ""),
                url=c.get("url", ""),
            )
            for c in resolved
        ]
    return chapter
