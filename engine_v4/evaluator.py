"""engine_v4.evaluator — heuristic faithfulness gate.

Ports the heuristic Q&A path of ``agents/06_evaluation_agent.py`` (skipping
Ragas). When the Gemma model is available it is asked to answer questions
about the chapter using only the chapter text; otherwise we fall back to a
keyword-overlap heuristic. Score >= 0.7 ⇒ approved.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from .limits import LLM
from .llm import call_gemma, get_gemma
from .types import Chapter

log = logging.getLogger("engine_v4.evaluator")

THRESHOLD = 0.70

_STOPWORDS = {
    "which", "where", "about", "their", "these", "those", "would",
    "could", "should", "there", "while", "until", "after", "before",
}


def _generate_questions(chapter_text: str, n: int = 5) -> list[str]:
    """Heuristic question generation from declarative-looking sentences."""
    sentences = [s.strip() for s in re.split(r"[.!?]", chapter_text) if len(s.strip()) > 30]
    questions = [f"What does the author mean by: '{s[:80]}...'?" for s in sentences[:n]]
    return questions or ["What is the main topic of this chapter?"]


def _answer_question(question: str, chapter_text: str) -> tuple[str, float]:
    """Answer a single question. Uses Gemma when available, else heuristic."""
    llm = get_gemma()
    if llm is not None:
        snippet = chapter_text[:2000]
        prompt = (
            "Using ONLY the following text, answer this question. "
            "If the answer is not in the text, say 'NOT FOUND'.\n"
            f"Text:\n'''{snippet}\n'''\n"
            f"Question: {question}\nAnswer:"
        )
        with LLM:
            answer = call_gemma(prompt, max_tokens=150, temperature=0.1)
        if answer:
            score = 0.0 if "NOT FOUND" in answer.upper() else 0.8
            return answer, score

    keywords = [
        w for w in re.findall(r"\b[a-zA-Z]{5,}\b", question)
        if w.lower() not in _STOPWORDS
    ]
    if not keywords:
        return "[heuristic] no keywords", 0.5
    found = sum(1 for kw in keywords if kw.lower() in chapter_text.lower())
    score = min(1.0, found / len(keywords))
    return f"[heuristic] keyword overlap {found}/{len(keywords)}", score


def score(chapter: Chapter) -> tuple[float, str]:
    """Return ``(mean_score, critique)``. Empty critique iff approved."""
    text = chapter.body_md or ""
    if len(text.strip()) < 40:
        return 0.0, f"Chapter {chapter.id} body is empty or too short."

    questions = _generate_questions(text, n=5)
    scores: list[float] = []
    weak: list[str] = []
    for q in questions:
        _ans, sc = _answer_question(q, text)
        scores.append(sc)
        if sc < 0.5:
            weak.append(q)

    mean = sum(scores) / len(scores) if scores else 0.0
    if mean >= THRESHOLD:
        return mean, ""
    critique = (
        f"Chapter {chapter.id} scored {mean:.2f} (threshold {THRESHOLD:.2f}). "
        "Improve faithfulness and coverage.\n"
        + "\n".join(f"- Question not well-answered: '{q}'" for q in weak[:3])
    )
    return mean, critique
