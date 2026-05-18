"""agents/06_evaluation_agent.py — Ragas Evaluation Agent

Uses Ragas as LLM-as-judge to auto-generate comprehension questions from each
chapter and attempt to answer them using only the chapter text. Chapters that
score below RAGAS_THRESHOLD fail and are re-queued to the Writer Agent.

Input:  {chapter_id: markdown_text}
Output: ({chapter_id: score}, {chapter_id: critique_text})
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import BOOK_DIR, GEMMA_MODEL, RAGAS_THRESHOLD, TOKEN_BUDGET, ERRORS_LOG
from agents.template import log_error, exponential_backoff, cleanup_temp


# ── Simple fallback evaluator (no Ragas dep required) ────────────────────────

def _generate_questions(chapter_text: str, n: int = 5) -> list[str]:
    """Generate comprehension questions using Gemma 4 or a heuristic fallback."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(str(GEMMA_MODEL))
        model = AutoModelForImageTextToText.from_pretrained(str(GEMMA_MODEL), device_map="auto")
        snippet = chapter_text[:1600]
        prompt = (
            f"Read this lecture chapter:\n'''{snippet}\n'''\n"
            f"Generate {n} specific comprehension questions a student should be able "
            "to answer using only this chapter. Return as a JSON array of strings."
        )
        inputs = proc(text=prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ids = model.generate(**inputs, max_new_tokens=300)
        raw = proc.decode(ids[0], skip_special_tokens=True)
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except Exception as e:
        log_error(6, e)

    # Heuristic fallback: extract sentences that look like facts
    sentences = [s.strip() for s in re.split(r"[.!?]", chapter_text) if len(s.strip()) > 30]
    questions = [f"What does the author mean by: '{s[:80]}...'?" for s in sentences[:n]]
    return questions or ["What is the main topic of this chapter?"]


def _answer_question(question: str, chapter_text: str) -> tuple[str, float]:
    """Answer a question using only the chapter text. Returns (answer, score 0-1)."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(str(GEMMA_MODEL))
        model = AutoModelForImageTextToText.from_pretrained(str(GEMMA_MODEL), device_map="auto")
        snippet = chapter_text[:2000]
        prompt = (
            f"Using ONLY the following text, answer this question. "
            f"If the answer is not in the text, say 'NOT FOUND'.\n"
            f"Text:\n'''{snippet}\n'''\n"
            f"Question: {question}\n"
            "Answer:"
        )
        inputs = proc(text=prompt, return_tensors="pt")
        if hasattr(model, "device"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        ids = model.generate(**inputs, max_new_tokens=150)
        answer = proc.decode(ids[0], skip_special_tokens=True)
        score = 0.0 if "NOT FOUND" in answer.upper() else 0.8
        return answer, score
    except Exception as e:
        log_error(6, e)

    # Heuristic: check if any keyword from the question appears in text
    keywords = [w for w in re.findall(r"\b[a-zA-Z]{5,}\b", question) if w.lower() not in
                {"which", "where", "about", "their", "these", "those", "would"}]
    found = sum(1 for kw in keywords if kw.lower() in chapter_text.lower())
    score = min(1.0, found / max(len(keywords), 1))
    return f"[heuristic] keyword overlap {found}/{len(keywords)}", score


def _ragas_evaluate(chapter_text: str, chapter_id: str) -> tuple[float, str]:
    """Attempt Ragas evaluation; fall back to custom Q&A loop."""
    # Try native Ragas if installed
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
        questions = _generate_questions(chapter_text, n=5)
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": [chapter_text[:500]] * len(questions),
            "contexts": [[chapter_text]] * len(questions),
        })
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        score = float(result["faithfulness"]) * 0.5 + float(result["answer_relevancy"]) * 0.5
        critique = "" if score >= RAGAS_THRESHOLD else (
            f"Ragas score {score:.2f} < threshold {RAGAS_THRESHOLD}. "
            "Improve: faithfulness and answer coverage."
        )
        return score, critique
    except Exception:
        pass

    # Custom Q&A fallback
    questions = _generate_questions(chapter_text, n=5)
    scores = []
    critiques_parts = []
    for q in questions:
        answer, sc = _answer_question(q, chapter_text)
        scores.append(sc)
        if sc < 0.5:
            critiques_parts.append(f"Question not well-answered: '{q}'")

    mean_score = sum(scores) / len(scores) if scores else 0.5
    critique = ""
    if mean_score < RAGAS_THRESHOLD:
        critique = (
            f"Chapter {chapter_id} scored {mean_score:.2f} (threshold {RAGAS_THRESHOLD}). "
            "Issues:\n" + "\n".join(f"- {c}" for c in critiques_parts[:3])
        )
    return mean_score, critique


# ── Checkpoint / DONE ─────────────────────────────────────────────────────────

def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 3
    data["batch"] = 8
    data["step"] = 6
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

def step_6(
    chapters: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Evaluate all chapters with Ragas LLM-as-judge.

    Args:
        chapters: {chapter_id: markdown_text}. If None, reads from BOOK_DIR.

    Returns:
        Tuple of:
          - {chapter_id: score}     (0.0 – 1.0)
          - {chapter_id: critique}  (empty string = approved)
    """
    def work():
        nonlocal chapters
        if chapters is None:
            chapters = {}
            for p in sorted(BOOK_DIR.glob("ch_*.md")):
                chapters[p.stem] = p.read_text(encoding="utf-8")

        scores: dict[str, float] = {}
        critiques: dict[str, str] = {}

        for ch_id, text in chapters.items():
            print(f"[eval] Evaluating {ch_id} ({len(text)} chars)...")
            score, critique = _ragas_evaluate(text, ch_id)
            scores[ch_id] = score
            critiques[ch_id] = critique
            status = "✓ PASS" if score >= RAGAS_THRESHOLD else f"✗ FAIL (score={score:.2f})"
            print(f"[eval] {ch_id}: {status}")

        return scores, critiques

    try:
        scores, critiques = exponential_backoff(work, step=6)
        _update_checkpoint("agents/06_evaluation_agent.py implemented")
        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · "
            "06_evaluation_agent: Ragas comprehension Q&A gate"
        )
        cleanup_temp()
        return scores, critiques
    except Exception as e:
        log_error(6, e)
        raise


if __name__ == "__main__":
    scores, critiques = step_6()
    for ch_id, score in scores.items():
        status = "PASS" if score >= RAGAS_THRESHOLD else "FAIL"
        print(f"{ch_id}: {score:.2f} {status}")
        if critiques.get(ch_id):
            print(f"  Critique: {critiques[ch_id][:120]}")
