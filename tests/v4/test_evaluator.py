"""Tests for engine_v4.evaluator."""
from unittest.mock import patch

from engine_v4 import evaluator
from engine_v4.types import Chapter


def _chapter(text: str) -> Chapter:
    return Chapter(id="ch_001", title="X", body_md=text)


def test_score_above_threshold_approves():
    text = (
        "## Lecture\n"
        "Photosynthesis converts sunlight into chemical energy in plants. "
        "Chloroplasts contain chlorophyll which absorbs photons and powers carbon fixation. "
        "Carbon dioxide and water react to produce glucose and oxygen as outputs. "
        "Mitochondria perform respiration using glucose to generate ATP."
    )
    # Force the heuristic path (no LLM available).
    with patch.object(evaluator, "get_gemma", return_value=None):
        sc, critique = evaluator.score(_chapter(text))
    assert sc >= evaluator.THRESHOLD
    assert critique == ""


def test_score_below_threshold_yields_critique():
    text = "## A\n" + ("alpha bravo charlie delta echo foxtrot golf hotel india juliet. " * 4)

    def fake_answer(q, t):
        return "NOT FOUND in text", 0.0

    with patch.object(evaluator, "_answer_question", side_effect=fake_answer):
        sc, critique = evaluator.score(_chapter(text))
    assert sc < evaluator.THRESHOLD
    assert critique
    assert "ch_001" in critique


def test_empty_chapter_fails_fast():
    sc, critique = evaluator.score(_chapter(""))
    assert sc == 0.0
    assert "empty" in critique.lower() or "short" in critique.lower()
