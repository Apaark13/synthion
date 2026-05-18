"""tests/test_eval.py — Smoke tests for agents/05_citation_agent.py
                        and agents/06_evaluation_agent.py"""
from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import pytest


# ── Citation tests ────────────────────────────────────────────────────────────

def test_find_references_detects_inline_citations():
    """_find_references should detect common inline citation formats."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "citation_mod",
        str(Path(__file__).parent.parent / "agents" / "05_citation_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = (
        "As shown by Vaswani et al., 2017, the transformer architecture "
        'is foundational. See "Attention Is All You Need" (2017) for details.'
    )
    refs = mod._find_references(text)
    assert len(refs) >= 1
    assert any("2017" in r for r in refs)


def test_inject_footnotes_appends_to_text():
    """_inject_footnotes should append footnote definitions at end of text."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "citation_mod",
        str(Path(__file__).parent.parent / "agents" / "05_citation_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    text = "The transformer model Vaswani et al., 2017 is important."
    citations = [{
        "ref_str": "Vaswani et al., 2017",
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "url": "https://arxiv.org/abs/1706.03762",
    }]
    result = mod._inject_footnotes(text, citations)
    assert "[^1]:" in result
    assert "Attention Is All You Need" in result


# ── Evaluation tests ──────────────────────────────────────────────────────────

def test_evaluation_pass_threshold():
    """Chapters with high keyword overlap should pass the threshold."""
    from config.settings import RAGAS_THRESHOLD
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_mod",
        str(Path(__file__).parent.parent / "agents" / "06_evaluation_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Mock out LLM calls; use heuristic path
    with patch.object(mod, "_generate_questions", return_value=["What is attention?",
                                                                  "How does softmax work?"]), \
         patch.object(mod, "_answer_question", return_value=("Good answer with attention and softmax", 0.9)):
        score, critique = mod._ragas_evaluate(
            "## Attention\nAttention mechanisms allow transformers to weigh token importance. "
            "Softmax normalises attention scores across all positions.",
            "ch_001"
        )

    assert score >= RAGAS_THRESHOLD
    assert critique == ""


def test_evaluation_fail_produces_critique():
    """Chapters with low scores should produce a non-empty critique."""
    from config.settings import RAGAS_THRESHOLD
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_mod2",
        str(Path(__file__).parent.parent / "agents" / "06_evaluation_agent.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with patch.object(mod, "_generate_questions", return_value=["What is quantum entanglement?"]), \
         patch.object(mod, "_answer_question", return_value=("NOT FOUND", 0.0)):
        score, critique = mod._ragas_evaluate("## Attention\nSimple text.", "ch_002")

    assert score < RAGAS_THRESHOLD
    assert len(critique) > 0
