"""tests/test_orchestrator.py — Smoke tests for agents/orchestrator.py"""
from __future__ import annotations
import pytest


def test_pipeline_state_typing():
    """PipelineState TypedDict must import without errors."""
    from agents.orchestrator import PipelineState
    state: PipelineState = {
        "url": "https://youtube.com/watch?v=test",
        "chapters": {},
        "citations": {},
        "critique": {},
        "retry_counts": {},
        "eval_scores": {},
        "approved": [],
    }
    assert state["url"] == "https://youtube.com/watch?v=test"
    assert isinstance(state["chapters"], dict)


def test_should_retry_no_unapproved():
    """_should_retry returns 'publish' when all chapters are approved."""
    from agents.orchestrator import _should_retry
    state = {
        "chapters": {"ch_001": "text"},
        "approved": ["ch_001"],
        "retry_counts": {},
    }
    assert _should_retry(state) == "publish"


def test_should_retry_with_unapproved_and_retries_left():
    """_should_retry returns 'retry' when chapters failed and retries remain."""
    from agents.orchestrator import _should_retry
    state = {
        "chapters": {"ch_001": "text", "ch_002": "text"},
        "approved": ["ch_001"],
        "retry_counts": {"ch_002": 0},  # 0 < RAGAS_MAX_RETRIES=3
    }
    assert _should_retry(state) == "retry"


def test_should_retry_exhausted():
    """_should_retry returns 'publish' when max retries reached."""
    from agents.orchestrator import _should_retry
    from config.settings import RAGAS_MAX_RETRIES
    state = {
        "chapters": {"ch_001": "text"},
        "approved": [],
        "retry_counts": {"ch_001": RAGAS_MAX_RETRIES},  # at or above max
    }
    assert _should_retry(state) == "publish"


def test_build_graph_compiles():
    """build_graph() should compile the LangGraph StateGraph without error."""
    pytest.importorskip("langgraph", reason="langgraph not installed")
    from agents.orchestrator import build_graph
    graph = build_graph()
    assert graph is not None
