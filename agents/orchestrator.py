"""agents/orchestrator.py — v3 LangGraph Multi-Agent Orchestrator

Defines the PipelineState TypedDict and wires all agents into a
LangGraph StateGraph with evaluation-gated retry loops.

v3 changes:
  - No HQ segment downloads (frames from 360p via scene detection)
  - Scene frames extracted in extract step, not upscale step
  - Batched writer call (one LLM call per video)

Graph edges:
    fetch → extract → pedagogical → kv_cache → writer
    writer → citation → evaluation
    evaluation → publish          (score ≥ RAGAS_THRESHOLD)
    evaluation → writer           (score < RAGAS_THRESHOLD, up to max retries)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import RAGAS_THRESHOLD, RAGAS_MAX_RETRIES, ERRORS_LOG
from agents.template import log_error

# ── State schema ──────────────────────────────────────────────────────────────

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class PipelineState(TypedDict, total=False):
    url: str
    # Layer 1
    low_res_path: str
    hq_segments: list[str]
    flagged_timestamps: list[float]
    video_duration_sec: float
    multimodal_log: list[dict]
    # Layer 2
    toc: dict
    chapters: dict[str, str]
    citations: dict[str, list]
    critique: dict[str, str]
    retry_counts: dict[str, int]
    # Layer 3
    eval_scores: dict[str, float]
    approved: list[str]
    hq_images: list[str]
    output_pdf: str
    output_epub: str


# ── Node wrappers ─────────────────────────────────────────────────────────────

def node_fetch(state: PipelineState) -> PipelineState:
    from agents import fetch_agent as fa
    result = fa.step_0(state["url"])
    return {
        **state,
        "low_res_path": str(result["low_res_path"]),
        "hq_segments": [],
        "flagged_timestamps": [],
        "video_duration_sec": result.get("video_duration_sec", 0),
    }


def node_extract(state: PipelineState) -> PipelineState:
    from agents import multimodal_extract as me
    log_path = me.step_1(Path(state["low_res_path"]) if state.get("low_res_path") else None)
    with open(log_path, encoding="utf-8") as f:
        log_entries = json.load(f)
    return {**state, "multimodal_log": log_entries}


def node_pedagogical(state: PipelineState) -> PipelineState:
    from agents import pedagogical_agent as pa
    toc = pa.step_2(state["multimodal_log"],
                    video_duration_sec=state.get("video_duration_sec"))
    return {**state, "toc": toc}


def node_kv_cache(state: PipelineState) -> PipelineState:
    from agents import kv_cache_agent as kv
    kv.step_3(state["multimodal_log"])
    return state


def node_writer(state: PipelineState) -> PipelineState:
    from agents import writer_agent as wa
    chapters = wa.step_4(
        toc=state["toc"],
        critique=state.get("critique", {}),
        multimodal_log=state.get("multimodal_log"),
    )
    retry_counts = dict(state.get("retry_counts") or {})
    for ch_id in chapters:
        if state.get("critique", {}).get(ch_id):
            retry_counts[ch_id] = retry_counts.get(ch_id, 0) + 1
    return {**state, "chapters": chapters, "retry_counts": retry_counts}


def node_citation(state: PipelineState) -> PipelineState:
    from agents import citation_agent as ca
    chapters, citations = ca.step_5(state["chapters"])
    return {**state, "chapters": chapters, "citations": citations}


def node_evaluation(state: PipelineState) -> PipelineState:
    from agents import evaluation_agent as ea
    scores, critiques = ea.step_6(state["chapters"])
    approved = [ch for ch, score in scores.items() if score >= RAGAS_THRESHOLD]
    new_critique = {ch: c for ch, c in critiques.items() if ch not in approved}
    return {
        **state,
        "eval_scores": scores,
        "critique": new_critique,
        "approved": approved,
    }


def _should_retry(state: PipelineState) -> str:
    """Routing function: 'retry' or 'publish'."""
    unapproved = [
        ch for ch in (state.get("chapters") or {})
        if ch not in (state.get("approved") or [])
    ]
    if not unapproved:
        return "publish"
    retry_counts = state.get("retry_counts") or {}
    if any(retry_counts.get(ch, 0) < RAGAS_MAX_RETRIES for ch in unapproved):
        return "retry"
    return "publish"


def node_publish(state: PipelineState) -> PipelineState:
    from agents import publish_agent as pub
    pdf, epub = pub.step_8(
        chapters=state["chapters"],
        citations=state.get("citations", {}),
        toc=state["toc"],
        images=state.get("hq_images", []),
    )
    return {**state, "output_pdf": str(pdf), "output_epub": str(epub)}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph StateGraph. Returns a compiled graph."""
    try:
        from langgraph.graph import StateGraph, END
    except ImportError as e:
        raise ImportError(
            "langgraph is required for the orchestrator. "
            "Install with: pip install langgraph"
        ) from e

    g = StateGraph(PipelineState)

    g.add_node("fetch",       node_fetch)
    g.add_node("extract",     node_extract)
    g.add_node("pedagogical", node_pedagogical)
    g.add_node("kv_cache",    node_kv_cache)
    g.add_node("writer",      node_writer)
    g.add_node("citation",    node_citation)
    g.add_node("evaluation",  node_evaluation)
    g.add_node("publish",     node_publish)

    g.set_entry_point("fetch")
    g.add_edge("fetch",       "extract")
    g.add_edge("extract",     "pedagogical")
    g.add_edge("pedagogical", "kv_cache")
    g.add_edge("kv_cache",    "writer")
    g.add_edge("writer",      "citation")
    g.add_edge("citation",    "evaluation")
    g.add_conditional_edges(
        "evaluation",
        _should_retry,
        {"retry": "writer", "publish": "publish"},
    )
    g.add_edge("publish",     END)

    return g.compile()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(url: str) -> PipelineState:
    """Run the full pipeline for a given YouTube URL."""
    graph = build_graph()
    initial_state: PipelineState = {
        "url": url,
        "chapters": {},
        "citations": {},
        "critique": {},
        "retry_counts": {},
        "eval_scores": {},
        "approved": [],
    }
    final_state = graph.invoke(initial_state)
    return final_state


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m agents.orchestrator <youtube-url>")
        sys.exit(1)
    result = run(sys.argv[1])
    print(f"\n[orchestrator] Pipeline complete.")
    print(f"  PDF:  {result.get('output_pdf')}")
    print(f"  EPUB: {result.get('output_epub')}")
    print(f"  Approved chapters: {result.get('approved')}")
