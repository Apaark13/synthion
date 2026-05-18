"""Smoke tests for engine_v4.runner — no network, no ffmpeg, no LLM."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine_v4 import run, run_many
from engine_v4.types import SourceSpec


def test_run_missing_file_raises_clean_error(tmp_path: Path):
    spec = SourceSpec(location=str(tmp_path / "does_not_exist.txt"), kind="text")
    with pytest.raises(FileNotFoundError):
        run(spec)


def test_run_many_with_text_specs_produces_textbook(tmp_path: Path):
    # Build a couple of trivial text-file specs. Even if ingest/understand
    # are not yet implemented (or fail), the runner must still produce a
    # textbook.md and a valid BuildResult.
    f1 = tmp_path / "a.txt"
    f1.write_text("Lecture A: gravity attracts.\n", encoding="utf-8")
    f2 = tmp_path / "b.txt"
    f2.write_text("Lecture B: light scatters.\n", encoding="utf-8")

    specs = [
        SourceSpec(location=str(f1), kind="text", title="Lecture A"),
        SourceSpec(location=str(f2), kind="text", title="Lecture B"),
    ]
    result = run_many(specs, max_concurrency=2)

    assert result.textbook_md is not None
    assert result.textbook_md.exists()
    assert len(result.states) == 2

    # run.json sidecar should exist and be parseable.
    summary = json.loads((result.workspace / "run.json").read_text(encoding="utf-8"))
    assert summary["sources_processed"] == 2
    assert summary["run_id"] == result.run_id


def test_run_many_rejects_non_spec():
    with pytest.raises(TypeError):
        run_many(["not a spec"])  # type: ignore[list-item]
