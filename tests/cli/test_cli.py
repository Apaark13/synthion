"""CLI smoke tests using click.testing.CliRunner."""
from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from cli.mte import _sniff_kind, cli


def test_sniff_kind_paths_and_urls():
    assert _sniff_kind("https://www.youtube.com/watch?v=abc") == "youtube"
    assert _sniff_kind("https://example.com/page") == "url"
    assert _sniff_kind("clip.mp4") == "video"
    assert _sniff_kind("audio.wav") == "audio"
    assert _sniff_kind("paper.pdf") == "pdf"
    assert _sniff_kind("notes.md") == "text"
    assert _sniff_kind("weird.bin") == "custom"


def test_cli_help_lists_subcommands():
    runner = CliRunner()
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0, res.output
    for cmd in ("build", "build-many", "sync", "info"):
        assert cmd in res.output


def test_cli_version_flag():
    res = CliRunner().invoke(cli, ["--version"])
    assert res.exit_code == 0
    assert "0.4.0" in res.output


def test_cli_info_runs():
    res = CliRunner().invoke(cli, ["info"])
    assert res.exit_code == 0, res.output
    assert "python:" in res.output
    assert "ffmpeg:" in res.output


def test_cli_build_text_source(tmp_path: Path, monkeypatch):
    src = tmp_path / "lesson.txt"
    src.write_text(
        "Linear regression fits a straight line through data.\n"
        "It minimises the sum of squared residuals via the normal equations.\n"
        "Gradient descent is an alternative iterative optimiser.\n" * 4,
        encoding="utf-8",
    )
    out = tmp_path / "ws"
    monkeypatch.setenv("MTE_WORKSPACE_ROOT", str(out))
    res = CliRunner().invoke(cli, ["build", str(src), "--kind", "text", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "run_id" in res.output


def test_cli_build_many_from_file(tmp_path: Path, monkeypatch):
    a = tmp_path / "a.txt"; a.write_text("alpha alpha alpha discussion of alpha topic.\n" * 6, encoding="utf-8")
    b = tmp_path / "b.txt"; b.write_text("beta beta beta discussion of beta topic.\n" * 6, encoding="utf-8")
    listing = tmp_path / "sources.txt"
    listing.write_text(f"{a}\n# comment\n{b}\n", encoding="utf-8")
    out = tmp_path / "ws"
    monkeypatch.setenv("MTE_WORKSPACE_ROOT", str(out))
    res = CliRunner().invoke(cli, [
        "build-many", "--from-file", str(listing),
        "--out", str(out), "--max-concurrency", "1", "--format", "json",
    ])
    assert res.exit_code == 0, res.output
    # Output may contain leading log lines; extract the JSON object.
    start = res.output.find("{")
    end = res.output.rfind("}")
    assert start >= 0 and end > start, f"no JSON in output:\n{res.output}"
    payload = json.loads(res.output[start:end + 1])
    assert "run_id" in payload


def test_cli_sync_with_stubbed_pipeline(tmp_path: Path, monkeypatch):
    transcript = tmp_path / "tx.json"
    transcript.write_text(json.dumps([
        {"start": 0.0, "end": 5.0, "text": "intro topic alpha"},
        {"start": 10.0, "end": 15.0, "text": "topic beta deep dive"},
    ]), encoding="utf-8")
    video = tmp_path / "fake.mp4"; video.write_bytes(b"\x00")

    from modules.visual_sync import scene as scene_mod
    from modules.visual_sync import extract as extract_mod
    from PIL import Image
    import numpy as np

    monkeypatch.setattr(scene_mod, "detect_scenes",
                        lambda *a, **kw: [(2.0, 0.9), (12.0, 0.95)])

    def _fake_extract(_v, timestamps, output_dir, **_kw):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out = []
        for ts in timestamps:
            p = Path(output_dir) / f"f_{ts:.3f}.png"
            arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(p)
            out.append((ts, str(p)))
        return out

    monkeypatch.setattr(extract_mod, "extract_frames", _fake_extract)

    out_dir = tmp_path / "sync_out"
    res = CliRunner().invoke(cli, [
        "sync", str(video), str(transcript), "--out", str(out_dir), "--ocr-backend", "none",
    ])
    assert res.exit_code == 0, res.output
    assert (out_dir / "figures.json").exists()
    data = json.loads((out_dir / "figures.json").read_text())
    assert "all" in data
