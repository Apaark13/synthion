"""Tests for engine_v4.figures."""
from pathlib import Path

from engine_v4.figures import resolve_anchors
from engine_v4.types import Frame


def _frame(ts: float, name: str = "f.png", caption: str = "A frame") -> Frame:
    return Frame(timestamp_sec=ts, path=Path(f"/tmp/{name}"), caption=caption)


def test_resolve_in_tolerance_replaces_with_image():
    body = "Intro.\n\n[FIG t=00:01:00]\n\nMore."
    frames = [_frame(55.0, "near.png", "Near frame")]
    out, used = resolve_anchors(body, frames, tolerance_sec=120)
    assert '<img src="/tmp/near.png"' in out
    assert "Near frame" in out
    assert "[FIG" not in out
    assert used == frames


def test_resolve_out_of_tolerance_drops_anchor():
    body = "Intro.\n\n[FIG t=00:10:00]\n\nMore."
    frames = [_frame(0.0, "far.png")]
    out, used = resolve_anchors(body, frames, tolerance_sec=60)
    assert "[FIG" not in out
    assert "<img" not in out
    assert used == []


def test_resolve_no_frames_drops_anchor():
    body = "x [FIG t=00:00:30] y"
    out, used = resolve_anchors(body, [], tolerance_sec=120)
    assert "[FIG" not in out
    assert used == []


def test_legacy_form_supported():
    body = "x [Figure: timestamp=00:00:10 desc=stuff] y"
    frames = [_frame(12.0, "img.png", "Legacy")]
    out, used = resolve_anchors(body, frames, tolerance_sec=30)
    assert '<img src="/tmp/img.png"' in out
    assert "Legacy" in out
    assert used == frames


def test_resolve_picks_nearest_of_many():
    body = "[FIG t=00:00:30]"
    frames = [_frame(0.0, "a.png", "A"), _frame(28.0, "b.png", "B"), _frame(120.0, "c.png", "C")]
    out, used = resolve_anchors(body, frames, tolerance_sec=60)
    assert '<img src="/tmp/b.png"' in out
    assert len(used) == 1 and used[0].path.name == "b.png"


def test_select_section_frames_picks_in_band():
    from engine_v4.figures import select_section_frames
    from engine_v4.types import OutlineSection
    sec = OutlineSection(id="ch_001", title="t", timestamps=[100.0, 200.0])
    frames = [_frame(t, f"f{int(t)}.png") for t in (50, 120, 140, 160, 180, 500)]
    picked = select_section_frames(sec, frames, n=3, pad_sec=10)
    # 120, 140, 160, 180 are in [90..210] band -> we picked 3 of them spaced.
    ts = [f.timestamp_sec for f in picked]
    assert len(picked) == 3
    assert all(90 <= t <= 210 for t in ts)


def test_inject_extra_frames_skips_already_used():
    from engine_v4.figures import inject_extra_frames
    f1 = _frame(10.0, "a.png", "A")
    f2 = _frame(20.0, "b.png", "B")
    body = "## Heading\n\nIntro paragraph.\n\n### Sub\nMore.\n"
    body2, used = inject_extra_frames(body, [f1, f2], already_used=[f1])
    # f1 already used → only f2 gets injected
    assert '<img src="/tmp/b.png"' in body2
    assert body2.count('<img src="/tmp/a.png"') == 0
    assert f2 in used
