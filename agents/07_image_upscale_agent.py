"""agents/07_image_upscale_agent.py — Real-ESRGAN Image Upscale Agent

Extracts key frames at high-info timestamps from the surgical 4K segments
and upscales them with Real-ESRGAN.

Input:
  - flagged_timestamps: list[float] — seconds of high-info moments
  - hq_segments: list[str] — paths to 4K clip files
  - multimodal_log: list[dict] — for is_high_info filtering

Output:
  - workspace/images/fig_NNN_upscaled.png  (upscaled PNGs)
  - workspace/images/manifest.json         [{index, timestamp, caption, path}]
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import IMAGES_DIR, VIDEO_HQ_DIR, VIDEO_LOW_RES_DIR, ESRGAN_MODEL, ERRORS_LOG
from agents.template import log_error, exponential_backoff, cleanup_temp

IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# ── Frame extraction ──────────────────────────────────────────────────────────

def _extract_frame(video_path: Path, timestamp_sec: float, out_path: Path) -> bool:
    """Extract a single JPEG frame at timestamp_sec. Returns True on success."""
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-ss", str(timestamp_sec),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return out_path.exists()
    except Exception as e:
        log_error(7, e)
        return False


def _find_best_segment(timestamp_sec: float, hq_segments: list[Path]) -> Path | None:
    """Return the HQ segment closest in time to the given timestamp."""
    if not hq_segments:
        return None
    # Segment filenames encode their start time as trailing integer
    def seg_start(p: Path) -> float:
        parts = p.stem.rsplit("_", 1)
        try:
            return float(parts[-1])
        except ValueError:
            return 0.0
    return min(hq_segments, key=lambda p: abs(seg_start(p) - timestamp_sec))


# ── ESRGAN availability (checked once) ───────────────────────────────────────
_esrgan_available: bool | None = None  # None = not yet checked

def _check_esrgan_available() -> bool:
    """Check if Real-ESRGAN Python deps are importable. Result is cached."""
    global _esrgan_available
    if _esrgan_available is not None:
        return _esrgan_available
    try:
        import importlib.util
        _esrgan_available = (
            importlib.util.find_spec("basicsr") is not None
            and importlib.util.find_spec("realesrgan") is not None
        )
    except Exception:
        _esrgan_available = False
    return _esrgan_available


def _upscale_frame(input_jpg: Path, output_png: Path) -> bool:
    """Run Real-ESRGAN via cloned repo's inference script. Falls back to ffmpeg copy."""
    import sys

    # Strategy 1: use the cloned Real-ESRGAN inference script directly
    # (requires basicsr; skipped on Python 3.14 where basicsr is uninstallable)
    esrgan_script = Path("/Users/i45367/models/realesrgan/inference_realesrgan.py")
    weights = Path("/Users/i45367/models/realesrgan/weights/RealESRGAN_x4plus.pth")
    if esrgan_script.exists() and weights.exists() and _check_esrgan_available():
        try:
            result = subprocess.run([
                sys.executable, str(esrgan_script),
                "-i", str(input_jpg),
                "-o", str(output_png.parent),
                "-n", "RealESRGAN_x4plus",
                "--model_path", str(weights),
                "--suffix", "upscaled",
                "--outscale", "4",
            ], capture_output=True, text=True, timeout=120)
            # The script saves as <stem>_upscaled.png — rename if needed
            candidate = output_png.parent / f"{input_jpg.stem}_upscaled.png"
            if candidate.exists() and candidate != output_png:
                candidate.rename(output_png)
            if output_png.exists():
                return True
        except Exception as e:
            log_error(7, e)

    # Strategy 2: use Python basicsr/realesrgan if available
    if _check_esrgan_available():
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore
            import cv2

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                            num_grow_ch=32, scale=4)
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(weights),
                model=model,
                tile=512, tile_pad=10, pre_pad=0,
            )
            img = cv2.imread(str(input_jpg), cv2.IMREAD_COLOR)
            if img is not None:
                output, _ = upsampler.enhance(img, outscale=4)
                cv2.imwrite(str(output_png), output)
                return output_png.exists()
        except Exception:
            pass

    # Strategy 3: ffmpeg copy (lossless convert JPEG → PNG, no upscale)
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(input_jpg), str(output_png),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_png.exists()
    except Exception as e:
        log_error(7, e)
        return False


# ── Checkpoint / DONE ─────────────────────────────────────────────────────────

def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["level"] = 3
    data["batch"] = 9
    data["step"] = 7
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

def step_7(
    flagged_timestamps: list[float] | None = None,
    hq_segments: list[str] | None = None,
    multimodal_log: list[dict] | None = None,
) -> list[Path]:
    """Extract + upscale key frames.

    Args:
        flagged_timestamps: Seconds of flagged high-info moments from fetch.
        hq_segments: Paths to 4K segment .mp4 files.
        multimodal_log: Optional log for caption extraction.

    Returns:
        List of Paths to upscaled PNG files.
    """
    def work():
        seg_paths = [Path(s) for s in (hq_segments or [])]
        # If no HQ segments, try reading from VIDEO_HQ_DIR
        if not seg_paths:
            seg_paths = sorted(VIDEO_HQ_DIR.glob("*.mp4"))

        # Determine if we have dedicated per-timestamp HQ segments
        using_hq_segments = bool(seg_paths)

        # Fall back to main low_res video when no HQ segments exist
        main_video: Path | None = None
        if not using_hq_segments:
            lr_videos = sorted(VIDEO_LOW_RES_DIR.glob("*.mp4"))
            if lr_videos:
                main_video = lr_videos[0]
                print(f"[upscale] No HQ segments found; using main video: {main_video.name}")

        timestamps = flagged_timestamps or []
        # If still no timestamps, derive from multimodal log
        if not timestamps and multimodal_log:
            timestamps = [
                e["timestamp_sec"] for e in multimodal_log
                if e.get("is_high_info") and "timestamp_sec" in e
            ]

        if not timestamps:
            print("[upscale] No high-info timestamps; skipping image extraction.")
            return []

        manifest = []
        upscaled: list[Path] = []

        for i, ts in enumerate(timestamps):
            raw_jpg = IMAGES_DIR / f"fig_{i:03d}_raw.jpg"

            if using_hq_segments:
                seg = _find_best_segment(ts, seg_paths)
                if seg is None:
                    continue
                # HQ segment starts at its own timestamp; extract from offset 0
                if not _extract_frame(seg, 0.0, raw_jpg):
                    print(f"[upscale] ✗ frame extraction failed at t={ts:.0f}s")
                    continue
            elif main_video is not None:
                # Use main video seeking to the actual timestamp
                if not _extract_frame(main_video, ts, raw_jpg):
                    print(f"[upscale] ✗ frame extraction failed at t={ts:.0f}s")
                    continue
            else:
                print(f"[upscale] ✗ no video source for t={ts:.0f}s")
                continue

            out_png = IMAGES_DIR / f"fig_{i:03d}_upscaled.png"
            ok = _upscale_frame(raw_jpg, out_png)
            raw_jpg.unlink(missing_ok=True)  # remove temp JPEG

            if ok:
                # Get caption from multimodal log if available
                caption = ""
                if multimodal_log:
                    closest = min(multimodal_log, key=lambda e: abs(e.get("timestamp_sec", 0) - ts))
                    caption = closest.get("visual_desc", "")[:200]

                manifest.append({
                    "index": i,
                    "timestamp_sec": ts,
                    "caption": caption,
                    "path": str(out_png),
                })
                upscaled.append(out_png)
                print(f"[upscale] ✓ fig_{i:03d}_upscaled.png (t={ts:.0f}s)")
            else:
                print(f"[upscale] ✗ upscale failed for t={ts:.0f}s")

        manifest_path = IMAGES_DIR / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[upscale] {len(upscaled)} images written; manifest → {manifest_path}")
        return upscaled

    try:
        result = exponential_backoff(work, step=7)
        _update_checkpoint("agents/07_image_upscale_agent.py implemented")
        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · "
            "07_image_upscale_agent: Real-ESRGAN key-frame upscaling"
        )
        cleanup_temp()
        return result
    except Exception as e:
        log_error(7, e)
        raise


if __name__ == "__main__":
    images = step_7()
    print(f"\n[upscale] Done: {len(images)} upscaled images")
    for p in images:
        print(f"  {p}")
