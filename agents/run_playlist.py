"""agents/run_playlist.py — Playlist-to-Textbook Runner

Processes a YouTube playlist end-to-end, producing one chapter per lecture
and assembling a single multi-chapter textbook.

Usage:
    python -m agents.run_playlist <playlist_url> [--max N] [--start N]

Each video is processed sequentially:
  1. Fetch (yt-dlp) + content sensing
  2. Multimodal extraction (Whisper + Gemma 4)
  3. Pedagogical ToC (content-aware chapter count)
  4. KV-cache store
  5. Writer (Gemma 4 prose)
  6. Citation
  7. Evaluation (Ragas gate)
  8. Image upscale
  9. Per-lecture chapter files accumulated

After all lectures: publish combined textbook.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import (
    WORKSPACE, BOOK_DIR, VIDEO_LOW_RES_DIR, VIDEO_HQ_DIR,
    TRANSCRIPT_DIR, KV_CACHE_DIR, IMAGES_DIR,
)
from agents.template import log_error


def _reset_workspace_for_lecture(lecture_idx: int, lecture_title: str):
    """Archive current workspace state and prepare a fresh run."""
    archive = WORKSPACE / "lectures" / f"lec_{lecture_idx:03d}"
    for src in [VIDEO_LOW_RES_DIR, VIDEO_HQ_DIR, TRANSCRIPT_DIR, IMAGES_DIR]:
        dst = archive / src.name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src, ignore_errors=True)
        src.mkdir(parents=True, exist_ok=True)
    # Clear KV-cache for new lecture
    for f in KV_CACHE_DIR.glob("*.npz"):
        f.unlink()


def run_playlist(playlist_url: str, max_videos: int = 0, start_at: int = 0) -> Path:
    """Process every video in a playlist and return the assembled textbook path.

    Args:
        playlist_url: YouTube playlist URL.
        max_videos: Maximum number of videos to process (0 = all).
        start_at: Skip the first N videos (0-based index).

    Returns:
        Path to the assembled textbook.md
    """
    from agents import fetch_agent, multimodal_extract, pedagogical_agent
    from agents import kv_cache_agent, writer_agent, citation_agent
    from agents import evaluation_agent, image_upscale_agent, publish_agent

    print(f"[playlist] Fetching video list from: {playlist_url}")
    videos = fetch_agent.get_playlist_videos(playlist_url)
    print(f"[playlist] Found {len(videos)} videos")

    if start_at > 0:
        videos = videos[start_at:]
        print(f"[playlist] Skipping first {start_at} videos; processing {len(videos)}")
    if max_videos > 0:
        videos = videos[:max_videos]
        print(f"[playlist] Limiting to {max_videos} videos")

    all_chapters: dict[str, str] = {}  # {lecture_prefixed_ch_id: markdown}
    all_manifests: list[dict] = []

    for idx, video in enumerate(videos):
        url = video["url"]
        title = video["title"]
        lecture_num = start_at + idx + 1
        print(f"\n{'='*60}")
        print(f"[playlist] Lecture {lecture_num}/{len(videos) + start_at}: {title}")
        print(f"{'='*60}")

        # Archive + reset workspace between lectures
        if idx > 0:
            _reset_workspace_for_lecture(lecture_num - 1, title)

        try:
            # Step 0: Fetch
            fetch_result = fetch_agent.step_0(url)
            duration_sec = fetch_result.get("video_duration_sec", 0)

            # Step 1: Extract
            log_path = multimodal_extract.step_1(fetch_result["low_res_path"])
            with open(log_path, encoding="utf-8") as f:
                mlog = json.load(f)

            # Step 2: Pedagogical (content-aware)
            toc = pedagogical_agent.step_2(mlog, video_duration_sec=duration_sec)

            # Prefix chapter IDs with lecture number to avoid collisions
            for sec in toc.get("sections", []):
                sec["id"] = f"lec{lecture_num:03d}_{sec['id']}"
                sec["prereqs"] = [
                    f"lec{lecture_num:03d}_{p}" for p in sec.get("prereqs", [])
                ]

            # Step 3: KV-cache
            kv_cache_agent.step_3(mlog, cache_id=f"lecture_{lecture_num:03d}")

            # Step 4: Writer
            chapters = writer_agent.step_4(toc=toc, multimodal_log=mlog)

            # Step 5: Citation
            cite_result = citation_agent.step_5(chapters)
            if isinstance(cite_result, tuple):
                chapters, _ = cite_result

            # Step 6: Evaluation
            eval_result = evaluation_agent.step_6(chapters)
            if isinstance(eval_result, tuple):
                scores, _ = eval_result
            else:
                scores = eval_result

            n_pass = sum(1 for v in scores.values()
                        if (v.get("approved") if isinstance(v, dict) else v >= 0.5))
            print(f"[playlist] Lecture {lecture_num}: {n_pass}/{len(chapters)} chapters passed eval")

            # Add lecture heading chapter
            lecture_heading = (
                f"# Lecture {lecture_num}: {title}\n\n"
                f"*Video: {url}*\n\n"
                f"*Duration: {duration_sec/60:.0f} minutes*\n\n---\n\n"
            )
            heading_id = f"lec{lecture_num:03d}_heading"
            all_chapters[heading_id] = lecture_heading
            all_chapters.update(chapters)

            # Step 7: Images
            images = image_upscale_agent.step_7(
                flagged_timestamps=fetch_result.get("flagged_timestamps", []),
                hq_segments=fetch_result.get("hq_segments", []),
                multimodal_log=mlog,
            )
            manifest_path = IMAGES_DIR / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    lecture_manifest = json.load(f)
                all_manifests.extend(lecture_manifest)

        except Exception as e:
            log_error(0, e)
            print(f"[playlist] ✗ Error processing lecture {lecture_num}: {e}")
            print(f"[playlist]   Continuing with next lecture...")
            continue

        print(f"[playlist] ✓ Lecture {lecture_num} complete ({len(chapters)} chapters)")

    # Final publish
    print(f"\n[playlist] Assembling full textbook ({len(all_chapters)} chapters)...")
    pdf_path, epub_path = publish_agent.step_8(all_chapters, all_manifests)

    # Write combined textbook path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary_path = WORKSPACE / "output" / f"playlist_run_{timestamp}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "playlist_url": playlist_url,
            "videos_processed": len(videos),
            "total_chapters": len(all_chapters),
            "epub": str(epub_path),
            "timestamp": timestamp,
        }, f, indent=2)

    print(f"[playlist] ✅ Done! {len(all_chapters)} chapters")
    print(f"[playlist] EPUB: {epub_path}")
    print(f"[playlist] Summary: {summary_path}")
    return epub_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process a YouTube playlist into a textbook")
    parser.add_argument("url", help="YouTube playlist URL")
    parser.add_argument("--max", type=int, default=0, help="Max videos to process (0=all)")
    parser.add_argument("--start", type=int, default=0, help="Skip first N videos")
    args = parser.parse_args()
    run_playlist(args.url, max_videos=args.max, start_at=args.start)
