"""agents/01_chunk_transcribe.py

Hardened chunk-and-transcribe agent (Batch 2).

Features
- Creates TRANSCRIPT_DIR if missing
- MLX ASR (Apple Silicon) preferred
- HuggingFace pipeline fallback
- Chunks WAVs using ffmpeg
- Writes chunk transcripts
- Combines into batch transcripts
- Writes final *_raw.txt
"""

from pathlib import Path
from datetime import datetime, timezone
import subprocess
import json
import torch

from config.settings import (
    WAV_DIR,
    TRANSCRIPT_DIR,
    CHUNK_SECS,
    ASR_MODEL,
)

from agents.template import (
    exponential_backoff,
    log_error,
    cleanup_temp,
)


# --------------------------------------------------
# Checkpoint Helpers
# --------------------------------------------------

def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"

    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}

    data["step"] = 1
    data.setdefault("completed_tasks", [])

    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)

    data["timestamp"] = datetime.now(timezone.utc).isoformat()

    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"

    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


# --------------------------------------------------
# ASR Loaders
# --------------------------------------------------

def _make_mlx_asr(model_path):
    try:
        import mlx.core as mx

        model = mx.load(model_path)

        class MLXASR:
            def __init__(self, mdl):
                self.model = mdl

            def __call__(self, audio_path):
                try:
                    res = self.model.transcribe(str(audio_path))
                    if isinstance(res, dict):
                        return res
                    return {"text": str(res)}
                except Exception as e:
                    return {"text": "", "error": str(e)}

        return MLXASR(model)

    except Exception:
        return None


def _make_pipeline_asr(model_path):
    try:
        from transformers import pipeline

        return pipeline(
            "automatic-speech-recognition",
            model=str(model_path),
            device=-1
        )
    except Exception:
        return None


# --------------------------------------------------
# Core Work Function
# --------------------------------------------------

def _work(wav_dir: Path):

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    # Prefer MLX
    asr = _make_mlx_asr(ASR_MODEL)

    if asr is None:
        print("⚠️ MLX ASR failed — using Transformers pipeline")
        asr = _make_pipeline_asr(ASR_MODEL)

    if asr is None:
        raise RuntimeError("No ASR backend available")

    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    batch_size = 15

    for wav in wav_files:

        stem = wav.stem
        print(f"🎧 Processing {wav.name}")

        # -------------------------
        # Get duration
        # -------------------------

        try:
            out = subprocess.check_output([
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(wav)
            ])
            duration = float(out.decode().strip())
        except Exception:
            duration = CHUNK_SECS
        print(f"⏱️ Duration: {duration:.2f} seconds")
        n_chunks = max(1, int((duration + CHUNK_SECS - 1) // CHUNK_SECS))

        temp_chunks = []

        # -------------------------
        # Chunk audio
        # -------------------------

        for i in range(n_chunks):

            start = i * CHUNK_SECS
            out_wav = TRANSCRIPT_DIR / f"{stem}_{i:03d}.wav"

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-t", str(CHUNK_SECS),
                "-i", str(wav),
                "-ar", "16000",
                "-ac", "1",
                str(out_wav)
            ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                temp_chunks.append(out_wav)
            except Exception:
                continue

        # -------------------------
        # Transcribe chunks
        # -------------------------

        chunk_text_files = []

        for idx, chunk_wav in enumerate(temp_chunks):

            chunk_txt = TRANSCRIPT_DIR / f"{stem}_{idx:03d}.txt"

            try:

                res = asr(str(chunk_wav))

                text = (
                    res.get("text")
                    if isinstance(res, dict)
                    else str(res)
                )

                if not text:
                    text = f"[EMPTY TRANSCRIPT] {chunk_wav.name}"

                with open(chunk_txt, "w", encoding="utf-8") as f:
                    f.write(text.strip())

                chunk_text_files.append(chunk_txt)

            except Exception:
                continue

        # -------------------------
        # Combine batches
        # -------------------------

        batch_texts = []

        for b in range(0, len(chunk_text_files), batch_size):

            group = chunk_text_files[b:b + batch_size]

            batch_idx = b // batch_size
            out_txt = TRANSCRIPT_DIR / f"{stem}_batch_{batch_idx:03d}.txt"

            try:
                with open(out_txt, "w", encoding="utf-8") as outf:

                    for p in group:

                        try:
                            outf.write(p.read_text(encoding="utf-8"))
                            outf.write("\n")
                        except Exception:
                            continue

                batch_texts.append(out_txt)

            except Exception:
                continue

        # Delete the original per-chunk text files now that batch files have been written
        for p in chunk_text_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        # -------------------------
        # Final raw transcript
        # -------------------------

        agg = TRANSCRIPT_DIR / f"{stem}_raw.txt"

        with open(agg, "w", encoding="utf-8") as f:

            for p in batch_texts:
                try:
                    f.write(p.read_text(encoding="utf-8"))
                    f.write("\n")
                except Exception:
                    continue

    return TRANSCRIPT_DIR


# --------------------------------------------------
# Public Step
# --------------------------------------------------

def step_1(wav_dir: Path = WAV_DIR) -> Path:

    try:

        out = exponential_backoff(
            lambda: _work(wav_dir),
            step=1
        )

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        _update_checkpoint(
            "agents/01_chunk_transcribe.py implemented"
        )

        _append_done(
            f"[v5] {datetime.now(timezone.utc).date()} · Copilot · agents/01_chunk_transcribe.py implemented"
        )

        cleanup_temp()

        return out

    except Exception as e:

        log_error(1, e)
        raise