"""
config/settings.py — v3 single source of truth for all agents.
Keep constants here; never hard-code paths or model names inside agent files.

v3 changes: Qwen MLX fast models, local MiniLM, removed HQ segment settings.
"""
from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
WORKSPACE   = REPO_ROOT / "workspace"
ERRORS_LOG  = REPO_ROOT / "ERRORS.log"

# ── Workspace sub-directories ─────────────────────────────────────
VIDEO_LOW_RES_DIR  = WORKSPACE / "video" / "low_res"   # 360p streams
VIDEO_HQ_DIR       = WORKSPACE / "video" / "hq_segments"  # legacy (v3: unused)
TRANSCRIPT_DIR     = WORKSPACE / "transcripts"          # *_multimodal_log.json
KV_CACHE_DIR       = WORKSPACE / "kv_cache"             # TurboQuant compressed context
BOOK_DIR           = WORKSPACE / "book"                 # chapter_NNN.md
IMAGES_DIR         = WORKSPACE / "images"               # scene-change PNGs + manifest.json
OUTPUT_DIR         = WORKSPACE / "output"               # textbook.pdf, textbook.epub

# ── Model paths (external, read-only) ─────────────────────────────
MODELS_ROOT    = REPO_ROOT.parent.parent / "models"
MOONSHINE_ROOT = Path("/Users/i45367/model/moonshine_env/moonshine_stable/models")

# Gemma 4 — high-quality prose writer (llama-cpp, 54 tok/s)
GEMMA_MODEL    = MODELS_ROOT / "gemma4"
GEMMA_GGUF     = GEMMA_MODEL / "gemma-4-E4B-it-Q4_K_M.gguf"
GEMMA_MMPROJ   = GEMMA_MODEL / "mmproj-F16.gguf"

# Qwen3.5-0.8B MLX — fast model for ToC/eval (83 tok/s)
QWEN_MLX_MODEL = MOONSHINE_ROOT / "qwen_mlx_4bit"

# Qwen3.5-2B MLX — thinking model for complex structure (58 tok/s)
QWEN2B_MLX_MODEL = MOONSHINE_ROOT / "qwen2b_mlx_4bit"

# MiniLM-L6-v2 — local semantic embeddings (384-dim, offline)
MINILM_LOCAL   = MOONSHINE_ROOT / "all-MiniLM-L6-v2"

# Real-ESRGAN — image upscaling (fallback: ffmpeg)
ESRGAN_MODEL   = MODELS_ROOT / "realesrgan"
ESRGAN_WEIGHTS = ESRGAN_MODEL / "weights" / "RealESRGAN_x4plus.pth"

# faster-whisper — transcription fallback when YouTube subs unavailable
WHISPER_MODEL  = MODELS_ROOT / "whisper-small"

# ── Fetch settings ────────────────────────────────────────────────
LOW_RES_FORMAT  = "bestvideo[height<=360]+bestaudio/best[height<=360]"
# v3: HQ segment download removed entirely — frames extracted from 360p

# ── Multimodal extraction settings ───────────────────────────────
SAMPLE_RATE          = 16_000   # Hz – audio resampling
FRAME_DIFF_THRESHOLD = 0.15    # frame-diff score to flag a high-info moment
SCENE_THRESHOLD      = 0.22    # ffmpeg scene detection sensitivity (0-1)
MAX_SCENE_FRAMES     = 80      # max unique frames to keep after dedup (v3.1: was 12)
PHASH_DEDUP_DIST     = 8       # hamming distance threshold for pHash dedup (tighter → more unique frames)
FRAMES_PER_CHAPTER   = 4       # auto-inject this many unique frames into each chapter (v3.1)
FRAMES_PER_CHAPTER_MIN = 2     # but never fewer than this when frames available

# ── KV-cache / TurboQuant settings ───────────────────────────────
KV_QUANTIZE_BITS = 4       # quantization bits for key/value cache compression

# ── Evaluation settings ───────────────────────────────────────────
RAGAS_THRESHOLD    = 0.70  # chapters below this score re-enter the writer loop
RAGAS_MAX_RETRIES  = 3     # max writer re-drafts per chapter before forced pass

# ── Generation token budget ───────────────────────────────────────
TOKEN_BUDGET = {
    "step0": {"max_in": 50,   "max_out": 50},    # fetch
    "step1": {"max_in": 500,  "max_out": 500},   # multimodal extract
    "step2": {"max_in": 600,  "max_out": 600},   # pedagogical / ToC (up for MLX speed)
    "step3": {"max_in": 2000, "max_out": 2000},  # writer (batched — one call per video)
    "step4": {"max_in": 200,  "max_out": 200},   # citation
    "step5": {"max_in": 400,  "max_out": 400},   # evaluation
    "step6": {"max_in": 100,  "max_out": 100},   # publish
}

