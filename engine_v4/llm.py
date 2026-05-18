"""engine_v4.llm — single shared model client with a global lock."""
from __future__ import annotations

import threading

from config.settings import GEMMA_GGUF, QWEN_MLX_MODEL

LLM_LOCK = threading.Lock()

_gemma = None
_mlx_model = None
_mlx_tokenizer = None


def get_gemma():
    global _gemma
    if _gemma is not None:
        return _gemma
    if not GEMMA_GGUF.exists():
        return None
    try:
        from llama_cpp import Llama
        _gemma = Llama(
            model_path=str(GEMMA_GGUF),
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False,
        )
    except Exception:
        _gemma = None
    return _gemma


def call_gemma(prompt: str, max_tokens: int = 1500, stop: list[str] | None = None,
               temperature: float = 0.3) -> str:
    llm = get_gemma()
    if llm is None:
        return ""
    with LLM_LOCK:
        try:
            resp = llm(prompt, max_tokens=max_tokens, temperature=temperature,
                       stop=stop or ["<<<CHAPTER_END>>>", "---END---"])
            return (resp["choices"][0]["text"] or "").strip()
        except Exception:
            return ""


def get_mlx():
    global _mlx_model, _mlx_tokenizer
    if _mlx_model is not None:
        return _mlx_model, _mlx_tokenizer
    if not QWEN_MLX_MODEL.exists():
        return None, None
    try:
        import mlx_lm
        _mlx_model, _mlx_tokenizer = mlx_lm.load(str(QWEN_MLX_MODEL))
    except Exception:
        _mlx_model, _mlx_tokenizer = None, None
    return _mlx_model, _mlx_tokenizer


def call_mlx(prompt: str, max_tokens: int = 600) -> str:
    model, tok = get_mlx()
    if model is None:
        return ""
    with LLM_LOCK:
        try:
            import mlx_lm, re
            out = mlx_lm.generate(model, tok,
                                  prompt=f"/no_think\n{prompt}",
                                  max_tokens=max_tokens, verbose=False)
            return re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
        except Exception:
            return ""
