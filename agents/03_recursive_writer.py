"""agents/03_recursive_writer.py

Hardened recursive writer (Batch 4).
- Reads batched 5-minute transcripts (TRANSCRIPT_DIR/*_batch_*.txt) produced by step_1.
- Retrieves top-K chunks from ChromaDB using local SentenceTransformer embeddings.
- Generates chapter Markdown via local mlx_lm Qwen 4-bit model (QWEN_FILE).
- Respects TOKEN_BUDGET["step3"] for generation guidance.
- Falls back to placeholder output if mlx_lm or the model dir are unavailable.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

import chromadb

from config.settings import (
    VECTOR_DIR, BOOK_DIR, QWEN_FILE, RAG_TOP_K, TOKEN_BUDGET,
    COLLECTION_NAME, ERRORS_LOG, EMBED_MODEL, TRANSCRIPT_DIR,
)
from agents.template import exponential_backoff, log_error, cleanup_temp

# ── MLX Qwen: load once at module level so we don't reload per chapter ────────
_qwen_model = None
_qwen_tokenizer = None

def _load_qwen():
    """Load the MLX Qwen model and tokenizer once; cache in module globals."""
    global _qwen_model, _qwen_tokenizer
    if _qwen_model is None:
        try:
            from mlx_lm import load
            _qwen_model, _qwen_tokenizer = load(str(QWEN_FILE))
        except Exception as e:
            log_error(3, e)
    return _qwen_model, _qwen_tokenizer


def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["step"] = 3
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


def _retrieve_contexts(texts, top_k: int = RAG_TOP_K):
    # Use the local SentenceTransformer for embedding queries so Chroma does
    # not attempt to download an ONNX model at runtime.
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(EMBED_MODEL))

        class LocalEmbedding:
            def __init__(self, model):
                self.model = model
            def name(self):
                return "local_sentence_transformers"

            def embed_documents(self, texts):
                res = self.model.encode(list(texts))
                out = []
                for emb in res:
                    if hasattr(emb, 'tolist'):
                        out.append(emb.tolist())
                    else:
                        out.append(list(emb))
                return out

            def embed_query(self, text):
                res = self.model.encode([text])
                emb = res[0]
                return emb.tolist() if hasattr(emb, 'tolist') else list(emb)

        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        try:
            coll = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=LocalEmbedding(model))
        except ValueError:
            coll = client.get_or_create_collection(name=COLLECTION_NAME)
        results = coll.query(query_texts=texts, n_results=top_k)
    except Exception as e:
        # Fall back to a simple query; log and return empty contexts on failure
        log_error(3, e)
        return ["" for _ in texts]
    contexts = []
    for docs in results.get("documents", []):
        if docs:
            contexts.append("\n".join(docs))
        else:
            contexts.append("")
    return contexts


def _call_qwen(prompt: str, max_tokens: int) -> str:
    """Generate text using the local MLX Qwen 4-bit model.

    Falls back to a short placeholder if mlx_lm is not installed or the
    model directory is missing.
    """
    model, tokenizer = _load_qwen()
    if model is not None and tokenizer is not None:
        try:
            from mlx_lm import generate
            text = generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
            )
            return text if isinstance(text, str) else str(text)
        except Exception as e:
            log_error(3, e)

    # Fallback: return a clearly-labelled draft so the pipeline doesn't break
    max_out = TOKEN_BUDGET.get("step3", {}).get("max_out", 800)
    return "# CHAPTER DRAFT (mlx_lm fallback)\n\n" + prompt[: max_out * 4]


def step_3(chunks: list = None, glossary: str = "") -> Path:
    """Generate chapter Markdown files from batched 5-min transcripts.

    If ``chunks`` is None or empty, reads all *_batch_*.txt files from
    TRANSCRIPT_DIR (produced by step_1) sorted alphabetically.
    Returns BOOK_DIR on success.
    """
    BOOK_DIR.mkdir(parents=True, exist_ok=True)

    def work():
        # ── resolve chunks from batch transcript files if not provided ────────
        if not chunks:
            batch_files = sorted(TRANSCRIPT_DIR.glob("*_batch_*.txt"))
            if not batch_files:
                # fall back to any _raw.txt files if no batch files exist
                batch_files = sorted(TRANSCRIPT_DIR.glob("*_raw.txt"))
            merged = [p.read_text(encoding="utf-8") for p in batch_files]
        else:
            merged = list(chunks)

        if not merged:
            merged = ["[empty transcript]"]

        contexts = _retrieve_contexts(merged, top_k=RAG_TOP_K)

        max_tok = TOKEN_BUDGET.get("step3", {}).get("max_out", 800)
        max_in  = TOKEN_BUDGET.get("step3", {}).get("max_in",  1200)

        for i, chunk in enumerate(merged):
            rag = contexts[i] if i < len(contexts) else ""
            # truncate raw transcript to max_in chars (proxy for token budget)

            print(len(chunk),i+1)
            print(f"Context retrieved for chunk {i+1}: {len(rag)} chars", rag)
            prompt = (
                f"Glossary:\n{glossary}\n\n"
                f"Context from related lectures:\n'''{rag}\n'''\n\n"
                "Write comprehensive, well-structured Markdown lecture notes "
                "with headings, bullet points, and key definitions."
                f"This whole operation is divided into {len(merged)} segments, and this is segment {i+1}. so CONTINUE this chapter based on the provided transcript and context."
                f"Transcript (5-minute segment):\n'''{chunk}\n'''\n\n"
            )
            out_text = _call_qwen(prompt, max_tokens=600)
            print(f"[writer] chapter_{i+1:03d}: {len(out_text)} chars")
            out_file = BOOK_DIR / f"chapter_{i+1:03d}.md"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(out_text)

        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return BOOK_DIR

    try:
        out = exponential_backoff(lambda: work(), step=3)
        _update_checkpoint("agents/03_recursive_writer.py implemented")
        _append_done(
            f"[v8] {datetime.now(timezone.utc).date()} · Copilot · "
            "03_recursive_writer: mlx_lm Qwen + batch-transcript feed"
        )
        cleanup_temp()
        return out
    except Exception as e:
        log_error(3, e)
        raise
