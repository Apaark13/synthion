"""agents/02_vectorise.py

Hardened vectorise agent (Batch 3).
- Embeds transcript files with SentenceTransformer (MiniLM).
- Upserts embeddings into ChromaDB PersistentClient.
- Uses exponential_backoff, logs errors, and cleans up temp files.
"""
from pathlib import Path
from datetime import datetime, timezone
import json

import chromadb
from sentence_transformers import SentenceTransformer

from config.settings import TRANSCRIPT_DIR, VECTOR_DIR, EMBED_MODEL, COLLECTION_NAME, ERRORS_LOG
from agents.template import exponential_backoff, log_error, cleanup_temp


def _update_checkpoint(task_name: str):
    jp = Path(__file__).parent.parent / "checkpoints" / "last_success.json"
    try:
        data = json.loads(jp.read_text())
    except Exception:
        data = {}
    data["step"] = 2
    data.setdefault("completed_tasks", [])
    if task_name not in data["completed_tasks"]:
        data["completed_tasks"].append(task_name)
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    jp.write_text(json.dumps(data, indent=4))


def _append_done(entry: str):
    done = Path(__file__).parent.parent / "DONE.md"
    with open(done, "a") as f:
        f.write("\n" + entry + "\n")


def step_2(transcript_dir: Path = TRANSCRIPT_DIR) -> Path:
    """Embed transcripts and upsert into ChromaDB. Returns VECTOR_DIR."""
    VECTOR_DIR = Path(__file__).parent.parent / ".." / "workspace" / "vector_db"
    VECTOR_DIR = VECTOR_DIR.resolve()
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    def work():
        model = SentenceTransformer(str(EMBED_MODEL))

        # Wrap local SentenceTransformer so Chroma uses it instead of attempting
        # to download an ONNX embedding model over the network.
        class LocalEmbedding:
            def __init__(self, model):
                self.model = model
            def name(self):
                return "local_sentence_transformers"

            def embed_documents(self, texts):
                # return list[list[float]]; accept numpy arrays or plain lists
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
            # Try to create/get collection with our local embedding function.
            collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=LocalEmbedding(model))
        except ValueError:
            # A persisted collection exists with a different embedding function
            # — use the existing collection without overriding the embedding_function.
            collection = client.get_or_create_collection(name=COLLECTION_NAME)

            ids, embeddings, docs = [], [], []
            for txt in Path(transcript_dir).glob("*.txt"):
                text = txt.read_text(encoding="utf-8")
                emb = model.encode(text).tolist()
                ids.append(txt.stem)
                embeddings.append(emb)
                docs.append(text)

            if ids:
                collection.add(ids=ids, embeddings=embeddings, documents=docs)

        finally:
            # Persist client if supported
            if hasattr(client, "persist"):
                try:
                    client.persist()
                except Exception:
                    pass
        # free any GPU memory
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return VECTOR_DIR

    try:
        out = exponential_backoff(lambda: work(), step=2)
        _update_checkpoint("agents/02_vectorise.py implemented")
        _append_done(f"[v6] {datetime.now(timezone.utc).date()} · Copilot · agents/02_vectorise.py implemented")
        cleanup_temp()
        return out
    except Exception as e:
        log_error(2, e)
        raise
