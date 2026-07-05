"""CPU embeddings via fastembed (ONNX) — the GPU stays free for vLLM.

If fastembed is unavailable or the model fails to load, HomeGuide degrades
gracefully to keyword-only (FTS5) search.
"""

import logging
import threading

import numpy as np

log = logging.getLogger("homeguide")

MODEL_NAME = "BAAI/bge-small-en-v1.5"  # 384-dim, ~130 MB, strong for its size

_model = None
_model_lock = threading.Lock()
_load_failed = False


def get_model():
    """Load the embedding model once; return None if it can't be loaded."""
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    with _model_lock:
        if _model is not None or _load_failed:
            return _model
        try:
            from fastembed import TextEmbedding

            log.info("Loading embedding model %s (first run downloads ~130 MB)...", MODEL_NAME)
            _model = TextEmbedding(MODEL_NAME)
            log.info("Embedding model ready.")
        except Exception:
            log.exception("Embedding model unavailable — falling back to keyword-only search.")
            _load_failed = True
    return _model


def available() -> bool:
    return get_model() is not None


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_passages(texts: list[str]) -> np.ndarray | None:
    model = get_model()
    if model is None:
        return None
    vecs = np.array(list(model.passage_embed(texts)), dtype=np.float32)
    return _normalize(vecs)


def embed_query(text: str) -> np.ndarray | None:
    model = get_model()
    if model is None:
        return None
    vec = np.array(list(model.query_embed(text)), dtype=np.float32)
    return _normalize(vec)[0]
