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


# How much of the document title to mix into each chunk vector. The appliance
# name usually exists only in the title, so without this "air fryer guarantee"
# cannot be steered to the right manual — BM25 alone cannot fix it, because RRF
# lets the vector ranking decide. Prepending the title to the passage *text*
# also works but distorts the body: at full strength "sausages cooking time"
# started returning a Toad in the Hole recipe instead of the cooking chart row,
# because every recipe chunk began "Ninja Air Fryer Quick Start Recipe Guide".
# Measured across both failure modes: 0.15 fixes the routing while leaving chart
# rows at rank 1, and 0.3 is already enough to lose them again.
TITLE_MIX = 0.15


def embed_passages_titled(texts: list[str], title: str) -> np.ndarray | None:
    """Passage vectors nudged toward their document's title."""
    vecs = embed_passages(texts)
    if vecs is None or not title.strip():
        return vecs
    title_vec = embed_passages([title])
    if title_vec is None:
        return vecs
    return _normalize(vecs + TITLE_MIX * title_vec[0])


def embed_query(text: str) -> np.ndarray | None:
    model = get_model()
    if model is None:
        return None
    vec = np.array(list(model.query_embed(text)), dtype=np.float32)
    return _normalize(vec)[0]
