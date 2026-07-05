"""Hybrid retrieval: FTS5 BM25 (exact terms like fault codes) + semantic
vector search, fused with reciprocal rank fusion (RRF).

Embeddings for all ready chunks are cached in memory as one numpy matrix;
at homelab scale (tens of manuals) this is a few MB.
"""

import logging
import re
import threading

import numpy as np

from . import db, embeddings

log = logging.getLogger("homeguide")

RRF_K = 60          # standard RRF damping constant
CANDIDATES = 24     # how many candidates each retriever contributes
EXCERPT_MAX = 800   # chars per excerpt returned to the agent
MIN_SIM = 0.60      # cosine floor — bge-small scores ~0.75+ for relevant, <0.55 for unrelated

# Voice queries are full sentences; without stopword removal BM25's OR-query
# rewards chunks that merely repeat "the"/"how"/"in".
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i in is it its "
    "me my of on or our s should t that the their there this to was we what when "
    "where which will with would you your".split()
)

_cache_lock = threading.Lock()
_cache: dict | None = None  # {"ids": np.ndarray, "matrix": np.ndarray, "doc_ids": np.ndarray}


def invalidate_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def _vector_cache() -> dict | None:
    global _cache
    with _cache_lock:
        if _cache is not None:
            return _cache
        conn = db.connect()
        rows = conn.execute(
            "SELECT c.id, c.doc_id, c.embedding FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id "
            "WHERE d.status = 'ready' AND c.embedding IS NOT NULL"
        ).fetchall()
        if not rows:
            return None
        _cache = {
            "ids": np.array([r["id"] for r in rows], dtype=np.int64),
            "doc_ids": np.array([r["doc_id"] for r in rows], dtype=np.int64),
            "matrix": np.vstack(
                [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
            ),
        }
        return _cache


def _allowed_doc_ids(category: str | None) -> set[int] | None:
    """None means no filter."""
    if not category:
        return None
    conn = db.connect()
    rows = conn.execute(
        "SELECT id FROM documents WHERE status = 'ready' AND lower(category) = lower(?)",
        (category,),
    ).fetchall()
    return {r["id"] for r in rows}


def _fts_ranked(query: str, allowed: set[int] | None) -> list[int]:
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query) if t.lower() not in STOPWORDS]
    if not tokens:
        return []
    match = " OR ".join(f'"{t}"' for t in tokens)
    conn = db.connect()
    rows = conn.execute(
        "SELECT c.id, c.doc_id FROM chunks_fts f "
        "JOIN chunks c ON c.id = f.rowid "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE chunks_fts MATCH ? AND d.status = 'ready' "
        "ORDER BY bm25(chunks_fts) LIMIT ?",
        (match, CANDIDATES),
    ).fetchall()
    return [r["id"] for r in rows if allowed is None or r["doc_id"] in allowed]


def _vector_ranked(query: str, allowed: set[int] | None) -> list[int]:
    cache = _vector_cache()
    if cache is None:
        return []
    qvec = embeddings.embed_query(query)
    if qvec is None:
        return []
    scores = cache["matrix"] @ qvec
    if allowed is not None:
        mask = np.isin(cache["doc_ids"], list(allowed))
        scores = np.where(mask, scores, -np.inf)
    order = np.argsort(-scores)[:CANDIDATES]
    return [int(cache["ids"][i]) for i in order if scores[i] >= MIN_SIM]


def hybrid_search(query: str, k: int = 4, category: str | None = None) -> list[dict]:
    allowed = _allowed_doc_ids(category)
    fts_ids = _fts_ranked(query, allowed)
    vec_ids = _vector_ranked(query, allowed)

    # Vector list first: on tied RRF scores the semantic ranking (calibrated
    # by MIN_SIM) should beat the noisier BM25 OR-query.
    fused: dict[int, float] = {}
    for ranked in (vec_ids, fts_ids):
        for rank, chunk_id in enumerate(ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    if not top:
        return []

    conn = db.connect()
    placeholders = ",".join("?" * len(top))
    rows = conn.execute(
        f"SELECT c.id, c.page, c.text, d.title, d.category FROM chunks c "
        f"JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({placeholders})",
        top,
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    results = []
    for chunk_id in top:
        r = by_id.get(chunk_id)
        if r is None:
            continue
        text = r["text"]
        if len(text) > EXCERPT_MAX:
            text = text[:EXCERPT_MAX].rsplit(" ", 1)[0] + "…"
        results.append(
            {
                "document": r["title"],
                "category": r["category"],
                "page": r["page"],
                "excerpt": text,
            }
        )
    return results
