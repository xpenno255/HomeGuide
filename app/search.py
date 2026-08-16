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

# Cosine floor. Measured against the Ninja AF500UK guide: chunks that actually
# answer the query score 0.78-0.82, while queries the library cannot answer top
# out at 0.53 ("what does E4 mean") to 0.68 ("descaling"). At the old 0.60 the
# agent was handed dehydrator chart rows for descaling questions and read them
# out. Anything that shares heavy vocabulary with the corpus ("how do I clean
# the air fryer" -> 0.75) still gets through; that is a corpus-coverage problem,
# not a threshold one.
MIN_SIM = 0.70

# Voice queries are full sentences; without stopword removal BM25's OR-query
# rewards chunks that merely repeat "the"/"how"/"in".
STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from has have how i in is it its "
    "me my of on or our s should t that the their there this to was we what when "
    "where which will with would you your".split()
)

# A token matching more than this share of the library carries no signal: in a
# single air-fryer manual "cooking" hits 77% of chunks and "air" 63%, so an
# OR-query containing them ranks chunks by how often they repeat the appliance's
# own vocabulary. The generic STOPWORDS list above cannot know this — the noise
# words are whatever the library happens to be about.
DF_MAX_RATIO = 0.5

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


def _selective_tokens(tokens: list[str]) -> list[str]:
    """Drop query tokens that match more than DF_MAX_RATIO of the library.

    Returns [] when nothing distinctive is left — a query made entirely of the
    corpus's own filler words should contribute no keyword ranking at all
    rather than an arbitrary one.
    """
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    if not total:
        return []
    ceiling = total * DF_MAX_RATIO
    keep = []
    for t in tokens:
        df = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", (f'"{t}"',)
        ).fetchone()["n"]
        if 0 < df <= ceiling:
            keep.append(t)
    return keep


def _fts_ranked(query: str, allowed: set[int] | None) -> tuple[list[int], set[int]]:
    """Return (ranked chunk ids, ids trustworthy without vector corroboration).

    BM25's real value over embeddings is exact identifiers — a fault code like
    "E4" is a token the vector model has no useful representation of. For plain
    English words the vector retriever is the better-calibrated of the two, so a
    chunk that matched only one ordinary word is weak evidence: "cleaning and
    care" matching the word "Cleaned" inside a mushroom dehydration row is a
    lexical coincidence, not an answer. Such hits are returned only if the
    vector side found them too.
    """
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query) if t.lower() not in STOPWORDS]
    tokens = _selective_tokens(tokens)
    if not tokens:
        return [], set()
    conn = db.connect()
    sql = (
        "SELECT c.id FROM chunks_fts f "
        "JOIN chunks c ON c.id = f.rowid "
        "JOIN documents d ON d.id = c.doc_id "
        "WHERE chunks_fts MATCH ? AND d.status = 'ready' "
    )
    params: list = [" OR ".join(f'"{t}"' for t in tokens)]
    # The category filter belongs in SQL: applied after LIMIT it would silently
    # shrink a filtered search to a handful of candidates.
    if allowed is not None:
        if not allowed:
            return [], set()
        sql += f"AND c.doc_id IN ({','.join('?' * len(allowed))}) "
        params.extend(allowed)
    sql += "ORDER BY bm25(chunks_fts) LIMIT ?"
    params.append(CANDIDATES)
    ranked = [r["id"] for r in conn.execute(sql, params).fetchall()]
    if not ranked:
        return [], set()

    # How many query tokens each candidate actually matched, and whether any of
    # them was identifier-shaped (contains a digit: "E4", "F21", "AF500").
    placeholders = ",".join("?" * len(ranked))
    hits: dict[int, int] = {}
    identifier: set[int] = set()
    for tok in tokens:
        rows = conn.execute(
            f"SELECT rowid AS id FROM chunks_fts WHERE chunks_fts MATCH ? AND rowid IN ({placeholders})",
            [f'"{tok}"', *ranked],
        ).fetchall()
        has_digit = any(ch.isdigit() for ch in tok)
        for r in rows:
            hits[r["id"]] = hits.get(r["id"], 0) + 1
            if has_digit:
                identifier.add(r["id"])

    trusted = {cid for cid in ranked if hits.get(cid, 0) >= 2 or cid in identifier}
    return ranked, trusted


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
    fts_ids, fts_trusted = _fts_ranked(query, allowed)
    vec_ids = _vector_ranked(query, allowed)

    # Keyword-only mode (no embedding model) has no second opinion to consult,
    # so every keyword hit stands on its own there.
    if vec_ids or embeddings.available():
        corroborated = set(vec_ids)
        fts_ids = [cid for cid in fts_ids if cid in fts_trusted or cid in corroborated]

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
