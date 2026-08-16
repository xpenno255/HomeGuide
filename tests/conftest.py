"""Shared fixtures.

Two things make this app awkward to test, and both are handled here:

1. Module-level state — `db` holds one shared connection and reads its paths
   from module globals, `search` caches every embedding in a module global.
   Each test gets a fresh temp DATA_DIR and a reset of both.
2. Embeddings — the real model is a 130 MB download and its scores are only
   meaningful against real documents. Mechanics (the similarity floor, the
   corroboration rule, RRF fusion) are tested against a stub with *exactly*
   controllable cosine scores; whether the real model actually separates good
   answers from bad ones is a corpus question, tested in
   test_retrieval_quality.py against the real manual.
"""

import numpy as np
import pytest

from app import db, embeddings, search

DIM = 8


def _unit(cosine: float) -> np.ndarray:
    """A unit vector whose dot product with the query vector is `cosine`."""
    cosine = max(-1.0, min(1.0, cosine))
    v = np.zeros(DIM, dtype=np.float32)
    v[0] = cosine
    v[1] = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
    return v


QUERY_VEC = _unit(1.0)


class StubEmbeddings:
    """Assigns a chosen cosine-to-query to any passage containing a marker.

    Register before ingesting, since embeddings are computed at upload time:
        stub.set("Sausages | 8", 0.82)
    """

    def __init__(self):
        self.scores: dict[str, float] = {}
        self.default = 0.0

    def set(self, marker: str, cosine: float) -> None:
        self.scores[marker] = cosine

    def _score(self, text: str) -> float:
        lowered = text.lower()
        for marker, cosine in self.scores.items():
            if marker.lower() in lowered:
                return cosine
        return self.default

    def embed_passages(self, texts):
        return np.vstack([_unit(self._score(t)) for t in texts]).astype(np.float32)

    def embed_query(self, text):
        return QUERY_VEC.copy()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Fresh database and file store, with all module state reset."""
    monkeypatch.setattr(db, "DATA_DIR", tmp_path)
    monkeypatch.setattr(db, "PDF_DIR", tmp_path / "pdfs")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "homeguide.db")
    monkeypatch.setattr(db, "_conn", None)
    search.invalidate_cache()
    yield tmp_path
    conn = getattr(db, "_conn", None)
    if conn is not None:
        conn.close()
    monkeypatch.setattr(db, "_conn", None)
    search.invalidate_cache()


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Semantic search on, with fully controlled similarity scores."""
    stub = StubEmbeddings()
    monkeypatch.setattr(embeddings, "embed_passages", stub.embed_passages)
    monkeypatch.setattr(embeddings, "embed_query", stub.embed_query)
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "get_model", lambda: stub)
    return stub


@pytest.fixture
def no_embeddings(monkeypatch):
    """Keyword-only mode: the model failed to load."""
    monkeypatch.setattr(embeddings, "embed_passages", lambda texts: None)
    monkeypatch.setattr(embeddings, "embed_query", lambda text: None)
    monkeypatch.setattr(embeddings, "available", lambda: False)
    monkeypatch.setattr(embeddings, "get_model", lambda: None)


@pytest.fixture
def add_doc(data_dir):
    """Insert a document from literal text and index it. Returns its id."""
    from app import ingest

    def _add(text: str, title: str = "Test manual", category: str = "manual") -> int:
        conn = db.connect()
        with db.lock:
            cur = conn.execute(
                "INSERT INTO documents (title, category, filename) VALUES (?, ?, ?)",
                (title, category, "doc.md"),
            )
            conn.commit()
        doc_id = cur.lastrowid
        db.pdf_path(doc_id, "doc.md").write_text(text, encoding="utf-8")
        ingest.ingest_document(doc_id)
        return doc_id

    return _add


@pytest.fixture
def client(data_dir, stub_embeddings):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as c:
        yield c
