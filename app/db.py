"""SQLite storage: documents, chunks, and an FTS5 index over chunk text.

A single shared connection guarded by `lock` is enough at homelab scale;
WAL mode keeps reads from blocking during ingestion.
"""

import logging
import os
import sqlite3
import threading
from pathlib import Path

log = logging.getLogger("homeguide")

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))
PDF_DIR = DATA_DIR / "pdfs"
DB_PATH = DATA_DIR / "homeguide.db"

lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    category    TEXT NOT NULL DEFAULT 'manual',
    filename    TEXT NOT NULL,
    pages       INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'processing',
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id    INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page      INTEGER NOT NULL,
    text      TEXT NOT NULL,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- porter stemming so voice phrasing matches manual phrasing (cook/cooking, descale/descaling).
-- `title` carries the document title into every chunk: the appliance name is
-- usually only in the title ("Ninja Air Fryer User Manual"), never in the body,
-- so without it "air fryer guarantee" cannot be steered to the right manual.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, title, tokenize='porter unicode61');
"""


def _migrate_fts_title(conn: sqlite3.Connection) -> None:
    """Add the `title` column to an index built before it existed.

    Rebuilt from the chunks already in the database, so no PDF is re-parsed and
    keyword search keeps working immediately. Embeddings are not rebuilt here —
    they only pick the title up on reindex.
    """
    columns = {r["name"] for r in conn.execute("PRAGMA table_info(chunks_fts)")}
    if not columns or "title" in columns:
        return
    log.info("Migrating chunks_fts to include document titles...")
    conn.executescript(
        "DROP TABLE chunks_fts;\n"
        "CREATE VIRTUAL TABLE chunks_fts USING fts5(text, title, tokenize='porter unicode61');"
    )
    conn.execute(
        "INSERT INTO chunks_fts (rowid, text, title) "
        "SELECT c.id, c.text, d.title FROM chunks c JOIN documents d ON d.id = c.doc_id"
    )
    conn.commit()
    log.info("Migration complete. Reindex documents to add titles to embeddings too.")


def connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA)
        _migrate_fts_title(conn)
        _conn = conn
    return _conn


def pdf_path(doc_id: int, filename: str) -> Path:
    suffix = Path(filename).suffix.lower() or ".pdf"
    return PDF_DIR / f"{doc_id}{suffix}"
