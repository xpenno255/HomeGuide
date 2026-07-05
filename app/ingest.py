"""Document ingestion: extract text (PDF via PyMuPDF, or plain text/markdown),
chunk it with page numbers preserved, embed, and index.
"""

import logging
import re
from pathlib import Path

from . import db, embeddings, search

log = logging.getLogger("homeguide")

CHUNK_TARGET = 900   # chars per chunk we aim for
CHUNK_MAX = 1400     # hard ceiling before splitting mid-block
CHUNK_OVERLAP = 150  # carried between forced splits so sentences aren't orphaned

TEXT_SUFFIXES = {".txt", ".md"}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return (page_number, text) pairs, 1-indexed."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(1, text)] if text.strip() else []

    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text))
    return pages


def _clean(text: str) -> str:
    text = text.replace("­", "")               # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(text: str) -> list[str]:
    """Split one page's text into chunks, preferring paragraph boundaries."""
    text = _clean(text)
    if not text:
        return []

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for block in blocks:
        if len(current) + len(block) + 2 <= CHUNK_TARGET:
            current = f"{current}\n\n{block}" if current else block
            continue
        flush()
        # Oversized single block: hard-split with overlap
        while len(block) > CHUNK_MAX:
            cut = block.rfind(". ", CHUNK_TARGET - 200, CHUNK_MAX)
            cut = cut + 1 if cut != -1 else CHUNK_MAX
            chunks.append(block[:cut].strip())
            block = block[max(cut - CHUNK_OVERLAP, 0):].strip()
        current = block
    flush()
    return chunks


def ingest_document(doc_id: int) -> None:
    """Background task: extract, chunk, embed, and index one document."""
    conn = db.connect()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return
    path = db.pdf_path(doc_id, row["filename"])

    try:
        pages = extract_pages(path)
        page_chunks = [(page_no, chunk) for page_no, text in pages for chunk in chunk_page(text)]
        if not page_chunks:
            raise ValueError("No extractable text found (scanned/image-only PDF?)")

        texts = [c for _, c in page_chunks]
        vectors = embeddings.embed_passages(texts)  # None => keyword-only mode

        with db.lock:
            for i, (page_no, chunk) in enumerate(page_chunks):
                blob = vectors[i].tobytes() if vectors is not None else None
                cur = conn.execute(
                    "INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)",
                    (doc_id, page_no, chunk, blob),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, text) VALUES (?, ?)",
                    (cur.lastrowid, chunk),
                )
            conn.execute(
                "UPDATE documents SET status = 'ready', pages = ?, chunk_count = ?, error = NULL WHERE id = ?",
                (len(pages), len(page_chunks), doc_id),
            )
            conn.commit()
        search.invalidate_cache()
        log.info("Indexed doc %s: %s pages, %s chunks.", doc_id, len(pages), len(page_chunks))
    except Exception as exc:
        log.exception("Ingestion failed for doc %s", doc_id)
        with db.lock:
            conn.execute(
                "UPDATE documents SET status = 'error', error = ? WHERE id = ?",
                (str(exc), doc_id),
            )
            conn.commit()


def delete_document(doc_id: int) -> bool:
    conn = db.connect()
    row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return False
    with db.lock:
        conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    db.pdf_path(doc_id, row["filename"]).unlink(missing_ok=True)
    search.invalidate_cache()
    return True
