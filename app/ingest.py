"""Document ingestion: extract text (PDF via PyMuPDF, or plain text/markdown),
chunk it with page numbers preserved, embed, and index.
"""

import logging
import re
from collections import Counter
from pathlib import Path

from . import db, embeddings, search

log = logging.getLogger("homeguide")

CHUNK_TARGET = 900   # chars per chunk we aim for
CHUNK_OVERLAP = 220  # trailing lines carried into the next chunk — keeps a table
                     # row's label and its values together across a split

TEXT_SUFFIXES = {".txt", ".md"}


def extract_pages(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (page_number, text, table_rows) triples, 1-indexed."""
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(1, text, [])] if text.strip() else []

    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append((i, text, _extract_table_rows(page)))
    return pages


def _extract_table_rows(page) -> list[str]:
    """Detected table rows as one 'cell | cell | ...' line each. Cooking charts
    et al. get indexed a second time this way: a whole row in one line ranks far
    better for 'chicken breast cooking time' than the same cells scattered
    across flowing text, and can never be split apart by chunking."""
    rows: list[str] = []
    try:
        tables = page.find_tables(strategy="lines").tables
    except Exception:
        return rows
    for table in tables:
        extracted = table.extract()
        # Only genuine tables: prose misdetected as a table produces 1-cell
        # "rows" that would duplicate the page text and crowd out real hits.
        multi_cell = sum(
            1 for row in extracted if sum(1 for c in row if c and c.strip()) >= 3
        )
        if multi_cell < 3:
            continue
        for row in extracted:
            # A row collapsed into one cell is still a complete, ordered row
            cells = [" ".join(c.split()) for c in row if c and c.strip()]
            line = " | ".join(cells)
            if cells and len(line) > 12:
                rows.append(line)
    return rows


def strip_repeated_lines(
    pages: list[tuple[int, str, list[str]]],
) -> list[tuple[int, str, list[str]]]:
    """Remove running headers/footers: short lines that repeat across a third
    or more of the pages (brand banners, URLs, page furniture). Left in, they
    pollute every chunk's ranking — a brand line like 'NINJA AIR FRYER' on
    every page makes all pages match 'air fryer' equally well."""
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for _, text, _ in pages:
        counts.update({ln.strip() for ln in text.split("\n") if ln.strip()})
    threshold = max(3, len(pages) // 3)
    boiler = {ln for ln, c in counts.items() if c >= threshold and len(ln) < 100}
    if not boiler:
        return pages
    return [
        (no, "\n".join(ln for ln in text.split("\n") if ln.strip() not in boiler), rows)
        for no, text, rows in pages
    ]


def _clean(text: str) -> str:
    text = text.replace("­", "")               # soft hyphens
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_BOILERPLATE = re.compile(r"^(?:[\w-]+(?:\.[\w-]+)+|[0-9 |/.-]+|page \d+.*)$", re.IGNORECASE)


def _page_header(lines: list[str]) -> str | None:
    """First real heading on the page (skipping URLs/page numbers), e.g.
    'Air Fry Cooking Chart'. Prefixed to every chunk so a table row keeps
    its context and queries like 'cooking time' match chart pages."""
    for line in lines[:8]:
        if _BOILERPLATE.match(line) or len(re.sub(r"[^A-Za-z]", "", line)) < 4:
            continue
        return line[:80]
    return None


def chunk_page(text: str) -> list[str]:
    """Split one page's text into line-aligned chunks with overlap.

    Manuals are full of charts where one table row spans several short lines
    (ingredient / amount / temp / time); splitting on lines with a trailing
    overlap keeps rows intact far better than paragraph splitting, which sees
    a chart page as one giant block and cuts it mid-row.
    """
    text = _clean(text)
    if not text:
        return []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for line in lines:
        if buf and size + len(line) > CHUNK_TARGET:
            chunks.append("\n".join(buf))
            # carry trailing lines into the next chunk
            kept: list[str] = []
            carried = 0
            for prev in reversed(buf):
                if carried + len(prev) > CHUNK_OVERLAP:
                    break
                kept.insert(0, prev)
                carried += len(prev) + 1
            buf, size = kept, carried
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))

    header = _page_header(lines)
    if header:
        chunks = [c if header in c[:300] else f"[{header}]\n{c}" for c in chunks]
    return chunks


def ingest_document(doc_id: int) -> None:
    """Background task: extract, chunk, embed, and index one document."""
    conn = db.connect()
    row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return
    path = db.pdf_path(doc_id, row["filename"])

    try:
        pages = strip_repeated_lines(extract_pages(path))
        page_chunks: list[tuple[int, str]] = []
        for page_no, text, rows in pages:
            page_chunks.extend((page_no, chunk) for chunk in chunk_page(text))
            if rows:
                header = _page_header(
                    [ln.strip() for ln in _clean(text).split("\n") if ln.strip()]
                )
                # One row per chunk: a self-contained "Sausages | ... | 200°C |
                # 10-13 mins" line is so term-dense it reliably outranks prose
                # that merely mentions the ingredient.
                page_chunks.extend(
                    (page_no, f"[{header}] {row}" if header else row) for row in rows
                )
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
