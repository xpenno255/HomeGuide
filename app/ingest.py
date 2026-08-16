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


_ENGLISH_MARKERS = frozenset(
    "the and with for this that your from are you will when then before after "
    "into have any use".split()
)
# Function words common in the other languages EU appliance manuals ship in, and
# rare or absent in English. Words that also exist in English ("die", "van",
# "non", "in", "no", "on", "man", "do", "care", "sue") are deliberately
# excluded — a false positive here deletes a real English page.
_FOREIGN_MARKERS = frozenset(
    "der und mit nicht oder das für ein sie auf "                    # de
    "les des avec pour dans une est vous sur être "                  # fr
    "los las con para del que por como este más "                    # es
    "della sono che gli alla nel questo può "                        # it
    "het een voor niet zijn aan kan worden uw "                      # nl
    "com não uma dos seu "                                           # pt
    "się nie lub jest aby że przez przed jeśli oraz może należy "    # pl
    "není nebo jsou které při také musí "                            # cs
    "ikke eller som skal til det "                                   # da/no
    "inte till för "                                                 # sv
    "että tai sekä kun jos ovat "                                    # fi
    "sau este pentru când "                                          # ro
    "vagy hogy egy ezt".split()                                      # hu
)

# Scripts other than Latin never produce English text. Counted separately
# because the word scoring above only sees Latin letters, so a Greek or
# Cyrillic page scores zero on every marker and would be kept by default.
_LATIN = re.compile(r"[A-Za-zÀ-ÿ]")
_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


def _is_probably_english(text: str) -> bool:
    """Keep a page unless it is confidently in another language.

    EU appliance manuals ship every language in one PDF — the Sage Precision
    Brewer manual is 156 pages of which 19 are English. Indexing the rest costs
    more than wasted space: the agent can quote Portuguese back at you, and
    `strip_repeated_lines` scales its threshold to the page count, so a document
    padded 8x stops recognising its own running boilerplate.

    Short or wordless pages (diagrams, part-label lists) are kept — dropping a
    real English page is far worse than keeping a foreign label list, and the
    embeddings are English-only anyway (bge-small-en-v1.5).
    """
    alpha = _ALPHA.findall(text)
    if len(alpha) >= 100:
        latin = sum(bool(_LATIN.match(c)) for c in alpha)
        if latin < len(alpha) * 0.5:
            return False

    words = re.findall(r"[a-zà-ÿąćęłńóśźżěščřůőűåøæ]+", text.lower())
    english = sum(w in _ENGLISH_MARKERS for w in words)
    foreign = sum(w in _FOREIGN_MARKERS for w in words)
    return not (foreign > english and foreign >= 3)


def extract_pages(path: Path) -> list[tuple[int, str, list[str]]]:
    """Return (page_number, text, table_rows) triples, 1-indexed.

    Page numbers are the document's real page numbers, so they stay correct for
    citation even when other-language pages are skipped.
    """
    if path.suffix.lower() in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [(1, text, [])] if text.strip() else []

    import pymupdf

    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = _page_text(page)
            if text.strip():
                pages.append((i, text, _extract_table_rows(page)))

    english = [p for p in pages if _is_probably_english(p[1])]
    # A wholly non-English manual is still better indexed than not indexed.
    return english or pages


# A gutter must be at least this fraction of the page width to separate columns.
GUTTER_MIN_RATIO = 0.02
# Blocks wider than this span columns (banners, footers) and cannot define them.
WIDE_BLOCK_RATIO = 0.6


def _column_bounds(blocks, width: float) -> list[float]:
    """x positions of the vertical whitespace gutters separating columns."""
    narrow = [b for b in blocks if (b[2] - b[0]) < WIDE_BLOCK_RATIO * width]
    if len(narrow) < 4:
        return []
    covered = bytearray(int(width) + 2)
    for b in narrow:
        for x in range(max(0, int(b[0])), min(int(width), int(b[2])) + 1):
            covered[x] = 1
    min_gutter = max(8, int(GUTTER_MIN_RATIO * width))
    first, last = int(min(b[0] for b in narrow)), int(max(b[2] for b in narrow))
    bounds: list[float] = []
    run_start: int | None = None
    for x in range(first, last + 1):
        if not covered[x]:
            if run_start is None:
                run_start = x
        else:
            if run_start is not None and x - run_start >= min_gutter:
                bounds.append((run_start + x) / 2)
            run_start = None
    return bounds


def _page_text(page) -> str:
    """Page text in column reading order.

    PyMuPDF's default order follows the PDF's internal draw order, which on a
    multi-column page interleaves the columns. Manuals are printed as two-page
    spreads, so this is not a cosmetic problem: on the AF500UK manual the
    CLEANING & MAINTENANCE heading sits in the left half while the right half
    holds TROUBLESHOOTING GUIDE, and the default order emitted the whole
    troubleshooting column first — separating every cleaning instruction from
    its own heading and labelling it with the wrong one.

    Blocks are grouped into columns by the vertical whitespace between them,
    then read top-to-bottom within each column. A page with no gutter wide
    enough to split on falls back to the default extraction unchanged.
    """
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
    if len(blocks) < 4:
        return page.get_text("text")
    bounds = _column_bounds(blocks, page.rect.width)
    if not bounds:
        return page.get_text("text")

    def order(b):
        column = sum(1 for x in bounds if b[0] >= x)
        return (column, round(b[1]), round(b[0]))

    return "\n".join(b[4].strip() for b in sorted(blocks, key=order))


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
            if not cells:
                continue
            line = " | ".join(cells)
            if len(cells) >= 2:
                if len(line) > 12:
                    rows.append(line)
            # A single-cell row is either a merged data row ("Asparagus Cut in
            # 2.5cm pieces, blanched 60°C 6-8 hours") or a section banner
            # spanning the table ("FRESH MEAT, POULTRY, FISH"). Banners carry no
            # answer but are short and header-prefixed, which scores them
            # absurdly well against short queries — they were landing in the top
            # 3 for "dishwasher safe parts". Real rows quote an amount, a
            # temperature or a time, so require a digit and some substance.
            elif len(line) >= 30 and any(ch.isdigit() for ch in line):
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

        # The appliance name usually appears only in the title the user gave the
        # document, never in the manual's own body text — the Ninja guarantee
        # section never says "air fryer". Both retrievers need it, or a library
        # with two appliances cannot tell whose guarantee is being asked about.
        # The vector side mixes it in rather than prepending it to the text; see
        # embeddings.TITLE_MIX for why the obvious version breaks the charts.
        title = row["title"]
        texts = [c for _, c in page_chunks]
        vectors = embeddings.embed_passages_titled(texts, title)

        with db.lock:
            for i, (page_no, chunk) in enumerate(page_chunks):
                blob = vectors[i].tobytes() if vectors is not None else None
                cur = conn.execute(
                    "INSERT INTO chunks (doc_id, page, text, embedding) VALUES (?, ?, ?, ?)",
                    (doc_id, page_no, chunk, blob),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (rowid, text, title) VALUES (?, ?, ?)",
                    (cur.lastrowid, chunk, title),
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


def _clear_chunks(conn, doc_id: int) -> None:
    conn.execute(
        "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = ?)",
        (doc_id,),
    )
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def reindex_document(doc_id: int) -> bool:
    """Re-chunk and re-embed a document from the file already on disk.

    Chunking and embedding happen at upload time, so a retrieval change only
    reaches existing documents through this — otherwise the only route is
    deleting every document and re-uploading the originals by hand.
    """
    conn = db.connect()
    row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return False
    if not db.pdf_path(doc_id, row["filename"]).exists():
        return False
    with db.lock:
        _clear_chunks(conn, doc_id)
        conn.execute(
            "UPDATE documents SET status = 'processing', chunk_count = 0, error = NULL WHERE id = ?",
            (doc_id,),
        )
        conn.commit()
    search.invalidate_cache()
    ingest_document(doc_id)
    return True


def delete_document(doc_id: int) -> bool:
    conn = db.connect()
    row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        return False
    with db.lock:
        _clear_chunks(conn, doc_id)
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
    db.pdf_path(doc_id, row["filename"]).unlink(missing_ok=True)
    search.invalidate_cache()
    return True
