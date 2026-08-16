"""Ingestion: chunking, page furniture, and table-row extraction.

Every case here comes from a real failure against manufacturer manuals, which
are dominated by tables and repeated page banners rather than prose.
"""

import pymupdf
import pytest

from app import db, ingest


class TestChunkPage:
    def test_short_page_is_one_chunk(self):
        assert ingest.chunk_page("Descaling\nRun a cycle with citric acid.") == [
            "Descaling\nRun a cycle with citric acid."
        ]

    def test_blank_page_yields_nothing(self):
        assert ingest.chunk_page("   \n\n  ") == []

    def test_long_page_splits_near_target(self):
        text = "\n".join(f"Step {i}: turn the dial to position {i}." for i in range(80))
        chunks = ingest.chunk_page(text)
        assert len(chunks) > 1
        # Overlap means chunks can exceed the target slightly, never wildly
        assert all(len(c) < ingest.CHUNK_TARGET * 1.6 for c in chunks)

    def test_split_chunks_overlap(self):
        """A table row split across a boundary must survive in one of the pieces."""
        text = "\n".join(f"Ingredient {i} | 200g | 180C | {i} mins" for i in range(60))
        chunks = ingest.chunk_page(text)
        assert len(chunks) > 1
        tail_of_first = chunks[0].split("\n")[-1]
        assert tail_of_first in chunks[1]

    def test_heading_prefixed_onto_later_chunks(self):
        """Chunks after the first lose the page heading, so a chart row split
        onto chunk 2 would otherwise have no idea it is a cooking chart."""
        rows = "\n".join(f"Ingredient {i} | 200g | 180C | {i} mins" for i in range(60))
        chunks = ingest.chunk_page(f"Air Fry Cooking Chart\n{rows}")
        assert len(chunks) > 1
        assert all("Air Fry Cooking Chart" in c for c in chunks)

    def test_heading_not_duplicated_when_already_present(self):
        """The first chunk already opens with the heading; prefixing it again
        would just dilute the chunk."""
        chunks = ingest.chunk_page("Air Fry Cooking Chart\nSausages | 200C")
        assert chunks[0].count("Air Fry Cooking Chart") == 1

    def test_urls_and_page_numbers_rejected_as_headings(self):
        chunks = ingest.chunk_page("ninjakitchen.co.uk\n34\nSausages | 200C | 10 mins")
        assert not chunks[0].startswith("[ninjakitchen")
        assert not chunks[0].startswith("[34]")


class TestStripRepeatedLines:
    def _pages(self, texts):
        return [(i, t, []) for i, t in enumerate(texts, start=1)]

    def test_banner_on_every_page_is_removed(self):
        pages = self._pages(["NINJA AIR FRYER\nContent %d" % i for i in range(6)])
        stripped = ingest.strip_repeated_lines(pages)
        assert all("NINJA AIR FRYER" not in text for _, text, _ in stripped)
        assert "Content 0" in stripped[0][1]

    def test_unique_lines_survive(self):
        pages = self._pages(["Banner\nUnique line %d" % i for i in range(6)])
        stripped = ingest.strip_repeated_lines(pages)
        assert all(f"Unique line {i}" in stripped[i][1] for i in range(6))

    def test_short_documents_untouched(self):
        """Under 4 pages, a repeated line is more likely real content."""
        pages = self._pages(["Repeated\nA", "Repeated\nB"])
        assert ingest.strip_repeated_lines(pages) == pages

    def test_long_repeated_line_kept(self):
        """Only short lines are page furniture; a repeated paragraph is content."""
        long_line = "This appliance must be earthed and connected " + "x" * 80
        pages = self._pages([f"{long_line}\nPage {i}" for i in range(6)])
        stripped = ingest.strip_repeated_lines(pages)
        assert long_line in stripped[0][1]


def _table_page(rows):
    """Render rows as a real PDF table so find_tables() can detect it.

    A row given as a single-element list spans the full width, which is how
    section banners and PyMuPDF-merged data rows actually appear.
    """
    doc = pymupdf.open()
    page = doc.new_page()
    x0, y0, width, height = 40, 50, 120, 20
    ncols = max(len(r) for r in rows)
    for r, row in enumerate(rows):
        span = ncols if len(row) == 1 else 1
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x0 + c * width,
                y0 + r * height,
                x0 + (c + span) * width,
                y0 + (r + 1) * height,
            )
            page.insert_textbox(rect, str(cell), fontsize=7)
            page.draw_rect(rect, color=(0, 0, 0), width=0.5)
    return doc, page


class TestTableRows:
    def test_multi_cell_rows_indexed_individually(self):
        rows = [
            ["Food", "Amount", "Temp", "Time"],
            ["Sausages", "8 (410g)", "200C", "10-13 mins"],
            ["Chicken wings", "1kg", "200C", "30-32 mins"],
            ["Lamb chops", "4 (340g)", "180C", "11-12 mins"],
        ]
        doc, page = _table_page(rows)
        extracted = ingest._extract_table_rows(page)
        doc.close()
        assert any("Sausages" in r and "10-13 mins" in r for r in extracted)
        assert any("Chicken wings" in r for r in extracted)

    def test_section_banner_row_rejected(self):
        """A single-cell banner spanning the table carries no answer, but is
        short and header-prefixed, which used to score it into the top 3."""
        rows = [
            ["Food", "Amount", "Temp", "Time"],
            ["FRESH MEAT, POULTRY, FISH"],
            ["Sausages", "8 (410g)", "200C", "10-13 mins"],
            ["Chicken wings", "1kg", "200C", "30-32 mins"],
            ["Lamb chops", "4 (340g)", "180C", "11-12 mins"],
        ]
        doc, page = _table_page(rows)
        extracted = ingest._extract_table_rows(page)
        doc.close()
        assert any("Sausages" in r for r in extracted), "real rows must still index"
        assert not any("FRESH MEAT" in r for r in extracted)

    def test_merged_single_cell_data_row_kept(self):
        """PyMuPDF often merges a row into one cell; if it quotes an amount or a
        time it is still a real chart row and must survive."""
        line = "Asparagus Cut in 2.5cm pieces, blanched 60C 6-8 hours"
        rows = [
            ["Food", "Prep", "Temp", "Time"],
            ["Sausages", "8 (410g)", "200C", "10-13 mins"],
            ["Chicken", "1kg", "200C", "30 mins"],
            [line],
        ]
        doc, page = _table_page(rows)
        extracted = ingest._extract_table_rows(page)
        doc.close()
        assert any("Asparagus" in r for r in extracted)

    def test_prose_misdetected_as_table_is_not_indexed(self):
        """Gated on 3+ genuinely multi-cell rows so prose isn't double-indexed."""
        rows = [[f"Some flowing sentence of prose, number {i}."] for i in range(5)]
        doc, page = _table_page(rows)
        extracted = ingest._extract_table_rows(page)
        doc.close()
        assert extracted == []

    def test_no_tables_on_plain_page(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Just a paragraph of text.")
        extracted = ingest._extract_table_rows(page)
        doc.close()
        assert extracted == []


class TestIngestDocument:
    def test_text_document_becomes_ready(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling\nRun a cycle with citric acid once a month.")
        conn = db.connect()
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        assert row["status"] == "ready"
        assert row["chunk_count"] > 0
        assert row["error"] is None

    def test_chunks_are_indexed_in_fts(self, add_doc, stub_embeddings):
        add_doc("Descaling\nRun a cycle with citric acid once a month.")
        conn = db.connect()
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?", ('"citric"',)
        ).fetchone()["n"]
        assert n == 1

    def test_empty_document_records_error(self, add_doc, stub_embeddings):
        doc_id = add_doc("   \n  \n ")
        conn = db.connect()
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        assert row["status"] == "error"
        assert "No extractable text" in row["error"]

    def test_embeddings_stored_when_available(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling instructions here.")
        conn = db.connect()
        row = conn.execute("SELECT embedding FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()
        assert row["embedding"] is not None

    def test_keyword_only_mode_stores_no_embeddings(self, add_doc, no_embeddings):
        doc_id = add_doc("Descaling instructions here.")
        conn = db.connect()
        row = conn.execute("SELECT embedding FROM chunks WHERE doc_id=?", (doc_id,)).fetchone()
        assert row["embedding"] is None


class TestReindexAndDelete:
    def test_reindex_does_not_duplicate_chunks(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling\nRun a cycle with citric acid.\nRinse thoroughly.")
        conn = db.connect()
        before = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        assert ingest.reindex_document(doc_id) is True
        after = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        assert after == before
        fts = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
        assert fts == before

    def test_reindex_leaves_document_ready(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling instructions.")
        ingest.reindex_document(doc_id)
        conn = db.connect()
        row = conn.execute("SELECT status FROM documents WHERE id=?", (doc_id,)).fetchone()
        assert row["status"] == "ready"

    def test_reindex_missing_document(self, data_dir, stub_embeddings):
        assert ingest.reindex_document(999) is False

    def test_reindex_missing_file(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling instructions.")
        db.pdf_path(doc_id, "doc.md").unlink()
        assert ingest.reindex_document(doc_id) is False

    def test_delete_removes_everything(self, add_doc, stub_embeddings):
        doc_id = add_doc("Descaling instructions.")
        path = db.pdf_path(doc_id, "doc.md")
        assert ingest.delete_document(doc_id) is True
        conn = db.connect()
        assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"] == 0
        assert conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"] == 0
        assert not path.exists()

    def test_delete_missing_document(self, data_dir, stub_embeddings):
        assert ingest.delete_document(999) is False
