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


ENGLISH_PAGE = (
    "DESCALING\nAfter regular use, hard water can cause mineral build up in the "
    "internal components. We recommend that you descale the machine regularly "
    "when the light is flashing, and then rinse the tank with fresh water."
)
GERMAN_PAGE = (
    "ENTKALKEN\nNach regelmäßigem Gebrauch kann hartes Wasser zu "
    "Mineralablagerungen führen. Wir empfehlen, das Gerät regelmäßig zu "
    "entkalken, und der Tank sollte mit frischem Wasser gespült werden."
)
FRENCH_PAGE = (
    "DÉTARTRAGE\nAprès une utilisation régulière, l'eau dure peut provoquer une "
    "accumulation de minéraux dans les composants. Nous vous recommandons de "
    "détartrer la machine avec des produits pour cet usage."
)


class TestLanguageFilter:
    """EU appliance manuals ship every language in one PDF — the Sage brewer
    manual is 156 pages of which 19 are English. Indexing the rest lets the
    agent quote Portuguese back at you, and inflates the page count that
    strip_repeated_lines scales its threshold to."""

    def test_english_kept(self):
        assert ingest._is_probably_english(ENGLISH_PAGE)

    @pytest.mark.parametrize("text", [GERMAN_PAGE, FRENCH_PAGE])
    def test_other_languages_dropped(self, text):
        assert not ingest._is_probably_english(text)

    def test_polish_dropped(self):
        """The Sage manual's Polish section leaked Polish descaling text into
        results for "descaling" until these markers were added."""
        assert not ingest._is_probably_english(
            "ODKAMIENIANIE\nNa wyświetlaczu mruga DESCALE, jeśli jest potrzebny "
            "proces odkamieniania. Nie należy przerywać procesu, aby nie "
            "uszkodzić ekspresu przez osad z wody."
        )

    @pytest.mark.parametrize(
        "text",
        [
            "ΑΦΑΛΑΤΩΣΗ\nΜετά από τακτική χρήση, το σκληρό νερό μπορεί να "
            "προκαλέσει συσσώρευση αλάτων στα εσωτερικά εξαρτήματα της συσκευής "
            "και να μειώσει τη ροή του νερού κατά την παρασκευή του καφέ.",
            "ОЧИСТКА ОТ НАКИПИ\nПосле регулярного использования жёсткая вода "
            "может вызвать образование накипи на внутренних компонентах "
            "прибора и ухудшить вкус приготовленного кофе.",
        ],
    )
    def test_non_latin_scripts_dropped(self, text):
        """Scored separately: the word matching only sees Latin letters, so a
        Greek or Cyrillic page hits zero on every marker and would be kept."""
        assert not ingest._is_probably_english(text)

    def test_wordless_page_kept(self):
        """Diagrams and part-label lists have no function words either way;
        dropping a real English page is worse than keeping a foreign one."""
        assert ingest._is_probably_english("A B C D\n1 2 3\nMAX MIN")

    def test_english_label_page_kept(self):
        """The brewer's English components page has 66 words and not one
        function word — a "needs English evidence" rule would delete it."""
        assert ingest._is_probably_english(
            "EN Components\nA. Water tank lid\nB. Water tank\nC. Filter holder\n"
            "D. Carafe lid\nE. Glass carafe\nF. Warming plate\nG. Control panel\n"
            "H. Bloom shower\nI. Drip stop outlet\nJ. Cone filter basket\n"
            "K. Flat bottom basket\nL. Measuring scoop\nM. Water filter\n"
            "N. Filter adapter\nRATED VOLTAGE 220-240V 50-60Hz 1650-1980W"
        )

    def test_empty_page_kept(self):
        assert ingest._is_probably_english("")

    def test_multilingual_pdf_keeps_only_english(self, tmp_path):
        doc = pymupdf.open()
        for text in (ENGLISH_PAGE, GERMAN_PAGE, FRENCH_PAGE):
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 400), text, fontsize=11)
        path = tmp_path / "multi.pdf"
        doc.save(path)
        doc.close()
        pages = ingest.extract_pages(path)
        assert len(pages) == 1
        assert "hard water" in pages[0][1]

    def test_page_numbers_survive_filtering(self, tmp_path):
        """Skipped pages must not renumber the rest, or citations go wrong."""
        doc = pymupdf.open()
        for text in (GERMAN_PAGE, FRENCH_PAGE, ENGLISH_PAGE):
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 400), text, fontsize=11)
        path = tmp_path / "multi.pdf"
        doc.save(path)
        doc.close()
        pages = ingest.extract_pages(path)
        assert [p[0] for p in pages] == [3]

    def test_all_foreign_document_is_still_indexed(self, tmp_path):
        """A wholly non-English manual is better indexed than not indexed."""
        doc = pymupdf.open()
        for text in (GERMAN_PAGE, FRENCH_PAGE):
            page = doc.new_page()
            page.insert_textbox(pymupdf.Rect(40, 40, 550, 400), text, fontsize=11)
        path = tmp_path / "foreign.pdf"
        doc.save(path)
        doc.close()
        assert len(ingest.extract_pages(path)) == 2


def _columned_page(columns, width=800, height=600):
    """Render text columns side by side, separated by a real gutter."""
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=height)
    col_w = width / len(columns)
    for c, lines in enumerate(columns):
        for i, line in enumerate(lines):
            rect = pymupdf.Rect(
                c * col_w + 20, 40 + i * 40, (c + 1) * col_w - 20, 40 + i * 40 + 34
            )
            page.insert_textbox(rect, line, fontsize=9)
    return doc, page


class TestColumnOrder:
    """Manuals are printed as two-page spreads, so PyMuPDF's default block order
    interleaves unrelated sections — the AF500UK manual emitted its whole
    TROUBLESHOOTING column before the CLEANING & MAINTENANCE heading's content.
    """

    def test_columns_read_top_to_bottom_not_across(self):
        doc, page = _columned_page(
            [
                ["CLEANING AND MAINTENANCE", "Wipe the main unit with a damp cloth.",
                 "Never immerse the unit in water."],
                ["TROUBLESHOOTING GUIDE", "Why is the unit beeping?",
                 "The food is finished cooking."],
            ]
        )
        text = ingest._page_text(page)
        doc.close()
        left = text.index("Never immerse")
        right = text.index("TROUBLESHOOTING GUIDE")
        assert left < right, f"columns interleaved:\n{text}"

    def test_heading_stays_with_its_own_column(self):
        doc, page = _columned_page(
            [
                ["CLEANING AND MAINTENANCE", "Wash the crisper plates by hand."],
                ["TROUBLESHOOTING GUIDE", "Why is the unit beeping?"],
            ]
        )
        chunks = ingest.chunk_page(ingest._page_text(page))
        doc.close()
        wash = next(c for c in chunks if "crisper" in c)
        assert "TROUBLESHOOTING" not in wash.split("Wash")[0]

    def test_single_column_page_is_unchanged(self):
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(50, 50, 500, 300),
                            "DESCALING\nRun a citric acid cycle once a month.", fontsize=10)
        assert ingest._page_text(page).split() == page.get_text("text").split()
        doc.close()

    def test_no_gutter_means_no_column_split(self):
        doc, page = _columned_page([["ONE COLUMN ONLY", "All the text is here."]])
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        assert ingest._column_bounds(blocks, page.rect.width) == []
        doc.close()

    def test_full_width_banner_does_not_hide_the_gutter(self):
        """A footer spanning both columns must not stop them being detected."""
        doc, page = _columned_page(
            [["LEFT HEADING", "Left column text."], ["RIGHT HEADING", "Right column text."]]
        )
        page.insert_textbox(
            pymupdf.Rect(20, 560, 780, 590), "ninjakitchen.co.uk  17  18", fontsize=8
        )
        blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]
        assert ingest._column_bounds(blocks, page.rect.width)
        doc.close()


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
