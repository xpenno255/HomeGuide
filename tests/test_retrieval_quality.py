"""Retrieval quality against the real manuals, with the real embedding model.

Everything else in this suite tests mechanics against a stub. This file is the
only place that answers the question that actually matters: does the system
return the right excerpt for a real question, and stay silent for one the
library cannot answer?

It needs real documents, so it is skipped by default:

    HOMEGUIDE_TEST_PDF=~/AF500UK_IG.pdf \
    HOMEGUIDE_TEST_MANUAL_PDF=~/AF500UK_manual.pdf \
    pytest tests/test_retrieval_quality.py

The two Ninja PDFs are different documents and the split matters:
  * the *Quick Start Recipe Guide* (HOMEGUIDE_TEST_PDF) holds the cooking
    charts, and nothing about cleaning, faults or the guarantee;
  * the *User Manual* (HOMEGUIDE_TEST_MANUAL_PDF) holds safeguards, cleaning,
    troubleshooting and the guarantee, and no cooking charts.
Tests needing the manual skip if only the recipe guide is configured. The first
run downloads the embedding model (~130 MB).
"""

import os
import shutil
from pathlib import Path

import pytest

RECIPES = os.environ.get("HOMEGUIDE_TEST_PDF", "")
MANUAL = os.environ.get("HOMEGUIDE_TEST_MANUAL_PDF", "")
BREWER = os.environ.get("HOMEGUIDE_TEST_BREWER_PDF", "")

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not RECIPES or not Path(RECIPES).exists(),
        reason="set HOMEGUIDE_TEST_PDF to the AF500UK recipe guide",
    ),
]

needs_manual = pytest.mark.skipif(
    not MANUAL or not Path(MANUAL).exists(),
    reason="set HOMEGUIDE_TEST_MANUAL_PDF to the AF500UK instruction manual",
)
needs_brewer = pytest.mark.skipif(
    not BREWER or not Path(BREWER).exists(),
    reason="set HOMEGUIDE_TEST_BREWER_PDF to the Sage Precision Brewer manual",
)

RECIPE_TITLE = "Ninja Air Fryer Quick Start Recipe Guide"
MANUAL_TITLE = "Ninja Air Fryer User Manual"
BREWER_TITLE = "Sage Precision Coffee Brewer Manual"

FAULT_CODES = """# Bosch Serie 6 Dishwasher — Troubleshooting

E4: The water flow sensor has detected a blockage in the filter system.
Clean the filters and check the spray arms rotate freely.

F21: Circulation pump fault. Switch the appliance off at the wall for two
minutes, then restart the programme.
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """One index, built once — embedding two manuals is slow."""
    from app import db, ingest, search

    tmp = tmp_path_factory.mktemp("corpus")
    db.DATA_DIR, db.PDF_DIR, db.DB_PATH = tmp, tmp / "pdfs", tmp / "homeguide.db"
    db._conn = None
    search.invalidate_cache()

    sources = [(RECIPE_TITLE, Path(RECIPES)), ("Bosch dishwasher manual", None)]
    if MANUAL and Path(MANUAL).exists():
        sources.insert(1, (MANUAL_TITLE, Path(MANUAL)))
    if BREWER and Path(BREWER).exists():
        sources.insert(0, (BREWER_TITLE, Path(BREWER)))

    conn = db.connect()
    for title, source in sources:
        name = source.name if source else "faults.md"
        with db.lock:
            cur = conn.execute(
                "INSERT INTO documents (title, category, filename) VALUES (?, ?, ?)",
                (title, "manual", name),
            )
            conn.commit()
        dest = db.pdf_path(cur.lastrowid, name)
        if source:
            shutil.copy(source, dest)
        else:
            dest.write_text(FAULT_CODES, encoding="utf-8")
        ingest.ingest_document(cur.lastrowid)

    assert conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"] > 50
    yield search
    conn.close()
    db._conn = None
    search.invalidate_cache()


def _excerpts(results):
    return " ".join(r["excerpt"].lower() for r in results)


# --- Cooking questions: the chart *row* must win, not the recipe prose that
# merely mentions the same ingredient. Adding the instruction manual to the
# library must not disturb this.
@pytest.mark.parametrize(
    "query,expected",
    [
        ("chicken breast cooking time", "chicken breast"),
        ("sausages cooking time", "sausages"),
        ("how long for chicken wings", "chicken wings"),
        ("what temperature for lamb chops", "lamb chops"),
    ],
)
def test_chart_row_wins(library, query, expected):
    results = library.hybrid_search(query, k=3)
    assert results, f"{query!r} returned nothing"
    assert expected in results[0]["excerpt"].lower(), (
        f"{query!r} -> {results[0]['excerpt'][:120]!r}"
    )
    assert results[0]["document"] == RECIPE_TITLE


# --- Questions this library genuinely cannot answer. Returning a plausible
# excerpt is worse than returning nothing: the agent reads it out as fact.
@pytest.mark.parametrize("query", ["lawnmower blade replacement", "how do i change a tyre"])
def test_unanswerable_queries_stay_silent(library, query):
    results = library.hybrid_search(query, k=3)
    assert results == [], f"{query!r} -> {[r['excerpt'][:70] for r in results]}"


@pytest.mark.skipif(
    bool(BREWER and Path(BREWER).exists()),
    reason="descaling is answerable once a machine that descales is in the library",
)
def test_descaling_silent_without_a_descaling_appliance(library):
    """Air fryers do not descale. This was the original example of a confident
    wrong answer: it used to return dehydrator chart rows."""
    assert library.hybrid_search("descaling", k=3) == []


# --- Exact identifiers are the one thing BM25 does that embeddings cannot.
@pytest.mark.parametrize("query", ["what does E4 mean", "E4", "dishwasher fault F21"])
def test_fault_codes_found(library, query):
    results = library.hybrid_search(query, k=3)
    assert results, f"{query!r} returned nothing"
    assert results[0]["document"] == "Bosch dishwasher manual"


# --- Care questions must reach the instruction manual, not the recipe guide.
# Before the manual was added these returned dehydrator chart rows.
@needs_manual
@pytest.mark.parametrize(
    "query,expected",
    [
        ("are the drawers dishwasher safe", "immerse"),
        ("can i put the main unit in water", "immerse"),
    ],
)
def test_care_questions_reach_the_manual(library, query, expected):
    results = library.hybrid_search(query, k=3)
    assert results, f"{query!r} returned nothing"
    assert results[0]["document"] == MANUAL_TITLE
    assert expected in results[0]["excerpt"].lower()


@needs_manual
def test_guarantee_section_found(library):
    """Every appliance has a guarantee section, so a bare "guarantee period"
    is genuinely ambiguous — assert it finds one, not which one."""
    results = library.hybrid_search("guarantee period", k=3)
    assert results and "guarantee" in _excerpts(results)


@needs_manual
@needs_brewer
@pytest.mark.xfail(
    reason="document titles are not indexed. Only chunk text is searchable, and "
    "the Ninja guarantee section never says 'air fryer' — the appliance name "
    "lives in the title alone — so naming the appliance cannot steer the "
    "search. Harmless with one appliance, wrong as soon as two of them have a "
    "guarantee section.",
    strict=False,
)
def test_naming_the_appliance_picks_the_right_guarantee(library):
    """Ambiguity should be resolvable the way a user would resolve it: by
    saying which appliance they mean."""
    results = library.hybrid_search("air fryer guarantee period", k=3)
    assert results and results[0]["document"] == MANUAL_TITLE


@needs_manual
def test_soapy_water_answer_is_retrievable(library):
    """The manual answers this directly: "If food residue is stuck on the
    crisper plates or drawer, place them in a sink filled with warm, soapy
    water and allow to soak." Reaching it needs the cleaning table to be read
    in column order, so this is the sharpest test of layout-aware extraction.
    """
    results = library.hybrid_search("food stuck on the crisper plate", k=3)
    assert results and results[0]["document"] == MANUAL_TITLE
    assert "soapy" in _excerpts(results)


@needs_manual
@pytest.mark.parametrize(
    "query",
    [
        pytest.param(
            "how do i clean the air fryer",
            marks=pytest.mark.xfail(
                reason="the cover page chunk ('10.4L Air Fryer / AF500UK / "
                "INSTRUCTIONS') carries no answer but still takes rank 1 — the "
                "prose equivalent of the table banners filtered in "
                "_extract_table_rows",
                strict=False,
            ),
        ),
        "how do i wash the crisper plates",
        "are the drawers dishwasher safe",
    ],
)
def test_cleaning_queries_answer_with_cleaning_content(library, query):
    """Rank 1 must be the manual, and must actually talk about cleaning —
    asserting the goal rather than the absence of one known-bad string."""
    top = library.hybrid_search(query, k=3)[0]
    assert top["document"] == MANUAL_TITLE, f"{query!r} -> {top['document']}"
    assert any(
        word in top["excerpt"].lower()
        for word in ("clean", "wash", "dishwasher", "soapy", "immerse")
    ), f"{query!r} -> {top['excerpt'][:100]!r}"


@needs_manual
@pytest.mark.xfail(
    reason="the manual says 'Guarantee' (UK usage) and never 'warranty', so the "
    "token is absent from the index and the vector side stays under MIN_SIM",
    strict=False,
)
def test_warranty_vocabulary_gap(library):
    """"guarantee period" works; "what is the warranty" finds nothing. The HA
    function description tells the agent the library covers warranties."""
    assert library.hybrid_search("what is the warranty", k=3)


# --- Multi-language manuals. The Sage brewer manual is 156 pages of which only
# ~19 are English; the rest is German, French, Dutch, Italian, Spanish and
# Portuguese. Indexing all of it let the agent quote Portuguese back, and
# inflated the page count that strip_repeated_lines scales its threshold to.
@needs_brewer
def test_only_english_pages_are_indexed(library):
    from app import db

    conn = db.connect()
    row = conn.execute(
        "SELECT pages, chunk_count FROM documents WHERE title = ?", (BREWER_TITLE,)
    ).fetchone()
    assert row["pages"] < 60, f"expected the English subset, indexed {row['pages']} pages"
    assert row["chunk_count"] < 200


@needs_brewer
@pytest.mark.parametrize(
    "query",
    ["descaling", "how do i descale the coffee machine", "how much coffee to use"],
)
def test_no_foreign_language_results(library, query):
    """Portuguese descaling instructions used to surface for "descaling"."""
    foreign = ("descalcificar", "entkalken", "détartrage", "ontkalken", "acumulação")
    text = _excerpts(library.hybrid_search(query, k=3))
    assert not any(w in text for w in foreign), f"{query!r} returned non-English text"


@needs_brewer
def test_descaling_is_answerable_from_the_brewer(library):
    """"descaling" was the canonical unanswerable query until a machine that
    actually descales joined the library."""
    results = library.hybrid_search("descaling", k=3)
    assert results and results[0]["document"] == BREWER_TITLE
    assert "descal" in _excerpts(results)


@needs_brewer
@pytest.mark.parametrize(
    "query,expected_doc",
    [
        ("how much coffee to use", BREWER_TITLE),
        ("how often should i replace the water filter", BREWER_TITLE),
        ("chicken breast cooking time", RECIPE_TITLE),
    ],
)
def test_queries_route_to_the_right_appliance(library, query, expected_doc):
    """Three appliances in one library must not bleed into each other."""
    results = library.hybrid_search(query, k=3)
    assert results and results[0]["document"] == expected_doc, (
        f"{query!r} -> {results[0]['document'] if results else 'nothing'}"
    )
