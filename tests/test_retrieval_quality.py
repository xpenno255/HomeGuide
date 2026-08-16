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

RECIPE_TITLE = "Ninja Air Fryer Quick Start Recipe Guide"
MANUAL_TITLE = "Ninja Air Fryer User Manual"

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
@pytest.mark.parametrize("query", ["descaling", "lawnmower blade replacement"])
def test_unanswerable_queries_stay_silent(library, query):
    results = library.hybrid_search(query, k=3)
    assert results == [], f"{query!r} -> {[r['excerpt'][:70] for r in results]}"


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
    results = library.hybrid_search("guarantee period", k=3)
    assert results and results[0]["document"] == MANUAL_TITLE
    assert "guarantee" in _excerpts(results)


@needs_manual
def test_cleaning_question_does_not_return_recipes(library):
    """Whatever else it returns, a cleaning question must not be answered out
    of the recipe book — that was the original failure."""
    results = library.hybrid_search("how do i clean the air fryer", k=3)
    assert results
    assert all(r["document"] != RECIPE_TITLE for r in results), (
        f"recipe guide leaked in: {[r['excerpt'][:60] for r in results]}"
    )


# --- Known defects, measured. These xfail deliberately: they are the current
# top of the backlog, not mysteries.
@needs_manual
@pytest.mark.xfail(
    reason="p10 is multi-column, so the cleaning table's text is separated from "
    "its CLEANING & MAINTENANCE heading and lands under [TROUBLESHOOTING GUIDE]. "
    "Needs layout-aware extraction; an all-caps heading heuristic was tried and "
    "regressed the cooking charts.",
    strict=False,
)
def test_soapy_water_answer_is_retrievable(library):
    """The manual answers this directly: "If food residue is stuck on the
    crisper plates or drawer, place them in a sink filled with warm, soapy
    water and allow to soak." Today the recipe guide's chip prose outranks it."""
    results = library.hybrid_search("food stuck on the crisper plate", k=3)
    assert results and results[0]["document"] == MANUAL_TITLE
    assert "soapy" in _excerpts(results)


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


@needs_manual
@pytest.mark.xfail(
    reason="cover page and parts-list chunks carry no answer but rank highly for "
    "short care queries — the prose equivalent of the table banners already "
    "filtered in _extract_table_rows",
    strict=False,
)
def test_page_furniture_does_not_outrank_real_answers(library):
    for query in ("how do i clean the air fryer", "how do i wash the crisper plates"):
        top = library.hybrid_search(query, k=1)[0]["excerpt"].lower()
        assert "instructions 10.4l" not in top
        assert "getting to know the control panel" not in top
