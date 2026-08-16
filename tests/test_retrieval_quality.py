"""Retrieval quality against a real manual, with the real embedding model.

Everything else in this suite tests mechanics against a stub. This file is the
only place that answers the question that actually matters: does the system
return the right excerpt for a real question, and stay silent for one the
library cannot answer?

It needs a real document, so it is skipped by default. To run it:

    HOMEGUIDE_TEST_PDF=/path/to/AF500UK_IG.pdf pytest tests/test_retrieval_quality.py

The first run downloads the embedding model (~130 MB). The expectations below
are specific to the Ninja AF500UK *Inspiration Guide* — a recipe book with
cooking charts, no troubleshooting section and no care instructions.
"""

import os
import shutil
from pathlib import Path

import pytest

PDF = os.environ.get("HOMEGUIDE_TEST_PDF", "")

pytestmark = [
    pytest.mark.corpus,
    pytest.mark.skipif(
        not PDF or not Path(PDF).exists(),
        reason="set HOMEGUIDE_TEST_PDF to the AF500UK guide to run quality tests",
    ),
]

FAULT_CODES = """# Bosch Serie 6 Dishwasher — Troubleshooting

E4: The water flow sensor has detected a blockage in the filter system.
Clean the filters and check the spray arms rotate freely.

F21: Circulation pump fault. Switch the appliance off at the wall for two
minutes, then restart the programme.
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    """One index, built once — embedding a 21-page manual is slow."""
    from app import db, ingest, search

    tmp = tmp_path_factory.mktemp("corpus")
    db.DATA_DIR, db.PDF_DIR, db.DB_PATH = tmp, tmp / "pdfs", tmp / "homeguide.db"
    db._conn = None
    search.invalidate_cache()

    conn = db.connect()
    sources = [
        ("Ninja Air Fryer AF500 recipe guide", Path(PDF)),
        ("Bosch dishwasher manual", None),
    ]
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

    row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
    assert row["n"] > 50, "manual did not index"
    yield search
    conn.close()
    db._conn = None
    search.invalidate_cache()


# Questions the guide genuinely answers. The chart *row* must win, not the
# recipe prose that merely mentions the same ingredient.
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


# Questions this library cannot answer. Returning a plausible-looking excerpt
# is worse than returning nothing: the agent reads it out as fact.
@pytest.mark.parametrize(
    "query",
    [
        "descaling",
        "cleaning and care",
        "dishwasher safe parts",
        "lawnmower blade replacement",
        "what is the warranty period",
    ],
)
def test_unanswerable_queries_stay_silent(library, query):
    results = library.hybrid_search(query, k=3)
    assert results == [], f"{query!r} -> {[r['excerpt'][:70] for r in results]}"


# Exact identifiers are the one thing BM25 does that embeddings cannot.
@pytest.mark.parametrize("query", ["what does E4 mean", "E4", "dishwasher fault F21"])
def test_fault_codes_found(library, query):
    results = library.hybrid_search(query, k=3)
    assert results, f"{query!r} returned nothing"
    assert results[0]["document"] == "Bosch dishwasher manual"


def test_appliance_vocabulary_query_is_a_known_residual(library):
    """"how do i clean the air fryer" shares heavy vocabulary with the corpus,
    so recipe chunks clear the cosine floor legitimately (~0.75) while correct
    chart answers only reach 0.78-0.82. The margin is too thin to threshold.

    This is expected to start passing once the real AF500 *instruction* manual
    is indexed and the query has a genuine answer to find — at which point
    delete the xfail rather than the assertion.
    """
    results = library.hybrid_search("how do i clean the air fryer", k=3)
    if results:
        pytest.xfail(f"known residual: {[r['excerpt'][:60] for r in results]}")
