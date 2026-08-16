"""Retrieval mechanics.

These use the stub embedder, so every cosine score is exact and the assertions
are about the *rules* — the similarity floor, the document-frequency filter,
the corroboration requirement, fusion order. Whether the real model separates
real answers from real noise is a different question, covered in
test_retrieval_quality.py.
"""

import pytest

from app import search


class TestSimilarityFloor:
    def test_hit_above_floor_returned(self, add_doc, stub_embeddings):
        stub_embeddings.set("Descaling", 0.80)
        add_doc("Descaling\nRun a cycle with citric acid.")
        assert search.hybrid_search("descaling", k=3)

    def test_hit_below_floor_suppressed(self, add_doc, stub_embeddings):
        """0.65 is the range unanswerable queries actually reached against the
        real corpus — dehydrator rows for "descaling"."""
        stub_embeddings.set("Mushrooms", 0.65)
        add_doc("Mushrooms\nCleaned with soft brush, dried at 60C for 6-8 hours.")
        assert search.hybrid_search("descaling", k=3) == []

    def test_floor_is_inclusive_at_threshold(self, add_doc, stub_embeddings):
        stub_embeddings.set("Descaling", search.MIN_SIM)
        add_doc("Descaling\nRun a cycle with citric acid.")
        assert search.hybrid_search("descaling", k=3)


class TestDocumentFrequencyFilter:
    """DF is measured over chunks, so these need a library above DF_MIN_CHUNKS."""

    @staticmethod
    def _corpus(add_doc, n=25):
        for i in range(n):
            add_doc(f"Cooking notes {i}\nCooking is described here in section {i}.",
                    title=f"Doc {i}")

    def test_token_matching_most_of_library_is_dropped(self, add_doc, stub_embeddings):
        """"cooking" hit 77% of chunks in a single air-fryer manual, so BM25 was
        ranking by how often a chunk repeats the appliance's own vocabulary."""
        self._corpus(add_doc)
        assert search._selective_tokens(["cooking"]) == []

    def test_rare_token_is_kept(self, add_doc, stub_embeddings):
        self._corpus(add_doc)
        add_doc("Descaling\nRun a cycle with citric acid.", title="Descale doc")
        assert search._selective_tokens(["citric"]) == ["citric"]

    def test_absent_token_is_dropped(self, add_doc, stub_embeddings):
        self._corpus(add_doc)
        assert search._selective_tokens(["nonexistentword"]) == []

    def test_query_of_only_common_words_contributes_nothing(self, add_doc, stub_embeddings):
        self._corpus(add_doc)
        ranked, _trusted = search._fts_ranked("cooking", None)
        assert ranked == []

    def test_filter_disabled_on_a_small_library(self, add_doc, stub_embeddings):
        """Below DF_MIN_CHUNKS every token trivially appears in "most" chunks.
        Applying the ratio there would discard the whole query and switch
        keyword search off, taking exact fault-code lookup with it."""
        add_doc("Troubleshooting\nE4: the water flow sensor detected a blockage.")
        assert search._selective_tokens(["e4"]) == ["e4"]


class TestCorroborationRule:
    """A keyword hit that matched one ordinary word is a lexical coincidence
    unless the vector side agrees. Identifiers are the exception: an exact
    fault code is the one thing BM25 does that embeddings cannot."""

    def test_single_ordinary_word_match_suppressed(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1  # vector side finds nothing
        add_doc("Mushrooms\nCleaned with a soft brush, then dried at 60C.")
        assert search.hybrid_search("cleaning and care", k=3) == []

    def test_identifier_token_survives_alone(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1
        add_doc("Troubleshooting\nE4: the water flow sensor detected a blockage.")
        results = search.hybrid_search("what does E4 mean", k=3)
        assert results and "E4" in results[0]["excerpt"]

    def test_bare_identifier_query_works(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1
        add_doc("Troubleshooting\nE4: the water flow sensor detected a blockage.")
        assert search.hybrid_search("E4", k=3)

    def test_letter_only_fault_code_survives_alone(self, add_doc, stub_embeddings):
        """LG dryers report OE, tE, dE and PS — codes with no digit for the
        digit rule to catch. They are trusted for being far too rare in the
        library to be a coincidence."""
        stub_embeddings.default = 0.1  # vector side finds nothing
        for i in range(25):
            add_doc(f"Operating notes {i}\nGeneral guidance for section {i}.", title=f"Doc {i}")
        add_doc(
            "Before Calling for Service\nOE DRAIN PUMP ERROR. The drain pump "
            "motor has malfunctioned. Unplug the appliance and call for service.",
            title="Dryer manual",
        )
        results = search.hybrid_search("what does OE mean", k=3)
        assert results and "OE" in results[0]["excerpt"]

    def test_bare_letter_only_code_works(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1
        for i in range(25):
            add_doc(f"Operating notes {i}\nGeneral guidance for section {i}.", title=f"Doc {i}")
        add_doc("Service\nOE DRAIN PUMP ERROR occurred.", title="Dryer manual")
        assert search.hybrid_search("OE", k=3)

    def test_common_word_is_not_made_distinctive_by_rarity(self, add_doc, stub_embeddings):
        """The rarity rule must not readmit the coincidence it was built
        around: a word that appears throughout the library stays untrusted."""
        stub_embeddings.default = 0.1
        for i in range(25):
            add_doc(f"Care notes {i}\nCleaning and care guidance for part {i}.", title=f"Doc {i}")
        assert search.hybrid_search("cleaning", k=3) == []

    def test_two_selective_tokens_survive_alone(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1
        add_doc("Descaling\nRun a citric acid cycle to descale the boiler.")
        assert search.hybrid_search("citric descale", k=3)

    def test_single_word_match_kept_when_vector_agrees(self, add_doc, stub_embeddings):
        stub_embeddings.set("Cleaned", 0.85)
        add_doc("Mushrooms\nCleaned with a soft brush, then dried at 60C.")
        assert search.hybrid_search("cleaning", k=3)

    def test_keyword_only_mode_relaxes_the_rule(self, add_doc, no_embeddings):
        """With no embeddings there is no second opinion, so keyword hits stand
        alone — a degraded mode, but better than returning nothing."""
        add_doc("Mushrooms\nCleaned with a soft brush, then dried at 60C.")
        assert search.hybrid_search("cleaning", k=3)


class TestCategoryFilter:
    def test_restricts_to_category(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\nRun a citric acid cycle.", title="Manual", category="manual")
        add_doc("Cover\nCitric acid is covered by this policy.",
                title="Policy", category="warranty")
        results = search.hybrid_search("citric", k=5, category="warranty")
        assert results and all(r["category"] == "warranty" for r in results)

    def test_unknown_category_returns_nothing(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\nRun a citric acid cycle.")
        assert search.hybrid_search("citric", k=5, category="nonexistent") == []

    def test_category_is_case_insensitive(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\nRun a citric acid cycle.", category="manual")
        assert search.hybrid_search("citric", k=5, category="MANUAL")


class TestResultShape:
    def test_result_fields(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\nRun a citric acid cycle.", title="Boiler manual")
        r = search.hybrid_search("citric", k=1)[0]
        assert set(r) == {"document", "category", "page", "excerpt"}
        assert r["document"] == "Boiler manual"
        assert r["page"] == 1

    def test_k_limits_results(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        for i in range(6):
            add_doc(f"Descaling {i}\nRun a citric acid cycle here.", title=f"Doc {i}")
        assert len(search.hybrid_search("citric", k=2)) == 2

    def test_long_excerpt_truncated(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\n" + "citric acid and water " * 200)
        r = search.hybrid_search("citric", k=1)[0]
        assert len(r["excerpt"]) <= search.EXCERPT_MAX + 1
        assert r["excerpt"].endswith("…")

    def test_empty_library_returns_nothing(self, data_dir, stub_embeddings):
        assert search.hybrid_search("anything", k=3) == []

    def test_processing_documents_are_not_searched(self, add_doc, stub_embeddings):
        from app import db

        stub_embeddings.set("citric", 0.85)
        doc_id = add_doc("Descaling\nRun a citric acid cycle.")
        conn = db.connect()
        with db.lock:
            conn.execute("UPDATE documents SET status='processing' WHERE id=?", (doc_id,))
            conn.commit()
        search.invalidate_cache()
        assert search.hybrid_search("citric", k=3) == []


class TestCacheInvalidation:
    def test_new_document_is_searchable_immediately(self, add_doc, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        add_doc("Descaling\nRun a citric acid cycle.", title="First")
        search.hybrid_search("citric", k=3)  # warm the cache
        add_doc("Second descaling\nAnother citric acid cycle.", title="Second")
        titles = {r["document"] for r in search.hybrid_search("citric", k=5)}
        assert "Second" in titles

    def test_deleted_document_disappears(self, add_doc, stub_embeddings):
        from app import ingest

        stub_embeddings.set("citric", 0.85)
        doc_id = add_doc("Descaling\nRun a citric acid cycle.")
        assert search.hybrid_search("citric", k=3)
        ingest.delete_document(doc_id)
        assert search.hybrid_search("citric", k=3) == []


class TestSynonymFallback:
    """Household words versus manufacturer words. Expansion runs only when the
    library was about to return nothing, so it cannot disturb a working query —
    rewriting every query was tried and diluted appliance names badly enough to
    send "air fryer guarantee" to the coffee machine."""

    def test_partner_word_is_appended(self):
        assert search.expand_synonyms("what is the warranty") == (
            "what is the warranty guarantee"
        )

    def test_no_change_when_both_words_present(self):
        q = "warranty and guarantee terms"
        assert search.expand_synonyms(q) == q

    def test_unrelated_query_untouched(self):
        q = "sausages cooking time"
        assert search.expand_synonyms(q) == q

    def test_expansion_is_case_insensitive(self):
        assert "guarantee" in search.expand_synonyms("WARRANTY period")

    def test_fallback_finds_the_manufacturer_word(self, add_doc, stub_embeddings):
        stub_embeddings.set("Guarantee", 0.85)
        add_doc("Guarantee\nYour appliance carries a two year Guarantee.")
        assert search.hybrid_search("warranty", k=3)

    def test_working_query_is_not_rewritten(self, add_doc, stub_embeddings):
        """The whole point of the fallback: a query that already returns
        something must be answered from exactly what it asked for."""
        stub_embeddings.set("Guarantee", 0.85)
        stub_embeddings.set("Warranty", 0.95)
        add_doc("Guarantee\nThe Guarantee covers manufacturing defects.", title="A")
        add_doc("Warranty\nThe Warranty covers accidental damage.", title="B")
        top = search.hybrid_search("warranty", k=1)[0]
        assert "accidental" in top["excerpt"], "fallback fired on a working query"

    def test_no_results_still_possible(self, add_doc, stub_embeddings):
        stub_embeddings.default = 0.1
        add_doc("Guarantee\nYour appliance carries a two year Guarantee.")
        assert search.hybrid_search("lawnmower blade replacement", k=3) == []
