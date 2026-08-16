"""HTTP surface, including the /query contract Home Assistant depends on."""

import io

import pytest


def _upload(client, name="manual.md", text="Descaling\nRun a citric acid cycle.", **form):
    payload = {"title": "Boiler manual", "category": "manual", **form}
    return client.post(
        "/api/upload",
        files={"file": (name, io.BytesIO(text.encode()), "text/markdown")},
        data=payload,
    )


class TestHealth:
    def test_reports_counts_and_modes(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["documents"] == 0
        assert body["chunks"] == 0
        assert set(body) == {"status", "documents", "chunks", "semantic_search", "llm"}

    def test_counts_only_ready_documents(self, client):
        _upload(client)
        assert client.get("/health").json()["documents"] == 1


class TestUpload:
    def test_accepted_and_indexed(self, client):
        resp = _upload(client)
        assert resp.status_code == 200
        assert resp.json()["status"] == "processing"
        docs = client.get("/api/documents").json()["documents"]
        assert docs[0]["status"] == "ready"

    def test_rejects_unsupported_type(self, client):
        resp = _upload(client, name="photo.jpg")
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_title_defaults_to_filename(self, client):
        _upload(client, name="bosch_dishwasher_manual.md", title="")
        docs = client.get("/api/documents").json()["documents"]
        assert docs[0]["title"] == "bosch dishwasher manual"

    def test_scanned_document_records_error(self, client):
        _upload(client, text="   \n  ")
        docs = client.get("/api/documents").json()["documents"]
        assert docs[0]["status"] == "error"
        assert "No extractable text" in docs[0]["error"]


class TestQuery:
    def test_get_returns_excerpts(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        _upload(client)
        body = client.get("/query", params={"q": "citric"}).json()
        assert body["results"]
        assert set(body["results"][0]) == {"document", "category", "page", "excerpt"}

    def test_no_match_returns_note_not_error(self, client):
        """The agent is told to say "not in the library" — it needs a clean
        empty result, not a failure."""
        _upload(client)
        body = client.get("/query", params={"q": "lawnmower blade replacement"}).json()
        assert body["results"] == []
        assert "No matching content" in body["note"]

    def test_missing_query_is_400(self, client):
        assert client.get("/query", params={"q": ""}).status_code == 400

    def test_k_is_clamped(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        for i in range(12):
            _upload(client, text=f"Descaling {i}\nRun a citric acid cycle.", title=f"D{i}")
        body = client.get("/query", params={"q": "citric", "k": 999}).json()
        assert len(body["results"]) <= 10

    def test_post_accepts_query_key(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        _upload(client)
        assert client.post("/query", json={"query": "citric"}).json()["results"]

    def test_post_accepts_q_alias(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        _upload(client)
        assert client.post("/query", json={"q": "citric"}).json()["results"]

    def test_post_rejects_non_integer_k(self, client):
        _upload(client)
        resp = client.post("/query", json={"query": "citric", "k": "abc"})
        assert resp.status_code == 400

    def test_category_filter_applies(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        _upload(client, category="manual")
        body = client.get("/query", params={"q": "citric", "category": "warranty"}).json()
        assert body["results"] == []


class TestDocuments:
    def test_list_shape(self, client):
        _upload(client)
        doc = client.get("/api/documents").json()["documents"][0]
        assert set(doc) >= {"id", "title", "category", "status", "pages", "chunk_count"}

    def test_original_file_downloadable(self, client):
        doc_id = _upload(client).json()["id"]
        resp = client.get(f"/api/documents/{doc_id}/file")
        assert resp.status_code == 200
        assert b"citric" in resp.content

    def test_file_404_for_unknown_document(self, client):
        assert client.get("/api/documents/999/file").status_code == 404

    def test_delete(self, client):
        doc_id = _upload(client).json()["id"]
        assert client.delete(f"/api/documents/{doc_id}").status_code == 200
        assert client.get("/api/documents").json()["documents"] == []

    def test_delete_404(self, client):
        assert client.delete("/api/documents/999").status_code == 404


class TestReindex:
    def test_rebuilds_without_duplicating(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        doc_id = _upload(client).json()["id"]
        before = client.get("/health").json()["chunks"]
        assert client.post(f"/api/documents/{doc_id}/reindex").status_code == 200
        assert client.get("/health").json()["chunks"] == before

    def test_document_still_searchable_after(self, client, stub_embeddings):
        stub_embeddings.set("citric", 0.85)
        doc_id = _upload(client).json()["id"]
        client.post(f"/api/documents/{doc_id}/reindex")
        assert client.get("/query", params={"q": "citric"}).json()["results"]

    def test_404_for_unknown_document(self, client):
        assert client.post("/api/documents/999/reindex").status_code == 404

    def test_409_when_original_file_is_gone(self, client):
        from app import db

        doc_id = _upload(client).json()["id"]
        db.pdf_path(doc_id, "manual.md").unlink()
        resp = client.post(f"/api/documents/{doc_id}/reindex")
        assert resp.status_code == 409


class TestStartup:
    def test_interrupted_ingest_marked_as_error(self, data_dir, stub_embeddings):
        """A container killed mid-ingest leaves 'processing' rows that would
        otherwise sit there forever."""
        from fastapi.testclient import TestClient

        from app import db
        from app.main import app

        conn = db.connect()
        with db.lock:
            conn.execute(
                "INSERT INTO documents (title, category, filename, status) "
                "VALUES ('Stuck', 'manual', 'x.md', 'processing')"
            )
            conn.commit()
        with TestClient(app) as client:
            doc = client.get("/api/documents").json()["documents"][0]
            assert doc["status"] == "error"
            assert "Interrupted" in doc["error"]


class TestAsk:
    def test_disabled_without_llm_configured(self, client, monkeypatch):
        from app import llm

        monkeypatch.setattr(llm, "BASE_URL", "")
        resp = client.post("/api/ask", json={"question": "how do I descale it?"})
        assert resp.status_code == 503

    def test_no_results_short_circuits_before_the_model(self, client, monkeypatch):
        """No excerpts means nothing to ground an answer in — the model must not
        be called at all, or it will answer from its own knowledge."""
        from app import llm

        monkeypatch.setattr(llm, "enabled", lambda: True)
        monkeypatch.setattr(
            llm, "ask", lambda *a, **k: pytest.fail("LLM called with no excerpts")
        )
        _upload(client)
        body = client.post("/api/ask", json={"question": "lawnmower blade"}).json()
        assert body["results"] == []
        assert "doesn't contain" in body["answer"]

    def test_missing_question_is_400(self, client, monkeypatch):
        from app import llm

        monkeypatch.setattr(llm, "enabled", lambda: True)
        assert client.post("/api/ask", json={"question": "  "}).status_code == 400

    def test_model_failure_is_502(self, client, stub_embeddings, monkeypatch):
        from app import llm

        def boom(*a, **k):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(llm, "enabled", lambda: True)
        monkeypatch.setattr(llm, "ask", boom)
        stub_embeddings.set("citric", 0.85)
        _upload(client)
        assert client.post("/api/ask", json={"question": "citric"}).status_code == 502

    def test_answer_returned_with_excerpts(self, client, stub_embeddings, monkeypatch):
        from app import llm

        monkeypatch.setattr(llm, "enabled", lambda: True)
        monkeypatch.setattr(llm, "ask", lambda q, results: f"Answer from {len(results)} excerpt(s)")
        stub_embeddings.set("citric", 0.85)
        _upload(client)
        body = client.post("/api/ask", json={"question": "citric"}).json()
        assert body["results"]
        assert "Answer from" in body["answer"]
