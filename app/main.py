"""HomeGuide — a local document library your Home Assistant voice agent can query.

Endpoints:
  GET  /                       web UI
  GET  /query?q=...            search (used by the HA agent; also accepts POST JSON)
  POST /api/upload             add a document (multipart: file, title, category)
  GET  /api/documents          list documents
  DELETE /api/documents/{id}   remove a document
  GET  /api/documents/{id}/file  original PDF
  GET  /health                 liveness + index stats
"""

import logging
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, StrictBool

from . import catalog, db, embeddings, ingest, llm, procedure, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("homeguide")

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
DEFAULT_K = 5
MAX_K = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = db.connect()
    # Anything mid-ingest when the container stopped is stale
    with db.lock:
        conn.execute(
            "UPDATE documents SET status = 'error', error = 'Interrupted during indexing — reindex or re-upload' "
            "WHERE status = 'processing'"
        )
        conn.commit()
    # Warm the embedding model off the request path
    threading.Thread(target=embeddings.get_model, daemon=True).start()
    yield


app = FastAPI(title="HomeGuide", docs_url="/api/docs", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    conn = db.connect()
    docs = conn.execute("SELECT COUNT(*) AS n FROM documents WHERE status = 'ready'").fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    return {
        "status": "ok",
        "documents": docs,
        "chunks": chunks,
        "semantic_search": embeddings.available(),
        "llm": llm.enabled(),
    }


def _run_query(q: str, k: int, category: str | None, appliance_id: str | None = None):
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")
    try:
        k = max(1, min(int(k), MAX_K))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="'k' must be an integer")
    appliance = None
    if appliance_id is not None:
        appliance, failure = catalog.scope_for(appliance_id, q)
        if failure:
            return failure
    reviewed = procedure.resolve(q, appliance, category)
    if reviewed:
        return reviewed
    results = search.hybrid_search(q, k=k, category=category,
                                   doc_ids=set(appliance['document_ids']) if appliance else None)
    if not results:
        return {
            "status": "no_match",
            "results": [],
            "retryable": False,
            "note": (
                "No matching content found for this query in the household document library. "
                "This search did not find an answer; it does not prove the information "
                "is absent from every document."
            ),
        }
    return {"status": "matched", "results": results,
            **({'appliance': {k: appliance[k] for k in ['id', 'name', 'manufacturer', 'model', 'region']}} if appliance else {})}


@app.get("/query")
def query_get(q: str = "", k: int = DEFAULT_K, category: str | None = None,
              appliance_id: str | None = None):
    return _run_query(q, k, category, appliance_id)


@app.post("/query")
async def query_post(payload: dict):
    return _run_query(
        payload.get("query") or payload.get("q") or "",
        payload.get("k") or DEFAULT_K,
        payload.get("category"),
        payload.get("appliance_id"),
    )


@app.post("/api/resolve")
def resolve_procedure(payload: dict):
    """Read-only reviewed answers for Assist; never performs an LLM call."""
    question = str(payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")
    return procedure.resolve(question) or {"status": "not_applicable"}


@app.post("/api/ask")
def ask(payload: dict):
    """UI-only: full RAG round-trip through the local LLM, mirroring what the
    Home Assistant agent does with the /query results."""
    question = (payload.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question")

    reviewed = _run_query(question, DEFAULT_K, None, payload.get("appliance_id"))
    if "answer" in reviewed:
        return reviewed
    if not llm.enabled():
        raise HTTPException(status_code=503, detail="No LLM configured (set LLM_BASE_URL)")
    results = reviewed["results"]
    if not results:
        return {
            "answer": "The document library doesn't contain anything about that.",
            "results": [],
        }
    try:
        answer = llm.ask(question, results)
        if not answer or not answer.strip():
            answer = "The document lookup returned no spoken answer. Please consult the referenced manual pages."
    except Exception as exc:
        log.exception("LLM request failed")
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}")
    return {"answer": answer, "results": results}


@app.post("/api/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile,
    title: str = Form(""),
    category: str = Form("manual"),
    appliance_id: str = Form(""),
    verified: bool = Form(False),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use PDF, TXT, or MD.")
    title = title.strip() or Path(file.filename).stem.replace("_", " ").replace("-", " ").strip()
    category = category.strip().lower() or "manual"

    conn = db.connect()
    if appliance_id and not conn.execute('SELECT 1 FROM appliances WHERE id=?', (appliance_id,)).fetchone():
        raise HTTPException(status_code=400, detail='Select an existing appliance before uploading.')
    with db.lock:
        cur = conn.execute(
            "INSERT INTO documents (title, category, filename) VALUES (?, ?, ?)",
            (title, category, file.filename),
        )
        conn.commit()
    doc_id = cur.lastrowid
    if appliance_id:
        catalog.link_document(doc_id, appliance_id, verified)

    dest = db.pdf_path(doc_id, file.filename)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    background.add_task(ingest.ingest_document, doc_id)
    return {"id": doc_id, "title": title, "status": "processing"}


@app.get("/api/documents")
def list_documents():
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, title, category, filename, pages, chunk_count, status, error, created_at, active "
        "FROM documents ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return {'documents': [dict(r) | {'appliances': [dict(a) for a in conn.execute(
        'SELECT appliance_id,verified FROM document_appliances WHERE doc_id=? ORDER BY appliance_id', (r['id'],))]} for r in rows]}


class ApplianceInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    kind: str = 'other'
    manufacturer: str = Field(default='', max_length=100)
    model: str = Field(default='', max_length=100)
    region: str = Field(default='', max_length=100)
    aliases: list[str] = Field(default_factory=list, max_length=20)


class DocumentLink(BaseModel):
    appliance_id: str
    verified: StrictBool


class DocumentAvailability(BaseModel):
    active: StrictBool


@app.get('/api/catalog')
def get_catalog():
    return catalog.get_catalog()


@app.get('/api/appliances')
def get_appliances():
    return {'appliances': catalog.list_appliances(), 'kinds': catalog.KINDS}


@app.post('/api/appliances')
def create_appliance(payload: ApplianceInput):
    try:
        return catalog.save_appliance(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.put('/api/appliances/{appliance_id}')
def update_appliance(appliance_id: str, payload: ApplianceInput):
    try:
        return catalog.save_appliance(payload.model_dump(), appliance_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail='Appliance not found') from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/api/documents/{doc_id}/appliances')
def associate_document(doc_id: int, payload: DocumentLink):
    try:
        catalog.link_document(doc_id, payload.appliance_id, payload.verified)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {'document_id': doc_id, **payload.model_dump()}


@app.patch('/api/documents/{doc_id}')
def set_document_availability(doc_id: int, payload: DocumentAvailability):
    conn = db.connect()
    with db.lock:
        cur = conn.execute('UPDATE documents SET active=? WHERE id=?', (int(payload.active), doc_id))
        conn.commit()
    if not cur.rowcount:
        raise HTTPException(status_code=404, detail='Document not found')
    search.invalidate_cache()
    return {'id': doc_id, 'active': payload.active}


@app.post("/api/reindex")
def reindex_all(background: BackgroundTasks):
    """Rebuild every document from its stored file.

    Sequential in one background task: reindexing is CPU-bound embedding work,
    and running documents in parallel would just thrash the cores while search
    serves half-rebuilt results for longer. Documents whose original file is
    missing are skipped and reported rather than failing the whole run.
    """
    conn = db.connect()
    rows = conn.execute("SELECT id, filename FROM documents ORDER BY id").fetchall()
    todo = [r["id"] for r in rows if db.pdf_path(r["id"], r["filename"]).exists()]
    skipped = [r["id"] for r in rows if r["id"] not in todo]

    def run(ids: list[int]) -> None:
        for doc_id in ids:
            ingest.reindex_document(doc_id)

    background.add_task(run, todo)
    return {"queued": todo, "skipped_missing_file": skipped}


@app.post("/api/documents/{doc_id}/reindex")
def reindex_document(doc_id: int, background: BackgroundTasks):
    conn = db.connect()
    row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not db.pdf_path(doc_id, row["filename"]).exists():
        raise HTTPException(status_code=409, detail="Original file missing — delete and re-upload")
    background.add_task(ingest.reindex_document, doc_id)
    return {"id": doc_id, "status": "processing"}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    if not ingest.delete_document(doc_id):
        raise HTTPException(status_code=404, detail="Document not found")
    return JSONResponse({"deleted": doc_id})


@app.get("/api/documents/{doc_id}/file")
def get_file(doc_id: int):
    conn = db.connect()
    row = conn.execute("SELECT filename FROM documents WHERE id = ?", (doc_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = db.pdf_path(doc_id, row["filename"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="File missing on disk")
    return FileResponse(path, filename=row["filename"])
