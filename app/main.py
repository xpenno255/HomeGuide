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
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import db, embeddings, ingest, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("homeguide")

app = FastAPI(title="HomeGuide", docs_url="/api/docs")

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
DEFAULT_K = 4
MAX_K = 10


@app.on_event("startup")
def startup() -> None:
    conn = db.connect()
    # Anything mid-ingest when the container stopped is stale
    with db.lock:
        conn.execute(
            "UPDATE documents SET status = 'error', error = 'Interrupted during indexing — delete and re-upload' "
            "WHERE status = 'processing'"
        )
        conn.commit()
    # Warm the embedding model off the request path
    threading.Thread(target=embeddings.get_model, daemon=True).start()


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
    }


def _run_query(q: str, k: int, category: str | None):
    q = (q or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Missing query")
    k = max(1, min(int(k), MAX_K))
    results = search.hybrid_search(q, k=k, category=category)
    if not results:
        return {
            "results": [],
            "note": "No matching content found in the household document library.",
        }
    return {"results": results}


@app.get("/query")
def query_get(q: str = "", k: int = DEFAULT_K, category: str | None = None):
    return _run_query(q, k, category)


@app.post("/query")
async def query_post(payload: dict):
    return _run_query(
        payload.get("query") or payload.get("q") or "",
        payload.get("k") or DEFAULT_K,
        payload.get("category"),
    )


@app.post("/api/upload")
async def upload(
    background: BackgroundTasks,
    file: UploadFile,
    title: str = Form(""),
    category: str = Form("manual"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Use PDF, TXT, or MD.")
    title = title.strip() or Path(file.filename).stem.replace("_", " ").replace("-", " ").strip()
    category = category.strip().lower() or "manual"

    conn = db.connect()
    with db.lock:
        cur = conn.execute(
            "INSERT INTO documents (title, category, filename) VALUES (?, ?, ?)",
            (title, category, file.filename),
        )
        conn.commit()
    doc_id = cur.lastrowid

    dest = db.pdf_path(doc_id, file.filename)
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    background.add_task(ingest.ingest_document, doc_id)
    return {"id": doc_id, "title": title, "status": "processing"}


@app.get("/api/documents")
def list_documents():
    conn = db.connect()
    rows = conn.execute(
        "SELECT id, title, category, filename, pages, chunk_count, status, error, created_at "
        "FROM documents ORDER BY created_at DESC, id DESC"
    ).fetchall()
    return {"documents": [dict(r) for r in rows]}


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
