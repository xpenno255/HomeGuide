# HomeGuide

A local document library your Home Assistant voice agent can query. Upload appliance
manuals, warranties and house documents (PDF/TXT/MD) through a web UI; the agent calls
one REST tool and gets back the most relevant excerpts with document and page references.

```
"What does fault code E4 mean on the dishwasher?"
        │
Home Assistant ── Extended OpenAI Conversation ── Gemma on vLLM (GPU)
        │  rest function: query_home_documents
        ▼
HomeGuide container (this repo, CPU only)
  FastAPI ── hybrid search ── SQLite FTS5 (BM25 keywords)
                          └── fastembed vectors (bge-small, ONNX on CPU)
        ▼
  {"results": [{"document": "Bosch dishwasher manual", "page": 43, "excerpt": "E4 …"}]}
```

Retrieval is **hybrid**: BM25 keyword search catches exact strings like fault codes
("E4"), semantic embeddings catch loosely-phrased voice queries, and the two rankings
are merged with reciprocal rank fusion. Embeddings run on CPU via ONNX — your A1000
stays fully dedicated to vLLM.

## 1. Run it

A prebuilt image is published to GHCR by [CI](.github/workflows/docker.yml) on every
push to `main`. On your Docker host, grab the compose file and start it:

```bash
mkdir homeguide && cd homeguide
curl -O https://raw.githubusercontent.com/xpenno255/HomeGuide/main/docker-compose.yml
docker compose up -d
```

Or clone the repo and build from source: swap `image:` for `build: .` in
[docker-compose.yml](docker-compose.yml), then `docker compose up -d --build`.

Then open `http://<docker-host>:8480`. On first startup the container downloads the
embedding model (~130 MB) into `./data/models`; until that finishes, search runs in
keyword-only mode (the header stat shows which mode is active).

Everything persists in `./data` (SQLite DB, original PDFs, model cache) — back that
folder up and you can rebuild the container freely.

## 2. Add your manuals

Use the web UI: drop in a PDF, give it a recognisable title (**include the appliance
name people actually say**, e.g. "Ninja Air Fryer AF300 manual", not "AF300-UM-EN-v2"),
pick a category, done. Indexing a typical manual takes a few seconds to ~1 minute on CPU.

Use the "Try a question" box to check what the agent will see for a given query —
if the right excerpt comes back there, the agent has what it needs.

> **Scanned PDFs:** HomeGuide extracts embedded text. A scanned/image-only manual will
> fail with "No extractable text" — run it through OCR first (e.g. `ocrmypdf in.pdf out.pdf`)
> and upload the result. Most manufacturer-download PDFs are fine as-is.

## 3. Wire up Home Assistant

1. Open **Settings → Devices & Services → Extended OpenAI Conversation → Configure**.
2. In the **Functions** field, append the contents of
   [homeassistant/query_home_documents.yaml](homeassistant/query_home_documents.yaml)
   to your existing function list.
3. Replace `HOMEGUIDE_HOST` with your Docker host's LAN IP.
4. Optionally add a line like this to your agent's prompt template, which noticeably
   improves how reliably a small model reaches for the tool:

   ```
   You have access to the household document library via query_home_documents.
   For any question about appliances, manuals, cooking times, fault codes,
   warranties or house paperwork, call it before answering, and cite the
   document and page. Never invent appliance instructions.
   ```

Then ask your voice assistant something like *"how long do I cook chicken breast in
the air fryer?"* or *"what does E4 mean on the dishwasher?"*.

## API

| Endpoint | Purpose |
|---|---|
| `GET /query?q=...&k=4&category=manual` | Search; `k` = excerpts (1–10), `category` optional. Also accepts `POST` with `{"query": "..."}`. |
| `GET /health` | Liveness, doc/chunk counts, search mode |
| `POST /api/upload` | Multipart: `file`, `title`, `category` |
| `GET /api/documents` | List library |
| `DELETE /api/documents/{id}` | Remove a document |
| `GET /api/documents/{id}/file` | Original file |

`/query` returns at most ~3 KB of excerpt text by default (4 excerpts × 800 chars), sized
so it fits comfortably in a small model's context. Raise `k` in the `resource_template`
if your context budget allows.

## Troubleshooting

- **Agent answers from its own knowledge instead of calling the tool** — strengthen the
  prompt line from step 4, and make sure the appliance name in the document title matches
  what you say out loud.
- **Agent says nothing was found but the UI search finds it** — check HA can reach the
  Docker host: `curl "http://HOMEGUIDE_HOST:8480/health"` from the HA machine.
- **Upload shows "error"** — hover the row for the reason; almost always a scanned PDF
  needing OCR (see above).
- **No auth by design** — this binds to your LAN with no authentication, like most
  homelab services. Don't expose port 8480 to the internet.
