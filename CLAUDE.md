# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

HomeGuide is a local RAG service for household documents (appliance manuals, warranties) that a Home Assistant voice agent queries as a tool. The agent side is Gemma served by vLLM, wired up through Extended OpenAI Conversation's `rest` function type ([homeassistant/query_home_documents.yaml](homeassistant/query_home_documents.yaml)) — the function description there is prompt engineering for a small model; edit it carefully.

## Development commands

There is no Docker on this machine; the production Docker host is a separate LAN machine (live app: `http://192.168.1.102:8480`). Develop against a venv:

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# Run locally (never point DATA_DIR at ./data used by anything real)
DATA_DIR=/tmp/homeguide-test FASTEMBED_CACHE_PATH=/tmp/homeguide-test/models \
  .venv/bin/uvicorn app.main:app --port 8490
```

There is no test suite; verification is exercising the API. The established flow after any ingestion/search change:

```bash
rm -rf /tmp/homeguide-test/homeguide.db* /tmp/homeguide-test/pdfs   # chunks are built at upload time
curl -X POST localhost:8490/api/upload -F file=@manual.pdf -F "title=..." -F "category=manual"
curl "localhost:8490/query?q=air%20fryer%20chicken%20breast%20cooking%20time&k=5"
```

Retrieval regression cases that must keep passing. The only document indexed in production is `AF500UK_IG.pdf`, which is the Ninja **Inspiration Guide** (a recipe book) — it has cooking charts but no troubleshooting section, no fault codes and no care instructions, so fault-code cases need a separate fixture:
- Cooking-chart queries ("chicken breast cooking time", "sausages cooking time", "what temperature for lamb chops") return the chart *row* at rank 1, not recipe prose
- Exact fault codes return the troubleshooting chunk — verify against a fixture containing codes, not the AF500UK guide. A bare "E4" must work as well as "what does E4 mean"; identifier tokens are the one thing BM25 does that the embeddings cannot
- Queries the library cannot answer ("descaling", "cleaning and care", "dishwasher safe parts", "lawnmower blade replacement") return zero results — the agent is told to say "not in the library" rather than guess

Known residual: "how do i clean the air fryer" still returns three recipe chunks. They clear the cosine floor legitimately (0.75) because the query shares heavy vocabulary with the corpus, and correct chart answers only score 0.78-0.82 — the 0.03 margin is too thin to threshold. This should resolve once the actual AF500 instruction manual is indexed and the query has a real answer to find; recheck it then rather than tightening `MIN_SIM`.

To test the LLM answer path (`/api/ask`) without real inference, run a stub OpenAI-compatible server (see git history for `/tmp/fake_vllm.py`) and set `LLM_BASE_URL` to it. Do NOT test against the Ollama on 192.168.1.102:11434 — it is deliberately dormant (vLLM holds the GPU) and completions hang forever.

## Deployment

Push to `main` → GitHub Actions builds `ghcr.io/xpenno255/homeguide:latest` (public, no auth to pull) → user runs `docker compose pull && up -d` on the Docker host. **Chunking/ingestion changes only reach existing documents after a reindex** — chunks and embeddings are computed at upload time and stored in SQLite. Use the Reindex button in the web UI, or `POST /api/documents/{id}/reindex`, which rebuilds from the original file already on disk. Search-time changes (thresholds, fusion) take effect immediately without one.

## Architecture

Single FastAPI app ([app/main.py](app/main.py)), single SQLite DB (one shared connection + `db.lock`, WAL mode). Two consumer-facing paths:

- `GET /query` — called by Home Assistant. Response size is budgeted for a small model's context (k excerpts × 800 chars); don't inflate it.
- `POST /api/ask` — web-UI only. Same search, then answers via any OpenAI-compatible server (`LLM_BASE_URL`, model auto-detected from `/v1/models`). Never used by HA.

**Search** ([app/search.py](app/search.py)) is hybrid, fused with RRF: FTS5 BM25 (porter tokenizer, stopwords stripped from the query) + cosine over fastembed vectors (bge-small-en-v1.5, CPU/ONNX — the GPU belongs to vLLM). All chunk embeddings are cached as one in-memory numpy matrix; `search.invalidate_cache()` must be called after any chunk mutation.

Four tuned behaviours embody real failures — none are arbitrary, and loosening any of them brings back confidently wrong answers:
- `MIN_SIM = 0.70` — measured on the AF500UK guide: chunks that answer the query score 0.78-0.82, unanswerable queries top out at 0.53-0.68. It was 0.60, which handed the agent dehydrator chart rows for "descaling".
- `DF_MAX_RATIO = 0.5` — query tokens matching over half the library are dropped before the BM25 OR-query. In a single air-fryer manual "cooking" appears in 77% of chunks and "air" in 63%, so without this BM25 ranks by how often a chunk repeats the appliance's own vocabulary. The hardcoded `STOPWORDS` list cannot cover this: the noise words are whatever the library happens to be about.
- **FTS hits need corroboration unless they matched ≥2 selective tokens or an identifier-shaped token** (one containing a digit: `E4`, `F21`). BM25's unique value over embeddings is exact identifiers; for ordinary English the vector side is better calibrated, so a lone word match like "cleaning" hitting "Cleaned with soft brush" in a mushroom row is a coincidence, not an answer. Keyword-only mode (no embedding model) relaxes this — there is no second opinion to consult.
- Vector-before-FTS ordering in RRF fusion — tie-breaks favor the calibrated retriever.

**Ingestion** ([app/ingest.py](app/ingest.py)) encodes the lessons from real manuals, which are dominated by tables and page furniture:
1. `strip_repeated_lines` removes running banners (short lines on ≥⅓ of pages) — otherwise "NINJA AIR FRYER" on every page makes all pages match appliance queries equally
2. Line-aware chunking with trailing overlap (not paragraph-based — chart pages have no blank lines and get cut mid-row otherwise)
3. Every chunk is prefixed with the page heading (`[Air Fry Cooking Chart]`)
4. Each detected table row (PyMuPDF `find_tables(strategy="lines")`) is additionally indexed as its own one-line chunk — this is what makes "sausages cooking time" hit `Sausages | 8 (410g) | ... | 200°C | 10-13 mins` at rank 1. Tables are gated on ≥3 multi-cell rows so prose misdetected as a table isn't double-indexed.

Embeddings degrade gracefully: if fastembed can't load, everything runs keyword-only (`/health` reports which mode).

The web UI is one self-contained file ([app/static/index.html](app/static/index.html)), vanilla JS, no build step.
