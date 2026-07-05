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

Retrieval regression cases that must keep passing (verified against the real Ninja AF500UK guide):
- Cooking-chart queries ("chicken breast cooking time", "sausages cooking time") return the chart *row* at rank 1, not recipe prose
- Exact fault codes ("what does E4 mean") return the troubleshooting chunk first
- Irrelevant queries ("lawnmower blade replacement") return zero results — the agent is told to say "not in the library" rather than guess

To test the LLM answer path (`/api/ask`) without real inference, run a stub OpenAI-compatible server (see git history for `/tmp/fake_vllm.py`) and set `LLM_BASE_URL` to it. Do NOT test against the Ollama on 192.168.1.102:11434 — it is deliberately dormant (vLLM holds the GPU) and completions hang forever.

## Deployment

Push to `main` → GitHub Actions builds `ghcr.io/xpenno255/homeguide:latest` (public, no auth to pull) → user runs `docker compose pull && up -d` on the Docker host. **Chunking/ingestion changes only take effect after documents are deleted and re-uploaded** — chunks and embeddings are computed at upload time and stored in SQLite.

## Architecture

Single FastAPI app ([app/main.py](app/main.py)), single SQLite DB (one shared connection + `db.lock`, WAL mode). Two consumer-facing paths:

- `GET /query` — called by Home Assistant. Response size is budgeted for a small model's context (k excerpts × 800 chars); don't inflate it.
- `POST /api/ask` — web-UI only. Same search, then answers via any OpenAI-compatible server (`LLM_BASE_URL`, model auto-detected from `/v1/models`). Never used by HA.

**Search** ([app/search.py](app/search.py)) is hybrid, fused with RRF: FTS5 BM25 (porter tokenizer, stopwords stripped from the query) + cosine over fastembed vectors (bge-small-en-v1.5, CPU/ONNX — the GPU belongs to vLLM). All chunk embeddings are cached as one in-memory numpy matrix; `search.invalidate_cache()` must be called after any chunk mutation. Two tuned constants embody real failures: `MIN_SIM = 0.60` (below it, vector hits are noise — measured ~0.75+ relevant vs <0.55 unrelated) and vector-before-FTS ordering in RRF fusion (tie-breaks favor the calibrated retriever).

**Ingestion** ([app/ingest.py](app/ingest.py)) encodes the lessons from real manuals, which are dominated by tables and page furniture:
1. `strip_repeated_lines` removes running banners (short lines on ≥⅓ of pages) — otherwise "NINJA AIR FRYER" on every page makes all pages match appliance queries equally
2. Line-aware chunking with trailing overlap (not paragraph-based — chart pages have no blank lines and get cut mid-row otherwise)
3. Every chunk is prefixed with the page heading (`[Air Fry Cooking Chart]`)
4. Each detected table row (PyMuPDF `find_tables(strategy="lines")`) is additionally indexed as its own one-line chunk — this is what makes "sausages cooking time" hit `Sausages | 8 (410g) | ... | 200°C | 10-13 mins` at rank 1. Tables are gated on ≥3 multi-cell rows so prose misdetected as a table isn't double-indexed.

Embeddings degrade gracefully: if fastembed can't load, everything runs keyword-only (`/health` reports which mode).

The web UI is one self-contained file ([app/static/index.html](app/static/index.html)), vanilla JS, no build step.
