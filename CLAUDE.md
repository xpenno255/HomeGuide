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

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest                     # fast: stubbed embeddings, no model download
```

`pytest` runs in under a second because embeddings are stubbed with exactly controllable cosine scores — those tests cover retrieval *mechanics* (the similarity floor, the DF filter, the corroboration rule, fusion, the API contract). CI runs exactly this.

Whether the real model actually separates good answers from bad is a corpus question, and lives in `tests/test_retrieval_quality.py`. It is skipped unless you point it at a real manual, and **it is the check that matters after any ingestion or retrieval change**:

```bash
HOMEGUIDE_TEST_PDF=~/path/AF500UK_IG.pdf .venv/bin/pytest tests/test_retrieval_quality.py
```

For manual poking, the API flow is:

```bash
curl -X POST localhost:8490/api/upload -F file=@manual.pdf -F "title=..." -F "category=manual"
curl "localhost:8490/query?q=air%20fryer%20chicken%20breast%20cooking%20time&k=5"
curl -X POST localhost:8490/api/documents/1/reindex   # after an ingestion change
```

The three retrieval behaviours that must keep passing are asserted in `tests/test_retrieval_quality.py` — chart *rows* at rank 1 for cooking questions, exact fault codes found (including a bare "E4"), and silence for questions the library cannot answer. Read that file rather than trusting a list here.

Two things it is easy to get wrong about the corpus:
- `AF500UK_IG.pdf` is the Ninja **Inspiration Guide** — a recipe book. It has cooking charts but no troubleshooting section, no fault codes and no care instructions, so fault-code cases use a separate inline fixture, not this PDF.
- One `xfail` is deliberate: "how do i clean the air fryer" returns recipe chunks that clear the cosine floor legitimately (~0.75), because the query shares heavy vocabulary with the corpus while correct chart answers only reach 0.78-0.82. The margin is too thin to threshold. Expect it to pass once the real AF500 *instruction* manual is indexed — at which point delete the xfail, don't tighten `MIN_SIM`.

`/api/ask` is covered by tests with `llm.ask` monkeypatched. To exercise it against real inference, set `LLM_BASE_URL` to an OpenAI-compatible server. Do NOT point it at the Ollama on 192.168.1.102:11434 — it is deliberately dormant (vLLM holds the GPU) and completions hang forever.

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

**Ingestion** ([app/ingest.py](app/ingest.py)) encodes the lessons from real manuals, which are dominated by tables, page furniture and other languages:

00. `_is_probably_english` skips pages that are confidently in another language. EU manuals ship every language in one PDF — the Sage brewer manual is 156 pages of which ~19 are English. Indexing the rest let the agent quote Portuguese back, *and* broke `strip_repeated_lines`, whose threshold scales with page count: at 156 pages it needed 52 repeats while the LCD legend (which contains the word `DESCALE`) repeated on ~23. Pages with few function words either way are kept — dropping a real English page is worse than keeping a foreign label list — and a wholly non-English manual is still indexed. Page numbers stay the document's real numbers so citations survive the skipping.

0. `_page_text` reads each page in **column order**, not PyMuPDF's default block order. Manuals are printed as two-page spreads: the AF500UK manual's page 10 is `CLEANING & MAINTENANCE` in the left half and `TROUBLESHOOTING GUIDE` in the right, and the default order emitted the entire troubleshooting column first — so every cleaning instruction was chunked away from its own heading and labelled `[TROUBLESHOOTING GUIDE]`. Columns are found from the vertical whitespace gutters between blocks (blocks wider than 60% of the page can't define a gutter, so footers don't mask it); a page with no wide-enough gutter falls back to default extraction untouched. Before this, "food stuck on the crisper plate" lost to recipe prose even though the manual answers it verbatim.
   > **Do not add mid-page section-heading detection.** Two independent attempts have been measured and reverted:
   > 1. *All-caps lines* — read recipe furniture (`COOK TIME: 22 MINUTES | MAKES: 4 SERVINGS`) as headings and knocked the cooking charts off rank 1.
   > 2. *Font size above the page's body size* — discriminates correctly on paper (real headings are 10-20pt against 7-8pt bodies, and it does reject the recipe furniture above), but still regressed retrieval: `sausages cooking time` returned Toad in the Hole prose instead of the chart row, and the soapy-water answer dropped from rank 1 to rank 2.
   >
   > The reason both fail is that the chunk label is not cosmetic — it is prepended to the chunk text, so it is indexed by FTS *and* embedded. Any change to labelling perturbs ranking across the whole library, and "first heading on the page" happens to be well tuned for the cooking charts. A third attempt needs to be judged on retrieval measurements, never on how correct the labels look.
   >
   > The remaining symptom is cosmetic: Sage p14 stacks REPLACING THE WATER FILTER and DESCALING in one column, so descaling chunks carry the wrong label. The descaling text is still *inside* those chunks, and the LLM answers the question correctly, which is why this is low priority.

1. `strip_repeated_lines` removes running banners (short lines on ≥⅓ of pages) — otherwise "NINJA AIR FRYER" on every page makes all pages match appliance queries equally
2. Line-aware chunking with trailing overlap (not paragraph-based — chart pages have no blank lines and get cut mid-row otherwise)
3. Every chunk is prefixed with the page heading (`[Air Fry Cooking Chart]`)
4. Each detected table row (PyMuPDF `find_tables(strategy="lines")`) is additionally indexed as its own one-line chunk — this is what makes "sausages cooking time" hit `Sausages | 8 (410g) | ... | 200°C | 10-13 mins` at rank 1. Tables are gated on ≥3 multi-cell rows so prose misdetected as a table isn't double-indexed.

Embeddings degrade gracefully: if fastembed can't load, everything runs keyword-only (`/health` reports which mode).

The web UI is one self-contained file ([app/static/index.html](app/static/index.html)), vanilla JS, no build step.
