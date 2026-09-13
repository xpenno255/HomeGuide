# HomeGuide

![HomeGuide](custom_components/homeguide/brand/icon.png)


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

### Optional: generated answers in the web UI

Point HomeGuide at the same vLLM server your Home Assistant agent uses (or any
OpenAI-compatible endpoint) and the "Try a question" box runs the full loop —
search the library, then have the model answer from the excerpts — so you can
test end-to-end without going through a voice pipeline. Uncomment the
`environment:` block in [docker-compose.yml](docker-compose.yml) and set
`LLM_BASE_URL` (e.g. `http://<vllm-host>:8000/v1`). The model name is
auto-detected from `/v1/models`, so `LLM_MODEL` is only needed if your server
hosts several. This only affects the web UI — the `/query` endpoint Home
Assistant calls is unchanged.

> **Scanned PDFs:** HomeGuide extracts embedded text. A scanned/image-only manual will
> fail with "No extractable text" — run it through OCR first (e.g. `ocrmypdf in.pdf out.pdf`)
> and upload the result. Most manufacturer-download PDFs are fine as-is.

## 3. Wire up Home Assistant

### Option A: the HomeGuide integration (recommended)

Registers `query_home_documents` with Home Assistant's native LLM tool
framework, so it appears as a tickable **HomeGuide** API in every conversation
agent that supports tool selection (Extended OpenAI Conversation, the official
OpenAI/Ollama integrations, ...). One install covers all your agents, and the
tool-usage guidance is injected automatically — no prompt editing.

1. Install with [HACS](#hacs-and-github-releases), or copy [custom_components/homeguide](custom_components/homeguide) into your HA `config/custom_components/`, then restart Home Assistant.
2. **Settings → Devices & Services → Add Integration → HomeGuide**, enter your
   HomeGuide URL (e.g. `http://192.168.1.50:8480`).
3. In your conversation agent's options, tick **HomeGuide** in the LLM API /
   tools selector.

### Option B: Extended OpenAI Conversation function YAML

For EOC setups without LLM API selection:

1. Open **Settings → Devices & Services → Extended OpenAI Conversation → Configure**.
2. In the **Functions** field, append the contents of
   [homeassistant/query_home_documents.yaml](homeassistant/query_home_documents.yaml)
   to your existing function list.
3. Replace `HOMEGUIDE_HOST` with your Docker host's LAN IP.
4. Optionally add a line like this to your agent's prompt template, which noticeably
   improves how reliably a small model reaches for the tool:

   ```
   Use query_home_documents for appliance instructions, fault explanations,
   maintenance, warranties and paperwork. Read current states, modes and
   sensor readings from Home Assistant's provided state, not from manuals.
   Search once per user question and answer from relevant excerpts, citing
   document and page. If the search is empty, irrelevant or unavailable,
   say the lookup could not answer and finish. Do not retry, rephrase or
   search the web unless the user explicitly requests another search.
   Never invent instructions or claim a failed search proves a manual
   lacks the information.
   ```

Then ask your voice assistant something like *"how long do I cook chicken breast in
the air fryer?"* or *"what does E4 mean on the dishwasher?"*.

Replace any existing "ANY question about appliances/settings MUST search" rule
with this guidance; keeping both creates a conflict for current-mode questions.
Keep these fixed instructions before the device list and put the changing time
last to preserve prefix-cache reuse. The native tool description carries the same
stopping guidance because some conversation integrations do not include API-level
prompt text in the rendered model request.

These are model instructions, not an enforced conversation limit. A caller that
needs a hard limit must also bound its tool loop. HomeGuide's `/query` endpoint
performs one retrieval and cannot stop another component from calling it again.

## API

| Endpoint | Purpose |
|---|---|
| `GET /query?q=...&k=4&category=manual` | Search; `k` = excerpts (1–10), `category` optional. Also accepts `POST` with `{"query": "..."}`. |
| `GET /health` | Liveness, doc/chunk counts, search mode |
| `POST /api/upload` | Multipart: `file`, `title`, `category` |
| `GET /api/documents` | List library |
| `POST /api/documents/{id}/reindex` | Rebuild chunks and embeddings from the stored file |
| `DELETE /api/documents/{id}` | Remove a document |
| `GET /api/documents/{id}/file` | Original file |

`/query` returns at most ~3 KB of excerpt text by default (4 excerpts × 800 chars), sized
so it fits comfortably in a small model's context. Raise `k` in the `resource_template`
if your context budget allows.

Successful searches include `status: "matched"` and the existing `results` list.
An empty search returns `status: "no_match"`, `results: []`, `retryable: false`,
and a note explaining that this search did not find an answer. This is not proof
that the information is absent from the entire library. The native HA integration
also returns `status: "unavailable"` and `retryable: false` on connection failure.
Existing consumers reading only `results` remain compatible.

For recognised short fault codes such as E4 or F21, a returned chunk must contain
the exact code in its body. Similarity to generic "fault" text is insufficient.
This conservative check does not cover every manufacturer's code notation, and
matching a code alone does not prove it belongs to the user's appliance: check
the returned document title too. Search-time changes need no document reindex.

## Troubleshooting

- **Agent answers from its own knowledge instead of calling the tool** — strengthen the
  prompt line from step 4, and make sure the appliance name in the document title matches
  what you say out loud.
- **Agent says nothing was found but the UI search finds it** — check HA can reach the
  Docker host: `curl "http://HOMEGUIDE_HOST:8480/health"` from the HA machine.
- **Upload shows "error"** — hover the row for the reason; almost always a scanned PDF
  needing OCR (see above).
- **Retrieval didn't improve after upgrading the image** — chunks and embeddings are built
  at upload time, so indexing improvements only reach documents you already have once you
  hit **Reindex** on them. It rebuilds from the stored original; nothing is re-uploaded.
- **No auth by design** — this binds to your LAN with no authentication, like most
  homelab services. Don't expose port 8480 to the internet.

### Appliance catalogue and uploads from Home Assistant (0.2.1)

In **Settings → Devices & services → HomeGuide → Configure**, use:

- **Connection and search settings** to set the HomeGuide URL and excerpt count.
- **Add an appliance** to record its name, type, manufacturer, exact model, region and spoken aliases. Leave unknown model/region fields blank rather than guessing.
- **Upload a guide** to select a PDF, TXT or Markdown file and confirm that it covers the selected appliance. HA forwards the file to HomeGuide; indexing runs there.
- **Confirm an existing guide** to associate a document already in the library.
- **Refresh catalogue** to refresh immediately. Automatic refresh runs every 60 seconds while the integration is loaded. The **HomeGuide library** sensor lists document indexing states and the currently supported appliances.

Only active, ready documents with confirmed appliance associations appear in Assist's tool catalogue. Uploads still processing or failed, archived documents, and unconfirmed associations are excluded. Confirmation is a human assertion, not automatic model detection. Changing an appliance's type, manufacturer, model or region clears its guide confirmations. Names and aliases can change without invalidating guides.

The Assist tool now requires `appliance_id` and `query`. The advertised IDs and metadata are sorted and contain no changing timestamps. HomeGuide filters both keyword and vector candidates to that appliance's confirmed documents **before** ranking limits. Unknown IDs and clear appliance-type conflicts return terminal responses; a failed lookup never broadens into another appliance's manuals. If the catalogue cannot refresh, the integration stops exposing the lookup tool until it recovers.

The language model still chooses whether to call the tool. The integration does not receive the original utterance in HA 2026.9's `LLMContext`, so it cannot guarantee that an unsupported question never causes an attempted call. The tool schema, instructions and backend scope prevent unsupported IDs from retrieving unrelated guides. Keep the conversation system prompt consistent: use a guide only for a supported matching appliance, use HA state for current readings, and do not require a manual lookup for every appliance-related question.

Catalogue API (existing unscoped `/query` and web uploads remain compatible):

| Endpoint | Purpose |
|---|---|
| `GET /api/catalog` | Stable ready catalogue and content revision |
| `GET /api/appliances` | All appliances, guide associations and supported types |
| `POST /api/appliances` | Create `{name, kind, manufacturer, model, region, aliases}` |
| `PUT /api/appliances/{id}` | Replace metadata; identity changes require reconfirmation |
| `POST /api/documents/{id}/appliances` | Associate `{appliance_id, verified: true/false}` |
| `PATCH /api/documents/{id}` | Archive/reactivate with `{active: false/true}` |
| `POST /api/upload` | Existing multipart upload plus `appliance_id` and `verified` |
| `GET /query?q=...&appliance_id=...` | Enforced appliance-scoped retrieval |

Existing documents are preserved on migration but are not automatically declared verified. The HomeGuide web upload remains usable; confirm its appliance association from HA afterward. Legacy REST-function YAML clients must be updated separately; install the HA LLM API integration to get the automatically refreshed catalogue.

Integration compatibility tests run in a separate environment with Home Assistant installed:

```bash
python3.14 -m venv /tmp/homeguide-ha-test
/tmp/homeguide-ha-test/bin/pip install -r requirements-ha-tests.txt
/tmp/homeguide-ha-test/bin/pytest homeassistant/tests
```

For Extended OpenAI Conversation versions that omit an LLM API's `api_prompt`, insert [the catalogue template](homeassistant/manual_catalogue_prompt.jinja) into the conversation system prompt before `Available Devices` and its CSV/time fields. This reads the integration's refreshed sensor, so supported names, aliases and model/region metadata update automatically. The catalogue must be treated separately from controllable HA entities: a manual can exist for an appliance with no HA entity. The tool description also contains the catalogue for agents that use native LLM APIs. Use the actual library sensor entity ID if HA gave it a different name.


## HACS and GitHub releases

Requires **Home Assistant 2026.9 or newer**. In HACS, open the menu → **Custom
repositories**, enter `https://github.com/xpenno255/HomeGuide`, and choose
**Integration**. Open HomeGuide, download the latest release, and restart Home
Assistant. Then add HomeGuide under **Settings → Devices & services** and enter
your backend URL (for this installation: `https://homeguide.xpennohome.uk`).
Existing manually installed HomeGuide entries can be adopted by HACS: install to
the same `custom_components/homeguide` location and restart; do not create a
second config entry.

HACS downloads the integration from the tagged `custom_components/homeguide`
tree. GitHub releases also include `homeguide.zip` for manual installation;
extract it into your HA config directory. This repository is usable as a HACS
custom repository; it is not submitted to the default HACS catalogue.

HACS installs the **HA integration only**. The HomeGuide backend remains a
separate Docker service. Update it to the matching release, for example
`ghcr.io/xpenno255/homeguide:0.3.0`, retaining its data volume. Release tags must
match the integration manifest version. GitHub Actions validates HACS and HA
metadata, runs integration tests, publishes a release ZIP and builds the backend
image. See the [HACS requirements](https://www.hacs.xyz/docs/publish/integration/)
and [release behaviour](https://www.hacs.xyz/docs/publish/start/).

## Reviewed microwave answers (0.3.0)

The exact reviewed NN-ST46KB UK manual can return source-backed Chaos Defrost
controls and weight checks. The optional **HomeGuide Assist** agent speaks those
answers directly and forwards other requests to your existing agent. Configure
its delegate in HomeGuide options, then select HomeGuide Assist in your voice
assistant settings. This avoids the measured empty-answer problem for reviewed
procedures. Details, supported questions and source checks are in
[Reviewed procedures](docs/reviewed-procedures.md).
