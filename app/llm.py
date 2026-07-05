"""Optional answer generation for the web UI via any OpenAI-compatible server
(e.g. the same vLLM instance that serves your Home Assistant agent).

Configured with LLM_BASE_URL; disabled when unset. The /query endpoint used by
Home Assistant never touches this — it's for the UI's "Try a question" box.
"""

import logging
import os
import threading

import httpx

log = logging.getLogger("homeguide")

BASE_URL = os.environ.get("LLM_BASE_URL", "").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "")
API_KEY = os.environ.get("LLM_API_KEY", "not-needed")
TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "60"))

SYSTEM_PROMPT = (
    "You are a household voice assistant with access to excerpts from the home's "
    "document library (appliance manuals, warranties, house documents). Answer the "
    "user's question using ONLY the provided excerpts. Mention which document and "
    "page the answer comes from. If the excerpts do not contain the answer, say the "
    "document library does not cover it — never guess. Keep answers short and plain: "
    "they are read aloud."
)

_resolve_lock = threading.Lock()
_resolved_model: str | None = None


def enabled() -> bool:
    return bool(BASE_URL)


def _headers() -> dict:
    return {"Authorization": f"Bearer {API_KEY}"}


def _model() -> str:
    """Use LLM_MODEL if set, else auto-detect the server's first model (vLLM serves one)."""
    global _resolved_model
    if MODEL:
        return MODEL
    if _resolved_model:
        return _resolved_model
    with _resolve_lock:
        if _resolved_model:
            return _resolved_model
        resp = httpx.get(f"{BASE_URL}/models", headers=_headers(), timeout=10)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if not models:
            raise RuntimeError("LLM server reports no models")
        _resolved_model = models[0]["id"]
        log.info("Auto-detected LLM model: %s", _resolved_model)
        return _resolved_model


def ask(question: str, results: list[dict]) -> str:
    excerpts = "\n\n".join(
        f"[{i}] {r['document']} ({r['category']}), page {r['page']}:\n{r['excerpt']}"
        for i, r in enumerate(results, start=1)
    )
    user_msg = f"Document library excerpts:\n\n{excerpts}\n\nQuestion: {question}"

    resp = httpx.post(
        f"{BASE_URL}/chat/completions",
        headers=_headers(),
        timeout=TIMEOUT,
        json={
            "model": _model(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.2,
            "max_tokens": 400,
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
