"""LLM API exposing the HomeGuide document library as an Assist tool.

Registers a "HomeGuide" API with Home Assistant's LLM framework, so any
conversation agent that supports LLM API selection (the built-in agents,
Extended OpenAI Conversation, Ollama, ...) can tick it and gain the
query_home_documents tool — no per-agent function YAML needed.
"""

import logging
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.util.json import JsonObjectType

from .const import (
    API_NAME,
    API_PROMPT,
    CONF_BASE_URL,
    CONF_NUM_RESULTS,
    DEFAULT_NUM_RESULTS,
    DOMAIN,
    REQUEST_TIMEOUT,
    TOOL_DESCRIPTION,
)

_LOGGER = logging.getLogger(__name__)


def _config(hass: HomeAssistant) -> dict[str, Any]:
    """Effective config: entry data overlaid with any options."""
    entry = next(iter(hass.config_entries.async_entries(DOMAIN)))
    return {**entry.data, **(entry.options or {})}


class QueryHomeDocumentsTool(llm.Tool):
    """Search the HomeGuide library and hand the excerpts to the agent."""

    name = "query_home_documents"
    description = TOOL_DESCRIPTION

    parameters = vol.Schema(
        {
            vol.Required(
                "query",
                description=(
                    "A concise search query in English. Include the appliance "
                    "name and the key terms, e.g. 'air fryer chicken cooking "
                    "time' or 'dishwasher fault code E4'."
                ),
            ): str,
        }
    )

    async def async_call(
        self,
        hass: HomeAssistant,
        tool_input: llm.ToolInput,
        llm_context: llm.LLMContext,
    ) -> JsonObjectType:
        """Call HomeGuide's /query endpoint.

        Errors are returned as data rather than raised: a small model handles
        {"error": ...} far better than a tool-call exception, and can tell the
        user the library is unreachable instead of stalling.
        """
        config = _config(hass)
        base_url = str(config[CONF_BASE_URL]).rstrip("/")
        query = tool_input.tool_args["query"]
        _LOGGER.debug("HomeGuide query: %s", query)

        session = async_get_clientsession(hass)
        try:
            resp = await session.get(
                f"{base_url}/query",
                params={
                    "q": query,
                    "k": int(config.get(CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS)),
                },
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
            )
            resp.raise_for_status()
            return await resp.json()
        except (aiohttp.ClientError, TimeoutError) as exc:
            _LOGGER.warning("HomeGuide unreachable at %s: %s", base_url, exc)
            return {
                "error": (
                    "The document library service could not be reached. "
                    "Tell the user the household document library is "
                    "currently unavailable."
                )
            }


class HomeGuideAPI(llm.API):
    """The HomeGuide LLM API — one tool, plus usage guidance for the agent."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialise with a stable id so agent selections survive restarts."""
        super().__init__(hass=hass, id=DOMAIN, name=API_NAME)

    async def async_get_api_instance(
        self, llm_context: llm.LLMContext
    ) -> llm.APIInstance:
        """Return the tool set for one conversation turn."""
        return llm.APIInstance(
            api=self,
            api_prompt=API_PROMPT,
            llm_context=llm_context,
            tools=[QueryHomeDocumentsTool()],
        )
