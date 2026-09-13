"""Constants for the HomeGuide integration."""

DOMAIN = "homeguide"

CONF_BASE_URL = "base_url"
CONF_NUM_RESULTS = "num_results"

DEFAULT_NUM_RESULTS = 5
REQUEST_TIMEOUT = 15  # seconds; /query is a local CPU search, normally <1s

API_NAME = "HomeGuide"

# Shown to the agent whenever the HomeGuide API is enabled. This is prompt
# engineering for small local models — the wording is what makes them reach
# for the tool instead of inventing appliance instructions; edit carefully.
API_PROMPT = (
    "The manual catalogue is separate from HA entities: an appliance does not need an HA entity to have a guide. Use only appliances in the tool’s supported-guides catalogue. Match the named appliance, model or alias, or a clearly established conversation referent. If none matches, explain that no guide is available and do not call the tool. Never substitute a different appliance. You have access to the household document library via the "
    "query_home_documents tool. Use it for manual instructions, explanations of "
    "appliance features or fault codes, maintenance, warranties and house paperwork. "
    "For a device's current state, mode or sensor reading, use Home Assistant's "
    "provided state instead; a manual cannot report live state. "
    "Search once per user question, then answer from the returned excerpts and cite "
    "the document and page. If results are empty, irrelevant or the service is "
    "unavailable, say the lookup could not answer the question and finish. "
    "Do not repeat or rephrase the lookup, or fall back to web search, unless the "
    "user explicitly requests another search. Never invent appliance instructions "
    "or claim that an unsuccessful search proves a manual lacks the information."
)

# Mirrors the tested description in homeassistant/query_home_documents.yaml.
TOOL_DESCRIPTION = (
    "Use the matching appliance_id for instructions, features, cooking, fault codes, "
    "cleaning or warranties. Match names and aliases; an exact model number is optional. "
    "Search once with the appliance, topic, quantities and exact error code. "
    "Answer from relevant excerpts with document/page. Empty or irrelevant results "
    "are terminal: say the lookup could not answer. Live states and controls use "
    "Home Assistant. If the requested appliance is absent from this list, say no "
    "guide is available without calling this tool."
)
