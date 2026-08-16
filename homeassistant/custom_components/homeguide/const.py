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
    "You have access to the household document library via the "
    "query_home_documents tool. For any question about appliances, manuals, "
    "cooking times or temperatures, error or fault codes, cleaning or "
    "maintenance, spare parts, warranties or house paperwork, call it before "
    "answering, and cite the document and page the answer came from. If it "
    "returns no results, say the information is not in the document library. "
    "Never invent appliance instructions."
)

# Mirrors the tested description in homeassistant/query_home_documents.yaml.
TOOL_DESCRIPTION = (
    "Search the household document library. It contains appliance user "
    "manuals, warranties and other house documents. Use this whenever the "
    "user asks about how an appliance works, cooking times or temperatures, "
    "program or setting names, error or fault codes, cleaning or descaling "
    "instructions, maintenance, spare parts, warranty coverage, or anything "
    "else that would be written in a manual or household document."
)
