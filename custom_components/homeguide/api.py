"""Assist tool with a catalogue snapshot and server-enforced document scope."""
import aiohttp
import voluptuous as vol
from homeassistant.helpers import llm
from .const import API_NAME, API_PROMPT, TOOL_DESCRIPTION, DOMAIN, CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS


def terminal(status, note):
    return {'status': status, 'results': [], 'retryable': False, 'note': note}


class QueryHomeDocumentsTool(llm.Tool):
    name = 'query_home_documents'

    def __init__(self, entry, appliances):
        self.entry = entry
        self.ids = {a['id'] for a in appliances}
        # Include the catalogue here too: some third-party agents omit api_prompt.
        lines = []
        for appliance in appliances:
            identity = ' '.join(appliance[k] for k in ('manufacturer', 'model', 'region') if appliance[k])
            aliases = ', '.join(appliance['aliases'])
            lines.append(f"{appliance['id']}: {appliance['name']}; {identity}; also called {aliases}.")
        self.description = 'Look up the uploaded manuals for these supported appliances:\n' + '\n'.join(lines) + '\n' + TOOL_DESCRIPTION
        self.parameters = vol.Schema({
            vol.Required('appliance_id', description='Select the matching supported appliance ID. If none matches, do not search; unsupported is a safe fallback.'): vol.In(sorted(self.ids) + ['unsupported']),
            vol.Required('query', description='Concise English manual question with the appliance, exact fault code, quantities and relevant feature.'): str,
        })

    async def async_call(self, hass, tool_input, llm_context):
        args = tool_input.tool_args
        coordinator = self.entry.runtime_data
        current_ids = {a['id'] for a in coordinator.data['catalog']['appliances']}
        if not coordinator.last_update_success:
            return terminal('unavailable', 'The guide catalogue is unavailable. Tell the user and finish.')
        if args['appliance_id'] not in self.ids & current_ids:
            return terminal('unsupported_appliance', 'No confirmed ready guide matches this appliance. Tell the user and finish.')
        config = {**self.entry.data, **self.entry.options}
        try:
            return await coordinator.client.request('GET', '/query', params={
                'q': args['query'], 'appliance_id': args['appliance_id'],
                'k': int(config.get(CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS))})
        except (aiohttp.ClientError, TimeoutError, ValueError):
            return terminal('unavailable', 'The document library could not be reached. Tell the user and finish.')


class HomeGuideAPI(llm.API):
    def __init__(self, hass, entry):
        super().__init__(hass=hass, id=DOMAIN, name=API_NAME)
        self.entry = entry

    async def async_get_api_instance(self, llm_context):
        coordinator = self.entry.runtime_data
        appliances = coordinator.data['catalog']['appliances'] if coordinator.last_update_success else []
        return llm.APIInstance(api=self, api_prompt=API_PROMPT, llm_context=llm_context,
                               tools=[QueryHomeDocumentsTool(self.entry, appliances)] if appliances else [])
