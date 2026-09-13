"""Config flow for HomeGuide: point HA at the HomeGuide container."""

import logging
from typing import Any

import aiohttp
import voluptuous as vol
from homeassistant.config_entries import ConfigFlow, ConfigFlowResult, OptionsFlow
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .const import CONF_BASE_URL, CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS, DOMAIN

_LOGGER = logging.getLogger(__name__)

STEP_USER_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_BASE_URL): str,
        vol.Optional(CONF_NUM_RESULTS, default=DEFAULT_NUM_RESULTS): vol.All(
            vol.Coerce(int), vol.Range(min=1, max=10)
        ),
    }
)


async def _validate(hass: HomeAssistant, base_url: str) -> dict[str, Any]:
    """Check the URL points at a live HomeGuide instance; return its stats."""
    session = async_get_clientsession(hass)
    resp = await session.get(
        f"{base_url.rstrip('/')}/health", timeout=aiohttp.ClientTimeout(total=10)
    )
    resp.raise_for_status()
    health = await resp.json()
    if health.get("status") != "ok":
        raise CannotConnect
    return health


class CannotConnect(Exception):
    """The URL did not answer like a HomeGuide instance."""


class HomeGuideConfigFlow(ConfigFlow, domain=DOMAIN):
    """Single-instance flow: just the base URL and an excerpt budget."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        errors: dict[str, str] = {}
        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].rstrip("/")
            try:
                health = await _validate(self.hass, base_url)
            except (aiohttp.ClientError, TimeoutError):
                errors["base_url"] = "cannot_connect"
            except CannotConnect:
                errors["base_url"] = "not_homeguide"
            else:
                return self.async_create_entry(
                    title=f"HomeGuide ({health.get('documents', '?')} documents)",
                    data={CONF_BASE_URL: base_url},
                    options={
                        CONF_NUM_RESULTS: user_input.get(
                            CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS
                        )
                    },
                )

        return self.async_show_form(
            step_id="user", data_schema=STEP_USER_SCHEMA, errors=errors
        )

    @staticmethod
    def async_get_options_flow(config_entry) -> "HomeGuideOptionsFlow":
        """Return the options flow."""
        return HomeGuideOptionsFlow()


class HomeGuideOptionsFlow(OptionsFlow):
    """Manage the connection, appliances and confirmed guides from HA."""

    @property
    def client(self):
        from .client import HomeGuideClient
        return HomeGuideClient(self.hass, {**self.config_entry.data, **self.config_entry.options})

    async def _finish(self, changes=None):
        return self.async_create_entry(data={**self.config_entry.options, **(changes or {})})

    async def async_step_init(self, user_input=None):
        return self.async_show_menu(step_id='init', menu_options=[
            'connection', 'assist', 'appliance', 'upload', 'link', 'refresh'])

    async def async_step_connection(self, user_input=None):
        errors = {}
        if user_input is not None:
            try:
                await _validate(self.hass, user_input[CONF_BASE_URL])
                from .client import HomeGuideClient
                await HomeGuideClient(self.hass, user_input).request('GET', '/api/catalog')
            except (aiohttp.ClientError, TimeoutError, CannotConnect, ValueError) as exc:
                _LOGGER.warning("HomeGuide connection validation failed: %s", exc)
                errors['base'] = 'cannot_connect'
            else:
                user_input[CONF_BASE_URL] = user_input[CONF_BASE_URL].rstrip('/')
                return await self._finish(user_input)
        config = {**self.config_entry.data, **self.config_entry.options}
        return self.async_show_form(step_id='connection', errors=errors, data_schema=vol.Schema({
            vol.Required(CONF_BASE_URL, default=config[CONF_BASE_URL]): str,
            vol.Required(CONF_NUM_RESULTS, default=config.get(CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS)):
                vol.All(vol.Coerce(int), vol.Range(min=1, max=10))}))

    async def async_step_assist(self, user_input=None):
        from homeassistant.components import conversation
        from homeassistant.helpers.selector import EntitySelector, EntitySelectorConfig
        from .conversation import HomeGuideConversation
        errors = {}
        if user_input is not None:
            agent = conversation.async_get_agent(self.hass, user_input['fallback_agent'])
            if agent is None or isinstance(agent, HomeGuideConversation):
                errors['base'] = 'invalid_agent'
            else:
                return await self._finish(user_input)
        return self.async_show_form(step_id='assist', errors=errors, data_schema=vol.Schema({
            vol.Required('fallback_agent', description={'suggested_value': self.config_entry.options.get('fallback_agent', '')}):
                EntitySelector(EntitySelectorConfig(domain='conversation'))}))

    async def async_step_appliance(self, user_input=None):
        errors = {}
        try:
            library = await self.client.request('GET', '/api/appliances')
            if user_input is not None:
                data = {**user_input, 'aliases': [a.strip() for a in user_input.get('aliases', '').split(',') if a.strip()]}
                await self.client.request('POST', '/api/appliances', json=data)
                return await self._finish()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            if user_input is None:
                return self.async_abort(reason='cannot_connect')
            errors['base'] = 'save_failed'
            library = {'kinds': {'other': []}}
        return self.async_show_form(step_id='appliance', errors=errors, data_schema=vol.Schema({
            vol.Required('name'): str,
            vol.Required('kind', default='other'): vol.In({k: k.replace('_', ' ').title() for k in library['kinds']}),
            vol.Optional('manufacturer', default=''): str,
            vol.Optional('model', default=''): str,
            vol.Optional('region', default=''): str,
            vol.Optional('aliases', default=''): str}))

    async def _appliances(self):
        library = await self.client.request('GET', '/api/appliances')
        return {a['id']: ' · '.join(v for v in (a['name'], a['manufacturer'], a['model'], a['region']) if v)
                for a in library['appliances']}

    async def async_step_upload(self, user_input=None):
        from homeassistant.helpers.selector import FileSelector, FileSelectorConfig
        errors = {}
        try:
            appliances = await self._appliances()
            if not appliances:
                return self.async_abort(reason='no_appliances')
            if user_input is not None:
                if not user_input.get('confirmed'):
                    errors['confirmed'] = 'confirmation_required'
                else:
                    filename, content = await self.hass.async_add_executor_job(
                        _read_uploaded_guide, self.hass, user_input['file'])
                    await self.client.upload(filename, content, user_input['appliance_id'],
                                             user_input.get('title', ''), 'manual')
                    return await self._finish()
        except (aiohttp.ClientError, TimeoutError, ValueError, OSError):
            # A consumed temporary upload must be selected again after a failure.
            if user_input is None:
                return self.async_abort(reason='cannot_connect')
            errors['base'] = 'upload_failed'
            appliances = await self._appliances()
        return self.async_show_form(step_id='upload', errors=errors, data_schema=vol.Schema({
            vol.Required('appliance_id'): vol.In(appliances),
            vol.Required('file'): FileSelector(FileSelectorConfig(accept='.pdf,.txt,.md')),
            vol.Optional('title', default=''): str,
            vol.Required('confirmed', default=False): bool}))

    async def async_step_link(self, user_input=None):
        errors = {}
        try:
            appliances = await self._appliances()
            if not appliances:
                return self.async_abort(reason='no_appliances')
            documents = (await self.client.request('GET', '/api/documents'))['documents']
            if user_input is not None:
                if not user_input.get('confirmed'):
                    errors['confirmed'] = 'confirmation_required'
                else:
                    await self.client.request('POST', f"/api/documents/{user_input['document_id']}/appliances",
                        json={'appliance_id': user_input['appliance_id'], 'verified': True})
                    return await self._finish()
        except (aiohttp.ClientError, TimeoutError, ValueError):
            return self.async_abort(reason='cannot_connect')
        return self.async_show_form(step_id='link', errors=errors, data_schema=vol.Schema({
            vol.Required('appliance_id'): vol.In(appliances),
            vol.Required('document_id'): vol.In({str(d['id']): f"{d['title']} ({d['status']})" for d in documents}),
            vol.Required('confirmed', default=False): bool}))

    async def async_step_refresh(self, user_input=None):
        await self.config_entry.runtime_data.async_request_refresh()
        return await self._finish()


def _read_uploaded_guide(hass, file_id):
    # process_uploaded_file deletes its temporary directory on exit; all file I/O
    # including cleanup belongs on the executor, not HA's event loop.
    from homeassistant.components.file_upload import process_uploaded_file
    with process_uploaded_file(hass, file_id) as path:
        if path.suffix.lower() not in {'.pdf', '.txt', '.md'}:
            raise ValueError('Unsupported guide format')
        return path.name, path.read_bytes()
