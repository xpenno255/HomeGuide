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
    """Let the excerpt budget be changed without re-adding the integration."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle options."""
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(
                {
                    vol.Optional(
                        CONF_NUM_RESULTS,
                        default=self.config_entry.options.get(
                            CONF_NUM_RESULTS, DEFAULT_NUM_RESULTS
                        ),
                    ): vol.All(vol.Coerce(int), vol.Range(min=1, max=10)),
                }
            ),
        )
