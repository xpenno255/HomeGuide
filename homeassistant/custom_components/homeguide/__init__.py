"""The HomeGuide integration: exposes the document library to Assist agents."""

import logging

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm

from .api import HomeGuideAPI
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Register the HomeGuide LLM API."""
    # async_on_unload runs the unregister callback on unload AND on reload,
    # so a reconfigure never trips "API homeguide is already registered".
    entry.async_on_unload(llm.async_register_api(hass, HomeGuideAPI(hass)))
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    _LOGGER.info("HomeGuide LLM API registered (%s)", entry.data.get("base_url"))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload the entry; unregistration happens via async_on_unload."""
    return True


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload on options change so the new URL/k take effect immediately."""
    await hass.config_entries.async_reload(entry.entry_id)
