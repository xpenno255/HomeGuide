"""HomeGuide: a refreshed, verified document catalogue for Assist."""
from homeassistant.helpers import llm
from .api import HomeGuideAPI
from .coordinator import HomeGuideCoordinator

PLATFORMS = ['sensor', 'conversation']


async def async_setup_entry(hass, entry):
    coordinator = HomeGuideCoordinator(hass, entry)
    await coordinator.async_config_entry_first_refresh()
    entry.runtime_data = coordinator
    entry.async_on_unload(llm.async_register_api(hass, HomeGuideAPI(hass, entry)))
    entry.async_on_unload(entry.add_update_listener(_async_update_listener))
    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    return True


async def async_unload_entry(hass, entry):
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)


async def _async_update_listener(hass, entry):
    await hass.config_entries.async_reload(entry.entry_id)
