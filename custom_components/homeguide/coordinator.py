"""Refresh readiness and the stable model-facing catalogue once per minute."""
from datetime import timedelta
import aiohttp
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator, UpdateFailed
import logging
from .client import HomeGuideClient


class HomeGuideCoordinator(DataUpdateCoordinator):
    def __init__(self, hass, entry):
        super().__init__(hass, logging.getLogger(__name__), name='HomeGuide',
                         config_entry=entry, update_interval=timedelta(seconds=60),
                         always_update=False)
        self.client = HomeGuideClient(hass, {**entry.data, **entry.options})

    async def _async_update_data(self):
        try:
            catalog = await self.client.request('GET', '/api/catalog')
            documents = await self.client.request('GET', '/api/documents')
            return {'catalog': catalog, 'documents': documents['documents']}
        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError) as exc:
            raise UpdateFailed(f'HomeGuide catalogue unavailable: {exc}') from exc
