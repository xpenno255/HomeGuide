"""HomeGuide HTTP client, shared by Assist, polling and the options flow."""
import aiohttp
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from .const import CONF_BASE_URL, REQUEST_TIMEOUT


class HomeGuideClient:
    def __init__(self, hass, config):
        self.session = async_get_clientsession(hass)
        self.base_url = config[CONF_BASE_URL].rstrip('/')

    async def request(self, method, path, **kwargs):
        async with self.session.request(method, self.base_url + path,
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT), **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def upload(self, filename, content, appliance_id, title, category):
        form = aiohttp.FormData()
        form.add_field('file', content, filename=filename)
        for key, value in {'appliance_id': appliance_id, 'verified': 'true',
                           'title': title, 'category': category}.items():
            form.add_field(key, value)
        return await self.request('POST', '/api/upload', data=form)
