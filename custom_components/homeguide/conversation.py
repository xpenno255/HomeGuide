"""Optional Assist front-end: reviewed answers bypass generation entirely."""
import re

import aiohttp
from homeassistant.components import conversation
from homeassistant.helpers import intent
from homeassistant.util import ulid


async def async_setup_entry(hass, entry, async_add_entities):
    async_add_entities([HomeGuideConversation(entry)])


class HomeGuideConversation(conversation.ConversationEntity):
    _attr_has_entity_name = False
    _attr_name = 'HomeGuide Assist'
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry):
        self.entry = entry
        self._attr_unique_id = f'{entry.entry_id}_assist'

    @property
    def supported_languages(self):
        return '*'

    async def async_process(self, user_input):
        """No model call for a verified procedure; preserve other agent behaviour.

        This entity deliberately does not edit another integration's chat log.
        Reviewed requests are self-contained; follow-ups must name the appliance
        and quantity. Delegated conversations retain their original session ID.
        """
        coordinator = self.entry.runtime_data
        result = None
        text = user_input.text.casefold()
        candidate = bool(re.search(r'\bdefrost(?:ing)?\b', text) and re.search(r'\bminc(?:e|ed)\b', text)
                         and not re.search(r'\b(?:turn|switch|lights?|set|timer|alarm)\b', text))
        if candidate and user_input.language.split('-')[0] == 'en':
            try:
                result = await coordinator.client.request('POST', '/api/resolve',
                                                          json={'question': user_input.text})
            except (aiohttp.ClientError, TimeoutError, ValueError):
                result = {'status': 'unavailable', 'answer': 'The document library could not be reached, so I cannot verify the defrost settings.'}
        if result and result.get('status') in {'matched', 'needs_weight', 'outside_supported_weight_range', 'unavailable'} and result.get('answer'):
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_speech(result['answer'])
            return conversation.ConversationResult(response=response,
                conversation_id=user_input.conversation_id or ulid.ulid_now())

        delegate_id = self.entry.options.get('fallback_agent')
        delegate = conversation.async_get_agent(self.hass, delegate_id) if delegate_id else None
        # Reject any HomeGuide instance, including a renamed entity, to avoid cycles.
        if delegate is not None and not isinstance(delegate, HomeGuideConversation):
            return await conversation.async_converse(self.hass, text=user_input.text,
                conversation_id=user_input.conversation_id, context=user_input.context,
                language=user_input.language, agent_id=delegate_id,
                device_id=user_input.device_id, satellite_id=user_input.satellite_id,
                extra_system_prompt=user_input.extra_system_prompt)
        response = intent.IntentResponse(language=user_input.language)
        response.async_set_speech('I could not find a reviewed answer. Select your existing conversation agent in the HomeGuide Assist options for other requests.')
        return conversation.ConversationResult(response=response,
            conversation_id=user_input.conversation_id or ulid.ulid_now())
