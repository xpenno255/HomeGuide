"""Runs against the installed Home Assistant 2026.9.2 classes."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
import homeassistant
import voluptuous as vol
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from custom_components.homeguide.api import QueryHomeDocumentsTool, HomeGuideAPI
from custom_components.homeguide.config_flow import HomeGuideOptionsFlow, _read_uploaded_guide
from homeassistant.core import HomeAssistant
from homeassistant.helpers.llm import ToolInput, LLMContext

APPLIANCE = dict(id='microwave', name='Kitchen microwave', kind='microwave', manufacturer='Panasonic', model='NN-ST46KB', region='UK', aliases=['Panasonic'], document_ids=[10])


def entry():
    client = SimpleNamespace(request=AsyncMock(return_value={'status':'matched','results':[]}))
    coordinator = SimpleNamespace(data={'catalog':{'appliances':[APPLIANCE]}}, last_update_success=True, client=client)
    return SimpleNamespace(data={'base_url':'https://homeguide.example'}, options={'num_results':5}, runtime_data=coordinator)


@pytest.mark.asyncio
async def test_scope_validation_refresh_and_stale_catalogue():
    e=entry(); tool=QueryHomeDocumentsTool(e,[APPLIANCE])
    with pytest.raises(vol.Invalid):
        tool.parameters({'query':'defrost'})
    with pytest.raises(vol.Invalid):
        tool.parameters({'query':'defrost','appliance_id':'dishwasher'})
    args=ToolInput(tool.name, {'query':'microwave defrost 500g', 'appliance_id':'microwave'})
    result=await tool.async_call(None,args,None)
    assert result['status']=='matched'
    assert e.runtime_data.client.request.call_args.kwargs['params']['appliance_id']=='microwave'
    e.runtime_data.client.request.reset_mock()
    e.runtime_data.data['catalog']['appliances']=[]
    assert (await tool.async_call(None,args,None))['status']=='unsupported_appliance'
    e.runtime_data.client.request.assert_not_awaited()
    e.runtime_data.last_update_success=False
    assert (await tool.async_call(None,args,None))['status']=='unavailable'


@pytest.mark.asyncio
async def test_catalogue_stable_and_unavailable_removes_tool(tmp_path):
    hass=HomeAssistant(str(tmp_path));e=entry()
    context=LLMContext('test',None,'en','conversation',None)
    api=HomeGuideAPI(hass,e)
    first=await api.async_get_api_instance(context)
    second=await api.async_get_api_instance(context)
    assert first.tools[0].description==second.tools[0].description
    assert 'NN-ST46KB' in first.tools[0].description
    e.runtime_data.last_update_success=False
    assert (await api.async_get_api_instance(context)).tools==[]


@pytest.mark.asyncio
async def test_options_menus_and_preserving_connection(monkeypatch):
    e=entry()
    monkeypatch.setattr(HomeGuideOptionsFlow, 'config_entry', property(lambda self:e))
    flow=HomeGuideOptionsFlow()
    menu=await flow.async_step_init()
    assert set(menu['menu_options'])=={'connection','assist','appliance','upload','link','refresh'}
    result=await flow._finish({'base_url':'https://new.example'})
    assert result['data']=={'num_results':5,'base_url':'https://new.example'}


@pytest.mark.asyncio
async def test_uploaded_file_read_and_cleanup_on_executor(tmp_path):
    from homeassistant.components.file_upload import FileUploadData
    hass=HomeAssistant(str(tmp_path))
    file_dir=tmp_path/'files'/'test-id';file_dir.mkdir(parents=True)
    (file_dir/'guide.md').write_text('Guide content')
    hass.data['file_upload']=FileUploadData(tmp_path/'files', {'test-id':'guide.md'})
    name,content=await hass.async_add_executor_job(_read_uploaded_guide,hass,'test-id')
    assert (name,content)==('guide.md',b'Guide content')
    assert not file_dir.exists()


@pytest.mark.asyncio
async def test_unsupported_id_is_terminal_without_http():
    e=entry();tool=QueryHomeDocumentsTool(e,[APPLIANCE])
    result=await tool.async_call(None,ToolInput(tool.name,{'query':'dishwasher E4','appliance_id':'unsupported'}),None)
    assert result['status']=='unsupported_appliance' and result['retryable'] is False
    e.runtime_data.client.request.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('status',['matched','outside_supported_weight_range','needs_weight'])
async def test_reviewed_conversation_never_calls_llm(tmp_path, monkeypatch, status):
    from custom_components.homeguide.conversation import HomeGuideConversation
    from homeassistant.components import conversation
    from homeassistant.core import Context
    e=entry();e.entry_id='test';e.options['fallback_agent']='conversation.gemma'
    e.runtime_data.client.request.return_value={'status':status,'answer':'Verified answer with a manual citation.'}
    agent=HomeGuideConversation(e);agent.hass=HomeAssistant(str(tmp_path))
    def forbidden(*args):pytest.fail('Reviewed answer must not reach generation')
    monkeypatch.setattr(conversation,'async_get_agent',forbidden)
    request=conversation.ConversationInput(text='microwave defrost 500g mince',context=Context(),conversation_id=None,device_id=None,satellite_id=None,language='en',agent_id='conversation.homeguide_assist')
    result=await agent.async_process(request)
    assert result.response.speech['plain']['speech']=='Verified answer with a manual citation.'
    assert result.conversation_id


@pytest.mark.asyncio
async def test_conversation_delegates_other_requests_preserving_context(tmp_path,monkeypatch):
    from custom_components.homeguide.conversation import HomeGuideConversation
    from homeassistant.components import conversation
    from homeassistant.core import Context
    e=entry();e.entry_id='test';e.options['fallback_agent']='conversation.gemma'
    e.runtime_data.client.request.return_value={'status':'not_applicable'}
    agent=HomeGuideConversation(e);agent.hass=HomeAssistant(str(tmp_path))
    delegate=SimpleNamespace()
    converse=AsyncMock(return_value='original response')
    monkeypatch.setattr(conversation,'async_converse',converse)
    monkeypatch.setattr(conversation,'async_get_agent',lambda *args:delegate)
    request=conversation.ConversationInput(text='Set study brightness to 50%',context=Context(),conversation_id='same-session',device_id='satellite-device',satellite_id='satellite.voice',language='en',agent_id='conversation.homeguide_assist',extra_system_prompt='Original context')
    assert await agent.async_process(request)=='original response'
    sent=converse.call_args.kwargs
    assert sent['agent_id']=='conversation.gemma'
    assert sent['conversation_id']==request.conversation_id and sent['context'] is request.context
    assert sent['text']==request.text and sent['extra_system_prompt']==request.extra_system_prompt
    e.runtime_data.client.request.assert_not_awaited()
    # Renamed HomeGuide entities must not produce recursive delegation.
    monkeypatch.setattr(conversation,'async_get_agent',lambda *args:agent)
    assert (await agent.async_process(request)).response.speech['plain']['speech']
