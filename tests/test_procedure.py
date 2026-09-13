"""Reviewed facts must never cross model, edition or library boundaries."""
import hashlib
from pathlib import Path

import pytest
from app import catalog, db, llm, procedure


@pytest.fixture
def reviewed(client, add_doc, monkeypatch):
    doc = add_doc('Test-only reviewed source.', 'Panasonic UK manual')
    a = catalog.save_appliance(dict(name='Microwave',kind='microwave',manufacturer='Panasonic',model='NN-ST46KB',region='UK'))
    catalog.link_document(doc,a['id'],True)
    record = dict(procedure.RECORDS[0], manual_sha256=hashlib.sha256(db.pdf_path(doc,'doc.md').read_bytes()).hexdigest())
    monkeypatch.setattr(procedure,'RECORDS',(record,))
    return a, doc


@pytest.mark.parametrize('amount,grams,status', [('200 g',200,'matched'),('500g',500,'matched'),('0.75 kg',750,'matched'),('1.2kg',1200,'matched'),('1500g',1500,'outside_supported_weight_range'),('100g',100,'outside_supported_weight_range'),('-500g',-500,'outside_supported_weight_range'),('1,500g',1500,'outside_supported_weight_range')])
def test_weight_and_answer_contract(client,reviewed,amount,grams,status):
    body=client.post('/api/resolve',json={'question':f'How do I defrost {amount} mince in the microwave?'}).json()
    assert body['status']==status and body['requested_weight_g']==grams
    assert body['source']['pages']==[32,33] and body['answer']
    if status=='matched':
        assert all(word in body['answer'] for word in ('Chaos Defrost','programme 7','Weight pads','Start','Source:'))
        assert 'procedure' in body
    else:
        assert 'procedure' not in body and 'Start' not in body['answer']


@pytest.mark.parametrize('amount',['','500g or 750g','1/2kg','500-750g','500–750g','500 lb','five hundred grams'])
def test_ambiguous_or_unparsed_weight_does_not_invent_settings(client,reviewed,amount):
    body=client.post('/api/resolve',json={'question':f'microwave defrost {amount} mince'}).json()
    assert body['status']=='needs_weight' and 'Start' not in body['answer']


@pytest.mark.parametrize('change',['model','region','unverified','inactive','processing','deleted_file','changed_file'])
def test_identity_and_source_lifecycle(client,reviewed,change):
    a,doc=reviewed
    if change in ('model','region'):
        catalog.save_appliance({**a,change:'different'},a['id'])
        catalog.link_document(doc,a['id'],True)
    elif change=='unverified':catalog.link_document(doc,a['id'],False)
    elif change=='inactive':client.patch(f'/api/documents/{doc}',json={'active':False})
    elif change=='processing':
        db.connect().execute("UPDATE documents SET status='processing' WHERE id=?",(doc,));db.connect().commit()
    elif change=='deleted_file':db.pdf_path(doc,'doc.md').unlink()
    else:db.pdf_path(doc,'doc.md').write_text('Wrong Turbo Defrost edition')
    assert client.post('/api/resolve',json={'question':'microwave defrost 500g mince'}).json()=={'status':'not_applicable'}


@pytest.mark.parametrize('query',['air fryer defrost 500g mince','microwave cook 500g mince','microwave manual defrost timing for 500g mince','microwave defrost 500g fish','microwave defrost 500g mince and chicken','microwave NN-ST45KW defrost 500g mince'])
def test_unreviewed_questions_keep_normal_retrieval(client,reviewed,query):
    assert client.post('/api/resolve',json={'question':query}).json()['status']=='not_applicable'


def test_query_and_web_answer_use_actual_registry_without_llm(client,reviewed,monkeypatch):
    def forbidden(*args):pytest.fail('Reviewed answers must not require inference')
    monkeypatch.setattr(llm,'ask',forbidden);monkeypatch.setattr(llm,'enabled',lambda:False)
    a,_=reviewed
    q='microwave defrost 500g mince'
    query=client.get('/query',params={'q':q,'appliance_id':a['id']}).json()
    web=client.post('/api/ask',json={'question':q,'appliance_id':a['id']}).json()
    assert query==web and query['procedure']['programme_number']==7
    assert client.get('/query',params={'q':q,'appliance_id':'unknown'}).json()['status']=='unsupported_appliance'
    assert 'procedure' not in client.get('/query',params={'q':q,'appliance_id':a['id'],'category':'warranty'}).json()


def test_missing_or_multiple_appliances_do_not_guess(client,reviewed,add_doc):
    a,doc=reviewed
    other=catalog.save_appliance({**a,'name':'Other microwave'})
    catalog.link_document(doc,other['id'],True)
    assert client.post('/api/resolve',json={'question':'microwave defrost 500g mince'}).json()['status']=='not_applicable'
    assert client.get('/query',params={'q':'defrost 500g mince','appliance_id':a['id']}).json()['status']=='matched'


def test_empty_generated_answer_has_useful_fallback(client,add_doc,stub_embeddings,monkeypatch):
    stub_embeddings.default=.95;add_doc('Clean the coffee machine with a damp cloth.')
    monkeypatch.setattr(llm,'enabled',lambda:True);monkeypatch.setattr(llm,'ask',lambda *args:'  ')
    body=client.post('/api/ask',json={'question':'coffee machine cleaning'}).json()
    assert 'no spoken answer' in body['answer'] and body['results']
