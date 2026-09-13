"""Verified catalogue lifecycle and retrieval isolation."""
import pytest
from app import catalog, db, search


def appliance(name='Microwave', kind='microwave', **kwargs):
    return catalog.save_appliance({'name': name, 'kind': kind, **kwargs})


def test_only_confirmed_ready_active_documents_are_advertised(client, add_doc):
    item = appliance(model='NN-ST46KB', region='UK', aliases=['Panasonic', 'kitchen microwave'])
    doc = add_doc('Chaos defrost minced meat programme 7.', 'UK microwave')
    before = catalog.get_catalog()
    catalog.link_document(doc, item['id'], False)
    assert catalog.get_catalog() == before
    catalog.link_document(doc, item['id'], True)
    ready = catalog.get_catalog()
    assert ready['appliances'][0]['document_ids'] == [doc]
    assert catalog.get_catalog() == ready
    assert client.patch(f'/api/documents/{doc}', json={'active': False}).status_code == 200
    assert not catalog.get_catalog()['appliances']
    client.patch(f'/api/documents/{doc}', json={'active': True})
    assert catalog.get_catalog() == ready
    conn = db.connect()
    for status in ('processing', 'error'):
        conn.execute('UPDATE documents SET status=? WHERE id=?', (status, doc)); conn.commit()
        assert not catalog.get_catalog()['appliances']


def test_model_change_requires_reconfirmation(client, add_doc):
    item = appliance(model='NN-ST46KB', region='UK')
    doc = add_doc('Chaos defrost minced meat programme 7.')
    catalog.link_document(doc, item['id'], True)
    catalog.save_appliance({**item, 'aliases': ['kitchen']}, item['id'])
    assert catalog.get_catalog()['appliances']
    catalog.save_appliance({**item, 'model': 'OTHER'}, item['id'])
    assert not catalog.get_catalog()['appliances']


def test_scope_blocks_unknown_and_explicit_wrong_appliance(client, add_doc, monkeypatch):
    item = appliance()
    doc = add_doc('Microwave cleaning instructions.')
    catalog.link_document(doc, item['id'], True)
    def forbidden(*a, **kw):
        pytest.fail('Retrieval must not run for rejected scope')
    monkeypatch.setattr(search, 'hybrid_search', forbidden)
    for identity, query, expected in [('unknown', 'microwave cleaning', 'unsupported_appliance'),
                                      (item['id'], 'dishwasher E4', 'appliance_mismatch')]:
        body = client.get('/query', params={'q': query, 'appliance_id': identity}).json()
        assert body['status'] == expected and body['results'] == [] and body['retryable'] is False
    assert catalog.scope_for(item['id'], 'microwave oven safe dish')[1] is None
    assert catalog.scope_for(item['id'], 'is this tray dishwasher safe')[1] is None


def test_scoped_search_cannot_leak_other_documents(client, add_doc, stub_embeddings):
    stub_embeddings.default = 0.95
    mine = add_doc('Chaos defrost minced meat programme 7.', 'Microwave')
    other = add_doc('Chaos defrost minced meat programme 7. E4 dishwasher error.', 'Wrong manual')
    item = appliance()
    catalog.link_document(mine, item['id'], True)
    body = client.get('/query', params={'q': 'microwave Chaos defrost minced meat', 'appliance_id': item['id']}).json()
    assert body['status'] == 'matched'
    assert {r['document'] for r in body['results']} == {'Microwave'}
    assert body['appliance']['id'] == item['id']
    # Exact codes in another manual are not corroboration for this scope.
    assert client.get('/query', params={'q': 'microwave error E4', 'appliance_id': item['id']}).json()['results'] == []


def test_upload_association_lifecycle_and_delete(client):
    item = appliance()
    response = client.post('/api/upload', data={'title': 'UK guide', 'appliance_id': item['id'], 'verified': 'true'},
        files={'file': ('guide.md', b'Chaos defrost minced meat uses programme 7. Select 500 g.', 'text/markdown')})
    assert response.status_code == 200
    doc = response.json()['id']
    assert catalog.get_catalog()['appliances'][0]['document_ids'] == [doc]
    assert client.delete(f'/api/documents/{doc}').status_code == 200
    assert not catalog.get_catalog()['appliances']
    bad = client.post('/api/upload', data={'appliance_id': 'missing'}, files={'file': ('guide.md', b'test')})
    assert bad.status_code == 400


def test_verification_is_a_boolean_and_invalid_metadata_rejected(client, add_doc):
    item = appliance()
    doc = add_doc('Manual cleaning guidance.')
    assert client.post(f'/api/documents/{doc}/appliances', json={'appliance_id': item['id'], 'verified': 'false'}).status_code == 422
    assert client.post('/api/appliances', json={'name': 'x', 'kind': 'nonexistent'}).status_code == 400
    assert client.post('/api/appliances', json={'name': '', 'aliases': []}).status_code == 422


def test_scope_filters_before_candidate_limit(client, add_doc, stub_embeddings):
    stub_embeddings.default = 0.99
    for i in range(search.CANDIDATES + 2):
        add_doc('Cleaning descaling water tank procedure.', f'Unrelated {i}')
    mine = add_doc('Cleaning descaling water tank procedure.', 'Selected guide')
    results = search.hybrid_search('cleaning descaling water tank', doc_ids={mine})
    assert results and {r['document'] for r in results} == {'Selected guide'}


def test_upgrade_preserves_existing_library(data_dir):
    import sqlite3
    legacy = sqlite3.connect(db.DB_PATH)
    legacy.executescript(db.SCHEMA.replace('    active      INTEGER NOT NULL DEFAULT 1,\n', ''))
    legacy.execute("INSERT INTO documents(title,filename,status) VALUES ('Old guide','old.pdf','ready')")
    legacy.execute("INSERT INTO chunks(doc_id,page,text) VALUES (1,4,'Existing guidance')")
    legacy.commit(); legacy.close()
    conn = db.connect()
    assert dict(conn.execute('SELECT title,active,status FROM documents').fetchone()) == {'title':'Old guide','active':1,'status':'ready'}
    assert conn.execute('SELECT text FROM chunks').fetchone()[0] == 'Existing guidance'
    assert catalog.get_catalog()['appliances'] == []
