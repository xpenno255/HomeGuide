"""Confirmed appliance/document associations, independent of retrieval rankings."""
import hashlib
import json
import re
import uuid

from . import db

KINDS = {
    'microwave': ['microwave'],
    'air_fryer': ['air fryer', 'airfryer'],
    'coffee_machine': ['coffee machine', 'coffee maker', 'coffee brewer'],
    'tumble_dryer': ['tumble dryer', 'dryer'],
    'washing_machine': ['washing machine', 'washer'],
    'dishwasher': ['dishwasher'],
    'radiator_valve': ['radiator valve', 'radiator thermostat', 'trv'],
    'oven': ['oven'],
    'fridge': ['fridge', 'refrigerator'],
    'freezer': ['freezer'],
    'household': ['house paperwork', 'house documents'],
    'other': [],
}


def normalize(text: str) -> str:
    return ' '.join(re.findall(r'\w+', text.casefold()))


def _decode(row):
    item = dict(row)
    item['aliases'] = json.loads(item['aliases'])
    return item


def list_appliances():
    conn = db.connect()
    rows = conn.execute('SELECT * FROM appliances ORDER BY id').fetchall()
    items = []
    for row in rows:
        item = _decode(row)
        item['documents'] = [dict(d) for d in conn.execute(
            'SELECT d.id, d.title, d.status, d.active, da.verified '
            'FROM document_appliances da JOIN documents d ON d.id=da.doc_id '
            'WHERE da.appliance_id=? ORDER BY d.id', (item['id'],))]
        items.append(item)
    return items


def get_catalog():
    entries = []
    for item in list_appliances():
        ready = [d for d in item.pop('documents') if d['verified'] and d['active'] and d['status'] == 'ready']
        if ready:
            entries.append({**item, 'document_ids': [d['id'] for d in ready]})
    # No clock or ephemeral indexing data in the model-facing catalogue.
    revision = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:16]
    return {'revision': revision, 'appliances': entries}


def save_appliance(data: dict, appliance_id: str | None = None):
    name = str(data.get('name', '')).strip()
    if not name or len(name) > 100:
        raise ValueError('Give the appliance a name of 1–100 characters.')
    kind = data.get('kind', 'other')
    if kind not in KINDS:
        raise ValueError('Unknown appliance type.')
    aliases = data.get('aliases', [])
    if isinstance(aliases, str):
        aliases = aliases.split(',')
    if not isinstance(aliases, list) or len(aliases) > 20 or not all(isinstance(a, str) and len(a) <= 100 for a in aliases):
        raise ValueError('Aliases must be a list of up to 20 short names.')
    aliases = sorted(set(a.strip() for a in aliases if a.strip()), key=str.casefold)
    fields = {key: str(data.get(key, '')).strip() for key in ['manufacturer', 'model', 'region']}
    if any(len(v) > 100 for v in fields.values()):
        raise ValueError('Manufacturer, model and region must each be at most 100 characters.')
    conn = db.connect()
    with db.lock:
        if appliance_id is not None:
            old = conn.execute('SELECT * FROM appliances WHERE id=?', (appliance_id,)).fetchone()
            if old is None:
                raise KeyError(appliance_id)
            # A guide confirmed for one model/region is not confirmed for another.
            if any(old[key] != fields[key] for key in fields) or old['kind'] != kind:
                conn.execute('UPDATE document_appliances SET verified=0 WHERE appliance_id=?', (appliance_id,))
            conn.execute('UPDATE appliances SET name=?,kind=?,manufacturer=?,model=?,region=?,aliases=? WHERE id=?',
                         (name, kind, fields['manufacturer'], fields['model'], fields['region'], json.dumps(aliases), appliance_id))
        else:
            appliance_id = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')[:40] or 'appliance'
            if appliance_id == 'unsupported' or conn.execute('SELECT 1 FROM appliances WHERE id=?', (appliance_id,)).fetchone():
                appliance_id += '_' + uuid.uuid4().hex[:6]
            conn.execute('INSERT INTO appliances(id,name,kind,manufacturer,model,region,aliases) VALUES (?,?,?,?,?,?,?)',
                         (appliance_id, name, kind, fields['manufacturer'], fields['model'], fields['region'], json.dumps(aliases)))
        conn.commit()
    return next(a for a in list_appliances() if a['id'] == appliance_id)


def link_document(doc_id: int, appliance_id: str, verified: bool):
    conn = db.connect()
    with db.lock:
        if not conn.execute('SELECT 1 FROM appliances WHERE id=?', (appliance_id,)).fetchone():
            raise KeyError('Appliance not found')
        if not conn.execute('SELECT 1 FROM documents WHERE id=?', (doc_id,)).fetchone():
            raise KeyError('Document not found')
        conn.execute('INSERT INTO document_appliances(doc_id,appliance_id,verified) VALUES (?,?,?) '
                     'ON CONFLICT(doc_id,appliance_id) DO UPDATE SET verified=excluded.verified',
                     (doc_id, appliance_id, int(verified)))
        conn.commit()


def scope_for(appliance_id: str, query: str):
    """Return a verified scope, or a terminal result. Never broaden a failed scope."""
    entries = get_catalog()['appliances']
    selected = next((a for a in entries if a['id'] == appliance_id), None)
    if selected is None:
        return None, {'status': 'unsupported_appliance', 'results': [], 'retryable': False,
                      'note': 'No ready, confirmed guide is available for that appliance. Say so and finish.'}
    # Catch unambiguous explicit subject conflicts, including an unregistered
    # dishwasher. Do not mistake "dishwasher safe" for the subject of a question.
    text = ' ' + normalize(query) + ' '
    text = re.sub(r'\b(?:dishwasher|microwave|oven) safe\b', '', text)
    kinds = {kind for kind, terms in KINDS.items() if any(' ' + term + ' ' in text for term in terms)}
    if 'microwave' in kinds:
        kinds.discard('oven')  # "microwave oven" is one appliance
    if len(kinds) == 1 and selected['kind'] not in kinds and selected['kind'] != 'other':
        return None, {'status': 'appliance_mismatch', 'results': [], 'retryable': False,
                      'note': 'The query names a different appliance from the selected guide. Do not use this guide for that appliance.'}
    return selected, None
