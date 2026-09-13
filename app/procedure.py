"""Small reviewed procedure registry; never infer controls from model vocabulary.

A record applies only to an active, ready, confirmed exact model/region manual
whose original bytes match the reviewed edition. Other requests keep normal RAG.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from . import catalog, db

RECORDS = tuple(json.loads(p.read_text()) for p in sorted((Path(__file__).parent / 'procedures').glob('*.json')))


@lru_cache(maxsize=64)
def _fingerprint(path: str, mtime: int, ctime: int, size: int) -> str:
    with open(path, 'rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def _source(appliance, record, category):
    for doc_id in appliance['document_ids']:
        row = db.connect().execute('SELECT * FROM documents WHERE id=?', (doc_id,)).fetchone()
        if not row or (category and row['category'] != category):
            continue
        path = db.pdf_path(doc_id, row['filename'])
        try:
            stat = path.stat()
            digest = _fingerprint(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except OSError:
            continue
        if digest == record['manual_sha256']:
            return {'document_id': doc_id, 'document': row['title'], 'pages': record['pages'],
                    'manual_sha256': digest, 'category': row['category']}
    return None


def is_mince_defrost(query):
    text = query.casefold()
    if re.search(r'\b(?:turn|switch|lights?|set|timer|alarm)\b', text):
        return False
    return bool(re.search(r'\bdefrost(?:ing)?\b', text) and re.search(r'\bminc(?:e|ed)\b', text)
                and not re.search(r'\b(?:manual|timed|power|watts?|fish|bread|chicken|pork|lamb|turkey|vegetarian|vegan|cooked|reheat|cook|cooking)\b', text))


def _weight(query):
    # Voice transcription commonly emits either 500 g or 0.5 kg. Multiple
    # quantities, fractions, signs and unrecognised units must not be guessed.
    text = re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', query.casefold())
    values = re.findall(r'(?<![\w.,])([+-]?\d+(?:\.\d+)?)\s*(kilograms?|kilos?|kg|grams?|g)\b', text)
    if len(values) != 1 or re.search(r'\d\s*[/–-]\s*\d|\b(?:or|between)\b', text):
        return None
    number, unit = values[0]
    if len(number) > 9:
        return None
    return float(number) * (1000 if unit.startswith('k') else 1)


def resolve(query: str, appliance: dict | None = None, category: str | None = None):
    if not is_mince_defrost(query):
        return None
    if appliance is None:
        text = ' ' + catalog.normalize(query) + ' '
        matches = [a for a in catalog.get_catalog()['appliances']
                   if any(' ' + catalog.normalize(term) + ' ' in text for term in
                          [a['name'], a['model'], *a['aliases'], *catalog.KINDS.get(a['kind'], [])] if term)]
        if len(matches) != 1:
            return None
        appliance, failure = catalog.scope_for(matches[0]['id'], query)
        if failure:
            return None
    for record in RECORDS:
        if appliance['kind'] != 'microwave' or any(catalog.normalize(appliance[k]) != catalog.normalize(record[k])
                                                  for k in ('manufacturer', 'model', 'region')):
            continue
        explicit_models = re.findall(r'\bnn[- ]?st[\w-]+', query, re.I)
        if any(catalog.normalize(model) != catalog.normalize(record['model']) for model in explicit_models):
            return None
        source = _source(appliance, record, category)
        if source is None:
            continue
        weight = _weight(query)
        minimum, maximum = (record['weight_range_g'][k] for k in ('minimum', 'maximum'))
        citation = f"{source['document']}, pages {', '.join(map(str, source['pages']))}."
        result = {'status': 'matched', 'appliance': {k: appliance[k] for k in ('id','name','manufacturer','model','region')},
                  'source': source, 'requested_weight_g': weight, 'retryable': False,
                  'weight_range_g': record['weight_range_g']}
        if weight is None:
            result['status'] = 'needs_weight'
            answer = f'Please give one weight in grams or kilograms; this automatic mince programme supports {minimum}–{maximum} g.'
        elif not minimum <= weight <= maximum:
            result['status'] = 'outside_supported_weight_range'
            answer = f'The automatic mince programme supports {minimum}–{maximum} g, so I cannot give automatic settings for {weight:g} g.'
        else:
            result['procedure'] = {k: record[k] for k in ('name','programme_number','weight_range_g','steps','standing_time_minutes')}
            answer = (f'For {weight:g} g of mince, select programme 7 with Chaos Defrost, enter {weight:g} g using the More/Less Weight pads, '
                      'then press Start; use a large shallow dish, break up the mince at the beeps and let it stand for 15–30 minutes afterwards.')
        result['answer'] = answer + ' Source: ' + citation
        result['answer_contract'] = ('Use the supplied answer and source. Include Chaos Defrost, programme 7, the entered weight and Start when settings are supplied. '
                                     'Standing time is after defrosting, not its duration. When no settings are supplied, explain the weight requirement; do not invent settings.')
        # Retain the legacy excerpt shape for clients that only display results.
        result['results'] = [{'document': source['document'], 'page': source['pages'][0],
                              'category': source['category'], 'excerpt': result['answer']}]
        return result
    return None
