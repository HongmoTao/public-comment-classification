"""Read prepared records and write small experiment outputs.

Classifier records contain sample_id, project_id, text, label, label_id,
and leakage_group_id. Labels are consecutive integers within each project.
"""
import json
from pathlib import Path


def read_json(path):
    text = Path(path).read_text(encoding='utf-8')
    return json.loads(text)


def read_rows(path, key=None):
    text = Path(path).read_text(encoding='utf-8')
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for line in text.splitlines():
            if line.strip():
                value.append(json.loads(line))
    if isinstance(value, dict):
        if key is not None and key in value:
            value = value[key]
        elif 'rows' in value:
            value = value['rows']
        else:
            value = [value]
    if not isinstance(value, list):
        raise ValueError('Expected a JSON record list or JSONL')
    for row in value:
        if not isinstance(row, dict):
            raise ValueError('Each record must be a JSON object')
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    path.write_text(text + '\n', encoding='utf-8')


def index_rows(rows):
    if not rows:
        raise ValueError('Prepared records must not be empty')
    required = ('sample_id', 'project_id', 'text', 'label', 'label_id', 'leakage_group_id')
    indexed = {}
    for row in rows:
        for field in required:
            if field not in row:
                raise ValueError('Missing required field: ' + field)
        sample_id = row['sample_id']
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError('Sample IDs must be nonempty strings')
        if sample_id in indexed:
            raise ValueError('Repeated sample ID')
        if not isinstance(row['text'], str) or not row['text'].strip():
            raise ValueError('Text must be a nonempty string')
        if type(row['label_id']) is not int or row['label_id'] < 0:
            raise ValueError('Label IDs must be nonnegative integers')
        for field in ('project_id', 'label', 'leakage_group_id'):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError('Expected a nonempty string: ' + field)
        indexed[sample_id] = row
    return indexed


def label_ids(rows):
    labels = set()
    for row in rows:
        labels.add(row['label_id'])
    return sorted(labels)


def sample_ids(rows):
    ids = []
    for row in rows:
        ids.append(row['sample_id'])
    return ids
