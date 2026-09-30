"""Verify deployment artifact URLs by streaming their complete pinned bytes."""
import hashlib
import json
from pathlib import Path
import urllib.request

root = Path(__file__).resolve().parents[1]
results = []
for item in json.loads((root / 'space/space_models.json').read_text()):
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(item['url'], timeout=120) as response:
        while block := response.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    assert size == item['bytes'] and digest.hexdigest() == item['sha256'], item['path']
    results.append({'path': item['path'], 'bytes': size, 'sha256_verified': True})
(root / 'artifacts/space-public-model-check.json').write_text(json.dumps(results, indent=2))
print(json.dumps(results))
