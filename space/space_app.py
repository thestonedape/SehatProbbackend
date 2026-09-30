"""Gradio Space packaging of the existing CPU FastAPI service.

No GPU acceleration or changed inference path. All predictions still pass through
the original /predict validation, admission and model service.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE if (SOURCE / 'main.py').exists() else SOURCE.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def materialize_models(manifest, root=ROOT):
    """Only pinned public artifacts; never accept downloaded pickle unchecked."""
    for item in manifest:
        target = root / item['path']
        if target.exists() and target.stat().st_size == item['bytes'] and sha256(target) == item['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.download')
        try:
            for attempt in range(3):
                try:
                    received = 0
                    with urllib.request.urlopen(item['url'], timeout=120) as response, temporary.open('wb') as stream:
                        while block := response.read(1024 * 1024):
                            received += len(block)
                            if received > item['bytes']:
                                raise RuntimeError('Artifact exceeds pinned size')
                            stream.write(block)
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
            if received != item['bytes'] or sha256(temporary) != item['sha256']:
                raise RuntimeError('Artifact checksum mismatch')
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def serving_app():
    manifest = json.loads((SOURCE / 'space_models.json').read_text())
    materialize_models(manifest)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import gradio as gr
    from main import app

    with gr.Blocks(title='Sehat Pro API', analytics_enabled=False) as status:
        gr.Markdown('## Sehat Pro API\nThe existing FastAPI inference service runs on CPU. '
                    'Use `/health`, `/ready`, `/docs` and `/predict`. '
                    'The frontend remains the existing Sehat Pro website. '
                    'This demo can sleep; model loading occurs once per process startup.')
    return gr.mount_gradio_app(app, status, path='/status')


if __name__ == '__main__':
    import uvicorn
    # ZeroGPU reserves PORT for its internal proxy; the public Gradio service
    # listens on GRADIO_SERVER_PORT (7860 by default).
    uvicorn.run(serving_app(), host='0.0.0.0', port=int(os.getenv('GRADIO_SERVER_PORT', '7860')), workers=1)
