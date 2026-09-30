"""Native real-model timing on deterministic synthetic images; not accuracy evidence."""
import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
import numpy as np
import psutil
from PIL import Image
from starlette.datastructures import UploadFile, Headers

parser = argparse.ArgumentParser()
parser.add_argument('--baseline-source', type=Path)
parser.add_argument('--requests', type=int, default=100)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
peak = [0]; done = threading.Event()
def sample_memory():
    process = psutil.Process()
    while not done.wait(.02): peak[0] = max(peak[0], process.memory_info().rss)
threading.Thread(target=sample_memory, daemon=True).start()
started = time.perf_counter()
if args.baseline_source:
    spec = importlib.util.spec_from_file_location('baseline_main', args.baseline_source)
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    asyncio.run(api.load_model_and_mappings())
else:
    import main as api
    api.load_model_and_mappings()
cold = time.perf_counter() - started
if api.model_efficientnet is None: raise RuntimeError('Real model failed to load')
rng = np.random.default_rng(7300930)
corpus = []
for i in range(50):
    pixels = rng.integers(0,256,(224,224,3),dtype=np.uint8)
    if i < 10: pixels[:] = i * 25
    buffer = io.BytesIO(); Image.fromarray(pixels).save(buffer,format='PNG'); corpus.append(buffer.getvalue())
def infer(data):
    if args.baseline_source:
        upload = UploadFile(io.BytesIO(data), filename='fixture.png', headers=Headers({'content-type':'image/png'}))
        return asyncio.run(api.predict_skin_disease(upload)).model_dump()
    return api.predict_bytes(data).model_dump()
infer(corpus[0])  # warmup excluded
latencies=[]; failures=[]; results=[]
for i in range(args.requests):
    started=time.perf_counter()
    try: results.append(infer(corpus[i % 50]))
    except Exception as exc: failures.append({'request':i,'type':type(exc).__name__})
    latencies.append(time.perf_counter()-started)
done.set()
report = {'runtime':'native Windows, sequential inference, real model', 'corpus':'50 deterministic synthetic PNG images, seed 7300930; numerical inputs only, not clinical accuracy evidence', 'requests':args.requests, 'warm_p50_seconds':float(np.percentile(latencies,50)), 'warm_p95_seconds':float(np.percentile(latencies,95)), 'throughput_per_second':args.requests/sum(latencies), 'cold_start_seconds':cold, 'peak_process_rss_bytes':peak[0], 'failures':failures, 'raw_latency_seconds':latencies, 'results':results, 'corpus_sha256':[hashlib.sha256(x).hexdigest() for x in corpus], 'model_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'final_model2.keras',root/'resultsskinwise/efficientnet_final.keras']}, 'container_memory': 'not measured; native RSS is not container memory', 'upload_acknowledgement': 'not measured by this inference benchmark'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in {'raw_latency_seconds','results','corpus_sha256','model_sha256'}}))
