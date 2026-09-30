"""Memory-limited container benchmark: cold start, bounded burst, cgroup peak memory, OOM events.

    python scripts/container_bench.py --image sde/sehat:upgrade --memory 512m --output artifacts/container-512m.json

Synthetic images exercise the serving path only; results are not accuracy evidence.
"""
import argparse
import io
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True)
parser.add_argument('--memory', default='512m')
parser.add_argument('--requests', type=int, default=100)
parser.add_argument('--concurrency', type=int, default=4)
parser.add_argument('--port', type=int, default=8010)
parser.add_argument('--env', action='append', default=[], help='extra KEY=VALUE for the container')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
name = f'sehat-bench-{args.port}'


def docker(*cmd, check=True):
    return subprocess.run(['docker', *cmd], capture_output=True, text=True, check=check).stdout.strip()


def cgroup(file):
    try:
        return docker('exec', name, 'cat', f'/sys/fs/cgroup/{file}')
    except subprocess.CalledProcessError:
        return ''


docker('rm', '-f', name, check=False)
env = ['-e', 'RATE_LIMIT_PER_MINUTE=100000'] + [x for kv in args.env for x in ('-e', kv)]
started = time.perf_counter()
docker('run', '-d', '--name', name, '--memory', args.memory, '--memory-swap', args.memory, '-p', f'{args.port}:8000', *env, args.image)
base = f'http://127.0.0.1:{args.port}'
cold, ready_body, samples, anon, done = None, None, [], [], threading.Event()


def sample():
    while not done.wait(0.5):
        value = cgroup('memory.current')
        if value.isdigit():
            samples.append(int(value))
        stat = dict(line.split() for line in cgroup('memory.stat').splitlines()[:40] if line)
        if 'anon' in stat:
            anon.append(int(stat['anon']))  # non-reclaimable; page cache is excluded


threading.Thread(target=sample, daemon=True).start()
with httpx.Client(timeout=120) as client:
    while time.perf_counter() - started < 600:
        if docker('inspect', '-f', '{{.State.Running}}', name, check=False) != 'true':
            break
        try:
            response = client.get(base + '/ready')
            if response.status_code == 200:
                cold, ready_body = time.perf_counter() - started, response.json()
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    results = []
    if cold is not None:
        buffer = io.BytesIO()
        rng = np.random.default_rng(7300930)
        images = []
        for _ in range(10):
            buffer = io.BytesIO()
            Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(buffer, format='PNG')
            images.append(buffer.getvalue())
        client.post(base + '/predict', files={'file': ('w.png', images[0], 'image/png')})

        def request(i):
            t = time.perf_counter()
            try:
                r = client.post(base + '/predict', files={'file': (f'{i}.png', images[i % 10], 'image/png')})
                status = r.status_code
            except httpx.HTTPError as exc:
                status = type(exc).__name__
            return {'request': i, 'status': status, 'seconds': time.perf_counter() - t}

        burst = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(request, range(args.requests)))
        duration = time.perf_counter() - burst
done.set()
events = dict(line.split() for line in cgroup('memory.events').splitlines() if line)
peak = cgroup('memory.peak')
state = json.loads(docker('inspect', '-f', '{{json .State}}', name))
image_bytes = int(docker('image', 'inspect', '-f', '{{.Size}}', args.image))
logs = subprocess.run(['docker', 'logs', '--tail', '15', name], capture_output=True, text=True).stderr[-3000:]
docker('rm', '-f', name, check=False)

ok = [x['seconds'] for x in results if x['status'] == 200]
report = {
    'scope': f'Docker Desktop (WSL2) container, --memory {args.memory} no swap, real models, synthetic 224x224 PNGs; not accuracy evidence',
    'image': args.image, 'image_bytes': image_bytes, 'memory_limit': args.memory,
    'cold_start_to_ready_seconds': cold, 'ready': ready_body,
    'requests': len(results), 'concurrency': args.concurrency,
    'status_counts': {str(s): sum(x['status'] == s for x in results) for s in {x['status'] for x in results}},
    'p50_seconds': float(np.percentile(ok, 50)) if ok else None,
    'p95_seconds': float(np.percentile(ok, 95)) if ok else None,
    'successful_per_second': len(ok) / duration if results else 0,
    'peak_cgroup_memory_bytes': int(peak) if peak.isdigit() else (max(samples) if samples else None),
    'peak_anon_memory_bytes_sampled': max(anon) if anon else None,
    'peak_source': 'memory.peak' if peak.isdigit() else 'sampled memory.current',
    'oom_kill_events': int(events.get('oom_kill', 0)) if events else None,
    'container_oom_killed': state.get('OOMKilled'), 'container_exit_code': state.get('ExitCode'),
    'log_tail': logs if cold is None else None,
    'raw_requests': results,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in {'raw_requests', 'log_tail'}}, indent=2))
if cold is None:
    print(logs)
