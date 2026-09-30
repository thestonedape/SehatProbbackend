"""Bounded input handling and process-local observability for a single API replica."""
import hashlib
import io
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

MODEL_VERSION = os.getenv('MODEL_VERSION', 'keras-v1')
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
slot = threading.Lock()
rate_buckets = OrderedDict()
metrics = {'requests': 0, 'errors': 0, 'duration_seconds': 0.0, 'predictions': 0, 'prediction_seconds': 0.0}
latency_bounds = (.005,.01,.025,.05,.1,.25,.5,1,2,5,10)
latency_counts = [0] * len(latency_bounds)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        try:
            payload = json.loads(message)
            if not isinstance(payload, dict): payload = {'event':message}
        except (ValueError, TypeError): payload = {'event':message}
        payload.update(level=record.levelname, logger=record.name)
        if record.exc_info: payload['exception_type'] = record.exc_info[0].__name__
        return json.dumps(payload)

async def validate_image(file):
    if file.content_type not in {'image/jpeg', 'image/png'}:
        raise HTTPException(415, 'Only JPEG and PNG images are supported')
    chunks, size = [], 0
    while chunk := await file.read(64 * 1024):
        size += len(chunk)
        if size > MAX_BYTES:
            raise HTTPException(413, 'Maximum image size is 10 MB')
        chunks.append(chunk)
    data = b''.join(chunks)
    await run_in_threadpool(validate_decoded_image, data, file.content_type)
    return data

def validate_decoded_image(data, content_type):
    expected = 'PNG' if content_type == 'image/png' else 'JPEG'
    if not (data.startswith(b'\x89PNG\r\n\x1a\n') if expected == 'PNG' else data.startswith(b'\xff\xd8\xff')):
        raise HTTPException(415, 'Image signature does not match its MIME type')
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != expected:
                raise HTTPException(415, 'Image format does not match its MIME type')
            if min(image.size) < 16 or max(image.size) > 8192 or image.width * image.height > MAX_PIXELS:
                raise HTTPException(422, 'Image dimensions exceed supported limits')
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise HTTPException(422, 'Image is corrupt or cannot be decoded') from exc

MAX_WAITING = int(os.getenv('PREDICTION_MAX_WAITING', '4'))
QUEUE_WAIT_SECONDS = float(os.getenv('PREDICTION_QUEUE_WAIT_SECONDS', '10'))
waiting = 0
waiting_lock = threading.Lock()

def _admit():
    """One active inference, a few bounded waiters; everything else is rejected at once."""
    global waiting
    if slot.acquire(blocking=False):
        return
    with waiting_lock:
        if waiting >= MAX_WAITING:
            metrics['rejected_overload'] = metrics.get('rejected_overload', 0) + 1
            raise HTTPException(429, 'Inference queue is full; retry later', headers={'Retry-After': '2'})
        waiting += 1
    try:
        if not slot.acquire(timeout=QUEUE_WAIT_SECONDS):
            metrics['rejected_wait_timeout'] = metrics.get('rejected_wait_timeout', 0) + 1
            raise HTTPException(429, 'Inference is busy; retry later', headers={'Retry-After': '2'})
    finally:
        with waiting_lock:
            waiting -= 1

@contextmanager
def prediction_slot():
    _admit()
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics['predictions'] += 1
        metrics['prediction_seconds'] += time.perf_counter() - started
        slot.release()

class admitted_prediction:
    """Async form: waiting happens in a worker thread, never on the event loop."""
    async def __aenter__(self):
        await run_in_threadpool(_admit)
        self.started = time.perf_counter()
    async def __aexit__(self, *exc):
        metrics['predictions'] += 1
        metrics['prediction_seconds'] += time.perf_counter() - self.started
        slot.release()

def rate_limit(request):
    # Bounded, per-process, direct-peer limiter. Do not trust arbitrary X-Forwarded-For.
    key = request.client.host if request.client else 'unknown'
    now = time.monotonic()
    count, expiry = rate_buckets.pop(key, (0, now + 60))
    if now >= expiry:
        count, expiry = 0, now + 60
    rate_buckets[key] = (count + 1, expiry)
    while len(rate_buckets) > 10000:
        rate_buckets.popitem(last=False)
    if count >= int(os.getenv('RATE_LIMIT_PER_MINUTE', '30')):
        raise HTTPException(429, 'Rate limit exceeded', headers={'Retry-After': str(max(1, int(expiry - now)))})

def model_checksum(path, variable):
    expected = os.getenv(variable)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    if expected and digest != expected:
        raise ValueError('Model checksum mismatch')
    return digest

def install_observability(app):
    if not logging.getLogger().handlers: logging.basicConfig(level=logging.INFO)
    for handler in logging.getLogger().handlers: handler.setFormatter(JSONFormatter())
    @app.middleware('http')
    async def observe(request, call_next):
        supplied = request.headers.get('X-Request-ID', '')
        request_id = supplied if re.fullmatch(r'[A-Za-z0-9_-]{1,64}', supplied) else uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse({'detail': 'Internal service error', 'request_id': request_id}, status_code=500)
        elapsed = time.perf_counter() - started
        metrics['requests'] += 1
        metrics['errors'] += int(response.status_code >= 400)
        metrics['duration_seconds'] += elapsed
        for index, bound in enumerate(latency_bounds): latency_counts[index] += int(elapsed <= bound)
        response.headers['X-Request-ID'] = request_id
        logging.getLogger('api.http').info(json.dumps({'event': 'request', 'request_id': request_id, 'method': request.method, 'status': response.status_code, 'duration_ms': round(elapsed * 1000, 3)}))
        return response
    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({'detail': exc.detail, 'request_id': getattr(request.state, 'request_id', None)}, status_code=exc.status_code, headers=exc.headers)
    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({'detail': 'Invalid request', 'request_id': getattr(request.state, 'request_id', None)}, status_code=422)
    @app.get('/metrics', include_in_schema=False)
    def prometheus_metrics():
        rows=[f'sehat_{name} {value}' for name, value in metrics.items()]
        rows += [f'sehat_request_duration_seconds_bucket{{le="{bound}"}} {count}' for bound,count in zip(latency_bounds,latency_counts)]
        rows += [f'sehat_request_duration_seconds_bucket{{le="+Inf"}} {metrics["requests"]}',f'sehat_request_duration_seconds_count {metrics["requests"]}',f'sehat_request_duration_seconds_sum {metrics["duration_seconds"]}']
        return Response('\n'.join(rows) + '\n', media_type='text/plain; version=0.0.4')
