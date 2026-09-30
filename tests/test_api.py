import time
import io
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import main
import reliability
import asyncio
import builtins
import pickle
import sys
from types import SimpleNamespace

class Model:
    output_shape = (None, 3)
    def __call__(self, data, training=False):
        return np.array([[0.60, 0.25, 0.15]], dtype=np.float32)

@pytest.fixture
def client(monkeypatch):
    def load():
        main.model_efficientnet = Model()
        main.model_vgg16 = None
        main.class_index = {'benign': 0, 'melanoma': 1, 'other': 2}
        main.index_class = {v: k for k, v in main.class_index.items()}
    monkeypatch.setattr(main, 'load_model_and_mappings', load)
    reliability.rate_buckets.clear()
    with TestClient(main.app) as api:
        yield api

def image(format='PNG', size=(32,32)):
    stream = io.BytesIO()
    Image.new('RGB', size, (100,120,140)).save(stream, format=format)
    return stream.getvalue()

def test_prediction_and_request_id(client):
    result = client.post('/predict', files={'file': ('a.png', image(), 'image/png')}, headers={'X-Request-ID':'test-123'})
    assert result.status_code == 200
    assert result.headers['X-Request-ID'] == 'test-123'
    body=result.json()
    assert body['predicted_class'] == 'benign'
    assert len(body['all_predictions']) == 3
    assert body['serving_mode'] == 'efficientnet'
    assert body['model_version']

@pytest.mark.parametrize('data,mime,expected', [(b'bad','text/plain',415),(b'bad','image/png',415),(b'\x89PNG\r\n\x1a\nbad','image/png',422),(image('JPEG'),'image/png',415),(image(size=(8,8)),'image/png',422)])
def test_invalid_inputs(client,data,mime,expected):
    result=client.post('/predict',files={'file':('a.png',data,mime)})
    assert result.status_code == expected
    assert 'request_id' in result.json()

def test_size_bound(client,monkeypatch):
    monkeypatch.setattr(reliability,'MAX_BYTES',10)
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 413

def test_unready_liveness(client,monkeypatch):
    monkeypatch.setattr(main,'model_efficientnet',None)
    assert client.get('/health').status_code == 200
    assert client.get('/ready').status_code == 503
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 503

def test_degraded_readiness(client):
    assert client.get('/ready').json()['mode']=='degraded'

def test_busy_backpressure(client,monkeypatch):
    # A waiter that cannot be admitted within the bounded wait is rejected, not queued forever.
    monkeypatch.setattr(reliability,'QUEUE_WAIT_SECONDS',0.1)
    with reliability.prediction_slot():
        assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 429
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 200

def test_full_wait_queue_rejects_immediately(client,monkeypatch):
    monkeypatch.setattr(reliability,'MAX_WAITING',0)
    with reliability.prediction_slot():
        started=time.perf_counter()
        assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 429
        assert time.perf_counter()-started < 1

def test_waiter_is_served_when_slot_frees(client):
    import threading
    reliability.slot.acquire()
    threading.Timer(0.3,reliability.slot.release).start()
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 200

def test_rate_limit(client,monkeypatch):
    monkeypatch.setenv('RATE_LIMIT_PER_MINUTE','1')
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 200
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code == 429

def test_preprocessing_equivalence():
    data=image()
    efficient=main.preprocess_image(data)
    assert efficient.shape == (1,224,224,3)
    assert np.allclose(efficient[0,0,0],[100,120,140])
    assert np.allclose(main.preprocess_image(data,'vgg16')[0,0,0],[140-103.939,120-116.779,100-123.68])

def test_metrics(client):
    response=client.get('/metrics')
    assert response.status_code==200
    assert 'sehat_requests' in response.text

def test_pixel_bound(client,monkeypatch):
    monkeypatch.setattr(reliability,'MAX_PIXELS',100)
    assert client.post('/predict',files={'file':('a.png',image(),'image/png')}).status_code==422

@pytest.mark.parametrize('failure,expected',[('efficientnet',False),('mapping',False),('vgg16',True),('vgg_checksum',True)])
def test_model_startup_failure_and_fallback(monkeypatch,failure,expected):
    def load(path,compile=False):
        assert compile is False
        if (failure=='efficientnet' and 'efficientnet' in path) or (failure=='vgg16' and 'final_model2' in path):raise OSError('Fixture unavailable')
        return Model()
    fake=SimpleNamespace(config=SimpleNamespace(threading=SimpleNamespace(set_inter_op_parallelism_threads=lambda n:None,set_intra_op_parallelism_threads=lambda n:None)),keras=SimpleNamespace(models=SimpleNamespace(load_model=load)),TensorSpec=lambda *a:None,float32='float32',function=lambda fn,**kwargs:fn)
    monkeypatch.setitem(sys.modules,'tensorflow',fake)
    def checksum(path,variable):
        if failure=='vgg_checksum' and variable=='VGG16_SHA256':raise ValueError('Fixture checksum mismatch')
    monkeypatch.setattr(main,'model_checksum',checksum)
    original_open=builtins.open
    def fixture_open(path,*args,**kwargs):
        if str(path)=='resultsskinwise/class_index.pkl':
            if failure=='mapping':raise FileNotFoundError('Fixture mapping missing')
            return io.BytesIO(pickle.dumps({'benign':0,'melanoma':1,'other':2}))
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',fixture_open)
    try:
        main.load_model_and_mappings()
        assert bool(main.model_efficientnet) is expected
        if expected:assert main.model_vgg16 is None
    finally:asyncio.run(main.shutdown_event())
