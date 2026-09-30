import hashlib
import importlib.util
import io
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('space_app', Path(__file__).resolve().parents[1] / 'space/space_app.py')
space = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space)


def test_artifact_verification_and_cleanup(tmp_path, monkeypatch):
    good = b'pinned model fixture'
    item = {'path':'models/model.bin','url':'https://example.invalid/model','bytes':len(good),'sha256':hashlib.sha256(good).hexdigest()}
    monkeypatch.setattr(space.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(good))
    space.materialize_models([item],tmp_path)
    assert (tmp_path/item['path']).read_bytes() == good
    monkeypatch.setattr(space.urllib.request,'urlopen',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('cached artifact must not download')))
    space.materialize_models([item],tmp_path)
    (tmp_path/item['path']).unlink()
    monkeypatch.setattr(space.urllib.request,'urlopen',lambda *a,**kw:io.BytesIO(b'x'*len(good)))
    with pytest.raises(RuntimeError,match='checksum'):
        space.materialize_models([item],tmp_path)
    assert not (tmp_path/item['path']).exists()
    assert not list(tmp_path.rglob('*.download'))


def test_transient_download_retry_is_bounded(tmp_path, monkeypatch):
    calls = []
    item = {'path':'model.bin','url':'https://example.invalid/model','bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}
    monkeypatch.setattr(space.time, 'sleep', lambda delay: None)
    def unavailable(*args, **kwargs):
        calls.append(1)
        raise space.urllib.error.URLError('transient fixture failure')
    monkeypatch.setattr(space.urllib.request, 'urlopen', unavailable)
    with pytest.raises(space.urllib.error.URLError):
        space.materialize_models([item], tmp_path)
    assert len(calls) == 3
    assert not list(tmp_path.rglob('*.download'))


def test_oversized_artifact_is_rejected_without_retry(tmp_path, monkeypatch):
    calls = []
    def download(*args, **kwargs):
        calls.append(1)
        return io.BytesIO(b'oversized')
    item = {'path':'model.bin','url':'https://example.invalid/model','bytes':1,'sha256':hashlib.sha256(b'a').hexdigest()}
    monkeypatch.setattr(space.urllib.request, 'urlopen', download)
    with pytest.raises(RuntimeError, match='pinned size'):
        space.materialize_models([item], tmp_path)
    assert len(calls) == 1
    assert not (tmp_path / 'model.bin').exists()
    assert not list(tmp_path.rglob('*.download'))
