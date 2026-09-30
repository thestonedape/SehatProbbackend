"""Real-model smoke check for the packaged Space, independent of lightweight CI."""
import io
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
from PIL import Image
from space_app import serving_app


def main():
    started = time.perf_counter()
    app = serving_app()
    with TestClient(app) as client:
        readiness = client.get('/ready')
        assert readiness.status_code == 200, readiness.text
        assert readiness.json()['mode'] == 'ensemble'
        assert client.get('/health').status_code == 200
        assert client.get('/status/').status_code == 200
        image = io.BytesIO()
        Image.new('RGB', (224, 224), (100, 120, 140)).save(image, format='PNG')
        before_prediction = time.perf_counter()
        result = client.post('/predict', files={'file': ('fixture.png', image.getvalue(), 'image/png')})
        assert result.status_code == 200, result.text
        prediction = result.json()
        assert len(prediction['all_predictions']) == 3
        assert prediction['serving_mode'] == 'ensemble'
        assert client.post('/predict', files={'file': ('bad.png', b'corrupt', 'image/png')}).status_code == 415
        report = {'scope': 'local real-model packaged Space smoke test; synthetic input, no accuracy claim',
                  'ready': readiness.json(), 'prediction': prediction,
                  'prediction_seconds': time.perf_counter() - before_prediction,
                  'startup_and_test_seconds': time.perf_counter() - started}
    output = Path(__file__).resolve().parents[1] / 'artifacts/space-local-smoke.json'
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
