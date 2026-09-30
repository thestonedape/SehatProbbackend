"""Experimental TensorFlow frozen serving graph; retains the original dtype policy.

This is not a float32 TFLite conversion. Never select it for serving before parity
and resource verification. Original Keras files remain the rollback.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['efficientnet', 'vgg16'], required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--disable-meta-optimizer', action='store_true')
parser.add_argument('--baseline-report', type=Path, help='Original default-runtime probabilities; required when changing optimizer settings')
args = parser.parse_args()
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
if args.disable_meta_optimizer:
    if not args.baseline_report:
        raise SystemExit('Optimizer experiment requires an original default-runtime baseline report')
    tf.config.optimizer.set_experimental_options({'disable_meta_optimizer': True})
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from main import preprocess_image
source = root / ('resultsskinwise/efficientnet_final.keras' if args.model == 'efficientnet' else 'final_model2.keras')
model = tf.keras.models.load_model(source, compile=False)
signature = [tf.TensorSpec([1, 224, 224, 3], tf.float32)]
original = tf.function(lambda batch: model(batch, training=False), input_signature=signature)
frozen = convert_variables_to_constants_v2(original.get_concrete_function(), lower_control_flow=False)
module = tf.Module()
module.infer = tf.function(lambda batch: {'probabilities': frozen(batch)[0]}, input_signature=signature)
destination = args.output_dir / args.model
tf.saved_model.save(module, str(destination), signatures={'serving_default': module.infer},
                    options=tf.saved_model.SaveOptions(experimental_custom_gradients=False))
restored = tf.saved_model.load(str(destination)).signatures['serving_default']
rng = np.random.default_rng(7300930)
baseline, candidate, hashes = [], [], []
for i in range(50):
    pixels = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    if i < 10:
        pixels[:] = i * 25
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format='PNG')
    raw = buffer.getvalue()
    hashes.append(hashlib.sha256(raw).hexdigest())
    batch = tf.convert_to_tensor(preprocess_image(raw, args.model))
    baseline.append(np.asarray(original(batch))[0].astype(float).tolist())
    candidate.append(np.asarray(restored(batch=batch)['probabilities'])[0].astype(float).tolist())
left, right = np.asarray(baseline), np.asarray(candidate)
if args.baseline_report:
    reference = json.loads(args.baseline_report.read_text())
    if reference['source_sha256'] != hashlib.sha256(source.read_bytes()).hexdigest() or reference['corpus_sha256'] != hashes:
        raise SystemExit('Reference model/corpus does not match')
    baseline = reference['original_probabilities']
    left = np.asarray(baseline)
difference = float(np.max(np.abs(left-right)))
labels = int(np.sum(left.argmax(axis=1) == right.argmax(axis=1)))
report = {'scope': 'experimental frozen TensorFlow graph; synthetic numerical verification, not accuracy evidence',
          'model': args.model, 'tensorflow_version': tf.__version__, 'keras_version': tf.keras.__version__,
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'original_dtype_policy': model.dtype_policy.name,
          'disable_meta_optimizer': args.disable_meta_optimizer,
          'inputs': 50, 'unchanged_labels': labels, 'max_absolute_probability_difference': difference,
          'passed': labels == 50 and difference <= 0.001, 'corpus_sha256': hashes,
          'original_probabilities': baseline, 'frozen_probabilities': candidate,
          'files': {str(path.relative_to(destination)): {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                    for path in destination.rglob('*') if path.is_file()}}
args.output_dir.mkdir(parents=True, exist_ok=True)
(args.output_dir / f'{args.model}-verification.json').write_text(json.dumps(report, indent=2))
print(json.dumps({key: value for key, value in report.items() if key not in {'files', 'original_probabilities', 'frozen_probabilities', 'corpus_sha256'}}))
if not report['passed']:
    raise SystemExit(1)
