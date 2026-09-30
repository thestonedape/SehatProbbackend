"""Float32 conversion experiment. Never changes the serving runtime automatically."""
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
root = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import main
main.load_model_and_mappings()
import tensorflow as tf
out = root/'artifacts'; out.mkdir(exist_ok=True)
report = {'adopted':False, 'corpus':'50 synthetic PNGs, seed 7300930; numerical verification, not accuracy evidence','models':{}}
rng = np.random.default_rng(7300930)
images=[]
for i in range(50):
    pixels=rng.integers(0,256,(224,224,3),dtype=np.uint8)
    if i<10: pixels[:]=i*25
    buffer=io.BytesIO();Image.fromarray(pixels).save(buffer,format='PNG');images.append(buffer.getvalue())
report['corpus_sha256']=[hashlib.sha256(x).hexdigest() for x in images]
for name,model in [('efficientnet',main.model_efficientnet),('vgg16',main.model_vgg16)]:
    if model is None:
        report['models'][name]={'passed':False,'error':'Model unavailable'};continue
    try:
        # Saved EfficientNet uses mixed-float16 compute, unsupported by the
        # built-in TFLite CPU ops. Clone its config with float32 compute and
        # unchanged weights; parity below must verify this precision change.
        config=json.loads(model.to_json())
        def float32_policy(value):
            if isinstance(value,dict):
                if value.get('class_name')=='DTypePolicy': value['config']['name']='float32'
                for key,item in list(value.items()):
                    if key=='dtype' and isinstance(item,str) and item in {'float16','mixed_float16'}: value[key]='float32'
                    else: float32_policy(item)
            elif isinstance(value,list):
                for item in value: float32_policy(item)
        float32_policy(config)
        converted_model=tf.keras.models.model_from_json(json.dumps(config))
        converted_model.set_weights(model.get_weights())
        converter=tf.lite.TFLiteConverter.from_keras_model(converted_model)
        converter.optimizations=[]
        converter.target_spec.supported_types=[tf.float32]
        converted=converter.convert()
        (out/f'{name}.tflite').write_bytes(converted)
        runtime=tf.lite.Interpreter(model_content=converted,num_threads=1);runtime.allocate_tensors()
        inp=runtime.get_input_details()[0];output=runtime.get_output_details()[0]
        differences=[];labels=[];rankings=[]
        for data in images:
            batch=main.preprocess_image(data,name)
            original=np.asarray(model(batch,training=False))[0]
            runtime.set_tensor(inp['index'],batch);runtime.invoke();actual=runtime.get_tensor(output['index'])[0]
            differences.append(float(np.max(np.abs(original-actual))))
            labels.append(bool(np.argmax(original)==np.argmax(actual)))
            rankings.append(bool(np.array_equal(np.argsort(original)[-3:][::-1],np.argsort(actual)[-3:][::-1])))
        report['models'][name]={'passed':max(differences)<=.001 and all(labels) and all(rankings),'maximum_absolute_probability_difference':max(differences),'unchanged_labels':sum(labels),'unchanged_top_three':sum(rankings),'raw_max_differences':differences,'bytes':len(converted)}
    except Exception as exc:
        report['models'][name]={'passed':False,'error_type':type(exc).__name__,'error':str(exc)[:2000]}
    (out/'tflite-verification.json').write_text(json.dumps(report,indent=2))
report['promotion_gate']='Individual numerical checks alone do not verify ensemble clinical ranking or resource improvement. TensorFlow remains serving runtime.'
(out/'tflite-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
