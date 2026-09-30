"""Bounded burst through the real ASGI API; native RSS is not container memory."""
import argparse
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import psutil
from PIL import Image
from fastapi.testclient import TestClient
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
parser=argparse.ArgumentParser();parser.add_argument('--requests',type=int,default=100);parser.add_argument('--concurrency',type=int,default=4);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
os.environ['RATE_LIMIT_PER_MINUTE']='10000'  # Capacity scenario isolates concurrency from the rate-limit suite.
import main
peak=[0];done=threading.Event()
def sample():
    while not done.wait(.02):peak[0]=max(peak[0],psutil.Process().memory_info().rss)
threading.Thread(target=sample,daemon=True).start()
buffer=io.BytesIO();Image.new('RGB',(224,224),(100,120,140)).save(buffer,format='PNG');image=buffer.getvalue()
with TestClient(main.app) as client:
    # Real-model warmup so startup graph tracing is measured separately from the burst.
    assert client.post('/predict',files={'file':('fixture.png',image,'image/png')}).status_code==200
    def request(index):
        start=time.perf_counter()
        response=client.post('/predict',files={'file':('fixture.png',image,'image/png')})
        return {'request':index,'status':response.status_code,'seconds':time.perf_counter()-start,'request_id':response.headers.get('X-Request-ID')}
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:results=list(executor.map(request,range(args.requests)))
    duration=time.perf_counter()-start
    ready=client.get('/ready').status_code
done.set()
report={'scope':'native Windows ASGI API, real models; bounded burst, not sustained arrival-rate capacity','requests':args.requests,'concurrency':args.concurrency,'status_counts':{str(s):sum(x['status']==s for x in results) for s in sorted({x['status'] for x in results})},'all_response_p50_seconds':float(np.percentile([x['seconds'] for x in results],50)),'all_response_p95_seconds':float(np.percentile([x['seconds'] for x in results],95)),'successful_inferences_per_second':sum(x['status']==200 for x in results)/duration,'duration_seconds':duration,'ready_after_burst':ready,'peak_process_rss_bytes':peak[0],'container_memory':'not measured','raw_requests':results}
args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='raw_requests'}))
