import json, sys, urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
IPS=['10.32.9.182','10.32.4.42']
token=Path('/mnt/data/butong/keeper/.token').read_text().strip()
def api(ip,path,payload=None):
 data=None if payload is None else json.dumps(payload).encode()
 req=urllib.request.Request(f'http://{ip}:9807/{path}',data=data,headers={'X-Keeper-Token':token,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=20) as r:return json.load(r)
if sys.argv[1]=='start':
 payload=json.loads(Path(sys.argv[2]).read_text())
 for ip in IPS:
  tasks=api(ip,'tasks')['tasks']; assert not any(t['status']=='running' for t in tasks), (ip,'running task')
  s=api(ip,'status'); assert len(s['gpus'])==8
  assert all(int(g.split('mem=')[1].split('/')[0])<1024 for g in s['gpus']), (ip,'busy GPUs')
 for ip in IPS:
  r=api(ip,'task/start',payload)
  with (ROOT/'ledger.jsonl').open('a') as f:f.write(json.dumps({'ip':ip,'task':r,'name':payload['name']})+'\n')
  print(json.dumps({'ip':ip,'result':r}))
else:
 for ip in IPS:
  ts=api(ip,'tasks')['tasks']; wanted=[t for t in ts if t.get('name','').startswith('pdd-wan14b-')]
  for t in wanted[:3]:
   print(json.dumps({'ip':ip,'task':t}))
   print(json.dumps(api(ip,'task/'+t['id']+'/log?n=12')))
