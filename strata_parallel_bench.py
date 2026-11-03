"""One-shot check: are parallel slots live, and does 2-way concurrency help?"""
import json, sys, time, urllib.request, concurrent.futures
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = 'http://192.168.1.85:8080'
KEY = 'sk-strata-3643ba2f77b5ccf84aba6bd7e42c8e20'

def get(path):
    req = urllib.request.Request(BASE + path, headers={'Authorization': 'Bearer ' + KEY})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode('utf-8'))

def completion(seed):
    body = {
        'model': 'default',
        'messages': [{'role': 'user',
                      'content': f'Reply with exactly this sequence, nothing else: {seed} one two three four five six seven eight nine ten'}],
        'max_tokens': 64,
        'temperature': 0,
        'stream': False,
    }
    req = urllib.request.Request(BASE + '/v1/chat/completions',
        data=json.dumps(body).encode('utf-8'),
        headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode('utf-8'))
    el = time.monotonic() - t0
    u = d.get('usage', {})
    return {'seconds': round(el, 2), 'completion_tokens': u.get('completion_tokens'),
            'tok_s': round((u.get('completion_tokens') or 0) / el, 2)}

props = get('/props')
print('props keys:', sorted(k for k in props if not isinstance(props[k], (list, dict)) or len(str(props[k])) < 120)[:40])
for k in ('parallel', 'slots', 'batch', 'slot_n', 'n_slots', 'n_ctx', 'ctx_size', 'model_type'):
    if k in props:
        print('PROP', k, '=', props[k])
try:
    slots = get('/slots')
    print('slots:', json.dumps(slots)[:800])
except Exception as e:
    print('slots endpoint:', e)

print('\n--- single request (baseline) ---')
print(json.dumps(completion('A')))

print('\n--- two requests concurrently ---')
t0 = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
    futs = [ex.submit(completion, s) for s in ('B', 'C')]
    r1, r2 = [f.result() for f in futs]
wall = time.monotonic() - t0
print('req1', json.dumps(r1))
print('req2', json.dumps(r2))
print('wall', round(wall, 2), 's; aggregate tok/s:',
      round(((r1['completion_tokens'] or 0) + (r2['completion_tokens'] or 0)) / wall, 2))
