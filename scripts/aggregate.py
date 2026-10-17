"""Flatten every repeat-run raw result into one CSV for charts and error analysis."""
import csv, glob, json, os, sys
RAW = r'C:\AI-Projects\coding-llm-benchmark\results\raw'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'repeat-flat.csv')
DIMS = ('correctness', 'code_quality', 'reasoning', 'repo_understanding', 'tool_use', 'hallucination', 'instruction')
rows = []
for f in glob.glob(os.path.join(RAW, '*', 'repeat', '*.json')):
    d = json.load(open(f, encoding='utf-8'))
    llm = d.get('llm') or {}
    if isinstance(llm, list):
        llm = (llm[-1] if llm else {}) or {}
        llm = llm if isinstance(llm, dict) else {}
    usage = llm.get('usage') or {}
    tok = d.get('tokens') or {}
    m = d.get('metrics') or {}
    r = {'model': d['model'], 'task': d.get('task') or d['test_id'][0], 'seed': d.get('seed'), 'status': d.get('status'),
         'total70': (d.get('scores') or {}).get('total70'), 'duration_s': d.get('duration_s'),
         'finish_reason': llm.get('finish_reason'), 'completion_tokens': tok.get('completion') or usage.get('completion_tokens'),
         'reasoning_tokens': tok.get('reasoning'), 'prompt_tokens': tok.get('prompt') or usage.get('prompt_tokens'),
         'decode_tps': (llm.get('timings') or {}).get('predicted_per_second'),
         'prompt_tps': (llm.get('timings') or {}).get('prompt_per_second'),
         'fake_toolcall': '<tool_call>' in (llm.get('content') or ''),
         'content_len': len(llm.get('content') or ''),
         'metrics': json.dumps({k: v for k, v in m.items() if not isinstance(v, (list,)) and k != 'files'}, ensure_ascii=False)[:600]}
    for k in DIMS:
        r[k] = (d.get('scores') or {}).get(k)
    rows.append(r)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['model'], r['task'], r['seed'] or 0)))
print(len(rows), 'rows ->', OUT)
