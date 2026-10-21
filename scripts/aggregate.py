"""Flatten every repeat-run raw result into one CSV for charts and error analysis."""
import csv, glob, json, os, re, sys
MODELS = {  # models in the main statistics; newer folders (rebuilds, 256k variants, ...) are never pulled in
    'devstral2-24b-q4km', 'gemma4-26b-a4b-qat', 'gemma4-31b-qat', 'glm47flash-iq4xs', 'gpt61sol-medium',
    'gpt6luna-medium', 'gpt6luna-xhigh', 'gptoss-20b-mxfp4', 'granite41-30b-q4km-kvq4', 'haiku55-medium',
    'haiku55-xhigh', 'museglimmer-30b-q4km', 'museglimmer-30b-q4km-dflash16', 'nemotron3nano-30b-iq4xs',
    'ornith10-q4km', 'ornith15-q4km', 'qwen36-27b-q4km', 'qwen36-35b-iq4xs', 'qwen38-27b-iq4xs',
    'qwen38flashnext-gsq-strata', 'qwen38flashnext-iq3xxs', 'qwen3coder-30b-iq4xs',
    'qwen3codernext-iq4xs-offload', 'sonnet55-medium',
}


def newest_runs():
    """One raw result per (model, task, seed): the newest timestamp in the file name (a re-grade supersedes older files)."""
    if 'BENCH_RAW' not in os.environ:
        sys.exit(r"Set BENCH_RAW to the benchmark's results\raw folder.")
    best = {}
    for f in glob.glob(os.path.join(os.environ['BENCH_RAW'], '*', 'repeat', '*.json')):
        d = json.load(open(f, encoding='utf-8'))
        if d['model'] not in MODELS:
            continue
        stamp = re.search(r'__(\d{8}-\d{6})\.json$', f).group(1)
        key = (d['model'], d.get('task') or d['test_id'][0], d.get('seed'))
        if key not in best or stamp > best[key][0]:
            best[key] = (stamp, d)
    return [d for _, d in best.values()]

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'repeat-flat.csv')
DIMS = ('correctness', 'code_quality', 'reasoning', 'repo_understanding', 'tool_use', 'hallucination', 'instruction')
rows = []
for d in newest_runs():
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
