"""Per-model pass rates of key sub-checks (error analysis), from repeat-run raw JSON."""
import collections, csv, glob, json, os
RAW = r'C:\AI-Projects\coding-llm-benchmark\results\raw'
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'checks.csv')
acc = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(os.path.join(RAW, '*', 'repeat', '*.json')):
    d = json.load(open(f, encoding='utf-8'))
    m, t, x = d['model'], d.get('task') or d['test_id'][0], d.get('metrics') or {}
    a = acc[m]
    if t == 'A':
        a['A program runs'].append(bool(x.get('runs')))
        a['A hidden data exact'].append((x.get('hidden_field_accuracy') or 0) >= 0.999)
    elif t == 'B':
        a['B all tests pass'].append(bool(x.get('all_tests_pass')))
    elif t == 'C':
        q = x.get('query_correct') or {}
        a['C 5/5 SQL correct'].append(bool(q) and all(q.values()))
        a['C Q5 2x faster'].append(bool(x.get('q5_faster_2x')))
    elif t == 'D':
        a['D vue-tsc passes'].append(bool(x.get('typescript_ok')))
        h = str(x.get('hidden_backend_tests') or '0/1').split('/')
        a['D hidden backend tests'].append(h[0] == h[1])
    elif t == 'E':
        r = x.get('recall_by_question') or {}
        for q, lab in (('Q3', 'E coupling'), ('Q4', 'E race-condition'), ('Q5', 'E memory-leak')):
            if q in r:
                a[lab + ' recall'].append(r[q])
    elif t == 'F':
        a['F right root cause'].append(bool(x.get('root_cause_correct')))
    elif t == 'G':
        a['G hidden tests pass'].append(bool(x.get('success')))
    elif t == 'H':
        v = x.get('variants') or {}
        mids = [bool(((vv.get('per_needle') or {}).get('middle'))) for vv in v.values() if isinstance(vv, dict)]
        fc = [vv.get('fields_correct', 0) / max(1, vv.get('fields_total', 9)) for vv in v.values() if isinstance(vv, dict)]
        if fc:
            a['H 9/9 fields'].append(all(c >= 0.999 for c in fc))
cols = sorted({k for a in acc.values() for k in a})
with open(OUT, 'w', newline='', encoding='utf-8-sig') as fh:
    w = csv.writer(fh); w.writerow(['model'] + cols)
    for m, a in sorted(acc.items()):
        w.writerow([m] + [round(sum(map(float, a[c])) / len(a[c]), 2) if a[c] else '' for c in cols])
print(open(OUT, encoding='utf-8-sig').read())
