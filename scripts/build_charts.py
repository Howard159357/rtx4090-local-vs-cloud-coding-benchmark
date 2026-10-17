"""Render every chart in its own process with a timeout and retry.

matplotlib on this Windows machine occasionally stalls mid-run (seen at random charts); a fresh process per chart
with a 90 s timeout makes the build reliable.  Usage: python build_charts.py [out_dir]"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'docs', 'assets', 'charts')
CHARTS = [f'c{i:02d}' for i in range(1, 20)]
env = dict(os.environ, MPLBACKEND='Agg')
failed = []
for c in CHARTS:
    for attempt in range(3):
        try:
            r = subprocess.run([sys.executable, '-I', os.path.join(HERE, 'make_charts.py'), OUT, c], env=env,
                               capture_output=True, text=True, timeout=90)
            print(c, (r.stdout.strip() or r.stderr.strip()[-300:]), flush=True)
            if r.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            print(c, f'timeout, retry {attempt + 1}', flush=True)
    else:
        failed.append(c)
print('FAILED: ' + ', '.join(failed) if failed else 'all charts built')
