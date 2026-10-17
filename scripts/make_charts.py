"""Charts for the RTX 4090 local-vs-cloud coding benchmark write-up.

Data: data/source/*.csv (copied from coding-llm-benchmark/benchmark-results), data/source/multiuser.json and
strata-perf.json, plus data/repeat-flat.csv and data/checks.csv built by aggregate.py / checks.py from the raw
repeat-run JSON. Output: docs/assets/charts/*.png (1080x1350). Usage: make_charts.py OUT_DIR [c01 c02 ...]"""
import csv
import json
import logging
import os
import statistics
import sys
import warnings
from collections import defaultdict

# Missing-glyph / font-weight warnings flood stderr; when a caller stops reading the pipe the process blocks.
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Patch, Rectangle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BR = os.path.join(HERE, 'data', 'source')
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'docs', 'assets', 'charts')
os.makedirs(OUT, exist_ok=True)

SURF, INK, INK2, MUTED, GRID = '#F9F9F4', '#1F2328', '#4B5563', '#8A8F98', '#E3E3DC'
GREEN, BLUE, ORANGE, PURPLE, TEAL, RED = '#76B900', '#3B6FC4', '#F28E2B', '#9B5FC9', '#1FA3A3', '#C8102E'
LOCAL, CLOUD = BLUE, ORANGE
GOOD = LinearSegmentedColormap.from_list('good', ['#F3F6EC', '#C9E29A', '#76B900', '#3F6B00'])
FOOT = ('RTX 4090 24 GB · 64 GB RAM | llama.cpp b10929 / b11609, Strata | 8 tasks x 3 runs, temp 0.6 | '
        'tasks & graders by Claude Opus 5.5 | not a leaderboard')

plt.rcParams.update({
    'font.family': ['Arial', 'DejaVu Sans'], 'font.size': 13, 'axes.edgecolor': GRID, 'axes.linewidth': 1,
    'axes.facecolor': SURF, 'figure.facecolor': SURF, 'xtick.color': INK2, 'ytick.color': INK2,
    'axes.labelcolor': INK2, 'text.color': INK, 'axes.titlecolor': INK,
})

STRATA, FN_LLAMA, FN_MTP = 'qwen38flashnext-gsq-strata', 'qwen38flashnext-iq3xxs', 'qwen38flashnext-iq3xxs-b11609-mtp'
MUSE, MUSE_DF = 'museglimmer-30b-q4km', 'museglimmer-30b-q4km-dflash16'
NAMES = {
    'sonnet55-medium': 'Claude Sonnet 5.5 · medium', 'gpt61sol-medium': 'GPT-6.1 Sol · medium',
    'haiku55-medium': 'Claude Haiku 5.5 · medium', 'gpt6luna-xhigh': 'GPT-6 Luna · xhigh',
    'haiku55-xhigh': 'Claude Haiku 5.5 · xhigh', 'gpt6luna-medium': 'GPT-6 Luna · medium',
    STRATA: 'Qwen3.8-Flash-Next · Strata*', FN_LLAMA: 'Qwen3.8-Flash-Next · llama.cpp*',
    FN_MTP: 'Qwen3.8-Flash-Next + MTP*',
    MUSE: 'Muse-Glimmer-30B', MUSE_DF: 'Muse-Glimmer-30B + DFlash',
    'qwen36-27b-q4km': 'Qwen3.6-27B', 'qwen36-27b-q4km-mtp6': 'Qwen3.6-27B + MTP',
    'ornith15-q4km': 'Ornith-1.5-35B-A3B', 'qwen36-35b-iq4xs': 'Qwen3.6-35B-A3B', 'gemma4-31b-qat': 'Gemma-4-31B',
    'ornith10-q4km': 'Ornith-1.0-35B', 'qwen38-27b-iq4xs': 'Qwen3.8-27B', 'qwen38-27b-iq4xs-mtp6': 'Qwen3.8-27B + MTP',
    'qwen3codernext-iq4xs-offload': 'Qwen3-Coder-Next 80B*', 'qwen3coder-30b-iq4xs': 'Qwen3-Coder-30B-A3B',
    'gptoss-20b-mxfp4': 'gpt-oss-20b', 'gemma4-26b-a4b-qat': 'Gemma-4-26B-A4B', 'devstral2-24b-q4km': 'Devstral-Small-2-24B',
    'granite41-30b-q4km-kvq4': 'Granite-4.1-30B', 'glm47flash-iq4xs': 'GLM-4.7-Flash',
    'nemotron3nano-30b-iq4xs': 'Nemotron-3-Nano-30B',
}
# How each local setup generates: dense on GPU, MoE, MoE with weights partly in RAM, + drafter, custom engine.
KIND = {MUSE: 'dense', 'qwen36-27b-q4km': 'dense', 'gemma4-31b-qat': 'dense', 'qwen38-27b-iq4xs': 'dense',
        'devstral2-24b-q4km': 'dense', 'granite41-30b-q4km-kvq4': 'dense',
        MUSE_DF: 'drafter', 'qwen36-27b-q4km-mtp6': 'drafter', 'qwen38-27b-iq4xs-mtp6': 'drafter', FN_MTP: 'drafter',
        'ornith15-q4km': 'moe', 'ornith10-q4km': 'moe', 'qwen36-35b-iq4xs': 'moe', 'qwen3coder-30b-iq4xs': 'moe',
        'gptoss-20b-mxfp4': 'moe', 'gemma4-26b-a4b-qat': 'moe', 'glm47flash-iq4xs': 'moe',
        'nemotron3nano-30b-iq4xs': 'moe', 'qwen3codernext-iq4xs-offload': 'moe', FN_LLAMA: 'moe', STRATA: 'engine'}
KIND_COL = {'dense': BLUE, 'moe': TEAL, 'drafter': GREEN, 'engine': PURPLE}
KIND_LAB = [('dense model', BLUE), ('MoE (few active params)', TEAL), ('+ speculative drafter', GREEN),
            ('Strata engine', PURPLE)]
CLOUD_IDS = ['sonnet55-medium', 'gpt61sol-medium', 'haiku55-medium', 'gpt6luna-xhigh', 'haiku55-xhigh', 'gpt6luna-medium']
TASKS = 'ABCDEFGH'
TASK_NAMES = {'A': 'Python', 'B': 'Debug', 'C': 'SQL', 'D': 'Multi-file', 'E': 'Repo read', 'F': 'Log RCA',
              'G': 'Agent', 'H': 'Long doc'}
WEIGHTS = {'G': 25, 'E': 20, 'B': 15, 'H': 15, 'D': 10, 'A': 5, 'C': 5, 'F': 5}
STAR = '* weights partly in system RAM (Flash-Next: 43-56 GB, Coder-Next: 16 GB)'


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


REP = {r['model']: r for r in read(os.path.join(BR, 'repeat-runs.csv'))}
MASTER = {r['model']: r for r in read(os.path.join(BR, 'master-table.csv'))}
LONG = read(os.path.join(BR, 'long-context.csv'))
P5 = read(os.path.join(BR, 'phase5-longctx.csv'))
P5B = read(os.path.join(BR, 'phase5b-256k.csv'))
MULTI = json.load(open(os.path.join(BR, 'multiuser.json'), encoding='utf-8'))
CHECKS = {r['model']: r for r in read(os.path.join(HERE, 'data', 'checks.csv'))}
FLAT = read(os.path.join(HERE, 'data', 'repeat-flat.csv'))
# Strata long-context H runs (results/raw/qwen38flashnext-gsq-strata/longctx): (tokens, ttft s, (start, mid, end))
STRATA_LC = {'128K': (116849, 27.99, (True, True, True)), '256K': (234890, 56.78, (True, False, True))}
LOCAL_IDS = [m for m in REP if m not in CLOUD_IDS and m != MUSE_DF]
RANKED = [m for m in sorted(REP, key=lambda m: -fnum(REP[m]['weighted_mean'])) if m != MUSE_DF]
TOP_LOCAL = (MUSE, STRATA)


# ---------------------------------------------------------------- layout helpers
def frame(title, subtitle, note_title, note, tsize=30, note_h=0.115):
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100)
    fig.text(0.065, 0.955, title, fontsize=tsize, fontweight='black', va='top')
    fig.text(0.065, 0.905, subtitle, fontsize=15, color=INK2, va='top')
    if note_title:
        fig.patches.append(FancyBboxPatch((0.065, 0.235 - note_h), 0.87, note_h,
                                          boxstyle='round,pad=0.01,rounding_size=0.012', transform=fig.transFigure,
                                          fc='white', ec='#CFCFC6', lw=1.2))
        fig.text(0.088, 0.222, note_title, fontsize=15, fontweight='bold', va='top')
        fig.text(0.088, 0.195, note, fontsize=13.5, color=INK2, va='top', linespacing=1.45)
    fig.text(0.065, 0.04, FOOT, fontsize=9.6, color=INK2)
    return fig


def style(ax, ylabel='', xlabel='', grid_axis='both'):
    ax.grid(True, color=GRID, lw=1, axis=grid_axis)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=13.5)
    ax.set_xlabel(xlabel, fontsize=13.5)
    ax.tick_params(length=0)


def card(fig, x, y, w, h, fc='white'):
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.006,rounding_size=0.012',
                                      transform=fig.transFigure, fc=fc, ec='#CFCFC6', lw=1.2))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'))
    plt.close(fig)
    print('saved', name)


def legend_bars(ax, items, loc='lower right', **kw):
    ax.legend(handles=[Patch(color=c, label=l) for l, c in items], loc=loc, frameon=False, fontsize=12, **kw)


def legend_dots(ax, items, loc='lower right', **kw):
    h = [Line2D([], [], marker='o', ls='', ms=10, color=c, label=l) for l, c in items]
    ax.legend(handles=h, loc=loc, frameon=False, fontsize=12, **kw)


def bold_ticks(ax, ms):
    for t, m in zip(ax.get_yticklabels(), ms):
        if m in TOP_LOCAL:
            t.set_fontweight('bold')


def ylabels(ms):
    return [('☁ ' if m in CLOUD_IDS else '') + NAMES[m] for m in ms]


# ---------------------------------------------------------------- 01 the test
def c01_tasks():
    fig = frame('Eight Tasks, One Factory', 'Claude Opus 5.5 wrote every prompt, fixture, hidden test and grader.', None, None)
    tasks = [
        ('A', 'Python CSV pipeline', 'Daily kWh, peak and data gaps from meter readings with bad rows mixed in.',
         'run on hidden data, field-level compare'),
        ('B', 'FastAPI debugging', 'Six planted bugs behind a production log and a failing test suite.',
         'original tests restored, async-blocking AST check'),
        ('C', 'SQL on a MES database', 'Five queries on SQLite incl. window functions; Q5 must get 2x faster.',
         'executed vs reference, EXPLAIN QUERY PLAN'),
        ('D', 'FastAPI + Vue 3 feature', 'New endpoint and page across backend and frontend, strict TypeScript.',
         'hidden backend tests, vue-tsc --noEmit'),
        ('E', 'Understand a 42-file repo', 'Architecture, data flow, coupling, race conditions, memory leaks.',
         'recall of planted issues, fake-file citations'),
        ('F', 'Incident log, ~37K tokens', 'One true root cause among symptoms and a CUDA-OOM red herring.',
         'root cause vs symptom, quoted timestamps exist'),
        ('G', 'Agent fixes a repo', 'Tools only (read, edit, run tests). Five bugs, no hints, 40 steps max.',
         'original + hidden tests on different data'),
        ('H', 'Long document, 3 rules', 'Rules planted at start, middle and end of a 32K-256K corpus.',
         '9 answer fields, each needs one rule'),
    ]
    cw, ch, gx, gy = 0.425, 0.168, 0.02, 0.016
    for i, (k, t, d, g) in enumerate(tasks):
        col, row = i % 2, i // 2
        x = 0.065 + col * (cw + gx)
        y = 0.86 - (row + 1) * ch - row * gy
        card(fig, x, y, cw, ch)
        fig.patches.append(Circle((x + 0.042, y + ch - 0.042), 0.026, transform=fig.transFigure, fc=GREEN, ec='none'))
        fig.text(x + 0.042, y + ch - 0.042, k, ha='center', va='center', fontsize=19, fontweight='bold', color='white')
        fig.text(x + 0.085, y + ch - 0.03, t, fontsize=15.5, fontweight='bold', va='top')
        fig.text(x + 0.085, y + ch - 0.058, f'weight {WEIGHTS[k]}%', fontsize=12.5, color=GREEN, fontweight='bold', va='top')
        fig.text(x + 0.022, y + 0.07, d, fontsize=12.2, color=INK2, va='top', wrap=True)
        fig.text(x + 0.022, y + 0.025, 'Graded by: ' + g, fontsize=10.8, color=MUTED, va='bottom', style='italic')
    for t in fig.texts:
        t._get_wrap_line_width = lambda w=cw: fig.bbox.width * (cw - 0.04)
    save(fig, '01_tasks')


# ---------------------------------------------------------------- 02 scoring
def c02_scoring():
    fig = frame('How a Score Is Earned', 'Seven dimensions per task, 0-10 each, 70 per task, weighted to 100.',
                'Grader self-checks before any model was scored',
                'Reference solutions score 70/70 on every task (G 69.2). A "do nothing" model scores low.\n'
                'Three grader bugs were found during the run (SQL column order, a false fake-file hit, CLI\n'
                'continuation joins); every affected model was re-graded and old results archived.')
    ax = fig.add_axes([0.17, 0.585, 0.75, 0.27])
    style(ax, '', 'Weight in the final score (%)', 'x')
    order = sorted(WEIGHTS, key=lambda k: WEIGHTS[k])
    cols = [GREEN if WEIGHTS[k] >= 15 else BLUE for k in order]
    bars = ax.barh([f'{k}  {TASK_NAMES[k]}' for k in order], [WEIGHTS[k] for k in order], color=cols, height=0.62, zorder=3)
    for b, k in zip(bars, order):
        ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f'{WEIGHTS[k]}%', va='center', fontsize=13,
                fontweight='bold')
    ax.set_xlim(0, 29)
    ax.set_title('Real-work tasks carry the weight: agent, repo reading, debugging, long docs = 75%', fontsize=13.5,
                 loc='left', fontweight='bold')
    dims = [('Correctness', 'runs the hidden tests / data'), ('Code quality', 'types, logging, error handling'),
            ('Reasoning', 'explains the real cause'), ('Repo understanding', 'names the right files'),
            ('Tool use', 'agent steps, no wasted calls'), ('Hallucination', 'cites only things that exist'),
            ('Instruction', 'format, no test edits, no new deps')]
    y0 = 0.535
    fig.text(0.065, y0, 'The seven dimensions', fontsize=15, fontweight='bold', va='top')
    for i, (d, w) in enumerate(dims):
        col, row = i % 2, i // 2
        x, y = 0.065 + col * 0.44, y0 - 0.045 - row * 0.04
        fig.patches.append(Rectangle((x, y - 0.012), 0.008, 0.024, transform=fig.transFigure, fc=BLUE, ec='none'))
        fig.text(x + 0.02, y, f'{d}', fontsize=13, fontweight='bold', va='center')
        fig.text(x + 0.2, y, w, fontsize=11.5, color=INK2, va='center')
    card(fig, 0.065, 0.255, 0.87, 0.075, fc='#EEF5E0')
    fig.text(0.088, 0.304, 'Correctness gate', fontsize=14, fontweight='bold', va='center')
    fig.text(0.088, 0.275, 'task score = mean(dimensions) x 7 x (0.5 + 0.05 x correctness)  ->  broken code with '
             'nice formatting cannot score high', fontsize=12, color=INK2, va='center')
    save(fig, '02_scoring')


# ---------------------------------------------------------------- 03 ranking
def c03_ranking():
    fig = frame('Cloud Leads, Two Local Setups Tie', 'Weighted score out of 100, mean of 3 runs. '
                'Whiskers show the worst and best run.', 'What the ranking says',
                'Four cloud settings clearly beat the best local setups; two sit within noise. Muse-Glimmer and\n'
                'Flash-Next on Strata tie at 92.5; Strata is steadier (±2.9 vs ±5.1). Flash-Next scores 88.1 on stock\n'
                f'llama.cpp with a different 3-bit file.\n{STAR}', note_h=0.13)
    ax = fig.add_axes([0.35, 0.31, 0.48, 0.58])
    style(ax, '', 'Weighted score (0-100)', 'x')
    ms = RANKED[::-1]
    for y, m in enumerate(ms):
        r = REP[m]
        c = CLOUD if m in CLOUD_IDS else PURPLE if m == STRATA else LOCAL
        lo, hi, mu = fnum(r['weighted_min']), fnum(r['weighted_max']), fnum(r['weighted_mean'])
        ax.plot([lo, hi], [y, y], color=c, lw=2, alpha=0.55, zorder=2)
        ax.scatter([mu], [y], s=100, color=c, ec=SURF, lw=2, zorder=3)
        ax.text(101, y, f'{mu:.1f} ± {fnum(r["weighted_std"]):.1f}', va='center', fontsize=11, color=INK)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ylabels(ms), fontsize=11)
    bold_ticks(ax, ms)
    ax.set_xlim(40, 100)
    ax.set_ylim(-0.7, len(ms) - 0.3)
    legend_dots(ax, [('cloud (subscription CLI)', CLOUD), ('local, llama.cpp', LOCAL), ('local, Strata engine', PURPLE)],
                loc='upper left')
    save(fig, '03_ranking')


# ---------------------------------------------------------------- 04 heatmap
def c04_heatmap():
    fig = frame('Where Each Model Loses Points', 'Mean score per task out of 70, 3 runs each. Darker = better.',
                'Reading the map',
                'A-D (clear specs) are nearly solved by the top local setups. E (reading a whole repo) is the gap\n'
                'that stays: no local setup reaches 62. Flash-Next on Strata reaches cloud level on G (agent) and C-D;\n'
                'Gemma-4-31B is strong except E; Qwen3.8 collapses on D (fake tool calls instead of files).')
    ax = fig.add_axes([0.35, 0.262, 0.6, 0.56])
    ms = RANKED
    data = [[fnum(REP[m][f'{t}_mean']) for t in TASKS] for m in ms]
    ax.imshow(data, cmap=GOOD, vmin=20, vmax=70, aspect='auto')
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=10.5,
                    color='white' if v > 55 else INK, fontweight='bold' if v >= 69.5 else 'normal')
    ax.set_xticks(range(8))
    ax.set_xticklabels([f'{t}\n{TASK_NAMES[t]}\n{WEIGHTS[t]}%' for t in TASKS], fontsize=11)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ylabels(ms), fontsize=10.5)
    bold_ticks(ax, ms)
    ax.axhline(len(CLOUD_IDS) - 0.5, color=INK, lw=2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, '04_task_heatmap')


# ---------------------------------------------------------------- 05 gap per task
def c05_gap():
    fig = frame('The Gap Left Is Reading a Repo', 'Per task, out of 70: best cloud, the two best local setups, and '
                'the local median.', 'Observed lesson',
                'The best local score matches or beats Sonnet 5.5 on B, C, D, G and H. Flash-Next on Strata even\n'
                'edges it on the agent task (69 vs 68). E, reading and critiquing a 42-file repo, is the one gap that\n'
                'stays: 7 points behind the cloud even for the best local setup. The median local model is far behind.')
    ax = fig.add_axes([0.22, 0.31, 0.56, 0.52])
    style(ax, '', 'Task score (0-70)', 'x')
    ys = list(range(8))[::-1]
    for y, t in zip(ys, TASKS):
        s = fnum(REP['sonnet55-medium'][f'{t}_mean'])
        mu, st = fnum(REP[MUSE][f'{t}_mean']), fnum(REP[STRATA][f'{t}_mean'])
        med = statistics.median(fnum(REP[m][f'{t}_mean']) for m in LOCAL_IDS)
        lo = min(med, mu, st, s)
        ax.plot([lo, max(s, mu, st)], [y, y], color=GRID, lw=6, zorder=1, solid_capstyle='round')
        ax.scatter([med], [y], s=110, color=MUTED, zorder=3, ec=SURF, lw=2)
        ax.scatter([mu], [y + 0.12], s=140, color=LOCAL, zorder=4, ec=SURF, lw=2)
        ax.scatter([st], [y - 0.12], s=140, color=PURPLE, zorder=4, ec=SURF, lw=2)
        ax.scatter([s], [y], s=150, color=CLOUD, zorder=5, ec=SURF, lw=2)
        best = max(mu, st)
        ax.text(71.5, y, f'{s - best:+.0f} / {s - med:+.0f}', va='center', fontsize=11.5,
                color=RED if s - best >= 6 else INK2, fontweight='bold' if s - best >= 6 else 'normal')
    ax.set_yticks(ys)
    ax.set_yticklabels([f'{t}  {TASK_NAMES[t]} ({WEIGHTS[t]}%)' for t in TASKS], fontsize=12.5)
    ax.set_xlim(40, 71)
    legend_dots(ax, [('Sonnet 5.5', CLOUD), ('Muse-Glimmer', LOCAL), ('Flash-Next · Strata', PURPLE),
                     ('local median', MUTED)], loc='lower left', bbox_to_anchor=(-0.36, 1.03), ncol=4,
                columnspacing=0.9, handletextpad=0.2)
    ax.text(71.5, 7.45, 'cloud lead vs\nbest local / median', fontsize=10, color=MUTED)
    save(fig, '05_cloud_local_gap')


# ---------------------------------------------------------------- 06 quality vs speed
def c06_quality_speed():
    fig = frame('Fast and Best Locally, at a RAM Cost', 'Local: score vs generation speed at 32K. Arrows = same '
                'weights with a speculative drafter.', 'Observed lesson',
                'Flash-Next on the Strata engine scores 92.5 at 170 tok/s, but parks 43-47 GB of experts in system\n'
                'RAM. Muse-Glimmer reaches 92.5 with everything on the GPU, and 127 tok/s only with DFlash. MoE\n'
                'A3B models are fast but top out near 84; the fastest, Nemotron, is the weakest coder.')
    ax = fig.add_axes([0.12, 0.31, 0.82, 0.55])
    style(ax, 'Weighted score (0-100)', 'Generation speed at 32K (tokens/s)')
    labels = {MUSE: (8, 6), 'qwen36-27b-q4km': (8, -14), 'ornith15-q4km': (8, 6),
              'qwen36-35b-iq4xs': (10, -3), 'gemma4-31b-qat': (8, -12), 'ornith10-q4km': (8, -16),
              'qwen38-27b-iq4xs': (-8, 8), 'qwen3codernext-iq4xs-offload': (8, -14), 'qwen3coder-30b-iq4xs': (-8, 8),
              'gptoss-20b-mxfp4': (8, 6), 'gemma4-26b-a4b-qat': (8, 4), 'devstral2-24b-q4km': (8, 4),
              'granite41-30b-q4km-kvq4': (8, 4), 'glm47flash-iq4xs': (8, 4), 'nemotron3nano-30b-iq4xs': (-8, 8),
              FN_LLAMA: (4, -15), STRATA: (-10, 8)}
    for m in LOCAL_IDS:
        x, y = fnum(MASTER[m]['gen_32k']), fnum(REP[m]['weighted_mean'])
        c = PURPLE if m == STRATA else LOCAL
        ax.scatter([x], [y], s=150 if m == STRATA else 120, color=c, ec=SURF, lw=2, zorder=3)
        dx, dy = labels.get(m, (8, 4))
        lab = 'Flash-Next · llama.cpp*' if m == FN_LLAMA else NAMES[m]
        ax.annotate(lab, (x, y), (dx, dy), textcoords='offset points', fontsize=11.2,
                    ha='left' if dx > 0 else 'right', color=PURPLE if m == STRATA else INK,
                    fontweight='bold' if m == STRATA else 'normal')
    for base, fast, tag in [(MUSE, MUSE_DF, 'DFlash'), ('qwen36-27b-q4km', 'qwen36-27b-q4km-mtp6', 'MTP'),
                            ('qwen38-27b-iq4xs', 'qwen38-27b-iq4xs-mtp6', 'MTP'), (FN_LLAMA, FN_MTP, 'MTP, b11609')]:
        x0, x1, y = fnum(MASTER[base]['gen_32k']), fnum(MASTER[fast]['gen_32k']), fnum(REP[base]['weighted_mean'])
        ax.annotate('', (x1, y), (x0 + 3, y), arrowprops=dict(arrowstyle='->', color=GREEN, lw=2.2))
        ax.scatter([x1], [y], s=110, color=GREEN, ec=SURF, lw=2, zorder=3)
        ax.text(x1 + 4, y + (0.9 if base == FN_LLAMA else -2.0), f'+{tag} {x1:.0f}', fontsize=10.5, color=GREEN,
                fontweight='bold')
    ax.set_xlim(15, 225)
    ax.set_ylim(42, 97)
    legend_dots(ax, [('llama.cpp', LOCAL), ('with speculative drafter', GREEN), ('Strata engine', PURPLE)],
                loc='lower center')
    save(fig, '06_quality_vs_speed')


# ---------------------------------------------------------------- 07 generation speed
def c07_generation():
    rows = []
    for m in [k for k in MASTER if MASTER[k]['type'] == 'local' and k in NAMES and k in KIND]:
        g32 = fnum(MASTER[m]['gen_32k'])
        p5 = [r for r in P5 if r['model'] == m and r['ctx'] == '128K']
        rows.append((m, g32, fnum(p5[0]['gen_tps']) if p5 else None))
    rows.sort(key=lambda r: r[1])
    fig = frame('Output Speed: Engines and Drafters Win', 'Generation tokens/s at 32K context (bar) and with the '
                'context filled to 128K (dot).', 'Observed lesson',
                'MoE models with ~3B active parameters generate 115-200 tok/s; dense 24-31B models 34-51, or 119-153\n'
                'with a drafter. Flash-Next, too big for 24 GB, runs at 27 tok/s on llama.cpp (41 with MTP) and at\n'
                '170 on Strata, which keeps hot experts in VRAM. At 128K most models keep 60-85% of their speed.')
    ax = fig.add_axes([0.35, 0.31, 0.58, 0.56])
    style(ax, '', 'Generation speed (tokens/s)', 'x')
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[KIND_COL[KIND[r[0]]] for r in rows], height=0.62, zorder=2)
    for y, r in zip(ys, rows):
        ax.text(r[1] + 3, y, f'{r[1]:.0f}', va='center', fontsize=10.5, fontweight='bold')
        if r[2]:
            ax.scatter([r[2]], [y], s=60, color=INK, marker='o', zorder=4, ec='white', lw=1.5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES[r[0]] for r in rows], fontsize=10.8)
    ax.set_xlim(0, 235)
    h = [Patch(color=c, label=l) for l, c in KIND_LAB]
    h.append(Line2D([], [], marker='o', ls='', color=INK, ms=8, label='at 128K context'))
    ax.legend(handles=h, loc='lower right', frameon=False, fontsize=11.5)
    save(fig, '07_generation_speed')


# ---------------------------------------------------------------- 08 reading speed
def c08_reading():
    rows = []
    for m in LOCAL_IDS:
        r = MASTER[m]
        p5 = [x for x in P5 if x['model'] == m and x['ctx'] == '128K']
        t128 = fnum(p5[0]['ttft_s']) if p5 else (STRATA_LC['128K'][1] if m == STRATA else None)
        rows.append((m, fnum(r['prompt_32k']), fnum(r['ttft_32k_s']), t128))
    rows.sort(key=lambda r: r[1])
    fig = frame('Reading Speed Sets the Wait', 'Prompt processing tokens/s at 32K, and time to first token at 32K and '
                'at ~110K tokens.', 'Observed lesson',
                'MoE models read 6,000-10,500 tok/s: 110K tokens in 16-23 s. Dense models read 2,400-3,300 tok/s and\n'
                'wait 40-60 s. Weights in system RAM are the slowest: Flash-Next on llama.cpp reads 216 tok/s, two\n'
                'minutes before the first token at 32K. Strata reads the same model at 3,200 tok/s (10 s).')
    ax = fig.add_axes([0.35, 0.31, 0.34, 0.56])
    style(ax, '', 'Prompt processing at 32K (tok/s)', 'x')
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[KIND_COL[KIND[r[0]]] for r in rows], height=0.62, zorder=2)
    for y, r in zip(ys, rows):
        ax.text(r[1] + 150, y, f'{r[1]:,.0f}', va='center', fontsize=10)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES[r[0]] for r in rows], fontsize=10.8)
    ax.set_xlim(0, 14000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
    ax2 = fig.add_axes([0.72, 0.31, 0.23, 0.56], sharey=ax)
    style(ax2, '', 'First token (s)', 'x')
    cap = 125
    for y, r in zip(ys, rows):
        if r[3]:
            ax2.plot([r[2], r[3]], [y, y], color=GRID, lw=4, zorder=1)
            ax2.scatter([r[3]], [y], s=70, color=ORANGE, zorder=3, ec=SURF, lw=1.5)
            ax2.text(r[3] + 3, y, f'{r[3]:.0f}', va='center', fontsize=9.5)
        ax2.scatter([min(r[2], cap)], [y], s=70, color=BLUE, zorder=3, ec=SURF, lw=1.5,
                    marker='>' if r[2] > cap else 'o')
        if r[2] > 30:
            ax2.text(min(r[2], cap) - 4, y + 0.42, f'{r[2]:.0f}', va='center', ha='right', fontsize=9.5,
                     color=RED if r[2] > 60 else INK)
    ax2.tick_params(labelleft=False)
    ax2.set_xlim(0, 135)
    ax2.legend(handles=[Line2D([], [], marker='o', ls='', ms=9, color=BLUE, label='32K'),
                        Line2D([], [], marker='o', ls='', ms=9, color=ORANGE, label='~110K')],
               loc='upper right', frameon=False, fontsize=11)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in KIND_LAB], loc='lower right', frameon=False, fontsize=10.5)
    save(fig, '08_reading_speed')


# ---------------------------------------------------------------- 09 time & tokens per round
def minutes_and_tokens():
    by = defaultdict(lambda: defaultdict(list))
    for r in FLAT:
        by[r['model']][r['task']].append((fnum(r['duration_s']) or 0, fnum(r['completion_tokens']) or 0))
    out = {}
    for m, ts in by.items():
        out[m] = (sum(statistics.mean(x[0] for x in v) for v in ts.values()) / 60,
                  sum(statistics.mean(x[1] for x in v) for v in ts.values()))
    out.update({'sonnet55-medium': (3.7, 28000), 'gpt61sol-medium': (11.2, 17000), 'haiku55-medium': (6.8, 88000),
                'gpt6luna-xhigh': (9.6, 40000), 'haiku55-xhigh': (20.2, 282000), 'gpt6luna-medium': (4.6, 13000)})
    return out


def c09_efficiency():
    mt = minutes_and_tokens()
    fig = frame('Score per Minute, Score per Token', 'Wall-clock minutes and output tokens for one full round of '
                '8 tasks, against the weighted score.', 'Observed lesson',
                'Flash-Next on Strata runs a round in 3.2 min, as fast as Sonnet 5.5 (3.7 min); the same model on\n'
                'llama.cpp takes 28 min. More thinking is not more score: Haiku xhigh spends 282K tokens and 20 min\n'
                'to score below Haiku medium. DFlash halves Muse-Glimmer\'s wall time.')
    ax = fig.add_axes([0.11, 0.625, 0.84, 0.245])
    ax2 = fig.add_axes([0.11, 0.315, 0.84, 0.235])
    style(ax, 'Score', 'Minutes per round (8 tasks)')
    style(ax2, 'Score', 'Output tokens per round (log scale)')
    show = set(CLOUD_IDS) | {MUSE, MUSE_DF, STRATA, FN_LLAMA, 'qwen36-27b-q4km', 'ornith15-q4km', 'qwen36-35b-iq4xs',
                             'gemma4-31b-qat', 'qwen3codernext-iq4xs-offload', 'gptoss-20b-mxfp4',
                             'nemotron3nano-30b-iq4xs', 'devstral2-24b-q4km'}
    short = {'sonnet55-medium': 'Sonnet 5.5', 'gpt61sol-medium': 'GPT-6.1 Sol', 'haiku55-medium': 'Haiku med',
             'haiku55-xhigh': 'Haiku xhigh', 'gpt6luna-medium': 'Luna med', 'gpt6luna-xhigh': 'Luna xhigh',
             MUSE: 'Muse-Glimmer', MUSE_DF: 'Muse + DFlash', STRATA: 'Flash-Next · Strata',
             FN_LLAMA: 'Flash-Next · llama.cpp', 'qwen3codernext-iq4xs-offload': 'Coder-Next 80B',
             'nemotron3nano-30b-iq4xs': 'Nemotron', 'qwen36-35b-iq4xs': 'Qwen3.6-35B', 'gemma4-31b-qat': 'Gemma-4-31B',
             'gptoss-20b-mxfp4': 'gpt-oss-20b', 'qwen36-27b-q4km': 'Qwen3.6-27B', 'ornith15-q4km': 'Ornith-1.5',
             'devstral2-24b-q4km': 'Devstral'}
    off1 = {'sonnet55-medium': (-6, 8), 'gpt61sol-medium': (9, -4), 'haiku55-medium': (0, 9), 'gpt6luna-xhigh': (0, -17),
            'gpt6luna-medium': (0, 9), 'haiku55-xhigh': (-10, 8), MUSE: (-10, -16), MUSE_DF: (-8, 5),
            'qwen3codernext-iq4xs-offload': (-10, -16), 'nemotron3nano-30b-iq4xs': (8, -4), STRATA: (-8, 7),
            FN_LLAMA: (-10, 7), 'gptoss-20b-mxfp4': (8, -4), 'devstral2-24b-q4km': (8, -4)}
    off2 = {'sonnet55-medium': (-4, 8), 'gpt61sol-medium': (-6, 8), 'haiku55-medium': (-6, 8), 'gpt6luna-xhigh': (6, 6),
            'gpt6luna-medium': (-8, -4), 'haiku55-xhigh': (-12, -16), MUSE: (8, -6), STRATA: (8, 6),
            'gptoss-20b-mxfp4': (8, -4), 'gemma4-31b-qat': (8, -4), 'nemotron3nano-30b-iq4xs': (8, -4)}
    for m in show:
        if m not in mt:
            continue
        mins, toks = mt[m]
        sc = fnum(REP[m]['weighted_mean'])
        c = CLOUD if m in CLOUD_IDS else GREEN if m == MUSE_DF else PURPLE if m == STRATA else LOCAL
        for a, x in ((ax, mins), (ax2, toks)):
            a.scatter([x], [sc], s=95, color=c, ec=SURF, lw=1.5, zorder=3)
        for a, x, off in ((ax, mins, off1), (ax2, toks, off2)):
            if m in off:
                dx, dy = off[m]
                a.annotate(short[m], (x, sc), (dx, dy), textcoords='offset points', fontsize=10.3, color=INK,
                           ha='left' if dx > 0 else 'right' if dx < 0 else 'center')
    ax.set_xlim(0, 30)
    ax2.set_xscale('log')
    ax2.set_xlim(8000, 500000)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
    for a in (ax, ax2):
        a.set_ylim(42, 103)
    legend_dots(ax, [('cloud', CLOUD), ('local', LOCAL), ('local + DFlash', GREEN), ('Strata', PURPLE)],
                loc='lower right')
    save(fig, '09_time_and_tokens')


# ---------------------------------------------------------------- 10 error analysis: sub-checks
def c10_checks():
    cols = [('A program runs', 'A program runs'), ('A hidden data exact', 'A hidden data exact'),
            ('B all tests pass', 'B all tests pass'), ('C 5/5 SQL correct', 'C 5/5 SQL correct'),
            ('C Q5 2x faster', 'C Q5 2x faster'), ('D vue-tsc passes', 'D vue-tsc passes'),
            ('D hidden backend tests', 'D hidden backend tests'), ('E coupling recall', 'E coupling recall'),
            ('E race-condition recall', 'E race-condition recall'), ('E memory-leak recall', 'E memory-leak recall'),
            ('F right root cause', 'F right root cause'), ('G hidden tests pass', 'G hidden tests pass'),
            ('H 9/9 fields', 'H 9/9 fields')]
    ms = RANKED
    data = [[fnum(CHECKS[m].get(c[0])) or 0 for c in cols] for m in ms]
    fig = frame('Error Analysis: Which Check Fails', 'Pass rate of grader checks over 3 runs. E columns: share of '
                'planted issues found.', 'Most common misses',
                'Spotting tight coupling (E) is the hardest check: only GPT-6.1 Sol and Luna xhigh are perfect, and no\n'
                'local setup averages above 0.81. Locally, the usual misses are code that crashes on malformed rows\n'
                '(A), vue-tsc errors (D) and a long-document rule (H). Only Muse-Glimmer and Strata get H right.')
    ax = fig.add_axes([0.32, 0.255, 0.64, 0.5])
    ax.imshow(data, cmap=GOOD, vmin=0, vmax=1, aspect='auto')
    for i, row in enumerate(data):
        for j, v in enumerate(row):
            ax.text(j, i, '✓' if v >= 0.999 else f'{v:.1f}'.lstrip('0') if v > 0 else '✗', ha='center', va='center',
                    fontsize=10 if 0 < v < 0.999 else 11.5, color='white' if v > 0.75 else (RED if v == 0 else INK),
                    fontweight='bold')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[1] for c in cols], rotation=50, ha='left', fontsize=10.5, rotation_mode='anchor')
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ylabels(ms), fontsize=10.3)
    bold_ticks(ax, ms)
    ax.axhline(len(CLOUD_IDS) - 0.5, color=INK, lw=2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, '10_error_checks')


# ---------------------------------------------------------------- 11 failure modes
def c11_failure_modes():
    modes = defaultdict(list)
    for r in FLAT:
        t70 = fnum(r['total70'])
        if t70 is None or t70 >= 35:
            continue
        m, t = r['model'], r['task']
        if r['fake_toolcall'] == 'True':
            k = 'Emits fake <tool_call> text, no files'
        elif r['finish_reason'] == 'length' or (m == 'haiku55-xhigh' and t == 'D'):
            k = 'Thinks until the token cap / stops half-way'
        elif t == 'A':
            k = 'Program crashes on malformed rows'
        elif t == 'G':
            k = 'Agent loops until the 40-step limit'
        elif t == 'F':
            k = 'Takes a symptom for the root cause'
        elif t == 'E':
            k = 'Misses coupling / races / leaks in repo'
        elif t == 'D':
            k = 'Cross-file feature breaks tests or tsc'
        elif t == 'B':
            k = 'Bug fix incomplete, tests still fail'
        elif t == 'C':
            k = 'SQL returns wrong rows'
        else:
            k = 'Misses planted long-document rules'
        modes[k].append(m)
    items = sorted(modes.items(), key=lambda kv: len(kv[1]))
    fig = frame('How Runs Fail', 'Every task run scoring below 35/70 (half marks), by failure mode. '
                f'{sum(len(v) for v in modes.values())} of {len(FLAT)} runs.', 'Observed lesson',
                'Cloud models produced one such run (Haiku xhigh stopped after the backend on D); neither Flash-Next\n'
                'setup produced any. Locally the biggest buckets are fragile code and agents that wander until the\n'
                'step limit. "Thinking to the cap" is rare but random, and it hits even Muse-Glimmer with DFlash.')
    ax = fig.add_axes([0.42, 0.31, 0.53, 0.55])
    style(ax, '', 'Runs below half marks', 'x')
    for y, (k, v) in enumerate(items):
        cl = sum(1 for m in v if m in CLOUD_IDS)
        ax.barh(y, len(v) - cl, color=LOCAL, height=0.6, zorder=2)
        if cl:
            ax.barh(y, cl, left=len(v) - cl, color=CLOUD, height=0.6, zorder=2)
        ex = sorted({NAMES[m] for m in v}, key=lambda n: -sum(1 for m in v if NAMES[m] == n))[0]
        ax.text(len(v) + 0.3, y, f'{len(v)}  e.g. {ex}', va='center', fontsize=10.2, color=INK2)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([k for k, _ in items], fontsize=11.5)
    ax.set_xlim(0, 21)
    ax.set_xticks(range(0, 13, 2))
    legend_bars(ax, [('local', LOCAL), ('cloud', CLOUD)])
    save(fig, '11_failure_modes')


# ---------------------------------------------------------------- 12 long context needles
def c12_needles():
    grid = defaultdict(dict)
    for r in LONG:
        k = {'32768': '32K', '65536': '64K'}[r['nominal_ctx']]
        grid[r['model']][k] = (r['start_needle'] == 'True', r['middle_needle'] == 'True', r['end_needle'] == 'True')
    for r in P5:
        if r['h_reused_from']:
            continue
        grid[r['model']][r['ctx']] = (r['start_rule'] == '✓', r['middle_rule'] == '✓', r['end_rule'] == '✓')
    kvq4 = set()
    for r in P5B:
        if r['h_fields_9'] and r['kv'].startswith('q4') and '256K' not in grid[r['model']]:
            grid[r['model']]['256K'] = (r['start'] == '1', r['middle'] == '1', r['end'] == '1')
            kvq4.add(r['model'])
    for c, (_, _, v) in STRATA_LC.items():
        grid[STRATA][c] = v
    ms = [m for m in RANKED if m in grid]
    ctxs = ['32K', '64K', '128K', '256K']
    fig = frame('Lost in the Middle', 'Task H: three binding rules at the start, middle and end. Was each one used?',
                'Observed lesson',
                '"Fits in context" is not "was read". Only Muse-Glimmer (to its 128K limit) and Flash-Next on Strata\n'
                'used all three rules at 128K. At 256K the best local result is 8/9 (Strata, Qwen3.8-27B): nobody\n'
                'found the middle rule. Cloud: all settings except GPT-6 Luna medium got 9/9.')
    ax = fig.add_axes([0.35, 0.3, 0.6, 0.55])
    ax.set_xlim(-0.5, len(ctxs) * 3.6 + 0.4)
    ax.set_ylim(len(ms) - 0.5, -1.3)
    ax.axis('off')
    for j, c in enumerate(ctxs):
        x0 = j * 3.6
        ax.text(x0 + 1, -1.15, c, ha='center', fontsize=13, fontweight='bold')
        for k, lab in enumerate(('S', 'M', 'E')):
            ax.text(x0 + k, -0.65, lab, ha='center', fontsize=10.5, color=MUTED)
    for i, m in enumerate(ms):
        ax.text(-0.9, i, NAMES[m], ha='right', va='center', fontsize=11,
                fontweight='bold' if m in TOP_LOCAL else 'normal')
        for j, c in enumerate(ctxs):
            v = grid[m].get(c)
            for k in range(3):
                x = j * 3.6 + k
                if v is None:
                    ax.add_patch(Rectangle((x - 0.42, i - 0.38), 0.84, 0.76, fc='none', ec=GRID, lw=1))
                    continue
                ok = v[k]
                ax.add_patch(Rectangle((x - 0.42, i - 0.38), 0.84, 0.76, fc=GREEN if ok else '#F4D6D6', ec='none'))
                ax.text(x, i, '✓' if ok else '✗', ha='center', va='center', fontsize=10.5,
                        color='white' if ok else RED, fontweight='bold')
        if m in kvq4:
            ax.text(3 * 3.6 + 2.55, i, 'q4', va='center', fontsize=9, color=MUTED)
    fig.text(0.065, 0.274, 'q4 = 256K needed KV cache q4_0 (Phase 5b). 32K/64K = temperature-0 run. '
             'empty = does not fit in 24 GB, or not run.', fontsize=10, color=INK2)
    fig.text(0.065, 0.252, 'S = start rule (UTC to Asia/Taipei)  ·  M = middle rule (meter M-17 replaced)  ·  '
             'E = end rule (maintenance excluded from peak)', fontsize=10, color=INK2)
    save(fig, '12_long_context_needles')


# ---------------------------------------------------------------- 13 long context cost
def c13_longctx_cost():
    series = {}
    for r in LONG:
        series.setdefault(r['model'], {})[int(r['doc_tokens'])] = fnum(r['ttft_s'])
    for r in P5:
        if r['h_reused_from']:
            continue
        series.setdefault(r['model'], {})[int(r['filled_tokens'])] = fnum(r['ttft_s'])
    series.setdefault(STRATA, {})[STRATA_LC['128K'][0]] = STRATA_LC['128K'][1]
    series[STRATA][225575] = 64.5  # strata 256K perf run (report 11.3)
    pick = [(MUSE, GREEN), ('qwen38-27b-iq4xs', BLUE), ('qwen36-35b-iq4xs', ORANGE),
            ('nemotron3nano-30b-iq4xs', TEAL), (STRATA, PURPLE)]
    fig = frame('What 256K Costs in Waiting', 'Time to first token as the prompt grows (task H corpus, no prefix cache).',
                'Observed lesson',
                'Dense Qwen3.8-27B waits 2.5 minutes at 222K tokens. Flash-Next on Strata waits ~65 s at 226K and still\n'
                'gets 8/9, the best local long-context result, using 46.6 GB of system RAM. MoE Qwen3.6-35B needs\n'
                '55 s (7/9). Muse-Glimmer stops at its 128K limit but reads every rule there.')
    ax = fig.add_axes([0.12, 0.31, 0.82, 0.54])
    style(ax, 'Time to first token (s)', 'Prompt tokens actually filled')
    for m, c in pick:
        pts = sorted(series[m].items())
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, lw=2.2, marker='o', ms=8, mec=SURF, mew=1.5)
        x, y = pts[-1]
        short = NAMES[m].replace(' · Strata*', ' · Strata').replace('Qwen3.8-Flash-Next', 'Flash-Next')
        ax.text(x + 4000, y + (2.5 if m == STRATA else -2.5 if m == 'qwen36-35b-iq4xs' else 0),
                f'{short}  {int(y + 0.5)}s', va='center', fontsize=11.2, color=INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v/1000:.0f}K'))
    ax.set_xlim(0, 340000)
    ax.set_ylim(0, 160)
    ax.annotate('Qwen3-Coder-Next 80B (part on CPU): 525 s at 222K, off the chart', (222329, 159), (8000, 152),
                fontsize=11, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=1.5), va='center')
    save(fig, '13_long_context_wait')


# ---------------------------------------------------------------- 14 reasoning level + stability
def c14_reasoning_stability():
    fig = frame('More Thinking, More Runs', 'Left: raising the reasoning level. Right: one temperature-0 run vs the '
                'mean of three seeded runs (local).', 'Observed lesson',
                'Raising effort helped GPT-6 Luna (+3.5, it stopped missing the mid-document rule) and hurt Claude\n'
                'Haiku (-2.3, one run over-thought and stopped half-way). Single runs mislead: Flash-Next on llama.cpp\n'
                'scored 79.3 once and 88.1 on average. Three runs is the minimum.')
    ax = fig.add_axes([0.1, 0.31, 0.25, 0.53])
    style(ax, 'Weighted score', '', 'y')
    for a, b, c, lab in [('gpt6luna-medium', 'gpt6luna-xhigh', ORANGE, 'GPT-6 Luna'),
                         ('haiku55-medium', 'haiku55-xhigh', PURPLE, 'Claude Haiku 5.5')]:
        ya, yb = fnum(REP[a]['weighted_mean']), fnum(REP[b]['weighted_mean'])
        ax.plot([0, 1], [ya, yb], color=c, lw=2.5, marker='o', ms=11, mec=SURF, mew=2)
        hi_x, hi_y = (0, ya) if ya > yb else (1, yb)
        ax.text(hi_x + (0.04 if hi_x == 0 else -0.04), hi_y + 0.25, f'{lab}\n{ya:.1f} to {yb:.1f}', va='bottom',
                ha='left' if hi_x == 0 else 'right', fontsize=11.5, color=c, fontweight='bold')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['medium', 'xhigh'], fontsize=12.5)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(91, 98)
    ax2 = fig.add_axes([0.65, 0.31, 0.3, 0.53])
    style(ax2, '', 'Weighted score', 'x')
    ms = [m for m in RANKED if m in LOCAL_IDS][::-1]
    for y, m in enumerate(ms):
        s, r = fnum(MASTER[m]['single_run_score']), fnum(REP[m]['weighted_mean'])
        ax2.plot([s, r], [y, y], color=GRID, lw=4, zorder=1)
        ax2.scatter([s], [y], s=55, color=MUTED, zorder=2)
        ax2.scatter([r], [y], s=85, color=LOCAL, zorder=3, ec=SURF, lw=1.5)
    ax2.set_yticks(range(len(ms)))
    ax2.set_yticklabels([NAMES[m] for m in ms], fontsize=10)
    ax2.set_xlim(40, 100)
    legend_dots(ax2, [('single run, temp 0', MUTED), ('3-run mean', LOCAL)], loc='lower right')
    save(fig, '14_reasoning_and_stability')


# ---------------------------------------------------------------- 15 routing
def c15_routing():
    fig = frame('What I Would Run Today', 'Routing decisions from this benchmark, for one RTX 4090 (64 GB RAM) and '
                'two subscriptions.', None, None)
    rows = [
        ('Best result, cloud allowed', 'Claude Sonnet 5.5 · medium', '98.8 · 3.7 min/round · 0 failed tasks', CLOUD),
        ('Most repeatable', 'GPT-6.1 Sol · medium', '97.7 ± 0.2 · but 11 min/round', CLOUD),
        ('Save subscription quota', 'Haiku 5.5 medium / GPT-6 Luna xhigh', 'both above 96, no failed tasks', CLOUD),
        ('Local, one user, fastest', 'Qwen3.8-Flash-Next on Strata', '92.5 ± 2.9 · ~170 tok/s · 43-47 GB RAM', PURPLE),
        ('Local, 2-4 users or easy to manage', 'Muse-Glimmer-30B + DFlash, -np 4', '92.5 · 237 tok/s total at 4 users · 19 GB',
         GREEN),
        ('Local, light and fast', 'Ornith-1.5-35B-A3B', '84.1 · 161 tok/s · VRAM only', LOCAL),
        ('Local, 256K context', 'Flash-Next on Strata / Qwen3.6-35B', '8/9 at 65 s, uses RAM / 7/9 at 55 s, VRAM only', LOCAL),
        ('Avoid for coding', 'Nemotron-3-Nano, GLM-4.7-Flash, Granite-4.1', 'fast or large context, below 57', RED),
    ]
    h, g = 0.083, 0.012
    for i, (need, pick, why, c) in enumerate(rows):
        y = 0.86 - (i + 1) * h - i * g
        card(fig, 0.065, y, 0.87, h)
        fig.patches.append(Rectangle((0.065, y), 0.012, h, transform=fig.transFigure, fc=c, ec='none'))
        fig.text(0.1, y + h - 0.022, need, fontsize=12.5, color=INK2, va='top')
        fig.text(0.1, y + 0.022, pick, fontsize=16, fontweight='bold', va='bottom')
        fig.text(0.92, y + h / 2, why, fontsize=12, color=INK2, va='center', ha='right')
    fig.text(0.065, 0.07, 'Cloud = subscription CLIs (Claude Code, Codex), no API fees. Local = llama.cpp or Strata, '
             '24 GB VRAM.', fontsize=12, color=INK2)
    save(fig, '15_routing')


# ---------------------------------------------------------------- 16 256K feasibility
def c16_fit_256k():
    rows = sorted(P5B, key=lambda r: (int(r['max_ctx']), -(fnum(r['ttft_256k_s']) or 0)))
    fig = frame('Who Can Hold 256K on 24 GB', 'Largest context verified without spilling into system RAM. '
                'Labels: first-token time at ~225K.', 'Observed lesson',
                'Halving the KV cache to q4_0 let six more llama.cpp models reach 256K. Strata holds it by putting the\n'
                'KV cache in system RAM: 64-65 s to first token, 130-143 tok/s, 8/9. Five MoE models on llama.cpp stay\n'
                'usable (39-58 s); dense 27B models and a VRAM-full Qwen3-Coder-30B wait 2.3-2.6 min.')
    ax = fig.add_axes([0.35, 0.31, 0.58, 0.56])
    style(ax, '', 'Largest verified context (tokens)', 'x')
    for y, r in enumerate(rows):
        kv = r['kv'].split(' ')[0]
        fits = r['fits_256k'] in ('yes', 'tight', 'slow')
        c = {'q8_0': GREEN, 'q4_0': BLUE, 'int8': PURPLE}.get(kv, MUTED) if fits else MUTED
        ax.barh(y, int(r['max_ctx']), color=c, height=0.62, zorder=2)
        t = fnum(r['ttft_256k_s'])
        if t:
            lab = f'{t:.0f} s' + ('  · VRAM nearly full' if r['fits_256k'] == 'tight' else '') + \
                  ('  · part on CPU' if r['fits_256k'] == 'slow' else '') + ('  · KV in RAM' if kv == 'int8' else '')
        else:
            lab = {'no': f'max {int(r["max_ctx"]) // 1024}K', 'native 128K': 'model limit 128K',
                   'native 198K': 'model limit 198K'}.get(r['fits_256k'], '')
        ax.text(int(r['max_ctx']) + 4000, y, lab, va='center', fontsize=10.3, color=RED if t and t > 300 else INK2)
    ax.axvline(262144, color=INK, lw=1.2, ls=':')
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAMES[r['model']] for r in rows], fontsize=10.8)
    ax.set_xlim(0, 390000)
    ax.set_xticks([0, 65536, 131072, 196608, 262144])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v / 1024:.0f}K'))
    legend_bars(ax, [('fits with KV q8_0', GREEN), ('fits only with KV q4_0', BLUE), ('Strata, KV in RAM', PURPLE),
                     ('does not reach 256K', MUTED)])
    save(fig, '16_fit_256k')


# ---------------------------------------------------------------- 17 multi-user
def c17_multiuser():
    names = list(MULTI)
    lab = {names[0]: 'Strata, default (queue)', names[1]: 'Strata, parallel 4',
           names[2]: 'Muse + DFlash, llama.cpp -np 4', names[3]: 'Muse, llama.cpp -np 4'}
    col = {names[0]: PURPLE, names[1]: TEAL, names[2]: GREEN, names[3]: BLUE}
    fig = frame('Two to Four Users: Muse Scales', 'Users send different coding requests at the same time '
                '(800 tokens each, temperature 0).', 'Observed lesson',
                'Strata is fastest for one user but processes one request at a time: total stays near 180 tok/s and\n'
                'the 4th user waits 14 s for a first token. Its parallel mode is worse on 24 GB. Muse-Glimmer on\n'
                'llama.cpp batches: 237 tok/s total at 4 users, every first token within 0.5 s.')
    ax = fig.add_axes([0.1, 0.56, 0.84, 0.29])
    style(ax, 'Total tokens/s (all users)', 'Concurrent users')
    for n in names:
        pts = [(x['clients'], x['aggregate_tok_s']) for x in MULTI[n]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col[n], lw=2.4, marker='o', ms=9, mec=SURF, mew=1.5)
        ax.text(4.08, pts[-1][1], f'{lab[n]}  {pts[-1][1]:.0f}', va='center', fontsize=11, color=INK)
    ax.set_xticks([1, 2, 4])
    ax.set_xlim(0.8, 5.9)
    ax.set_ylim(0, 270)
    ax2 = fig.add_axes([0.35, 0.3, 0.55, 0.19])
    style(ax2, '', 'At 4 users: longest wait for a first token (s)', 'x')
    order = sorted(names, key=lambda n: MULTI[n][-1]['max_wait_first_token_s'])
    for y, n in enumerate(order):
        x = MULTI[n][-1]
        ax2.barh(y, x['max_wait_first_token_s'], color=col[n], height=0.6, zorder=2)
        ax2.text(x['max_wait_first_token_s'] + 0.2, y, f'{x["max_wait_first_token_s"]:.1f} s  ·  all done in '
                 f'{x["wall_s"]:.1f} s', va='center', fontsize=10.5, color=INK2)
    ax2.set_yticks(range(len(order)))
    ax2.set_yticklabels([lab[n] for n in order], fontsize=10.8)
    ax2.set_xlim(0, 24)
    save(fig, '17_multi_user')


# ---------------------------------------------------------------- 18 same weights, different engine
def c18_engine():
    rows = [('llama.cpp b10929 · unsloth UD-IQ3_XXS', 'qwen38flashnext-iq3xxs', BLUE),
            ('llama.cpp b11609 · unsloth UD-IQ3_XXS', 'qwen38flashnext-iq3xxs-b11609', BLUE),
            ('llama.cpp b11609 + MTP · unsloth', FN_MTP, GREEN),
            ('llama.cpp b11609 · GSQ-RCO IQ3_XXS', 'qwen38flashnext-gsq-iq3xxs-b11609', BLUE),
            ('Strata · GSQ-RCO IQ3_XXS', STRATA, PURPLE)]
    fig = frame('Same Model, Five Times Faster', 'Qwen3.8-Flash-Next (76 GB at 3-bit) on one 24 GB GPU: engine and '
                'build at 32K context.', 'Observed lesson',
                'On the same GSQ-RCO file, Strata generates 5.2x and reads 10.9x faster than stock llama.cpp, because\n'
                'it keeps frequently used experts in VRAM and moves the KV cache to RAM. Upgrading llama.cpp alone\n'
                'gives +9%; MTP another 1.4x. Score: 92.5 on Strata vs 88.1 on llama.cpp with the unsloth file.')
    for i, (key, xl, fmt) in enumerate([('gen_32k', 'Generation at 32K (tok/s)', '{:.0f}'),
                                        ('prompt_32k', 'Prompt processing at 32K (tok/s)', '{:,.0f}')]):
        ax = fig.add_axes([0.42, 0.62 - i * 0.3, 0.5, 0.22])
        style(ax, '', xl, 'x')
        vals = [fnum(MASTER[k][key]) for _, k, _ in rows]
        ys = list(range(len(rows)))[::-1]
        ax.barh(ys, vals, color=[c for *_, c in rows], height=0.6, zorder=2)
        for y, v in zip(ys, vals):
            ax.text(v * 1.02 + 2, y, fmt.format(v), va='center', fontsize=11, fontweight='bold')
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=10.8)
        ax.set_xlim(0, max(vals) * 1.2)
    save(fig, '18_engine_matters')


# ---------------------------------------------------------------- 19 llama.cpp b10929 -> b11609
def c19_version():
    rows = []
    for k in MASTER:
        if not k.endswith('-b11609') or k.startswith('qwen38flashnext-gsq'):
            continue
        base = k[:-len('-b11609')]
        if base not in MASTER or base not in NAMES:
            continue
        g0, g1 = fnum(MASTER[base]['gen_32k']), fnum(MASTER[k]['gen_32k'])
        p0, p1 = fnum(MASTER[base]['prompt_32k']), fnum(MASTER[k]['prompt_32k'])
        rows.append((base, (g1 / g0 - 1) * 100, (p1 / p0 - 1) * 100))
    rows.sort(key=lambda r: r[1])
    fig = frame('New llama.cpp: Faster Reading', 'Change from b10929 (Sept 12) to b11609 (Oct 14) at 32K context, '
                'same models and settings.', 'Observed lesson',
                'b11609 reads faster for 15 of 19 setups (mostly +3 to +9%), so first tokens come sooner. Generation\n'
                'barely moves, except offloaded MoE (+9%) and Ornith (+4%). Muse-Glimmer + DFlash loses 7-13% with a\n'
                'filled context (equal on short prompts), so it stays on b10929. Flash-Next MTP needs b11609.')
    ax = fig.add_axes([0.35, 0.31, 0.58, 0.56])
    style(ax, '', 'Change vs b10929 (%)', 'x')
    ys = range(len(rows))
    ax.barh([y + 0.18 for y in ys], [r[1] for r in rows], height=0.36, color=GREEN, zorder=2, label='generation')
    ax.barh([y - 0.18 for y in ys], [r[2] for r in rows], height=0.36, color=BLUE, zorder=2, label='prompt processing')
    ax.axvline(0, color=INK, lw=1)
    for y, r in zip(ys, rows):
        ax.text(max(r[1], r[2], 0) + 0.4, y, f'{r[1]:+.0f} / {r[2]:+.0f}', va='center', fontsize=9.8, color=INK2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([NAMES[r[0]] for r in rows], fontsize=10.8)
    ax.set_xlim(-15, 17)
    ax.legend(loc='lower right', frameon=False, fontsize=11.5)
    save(fig, '19_llamacpp_version')


ALL = [c01_tasks, c02_scoring, c03_ranking, c04_heatmap, c05_gap, c06_quality_speed, c07_generation, c08_reading,
       c09_efficiency, c10_checks, c11_failure_modes, c12_needles, c13_longctx_cost, c14_reasoning_stability,
       c15_routing, c16_fit_256k, c17_multiuser, c18_engine, c19_version]

if __name__ == '__main__':
    only = set(sys.argv[2:])
    for fn in ALL:
        if not only or fn.__name__.split('_')[0] in only:
            fn()
