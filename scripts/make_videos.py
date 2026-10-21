"""Five short silent LinkedIn videos (1080x1350, 30 fps, H.264) replayed from measured benchmark timings.

  01_first_token_race.mp4  time to first token at ~225K tokens, data/source/phase5b-256k.csv (ttft_256k_s), 10x compressed, Coder-Next omitted
  02_four_users.mp4        four simultaneous users, data/source/multiuser.json (4-client entries), real time
  03_agent_replay.mp4      Task G agent transcripts (Strata Flash-Next vs Nemotron-3-Nano), two terminal panes replayed from
                           results/raw/<model>/repeat/G@s11__*.json of the benchmark repo, 2x compressed
  04_typing_speed.mp4      before / after: the same ~260-token function typed at measured speeds, llama.cpp vs Strata on the
                           same file and Muse-Glimmer without / with DFlash (master-table.csv, gen_32k), real time
  05_round_clock.mp4       one round of all 8 tasks per setup as consecutive segments, per-task mean duration_s from
                           data/repeat-flat.csv, scores from data/source/repeat-runs.csv, 120x compressed

Every frame is drawn with Pillow and piped as raw RGB to the ffmpeg binary shipped by the imageio-ffmpeg pip package
(no matplotlib, no system ffmpeg). Output: videos/*.mp4 plus three check frames per video (videos/check_0N_{a,b,c}.png).
Usage: make_videos.py [01] [02] [03] [04] [05]   (default: all)
"""
import csv
import glob
import json
import os
import re
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, 'data', 'source')
OUT = os.path.join(HERE, 'videos')
os.makedirs(OUT, exist_ok=True)

W, H, FPS = 1080, 1350, 30
SURF, INK, INK2, MUTED, GRID = '#F9F9F4', '#1F2328', '#4B5563', '#8A8F98', '#E3E3DC'
GREEN, BLUE, ORANGE, PURPLE, TEAL, RED = '#76B900', '#3B6FC4', '#F28E2B', '#9B5FC9', '#1FA3A3', '#C8102E'
FOOT = ('RTX 4090 24 GB · 64 GB RAM | llama.cpp b10929 / b11609, Strata | replayed from measured timings | '
        'not a leaderboard')
FONTS = 'C:/Windows/Fonts'
REG, BOLD = f'{FONTS}/arial.ttf', f'{FONTS}/arialbd.ttf'
BLACK = f'{FONTS}/ariblk.ttf' if os.path.exists(f'{FONTS}/ariblk.ttf') else BOLD
MONO = f'{FONTS}/consola.ttf' if os.path.exists(f'{FONTS}/consola.ttf') else f'{FONTS}/cour.ttf'
MARGIN = 60

_fonts = {}


def font(path, size):
    key = (path, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(path, size)
    return _fonts[key]


def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(c1, c2, t):
    """Blend two hex colours, t=0 -> c1, t=1 -> c2."""
    a, b = hex_rgb(c1), hex_rgb(c2)
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


def fit_font(path, text, max_w, size, min_size=36):
    """Largest font size <= size at which text fits into max_w."""
    while size > min_size and font(path, size).getlength(text) > max_w:
        size -= 2
    return font(path, size)


def wrap(text, f, max_w):
    lines, cur = [], ''
    for word in text.split():
        probe = (cur + ' ' + word).strip()
        if f.getlength(probe) <= max_w or not cur:
            cur = probe
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_check(d, x, y, size, color, width=4):
    """Tick mark whose box is (x, y, x+size, y+size)."""
    d.line([(x, y + size * 0.55), (x + size * 0.38, y + size * 0.9), (x + size, y + size * 0.12)],
           fill=color, width=width, joint='curve')


def new_frame():
    img = Image.new('RGB', (W, H), SURF)
    return img, ImageDraw.Draw(img)


def draw_header(d, title, subtitle, right_text=None, right_label=None):
    """Title + subtitle like the charts. Optional big right-aligned readout (used for the stopwatch). Returns y below."""
    reserved = 0
    if right_text:
        f_r = font(BOLD, 48)
        reserved = max(f_r.getlength(right_text), font(REG, 20).getlength(right_label or '')) + 30
        d.text((W - MARGIN, 45), right_text, font=f_r, fill=INK, anchor='ra')
        if right_label:
            d.text((W - MARGIN, 102), right_label, font=font(REG, 20), fill=MUTED, anchor='ra')
    f_t = fit_font(BLACK, title, W - 2 * MARGIN - reserved, 60)
    d.text((MARGIN, 45), title, font=f_t, fill=INK)
    y = 45 + f_t.size + 22
    f_s = font(REG, 28)
    for line in wrap(subtitle, f_s, W - 2 * MARGIN):
        d.text((MARGIN, y), line, font=f_s, fill=INK2)
        y += 36
    return y


def draw_lesson(d, y0, y1, heading, body):
    d.rounded_rectangle([MARGIN, y0, W - MARGIN, y1], radius=18, fill='white', outline=GRID, width=2)
    x, y = MARGIN + 34, y0 + 26
    d.text((x, y), heading, font=font(BOLD, 28), fill=INK)
    y += 46
    f = font(REG, 24)
    for line in wrap(body, f, W - 2 * MARGIN - 68):
        d.text((x, y), line, font=f, fill=INK2)
        y += 33


def draw_footer(d, note=None):
    if note:
        d.text((MARGIN, 1262), note, font=font(REG, 18), fill=MUTED)
    d.text((MARGIN, 1296), FOOT, font=font(REG, 17), fill=INK2)


class Encoder:
    """Pipe raw RGB frames into ffmpeg; also saves check frames at the requested fractions of the video."""

    def __init__(self, path, n_frames, check_prefix, check_at=(0.05, 0.50, 0.98)):
        self.n = n_frames
        self.checks = {max(0, min(n_frames - 1, int(round(fr * n_frames)))): f'{check_prefix}_{tag}.png'
                       for fr, tag in zip(check_at, 'abc')}
        self.log = open(path + '.ffmpeg.log', 'w')
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'info', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', path]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=self.log, stderr=self.log)
        self.i = 0

    def write(self, img):
        if self.i in self.checks:
            img.save(self.checks[self.i])
        self.proc.stdin.write(img.tobytes())
        self.i += 1

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.log.close()
        if self.proc.returncode:
            raise RuntimeError(f'ffmpeg failed with code {self.proc.returncode}, see log')
        os.remove(self.log.name)


def fmt_clock(s):
    return f'{int(s // 60)}:{int(s % 60):02d}'


# ------------------------------------------------------------------------------------------------- video 1: TTFT race
RACE_MODELS = [  # key, display name, tag line, colour
    ('nemotron3nano-30b-iq4xs', 'Nemotron-3-Nano-30B', 'MoE', TEAL),
    ('qwen36-35b-iq4xs', 'Qwen3.6-35B-A3B', 'MoE', TEAL),
    ('gemma4-26b-a4b-qat', 'Gemma-4-26B-A4B', 'MoE', TEAL),
    ('ornith10-q4km', 'Ornith-1.0-35B', 'MoE', TEAL),
    ('ornith15-q4km', 'Ornith-1.5-35B-A3B', 'MoE', TEAL),
    ('qwen38flashnext-gsq-strata', 'Qwen3.8-Flash-Next · Strata', 'Strata · KV cache in system RAM', PURPLE),
    ('qwen3coder-30b-iq4xs', 'Qwen3-Coder-30B-A3B', 'MoE', TEAL),
    ('qwen38-27b-iq4xs', 'Qwen3.8-27B', 'dense', BLUE),
    ('qwen36-27b-q4km', 'Qwen3.6-27B', 'dense', BLUE),
    ('qwen38-27b-iq4xs-mtp6', 'Qwen3.8-27B + MTP', 'dense · multi-token prediction', BLUE),
    ('qwen3codernext-iq4xs-offload', 'Qwen3-Coder-Next 80B', 'MoE · weights partly on CPU', BLUE),
]


def load_race():
    with open(os.path.join(SRC, 'phase5b-256k.csv'), newline='', encoding='utf-8') as fh:
        ttft = {r['model']: float(r['ttft_256k_s']) for r in csv.DictReader(fh) if r['ttft_256k_s'].strip()}
    # Coder-Next (525 s, weights partly on CPU) is left out of the race and mentioned in a note; it would triple the video.
    rows = [(name, tag, color, ttft[key]) for key, name, tag, color in RACE_MODELS
            if key in ttft and key != 'qwen3codernext-iq4xs-offload']
    return sorted(rows, key=lambda r: r[3])


def video_race():
    rows = load_race()
    t_max = max(r[3] for r in rows)                       # 155 s
    compress, hold = 10.0, 2.5
    n_frames = int(round((t_max / compress + hold) * FPS))
    enc = Encoder(os.path.join(OUT, '01_first_token_race.mp4'), n_frames, os.path.join(OUT, 'check_01'))

    axis_max = 180.0                                      # seconds mapped to the full bar width
    bar_x0, bar_x1 = 420, W - MARGIN - 10
    scale = (bar_x1 - bar_x0) / axis_max
    rows_y0, rows_y1 = 345, 1005
    pitch = (rows_y1 - rows_y0) / len(rows)
    bar_h = 30
    f_name, f_tag, f_time, f_small = font(BOLD, 23), font(REG, 17), font(BOLD, 23), font(REG, 18)

    for i in range(n_frames):
        t_real = min(i / FPS * compress, t_max)
        img, d = new_frame()
        y = draw_header(d, 'Who Answers First at 225K Tokens',
                        'Time to first token after sending a ~225K-token document, 24 GB GPU. 10x time compression.')
        d.text((MARGIN, y + 2), 'Not in the race: Qwen3-Coder-Next 80B with weights partly on CPU answers after 525 s (8.8 min).',
               font=font(REG, 20), fill=MUTED)

        # elapsed clock (real seconds) top right of the race area, legend on the left
        clock_y = 232
        f_clock = font(BLACK, 64)
        d.text((W - MARGIN, clock_y - 8), fmt_clock(t_real), font=f_clock, fill=INK, anchor='ra')
        d.text((W - MARGIN, clock_y + 66), 'elapsed, real time', font=f_small, fill=MUTED, anchor='ra')
        lx = MARGIN
        for label, color in (('MoE', TEAL), ('dense', BLUE), ('Strata engine', PURPLE)):
            d.rounded_rectangle([lx, clock_y + 22, lx + 22, clock_y + 44], radius=5, fill=color)
            d.text((lx + 32, clock_y + 22), label, font=font(REG, 21), fill=INK2)
            lx += 32 + font(REG, 21).getlength(label) + 36

        # time ticks behind the bars
        for sec in range(0, int(axis_max) + 1, 30):
            x = bar_x0 + sec * scale
            d.line([(x, rows_y0 - 6), (x, rows_y1 + 4)], fill=GRID, width=2)
            lab = '0' if sec == 0 else f'{sec} s'
            d.text((x, rows_y1 + 10), lab, font=f_small, fill=MUTED, anchor='ma')

        for k, (name, tag, color, ttft) in enumerate(rows):
            cy = rows_y0 + pitch * (k + 0.5)
            d.text((MARGIN, cy - 21), name, font=f_name, fill=INK)
            d.text((MARGIN, cy + 5), tag, font=f_tag, fill=MUTED)
            done = t_real >= ttft
            end = bar_x0 + min(t_real, ttft) * scale
            y0, y1 = cy - bar_h / 2, cy + bar_h / 2
            if done:
                d.rounded_rectangle([bar_x0, y0, end, y1], radius=7, fill=color)
                label = f'{int(ttft + 0.5)} s' + (f'  ({ttft / 60:.1f} min)' if ttft >= 90 else '')  # half-up: 64.5 -> 65
                lw = f_time.getlength(label)
                if end + 34 + lw <= bar_x1:
                    draw_check(d, end + 10, y0 + 4, 22, color, 4)
                    d.text((end + 40, cy - 13), label, font=f_time, fill=INK)
                else:                                      # bar reaches the edge: label inside, in white
                    d.text((end - 12, cy - 13), label, font=f_time, fill='white', anchor='ra')
                    draw_check(d, end - lw - 44, y0 + 4, 22, 'white', 4)
            else:
                d.rounded_rectangle([bar_x0, y0, bar_x0 + ttft * scale, y1], radius=7, fill=mix(SURF, GRID, 0.6))
                if end > bar_x0 + 8:
                    d.rounded_rectangle([bar_x0, y0, end, y1], radius=7, fill=mix(color, SURF, 0.45))
                dots = '.' * (1 + int(i / FPS * 3) % 3)
                tip = max(end, bar_x0)
                d.ellipse([tip - 6, cy - 6, tip + 6, cy + 6], fill=color)
                d.text((tip + 16, cy - 11), 'reading' + dots, font=font(REG, 20), fill=MUTED)

        draw_lesson(d, 1050, 1230, 'Observed lesson',
                    'MoE models answer in 39-58 s. Strata holds 256K by keeping the KV cache in RAM: 65 s. '
                    'Dense 27B models and a VRAM-full Qwen3-Coder-30B wait 2.3-2.6 min.')
        draw_footer(d)
        enc.write(img)
    enc.close()
    return n_frames / FPS


# ------------------------------------------------------------------------------------------- video 2: four users
SNIPPETS = [
    ('Python: dedupe a CSV and keep the newest row', '''\
import csv
from datetime import datetime

def load_rows(path):
    with open(path, newline="") as fh:
        yield from csv.DictReader(fh)

def parse_ts(s):
    return datetime.fromisoformat(s)

def newest_by_email(rows):
    seen = {}
    for row in rows:
        key = row["email"].strip().lower()
        ts = parse_ts(row["updated"])
        if key not in seen or ts > seen[key][0]:
            seen[key] = (ts, row)
    return [r for _, r in seen.values()]

def write_rows(path, rows):
    fields = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fields)
        w.writeheader()
        w.writerows(rows)

if __name__ == "__main__":
    rows = list(load_rows("users.csv"))
    rows = newest_by_email(rows)
    write_rows("users_clean.csv", rows)
    print(f"kept {len(rows)} rows")
'''),
    ('FastAPI: paginated /orders endpoint', '''\
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from .db import get_db
from .models import Order
from .schemas import OrderOut, Page

router = APIRouter(prefix="/orders")

@router.get("", response_model=Page[OrderOut])
def list_orders(
    page: int = Query(1, ge=1),
    size: int = Query(20, le=100),
    status: str | None = None,
    db: Session = Depends(get_db),
):
    q = db.query(Order)
    if status:
        q = q.filter(Order.status == status)
    total = q.count()
    q = q.order_by(Order.created_at.desc())
    items = q.offset((page - 1) * size)
    items = items.limit(size).all()
    return Page(items=items, total=total,
                page=page, size=size)
'''),
    ('Vue: searchable dropdown component', '''\
<script setup lang="ts">
import { ref, computed } from "vue"
const props = defineProps<{ items: string[] }>()
const emit = defineEmits<{ select: [string] }>()
const query = ref("")
const open = ref(false)
const hits = computed(() => {
  const q = query.value.toLowerCase()
  return props.items
    .filter(i => i.toLowerCase().includes(q))
    .slice(0, 8)
})
function pick(v: string) {
  emit("select", v)
  query.value = v
  open.value = false
}
</script>

<template>
  <div class="dd" @focusout="open = false">
    <input v-model="query" @focus="open = true"
           placeholder="Search..." />
    <ul v-if="open && hits.length">
      <li v-for="h in hits" :key="h"
          @mousedown.prevent="pick(h)">
        {{ h }}
      </li>
    </ul>
  </div>
</template>
'''),
    ('SQL: monthly revenue with 3-month rolling average', '''\
WITH monthly AS (
  SELECT date_trunc('month', paid_at) AS month,
         SUM(amount_cents) / 100.0  AS revenue
  FROM payments
  WHERE status = 'succeeded'
    AND paid_at >= now() - interval '24 months'
  GROUP BY 1
),
rolling AS (
  SELECT month, revenue,
         AVG(revenue) OVER (
           ORDER BY month
           ROWS BETWEEN 2 PRECEDING
                AND CURRENT ROW
         ) AS avg_3m
  FROM monthly
)
SELECT to_char(month, 'YYYY-MM') AS month,
       round(revenue, 2)          AS revenue,
       round(avg_3m, 2)           AS avg_3m,
       round(100.0 * (revenue - avg_3m)
             / NULLIF(avg_3m, 0), 1) AS pct
FROM rolling
ORDER BY month;
'''),
]


def load_multiuser():
    with open(os.path.join(SRC, 'multiuser.json'), encoding='utf-8') as fh:
        data = json.load(fh)
    keys = list(data)
    setups = []
    for idx, label, color in ((0, 'Strata, default: one request at a time', PURPLE),
                              (2, 'Muse-Glimmer + DFlash, llama.cpp -np 4', GREEN)):
        entry = next(e for e in data[keys[idx]] if e['clients'] == 4)
        reqs = [entry['requests'][str(c)] for c in range(4)]
        setups.append(dict(label=label, color=color, reqs=reqs, agg=entry['aggregate_tok_s'], wall=entry['wall_s']))
    return setups


def draw_pane(d, box, user_no, req, snippet, t, color):
    x0, y0, x1, y1 = box
    title, code = snippet
    d.rounded_rectangle(box, radius=12, fill='#22272E')
    d.rounded_rectangle([x0, y0, x1, y0 + 36], radius=12, fill='#2D333B')
    d.rectangle([x0, y0 + 24, x1, y0 + 36], fill='#2D333B')
    f_h, f_mono, f_stat = font(BOLD, 19), font(MONO, 16), font(BOLD, 18)
    d.text((x0 + 14, y0 + 9), f'User {user_no}', font=f_h, fill='white')
    d.text((x0 + 92, y0 + 10), title, font=font(REG, 16), fill='#9DA5B0')
    ttft, total, tok_s = req['ttft_s'], req['total_s'], req['tok_s']
    body_x, body_y = x0 + 14, y0 + 48
    if t < ttft:
        d.text((body_x, body_y + 14), 'waiting for first token…', font=font(REG, 22), fill='#E8E9EB')
        d.text((body_x, body_y + 48), f'{t:.1f} s', font=font(BLACK, 40), fill='#FF6B6B')
        return
    frac = min(1.0, (t - ttft) / max(total - ttft, 0.01))
    shown = code[:int(len(code) * frac)]
    lines = shown.splitlines()[-5:]
    done = t >= total
    y = body_y
    for k, line in enumerate(lines):
        line = line[:48]
        d.text((body_x, y), line, font=f_mono, fill='#8F96A0' if done else '#D6DBE1')
        if k == len(lines) - 1 and not done and int(t * 4) % 2 == 0:
            cx = body_x + f_mono.getlength(line) + 2
            d.rectangle([cx, y + 1, cx + 9, y + 18], fill='#D6DBE1')
        y += 18
    # status strip at the bottom of the pane (streaming or done)
    d.rectangle([x0 + 1, y1 - 34, x1 - 1, y1 - 1], fill='#22272E')
    d.rounded_rectangle([x0, y1 - 34, x1, y1], radius=12, fill='#22272E')
    if done:
        draw_check(d, body_x, y1 - 30, 22, color, 4)
        d.text((body_x + 32, y1 - 30), f'done in {total:.1f} s  ·  {tok_s:.0f} tok/s', font=font(BOLD, 20), fill=color)
    else:
        d.text((body_x, y1 - 29), f'streaming  ·  first token at {ttft:.1f} s  ·  {tok_s:.0f} tok/s',
               font=f_stat, fill=color)


def video_users():
    setups = load_multiuser()
    t_end = max(s['wall'] for s in setups)
    hold = 3.0
    n_frames = int(round((t_end + hold) * FPS))
    enc = Encoder(os.path.join(OUT, '02_four_users.mp4'), n_frames, os.path.join(OUT, 'check_02'))

    pane_w, pane_h, gap = 470, 168, 14
    half_tops = (205, 615)

    for i in range(n_frames):
        t = min(i / FPS, t_end)
        img, d = new_frame()
        draw_header(d, 'Four People, One RTX 4090',
                    'Four different coding requests sent at the same moment, replayed at measured timings.',
                    right_text=f'{t:4.1f} s', right_label='elapsed, real time')
        for s, top in zip(setups, half_tops):
            d.rectangle([MARGIN, top + 4, MARGIN + 6, top + 28], fill=s['color'])
            d.text((MARGIN + 18, top), s['label'], font=font(BOLD, 24), fill=INK)
            started = any(t >= r['ttft_s'] for r in s['reqs'])
            all_done = t >= s['wall']
            if all_done:
                ctr = f"all done in {s['wall']:.1f} s  ·  {s['agg']:.0f} tok/s combined"
            elif started:
                ctr = f"all users combined: {s['agg']:.0f} tok/s"
            else:
                ctr = 'all users combined: — tok/s'
            d.text((W - MARGIN, top + 4), ctr, font=font(REG, 20), fill=s['color'] if started else MUTED, anchor='ra')
            for u in range(4):
                col, row = u % 2, u // 2
                x0 = MARGIN + col * (pane_w + gap)
                y0 = top + 42 + row * (pane_h + gap)
                draw_pane(d, (x0, y0, x0 + pane_w, y0 + pane_h), u + 1, s['reqs'][u], SNIPPETS[u], t, s['color'])

        draw_lesson(d, 1035, 1215, 'Observed lesson',
                    'Strata is fastest for one person but serves one request at a time: the 4th user waits 14 s. '
                    'Muse-Glimmer on llama.cpp batches: every first token within 0.5 s, 237 tok/s combined, '
                    'all done in 13.5 s.')
        draw_footer(d, 'Code text is illustrative; timings (first token, finish, tok/s) are measured.')
        enc.write(img)
    enc.close()
    return n_frames / FPS


# ------------------------------------------------------------------------------------------ video 3: agent replay
AGENT_RUNS = [  # label, accent colour, model directory, transcript file (glob)
    ('Qwen3.8-Flash-Next · Strata', PURPLE, 'qwen38flashnext-gsq-strata', 'G@s11__20261014-222030.json'),
    ('Nemotron-3-Nano-30B', BLUE, 'nemotron3nano-30b-iq4xs', 'G@s11__*.json'),
]
STEP_LIMIT = 40
PANE_BG, PANE_HEAD, PANE_TEXT, PANE_DIM, PANE_RED = '#22272E', '#2D333B', '#D6DBE1', '#9DA5B0', '#FF6B6B'
EDIT_TOOLS = ('edit_file', 'write_file', 'write')


def _arg_summary(args):
    if isinstance(args, dict):
        for k in ('path', 'command', 'cmd', 'file'):
            if k in args:
                return ' '.join(str(args[k]).split())
    return ''


def _result_summary(tool, args, result, first_test_run_failed=None):
    """Short '→ ...' text for test runs and refused commands; '' when the result is not worth a word."""
    text = str(result or '')
    is_test = tool == 'run_tests' or (tool == 'run_command' and 'pytest' in _arg_summary(args))
    if text.startswith('ERROR: command not allowed'):
        return 'not allowed'
    if not is_test:
        return 'FINISHED' if tool == 'finish' else ''
    m_fail, m_pass = re.search(r'(\d+) failed', text), re.search(r'(\d+) passed', text)
    if m_fail:
        return f'{m_fail.group(1)} failed'
    if m_pass:
        return f'{m_pass.group(1)} passed'
    if 'no tests ran' in text or 'not found' in text:
        return 'no tests ran'
    if 'AssertionError' in text or 'FAILED' in text or 'Error' in text:
        # the stored result is truncated before pytest's summary line; the first run before any edit fails the
        # baseline count recorded in metrics
        return f'{first_test_run_failed} failed' if first_test_run_failed else 'failing'
    return 'ran'


def load_agent_runs():
    if 'BENCH_RAW' not in os.environ:
        sys.exit(r"Set BENCH_RAW to the benchmark's results\raw folder.")
    runs = []
    for label, color, sub, pattern in AGENT_RUNS:
        path = sorted(glob.glob(os.path.join(os.environ['BENCH_RAW'], sub, 'repeat', pattern)))[0]
        with open(path, encoding='utf-8') as fh:
            data = json.load(fh)
        m = data['metrics']
        calls, seen, edited, repeats = [], set(), False, 0
        for s in data['transcript']:
            tool, args = s['tool'], s.get('args')
            key = (tool, json.dumps(args, sort_keys=True))
            # a call is a wasted repeat when it is identical to an earlier one and nothing was edited in between
            repeat = key in seen
            if tool in EDIT_TOOLS:
                seen.clear()
                edited = True
            seen.add(key)
            repeats += repeat
            is_test = tool == 'run_tests' or (tool == 'run_command' and 'pytest' in _arg_summary(args))
            res = _result_summary(tool, args, s.get('result'),
                                  m.get('baseline_failed') if (is_test and not edited) else None)
            calls.append(dict(step=s['step'], tool=tool, arg=_arg_summary(args), res=res, repeat=repeat))
        assert len(calls) == m['tool_calls'], (label, len(calls), m['tool_calls'])
        assert max(c['step'] for c in calls) == m['steps'], (label, m['steps'])
        runs.append(dict(label=label, color=color, calls=calls, elapsed=float(m['elapsed_s']), steps=m['steps'],
                         tool_calls=m['tool_calls'], success=m['success'], hidden=m['hidden_tests'],
                         finished=m['finished_called'], repeats=repeats, file=os.path.basename(path)))
    return runs


def clip(text, f, max_w):
    if f.getlength(text) <= max_w:
        return text
    while text and f.getlength(text + '…') > max_w:
        text = text[:-1]
    return text + '…'


def draw_cross(d, x, y, size, color, width=4):
    d.line([(x + 2, y + 2), (x + size - 2, y + size - 2)], fill=color, width=width)
    d.line([(x + size - 2, y + 2), (x + 2, y + size - 2)], fill=color, width=width)


def draw_agent_pane(d, box, run, t_real, blink, n_lines=20):
    """Terminal pane replaying one transcript up to real time t_real. Returns (calls shown, current step)."""
    x0, y0, x1, y1 = box
    calls, elapsed, color = run['calls'], run['elapsed'], run['color']
    n = len(calls)
    shown = sum(1 for k in range(n) if (k + 0.5) / n * elapsed <= t_real)
    done = t_real >= elapsed
    d.rounded_rectangle(box, radius=12, fill=PANE_BG)
    d.rounded_rectangle([x0, y0, x1, y0 + 36], radius=12, fill=PANE_HEAD)
    d.rectangle([x0, y0 + 24, x1, y0 + 36], fill=PANE_HEAD)
    f_h = font(REG, 17)
    d.text((x0 + 14, y0 + 9), f'elapsed {min(t_real, elapsed):4.1f} s, real time', font=f_h, fill='white')
    d.text((x1 - 14, y0 + 9), f'{shown} tool calls', font=f_h, fill=PANE_DIM, anchor='ra')

    f_m, lh = font(MONO, 16), 24
    body_x, y = x0 + 14, y0 + 50
    inner_w = x1 - x0 - 28
    for c in calls[:shown][-n_lines:]:
        if c['repeat']:
            prefix, main, tag = '▸ ' + c['tool'] + '  ', c['arg'], '  (repeat)'
            cols = (PANE_RED, PANE_RED, PANE_RED)
        else:
            prefix, main = '▸ ' + c['tool'] + '  ', c['arg']
            tag = ('  → ' + c['res']) if c['res'] else ''
            tag_col = PANE_RED if ('failed' in c['res'] or 'not allowed' in c['res'] or 'failing' in c['res']) else \
                (mix(GREEN, '#FFFFFF', 0.3) if ('passed' in c['res'] or c['res'] == 'FINISHED') else PANE_DIM)
            cols = (mix(color, '#FFFFFF', 0.45), PANE_TEXT, tag_col)
        main = clip(main[:40], f_m, inner_w - f_m.getlength(prefix) - f_m.getlength(tag))
        x = body_x
        for part, col in zip((prefix, main, tag), cols):
            d.text((x, y), part, font=f_m, fill=col)
            x += f_m.getlength(part)
        y += lh
    if not done and blink:
        d.rectangle([body_x, y + 3, body_x + 9, y + 20], fill=PANE_TEXT)

    # status strip at the bottom of the pane
    d.rectangle([x0 + 1, y1 - 70, x1 - 1, y1 - 1], fill=PANE_BG)
    d.rounded_rectangle([x0, y1 - 70, x1, y1], radius=12, fill=PANE_BG)
    d.line([(x0 + 14, y1 - 70), (x1 - 14, y1 - 70)], fill=PANE_HEAD, width=2)
    f_s, f_s2 = font(BOLD, 20), font(REG, 18)
    if done and run['success']:
        draw_check(d, body_x, y1 - 58, 22, GREEN, 4)
        d.text((body_x + 32, y1 - 60), f"finished · hidden tests {run['hidden']}", font=f_s, fill=GREEN)
        d.text((body_x + 32, y1 - 32), f"{run['steps']} turns, {run['tool_calls']} tool calls", font=f_s2, fill=PANE_TEXT)
    elif done:
        draw_cross(d, body_x, y1 - 58, 22, PANE_RED, 4)
        d.text((body_x + 32, y1 - 60), f"step limit reached · hidden tests {run['hidden']}", font=f_s, fill=PANE_RED)
        why = 'never called finish' if not run['finished'] else 'called finish'
        d.text((body_x + 32, y1 - 32), f"{why} · {run['steps']} turns, {run['tool_calls']} tool calls", font=f_s2,
               fill=PANE_TEXT)
    else:
        dots = '.' * (1 + int(t_real * 1.5) % 3)
        d.text((body_x, y1 - 58), 'working' + dots, font=f_s, fill=mix(color, '#FFFFFF', 0.45))
    step = calls[shown - 1]['step'] if shown else 0
    return shown, step


def video_agent():
    runs = load_agent_runs()
    t_max = max(r['elapsed'] for r in runs)                # 34.8 s
    compress, hold = 2.0, 2.5
    n_frames = int(round((t_max / compress + hold) * FPS))
    enc = Encoder(os.path.join(OUT, '03_agent_replay.mp4'), n_frames, os.path.join(OUT, 'check_03'))

    pane_w, gap, pane_y0, pane_y1 = 470, 20, 262, 905
    for i in range(n_frames):
        t_real = min(i / FPS * compress, t_max)
        img, d = new_frame()
        draw_header(d, 'Seven Steps or Forty',
                    'Task G: fix 5 hidden bugs with tools only. Two real agent transcripts, replayed.',
                    right_text=f'{t_real:4.1f} s', right_label='elapsed, real time · 2x compressed')
        for k, run in enumerate(runs):
            x0 = MARGIN + k * (pane_w + gap)
            box = (x0, pane_y0, x0 + pane_w, pane_y1)
            shown, step = draw_agent_pane(d, box, run, t_real, int(t_real * 2) % 2 == 0)
            d.rectangle([x0, pane_y0 - 46, x0 + 6, pane_y0 - 20], fill=run['color'])
            d.text((x0 + 18, pane_y0 - 50), run['label'], font=font(BOLD, 24), fill=INK)
            done = t_real >= run['elapsed']
            step_col = (GREEN if run['success'] else RED) if done else INK2
            d.text((x0 + pane_w, pane_y0 - 46), f'step {step} / {STEP_LIMIT}', font=font(BOLD, 22), fill=step_col, anchor='ra')
            # running counters under the pane
            seen = run['calls'][:shown]
            reps = sum(c['repeat'] for c in seen)
            reads = sum(c['tool'] == 'read_file' for c in seen)
            edits = sum(c['tool'] in EDIT_TOOLS for c in seen)
            f_c = font(REG, 20)
            d.text((x0, pane_y1 + 14), f'repeated calls: {reps}', font=f_c, fill=RED if reps else INK2)
            d.text((x0 + pane_w, pane_y1 + 14), f'files read {reads} · edits {edits}', font=f_c, fill=INK2, anchor='ra')

        draw_lesson(d, 1030, 1230, 'Observed lesson',
                    'The top local agent reads the right files, edits, re-runs the tests and finishes. The weakest one '
                    'spends all 40 steps on shell commands, many of them not allowed, never reads or edits a file, '
                    'and never calls finish.')
        draw_footer(d, 'Tool calls are replayed from the saved transcripts; "(repeat)" marks a call identical to an '
                       'earlier one with no edit in between.')
        enc.write(img)
    enc.close()
    return n_frames / FPS


# ------------------------------------------------------------------------------------------ video 4: typing speed
SLATE = '#8C9AB0'   # "before" panes: muted, so the eye goes to the optimised side
TYPING_ROWS = [  # row label, what changed, before (master-table key, pane label, colour), after (same)
    ('Qwen3.8-Flash-Next · 71 GB at 3-bit, mostly in system RAM', 'change: the engine',
     ('qwen38flashnext-gsq-iq3xxs-b11609', 'llama.cpp b11609 · GSQ-RCO file', SLATE),
     ('qwen38flashnext-gsq-strata', 'Strata · same GSQ-RCO file', PURPLE)),
    ('Muse-Glimmer-30B · fully on the GPU', 'change: + DFlash drafter',
     ('museglimmer-30b-q4km', 'llama.cpp b10929', SLATE),
     ('museglimmer-30b-q4km-dflash16', 'llama.cpp b10929 + DFlash', GREEN)),
]
TYPING_SETUPS = [s for row in TYPING_ROWS for s in row[2:]]   # before, after, before, after
CHARS_PER_TOKEN = 4
TYPING_CODE = '''\
import csv
from collections import defaultdict
from datetime import datetime


def parse(row):
    try:
        ts = datetime.fromisoformat(row["ts"])
        kw = float(row["kw"])
    except (KeyError, ValueError):
        return None
    if kw < 0 or row.get("status") != "1":
        return None
    return ts.date().isoformat(), kw * 0.25


def daily_kwh(path):
    totals = defaultdict(float)
    skipped = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            hit = parse(row)
            if hit is None:
                skipped += 1
                continue
            day, kwh = hit
            totals[day] += kwh
    rows = [(d, round(totals[d], 2))
            for d in sorted(totals)]
    return rows, skipped


def print_report(path):
    rows, skipped = daily_kwh(path)
    for day, kwh in rows:
        print(f"{day}  {kwh:>9.2f} kWh")
    total = sum(k for _, k in rows)
    print(f"total  {total:>9.2f} kWh")
    print(f"skipped {skipped} rows")


if __name__ == "__main__":
    print_report("meter.csv")
'''


def load_typing_speeds():
    with open(os.path.join(SRC, 'master-table.csv'), newline='', encoding='utf-8-sig') as fh:
        rows = {}
        for r in csv.DictReader(fh):
            rows.setdefault(r['model'], r)
    return [dict(key=key, label=label, color=color, tok_s=float(rows[key]['gen_32k']))
            for key, label, color in TYPING_SETUPS]


def draw_typing_pane(d, box, setup, t, n_tokens, blink, n_lines=10):
    x0, y0, x1, y1 = box
    color, tok_s = setup['color'], setup['tok_s']
    code = TYPING_CODE
    t_done = len(code) / (tok_s * CHARS_PER_TOKEN)
    d.rounded_rectangle(box, radius=12, fill=PANE_BG)
    d.rounded_rectangle([x0, y0, x1, y0 + 36], radius=12, fill=PANE_HEAD)
    d.rectangle([x0, y0 + 24, x1, y0 + 36], fill=PANE_HEAD)
    d.rectangle([x0 + 12, y0 + 9, x0 + 17, y0 + 27], fill=color)
    d.text((x0 + 26, y0 + 9), setup['label'], font=font(BOLD, 17), fill='white')
    f_m, lh = font(MONO, 17), 27
    body_x, y = x0 + 14, y0 + 50
    n_chars = int(min(len(code), max(0.0, t) * tok_s * CHARS_PER_TOKEN))
    done = n_chars >= len(code)
    shown = code[:n_chars]
    lines = shown.split('\n')[-n_lines:] if n_chars else []
    for k, line in enumerate(lines):
        d.text((body_x, y), line, font=f_m, fill=PANE_DIM if done else PANE_TEXT)
        if k == len(lines) - 1 and not done and blink:
            cx = body_x + f_m.getlength(line) + 2
            d.rectangle([cx, y + 2, cx + 10, y + 22], fill=PANE_TEXT)
        y += lh
    if not lines and not done and blink:
        d.rectangle([body_x, y + 2, body_x + 10, y + 22], fill=PANE_TEXT)
    d.rectangle([x0 + 1, y1 - 44, x1 - 1, y1 - 1], fill=PANE_BG)
    d.rounded_rectangle([x0, y1 - 44, x1, y1], radius=12, fill=PANE_BG)
    d.line([(x0 + 14, y1 - 44), (x1 - 14, y1 - 44)], fill=PANE_HEAD, width=2)
    tokens = int(round(n_chars / CHARS_PER_TOKEN))
    if done:
        draw_check(d, body_x, y1 - 34, 22, color, 4)
        d.text((body_x + 32, y1 - 36), f'done in {t_done:.1f} s  ·  {n_tokens} tokens  ·  {tok_s:.0f} tok/s',
               font=font(BOLD, 20), fill=color)
    else:
        d.text((body_x, y1 - 36), f'{tokens} / {n_tokens} tokens  ·  {tok_s:.0f} tok/s', font=font(BOLD, 20),
               fill=mix(color, '#FFFFFF', 0.25))


def video_typing():
    setups = load_typing_speeds()
    n_tokens = int(round(len(TYPING_CODE) / CHARS_PER_TOKEN))
    t_type = max(len(TYPING_CODE) / (s['tok_s'] * CHARS_PER_TOKEN) for s in setups)   # slowest pane, ~9.6 s
    lead = 1.0                                                                        # panes empty, speeds visible
    hold = max(2.5, 15.0 - lead - t_type)                                             # at least 2.5 s, video >= 15 s
    n_frames = int(round((lead + t_type + hold) * FPS))
    enc = Encoder(os.path.join(OUT, '04_typing_speed.mp4'), n_frames, os.path.join(OUT, 'check_04'))

    # two rows (one model each), BEFORE on the left, AFTER on the right, an arrow with the speed-up in between
    gap = 56
    pane_w = (W - 2 * MARGIN - gap) // 2
    pane_h, col_y, row_label_h, row_gap = 318, 240, 36, 22
    rows_y = [col_y + 40, col_y + 40 + row_label_h + pane_h + row_gap]
    v = {s['key']: s['tok_s'] for s in setups}
    gains = [v[r[3][0]] / v[r[2][0]] for r in TYPING_ROWS]
    pairs = ' and '.join(f"{v[r[2][0]]:.0f} / {v[r[3][0]]:.0f}" for r in TYPING_ROWS)
    for i in range(n_frames):
        t = min(i / FPS - lead, t_type)
        img, d = new_frame()
        y = draw_header(d, 'Before and After, Same GPU',
                        "Left: before. Right: one change. Characters appear at each setup's measured tokens/s "
                        '(32K context); the code is illustrative.')
        # clock on its own line under the subtitle (the header's right-hand readout would touch the 2-line subtitle)
        f_clock = font(BOLD, 34)
        clock = f'{max(t, 0.0):4.1f} s'
        d.text((W - MARGIN, y - 6), clock, font=f_clock, fill=INK, anchor='ra')
        d.text((W - MARGIN - f_clock.getlength(clock) - 14, y + 6), 'elapsed, real time', font=font(REG, 20),
               fill=MUTED, anchor='ra')
        f_col = font(BOLD, 22)
        d.text((MARGIN, col_y), 'BEFORE', font=f_col, fill=MUTED)
        d.text((MARGIN + pane_w + gap, col_y), 'AFTER', font=f_col, fill=INK)
        blink = int(i / FPS * 4) % 2 == 0
        for r, (label, change, before, after) in enumerate(TYPING_ROWS):
            ry = rows_y[r]
            d.text((MARGIN, ry), label, font=font(BOLD, 21), fill=INK)
            d.text((W - MARGIN, ry + 1), change, font=font(REG, 20), fill=INK2, anchor='ra')
            y0 = ry + row_label_h
            for c, s in enumerate(setups[2 * r: 2 * r + 2]):
                x0 = MARGIN + c * (pane_w + gap)
                draw_typing_pane(d, (x0, y0, x0 + pane_w, y0 + pane_h), s, t, n_tokens, blink, n_lines=8)
            # arrow and speed-up between the two panes
            cx, cy = MARGIN + pane_w + gap // 2, y0 + pane_h // 2
            d.text((cx, cy - 20), f'{gains[r]:.1f}x', font=font(BOLD, 21), fill=INK, anchor='mb')
            d.line([(cx - 20, cy), (cx + 12, cy)], fill=INK, width=4)
            d.polygon([(cx + 22, cy), (cx + 8, cy - 10), (cx + 8, cy + 10)], fill=INK)
        b1, a1, b2, a2 = (v[k] for r in TYPING_ROWS for k in (r[2][0], r[3][0]))
        draw_lesson(d, 1035, 1230, 'Observed lesson',
                    f'Same GPU, one change per row. The identical 3-bit Flash-Next file writes {b1:.0f} tok/s on '
                    f'llama.cpp and {a1:.0f} on Strata ({gains[0]:.1f}x). Muse-Glimmer, fully on the GPU, goes from '
                    f'{b2:.0f} to {a2:.0f} tok/s with its DFlash drafter ({gains[1]:.1f}x).')
        draw_footer(d, f'Code is illustrative; speeds are measured ({n_tokens} tokens at 1 token ≈ {CHARS_PER_TOKEN} '
                       f'characters; {pairs} tok/s).')
        enc.write(img)
        if i == n_frames - 1:                                    # the final frame doubles as the LinkedIn thumbnail
            os.makedirs(os.path.join(OUT, 'thumbnails'), exist_ok=True)
            img.save(os.path.join(OUT, 'thumbnails', '04_typing_speed_thumbnail.jpg'), quality=92)
    enc.close()
    return n_frames / FPS


# ------------------------------------------------------------------------------------------- video 5: round clock
ROUND_SETUPS = [  # key, display name, tag, colour
    ('sonnet55-medium', 'Claude Sonnet 5.5 · medium', 'cloud', ORANGE),
    ('gpt61sol-medium', 'GPT-6.1 Sol · medium', 'cloud', ORANGE),
    ('haiku55-medium', 'Claude Haiku 5.5 · medium', 'cloud', ORANGE),
    ('qwen38flashnext-gsq-strata', 'Qwen3.8-Flash-Next · Strata', 'local · Strata', PURPLE),
    ('museglimmer-30b-q4km-dflash16', 'Muse-Glimmer-30B + DFlash', 'local', GREEN),
    ('museglimmer-30b-q4km', 'Muse-Glimmer-30B', 'local', BLUE),
    ('ornith15-q4km', 'Ornith-1.5-35B-A3B', 'local', BLUE),
    ('qwen36-35b-iq4xs', 'Qwen3.6-35B-A3B', 'local', BLUE),
]
ROUND_NOT_SHOWN = 'qwen38flashnext-iq3xxs'          # Flash-Next on stock llama.cpp, ~28 min: mentioned in a note only
TASKS = 'ABCDEFGH'


def load_round():
    """Per setup: per-task mean duration (s) over its runs, round time = sum of the 8 means, weighted score."""
    acc = {}
    with open(os.path.join(HERE, 'data', 'repeat-flat.csv'), newline='', encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if r['duration_s'].strip():
                acc.setdefault(r['model'], {}).setdefault(r['task'], []).append(float(r['duration_s']))
    with open(os.path.join(SRC, 'repeat-runs.csv'), newline='', encoding='utf-8-sig') as fh:
        score = {r['model']: float(r['weighted_mean']) for r in csv.DictReader(fh)}

    def one(key):
        per = [sum(acc[key][t]) / len(acc[key][t]) for t in TASKS]
        return dict(key=key, per=per, total=sum(per), score=score[key])

    setups = []
    for key, name, tag, color in ROUND_SETUPS:
        s = one(key)
        s.update(name=name, tag=tag, color=color)
        setups.append(s)
    return setups, one(ROUND_NOT_SHOWN)


def video_round():
    setups, not_shown = load_round()
    for s in setups + [not_shown]:
        print(f"  {s['key']:32s} round {s['total'] / 60:5.2f} min  score {s['score']:.1f}  "
              f"tasks {' '.join(f'{v:.0f}' for v in s['per'])} s")
    t_max = max(s['total'] for s in setups)               # 990 s, Muse-Glimmer
    compress, hold = 120.0, 1.5
    n_frames = int(round((t_max / compress + hold) * FPS))
    enc = Encoder(os.path.join(OUT, '05_round_clock.mp4'), n_frames, os.path.join(OUT, 'check_05'))

    axis_max = 18 * 60.0                                  # seconds mapped to the full track width
    bar_x0, bar_x1 = 400, W - MARGIN
    scale = (bar_x1 - bar_x0) / axis_max
    rows_y0, rows_y1 = 345, 1000
    pitch = (rows_y1 - rows_y0) / len(setups)
    bar_h, gap = 34, 2
    f_name, f_tag, f_done, f_small, f_seg = font(BOLD, 23), font(REG, 17), font(BOLD, 21), font(REG, 18), font(BOLD, 18)
    track = mix(SURF, GRID, 0.6)
    best = setups[0]
    strata = next(s for s in setups if s['key'] == 'qwen38flashnext-gsq-strata')
    muse = next(s for s in setups if s['key'] == 'museglimmer-30b-q4km')
    dflash = next(s for s in setups if s['key'] == 'museglimmer-30b-q4km-dflash16')
    ornith = next(s for s in setups if s['key'] == 'ornith15-q4km')

    for i in range(n_frames):
        t_real = min(i / FPS * compress, t_max)
        img, d = new_frame()
        y = draw_header(d, 'One Round of Eight Tasks',
                        'Wall-clock time to answer all 8 tasks, mean of 3 runs. Cloud through subscription CLIs.')
        d.text((MARGIN, y + 2), f"Not shown: the same Flash-Next weights on stock llama.cpp take "
                                f"{not_shown['total'] / 60:.0f} min.", font=font(REG, 20), fill=MUTED)

        # big clock top right, legend on the left (same band as video 01)
        clock_y = 240
        f_clock = font(BLACK, 64)
        d.text((W - MARGIN, clock_y - 8), fmt_clock(t_real), font=f_clock, fill=INK, anchor='ra')
        d.text((W - MARGIN, clock_y + 66), 'elapsed, real time · 120x', font=f_small, fill=MUTED, anchor='ra')
        lx = MARGIN
        for label, color in (('cloud', ORANGE), ('local llama.cpp', BLUE), ('local + DFlash', GREEN),
                             ('Strata engine', PURPLE)):
            d.rounded_rectangle([lx, clock_y + 22, lx + 22, clock_y + 44], radius=5, fill=color)
            d.text((lx + 32, clock_y + 22), label, font=font(REG, 21), fill=INK2)
            lx += 32 + font(REG, 21).getlength(label) + 30

        # time axis behind the rows, ticks every 2 min
        for m in range(0, 19, 2):
            x = bar_x0 + m * 60 * scale
            d.line([(x, rows_y0 - 6), (x, rows_y1 + 4)], fill=GRID, width=2)
            d.text((x, rows_y1 + 10), '0' if m == 0 else f'{m} min', font=f_small, fill=MUTED, anchor='ma')

        for k, s in enumerate(setups):
            cy = rows_y0 + pitch * (k + 0.5)
            y0, y1 = cy - bar_h / 2, cy + bar_h / 2
            d.text((MARGIN, cy - 21), s['name'], font=f_name, fill=INK)
            d.text((MARGIN, cy + 5), s['tag'], font=f_tag, fill=MUTED)
            total, color = s['total'], s['color']
            done = t_real >= total
            end = bar_x0 + min(t_real, total) * scale
            # faint track as long as the full round
            d.rounded_rectangle([bar_x0, y0, bar_x0 + total * scale, y1], radius=7, fill=track)
            # DFlash is the same weights as Muse-Glimmer (quality within noise); its own 3-run mean is dragged down by
            # thinking-to-cap runs, so show the base model's score rather than a misleading separate one.
            if s['key'] == 'museglimmer-30b-q4km-dflash16':
                label = f"{total / 60:.1f} min · same weights, 92.5"
            else:
                label = f"{total / 60:.1f} min · score {s['score']:.1f}"
            lw = f_done.getlength(label)
            inside = done and end + 40 + lw > bar_x1          # no room right of the bar: badge inside the bar
            badge_x0 = end - lw - 60
            # task segments A-H, consecutive, filling left to right; alternating base / darker shade, 2 px cream gap
            start = 0.0
            for j, dur in enumerate(s['per']):
                if t_real <= start:
                    break
                sx0 = bar_x0 + start * scale + (gap / 2 if j else 0)
                sx1 = bar_x0 + min(t_real, start + dur) * scale - (gap / 2 if j < 7 else 0)
                shade = color if j % 2 == 0 else mix(color, '#000000', 0.22)
                if sx1 > sx0:
                    d.rectangle([sx0, y0, sx1, y1], fill=shade)
                    cx = (sx0 + sx1) / 2
                    if dur * scale - gap >= 26 and sx1 - sx0 >= 20 and not (inside and cx + 10 > badge_x0):
                        d.text((cx, cy), TASKS[j], font=f_seg, fill='white', anchor='mm')
                start += dur
            if done:
                if not inside:
                    draw_check(d, end + 10, y0 + 6, 22, color, 4)
                    d.text((end + 40, cy - 12), label, font=f_done, fill=INK)
                else:                                      # bar reaches the edge: dark badge at the bar end, white label
                    d.rounded_rectangle([badge_x0, y0, end, y1], radius=7, fill=mix(color, '#000000', 0.5))
                    d.text((end - 12, cy - 12), label, font=f_done, fill='white', anchor='ra')
                    draw_check(d, end - lw - 44, y0 + 6, 22, 'white', 4)
            else:
                elapsed = fmt_clock(t_real)
                tw = f_small.getlength(elapsed)
                if end + 12 + tw <= bar_x1 + 10:
                    d.text((end + 12, cy - 11), elapsed, font=f_small, fill=MUTED)
                else:
                    d.text((end - 10, cy - 11), elapsed, font=f_small, fill='white', anchor='ra')

        draw_lesson(d, 1050, 1230, 'Observed lesson',
                    f"Claude Sonnet 5.5 is the best and among the fastest: {best['total'] / 60:.1f} min for a round. "
                    f"Strata runs the same 8 tasks in {strata['total'] / 60:.1f} min at {strata['score']:.1f}. "
                    f"Muse-Glimmer needs {muse['total'] / 60:.1f} min, {dflash['total'] / 60:.0f} with its DFlash drafter at the same quality. "
                    f"Ornith is as fast as Strata but scores {ornith['score']:.0f}.")
        draw_footer(d, "Segments A-H are the eight tasks; each width is that task's mean duration for the setup.")
        enc.write(img)
    enc.close()
    return n_frames / FPS


if __name__ == '__main__':
    which = sys.argv[1:] or ['01', '02', '03', '04', '05']
    if '01' in which:
        print(f'01_first_token_race.mp4: {video_race():.1f} s')
    if '02' in which:
        print(f'02_four_users.mp4: {video_users():.1f} s')
    if '03' in which:
        print(f'03_agent_replay.mp4: {video_agent():.1f} s')
    if '04' in which:
        print(f'04_typing_speed.mp4: {video_typing():.1f} s')
    if '05' in which:
        print(f'05_round_clock.mp4: {video_round():.1f} s')
