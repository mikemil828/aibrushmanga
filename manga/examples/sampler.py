"""Tool sampler: every pen, hatching over a gradient, screentones, focus/speed lines and panels on one sheet."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from manga import Sheet, hatch, tone, effects

out = sys.argv[1] if len(sys.argv) > 1 else 'sampler'
W, H = 1600, 1200
t = time.time()
sh = Sheet(W, H, px_per_mm=8.0, journal=out + '.jsonl')
rows = effects.layout(W, H, [(1, [1, 1]), (1, [1.3, 1]), (1, [1, 1, 1])], margin_mm=6, px_per_mm=8)
P = [p for r in rows for p in r]
def box(p): return (p[0][0], p[0][1], p[2][0], p[2][1])

# 1. pens: one wave per pen, pressure from light to hard
x0, y0, x1, y1 = box(P[0])
with sh.clip(P[0]):
    for i, pen in enumerate(['g_pen', 'saji', 'maru', 'brush', 'fineliner']):
        y = y0 + 40 + i * 62
        for j, pr in enumerate([0.3, 0.6, 1.0]):
            xa = x0 + 30 + j * 230
            sh.stroke([(xa, y + 20), (xa + 70, y - 12), (xa + 140, y + 18), (xa + 200, y - 5)], pen=pen, pressure=pr)
# 2. hatching over a left-to-right gradient
x0, y0, x1, y1 = box(P[1])
g = np.zeros((H, W), np.float32)
g[:, int(x0):int(x1)] = np.linspace(0, 1, int(x1) - int(x0))[None, :]
m = np.zeros((H, W), bool); m[int(y0) + 10:int(y1) - 10, int(x0) + 10:int(x1) - 10] = True
hatch(sh, g, m, pen='maru', angle=60, spacing_mm=0.8, seed=3)
# 3. tones: dots at four levels, a gradient, lines, sand; scratched highlight
x0, y0, x1, y1 = box(P[2])
w = (x1 - x0) / 5
for i, lv in enumerate([0.1, 0.3, 0.5, 0.75]):
    tone(sh, [(x0 + i * w, y0), (x0 + (i + 1) * w, y0), (x0 + (i + 1) * w, y1), (x0 + i * w, y1)], level=lv, lpi=32)
tone(sh, [(x0 + 4 * w, y0), (x1, y0), (x1, (y0 + y1) / 2), (x0 + 4 * w, (y0 + y1) / 2)], level=0.4, lpi=32, kind='line', angle=0)
tone(sh, [(x0 + 4 * w, (y0 + y1) / 2), (x1, (y0 + y1) / 2), (x1, y1), (x0 + 4 * w, y1)], level=0.35, kind='noise', seed=5)
with sh.clip(P[2]):
    for k in range(6):
        sh.stroke([(x0 + 2.2 * w + k * 9, y0 + 30), (x0 + 1.6 * w + k * 9, y1 - 30)], pen='white', pressure=0.5, mode='white')
# 4. gradient tone
x0, y0, x1, y1 = box(P[3])
tone(sh, P[3], level=(0.05, 0.7, (x0, y0), (x0, y1)), lpi=32)
# 5. focus lines
x0, y0, x1, y1 = box(P[4])
with sh.clip(P[4]):
    effects.focus_lines(sh, ((x0 + x1) / 2, (y0 + y1) / 2), (70, 50), (x0, y0, x1, y1), count=200, seed=2)
# 6. speed lines
x0, y0, x1, y1 = box(P[5])
with sh.clip(P[5]):
    effects.speed_lines(sh, (x0, y0, x1, y1), angle=-15, count=120, seed=4)
# 7. flash + beta
x0, y0, x1, y1 = box(P[6])
with sh.clip(P[6]):
    sh.fill(P[6])
    sh.stroke([((x0 + x1) / 2, (y0 + y1) / 2)], pen='white', pressure=1.0, mode='white')
    c = ((x0 + x1) / 2, (y0 + y1) / 2)
    yy, xx = np.mgrid[0:H, 0:W]
    disk = (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < 110 ** 2
    sh.fill(disk, mode='white')
    effects.flash(sh, c, 30, 120, count=160, seed=7)
for p in P:
    effects.border(sh, p)
effects.caption(sh, (P[0][0][0] + 8, P[0][2][1] - 40), 'PENS: G / SAJI / MARU / BRUSH / LINER', size_mm=2.6)
sh.png(out + '.png')
sh.close()
print('done', time.time() - t, sh.stats)
