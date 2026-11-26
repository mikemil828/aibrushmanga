"""Re-draw a grey manga image as ink: traced lines as pen strokes, one stock tone per area, solid blacks.

Usage: python manga/examples/replicate_drawn.py image.png   (writes drawn.png and drawn.jsonl)
"""
import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..'))
from PIL import Image
from scipy import ndimage as nd
from manga import Sheet, tone, prep, trace

ref = np.asarray(Image.open(sys.argv[1]).convert('RGB'), np.float32) / 255
H, W = ref.shape[:2]
gray = ref.mean(2)
d = 1 - gray
t0 = time.time()
lines = prep.xdog_lines(ref, sigma=0.9, p=24, eps=-0.015, phi=80, tidy=12) > 0.5
sh = Sheet(W, H, px_per_mm=8.0, journal='drawn.jsonl')
# 1. tone, area by area (laid first: ink goes over it)
t = time.time()
barrier = nd.binary_dilation(lines, iterations=1)
lv = trace.area_tones(d, barrier)
tone(sh, None, level=np.where(lv > 0.6, lv, lv * 0.8), lpi=65)
# white edge (fuchidori): scrape a 1 px paper margin round every line so ink reads through the tone
blk0 = nd.gaussian_filter((nd.gaussian_filter(d, 1.0) > 0.80).astype(np.float32), 1.0) > 0.5
sh.fill(nd.binary_dilation(lines, iterations=1) & ~blk0, mode='white')
print('tones', round(time.time() - t), 's', flush=True)
# 2. ink: centre lines as pen strokes, thick masses as beta
t = time.time()
n, beta = trace.ink_line_art(sh, lines, seed=3, weight=1.5)
# solid blacks: the darkest masses (hair shadow, deep shade) are filled, as an inker blocks them in
blk = nd.gaussian_filter((nd.gaussian_filter(d, 1.0) > 0.80).astype(np.float32), 1.0) > 0.5
lab, k = nd.label(blk)
if k:
    sz = nd.sum(blk, lab, range(1, k + 1)); blk = np.isin(lab, 1 + np.nonzero(sz >= 30)[0])
sh.fill(blk)
print('strokes', n, round(time.time() - t), 's', flush=True)
sh.png('drawn.png'); sh.close()
out = np.asarray(sh.image().convert('L'), np.float32) / 255
mae = np.abs(nd.gaussian_filter(out, 3) - nd.gaussian_filter(gray, 3)).mean()
print('total', round(time.time() - t0), 's  blurred-MAE', round(float(mae), 3))
print('saved drawn.png')
