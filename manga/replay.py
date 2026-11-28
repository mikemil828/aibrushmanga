"""Replay a manga journal, optionally at another scale:  python -m manga.replay page.jsonl --scale 3 --out big.png"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from .sheet import Sheet
from .pens import Pen
from .tone import _tone as _tone_fn
from . import effects


def _load_array(jdir, name):
    a = np.asarray(Image.open(os.path.join(jdir, name)))
    if a.dtype == np.uint8:
        return a > 127
    return a.astype(np.float32) / 65535.0


def _mask(jdir, m):
    if m is None:
        return None
    if 'poly' in m:
        return [tuple(p) for p in m['poly']]
    return _load_array(jdir, m['file'])


# paper sizes in mm, for --fit
PAPERS = {'letter': (215.9, 279.4), 'a4': (210.0, 297.0), 'b4': (257.0, 364.0), 'legal': (215.9, 355.6)}


def replay(path, scale=1.0, until=None, progress=False, snap=False, on_op=None):
    jdir = path + '.d'
    sh = None
    clips = []
    t0 = time.time()
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if until is not None and i >= until:
                break
            d = json.loads(line)
            op = d['op']
            if op == 'sheet':
                sh = Sheet(d['width'], d['height'], d['px_per_mm'], tuple(d['paper']), tuple(d['ink']),
                           seed=d['seed'], scale=scale)
                sh._replaying = True
                sh.snap_screens = snap
            elif op == 'stroke':
                sh._n += 1
                sh.stats['strokes'] += 1
                sh._stroke(np.asarray(d['points'], float), Pen.from_dict(d['pen']), d['pressure'], d['seed'],
                           d['mode'], tuple(d['taper']) if d['taper'] is not None else None, d.get('smooth', True))
            elif op == 'fill':
                sh._n += 1
                sh._put(sh.mask(_mask(jdir, d['mask'])) * d['amount'], 0, 0, d['mode'])
            elif op == 'apply':
                sh._n += 1
                sh._put(sh.raster(_load_array(jdir, d['cov']['file'])), 0, 0, d['mode'])
            elif op == 'tone':
                sh._n += 1
                lv = d['level']
                if isinstance(lv, dict) and 'file' in lv:
                    lv = _load_array(jdir, lv['file'])
                _tone_fn(sh, _mask(jdir, d['mask']), lv, d['lpi'], d['angle'], d['kind'],
                                tuple(d['phase']), d['mode'], d['seed'], d.get('grain_mm', 0.12),
                         d.get('soft_mm', 0.0))
            elif op == 'clip':
                sh._n += 1
                cm = sh.mask(_mask(jdir, d['mask']))
                sh._clips.append(cm)
            elif op == 'unclip':
                sh._clips.pop()
            elif op == 'border':
                sh._n += 1
                effects._border(sh, [tuple(p) for p in d['poly']], d['width_mm'])
            elif op == 'caption':
                effects.caption(sh, tuple(d['xy']), d['text'], d['size_mm'], d['pad_mm'], d['box'], d['bold'],
                                d.get('face'))
            if on_op is not None and sh is not None:
                on_op(i, op, sh)                        # after each journal line: frames for an animation
            if progress and i % 2000 == 0:
                print(f'  line {i}  {time.time() - t0:.0f}s', flush=True)
    return sh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('journal')
    ap.add_argument('--scale', type=float, default=1.0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--bw', action='store_true', help='pure 1-bit black and white output')
    ap.add_argument('--until', type=int, default=None, help='stop before this journal line')
    ap.add_argument('--dpi', type=float, default=None,
                    help="printer resolution, e.g. 600: renders for that printer (sets the scale, tags the PNG, "
                         "snaps screens to whole pixels). Print the PNG at 100%%/actual size.")
    ap.add_argument('--fit', choices=sorted(PAPERS), default=None,
                    help='with --dpi: shrink the page to fit this paper (0.25 in margins) instead of its own size')
    a = ap.parse_args()
    t = time.time()
    scale = a.scale
    if a.dpi:
        with open(a.journal, encoding='utf-8') as f:
            d = json.loads(f.readline())
        w_mm, h_mm = d['width'] / d['px_per_mm'], d['height'] / d['px_per_mm']
        shrink = 1.0
        if a.fit:
            pw, ph = PAPERS[a.fit]
            shrink = min(1.0, (pw - 12.7) / w_mm, (ph - 12.7) / h_mm)
        scale = a.dpi / 25.4 * shrink / d['px_per_mm']
    sh = replay(a.journal, scale, a.until, progress=True, snap=bool(a.dpi or a.bw))
    sh.png(a.out, bw=a.bw, dpi=a.dpi)
    print(f'{a.out}: {sh.W}x{sh.H}, {sh.stats["strokes"]} strokes, {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
