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


def replay(path, scale=1.0, until=None, progress=False):
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
                                tuple(d['phase']), d['mode'], d['seed'], d.get('grain_mm', 0.12))
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
    a = ap.parse_args()
    t = time.time()
    sh = replay(a.journal, a.scale, a.until, progress=True)
    sh.png(a.out, bw=a.bw)
    print(f'{a.out}: {sh.W}x{sh.H}, {sh.stats["strokes"]} strokes, {time.time() - t:.0f}s')


if __name__ == '__main__':
    main()
