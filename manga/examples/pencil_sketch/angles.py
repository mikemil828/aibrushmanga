"""The pencil-sketch girl from six angles, built by character.py on the same construction.
Run from this folder: python3 angles.py  ->  angles.png (ink) and angles-sketch.png (pencil under the ink)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import Sheet, effects            # noqa: E402
from character import draw_character        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# (label, turn, tilt, nod): facing us, three-quarter, profile; looking down, looking up, turned the other way
POSES = [('facing', 0.0, 0.0, 0.0), ('three-quarter', 0.6, -4.0, 0.0), ('profile', 1.0, 0.0, 0.0),
         ('looking down', 0.35, -6.0, 0.55), ('looking up', -0.3, 5.0, -0.5), ('other side', -0.7, 6.0, 0.15)]
CW, CH, R = 680, 820, 140
sh = Sheet(3 * CW, 2 * CH, px_per_mm=8.0, journal=os.path.join(HERE, 'angles.jsonl'), seed=7)
for i, (label, turn, tilt, nod) in enumerate(POSES):
    x0, y0 = CW * (i % 3), CH * (i // 3)
    draw_character(sh, (x0 + CW / 2, y0 + 300), R, turn=turn, tilt=tilt, nod=nod, seed=10 + i)
    effects.border(sh, [(x0 + 15, y0 + 15), (x0 + CW - 15, y0 + 15), (x0 + CW - 15, y0 + CH - 15),
                        (x0 + 15, y0 + CH - 15)], width_mm=0.6)
sh.png(os.path.join(HERE, 'angles-sketch.png'), sketch=True)
sh.png(os.path.join(HERE, 'angles.png'))
sh.close()
print('strokes', sh.stats['strokes'])
