"""Loomis turnaround: sketch.head from facing us to a full profile, looking down and up, leaning and turned
away. Run from this folder: python3 turnaround.py  ->  turnaround.png (pencil construction in blue, the face
outline the construction returns inked in black, eye openings roughed in pencil)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import Sheet                       # noqa: E402
from manga import sketch as sk                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# (turn, tilt, nod): facing, 30 degrees, three-quarter, nearly profile, profile;
# then looking down, looking up, turned the other way and down, leaning, profile facing left
POSES = [(0, 0, 0), (0.5, 0, 0), (0.71, 0, 0), (0.92, 0, 0), (1.0, 0, 0),
         (0.5, 0, 0.6), (0.5, 0, -0.6), (-0.6, 8, 0.3), (0, -10, 0), (-1.0, 0, 0)]
R = 150
sh = Sheet(5 * 500, 2 * 650, px_per_mm=8.0, seed=3)
for i, (turn, tilt, nod) in enumerate(POSES):
    cx, cy = 250 + 500 * (i % 5), 220 + 650 * (i // 5)
    h = sk.head(sh, (cx, cy), R, turn=turn, tilt=tilt, nod=nod, seed=i)
    for e in 'lr':
        if h['eye_vis_' + e]:
            sk.curve(sh, h['eye_shape_' + e] + h['eye_shape_' + e][:1], pressure=0.5, seed=50 + i)
    sh.stroke(h['jaw'], pen='maru', pressure=0.5)
sh.png(os.path.join(HERE, 'turnaround.png'), sketch=True)
