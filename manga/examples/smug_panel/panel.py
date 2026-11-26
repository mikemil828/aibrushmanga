"""Mio's smug 'heh.': one panel, 200 x 150 mm. Run: python3 panel.py panel.jsonl"""
import sys, os
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..', '..')); sys.path.insert(0, _here)
import numpy as np
from manga import Sheet, tone, effects
import face, mio

W, H, PPM = 1600, 1200, 8.0
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else 'panel.jsonl', seed=4)
m = 4 * PPM
box = (m, m, W - m, H - m)
frame = [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
with sh.clip(np.pad(np.ones((int(box[3] - box[1]), int(box[2] - box[0])), bool),
                    ((int(box[1]), H - int(box[3])), (int(box[0]), W - int(box[2]))))):
    # background: a soft gradient tone, darker to the corners, and focus lines on her face
    tone(sh, None, level=(0.30, 0.06, (0, 0), (0, H)), lpi=60)
    effects.focus_lines(sh, (770, 640), (560, 520), box, count=160, pressure=(0.35, 0.8), seed=3)
    mio.body(sh, W, H)
    face.fill_face(sh)
    mio.hair(sh)
    face.features(sh)
    effects.caption(sh, (1110, 760), 'heh.', size_mm=6.5, pad_mm=0, box=False, face='dialogue')
effects.border(sh, frame, 0.8)
print('strokes', sh.stats['strokes'])
sh.png(os.path.join(_here, 'panel.png')); sh.close()
