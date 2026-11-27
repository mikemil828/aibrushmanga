"""Review sheets for the nine-expression panel, so each round starts with a look instead of a new throwaway script.

    python3 review.py                          # eyes.png, mouths.png: one part of every face, in a grid
    python3 review.py --old prev.png           # also before-after.png against an earlier panel.png
    DUMP=m.npz python3 panel.py && python3 review.py --face 5 --part eyes --masks m.npz   # layers-5-eyes.png
    python3 review.py --pin 79 43              # pin-79-43.png: reference | drawing | classes round a comment pin
    python3 review.py --each iris --masks m.npz   # each-iris.png: every detected iris beside the reference
                                               # (iris, mouth, teeth: from DUMP; brows: no build needed)
    python3 review.py --each brows             # each-brows.png, detection only (no build): about 8 s

A part is a box in cell fractions (y0, y1, x0, x1), the same as panel.py's zones.
"""
import argparse, sys, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import tools
import prep as ref

M = 88
PARTS = {"eyes": (0.18, 0.55, 0.18, 0.82), "brows": (0.15, 0.40, 0.18, 0.82), "mouth": (0.52, 0.85, 0.28, 0.72),
         'face': (0.0, 1.0, 0.1, 0.9)}
CLASSES = [('bg', (235, 235, 255)), ('skin', (255, 240, 225)), ('shadow', (255, 220, 150)), ('white', (210, 210, 210)),
           ('pink', (255, 120, 160)), ('blush', (255, 190, 200)), ('amber', (255, 170, 0)), ('brown', (150, 60, 0)),
           ('hair', (60, 90, 90)), ('shine', (120, 200, 200)), ('jacket', (40, 40, 140)), ('black', (0, 0, 0)),
           ('line', (0, 160, 0))]
MASK_COLOURS = [(0, 0, 255), (255, 0, 0), (255, 255, 0), (200, 0, 200), (0, 255, 0)]


def box(i, part, page=True):
    x0, y0, x1, y1 = ref.cell(i)
    fy0, fy1, fx0, fx1 = PARTS[part]
    b = (x0 + fx0 * (x1 - x0), y0 + fy0 * (y1 - y0), x0 + fx1 * (x1 - x0), y0 + fy1 * (y1 - y0))
    return tuple(v + M for v in b) if page else b


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--old'); ap.add_argument('--face', type=int); ap.add_argument('--part', default='eyes')
    ap.add_argument('--masks', help='npz written by DUMP=... python3 panel.py')
    ap.add_argument('--pin', type=float, nargs=2, metavar=('X%', 'Y%'), help='a comment pin on panel.png')
    ap.add_argument('--each', help='a mask name: every piece of it beside the reference (from --masks, or from '
                                   'panel.py run up to its drawing)')
    a = ap.parse_args()
    tools.strip('panel.png', [box(i, 'eyes') for i in range(9)], cols=3, out='eyes.png')
    tools.strip('panel.png', [box(i, 'mouth') for i in range(9)], cols=3, width=300, out='mouths.png')
    if a.old:
        tools.before_after(a.old, 'panel.png', [box(i, 'face') for i in range(9)], width=700, out='before-after.png')
    if a.face is not None:
        P = np.load('parts.npz')
        masks = [(P[k], c) for k, c in CLASSES]
        if a.masks:
            D = np.load(a.masks)
            masks += [(D[k], MASK_COLOURS[j % len(MASK_COLOURS)]) for j, k in enumerate(D.files)
                      if D[k].dtype == bool and D[k].shape == P['skin'].shape]
            print('masks over the classes, in order:', [k for k in D.files])
        tools.layers('ref.png', 'panel.png', box(a.face, a.part, page=False), masks, offset=(M, M),
                     out=f'layers-{a.face}-{a.part}.png')
    if a.pin:
        W, H = Image.open('panel.png').size
        x, y = a.pin[0] / 100 * W - M, a.pin[1] / 100 * H - M
        tools.triptych('ref.png', 'panel.png', np.load('parts.npz'), [(x, y)], offset=(M, M),
                       out=f'pin-{a.pin[0]:g}-{a.pin[1]:g}.png')
        print('pin at reference px', int(x), int(y))
    if a.each:
        if a.masks and a.each in np.load(a.masks).files:
            m = np.load(a.masks)[a.each]
        else:                                        # the masks decided before any ink: no full build needed
            ns = tools.run_until('panel.py', 'with sh.clip(PANEL):')
            if a.each not in ns:                     # irises, mouths, teeth are found while drawing
                sys.exit(f'{a.each} is decided while drawing: DUMP=m.npz python3 panel.py, then --masks m.npz')
            m = ns[a.each]
        bx = tools.boxes_of(m, pad=14, min_px=150)
        tools.beside('ref.png', 'panel.png', bx, offset=(M, M), size=(ref.PW, ref.PH), out=f'each-{a.each}.png')
        print(len(bx), 'pieces of', a.each)
    print('wrote eyes.png, mouths.png' + (', before-after.png' if a.old else '') +
          (f', layers-{a.face}-{a.part}.png' if a.face is not None else ''))
