"""Read the reference into named parts, at the panel's own resolution.

The reference is a flat-colour sheet, so colour separates the parts cleanly: hair, ink lines, the white of the
eyes, iris layers, mouth inside and tongue, skin shadow, blush, the gloom band of the frightened face. panel.py
decides how each part is drawn; nothing here draws.

    python3 prep.py        # writes parts.npz and parts_preview.png
"""
import numpy as np
from PIL import Image
from scipy import ndimage as nd

PW, PH = 2736, 1539                       # the panel's inside, page px (reference x 1.425)
S = PW / 1920
COLS, ROWS = 4, 2                         # eight faces
CELL_W, CELL_H = PW / COLS, PH / ROWS


def cell(i):
    """Page-px box (x0, y0, x1, y1) of face i (0..7, row by row) inside the panel."""
    c, r = i % COLS, i // COLS
    return int(c * CELL_W), int(r * CELL_H), int((c + 1) * CELL_W), int((r + 1) * CELL_H)


def read(path='ref.png'):
    im = Image.open(path).convert('RGB').resize((PW, PH), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    P = {}
    P['hair'] = (g > r + 6) & (b > r + 4) & (mx < 130)                  # dark teal: hair and the thick brows
    P['black'] = (mx < 75) & ~P['hair']                                  # lash bands, pupils, drawn lines
    P['amber'] = (r > 150) & (r - b > 90) & (g > 90) & (g - b > 40)        # iris, not the tongue
    P['yellow'] = P['amber'] & (g > 185)                                 # the bright crescent low in the iris
    P['maroon'] = (r > 50) & (r < 150) & (r - g > 25) & (mx < 160) & ~P['hair'] & ~P['black']
    P['pink'] = (r > 205) & (r - g > 70) & (g < 180) & ~P['amber']       # tongue, lips, iris mid tone, strong blush
    P['white'] = (mn > 226) & (mx - mn < 16) & (b >= r - 6)              # shirt, eye whites, teeth, highlights
    P['shadow'] = (r > 185) & (r <= 240) & (r - g > 38) & (r - g <= 72) & (g - b > 8) & ~P['amber']
    P['gloom'] = (r > 190) & (r < 230) & (np.abs(g - b) < 10) & (r - g > 16) & (r - g < 42)
    P['blush'] = (r > 243) & (r - g > 40) & (r - g < 70) & (b > 150)
    P['skin'] = (r > 238) & (g > 192) & (g < 234) & (b > 165) & (b < 218) & (r - b > 28)   # lit skin, blush included
    # thin drawn lines of any colour: darker than what lies either side (creases, nose, mouth, folds, strands)
    k = int(round(7 * S))
    top = nd.grey_closing(lum, size=(k, k)) - lum
    P['line'] = (top > 26) & ~P['white']
    P['line_soft'] = (top > 13) & ~P['white']                            # the faintest drawn lines (mouth, nose)
    P['strand'] = (top > 8) & (mx < 135)                                 # strand lines inside the hair
    P['fold'] = (top > 5) & (top <= 26) & (mn > 200) & (b >= r - 4)      # soft grey folds on the white shirt
    return a / 255, P


def preview(P, out='parts_preview.png'):
    o = np.full(P['hair'].shape + (3,), 255, np.uint8)
    for kname, c in [('hair', (0, 90, 90)), ('white', (205, 205, 255)), ('shadow', (150, 100, 60)),
                     ('gloom', (150, 80, 200)), ('blush', (255, 170, 170)), ('amber', (255, 170, 0)),
                     ('yellow', (255, 240, 0)), ('maroon', (120, 0, 0)), ('pink', (255, 0, 150)),
                     ('fold', (120, 160, 255)), ('strand', (0, 40, 40)), ('line', (90, 90, 90)), ('black', (0, 0, 0))]:
        o[P[kname]] = c
    Image.fromarray(o).save(out)


if __name__ == '__main__':
    rgb, P = read()
    np.savez_compressed('parts.npz', **P)
    preview(P)
    print({k: int(v.sum()) for k, v in P.items()})
