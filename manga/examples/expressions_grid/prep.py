"""Read the reference into named parts, at the panel's own resolution.

The reference is a flat-colour sheet of nine faces in a 3 x 3 grid with thin dividers. Colour separates most
parts: hair (dark teal with lighter teal shine), the navy jacket, ink lines, lit skin, skin shadow, eye whites,
iris layers, mouth and tongue, blush. panel.py decides how each part is drawn; nothing here draws.

    python3 prep.py        # writes parts.npz and parts_preview.png
"""
import numpy as np
from PIL import Image
from scipy import ndimage as nd

X0, X1, Y0, Y1 = 114, 1822, 0, 1080       # the part of the reference inside the outer frame
S = 1.6                                   # page px per reference px
PW, PH = int(round((X1 - X0) * S)), int(round((Y1 - Y0) * S))
COLS = [114, 682.5, 1252.5, 1822]         # reference x of the vertical dividers (outer ends: the frame)
ROWS = [0, 363.5, 724.5, 1080]            # reference y of the horizontal dividers


def to_page(x, y):
    """Reference px -> panel px (inside the frame)."""
    return (x - X0) * S, (y - Y0) * S


def cell(i):
    """Panel-px box (x0, y0, x1, y1) of face i (0..8, row by row)."""
    c, r = i % 3, i // 3
    x0, y0 = to_page(COLS[c], ROWS[r]); x1, y1 = to_page(COLS[c + 1], ROWS[r + 1])
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def read(path='ref.png'):
    im = Image.open(path).convert('RGB').crop((X0, Y0, X1, Y1)).resize((PW, PH), Image.LANCZOS)
    a = np.asarray(im).astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(2), a.min(2)
    lum = 0.3 * r + 0.59 * g + 0.11 * b
    P = {}
    P['bg'] = (mn > 212) & (b - r > 4) & (b - g > 3)
    P['hair'] = (g > r + 8) & (np.abs(b - g) < 9) & (mx < 160)                # teal: hair, its shine, brows
    P['shine'] = P['hair'] & (g > 100)
    P['jacket'] = (b > g + 4) & (b > r + 8) & (mx < 110) & ~P['hair']          # navy collar
    P['zip'] = (b > r + 12) & (mx > 130) & (mx < 215) & (mx - mn > 18) & ~P['bg']
    P['black'] = (mx < 72) & ~P['hair'] & ~P['jacket']                         # ink lines, lash bands, pupils
    P['brown'] = (r > 45) & (r < 170) & (r - b > 25) & (r - g > 15) & ~P['black'] & ~P['hair']   # lines, iris top
    P['amber'] = (r > 175) & (r - b > 95) & (g > 130)
    P['pink'] = (r > 190) & (r - g > 55) & (g < 165) & ~P['amber']             # mouth inside, tongue
    P['skin'] = (r > 232) & (r - b > 40) & (g > 196)
    P['shadow'] = (r > 190) & (r - b > 45) & (g > 140) & (g <= 196) & ~P['amber'] & ~P['pink']
    P['blush'] = (r > 228) & (r - g > 52) & (b > 138) & (g < 200) & ~P['pink']   # cheeks: a flush or a band
    P['white'] = (mn > 205) & (b - r < 5) & ~P['bg'] & ~P['skin']              # eye whites, teeth, catch-lights
    k = int(round(5 * S))
    top = nd.grey_closing(lum, size=(k, k)) - lum
    P['line'] = (top > 26) & ~P['white'] & ~P['bg']
    P['line_soft'] = (top > 12) & ~P['white']
    P['strand'] = (top > 8) & (mx < 165) & P['hair']
    return a / 255, P


def preview(P, out='parts_preview.png'):
    o = np.full(P['hair'].shape + (3,), 255, np.uint8)
    for kname, c in [('bg', (235, 235, 255)), ('skin', (255, 235, 215)), ('shadow', (200, 140, 90)),
                     ('hair', (0, 110, 110)), ('shine', (120, 220, 220)), ('jacket', (40, 40, 150)),
                     ('zip', (255, 0, 255)), ('white', (190, 190, 190)), ('amber', (255, 170, 0)),
                     ('pink', (255, 0, 150)), ('blush', (255, 150, 180)), ('brown', (130, 60, 0)), ('line', (0, 200, 0)), ('black', (0, 0, 0))]:
        o[P[kname]] = c
    Image.fromarray(o).save(out)


if __name__ == '__main__':
    rgb, P = read()
    np.savez_compressed('parts.npz', **P)
    preview(P)
    print(PW, PH, {k: int(v.sum()) for k, v in P.items()})
