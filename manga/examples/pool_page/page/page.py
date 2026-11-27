"""A manga page drawn from the pool reference: an introduction page for the character.

Decided before drawing:
  * B4 original, 257 x 364 mm at 8 px/mm. Left column: two small panels; right: one tall panel, the page's anchor.
  * Panel 1 (top left): where we are. Palm, umbrella and loungers by the water, all in light maru line,
    sky in a soft gradient tone, water in one light tone with ripples scraped out. Caption box.
  * Panel 2 (bottom left): her face up close, the emotional beat. Background dropped for a flash (uni-flash),
    the face drawn in full: hand-drawn eyes, black hair with gloss, blush hatching. Speech balloon.
  * Panel 3 (right, tall): the full figure, waving. G-pen contour, maru inner lines, two-tone cel shading per
    material (skin, swimsuit, scales), black hair. The pool scene behind her stays light and sparse.
"""
import sys, time, numpy as np
_here = __import__('os').path.dirname(__file__); sys.path.insert(0, __import__('os').path.join(_here, '..', '..', '..', '..')); sys.path.insert(0, _here)
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
from manga import Sheet, tone, trace, prep, effects, geom
import figure

W, H, PPM = 2056, 2912, 8.0
t0 = time.time()
sh = Sheet(W, H, px_per_mm=PPM, journal=sys.argv[1] if len(sys.argv) > 1 else 'page.jsonl', seed=3)
rows = effects.layout(W, H, [(1.0, [0.78, 1.22])], margin_mm=11, gutter_mm=(4, 6), px_per_mm=PPM)
L, R = rows[0]
lx0, ly0 = L[0]; lx1, ly1 = L[2]
rx0, ry0 = R[0]; rx1, ry1 = R[2]
gy = 6 * PPM
hh = (ly1 - ly0 - 2 * gy)
y1a = ly0 + hh * 0.36; y2a = y1a + gy; y2b = y2a + hh * 0.25; y3a = y2b + gy
P1 = [(lx0, ly0), (lx1, ly0), (lx1, y1a), (lx0, y1a)]
P4 = [(lx0, y2a), (lx1, y2a - 40), (lx1, y2b - 40), (lx0, y2b)]        # the wave: a slanted strip
P2 = [(lx0, y3a), (lx1, y3a - 40), (lx1, ry1), (lx0, ry1)]
P3 = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]


def ibox(p):
    xs, ys = [q[0] for q in p], [q[1] for q in p]
    return int(min(xs)), int(min(ys)), int(np.ceil(max(xs))), int(np.ceil(max(ys)))


def pmask(p):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(q) for q in p], fill=1)
    return np.asarray(im, bool)


def place(a, box):
    X0, Y0, X1, Y1 = box
    out = np.zeros((H, W), a.dtype); out[Y0:Y1, X0:X1] = a; return out


def background(box, crop, light=0.55, min_len=18, sky_to=None, water=None, drop=None):
    """The scene, drawn lightly: pale area tones, thin maru lines with the scraps left out."""
    rgb, mat, to_page, s = figure.source(box, crop)
    d = 1 - rgb.mean(2)
    fig = nd.binary_dilation(mat > 0, iterations=int(2 * s))
    lines = prep.xdog_lines(rgb, sigma=0.9 * s, p=24, eps=-0.015, phi=80, tidy=int(12 * s * s)) > 0.5
    lines = nd.binary_closing(lines, iterations=max(1, int(s / 1.2)))
    lines &= ~fig
    # specks and crumbs of line (float texture, water glints) print as dirt: drop pieces that are small in
    # both size and extent; long thin pieces (tile joints) are kept, and their gaps bridged along the line
    lab_b, nb = nd.label(lines, structure=np.ones((3, 3)))
    if nb:
        ext = np.array([max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in nd.find_objects(lab_b)])
        crumb = np.zeros(nb + 1, bool); crumb[1:] = ext < 9 * s
        # blots: short, thick pieces (the float's print, glints) would fill as black diamonds
        thick = nd.maximum(nd.distance_transform_edt(lines), lab_b, np.arange(1, nb + 1))
        crumb[1:] |= (thick > 2.2 * s) & (ext < 30 * s)
        lines &= ~crumb[lab_b]
    if water is not None:                        # bridge the floor tiles' joints (below the pool) only
        floor = (np.arange(lines.shape[0])[:, None] > to_page(0, water[1])[1] - box[1]) & np.ones_like(lines)
        for k_ in (np.ones((1, int(5 * s)), bool), np.ones((int(5 * s), 1), bool)):
            lines |= nd.binary_closing(lines, structure=k_) & ~fig & floor
    if drop is not None:                         # parts inked by hand elsewhere
        lines &= ~drop(to_page, s, box)
    lv = np.minimum(trace.area_tones(d, nd.binary_dilation(lines, iterations=1)), 0.75) * light
    lv[fig] = 0
    if sky_to is not None:                       # sky: a soft gradient tone, darker at the top
        yy = np.arange(lv.shape[0])[:, None] + np.zeros((1, lv.shape[1]))
        ys = np.array([to_page(0, 0)[1], to_page(0, sky_to)[1]]) - box[1]
        sky = (yy < ys[1]) & (nd.gaussian_filter(d, 2) < 0.35) & ~fig
        lv = np.where(sky, np.clip(0.30 - 0.26 * (yy - ys[0]) / max(ys[1] - ys[0], 1), 0.04, 0.30), lv)
    if water is not None:                        # the pool: one light tone, ripples scraped out below
        wy0, wy1 = to_page(0, water[0])[1] - box[1], to_page(0, water[1])[1] - box[1]
        yy = np.arange(lv.shape[0])[:, None] + np.zeros((1, lv.shape[1]))
        wat = (yy > wy0) & (yy < wy1) & ~fig
        lv = np.where(wat, 0.30, lv)
    tone(sh, place(lv > 0, box), level=place(lv.astype(np.float32), box), lpi=figure.LPI)
    if water is not None:
        gw = nd.gaussian_filter(rgb.mean(2), 1.2 * s)
        rip = wat & (gw - nd.gaussian_filter(gw, 8 * s) > 0.06)
        sh.fill(place(nd.binary_opening(rip, iterations=1), box), mode='white')
    n, _ = trace.ink_line_art(sh, place(lines, box), pens=('maru', 'maru'), weight=0.9, min_len_px=min_len * s,
                              min_px=1.0, seed=box[0] + box[1])
    return n, fig


def balloon(center, rx, ry, text, tail, size_mm=5.0):
    cx, cy = center
    t = np.linspace(0, 2 * np.pi, 120)
    # tail: a narrow wedge from the balloon's edge toward the speaker
    ang = np.arctan2(tail[1] - cy, tail[0] - cx)
    a0, a1 = ang - 0.16, ang + 0.16
    e0 = (cx + rx * np.cos(a0), cy + ry * np.sin(a0)); e1 = (cx + rx * np.cos(a1), cy + ry * np.sin(a1))
    shape = [(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t if not (a0 < a < a1 or a0 < a - 2 * np.pi < a1)]
    sh.fill([(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in t], mode='white')
    sh.fill([e0, tuple(tail), e1], mode='white')
    ell = np.array([(cx + rx * np.cos(a), cy + ry * np.sin(a)) for a in np.linspace(a1, a0 + 2 * np.pi, 140)])
    sh.stroke(np.vstack([[tail], ell, [tail]]), pen='g_pen', pressure=0.35, taper=(0, 0), smooth=False)
    f = effects._font(size_mm * PPM, face='dialogue')
    lines = text.split('\n')
    tmp = ImageDraw.Draw(Image.new('L', (1, 1)))
    bbs = [tmp.textbbox((0, 0), ln, font=f) for ln in lines]
    lh = size_mm * PPM * 1.2
    y = cy - lh * len(lines) / 2
    for ln, bb in zip(lines, bbs):
        effects.caption(sh, (cx - (bb[2] - bb[0]) / 2 - 0.5, y - bb[1] * 0.5), ln, size_mm=size_mm, pad_mm=0, box=False,
                        face='dialogue')
        y += lh


strokes = 0
# ---- Panel 1: the place
b1 = ibox(P1)
with sh.clip(pmask(P1)):
    import panel1
    s1 = (b1[3] - b1[1]) / (1085 - 650)                    # hand-drawn: frame ref y 650-1085, x centred on the pole
    tp1 = lambda rx, ry: (b1[0] + (b1[2] - b1[0]) / 2 + (rx - 112) * s1, b1[1] + (ry - 650) * s1)
    panel1.draw_panel1(sh, b1, tp1, s1, W, H)
effects.border(sh, P1, 0.8)
effects.caption(sh, (b1[0] + 3 * PPM, b1[1] + 3 * PPM), 'RESORT POOL.\n2 P.M.', size_mm=3.8, face='caption')

# ---- Panel 3: the full figure
b3 = ibox(P3)
with sh.clip(pmask(P3)):
    import umbrella3
    n, fig3 = background(b3, (130, 0, 1035, 1920), light=0.5, min_len=34, sky_to=880, water=(1090, 1530),
                         drop=umbrella3.drawn_zone_and_sky)
    strokes += n
    import umbrella3
    _, _, tp3, s3 = figure.source(b3, (130, 0, 1035, 1920))
    # the shadowed corner inside the tail's curl (a potted plant in deep shade): one deliberate dark tone with
    # the lit leaves left as clean light shapes, instead of the scattered black blots the tracing made
    _rgbp, _matp = figure.source(b3, (130, 0, 1035, 1920))[:2]
    _yy3, _xx3 = np.mgrid[:_matp.shape[0], :_matp.shape[1]]
    _rx = (_xx3 + b3[0] - tp3(0, 0)[0]) / s3; _ry = (_yy3 + b3[1] - tp3(0, 0)[1]) / s3
    _pocket = (_matp == 0) & (_rx > 925) & (_ry > 860) & (_ry < 1070)
    _pocket = nd.binary_opening(_pocket, iterations=2)
    _lp, _np_ = nd.label(_pocket)                       # only the enclosed curl, not the strip above the tail
    if _np_:
        _cy = nd.mean(_ry, _lp, np.arange(1, _np_ + 1))
        _sz = nd.sum(_pocket, _lp, np.arange(1, _np_ + 1))
        _keep = np.zeros(_np_ + 1, bool); _keep[1:] = (_cy > 940) & (_sz > 400 * s3 * s3)
        _pocket = _keep[_lp]
    if _pocket.any():
        _g = nd.gaussian_filter(_rgbp.mean(2), 1.2 * s3)
        _leaf = _pocket & (_g > 0.55)
        _leaf = nd.binary_opening(_leaf, iterations=int(1.5 * s3))
        sh.fill(place(_pocket, b3), mode='white')
        tone(sh, place(_pocket & ~_leaf, b3), level=0.72, lpi=figure.LPI)
        tone(sh, place(_leaf, b3), level=0.15, lpi=figure.LPI)
        prep.ink_outline(sh, place(_leaf, b3), pen='maru', pressure=(0.3, 0.5), seed=77)
    umbrella3.draw_umbrella(sh, tp3, s3)
    import bushes3
    _rgb3, _mat3 = figure.source(b3, (130, 0, 1035, 1920))[:2]
    bushes3.draw_bushes(sh, _rgb3, _mat3, tp3, s3, b3, figure.LPI)
    sh.fill(place(figure.source(b3, (130, 0, 1035, 1920))[1] > 0, b3), mode='white')
    r3 = figure.draw_figure(sh, b3, (130, 0, 1035, 1920))
    strokes += r3['strokes']
    # the gap between her thighs shows the deck behind her: carry the gutter grate's slanted bars through it,
    # same pitch and slant as the bars either side, so the background reads as continuous
    _gap3 = np.zeros((H, W), bool)
    _gap3[b3[1]:b3[3], b3[0]:b3[2]] = r3['mat'] == 0
    _win = np.zeros((H, W), bool); _win[2290:2420, 1370:1445] = True
    _gap3 = nd.binary_erosion(_gap3 & _win, iterations=2)
    _up = _gap3.copy(); _up[2300:] = False
    _up |= nd.binary_erosion(np.pad(r3['mat'] == 0, ((b3[1], H - b3[3]), (b3[0], W - b3[2]))) &
                             (np.arange(H)[:, None] >= 2150) & (np.arange(H)[:, None] < 2300) &
                             (np.arange(W)[None, :] >= 1370) & (np.arange(W)[None, :] < 1445), iterations=2)
    # above the grate: the deck's light tone, evenly, instead of the reference's stray grey patch and specks
    sh.fill(_up, mode='white')
    tone(sh, _up, level=0.15, lpi=figure.LPI)
    with sh.clip(_gap3):
        for _x in (1381.2, 1402.0, 1422.8, 1443.6):
            sh.fill([(_x, 2313), (_x + 8, 2313), (_x - 19, 2400), (_x - 27, 2400)])
effects.border(sh, P3, 0.8)

# ---- Panel 2: the face, with a flash behind it
b2 = ibox(P2)
CROP2 = (300, 150, 720, 760)
with sh.clip(pmask(P2)):
    rgb2, mat2, tp2, s2 = figure.source(b2, CROP2)
    c = tp2(510, 400)
    effects.flash(sh, c, r_in=0.42 * (b2[2] - b2[0]), r_out=1.2 * (b2[3] - b2[1]), count=150, seed=4)
    sh.fill(place(nd.binary_dilation(mat2 > 0, iterations=3), b2), mode='white')
    r2 = figure.draw_figure(sh, b2, CROP2, dict(contour_mm=0.55, content=True))   # eyes closed, content
    strokes += r2['strokes']
    balloon((b2[0] + 0.30 * (b2[2] - b2[0]), b2[1] + 0.12 * (b2[3] - b2[1])), 0.22 * (b2[2] - b2[0]),
            0.075 * (b2[3] - b2[1]), 'OVER\nHERE!!', tail=tp2(470, 330), size_mm=4.6)
effects.border(sh, P2, 0.8)

# ---- Panel 4 (middle left): the wave, hand up close, with motion marks
b4 = ibox(P4)
CROP4 = (680, 250, 1000, 560)
with sh.clip(pmask(P4)):
    rgb4, mat4, tp4, s4 = figure.source(b4, CROP4)
    effects.speed_lines(sh, b4, angle=-12, count=70, length=(0.3, 0.8), seed=6)
    _cols4 = figure.SRCBOX[0][0] + np.arange(mat4.shape[1]) / s4
    _rows4 = figure.SRCBOX[0][1] + np.arange(mat4.shape[0]) / s4
    _keep4 = np.zeros(mat4.shape, bool)
    for _t in np.linspace(0, 1, 40):
        _keep4 |= (_cols4[None, :] - (716 - 23 * _t)) ** 2 + (_rows4[:, None] - (463 - 4 * _t)) ** 2 <= (14.5 - 3.5 * _t) ** 2
    mat4 = np.where(_keep4, 1, mat4)
    mat4 = np.where((_cols4 < 712)[None, :] & ~_keep4, 0, mat4)   # the thumb reaches past the cut
    _lab4, _n4 = nd.label(mat4 > 0)
    if _n4 > 1:
        _sz4 = nd.sum(mat4 > 0, _lab4, np.arange(1, _n4 + 1))
        mat4 = np.where(_lab4 == 1 + int(np.argmax(_sz4)), mat4, 0)
    sh.fill(place(nd.binary_dilation(mat4 > 0, iterations=3), b4), mode='white')
    # the sliver of her face at the panel's left edge is cut: it read as a broken shape, and the hand is the beat
    r4 = figure.draw_figure(sh, b4, CROP4, dict(contour_mm=0.55, blush=False, inner_pen='saji', inner_w=1.5, inner_min=14,
                                              even=0.85, cut_left=712, cut_keep=((716, 463, 14.5), (693, 459, 11)), blk_min=60))
    strokes += r4['strokes']
    # two specks where the contour turns sharply at the finger webs: scrape them and lay the skin tone back
    _yy, _xx = np.mgrid[:H, :W]
    _sp = np.zeros((H, W), bool)
    for (_cx, _cy) in [(330, 1405), (406, 1401)]:
        _sp |= (_xx - _cx) ** 2 + (_yy - _cy) ** 2 <= 16
    sh.fill(_sp, mode='white')
    tone(sh, _sp, level=0.40, lpi=figure.LPI)
    # wave marks: short arcs either side of the fingers, the shorthand for a hand in motion
    for (cx, cy, sgn) in [(735, 330, -1), (925, 345, 1)]:
        for k, rr in enumerate((26, 40, 54)):
            a = np.linspace(-0.55, 0.55, 12)
            pts = [tp4(cx + sgn * rr * np.cos(t) - sgn * 0, cy + rr * np.sin(t)) for t in a]
            sh.stroke(pts, pen='g_pen', pressure=0.5, taper=(0.6, 0.9), seed=40 + k)
effects.border(sh, P4, 0.8)

sh.png('page.png'); sh.close()
print('strokes', strokes + sh.stats.get('strokes', 0) * 0, 'sheet strokes', sh.stats['strokes'], round(time.time() - t0), 's')
