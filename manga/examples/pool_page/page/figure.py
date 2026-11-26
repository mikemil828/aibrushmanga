"""Draw the girl from the reference into a page region, material by material.

Order (as an inker works): tone per material with cel shadows -> white edge round lines -> inner lines (maru)
-> contour (G-pen, heavier) -> hair blacks, gloss and strands -> eye and mouth blacks -> blush hatching.
"""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', '..', '..'))
from PIL import Image
from scipy import ndimage as nd
from manga import tone, trace, prep, geom
import eyes_hand
EYEZONE = eyes_hand.eye_region(MAT.shape[1] if False else 1080, 1920, pad=7)

REF = Image.open('../ref.png').convert('RGB')
MAT = np.load('./mat.npy')
SKIN, SUIT, SCALES, HAIR, PAPER, IRIS = 1, 2, 3, 4, 5, 6
BLUSH = np.load('./blush.npy')
LPI = 60
SRCBOX = [None]


def source(box, crop):
    """Reference crop (x0, y0, x1, y1 in ref px) cover-fitted to a page box. Returns rgb, mat, to_page, scale."""
    X0, Y0, X1, Y1 = box
    pw, ph = X1 - X0, Y1 - Y0
    cx0, cy0, cx1, cy1 = crop
    s = max(pw / (cx1 - cx0), ph / (cy1 - cy0))
    RW, RH = REF.size
    s = max(s, pw / RW, ph / RH)                 # never ask for more than the reference holds
    vw, vh = pw / s, ph / s
    vx0, vy0 = cx0 + ((cx1 - cx0) - vw) / 2, cy0 + ((cy1 - cy0) - vh) / 2
    vx0, vy0 = max(0.0, min(vx0, RW - vw)), max(0.0, min(vy0, RH - vh))
    vw, vh = min(vw, RW - vx0), min(vh, RH - vy0)
    SRCBOX[0] = (vx0, vy0, vx0 + vw, vy0 + vh)
    rgb = np.asarray(REF.resize((pw, ph), Image.BICUBIC, box=(vx0, vy0, vx0 + vw, vy0 + vh)), np.float32) / 255
    mat = np.asarray(Image.fromarray(MAT).resize((pw, ph), Image.NEAREST, box=(vx0, vy0, vx0 + vw, vy0 + vh)))
    to_page = lambda rx, ry: (X0 + (rx - vx0) * s, Y0 + (ry - vy0) * s)
    return np.clip(rgb, 0, 1), mat, to_page, s


def smooth(m, sigma):
    return nd.gaussian_filter(m.astype(np.float32), sigma) > 0.5


def place(sh, a, box):
    X0, Y0, X1, Y1 = box
    out = np.zeros((sh.H0, sh.W0), a.dtype) if hasattr(sh, 'H0') else np.zeros((sh.H // sh.k, sh.W // sh.k), a.dtype)
    out[Y0:Y1, X0:X1] = a
    return out


def cel(d, m, base, shadow, delta, sigma, min_area):
    """Two flat tones in a material: base, and a shadow shape where the reference is `delta` darker than its median."""
    if not m.any():
        return None, None
    med = np.median(d[m])
    sh_m = smooth((d > med + delta) & m, sigma) & m
    lab, n = nd.label(sh_m)
    if n:
        sz = nd.sum(sh_m, lab, range(1, n + 1))
        sh_m = np.isin(lab, 1 + np.nonzero(sz >= min_area)[0])
    lv = np.where(m, base, 0.0)
    lv = np.where(sh_m, shadow, lv)
    return lv.astype(np.float32), sh_m


def draw_figure(sh, box, crop, opt=None):
    o = dict(skin=(0.30, 0.52, 0.06), suit=(0.0, 0.14, 0.09), scales=(0.10, 0.28, 0.08), contour_w=2.2, inner_w=1.0,
             inner_min=10, white_edge=1, blush=True, strands=0.6)
    o.update(opt or {})
    rgb, mat, to_page, s = source(box, crop)
    X0, Y0, X1, Y1 = box
    d = 1 - rgb.mean(2)
    ds = nd.gaussian_filter(d, 1.2 * s)
    fig = mat > 0
    eyez = np.asarray(Image.fromarray(EYEZONE).resize((X1 - X0, Y1 - Y0), Image.NEAREST, box=SRCBOX[0]))
    hair = mat == HAIR
    # lines of the figure, at the panel's own scale
    lines = prep.xdog_lines(rgb, sigma=0.9 * s, p=24, eps=-0.015, phi=80, tidy=int(12 * s * s)) > 0.5
    lines &= nd.binary_dilation(fig, iterations=max(2, int(2 * s)))
    lines &= ~eyez
    hair = hair & ~eyez
    mat = np.where((mat == HAIR) & eyez, SKIN, mat)      # round the hand-drawn eyes: skin, not a hole
    edge = fig ^ nd.binary_erosion(fig, iterations=1)
    near_edge = nd.binary_dilation(edge, iterations=max(3, int(3 * s)))
    contour = lines & near_edge
    inner = lines & ~near_edge & ~hair
    # 1. tones, material by material
    lv = np.zeros(d.shape, np.float32)
    shadows = {}
    for key, k in (('skin', SKIN), ('suit', SUIT), ('scales', SCALES)):
        base, shad, delta = o[key]
        l, shm = cel(ds, mat == k, base, shad, delta, 2.0 * s, int(40 * s * s))
        if l is not None:
            lv = np.maximum(lv, l)
            shadows[key] = shm
    # irises: darker under the upper lid, lighter at the bottom, catch-lights left as paper
    iris = mat == IRIS
    if iris.any():
        lab, n = nd.label(iris)
        for i in range(1, n + 1):
            ys = np.nonzero(lab == i)[0]
            t = np.clip((np.arange(d.shape[0]) - ys.min()) / max(np.ptp(ys), 1), 0, 1)[:, None]
            lv = np.where(lab == i, 0.85 - 0.45 * t, lv)
        lv = np.where(iris & (ds < 0.30), 0.0, lv)
    tone(sh, place(sh, lv > 0, box), level=place(sh, lv, box), lpi=LPI)
    # 2. white edge (fuchidori) so the lines read through the dots
    if o['white_edge']:
        sh.fill(place(sh, nd.binary_dilation(contour, iterations=o['white_edge']) & fig & ~hair, box), mode='white')
    # 3. inner lines: maru, light; skip texture scraps on the skin
    n_in, _ = trace.ink_line_art(sh, place(sh, inner, box), pens=('maru', 'maru'), weight=o['inner_w'],
                                 min_len_px=o['inner_min'] * s, seed=11, min_px=1.0)
    # 4. hair: black, gloss scraped back, a few strands in white
    if hair.any():
        sh.fill(place(sh, hair, box))
        med = np.median(ds[hair])
        gloss = nd.binary_opening(hair & (ds < med - 0.22), iterations=max(1, int(s)))
        # keep only streak-like shine (long and thin, along the strands); round blobs read as holes
        lab, n = nd.label(gloss)
        keepg = np.zeros(n + 1, bool)
        for i, sl in enumerate(nd.find_objects(lab), 1):
            ys, xs = np.nonzero(lab[sl] == i)
            if len(ys) < 6:
                continue
            ev = np.linalg.eigvalsh(np.cov(np.vstack([xs, ys])) + 1e-6 * np.eye(2))
            keepg[i] = ev[1] / max(ev[0], 1e-6) > 9.0
        gloss = keepg[lab] & False             # the shine is drawn by hand below instead
        ring = geom.catmull(np.array([(392, 285), (425, 243), (475, 222), (530, 222), (582, 242), (612, 280)], float), per=10)
        rng2 = np.random.default_rng(21)
        for i, (qx, qy) in enumerate(ring[2:-2:2]):
            vx, vy = qx - 500, qy - 380
            nv = np.hypot(vx, vy); vx, vy = vx / nv, vy / nv
            Lr = rng2.uniform(9, 20) * (0.6 if i % 3 == 1 else 1.0)
            off = rng2.uniform(-3, 3)
            a = to_page(qx + vx * (off + Lr * 0.5), qy + vy * (off + Lr * 0.5))
            b = to_page(qx - vx * (Lr * 0.5 - off), qy - vy * (Lr * 0.5 - off))
            if not (X0 <= a[0] <= X1 and Y0 <= a[1] <= Y1):
                continue
            sh.stroke([a, b], pen='white', pressure=0.18 + 0.12 * rng2.random(), mode='white',
                      taper=(0.5 * s, 0.9 * s), smooth=False, seed=int(rng2.integers(1 << 30)))
        strands = lines & nd.binary_erosion(hair, iterations=int(3 * s)) & ~gloss
        rng = np.random.default_rng(5)
        n_st = 0
        for pts, a, b in trace.trace(trace.thin(strands)):
            if len(pts) < 2 or geom.arclen(pts)[-1] < 25 * s or rng.random() > o['strands'] \
                    or np.hypot(*(pts[0] - pts[-1])) < 0.3 * geom.arclen(pts)[-1]:
                continue
            Q = geom.simplify(pts, 0.8) + [X0, Y0]
            sh.stroke(Q, pen='white', pressure=0.06 * s, mode='white', taper=(0.8 * s, 1.6 * s),
                      seed=int(rng.integers(1 << 30)))
            n_st += 1
    # 5. contour: G-pen, the heaviest line on the figure
    n_c, _ = trace.ink_line_art(sh, place(sh, contour, box), pens=('g_pen', 'g_pen'), weight=o['contour_w'],
                                min_len_px=6 * s, seed=12, min_px=1.6 * s)
    # 6. blacks inside the figure: pupils, mouth, deepest folds
    blk = smooth(((ds > 0.80) | (iris & (ds > 0.68))) & fig & ~hair & ~eyez, 1.0)
    lab, n = nd.label(blk)
    if n:
        sz = nd.sum(blk, lab, range(1, n + 1))
        blk = np.isin(lab, 1 + np.nonzero(sz >= 20 * s * s)[0])
        sh.fill(place(sh, blk, box))
    eyes_hand.draw_eyes(sh, to_page, s)
    # 7. blush: a few short hatched strokes on each cheek, the manga shorthand for a flush
    if o['blush']:
        rng = np.random.default_rng(9)
        for bx0, by0, bx1, by1 in BLUSH:
            p0, p1 = to_page(bx0, by0), to_page(bx1, by1)
            if p1[0] < X0 or p0[0] > X1 or p1[1] < Y0 or p0[1] > Y1:
                continue
            w, h = p1[0] - p0[0], p1[1] - p0[1]
            k = 8
            for i in range(k):
                cx = p0[0] + w * (i + 0.5) / k + rng.normal(0, w * 0.01)
                cy = p0[1] + h * 0.5 + h * 0.25 * np.sin(np.pi * (i + 0.5) / k - np.pi / 2) * 0.3
                L = h * (0.55 + 0.35 * np.sin(np.pi * (i + 0.5) / k))
                dx, dy = 0.45 * L * 0.45, 0.45 * L
                sh.stroke([(cx - dx, cy + dy), (cx + dx, cy - dy)], pen='maru', pressure=0.35,
                          taper=(0.5, 0.5), smooth=False, seed=int(rng.integers(1 << 30)))
    return dict(lines=lines, fig=fig, to_page=to_page, s=s, strokes=n_in + n_c, mat=mat, rgb=rgb)
