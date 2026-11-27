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
import hair as hair_mod
import contour as contour_mod
EYEZONE = eyes_hand.eye_region(MAT.shape[1] if False else 1080, 1920, pad=7)

REF = Image.open('../ref.png').convert('RGB')
MAT = np.load('./mat.npy')
SKIN, SUIT, SCALES, HAIR, PAPER, IRIS = 1, 2, 3, 4, 5, 6
BLUSH = np.load('./blush.npy')
LPI = 60
SRCBOX = [None]
LAST_BLEED = [0]


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
    if s > 1.2:
        # blown up from the reference, the material map comes out in stair steps and lumps (fingers turn
        # uneven): smooth every material's edge at the panel's own scale, then take the strongest at each pixel
        labels = np.unique(mat)
        sig = 0.9 * s
        score = np.stack([nd.gaussian_filter((mat == k).astype(np.float32), sig) for k in labels])
        mat = labels[np.argmax(score, 0)].astype(mat.dtype)
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
    o = dict(skin=(0.40, 0.58, 0.06), suit=(0.0, 0.14, 0.09), scales=(0.12, 0.12, 0.08), contour_w=2.2, inner_w=1.0,
             inner_min=10, white_edge=1, blush=True, strands=0.6, hair_density=0.10, contour='even', contour_mm=0.42, even=0.7, inner_pen='maru')
    o.update(opt or {})
    rgb, mat, to_page, s = source(box, crop)
    X0, Y0, X1, Y1 = box
    d = 1 - rgb.mean(2)
    ds = nd.gaussian_filter(d, 1.2 * s)
    # a panel can leave out part of the figure that only peeks in at its edge (ref x below cut_left)
    if o.get('cut_left'):
        vx0 = SRCBOX[0][0]
        cols = vx0 + np.arange(mat.shape[1]) / s
        cut = (cols < o['cut_left'])[None, :] & np.ones_like(mat, bool)
        keep = np.zeros_like(cut)
        if o.get('cut_keep'):
            # a finger reaching past the cut (e.g. the thumb): drawn as a tapered capsule from (x0,y0,r0) at its
            # base to a rounded tip (x1,y1,r1), ref px. The traced skin there merges with what it touches.
            (x0, y0, r0), (x1, y1, r1) = o['cut_keep']
            rows = SRCBOX[0][1] + np.arange(mat.shape[0]) / s
            for tt in np.linspace(0, 1, 40):
                cx, cy, rr = x0 + (x1 - x0) * tt, y0 + (y1 - y0) * tt, r0 + (r1 - r0) * tt
                keep |= ((cols[None, :] - cx) ** 2 + (rows[:, None] - cy) ** 2 <= rr * rr)
            keep &= cut
            mat = np.where(keep, 1, mat)
        mat = np.where(cut & ~keep, 0, mat)
        # keep only the main piece (the hand), and round the cut end so it reads as a fingertip, not a sawn edge
        lab_c, nc = nd.label(mat > 0)
        if nc > 1:
            szc = nd.sum(mat > 0, lab_c, np.arange(1, nc + 1))
            mat = np.where(lab_c == 1 + int(np.argmax(szc)), mat, 0)
        near_cut = (cols < o['cut_left'] + 14)[None, :] & np.ones_like(mat, bool)
        rounded = nd.gaussian_filter((mat > 0).astype(np.float32), 4 * s) > 0.5
        mat = np.where(near_cut & ~rounded & ~keep, 0, mat)
    # scales: clean shapes. The traced patches came out sawtoothed and full of small islands; round their
    # edges at the panel's scale, fill the pinholes and drop specks, so each patch reads as one smooth piece
    sc = mat == SCALES
    if sc.any() and o.get('smooth_scales', True):
        scs = nd.gaussian_filter(sc.astype(np.float32), 2.2 * s) > 0.5
        scs = nd.binary_fill_holes(scs) & (mat > 0)
        lab_s, ns = nd.label(scs)
        if ns:
            sz = nd.sum(scs, lab_s, np.arange(1, ns + 1))
            keep = np.zeros(ns + 1, bool); keep[1:] = sz >= 80 * s * s
            scs = keep[lab_s]
        # pixels that stop being scales take the skin's place; skin pixels inside the new shape become scales
        mat = np.where(sc & ~scs, SKIN, mat)
        mat = np.where(scs & (mat == SKIN), SCALES, mat)
    fig = mat > 0
    _zone = EYEZONE | eyes_hand.mouth_region(EYEZONE.shape[1], EYEZONE.shape[0]) if o.get('content') else EYEZONE
    eyez = np.asarray(Image.fromarray(_zone).resize((X1 - X0, Y1 - Y0), Image.NEAREST, box=SRCBOX[0]))
    if o.get('content'):
        # the content face: the open eyes and mouth become plain skin, shaded flat from the skin around them
        _fz = eyes_hand.face_redraw_region(EYEZONE.shape[1], EYEZONE.shape[0], pad=11)
        fz = np.asarray(Image.fromarray(_fz).resize((X1 - X0, Y1 - Y0), Image.NEAREST, box=SRCBOX[0]))
        eyez = eyez | (fz & (mat != HAIR))
        mat = np.where(fz & (mat != HAIR), SKIN, mat)
        fig = mat > 0
    hair = mat == HAIR
    # inside the hand-drawn zones (eyes, plain swimsuit) the reference's shading is not ours: take the shade
    # from just outside, so no cel-shadow edge traces the shapes we left out
    _ei = nd.distance_transform_edt(eyez, return_distances=False, return_indices=True)
    ds = ds[tuple(_ei)]
    if o.get('content'):
        _ring = nd.binary_dilation(fz, iterations=int(6 * s)) & ~fz & (mat == SKIN)
        ds = np.where(fz, np.median(ds[_ring]) if _ring.any() else ds, ds)
    # fabric shading reads as broad soft shapes, never as the outline of something printed on it
    ds = np.where(mat == SUIT, nd.gaussian_filter(ds, 7 * s), ds)
    # lines of the figure, at the panel's own scale
    lines = prep.xdog_lines(rgb, sigma=0.9 * s, p=24, eps=-0.015, phi=80, tidy=int(12 * s * s)) > 0.5
    lines &= nd.binary_dilation(fig, iterations=max(2, int(2 * s)))
    # background lines that run into the figure (umbrella ribs, pool edges) stop at its outline: a line piece
    # inside the figure is kept only if its traced line lies mostly inside the figure
    full_lines = prep.xdog_lines(rgb, sigma=0.9 * s, p=24, eps=-0.015, phi=80, tidy=int(12 * s * s)) > 0.5
    core = nd.binary_erosion(fig, iterations=max(2, int(2 * s)))
    lab, nl = nd.label(full_lines, structure=np.ones((3, 3)))
    if nl:
        # a figure's own outline lies on its edge, so "inside" means inside the figure or on its edge
        near = nd.binary_dilation(fig, iterations=max(3, int(3 * s)))
        inside = nd.mean(near, lab, np.arange(1, nl + 1))
        size = nd.sum(full_lines, lab, np.arange(1, nl + 1))
        out_px = size * (1 - inside)
        bad = (inside < 0.5) & (out_px > 30 * s)            # most of the line is out in the scene
        bled = np.concatenate([[False], bad])[lab]
        LAST_BLEED[0] = int((lines & bled & core).sum())
        lines &= ~bled | ~core
        lines &= ~(bled & nd.binary_dilation(core, iterations=1))
    lines &= ~eyez
    hair = hair & ~eyez
    mat = np.where((mat == HAIR) & eyez, SKIN, mat)      # round the hand-drawn eyes: skin, not a hole
    edge = fig ^ nd.binary_erosion(fig, iterations=1)
    near_edge = nd.binary_dilation(edge, iterations=max(3, int(3 * s)))
    contour = lines & near_edge
    inner = lines & ~near_edge & ~hair
    # lines on a boundary between materials (swimsuit edge, arm over body) are form lines, not detail
    mb = np.zeros_like(fig)
    for k in (SKIN, SUIT, SCALES):
        mk = nd.binary_opening(mat == k, iterations=max(1, int(s)))
        mb |= mk ^ nd.binary_erosion(mk, iterations=1)
    mb &= nd.binary_erosion(fig, iterations=max(3, int(3 * s)))
    matline = inner & nd.binary_dilation(mb, iterations=max(2, int(2.5 * s))) if o['contour'] != 'trace' else np.zeros_like(fig)
    inner = inner & ~matline
    # dots of line (a pore or a crease end in the reference) print as specks: drop pieces under a few px
    lab_d, nd_ = nd.label(inner, structure=np.ones((3, 3)))
    if nd_:
        szd = nd.sum(inner, lab_d, np.arange(1, nd_ + 1))
        ext = np.array([max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in nd.find_objects(lab_d)])
        dot = np.zeros(nd_ + 1, bool); dot[1:] = ext < 5 * s
        inner &= ~dot[lab_d]
    # on the plain white swimsuit only the long form lines stay (the ribs' shadow edges, the hip line);
    # short traced scraps there read as dirt on the fabric
    on_suit = inner & (mat == SUIT)
    lab_i, ni = nd.label(on_suit, structure=np.ones((3, 3)))
    if ni:
        szi = nd.sum(on_suit, lab_i, np.arange(1, ni + 1))
        short = np.zeros(ni + 1, bool); short[1:] = szi < 80 * s
        inner &= ~short[lab_i]
    # 1. tones, material by material
    lv = np.zeros(d.shape, np.float32)
    shadows = {}
    for key, k in (('skin', SKIN), ('suit', SUIT), ('scales', SCALES)):
        base, shad, delta = o[key]
        # fabric shadow islands smaller than a thumbprint read as stains: the suit needs bigger shapes
        l, shm = cel(ds, mat == k, base, shad, delta, 2.0 * s, int((160 if k == SUIT else 40) * s * s))
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
    n_in, _ = trace.ink_line_art(sh, place(sh, inner, box), pens=(o['inner_pen'], o['inner_pen']), weight=o['inner_w'],
                                 min_len_px=o['inner_min'] * s, seed=11, min_px=1.0, even=o['even'])
    # 4. hair: strands that follow the flow over a mid tone, black in the deepest shadow, paper where it shines
    if hair.any():
        cx, cy = to_page(505, 160)
        # the shine band (paper) belongs on the crown only; over the forehead the locks stay toned
        Yr = (np.arange(hair.shape[0])[:, None] + Y0 - to_page(0, 0)[1]) / s + 0 * np.arange(hair.shape[1])[None, :]
        hair_mod.draw_hair(sh, hair, rgb.mean(2), lines, box, s, style='tone', density=o['hair_density'],
                           ring=Yr < 300,
                           root=np.array([cx - X0, cy - Y0]), deep_at=0.88)
    # 5. contour: G-pen, the heaviest line on the figure
    if o['contour'] == 'trace':
        n_c, _ = trace.ink_line_art(sh, place(sh, contour, box), pens=('g_pen', 'g_pen'), weight=o['contour_w'],
                                    min_len_px=6 * s, seed=12, min_px=1.6 * s)
    else:
        # the silhouette as one even line; contour lines inside the figure (arm over body, thighs) traced, evened
        n_c = contour_mod.draw_contour(sh, fig, box, s, base_mm=o['contour_mm'])
        inside = (contour & nd.binary_erosion(fig, iterations=max(3, int(3 * s)))) | matline
        n2, _ = trace.ink_line_art(sh, place(sh, inside, box), pens=('saji', 'saji'), weight=1.3,
                                   min_len_px=10 * s, seed=13, min_px=1.4 * s, even=0.85)
        n_c += n2
        # the swimsuit's edge: a lighter even line round the suit, where the fabric meets skin
        suit_m = nd.binary_opening(mat == SUIT, iterations=max(1, int(s)))
        if suit_m.any():
            n_c += contour_mod.draw_contour(sh, suit_m, box, s, base_mm=0.22, swing=0.25, pen='saji', seed=14)
    # 6. blacks inside the figure: pupils, mouth, deepest folds
    blk = smooth(((ds > 0.80) | (iris & (ds > 0.68))) & fig & ~hair & ~eyez, 1.0)
    lab, n = nd.label(blk)
    if n:
        sz = nd.sum(blk, lab, range(1, n + 1))
        blk = np.isin(lab, 1 + np.nonzero(sz >= o.get('blk_min', 20) * s * s)[0])
        sh.fill(place(sh, blk, box))
    if o.get('content'):
        eyes_hand.draw_closed_eyes(sh, to_page, s)
        eyes_hand.draw_smile(sh, to_page, s)
    else:
        eyes_hand.draw_eyes(sh, to_page, s)
    skin_face = (mat > 0) & (mat != HAIR)
    eyes_hand.draw_cheek_scales(sh, to_page, s, face=place(sh, skin_face, box))
    # the face contour runs along the patch's outer edge: ink it again on top of the pale patch
    with sh.clip(place(sh, nd.binary_dilation(eyez, iterations=int(4 * s)) & ~nd.binary_erosion(fig, iterations=int(4 * s)), box)):
        contour_mod.draw_contour(sh, fig, box, s, base_mm=o['contour_mm'], seed=15)
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
