"""The girl from draw.py, built so she can be drawn from any angle.

draw.py places her hair, neck and collar by hand for one pose. Here every part is built in 3D in head units
(x across, y down, z toward the viewer, cranium radius 1) on the construction sketch.head returns, and
projected with the head's own turn, tilt and nod; the body turns half as far as the head. Layers, back to
front: shoulders and collar, neck, back hair (the bob behind the face), face, features, the locks on the
side facing us, the fringe. Each layer is filled with paper over the ones behind, so nothing needs clipping.

Same plan as draw.py: eyes first, then the face contour, hair as a light tone with a few flow strands, the
body broad and plain. Use: draw_character(sh, (cx, cy), R, turn=.., tilt=.., nod=..).
"""
import numpy as np
from PIL import Image, ImageDraw

from manga import tone, face
from manga import sketch as sk

# bangs: (across, how far the tip falls below the brow line, in head units); notches rise toward the parting
CLUMPS = [(-0.78, 0.32), (-0.6, 0.08), (-0.4, 0.2), (-0.15, 0.24), (0.05, 0.03), (0.26, 0.17), (0.46, 0.06),
          (0.72, 0.34)]
NOTCH = [0.42, 0.22, 0.48, 0.3, 0.4, 0.26, 0.5]
PART = (-0.25, -0.98)                   # the parting, on her right (viewer's left when she faces us)
HAIR_R = 1.13                           # the hair's outer shell over the cranium


def _lerp(a, b, t):
    return tuple(np.add(a, np.multiply(np.subtract(b, a), t)))


def _shell(x, y, rad=HAIR_R):
    """Point on a sphere of radius rad round the cranium centre, on its front, at (x, y)."""
    return (x, y, float(np.sqrt(max(rad * rad - x * x - y * y, 0.0))))


def _hull(pts):
    pts = sorted(map(tuple, np.asarray(pts, float)))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) -
                                   (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(q)
        return h
    lo, up = half(pts), half(pts[::-1])
    return lo[:-1] + up[:-1]


def draw_character(sh, c, R, turn=0.3, tilt=-5.0, nod=0.0, seed=1, pencil=True):
    W, H = sh.W0, sh.H0

    def poly_mask(pts):
        im = Image.new('L', (W, H), 0)
        if len(pts) > 2:
            ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=255)
        return np.asarray(im) > 127

    def ellipse_pts(cc, rx, ry, a0=0, a1=360, n=40):
        t = np.deg2rad(np.linspace(a0, a1, n))
        cr, sr = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt))
        x, y = rx * np.cos(t), ry * np.sin(t)
        return [(cc[0] + cr * a - sr * b, cc[1] + sr * a + cr * b) for a, b in zip(x, y)]

    h = sk.head(sh, c, R, turn=turn, tilt=tilt, nod=nod, seed=seed, draw=pencil)
    P, F = h['project'], h['facing']
    yaw = np.arcsin(np.clip(turn, -1, 1))
    side_near = -1 if turn > 0 else 1          # which side of her head (x sign) faces us when turned

    # ---------------------------------------------------------------- body (turns half as far as the head)
    by = yaw * 0.5
    cb, sb = np.cos(by), np.sin(by)

    def B(pts3):
        pts3 = np.atleast_2d(np.asarray(pts3, float))
        x, y, z = pts3[:, 0], pts3[:, 1], pts3[:, 2]
        return [(c[0] + R * (cb * x + sb * z), c[1] + R * y) for x, y, z in zip(x, y, z)]

    # neck: a cylinder, so its outline is two lines a constant width apart along its axis, from inside the
    # head (under the jaw) to its base on the body
    top = P([0.0, 1.0, -0.25])
    base = B([0.0, 1.95, -0.05])[0]
    ax = np.subtract(base, top)
    ax_n = np.array([-ax[1], ax[0]]) / max(np.hypot(*ax), 1e-6)
    nw = 0.4 * R
    neck_l = [tuple(top - ax_n * nw), tuple(np.add(base, -ax_n * nw * 1.12))]
    neck_r = [tuple(top + ax_n * nw), tuple(np.add(base, ax_n * nw * 1.12))]
    # shoulders: from the neck's sides (the trapezius rises into it) out and down to the arms
    sh_pts = {}
    for s in (-1, 1):
        sh_pts[s] = B([(s * 0.44, 1.72, -0.05), (s * 0.95, 2.02, -0.05), (s * 1.55, 2.25, -0.05),
                       (s * 2.0, 2.7, -0.05), (s * 2.25, 3.3, -0.05)])
    V = B([0.0, 2.85, 0.5])[0]
    lapel = {}
    for s in (-1, 1):
        lapel[s] = B([(s * 0.42, 1.95, 0.22), (s * 1.0, 2.15, 0.3), (s * 0.55, 2.9, 0.45), (0.0, 2.85, 0.5)])
    body_m = poly_mask(sh_pts[-1][::-1] + sh_pts[1] + [(sh_pts[1][-1][0], H), (sh_pts[-1][-1][0], H)])
    sh.fill(body_m, mode='white')
    for s in (-1, 1):
        sh.stroke(sh_pts[s], pen='saji', pressure=0.7, taper=(0.5, 0.0))
    col_m = np.zeros((H, W), bool)
    for s in (-1, 1):
        col_m |= poly_mask(lapel[s])
    tone(sh, col_m, level=0.48, lpi=60)
    for s in (-1, 1):
        L = lapel[s]
        sh.stroke(L[1:], pen='saji', pressure=0.65, taper=(0.5, 0.3))
        sh.stroke([L[0], L[3]], pen='saji', pressure=0.6, taper=(0.5, 0.3))
        a, b = _lerp(L[1], L[0], 0.25), _lerp(L[2], V, 0.25)
        sh.stroke([a, b], pen='white', pressure=0.35, mode='white', taper=(0.5, 0.5))
    for d in (-1, 1):
        t = B([(d * 0.05, 2.85, 0.5), (d * 0.2, 3.25, 0.5), (d * 0.08, 3.35, 0.5), (-d * 0.01, 2.93, 0.5)])
        m_t = poly_mask(t)
        sh.fill(m_t, mode='white')
        tone(sh, m_t, level=0.48, lpi=60)
        sh.stroke(t + [t[0]], pen='saji', pressure=0.55, taper=(0, 0))
    sh.fill(poly_mask(ellipse_pts((V[0], V[1] + 0.01 * R), 0.1 * R, 0.08 * R)))
    # ---------------------------------------------------------------- back hair: the bob behind the face
    # the shell over the cranium plus a curtain hanging behind down to the jaw; the far lock when it is
    # turned away. One convex outline; the face and front hair are laid over it.
    th = np.linspace(0, 2 * np.pi, 48)
    ph = np.linspace(-np.pi / 2, np.pi / 2, 13)
    shell3 = [(HAIR_R * np.cos(p) * np.cos(t), HAIR_R * np.sin(p), HAIR_R * np.cos(p) * np.sin(t))
              for t in th for p in ph]
    shell3 = [q for q in shell3 if q[1] < 0.45]                    # the bob ends round the jaw, not below
    curtain3 = [(s * x, y, z) for s in (-1, 1) for x in (0.75, 1.0) for y in (0.45, 0.95) for z in (-0.85, -0.2)]

    def lock3(s):
        """A side lock: hangs from the temple past the cheek to a point by the jaw (s = side, -1 or 1)."""
        return [(s * 0.74, -0.45, 0.5), (s * 0.68, 0.2, 0.42), (s * 0.68, 0.75, 0.32), (s * 0.82, 1.36, 0.12),
                (s * 1.14, 0.5, 0.05), (s * 1.1, -0.4, 0.25), (s * 1.06, 0.95, -0.25), (s * 1.0, 1.1, -0.5),
                (s * 1.06, 0.05, -0.45)]
    front_locks = [s for s in (-1, 1) if F(np.array([s, 0.0, 0.25])) > -0.25]
    back3 = shell3 + curtain3
    for s in (-1, 1):
        if s not in front_locks:
            back3 += lock3(s)
    back = _hull(P(back3))
    back_m = poly_mask(back)
    sh.fill(back_m, mode='white')
    tone(sh, back_m, level=0.24, lpi=60)
    sh.stroke(back + back[:1], pen='g_pen', pressure=0.8, taper=(0, 0))

    # the neck stands in front of the hair behind it
    neck_m = poly_mask(neck_l + neck_r[::-1])
    sh.fill(neck_m, mode='white')
    sh.stroke(neck_l, pen='g_pen', pressure=0.6, taper=(2.0, 0.5))
    sh.stroke(neck_r, pen='g_pen', pressure=0.55, taper=(2.0, 0.5))

    # ---------------------------------------------------------------- face: paper over the back hair
    jaw = h['jaw']
    # the face: its outline closed over the forehead below the hairline (the fringe covers that edge)
    brow_top = [tuple(q) for q, x in zip(P([_shell(x, -0.42, 1.0) for x in np.linspace(0.85, -0.85, 15)]),
                                         np.linspace(0.85, -0.85, 15))
                if F(np.array(_shell(x, -0.42, 1.0))) > 0]
    face_m = poly_mask(jaw + brow_top)
    sh.fill(face_m, mode='white')
    ci = int(np.argmax([p[1] for p in jaw]))
    eye_y = min(h['eye_l'][1], h['eye_r'][1])
    sh.stroke([p for p in jaw if p[1] > eye_y - 0.05 * R], pen='g_pen', pressure=0.7, taper=(1.5, 2.0))

    def on_face(pts2, n=None):
        """Head-unit (x, y) points on the face surface, projected, kept where the surface faces us."""
        out = []
        for x, y in pts2:
            z = sk._face_z(x, y)
            nrm = np.array([2 * sk.FACE_CURVE * x, 0.0, 1.0])
            if F(nrm / np.linalg.norm(nrm)) > 0.08:
                out.append(tuple(P([x, y, z])))
        return out

    # ---------------------------------------------------------------- eyes (primary)
    openings, irises = {}, {}
    for e, sx in (('l', -1), ('r', 1)):
        if not h['eye_vis_' + e] or h['eye_upper_' + e] is None:
            continue
        rx, ry = h['eye_size_' + e]
        U, Lo = np.array(h['eye_upper_' + e]), np.array(h['eye_lower_' + e])
        n = len(U)
        # the band's weight grows toward the outer corner; where the lid wraps round the eyeball (the rounded
        # corner of a turned eye: the lid turns down there) it tapers away again, and the flick leaves from
        # the corner's top, not from the bottom of the wrap
        dy_ = np.r_[0, np.diff(U[:, 1])]
        dx_ = np.abs(np.r_[0, np.diff(U[:, 0])])
        turn_dn = np.nonzero((dy_ > 1.2 * dx_) & (np.arange(n) > n // 2))[0]
        kc = int(turn_dn[0]) - 1 if len(turn_dn) else n - 1
        w = np.r_[np.linspace(0, 1, kc + 1), np.linspace(1, 0.1, n - kc)[1:]]
        out_dir = np.sign(U[-1, 0] - U[0, 0]) or 1.0
        opening = poly_mask([tuple(p) for p in U] + [tuple(p) for p in Lo[::-1]])
        openings[e] = opening
        band_o = [(x + out_dir * 0.06 * rx * wi, y - (0.07 + 0.13 * wi) * ry) for (x, y), wi in zip(U, w)]
        band_i = [(x, y + min((0.06 + 0.16 * wi) * ry, 0.45 * (ly - y))) for (x, y), (_, ly), wi in zip(U, Lo, w)]
        band_m = poly_mask(band_o + band_i[::-1])
        # she looks at the viewer: the iris sits where the eyeball faces us (its centre on the page), on the
        # lower lid; its width follows the opening's foreshortening
        bc, rb = h['eye_ball_' + e]
        irx, iry = 0.62 * rx, 0.95 * ry
        icx = float(np.clip(bc[0], U[:, 0].min() + irx * 0.8, U[:, 0].max() - irx * 0.8)) \
            if U[:, 0].max() - U[:, 0].min() > 2 * irx else float(np.mean(U[:, 0]))
        srt = Lo[np.argsort(Lo[:, 0])]
        ic = (icx, float(np.interp(icx, srt[:, 0], srt[:, 1])) - 0.88 * iry)
        iris_m = poly_mask(ellipse_pts(ic, irx, iry)) & opening
        irises[e] = iris_m
        pupil_m = poly_mask(ellipse_pts((ic[0], ic[1] + 0.03 * iry), 0.34 * irx, 0.48 * iry, n=24))
        lights = (poly_mask(ellipse_pts((ic[0] - 0.32 * irx, ic[1] - 0.08 * iry), 0.26 * irx, 0.2 * iry, n=20)) |
                  poly_mask(ellipse_pts((ic[0] + 0.35 * irx, ic[1] + 0.45 * iry), 0.1 * irx, 0.08 * iry, n=12))) \
            & ~band_m
        if iris_m.sum() > 20:
            face.manga_iris(sh, iris_m, pupil=pupil_m, lights=lights, solid=0.2, fade=0.86, peak=0.88,
                            pupil_r=0.16, sparkles=0, rim=0.7, seed=3)
            sh.fill(pupil_m & iris_m & ~lights)
        sh.fill(band_m)
        oc = band_o[kc]
        sh.stroke([oc, (oc[0] + out_dir * 0.25 * rx, oc[1] - 0.05 * ry),
                   (oc[0] + out_dir * 0.42 * rx, oc[1] - 0.22 * ry)], pen='g_pen', pressure=0.85, taper=(0.3, 1.6))
        k0, k1 = int(0.3 * n), int(0.92 * n)
        if k1 - k0 > 1:
            sh.stroke([(x, y - 0.32 * ry) for x, y in U[k0:k1]], pen='maru', pressure=0.45)
        under = [tuple(p) for p in Lo if abs(p[0] - ic[0]) < 0.95 * irx]
        if len(under) > 1:
            sh.stroke(under, pen='maru', pressure=0.5, taper=(1.0, 1.0))
        # brow: on the face, thick at its head by the centre line, thin toward the temple
        brow = on_face([(sx * x, sk.HEAD['brow'] - 0.03 - 0.07 * np.sin(np.pi * (x - 0.14) / 0.56))
                        for x in np.linspace(0.14, 0.7, 10)])
        if len(brow) > 1:
            sh.stroke(brow, pen='g_pen', pressure=0.55, taper=(0.2, 2.5))

    # ---------------------------------------------------------------- nose, mouth, ear, blush
    if abs(turn) < 0.85:
        ny = sk.HEAD['nose']
        nose = on_face([(0.04, ny - 0.06), (0.08, ny + 0.02), (0.02, ny + 0.04)])
        if len(nose) > 1:
            sh.stroke(nose, pen='maru', pressure=0.55, taper=(0.6, 0.4))
    my = sk.HEAD['mouth']
    left = on_face([(x, my - 0.05 * (x / 0.2) ** 2 + 0.02) for x in np.linspace(-0.2, -0.02, 6)])
    right = on_face([(x, my - 0.04 * (x / 0.2) ** 2 + 0.02) for x in np.linspace(0.02, 0.15, 5)])
    for seg, tp in ((left, (0.6, 0.3)), (right, (0.3, 0.6))):
        if len(seg) > 1:
            sh.stroke(seg, pen='maru', pressure=0.68, taper=tp)
    lip = on_face([(-0.04, my + 0.12), (0.03, my + 0.12)])
    if len(lip) > 1:
        sh.stroke(lip, pen='maru', pressure=0.35)
    for e, s in (('l', -1), ('r', 1)):
        es = h['ear_shape_' + e]
        if es is not None and F(np.array([s, 0, 0])) > 0.15:
            sh.stroke(es, pen='g_pen', pressure=0.55)
        cheek = on_face([(s * 0.5, sk.HEAD['eye'] + 0.42)])
        if cheek:
            near = (s == side_near) or abs(turn) < 0.05
            face.blush(sh, cheek[0], (0.16 if near else 0.1) * R, 0.06 * R, n=5 if near else 3, seed=1 + (s > 0))

    # ---------------------------------------------------------------- front hair: locks on our side, fringe
    hair_front = np.zeros((H, W), bool)
    for s in front_locks:
        L = _hull(P(lock3(s)))
        # the lock rises to the crown over the temple; above its middle it stays inside the hair's outline
        mid_y = P([s * 0.9, 0.4, 0.2])[1]
        m = poly_mask(L) & (back_m | (np.arange(H)[:, None] > mid_y))
        hair_front |= m
        sh.fill(m, mode='white')
        tone(sh, m, level=0.24, lpi=60)
        # its edges, except where they run across the hair behind (the lock's top): only over the face or
        # against the paper
        inner = back_m & ~face_m
        LL = L + L[:1]
        # walk each edge and ink only the runs that bound the drawn lock (not the part clipped away above the
        # hair's outline) and don't sit inside the hair behind
        for a, b in zip(LL, LL[1:]):
            n = max(int(np.hypot(b[0] - a[0], b[1] - a[1]) / 3), 2)
            run = []
            for t in np.linspace(0, 1, n + 1):
                x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                xi, yi = int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))
                near_m = m[max(yi - 3, 0):yi + 4, max(xi - 3, 0):xi + 4].any()
                win = inner[max(yi - 6, 0):yi + 7, max(xi - 6, 0):xi + 7]
                ok = near_m and not win.all() and (y > mid_y - 0.5 * R or face_m[yi, xi])
                if ok:
                    run.append((x, y))
                if (not ok or t == 1) and len(run) > 1:
                    sh.stroke(run, pen='g_pen', pressure=0.6, taper=(0, 0))
                if not ok:
                    run = []
        # two strands down the lock
        a, b = P([s * 0.8, -0.1, 0.62]), P([s * 0.84, 1.15, 0.3])
        a2, b2 = P([s * 0.98, 0.0, 0.45]), P([s * 0.92, 1.1, 0.1])
        for p, q in ((a, b), (a2, b2)):
            sh.stroke([tuple(p), _lerp(p, q, 0.5), tuple(q)], pen='maru', pressure=0.38, taper=(3.0, 3.0))
    # fringe: from the parting over the forehead to clump tips, notched between them; clumps on the far side
    # of a turned head drop out where the forehead turns away
    part3 = _shell(*PART)
    tips3, keep = [], []
    for u, drop in CLUMPS:
        y = sk.HEAD['brow'] + drop - 0.12
        t3 = _shell(u * 0.98, y, 1.04)
        tips3.append(t3)
        keep.append(F(np.array(t3) / np.linalg.norm(t3)) > -0.05)
    edge3 = [tips3[0]]
    valleys3 = []
    for (a, b), f in zip(zip(tips3, tips3[1:]), NOTCH):
        v = _lerp(_lerp(a, b, 0.5), part3, f)
        v = _shell(v[0], v[1], 1.05)
        valleys3.append(v)
        edge3 += [v, b]
    top3 = [_shell(x, -0.6, 1.1) for x in np.linspace(0.85, -0.85, 9)]
    fr_m = poly_mask(list(P(edge3 + top3)))
    fr_m &= poly_mask(back) | face_m
    hair_front |= fr_m
    sh.fill(fr_m, mode='white')
    tone(sh, fr_m, level=0.24, lpi=60)
    part2 = tuple(P(part3))
    tips2, valleys2 = P(tips3), P(valleys3)
    head_m = back_m | face_m
    for k, t in enumerate(tips2):
        ti = (int(np.clip(t[1], 0, H - 1)), int(np.clip(t[0], 0, W - 1)))
        if not keep[k] or not head_m[ti]:
            continue
        sweep = -0.05 if CLUMPS[k][0] < PART[0] else 0.05
        for v in ([valleys2[k - 1]] if k > 0 else []) + ([valleys2[k]] if k < len(valleys2) else []):
            d = np.subtract(t, v)
            nrm = np.array([d[1], -d[0]]) / max(np.hypot(*d), 1e-6)
            mid = np.add(_lerp(v, t, 0.5), nrm * sweep * R)
            sh.stroke([_lerp(v, part2, 0.3), tuple(v), tuple(mid), tuple(t)], pen='g_pen', pressure=0.5,
                      taper=(2.0, 3.0))
    # the ring of light across the crown, scraped back to paper, with a jagged lower edge
    hc = P([0, -0.05, 0])
    hr = HAIR_R * R
    ring = ellipse_pts((hc[0], hc[1] + 8), hr * 0.76, hr * 0.74, 196, 336, 15)
    for k, a in enumerate(np.linspace(336, 196, 29)):
        rr = 0.7 if k % 2 == 0 else 0.64
        ring += ellipse_pts((hc[0], hc[1] + 8), hr * rr, hr * (rr - 0.02), a, a, 1)
    from scipy.ndimage import binary_dilation
    near_locks = binary_dilation(hair_front & ~fr_m, iterations=int(0.12 * R))
    sh.fill(poly_mask(ring) & ((back_m & ~face_m) | fr_m) & ~near_locks, mode='white')     # crown and fringe only
    # flow strands from the parting
    for a in (168, 186, 204, 334, 352, 370):
        end = ellipse_pts(hc, hr * 0.9, hr * 0.92, a, a, 1)[0]
        mid = ellipse_pts(hc, hr * 0.55, hr * 0.6, (a + 270) / 2, (a + 270) / 2, 1)[0]
        sh.stroke([_lerp(part2, mid, 0.3), _lerp(mid, end, 0.3), end], pen='maru', pressure=0.42, taper=(3.0, 3.0))

    # ---------------------------------------------------------------- cast shadows (light from upper left)
    skin = face_m & ~hair_front
    for e in openings:
        skin &= ~openings[e]
    dy, dx = int(0.1 * R), int(0.035 * R)
    fs = np.zeros_like(fr_m)
    fs[dy:, dx:] = fr_m[:-dy, :-dx]
    fs &= skin & (np.arange(H)[:, None] < eye_y + 0.1 * R)
    tone(sh, fs, level=0.13, lpi=60)
    for e in openings:
        op = openings[e]
        ys = np.nonzero(op.any(axis=1))[0]
        if len(ys):
            lid = op & ~irises[e] & (np.arange(H)[:, None] < ys.min() + 0.32 * (ys.max() - ys.min()))
            tone(sh, lid, level=0.26, lpi=60)
    # the jaw's shadow on the neck
    js = np.zeros_like(face_m)
    d = int(0.12 * R)
    js[d:, :] = face_m[:-d, :]
    tone(sh, js & ~face_m & neck_m & ~hair_front & ~back_m, level=0.38, lpi=60)
    return h
