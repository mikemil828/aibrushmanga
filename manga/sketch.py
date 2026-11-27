"""Pencil construction on the sketch layer: decide the drawing before any ink goes down.

Everything here draws with the pencil (Sheet.pencil, mode='sketch'), which is shown in non-photo blue under
the ink on request (Sheet.png(..., sketch=True), replay --sketch on) and never printed.

The helpers also return the points they constructed, so the inking pass can follow the decided sketch
instead of guessing again:

    sh = Sheet(1200, 1600)
    h = sketch.head(sh, (600, 520), 230, turn=0.3, tilt=-5)     # cranium, jaw, centre and feature lines
    sketch.ellipse(sh, h['eye_r'], *h['eye_size'])             # rough eye
    ...look at sh.png('s.png', sketch='only'), move things, then ink along h['jaw'], h['eye_r'], ...

Coordinates are base px, like every other call.
"""
import numpy as np

# manga head proportions, in cranium radii below the cranium centre (top of head = -1)
HEAD = {'hairline': -0.42, 'brow': 0.12, 'eye': 0.36, 'nose': 0.82, 'mouth': 1.12, 'chin': 1.52}


def _rng(sh, seed):
    return np.random.default_rng(seed if seed is not None else (sh.seed * 7907 + sh._n * 104729) & 0x7fffffff)


def line(sh, a, b, pressure=0.3, overshoot=0.04, seed=None):
    """A ruled guide line from a to b, run a little past both ends as a hand does."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = (b - a) * overshoot
    sh.pencil([tuple(a - d), tuple(b + d)], pressure=pressure, smooth=False, seed=seed)


def ellipse(sh, c, rx, ry=None, rot=0.0, pressure=0.4, passes=2, loose=0.03, seed=None):
    """A loose freehand ellipse: `passes` overlapping laps, each a little off, overshooting its start."""
    ry = rx if ry is None else ry
    rng = _rng(sh, seed)
    cr, sr = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
    for k in range(passes):
        a0 = rng.uniform(0, 2 * np.pi)
        t = a0 + np.linspace(0, 2 * np.pi * 1.08, 40)
        jx = 1 + loose * rng.standard_normal()
        jy = 1 + loose * rng.standard_normal()
        off = rng.standard_normal(2) * loose * min(rx, ry) * 0.5
        x, y = rx * jx * np.cos(t), ry * jy * np.sin(t)
        pts = [(c[0] + off[0] + cr * xi - sr * yi, c[1] + off[1] + sr * xi + cr * yi) for xi, yi in zip(x, y)]
        sh.pencil(pts, pressure=pressure * (1.0 if k == 0 else 0.75), seed=int(rng.integers(1 << 30)))


def curve(sh, pts, pressure=0.45, seed=None):
    """A free pencil line through points (gesture lines, rough contours, hair flow)."""
    sh.pencil([tuple(p) for p in pts], pressure=pressure, seed=seed)


# Loomis construction in head units (cranium radius = 1; x to the head's own left as we see it facing us,
# y down, z toward the viewer). The ball is sliced flat on both sides at |x| = SIDE: the side plane, a disc of
# radius sqrt(1 - SIDE^2). Its centre is where the brow line crosses it, the ear sits just behind its vertical
# line between the brow and nose lines, and the jaw drops from below it to the chin.
SIDE = 0.82
# profile of the face down its centre line: depth (z) at each height (y)
PROFILE = ([-1.0, HEAD['hairline'], HEAD['brow'], HEAD['eye'], 0.66, HEAD['nose'], 0.9, HEAD['mouth'], 1.28,
            1.42, HEAD['chin']],
           [0.0, 0.91, 0.99, 0.94, 0.99, 1.04, 0.93, 0.92, 0.85, 0.86, 0.76])
# cheek down the front edge of the side plane to the jaw corner (below the plane's vertical line), then to the chin
CHEEK = [(0.82, HEAD['brow'], 0.55), (0.81, 0.45, 0.42), (0.76, 0.8, 0.22), (0.69, 1.02, 0.02)]
# the jaw leaves its corner steeply and rounds off toward the chin, so it bows down and out (never a ruler line)
JAW = [(0.69, 1.02, 0.02), (0.64, 1.16, 0.2), (0.5, 1.31, 0.4), (0.3, 1.43, 0.58), (0.13, 1.49, 0.7),
       (0.0, HEAD['chin'], 0.76)]
RAMUS = (0.8, 0.8, -0.1)          # the back of the jaw rises from its corner to just under the ear
EYE_SPAN, EYE_OUT, EYE_SET, EYE_LOW = 65.0, 22.0, 1.1, 0.8  # opening half-width on the ball (deg), outward aim, depth, lower lid
FACE_CURVE = 0.9     # how fast the front of the face falls back toward the sides: z = profile - FACE_CURVE * x^2


def _face_z(x, y):
    return np.interp(y, *PROFILE) - FACE_CURVE * x * x


def _upper_envelope(x, y):
    """y raised to its concave upper hull over x (x need not be sorted); the order of the points is kept."""
    order = np.argsort(x)
    hull = []
    for i in order:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = (x[hull[-2]], y[hull[-2]]), (x[hull[-1]], y[hull[-1]])
            if (x2 - x1) * (y[i] - y1) - (y2 - y1) * (x[i] - x1) >= 0:
                hull.pop()
            else:
                break
        hull.append(i)
    hx, hy = x[hull], y[hull]
    return np.maximum(y, np.interp(x, hx, hy))


def _hull(pts):
    """Convex hull of 2D points, counter-clockwise (monotone chain)."""
    pts = sorted(map(tuple, np.asarray(pts, float)))
    def half(seq):
        h = []
        for q in seq:
            while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (q[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (q[0] - h[-2][0]) <= 0:
                h.pop()
            h.append(q)
        return h
    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _resample(pts, n):
    pts = np.asarray(pts, float)
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    t = np.linspace(0, d[-1], n)
    return np.stack([np.interp(t, d, pts[:, k]) for k in range(pts.shape[1])], 1)


def head(sh, c, r, turn=0.0, tilt=0.0, nod=0.0, pressure=0.35, draw=True, seed=None):
    """Construct a manga head the Loomis way: a ball for the cranium, sliced flat at the sides (the side
    planes), a centre line and a brow line wrapped round it, the jaw hung from below the side plane to the
    chin, and feature lines across the face. It is built in 3D and turned, so it holds from facing us to a
    full profile and when the head looks up or down.

    c, r  -- cranium centre and radius (base px). The chin lands about 1.5 r below the centre.
    turn  -- -1..1, how far the face turns to the viewer's right: sin of the angle (0 = facing us,
             0.5 = 30 degrees, 0.71 = a true three-quarter, 1 = profile facing right).
    tilt  -- degrees the head leans (positive = clockwise on the page).
    nod   -- -1..1, positive looks down (up to 30 degrees): feature lines bow downward round the face.

    Returns a dict of constructed points for inking: 'eye_l', 'eye_r' (centres, viewer's left/right),
    'eye_size' (rx, ry) facing us and 'eye_size_l', 'eye_size_r' foreshortened by the turn, 'eye_vis_l',
    'eye_vis_r', 'eye_shape_l/r' (opening outline), 'eye_upper_l/r' and 'eye_lower_l/r' (lids, inner corner to
    outer, None when hidden), 'brow_l', 'brow_r', 'nose', 'mouth', 'chin', 'jaw' (the face outline, viewer's left side
    top to chin, then the right side back up; on the far side it follows the profile of brow, nose, mouth and
    chin wherever that stands out past the cheek), 'ear_l', 'ear_r' (centres, None when hidden), 'ear_shape_l',
    'ear_shape_r' (outlines), 'side_l', 'side_r' (side-plane discs, None when facing away), 'centre' (centre
    line polyline), 'profile' (the centre line below the brow), 'hairline', 'cranium' (c, r), 'far' (+1 when
    the far side is the viewer's right) and 'lines' (brow, eye, nose and mouth lines across the face).
    """
    c = np.asarray(c, float)
    yaw = np.arcsin(np.clip(turn, -1.0, 1.0))
    pitch = np.deg2rad(30.0) * float(np.clip(nod, -1, 1))
    roll = np.deg2rad(tilt)
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    # the head turns about the neck, not the cranium centre: pivot a little below and behind it
    pivot = np.array([0.0, 0.5, -0.1])

    def rot(v):
        v = np.asarray(v, float)
        x, y, z = v[..., 0], v[..., 1], v[..., 2]
        x, z = cy * x + sy * z, -sy * x + cy * z                  # turn
        y, z = cp * y + sp * z, -sp * y + cp * z                  # nod (positive: chin toward the chest)
        return np.stack([x, y, z], -1)

    def proj(v):
        v = rot(np.asarray(v, float) - pivot) + rot(pivot) * 0 + pivot
        x, y = v[..., 0] * r, v[..., 1] * r
        return np.stack([c[0] + cr * x - sr * y, c[1] + sr * x + cr * y], -1), v[..., 2]

    def facing(n):
        return rot(n)[..., 2]

    def P(x, y):
        """Point on the front of the face, its screen position and whether it faces us."""
        z = _face_z(x, y)
        n = np.array([2 * FACE_CURVE * x, 0.0, 1.0])
        p, _ = proj([x, y, z])
        return p, facing(n / np.linalg.norm(n)) > 0.05

    def across(y, half, n=17):
        pts = [P(x, y) for x in np.linspace(-half, half, n)]
        return [tuple(p) for p, vis in pts if vis]

    out = {'cranium': (tuple(c), r), 'far': 1 if turn > 0 else -1}
    # centre line: a meridian over the ball, then down the profile to the chin
    ys = np.linspace(-0.97, HEAD['chin'], 26)
    cen3 = np.stack([np.zeros_like(ys), ys,
                     np.where(ys < HEAD['hairline'], np.sqrt(np.clip(1 - ys * ys, 0, 1)), np.interp(ys, *PROFILE))], 1)
    cen2, _ = proj(cen3)
    cen_vis = facing(np.array([0.0, 0.0, 1.0])) > -0.25        # the centre line shows until the head turns away
    out['centre'] = [tuple(p) for p in cen2]
    out['profile'] = [tuple(p) for p, y in zip(cen2, ys) if y >= HEAD['brow']]
    # side planes and ears
    rad = np.sqrt(1 - SIDE * SIDE)
    th = np.linspace(0, 2 * np.pi, 41)
    for e, sx in (('l', -1), ('r', 1)):
        n_side = np.array([sx, 0.0, 0.0])
        vis = facing(n_side) > 0.02
        disc = np.stack([np.full_like(th, sx * SIDE), rad * np.sin(th), rad * np.cos(th)], 1)
        out['side_' + e] = [tuple(p) for p in proj(disc)[0]] if vis else None
        # the ear: just behind the plane's vertical line, brow line to nose line, its back edge flared out
        ec = np.array([sx * (SIDE + 0.02), (HEAD['brow'] + HEAD['nose']) / 2, -0.17])
        hdir = np.array([sx * np.sin(np.deg2rad(25)), 0.0, -np.cos(np.deg2rad(25))])
        a, b = 0.2, (HEAD['nose'] - HEAD['brow']) / 2
        t = np.linspace(0, 2 * np.pi, 33)
        # a little egg: wider at the top
        shape = ec + np.outer(a * np.cos(t) * (1 - 0.25 * np.sin(t)), hdir) + np.outer(b * np.sin(t), [0, 1, 0])
        ear_vis = facing(n_side) > -0.35
        out['ear_' + e] = tuple(proj(ec)[0]) if ear_vis else None
        out['ear_shape_' + e] = [tuple(p) for p in proj(shape)[0]] if ear_vis else None
    # eyes: each is an eyeball set into its socket behind the face surface. The opening (lids) lies on the
    # ball and faces forward and a little outward (EYE_OUT), so the outer corner sits further back than the
    # inner one. Its outline on the page is the hull of the projected lids: an almond facing us, a wedge
    # opening toward the nose in profile, set back from the nose line by the socket's depth.
    eye_dx, ex, ey = 0.42, 0.3, 0.29
    out['eye_size'] = (ex * r, ey * r)
    span = np.deg2rad(EYE_SPAN)
    re = ex / np.sin(span)                                   # ball radius so the opening is 2 ex wide facing us
    for e, sx in (('l', -1), ('r', 1)):
        x = sx * eye_dx
        p, vis = P(x, HEAD['eye'])
        phi = np.linspace(-span, span, 21)                   # across the opening, inner corner (-) to outer (+)
        tt = np.linspace(0, 1, 21)
        # manga lids: the upper one full and arched, highest a little toward the outer corner; the lower
        # one shallower, lowest a little toward the inner corner
        lid_up = np.sin(np.pi * tt ** 0.85) ** 0.6
        lid_lo = np.sin(np.pi * tt ** 1.15) ** 0.8
        a = phi + np.deg2rad(EYE_OUT)                       # aim of each lid point round the ball, from forward
        # shift the ball inward so that, facing us, the opening is centred on the eye's place on the face
        bx = x - sx * re * (np.sin(a[-1]) + np.sin(a[0])) / 2
        zc = _face_z(bx, HEAD['eye']) - EYE_SET * re
        ball = np.array([bx, HEAD['eye'], zc])
        lx = ball[0] + sx * re * np.sin(a)
        lz = ball[2] + re * np.cos(a)
        up = np.stack([lx, ball[1] - ey * lid_up, lz], 1)
        lo = np.stack([lx, ball[1] + EYE_LOW * ey * lid_lo, lz], 1)
        # the cornea bulges a little in front of the ball between the lids: in profile the front of the
        # opening rounds forward instead of being cut off flat
        cy_ = np.linspace(-0.8 * ey, 0.55 * ey, 7)
        bulge = 1 + 0.12 * np.cos(np.pi * (cy_ - cy_.mean()) / (cy_[-1] - cy_[0]))
        cornea = np.stack([np.full(7, ball[0]), ball[1] + cy_, ball[2] + re * bulge], 1)
        pts2 = proj(np.r_[up, lo[::-1], cornea])[0]
        hull = _hull(pts2)
        # the centre is the middle of the opening mirrored top to bottom, so it stays on the eye line
        box = proj(np.r_[up, up * [1, -1, 1] + [0, 2 * ball[1], 0]])[0]
        lo2, hi2 = box.min(0), box.max(0)
        tan = np.array([1.0, 0.0, -2 * FACE_CURVE * x])
        tan /= np.linalg.norm(tan)
        shrink = np.linalg.norm(rot(tan)[:2])
        # the centre follows the opening: in a turn the near eye's middle slides back from the nose line
        ctr = (lo2 + hi2) / 2 if vis else p
        out['eye_' + e] = (float(ctr[0]), float(ctr[1]))
        out['eye_vis_' + e] = bool(vis)
        out['eye_size_' + e] = (ex * r * max(shrink, 0.15), ey * r * max(abs(cp), 0.5))
        out['eye_shape_' + e] = [tuple(q) for q in hull] if vis else None
        # the lids themselves, inner corner to outer, only where the ball under them faces us (in profile
        # the half toward the nose turns edge-on and drops out, leaving the wedge's two sides)
        nrm = np.stack([np.sin(a) * sx, np.zeros_like(a), np.cos(a)], 1)
        show = facing(nrm) > -0.05
        for name, lid3 in (('upper', up), ('lower', lo)):
            q = proj(lid3)[0][show]
            out['eye_%s_%s' % (name, e)] = [tuple(v) for v in q] if vis and len(q) > 1 else None
        out['brow_' + e] = tuple(P(x, HEAD['brow'])[0])
    for k in ('nose', 'mouth', 'chin'):
        out[k] = tuple(P(0.0, HEAD[k])[0])
    out['hairline'] = across(HEAD['hairline'], 0.62)
    out['lines'] = {k: across(HEAD[k], 0.78 if k in ('brow', 'eye') else 0.42) for k in ('brow', 'eye', 'nose', 'mouth')}
    # face outline. Each side: cheek then jaw, seen where its surface faces us; on the side turned away, the
    # silhouette is whichever stands further out, that cheek or the profile down the centre line.
    contour = _resample(CHEEK + JAW[1:], 30)
    sides = {}
    for e, sx in (('l', -1), ('r', 1)):
        if facing(np.array([sx, 0.0, 0.0])) > 0.12:
            # this side plane faces us, so its cheek is inside the face: the outline there is the jaw, from
            # its corner up toward the ear (the ramus, which the ear and hair cover) and down to the chin
            pts3 = _resample([RAMUS] + JAW, 30)
        else:
            pts3 = contour
        sides[e] = proj(pts3 * np.array([sx, 1, 1]))[0]
    def prof_for(pts3):
        return proj(np.stack([np.zeros(len(pts3)), pts3[:, 1], np.interp(pts3[:, 1], *PROFILE)], 1))[0]
    prof2 = prof_for(contour)
    # which way is "out" on the screen for the far side: from the face centre toward that side
    right = np.array([cr, sr])
    far = 'r' if turn > 0 else 'l'
    s_out = 1.0 if far == 'r' else -1.0
    if abs(turn) > 1e-6:
        # On the side turned away the outline is the limb of the cheek, not its edge: across each height the
        # face rounds from the centre line (profile depth) out to the cheek/jaw contour like a quarter
        # ellipse, and the outline is where that cross-section stands out furthest on the page. It follows
        # the cheekbone and swings in toward the chin as a curve, where the bare contour gave a ruler line.
        sxf = 1.0 if far == 'r' else -1.0
        # The search starts 45 degrees off the centre line (the centre is the profile, added below with its nose
        # and lips) and narrows to the contour itself as the head nears profile, where the cheek falls behind
        # the nose and lips and the full cross-section would hide them.
        th0 = np.deg2rad(45 + 45 * np.clip((abs(turn) - 0.6) / 0.3, 0, 1))
        th = np.linspace(th0, np.pi / 2, 24)
        limb = []
        for xe, ye, ze in contour:
            zf = np.interp(ye, *PROFILE)
            w = abs(xe)
            cand = np.stack([sxf * w * np.sin(th), np.full_like(th, ye), ze + (zf - ze) * np.cos(th)], 1)
            q = proj(cand)[0]
            limb.append(q[np.argmax((q - c) @ right * s_out)])
        sides[far] = np.array(limb)
        # the jaw corner drops behind the cheek: keep the outer envelope
        down = np.array([-sr, cr])
        rel = sides[far] - c
        dn, ou = rel @ down, rel @ right * s_out
        ou = _upper_envelope(dn, ou)
        cheek = c + np.outer(dn, down) + np.outer(ou * s_out, right)
        use_prof = (prof2 - cheek) @ right * s_out > 0
        sides[far] = np.where(use_prof[:, None], prof2, cheek)
    out['jaw'] = [tuple(p) for p in sides['l']] + [tuple(p) for p in sides['r'][::-1]]
    if draw:
        rng = _rng(sh, seed)
        s = lambda: int(rng.integers(1 << 30))
        ellipse(sh, c + (proj([0, 0, 0])[0] - c), r, pressure=pressure, seed=s())
        for e in 'lr':
            if out['side_' + e] is not None:
                curve(sh, out['side_' + e], pressure=pressure * 0.8, seed=s())
                # the side plane's cross: vertical line down to the jaw corner, horizontal along the brow line
                sx = -1 if e == 'l' else 1
                curve(sh, [tuple(proj([sx * SIDE, y, 0.0])[0]) for y in np.linspace(-rad, 1.02, 8)],
                      pressure=pressure * 0.6, seed=s())
            if out['ear_shape_' + e] is not None:
                curve(sh, out['ear_shape_' + e], pressure=pressure * 0.8, seed=s())
        curve(sh, out['jaw'], pressure=pressure, seed=s())
        if cen_vis:
            curve(sh, out['centre'], pressure=pressure * 0.7, seed=s())
        # brow line all the way round the ball, front part only
        bl = [proj([np.sin(a), HEAD['brow'], np.cos(a)]) for a in np.linspace(-np.pi / 2, np.pi / 2, 25)]
        bl = [tuple(p) for (p, _), a in zip(bl, np.linspace(-np.pi / 2, np.pi / 2, 25))
              if facing(np.array([np.sin(a), 0, np.cos(a)])) > 0]
        if len(bl) > 1:
            curve(sh, bl, pressure=pressure * 0.6, seed=s())
        for k, pts in out['lines'].items():
            if k != 'brow' and len(pts) > 1:
                curve(sh, pts, pressure=pressure * 0.6, seed=s())
    return out


def figure(sh, joints, head_r, pressure=0.35, seed=None):
    """A stick-and-bean figure: joints {'neck','chest','pelvis','l_shoulder','l_elbow','l_hand', ... 'r_foot'}.

    Draws the ribcage and pelvis as ellipses, limbs as lines and small circles at the joints. Any joint may be
    left out. Returns nothing; the joints you passed are the construction.
    """
    rng = _rng(sh, seed)
    s = lambda: int(rng.integers(1 << 30))
    J = {k: np.asarray(v, float) for k, v in joints.items()}
    if 'chest' in J and 'neck' in J:
        d = J['chest'] - J['neck']
        ang = np.degrees(np.arctan2(d[1], d[0])) - 90
        ellipse(sh, J['chest'], 1.05 * head_r, 1.25 * head_r, rot=ang, pressure=pressure, passes=1, seed=s())
        line(sh, J['neck'], J['chest'], pressure, seed=s())
    if 'pelvis' in J:
        ellipse(sh, J['pelvis'], 1.0 * head_r, 0.65 * head_r, pressure=pressure, passes=1, seed=s())
        if 'chest' in J:
            curve(sh, [J['chest'], (J['chest'] + J['pelvis']) / 2 + [0.1 * head_r, 0], J['pelvis']], pressure, s())
    for side in ('l', 'r'):
        for chain in (('neck', 'shoulder', 'elbow', 'hand'), ('pelvis', 'hip', 'knee', 'foot')):
            pts = [J[n if n in ('neck', 'pelvis') else f'{side}_{n}'] for n in chain
                   if (n if n in ('neck', 'pelvis') else f'{side}_{n}') in J]
            for a, b in zip(pts, pts[1:]):
                line(sh, a, b, pressure, overshoot=0.0, seed=s())
            for p in pts[1:]:
                ellipse(sh, p, 0.12 * head_r, pressure=pressure * 0.8, passes=1, seed=s())


def perspective(sh, box, horizon_y, vps, rays=12, pressure=0.25, seed=None):
    """A perspective grid inside box (x0, y0, x1, y1): the horizon line and `rays` lines from each
    vanishing point (vps: list of (x, y); a point may lie outside the box). Clipped to the box."""
    rng = _rng(sh, seed)
    x0, y0, x1, y1 = box
    with sh.clip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        line(sh, (x0, horizon_y), (x1, horizon_y), pressure * 1.3, overshoot=0.0, seed=int(rng.integers(1 << 30)))
        far = 4 * max(x1 - x0, y1 - y0)
        for vp in vps:
            vp = np.asarray(vp, float)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            base = np.arctan2(cy - vp[1], cx - vp[0])
            span = np.pi * 0.9 if x0 <= vp[0] <= x1 else np.pi * 0.45
            for a in base + np.linspace(-span / 2, span / 2, rays):
                line(sh, vp, vp + far * np.array([np.cos(a), np.sin(a)]), pressure, overshoot=0.0,
                     seed=int(rng.integers(1 << 30)))
