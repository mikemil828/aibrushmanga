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


def head(sh, c, r, turn=0.0, tilt=0.0, nod=0.0, pressure=0.35, draw=True, seed=None):
    """Construct a manga head: cranium circle, jaw to chin, centre line and feature lines.

    c, r  -- cranium centre and radius (base px). The chin lands about 1.5 r below the centre.
    turn  -- -1..1, how far the face turns to the viewer's right (0 = facing us, 0.5 = three-quarter).
    tilt  -- degrees the head leans (positive = clockwise on the page).
    nod   -- -1..1, positive looks down: feature lines bow downward, as they wrap round the face.

    Returns a dict of constructed points for inking: 'eye_l', 'eye_r' (centres, viewer's left/right),
    'eye_size' (rx, ry) facing us and 'eye_size_l', 'eye_size_r' narrowed by the turn, 'brow_l', 'brow_r',
    'nose', 'mouth', 'chin', 'jaw' (polyline ear to chin to ear, hidden parts left out), 'ear_l', 'ear_r' (None when hidden), 'centre' (centre line polyline),
    'hairline', 'cranium' (c, r) and 'far' (+1 when the far side is the viewer's right).
    """
    c = np.asarray(c, float)
    yaw = np.arcsin(np.clip(turn, -0.95, 0.95))
    ct, st = np.cos(np.deg2rad(tilt)), np.sin(np.deg2rad(tilt))

    def P(u, v, depth=1.0):
        """Point on the face: u across (-1 = viewer's left edge of the front), v down in radii."""
        phi = np.arcsin(np.clip(u, -1, 1)) + yaw
        x = r * depth * np.sin(phi)
        y = r * v + nod * 0.22 * r * np.cos(phi) * (v > -0.5)
        return c + np.array([ct * x - st * y, st * x + ct * y]), np.cos(phi) > 0.05

    def across(v, half=0.85, depth=1.0, n=17):
        pts = [P(u, v, depth) for u in np.linspace(-half, half, n)]
        return [tuple(p) for p, vis in pts if vis]

    # jaw: the face narrows from the cranium's width to a pointed chin
    jaw_v = np.linspace(HEAD['eye'] - 0.1, HEAD['chin'], 14)
    jaw_w = np.interp(jaw_v, [HEAD['eye'] - 0.1, 0.75, 1.15, HEAD['chin']], [0.93, 0.86, 0.6, 0.0])
    left = [P(-w, v) for w, v in zip(jaw_w, jaw_v)]
    right = [P(w, v) for w, v in zip(jaw_w, jaw_v)]
    jaw = [tuple(p) for p, vis in left if vis] + [tuple(p) for p, vis in right[::-1] if vis]
    # centre line: an arc over the cranium (a meridian), then down the front of the face to the chin
    centre = []
    for v in np.linspace(-0.97, HEAD['chin'], 18):
        depth = np.sqrt(max(1 - v * v, 0.0)) if v < 0.3 else np.interp(v, [0.3, HEAD['chin']], [0.954, 0.85])
        centre.append(tuple(P(0, v, depth)[0]))
    eye_dx = 0.42
    eye_l, vis_l = P(-eye_dx, HEAD['eye'])
    eye_r, vis_r = P(eye_dx, HEAD['eye'])
    ear_l, ear_lv = P(-1.0, (HEAD['brow'] + HEAD['nose']) / 2, depth=0.98)
    ear_r, ear_rv = P(1.0, (HEAD['brow'] + HEAD['nose']) / 2, depth=0.98)
    out = {
        'cranium': (tuple(c), r), 'centre': centre, 'jaw': jaw,
        'eye_l': tuple(eye_l), 'eye_r': tuple(eye_r),
        'eye_size': (0.28 * r, 0.23 * r),
        # each eye narrowed by how far it has turned away (the far eye is the narrow one)
        'eye_size_l': (0.28 * r * max(np.cos(np.arcsin(-eye_dx) + yaw), 0.2) / np.cos(np.arcsin(eye_dx)), 0.23 * r),
        'eye_size_r': (0.28 * r * max(np.cos(np.arcsin(eye_dx) + yaw), 0.2) / np.cos(np.arcsin(eye_dx)), 0.23 * r),
        'brow_l': tuple(P(-eye_dx, HEAD['brow'])[0]), 'brow_r': tuple(P(eye_dx, HEAD['brow'])[0]),
        'nose': tuple(P(0, HEAD['nose'])[0]), 'mouth': tuple(P(0, HEAD['mouth'])[0]),
        'chin': tuple(P(0, HEAD['chin'])[0]), 'hairline': across(HEAD['hairline'], 0.7),
        'ear_l': tuple(ear_l) if ear_lv else None, 'ear_r': tuple(ear_r) if ear_rv else None,
        'far': 1 if turn > 0 else -1,
        'lines': {k: across(HEAD[k], 0.8 if k in ('brow', 'eye') else 0.45) for k in ('brow', 'eye', 'nose', 'mouth')},
    }
    if draw:
        rng = _rng(sh, seed)
        ellipse(sh, c, r, pressure=pressure, seed=int(rng.integers(1 << 30)))
        curve(sh, jaw, pressure=pressure, seed=int(rng.integers(1 << 30)))
        curve(sh, centre, pressure=pressure * 0.7, seed=int(rng.integers(1 << 30)))
        for k, pts in out['lines'].items():
            if len(pts) > 1:
                curve(sh, pts, pressure=pressure * 0.6, seed=int(rng.integers(1 << 30)))
        for e in (out['ear_l'], out['ear_r']):
            if e is not None:
                ellipse(sh, e, 0.12 * r, 0.24 * r, rot=tilt, pressure=pressure * 0.8, passes=1,
                        seed=int(rng.integers(1 << 30)))
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
