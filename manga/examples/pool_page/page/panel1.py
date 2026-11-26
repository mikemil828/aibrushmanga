"""Panel 1 drawn by hand from points read off the reference (ref px): umbrella, sea, loungers, sky.

At 3x the reference, traced lines break up, so this panel is drawn as an inker would: ruled pole and ribs,
a scalloped canopy rim, black lounger slats, the sea in line tone with sparkle scraped out.
"""
import numpy as np
from PIL import Image, ImageDraw
from manga import tone, geom
from manga.pens import get as get_pen


def draw_panel1(sh, box, to_page, s, W, H):
    P = lambda pts: [to_page(x, y) for x, y in pts]                       # noqa: E731
    X0, Y0, X1, Y1 = box

    def m(pts):
        im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
        return np.asarray(im, bool)
    full = m([(X0, Y0), (X1, Y0), (X1, Y1), (X0, Y1)])
    hy = to_page(0, 895)[1]                       # horizon
    sea_y1 = to_page(0, 1000)[1]
    # sky: gradient tone, darkest at the top
    sky = m([(X0, Y0), (X1, Y0), (X1, hy), (X0, hy)])
    tone(sh, sky, level=(0.32, 0.04, (X0, Y0), (X0, hy)), lpi=60)
    rng = np.random.default_rng(3)
    # sea: horizontal line tone, sparkle scraped out near the horizon where the sun hits
    sea = m([(X0, hy), (X1, hy), (X1, sea_y1), (X0, sea_y1)])
    tone(sh, sea, level=0.35, lpi=55, angle=90, kind='line')
    for i in range(70):
        x = rng.uniform(X0, X1); y = hy + (sea_y1 - hy) * rng.random() ** 1.6
        L = rng.uniform(6, 26) * s * (1 - (y - hy) / (sea_y1 - hy) * 0.5)
        sh.stroke([(x - L / 2, y), (x + L / 2, y)], pen='white', pressure=0.15, mode='white', taper=(0.4, 0.4),
                  smooth=False, seed=int(rng.integers(1 << 30)))
    sh.stroke([(X0, hy), (X1, hy)], pen='maru', pressure=0.5, taper=(0, 0), smooth=False)
    # deck: a light tone with a few ruled board joints in perspective
    deck = m([(X0, sea_y1), (X1, sea_y1), (X1, Y1), (X0, Y1)])
    tone(sh, deck, level=0.15, lpi=60)
    sh.stroke([(X0, sea_y1), (X1, sea_y1)], pen='saji', pressure=0.6, taper=(0, 0), smooth=False)
    # deck boards: ruled joints running toward a vanishing point beyond the frame
    vx, vy = to_page(110, 700)
    for xr in np.linspace(-200, 420, 12):
        a = to_page(xr, 1085); t = (sea_y1 - a[1]) / (vy - a[1])
        sh.stroke([a, (a[0] + (vx - a[0]) * t, sea_y1)], pen='maru', pressure=0.3, taper=(0, 0), smooth=False)
    # pole base
    bx, by = to_page(102, 1052)
    sh.fill([(bx + 14 * s * np.cos(t), by + 4 * s * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 30)])
    # loungers, drawn from the reference: a tubular white frame round a black slatted back, a thick cushion,
    # and V-shaped tube legs; each casts a soft shadow on the deck
    def tube(pts, w, cap=True):
        """A white tube with an inked edge on both sides: a wide black stroke, then a narrower white one."""
        Q = P(pts)
        sh.stroke(np.column_stack([Q, np.ones(len(Q))]), pen=get_pen('marker').with_(min_mm=w + 0.5, max_mm=w + 0.5,
                  wobble_mm=0.0), taper=(0, 0), smooth=False)
        sh.stroke(np.column_stack([Q, np.ones(len(Q))]), pen=get_pen('marker').with_(min_mm=w - 0.15,
                  max_mm=w - 0.15, wobble_mm=0.0), taper=(0, 0), smooth=False, mode='white')

    def shrink(quad, d):
        c = np.mean(quad, 0)
        out = []
        for p in quad:
            v = c - p
            out.append(tuple(p + v / np.hypot(*v) * d))
        return out

    def rounded(pts, r=0.18, it=2):
        """Chaikin corner cutting: a tube frame bends, it has no sharp corners."""
        q = np.array(pts, float)
        for _ in range(it):
            n = len(q); out = []
            for i in range(n):
                a_, b_ = q[i], q[(i + 1) % n]
                out += [a_ + (b_ - a_) * r, a_ + (b_ - a_) * (1 - r)]
            q = np.array(out)
        return q

    # the right lounger read off the reference (seen from behind, three-quarter); the left one is the same
    # lounger further along the deck
    CH = dict(
        shadow=[(70, 1058), (150, 1056), (240, 1056), (272, 1049), (272, 1073), (80, 1076)],
        legs=[[(111, 1029), (116, 1062), (134, 1064), (140, 1031)],
              [(201, 1029), (208, 1062), (224, 1063), (232, 1027)],
              [(253, 1021), (257, 1046), (264, 1046), (267, 1019)]],
        under=[(160, 1033), (177, 1033), (173, 1053), (164, 1053)],
        rail=[(108, 1028), (205, 1028), (232, 1025), (270, 1016)],
        seat=[(229, 999), (272, 1004), (272, 1013), (232, 1020)],
        back=[(122, 942), (205, 941), (234, 1021), (147, 1025)],
        panel=[(128.5, 952), (198.5, 951), (224, 1015), (153, 1018.5)])
    for dx, dy in [(-190, -2), (0, 0)]:
        D = lambda pts: [(x + dx, y + dy) for x, y in pts]          # noqa: E731
        tone(sh, m(P(D(CH['shadow']))), level=0.45, lpi=60)
        # legs: open tube loops, white with an inked edge; behind them, the base rail
        for lg in CH['legs']:
            tube(D(lg), 1.05)
        # the far leg under the seat, in the seat's shadow
        tone(sh, m(P(D(CH['under']))), level=0.7, lpi=60)
        tube(D(CH['rail']), 1.15)
        # seat cushion beyond the back: lit top, toned front edge
        seat = P(D(CH['seat']))
        sh.fill(seat, mode='white')
        tone(sh, m([seat[3], seat[2], (seat[2][0], seat[2][1] - 3 * s), (seat[3][0], seat[3][1] - 3 * s)]),
             level=0.3, lpi=60)
        sh.stroke(np.vstack([seat, seat[:1]]), pen='saji', pressure=0.7, taper=(0, 0), smooth=False)
        # back: a white tube frame with rounded corners round a black slatted panel
        outer = rounded(P(D(CH['back'])))
        inner = np.array(P(D(CH['panel'])))
        sh.fill([tuple(p) for p in outer], mode='white')
        sh.fill([tuple(p) for p in rounded(inner, 0.12, 1)])
        (a0, b0, c0, d0) = inner
        for tt in np.linspace(0.12, 0.92, 8):
            p0, p1 = a0 + (d0 - a0) * tt, b0 + (c0 - b0) * tt
            sh.stroke([tuple(p0 + (p1 - p0) * 0.1), tuple(p0 + (p1 - p0) * 0.9)], pen='white', pressure=0.14,
                      mode='white', taper=(0.3, 0.3), smooth=False)
        # the frame's right rail turns from the light: one tone on it, a shine line down the left rail
        o4 = np.array(P(D(CH['back'])))
        tone(sh, m([tuple(o4[1]), tuple(o4[2]), tuple(inner[2]), tuple(inner[1])]) & m([tuple(p) for p in outer]),
             level=0.3, lpi=60)
        sh.stroke(np.vstack([outer, outer[:1]]), pen='g_pen', pressure=0.55, taper=(0, 0), smooth=False)
        sh.stroke(np.vstack([rounded(inner, 0.12, 1), rounded(inner, 0.12, 1)[:1]]), pen='maru', pressure=0.5,
                  taper=(0, 0), smooth=False)
    # umbrella pole: two ruled lines with black between, behind the canopy
    px0, py0 = to_page(102, 772); px1, py1 = to_page(102, 1052)
    w = 2.6 * s
    sh.fill([(px0 - w, py0), (px0 + w, py0), (px1 + w, py1), (px1 - w, py1)])
    # canopy underside: light tone between the rim and the hub, ribs ruled with the saji pen
    apex, hub = (103, 687), (102, 772)
    rim_top = [(-40, 742), (0, 750), (60, 757), (102, 760), (150, 757), (205, 752), (250, 742)]
    rim_low = [(-40, 760), (0, 770), (45, 788), (102, 778), (165, 790), (205, 770), (250, 758)]
    under = P(rim_top + rim_low[::-1])
    sh.fill(under, mode='white')
    tone(sh, m(P(rim_low + [hub])), level=0.45, lpi=60)
    for tip in [(-20, 765), (45, 788), (165, 790), (230, 762)]:
        sh.stroke(P([hub, tip]), pen='saji', pressure=0.85, taper=(0, 0.2), smooth=False)
    # canopy top: paper, outlined with the G-pen; seams from the apex to each rim point
    top = P([apex] + rim_top[1:-1] + [apex])
    canopy = P([apex, (-40, 742), (0, 750), (60, 757), (102, 760), (150, 757), (205, 752), (250, 742)])
    sh.fill(canopy, mode='white')
    for q in [(0, 750), (60, 757), (150, 757), (205, 752)]:
        sh.stroke(P([apex, q]), pen='maru', pressure=0.45, taper=(0.3, 0.3), smooth=False)
    # canopy shading: the right side turned away from the sun, one light tone
    tone(sh, m(P([apex, (102, 760), (150, 757), (205, 752), (250, 742)])), level=0.18, lpi=60)
    sh.stroke(P([(-40, 742), apex, (250, 742)]), pen='g_pen', pressure=0.6, taper=(0, 0), smooth=False)
    # scalloped rim: a run of shallow arcs between rib tips
    pts = []
    for (ax, ay), (bx, by) in zip(rim_low[:-1], rim_low[1:]):
        for t in np.linspace(0, 1, 8, endpoint=False):
            pts.append((ax + (bx - ax) * t, ay + (by - ay) * t + 4 * np.sin(np.pi * t)))
    pts.append(rim_low[-1])
    sh.stroke(P(pts), pen='g_pen', pressure=0.5, taper=(0, 0))
    sh.stroke(P(rim_top), pen='maru', pressure=0.6, taper=(0, 0))
    for a, b in zip(rim_top, rim_low):
        sh.stroke(P([a, b]), pen='maru', pressure=0.5, taper=(0, 0), smooth=False)
    # finial on the apex
    ax_, ay_ = to_page(*apex)
    sh.fill([(ax_ + 3.5 * s * np.cos(t), ay_ - 2 * s + 3 * s * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 20)])
    # palm fronds hanging into the top of the frame: a few tapered brush strokes
    for x0_, y0_, x1_, y1_ in [(-60, 650, -20, 700), (-50, 650, -45, 720), (-40, 650, 0, 690), (-30, 650, 20, 675), (-55, 650, -75, 705)]:
        a = to_page(x0_, y0_); b = to_page(x1_, y1_)
        mid = ((a[0] + b[0]) / 2 + 6 * s, (a[1] + b[1]) / 2)
        sh.stroke([a, mid, b], pen='brush', pressure=0.35, taper=(0.5, 3.0))
