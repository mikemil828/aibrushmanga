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

    for back, seat, legs, shadow in [
        ([(-48, 945), (12, 940), (46, 1006), (-14, 1008)], (-60, 100, 1004, 1016),
         [[(-8, 1016), (12, 1058), (34, 1016)], [(58, 1016), (75, 1050), (92, 1016)]],
         [(-60, 1052), (96, 1040), (104, 1060), (-60, 1070)]),
        ([(122, 943), (205, 942), (234, 1022), (141, 1024)], (108, 270, 1020, 1032),
         [[(112, 1032), (122, 1064), (140, 1032)], [(188, 1032), (205, 1064), (222, 1032)],
          [(240, 1030), (256, 1058), (268, 1030)]],
         [(100, 1058), (270, 1052), (278, 1070), (108, 1075)]),
    ]:
        tone(sh, m(P(shadow)), level=0.40, lpi=60)
        # legs first (behind the seat)
        for lg in legs:
            tube(lg, 0.55)
        # back: frame, black slatted panel, slats in white
        outer = np.array(P(back))
        inner = np.array(shrink(outer, 4.5 * s))
        sh.fill([tuple(p) for p in outer], mode='white')
        sh.fill([tuple(p) for p in inner])
        (a0, b0, c0, d0) = inner
        for t in np.linspace(0.1, 0.9, 8):
            p0, p1 = a0 + (d0 - a0) * t, b0 + (c0 - b0) * t
            sh.stroke([tuple(p0 + (p1 - p0) * 0.07), tuple(p0 + (p1 - p0) * 0.93)], pen='white', pressure=0.12,
                      mode='white', taper=(0.2, 0.2), smooth=False)
        sh.stroke(np.vstack([outer, outer[:1]]), pen='saji', pressure=0.75, taper=(0, 0), smooth=False)
        sh.stroke(np.vstack([inner, inner[:1]]), pen='maru', pressure=0.6, taper=(0, 0), smooth=False)
        # the frame's shine: a thin white line down the lit (left) rail is already paper; shade the right rail
        rail = [tuple(outer[1]), tuple(outer[2]), tuple(inner[2]), tuple(inner[1])]
        tone(sh, m(rail), level=0.25, lpi=60)
        # cushion: top face paper, front face toned, rounded ends
        x0_, x1_, yt, yb = seat
        top = P([(x0_, yt), (x1_, yt), (x1_, yt + 4), (x0_, yt + 4)])
        front = P([(x0_, yt + 4), (x1_, yt + 4), (x1_ - 1, yb), (x0_ + 1, yb)])
        sh.fill(top + front[2:], mode='white')
        tone(sh, m(front), level=0.22, lpi=60)
        sh.stroke(np.array(top[:2] + [front[2], front[3], top[0]]), pen='saji', pressure=0.75, taper=(0, 0),
                  smooth=False)
        sh.stroke(np.array([top[3], top[2]]), pen='maru', pressure=0.45, taper=(0, 0), smooth=False)
        # seams on the cushion top
        for xs_ in np.linspace(x0_, x1_, 5)[1:-1]:
            sh.stroke(P([(xs_, yt + 0.5), (xs_, yt + 3.5)]), pen='maru', pressure=0.3, taper=(0, 0), smooth=False)
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
