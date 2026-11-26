"""Panel 1 drawn by hand from points read off the reference (ref px): umbrella, sea, loungers, sky.

At 3x the reference, traced lines break up, so this panel is drawn as an inker would: ruled pole and ribs,
a scalloped canopy rim, black lounger slats, the sea in line tone with sparkle scraped out.
"""
import numpy as np
from PIL import Image, ImageDraw
from manga import tone, geom


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
    # loungers: backrest, seat and legs in saji; slats solid black with white gaps
    for back, seat, legs in [
        ([(-30, 940), (25, 943), (60, 1015), (5, 1015)], [(-30, 1025), (95, 1020), (95, 1034), (-30, 1040)],
         [[(5, 1035), (30, 1070)], [(30, 1035), (5, 1070)], [(70, 1030), (90, 1065)]]),
        ([(122, 943), (193, 942), (235, 1015), (165, 1016)], [(115, 1026), (275, 1016), (275, 1030), (115, 1040)],
         [[(130, 1040), (150, 1075)], [(150, 1040), (130, 1075)], [(240, 1030), (262, 1068)], [(262, 1030), (240, 1068)]]),
    ]:
        bp = P(back)
        sh.fill(bp)
        (a, b, c, d) = [np.array(p) for p in bp]
        for t in np.linspace(0.12, 0.88, 7):            # white gaps between slats
            p0, p1 = a + (d - a) * t, b + (c - b) * t
            sh.stroke([tuple(p0 + (p1 - p0) * 0.06), tuple(p0 + (p1 - p0) * 0.94)], pen='white', pressure=0.15,
                      mode='white', taper=(0, 0), smooth=False)
        sh.stroke(bp + [bp[0]], pen='saji', pressure=0.8, taper=(0, 0), smooth=False)
        sp = P(seat)
        sh.fill(sp, mode='white')
        sh.stroke(sp + [sp[0]], pen='saji', pressure=0.7, taper=(0, 0), smooth=False)
        for lg in legs:
            sh.stroke(P(lg), pen='saji', pressure=0.9, taper=(0, 0), smooth=False)
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
