"""Manga effects and page furniture: focus lines, speed lines, flash, panel layouts, borders, captions."""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def focus_lines(sheet, center, inner, outer_box, count=220, pen='g_pen', pressure=(0.55, 1.0),
                reach=0.35, bundles=True, seed=0):
    """Concentrated lines (shuchu-sen): strokes from outside toward `center`, stopping around an inner ellipse.

    center    -- (x, y) base px.  inner -- (rx, ry) of the clear ellipse.
    outer_box -- (x0, y0, x1, y1): strokes start on (or beyond) this box, usually the panel.
    reach     -- how irregular the inner ends are (fraction of the inner radius).
    """
    rng = np.random.default_rng(seed)
    cx, cy = center
    rx, ry = inner
    x0, y0, x1, y1 = outer_box
    big = 2 * max(x1 - x0, y1 - y0)
    if bundles:
        nb = count // 5 + 1                      # bundles spread evenly round the circle, jittered
        base = (np.arange(nb) + rng.uniform(0.15, 0.85, nb)) * 2 * np.pi / nb
        ang = np.concatenate([b + rng.normal(0, 0.03, rng.integers(3, 8)) for b in base])[:count]
    else:
        ang = (np.arange(count) + rng.uniform(0, 1, count)) * 2 * np.pi / count
    n = 0
    for a in ang:
        dx, dy = np.cos(a), np.sin(a)
        # where the ray leaves the box
        ts = []
        for t in ((x0 - cx) / dx if dx else np.inf, (x1 - cx) / dx if dx else np.inf,
                  (y0 - cy) / dy if dy else np.inf, (y1 - cy) / dy if dy else np.inf):
            if t > 0:
                ts.append(t)
        t_out = min(ts) if ts else big
        e = 1.0 / np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)       # ellipse radius along the ray
        t_in = e * (1 + reach * rng.uniform(-0.3, 1.0))
        if t_in >= t_out:
            continue
        t_start = t_out + 4
        p = rng.uniform(*pressure)
        pts = [(cx + dx * t_start, cy + dy * t_start, p), (cx + dx * t_in, cy + dy * t_in, p * 0.9)]
        sheet.stroke(pts, pen=pen, taper=(0.0, (t_start - t_in) / sheet.ppm0 * rng.uniform(0.5, 0.85)),
                     seed=int(rng.integers(1 << 30)))
        n += 1
    return n


def speed_lines(sheet, box, angle=0.0, count=90, length=(0.3, 0.9), pen='maru', pressure=(0.4, 0.9),
                fade='both', seed=0):
    """Parallel speed lines (ryusen) inside box (x0, y0, x1, y1). length -- fraction of the box span."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    th = np.deg2rad(angle)
    d = np.array([np.cos(th), np.sin(th)])
    nv = np.array([-d[1], d[0]])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) - [cx, cy]
    span_d = np.ptp(corners @ d)
    span_n = np.ptp(corners @ nv)
    for _ in range(count):
        o = rng.uniform(-span_n / 2, span_n / 2)
        L = rng.uniform(*length) * span_d
        c = rng.uniform(-span_d / 2 + L / 2, span_d / 2 - L / 2) if span_d > L else 0
        a = np.array([cx, cy]) + nv * o + d * (c - L / 2)
        b = np.array([cx, cy]) + nv * o + d * (c + L / 2)
        p = rng.uniform(*pressure)
        tl = L / sheet.ppm0
        taper = {'both': (tl * 0.35, tl * 0.45), 'end': (0.3, tl * 0.7), 'start': (tl * 0.7, 0.3)}[fade]
        sheet.stroke([(a[0], a[1], p), (b[0], b[1], p)], pen=pen, taper=taper, seed=int(rng.integers(1 << 30)))
    return count


def flash(sheet, center, r_in, r_out, count=160, pen='maru', pressure=(0.4, 0.9), seed=0):
    """Beta-flash / uni-flash: thin radial lines between two radii, tapered at both ends."""
    rng = np.random.default_rng(seed)
    cx, cy = center
    for a in rng.uniform(0, 2 * np.pi, count):
        ri = r_in * rng.uniform(0.85, 1.15)
        ro = r_out * rng.uniform(0.6, 1.1)
        dx, dy = np.cos(a), np.sin(a)
        p = rng.uniform(*pressure)
        L = (ro - ri) / sheet.ppm0
        sheet.stroke([(cx + dx * ri, cy + dy * ri, p), (cx + dx * ro, cy + dy * ro, p)], pen=pen,
                     taper=(L * 0.3, L * 0.6), seed=int(rng.integers(1 << 30)))
    return count


def layout(width, height, rows, margin_mm=10.0, gutter_mm=(3.0, 6.0), px_per_mm=8.0, slant=None):
    """Panel rectangles for a page. rows -- list of (height_weight, [width_weights...]).

    gutter_mm -- (between columns, between rows): manga uses narrower vertical gutters than horizontal ones.
    slant     -- optional {(row, col): (dx_top, dx_bottom)} to slant a panel's right edge (base px).
    Returns a list of rows of polygons [(x, y) x 4].
    """
    m = margin_mm * px_per_mm
    gx, gy = gutter_mm[0] * px_per_mm, gutter_mm[1] * px_per_mm
    hw = np.array([r[0] for r in rows], float)
    avail_h = height - 2 * m - gy * (len(rows) - 1)
    hs = avail_h * hw / hw.sum()
    out = []
    y = m
    for ri, (_, cols) in enumerate(rows):
        cw = np.array(cols, float)
        avail_w = width - 2 * m - gx * (len(cols) - 1)
        ws = avail_w * cw / cw.sum()
        x = m
        row = []
        for ci, w in enumerate(ws):
            poly = [(x, y), (x + w, y), (x + w, y + hs[ri]), (x, y + hs[ri])]
            row.append(poly)
            x += w + gx
        out.append(row)
        y += hs[ri] + gy
    if slant:
        for (ri, ci), (dt, db) in slant.items():
            p = out[ri][ci]
            p[1] = (p[1][0] + dt, p[1][1])
            p[2] = (p[2][0] + db, p[2][1])
            if ci + 1 < len(out[ri]):
                q = out[ri][ci + 1]
                q[0] = (q[0][0] + dt, q[0][1])
                q[3] = (q[3][0] + db, q[3][1])
    return out


def border(sheet, poly, width_mm=0.8):
    """Ruled panel border: a crisp closed outline of constant width, square corners (drawn 4x and downsampled)."""
    sheet._log({'op': 'border', 'poly': [[float(x), float(y)] for x, y in poly], 'width_mm': width_mm})
    sheet._n += 1
    _border(sheet, poly, width_mm)


def _border(sheet, poly, width_mm):
    ss = 4
    k = sheet.k * ss
    P = np.array(poly, float) * k
    w = width_mm * sheet.ppm * ss
    x0, y0 = np.floor(P.min(0) - w).astype(int)
    x1, y1 = np.ceil(P.max(0) + w).astype(int)
    im = Image.new('L', (x1 - x0, y1 - y0), 0)
    pts = [(x - x0, y - y0) for x, y in P]
    ImageDraw.Draw(im).line(pts + [pts[0], pts[1]], fill=255, width=max(1, int(round(w))), joint='curve')
    im = im.resize(((x1 - x0) // ss, (y1 - y0) // ss), Image.BOX)
    was = sheet._replaying
    sheet._replaying = True
    try:
        sheet._clips, saved = [], sheet._clips
        sheet._put(np.asarray(im, np.float32) / 255.0, x0 // ss, y0 // ss, 'ink')
    finally:
        sheet._clips = saved
        sheet._replaying = was


FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
# lettering faces bundled with the engine (SIL Open Font License, see fonts/OFL-ComicNeue.txt):
# 'dialogue' for balloons, 'caption' for narration boxes, as English manga lettering does
FACES = {'dialogue': 'ComicNeue-Bold.ttf', 'caption': 'ComicNeue-BoldItalic.ttf'}


def _font(size_px, bold=True, face=None):
    names = ['DejaVuSans-Bold.ttf', 'FreeSansBold.ttf'] if bold else ['DejaVuSans.ttf', 'FreeSans.ttf']
    if face:
        names = [os.path.join(FONTS, FACES.get(face, face))] + names
    # the bundled face last, so a system without DejaVu/FreeSans (Windows, macOS) still letters at the asked size
    names.append(os.path.join(FONTS, FACES['dialogue']))
    for n in names:
        try:
            return ImageFont.truetype(n, int(size_px))
        except OSError:
            continue
    return ImageFont.load_default(int(size_px))


def caption(sheet, xy, text, size_mm=3.6, pad_mm=1.6, box=True, bold=True, face=None):
    """Narration box with lettering. xy = top-left (base px). Lettering is rendered as a raster and journaled."""
    k = sheet.k
    size = size_mm * sheet.ppm
    f = _font(size, bold, face)
    lines = text.split('\n')
    tmp = ImageDraw.Draw(Image.new('L', (1, 1)))
    ws = [tmp.textbbox((0, 0), ln, font=f) for ln in lines]
    tw = max(b[2] - b[0] for b in ws)
    lh = size * 1.25
    th = lh * len(lines)
    pad = pad_mm * sheet.ppm
    bw, bh = int(tw + 2 * pad), int(th + 2 * pad)
    x0, y0 = int(xy[0] * k), int(xy[1] * k)
    sheet._log({'op': 'caption', 'xy': list(xy), 'text': text, 'size_mm': size_mm, 'pad_mm': pad_mm,
                'box': box, 'bold': bold, 'face': face})
    sheet._n += 1
    was = sheet._replaying
    sheet._replaying = True
    try:
        if box:
            white = np.ones((bh, bw), np.float32)
            sheet._put(white, x0, y0, 'white')
            ring = np.zeros((bh, bw), np.float32)
            t = max(1, int(round(0.3 * sheet.ppm)))
            ring[:t, :] = ring[-t:, :] = 1
            ring[:, :t] = ring[:, -t:] = 1
            sheet._put(ring, x0, y0, 'ink')
        im = Image.new('L', (bw, bh), 0)
        dr = ImageDraw.Draw(im)
        for i, ln in enumerate(lines):
            dr.text((pad, pad + i * lh), ln, font=f, fill=255)
        sheet._put(np.asarray(im, np.float32) / 255, x0, y0, 'ink')
    finally:
        sheet._replaying = was
    return (xy[0], xy[1], xy[0] + bw / k, xy[1] + bh / k)


def halo(sheet, figure, width_px=None, width_mm=1.2):
    """A white gap round a figure (kuchinuki), as published pages cut a figure off from a patterned or toned
    background: paper `width` wide just outside the figure's mask. Lay the background first, then the halo, then
    the figure's outline over its edge. `figure` is a bool mask at the sheet's size."""
    import numpy as np
    from scipy import ndimage as nd
    w = width_px if width_px is not None else width_mm * sheet.ppm
    f = np.asarray(figure, bool)
    ring = (nd.distance_transform_edt(~f) <= w) & ~f
    if ring.any():
        sheet.fill(ring, mode='white')
    return ring
