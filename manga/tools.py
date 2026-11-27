"""Working aids for drawing from a reference: read coordinates, test a fragment, compare with the reference.

    gridcrop('ref.png', (620, 960, 760, 1240), step=10, scale=4, out='g_fig.png')   # read polygon points off it
    fragment(lambda sh: walker(sh), (W, H), box=(1000, 1500, 1350, 2150), out='wt.png', zoom=2)
    compare('panel.png', 'ref.png', box_page, box_ref, out='cmp.png')               # drawn | reference
    layers('ref.png', 'panel.png', box, [(P['pink'], (255, 120, 160)), (iris, (0, 0, 255))], offset=(88, 88))
    before_after('old.png', 'panel.png', boxes, out='ba.png')                        # what the client sees
    strip('panel.png', eye_boxes, cols=4)                                            # one part of every figure
    probe(P, 'ref.png', [(400, y) for y in range(360, 460, 3)])                      # can colour separate it?
"""
import numpy as np
from scipy import ndimage as nd
from PIL import Image, ImageDraw

from .sheet import Sheet


def _img(a):
    return a if isinstance(a, Image.Image) else Image.open(a)


def gridcrop(img, box, step=50, scale=2, out=None, color=(255, 0, 0)):
    """A crop of `img` enlarged `scale` times with a labelled grid every `step` px in the image's own coordinates,
    for reading polygon points and stroke paths off a reference."""
    im = _img(img).convert('RGB')
    x0, y0, x1, y1 = box
    c = im.crop(box).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    for x in range((x0 // step + 1) * step, x1, step):
        X = (x - x0) * scale
        d.line([(X, 0), (X, c.height)], fill=color)
        d.text((X + 2, 2), str(x), fill=color)
    for y in range((y0 // step + 1) * step, y1, step):
        Y = (y - y0) * scale
        d.line([(0, Y), (c.width, Y)], fill=color)
        d.text((2, Y + 2), str(y), fill=color)
    if out:
        c.save(out)
    return c


def fragment(draw, page_size, box, px_per_mm=8.0, out=None, zoom=1, seed=7):
    """Run one or more drawing functions f(sheet) on a fresh sheet clipped to `box` (page px) and return the crop.
    A few seconds instead of a whole page: try two or three variants of a part side by side before the sheet."""
    sh = Sheet(page_size[0], page_size[1], px_per_mm=px_per_mm, seed=seed)
    x0, y0, x1, y1 = box
    with sh.clip([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]):
        for f in (draw if isinstance(draw, (list, tuple)) else [draw]):
            f(sh)
    c = sh.image().crop(box)
    if zoom != 1:
        c = c.resize((c.width * zoom, c.height * zoom), Image.LANCZOS)
    if out:
        c.save(out)
    return c


def compare(drawn, ref, box_drawn, box_ref, out=None, height=600):
    """The drawn area and the same area of the reference side by side, at the same height."""
    a = _img(drawn).convert('RGB').crop(box_drawn)
    b = _img(ref).convert('RGB').crop(box_ref)
    a = a.resize((round(a.width * height / a.height), height), Image.LANCZOS)
    b = b.resize((round(b.width * height / b.height), height), Image.LANCZOS)
    c = Image.new('RGB', (a.width + b.width + 10, height), (255, 0, 0))
    c.paste(a, (0, 0)); c.paste(b, (a.width + 10, 0))
    if out:
        c.save(out)
    return c


def pin(page, x_pct, y_pct, size=500, ref=None, to_ref=None, out=None, height=600):
    """The area round a review comment pinned at (x_pct, y_pct) percent of the page, and, given `ref` and a
    page->ref mapping `to_ref(x, y)`, the same area of the reference beside it. Returns (image, page box).
    Comments come pinned as "68% from the left, 76% from the top": this is the first look at every one."""
    im = _img(page).convert('RGB')
    cx, cy = im.width * x_pct / 100, im.height * y_pct / 100
    h = size / 2
    box = tuple(int(round(v)) for v in (max(0, cx - h), max(0, cy - h), min(im.width, cx + h), min(im.height, cy + h)))
    if ref is None or to_ref is None:
        c = im.crop(box)
        if out:
            c.save(out)
        return c, box
    (rx0, ry0), (rx1, ry1) = to_ref(box[0], box[1]), to_ref(box[2], box[3])
    rbox = tuple(int(round(v)) for v in (min(rx0, rx1), min(ry0, ry1), max(rx0, rx1), max(ry0, ry1)))
    return compare(im, ref, box, rbox, out=out, height=height), box


def cut_parts(mask, kept, min_px=30):
    """What a cut takes away: `mask` is a figure before the cut, `kept` the part that survives (both bool, same
    shape). Returns one dict per piece of the figure that the cut touches, largest loss first:
    {'lost': px removed, 'kept': px left, 'bbox': (x0, y0, x1, y1) of the removed part}. A piece that keeps most
    of itself but loses its end is usually a finger, a foot or a tail the cut sawed off, and wants a look."""
    from scipy import ndimage as nd
    lab, n = nd.label(mask)
    gone = mask & ~kept
    out = []
    for i, sl in enumerate(nd.find_objects(lab), 1):
        piece = lab[sl] == i
        lost = int((gone[sl] & piece).sum())
        if lost < min_px:
            continue
        ys, xs = np.nonzero(gone[sl] & piece)
        out.append(dict(lost=lost, kept=int((kept[sl] & piece).sum()),
                        bbox=(int(xs.min() + sl[1].start), int(ys.min() + sl[0].start),
                              int(xs.max() + sl[1].start + 1), int(ys.max() + sl[0].start + 1))))
    return sorted(out, key=lambda d: -d['lost'])


# ---- learned on the expressions panel: each round I rewrote the same three throwaway scripts ----

def _crop_scaled(im, box, width):
    c = im.crop(tuple(int(round(v)) for v in box))
    return c.resize((width, max(1, round(c.height * width / c.width))), Image.LANCZOS)


def layers(ref, drawn, box, masks=(), offset=(0, 0), out=None, zoom=3, back=(128, 128, 128)):
    """Why a part came out wrong: the reference, the drawing and the masks behind it, side by side for one `box`
    (reference px). `masks` is a list of (bool array at reference size, (r, g, b)), painted in order over grey, so
    later ones win: e.g. the colour classes from prep, then a mask the script built (iris, teeth, tips).
    `offset` is where reference (0, 0) lands on the drawing (the panel margin). Read the three together: a
    wrong drawing over a right mask is a drawing bug; a wrong mask is a prep or detection bug."""
    x0, y0, x1, y1 = [int(v) for v in box]
    a = np.asarray(_img(ref).convert('RGB'))
    if a.shape[:2] != (masks[0][0].shape if masks else a.shape[:2]):
        a = np.asarray(_img(ref).convert('RGB').resize((masks[0][0].shape[1], masks[0][0].shape[0]), Image.LANCZOS))
    d = np.asarray(_img(drawn).convert('RGB'))
    ox, oy = offset
    m = np.empty((y1 - y0, x1 - x0, 3), np.uint8); m[:] = back
    for mask, col in masks:
        m[np.asarray(mask, bool)[y0:y1, x0:x1]] = col
    row = np.concatenate([a[y0:y1, x0:x1], d[y0 + oy:y1 + oy, x0 + ox:x1 + ox], m], 1)
    c = Image.fromarray(row).resize((row.shape[1] * zoom, row.shape[0] * zoom), Image.NEAREST)
    if out:
        c.save(out)
    return c


def before_after(old, new, boxes, out=None, width=1000, gap=12):
    """One sheet for a review round: for each box (drawing px), the previous drawing on the left and the new one
    on the right. Copy the current output aside before rebuilding and pass it as `old`; this sheet is what goes
    to the client, so they see each change without hunting for it."""
    a, b = _img(old).convert('RGB'), _img(new).convert('RGB')
    rows = [(_crop_scaled(a, bx, width), _crop_scaled(b, bx, width)) for bx in boxes]
    c = Image.new('RGB', (2 * width + gap, sum(r[0].height + gap for r in rows) - gap), 'white')
    y = 0
    for l, r in rows:
        c.paste(l, (0, y)); c.paste(r, (width + gap, y)); y += l.height + gap
    if out:
        c.save(out)
    return c


def strip(img, boxes, cols=4, width=None, out=None):
    """The same part of several figures in one grid (all eyes, all mouths): a fix to one face often breaks
    another, and this shows every face at once. `boxes` are drawing px; `width` scales each tile."""
    im = _img(img).convert('RGB')
    tiles = [im.crop(tuple(int(round(v)) for v in bx)) for bx in boxes]
    if width:
        tiles = [t.resize((width, max(1, round(t.height * width / t.width))), Image.LANCZOS) for t in tiles]
    tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    c = Image.new('RGB', (tw * min(cols, len(tiles)), th * rows), 'white')
    for k, t in enumerate(tiles):
        c.paste(t, ((k % cols) * tw, (k // cols) * th))
    if out:
        c.save(out)
    return c


def probe(parts, img, pts):
    """What the reference has at each point: [(x, y, (r, g, b), [class names])]. `parts` is a dict (or npz) of
    bool masks at the image's size. Sample a line across a feature (down the middle of a mouth, along a faint
    brow) before guessing thresholds: it shows at once whether colour can separate it at all."""
    a = np.asarray(_img(img).convert('RGB'))
    names = list(parts.keys()) if hasattr(parts, 'keys') else list(parts.files)
    if a.shape[:2] != parts[names[0]].shape:
        h, w = parts[names[0]].shape
        a = np.asarray(_img(img).convert('RGB').resize((w, h), Image.LANCZOS))
    out = []
    for x, y in pts:
        x, y = int(round(x)), int(round(y))
        out.append((x, y, tuple(int(v) for v in a[y, x]), [k for k in names if parts[k][y, x]]))
    return out


def keep(path=None, **arrays):
    """Save the masks a drawing script built, for `layers` afterwards. Does nothing unless `path` is given or
    the DUMP environment variable is set, so it can stay in the script: `DUMP=m.npz python3 panel.py`."""
    import os
    path = path or os.environ.get('DUMP')
    if path:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


CLASS_COLOURS = {'bg': (235, 235, 255), 'skin': (255, 225, 195), 'shadow': (205, 155, 120), 'white': (215, 215, 215),
                 'pink': (255, 120, 160), 'blush': (255, 190, 200), 'amber': (255, 170, 0), 'brown': (140, 80, 0),
                 'hair': (200, 60, 60), 'shine': (255, 0, 255), 'jacket': (40, 40, 140), 'line_soft': (120, 120, 255),
                 'line': (0, 0, 255), 'strand': (0, 180, 0), 'black': (0, 0, 0)}


def classmap(parts, box=None, colours=None):
    """The colour classes from prep as one picture: each class in its own colour, later ones in `colours` on
    top (so lines and black win). `box` (x0, y0, x1, y1) crops it. Unknown classes are skipped."""
    colours = colours or CLASS_COLOURS
    names = list(parts.keys()) if hasattr(parts, 'keys') else list(parts.files)
    h, w = parts[names[0]].shape
    x0, y0, x1, y1 = [int(v) for v in (box or (0, 0, w, h))]
    o = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    for k, c in colours.items():
        if k in names:
            o[np.asarray(parts[k], bool)[y0:y1, x0:x1]] = c
    return Image.fromarray(o)


def _ref_at(ref, shape):
    im = _img(ref).convert('RGB')
    return im if im.size == (shape[1], shape[0]) else im.resize((shape[1], shape[0]), Image.LANCZOS)


def triptych(ref, drawn, parts, pts, half=90, offset=(0, 0), zoom=2, out=None):
    """Reference | drawing | colour classes round each point (x, y in reference px, at the classes' size), one
    row per point: the first look at a pinned comment or a missing part. If the classes are right and the drawing
    is wrong, the bug is in drawing; if the classes are wrong, it is in prep or detection. `offset` is where
    reference (0, 0) lands on the drawing (the panel margin)."""
    names = list(parts.keys()) if hasattr(parts, 'keys') else list(parts.files)
    shape = parts[names[0]].shape
    r, d = _ref_at(ref, shape), _img(drawn).convert('RGB')
    hw, ox, oy = int(half * 16 / 9), offset[0], offset[1]
    rows = []
    for x, y in pts:
        b = (int(x) - hw, int(y) - half, int(x) + hw, int(y) + half)
        row = Image.new('RGB', (3 * 2 * hw, 2 * half), 'white')
        row.paste(r.crop(b), (0, 0)); row.paste(d.crop((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)), (2 * hw, 0))
        row.paste(classmap(parts, b), (4 * hw, 0))
        rows.append(row)
    c = Image.new('RGB', (rows[0].width, sum(rw.height for rw in rows)), 'white')
    for k, rw in enumerate(rows):
        c.paste(rw, (0, k * rows[0].height))
    c = c.resize((c.width * zoom, c.height * zoom), Image.NEAREST)
    if out:
        c.save(out)
    return c


def beside(ref, drawn, boxes, offset=(0, 0), size=None, gap=6, out=None):
    """For each box (reference px): the reference and the drawing side by side, stacked down the sheet. With one
    box per figure (the brows of every face, every ear) it is the whole-sheet check: are all of them right, not
    just the one a comment pointed at? `size` is the reference's size when it differs from the image file's."""
    d = _img(drawn).convert('RGB')
    r = _ref_at(ref, (size[1], size[0])) if size else _img(ref).convert('RGB')
    ox, oy = offset
    rows = []
    for bx in boxes:
        b = tuple(int(round(v)) for v in bx)
        a, p = r.crop(b), d.crop((b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy))
        rw = Image.new('RGB', (2 * a.width + gap, a.height), 'white'); rw.paste(a, (0, 0)); rw.paste(p, (a.width + gap, 0))
        rows.append(rw)
    W = max(rw.width for rw in rows)
    c = Image.new('RGB', (W, sum(rw.height + gap for rw in rows) - gap), (255, 0, 0))
    y = 0
    for rw in rows:
        c.paste(rw, (0, y)); y += rw.height + gap
    if out:
        c.save(out)
    return c


def boxes_of(mask, pad=10, min_px=1):
    """A box (x0, y0, x1, y1) round each connected piece of `mask`, `pad` px wider, ordered row by row: feed it
    to `beside` or `strip` to see every iris, every brow, every detected thing at once."""
    lab, n = nd.label(np.asarray(mask, bool))
    out = []
    for k, sl in enumerate(nd.find_objects(lab), 1):
        if (lab[sl] == k).sum() >= min_px:
            out.append((sl[1].start - pad, sl[0].start - pad, sl[1].stop + pad, sl[0].stop + pad))
    h = max((b[3] - b[1] for b in out), default=1)
    return sorted(out, key=lambda b: (b[1] // h, b[0]))


def run_until(script, marker, **env):
    """Run a drawing script up to the line holding `marker` and return its variables, without drawing the rest.
    Detection (brows, irises, mouths) is decided before any ink, so it can be tried in seconds instead of the
    whole build: `ns = tools.run_until('panel.py', '# ---- each brow'); ns['brows']`. Runs in the script's own
    folder, with that folder on sys.path; argv[1] is a throwaway path, so a script that journals to argv[1] leaves
    its real journal alone. `env` adds variables before the run."""
    import os, sys, tempfile
    script = os.path.abspath(script)
    src = open(script).read()
    k = src.find(marker)
    if k < 0:
        raise ValueError(f'{marker!r} not in {script}')
    src = src[:src.rfind('\n', 0, k) + 1]
    here, cwd = os.path.dirname(script), os.getcwd()
    ns = {'__name__': '__run_until__', '__file__': script, **env}
    argv = sys.argv
    sys.path.insert(0, here); os.chdir(here); sys.argv = [script, os.path.join(tempfile.mkdtemp(), 'journal.jsonl')]   # not the real journal
    try:
        exec(compile(src, script, 'exec'), ns)
    finally:
        os.chdir(cwd); sys.path.remove(here); sys.argv = argv
    return ns
