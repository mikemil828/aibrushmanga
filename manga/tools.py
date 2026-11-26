"""Working aids for drawing from a reference: read coordinates, test a fragment, compare with the reference.

    gridcrop('ref.png', (620, 960, 760, 1240), step=10, scale=4, out='g_fig.png')   # read polygon points off it
    fragment(lambda sh: walker(sh), (W, H), box=(1000, 1500, 1350, 2150), out='wt.png', zoom=2)
    compare('panel.png', 'ref.png', box_page, box_ref, out='cmp.png')               # drawn | reference
"""
import numpy as np
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
