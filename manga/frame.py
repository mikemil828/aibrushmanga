"""A reference image placed on the page.

Every drawing from a reference needs the same plumbing: convert reference pixels to page pixels, scale masks and
tone levels up to the page, lay tone from a reference-sized level map, and fill small hand-placed polygons.
`Frame` holds it, so a page script can read coordinates off the reference and draw with them directly.

    fr = Frame.fit((W, H), ref.shape[1::-1], box=(80, 80, 1920, 2840))
    fr.lay(sh, sky_mask, sky_levels)                 # tone from reference-sized arrays
    sh.stroke(fr.pts([(612, 688), (640, 704)]))      # reference coordinates -> page
    sh.fill(fr.poly(HAT))                            # a hand-placed polygon (reference px), smoothed
"""
import numpy as np
from PIL import Image, ImageDraw

from . import geom
from .tone import tone


class Frame:
    def __init__(self, page_size, ref_size, origin=(0.0, 0.0), scale=1.0, lpi=60.0):
        self.W, self.H = int(page_size[0]), int(page_size[1])
        self.RW, self.RH = int(ref_size[0]), int(ref_size[1])
        self.ox, self.oy = float(origin[0]), float(origin[1])
        self.s = float(scale)
        self.lpi = float(lpi)

    @classmethod
    def fit(cls, page_size, ref_size, box, lpi=60.0):
        """Scale the reference to the box's width, top-left at the box's corner (the box's height should match the
        reference's aspect ratio; anything below the box is cut by the panel clip)."""
        x0, y0, x1, _ = box
        return cls(page_size, ref_size, (x0, y0), (x1 - x0) / ref_size[0], lpi)

    # ---- coordinates
    def P(self, x, y):
        """One reference point -> page point."""
        return (self.ox + self.s * x, self.oy + self.s * y)

    def pts(self, seq):
        """A sequence of (x, y) reference points -> list of page points."""
        return [self.P(x, y) for x, y in np.asarray(seq, float)[:, :2]]

    def to_page(self, x, y):
        return self.P(x, y)

    # ---- rasters
    def up(self, a):
        """A reference-sized float array scaled onto the page (zeros outside the frame)."""
        im = Image.fromarray(np.asarray(a, np.float32), 'F').resize((round(self.RW * self.s), round(self.RH * self.s)),
                                                                     Image.BILINEAR)
        out = np.zeros((self.H, self.W), np.float32)
        x0, y0 = int(round(self.ox)), int(round(self.oy))
        h, w = min(im.height, self.H - y0), min(im.width, self.W - x0)
        out[y0:y0 + h, x0:x0 + w] = np.asarray(im)[:h, :w]
        return out

    def upm(self, m):
        """A reference-sized bool mask scaled onto the page."""
        return self.up(np.asarray(m, np.float32)) > 0.5

    def lay(self, sh, mask, level, kind='dot', angle=45.0, lpi=None):
        """Tone over a reference mask; level is a number or a reference-sized array.
        For kind='line', angle=90 gives horizontal lines (still water), 0 vertical ones."""
        m = self.upm(mask)
        lv = np.where(mask, level, 0) if np.ndim(level) else np.where(mask, float(level), 0.0)
        tone(sh, m, level=self.up(lv), lpi=lpi or self.lpi, kind=kind, angle=angle)

    # ---- hand-placed shapes (reference px)
    def poly(self, pts, smooth=6):
        """Page mask of a closed polygon given in reference px; smooth > 0 rounds it through a Catmull-Rom spline."""
        p = np.asarray(pts, float)
        if smooth:
            p = geom.catmull(np.vstack([p, p[:1]]), smooth)
        im = Image.new('L', (self.W, self.H), 0)
        ImageDraw.Draw(im).polygon([self.P(x, y) for x, y in p[:, :2]], fill=1)
        return np.asarray(im, bool)

    def disc(self, c, r):
        """Page mask of a disc, centre and radius in reference px."""
        yy, xx = np.mgrid[:self.H, :self.W]
        cx, cy = self.P(*c)
        return (xx - cx) ** 2 + (yy - cy) ** 2 < (r * self.s) ** 2

    def line(self, sh, pts, per=8, **kw):
        """A stroke through reference points, smoothed through a spline when it has more than two."""
        Q = geom.catmull(np.array(pts, float), per) if len(pts) > 2 else np.array(pts, float)
        sh.stroke(self.pts(Q), **kw)
