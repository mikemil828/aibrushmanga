"""The two bushes behind her tail in the tall panel, drawn as an inker would: black leaf masses with leaf-shaped
light breaking through on top, one mid tone for the lit leaves, and a light scalloped outline round the clump.

Traced, they broke into black specks; here the reference's own leaf shapes are kept, but posterized into three
values at the page's scale and cleaned of anything smaller than a leaf."""
import numpy as np
from scipy import ndimage as nd
from PIL import Image, ImageDraw
from manga import tone

# where the bushes are (ref px): the tall one left of the tail's curl, the low one by the water's edge
AREA = [(690, 812), (912, 812), (912, 895), (925, 895), (925, 1045), (690, 1045)]   # right of x=912: the pole
BLACK, MID, LIT = 0.28, 0.55, 0.80   # reference grey (0 black .. 1 white): ink / dark tone / light tone / paper


def bush_mask(rgb, mat, to_page, s, box):
    X0, Y0, X1, Y1 = box
    im = Image.new('L', (X1 - X0, Y1 - Y0), 0)
    ImageDraw.Draw(im).polygon([tuple(np.array(to_page(*p)) - (X0, Y0)) for p in AREA], fill=1)
    area = np.asarray(im, bool)
    g = nd.gaussian_filter(rgb.mean(2), 1.1 * s)
    fig = nd.binary_dilation(mat > 0, iterations=int(2 * s))
    # the clumps are where the reference is dark; the lit leaves round them belong to them, the sea does not
    seed = (g < 0.36) & area & ~fig
    seed = nd.binary_opening(seed, iterations=int(1.5 * s))
    mass = nd.binary_dilation(seed, iterations=int(5 * s)) & (g < LIT + 0.05) & area & ~fig
    # the tall bush is dark clover leaves against the sky: no lit fringe round it
    yy = np.arange(mass.shape[0])[:, None] + Y0
    upper = yy < to_page(0, 952)[1]
    mass &= ~upper | (g < 0.48)
    # round, not stair-stepped: leaves are lobes
    mass = nd.gaussian_filter(mass.astype(np.float32), 1.6 * s) > 0.5
    lab, n = nd.label(mass)
    if n:
        sz = nd.sum(mass, lab, np.arange(1, n + 1))
        mass = np.isin(lab, 1 + np.nonzero(sz >= 400 * s * s)[0])
    return mass & ~fig, g, fig, upper


def draw_bushes(sh, rgb, mat, to_page, s, box, lpi):
    X0, Y0, X1, Y1 = box
    mass, g, fig, upper = bush_mask(rgb, mat, to_page, s, box)
    if not mass.any():
        return mass

    def place(a):
        out = np.zeros((sh.H0, sh.W0) if hasattr(sh, 'H0') else (sh.H // sh.k, sh.W // sh.k), bool)
        out[Y0:Y1, X0:X1] = a
        return out

    sh.fill(place(mass), mode='white')
    # tall bush: black leaves, the paper showing between them; low bush: lit leaves in two tones over black
    tone(sh, place(mass & ~upper & (g >= BLACK) & (g < MID)), level=0.45, lpi=lpi)
    tone(sh, place(mass & (g >= MID) & (g < LIT)), level=0.18, lpi=lpi)
    black = mass & (g < np.where(upper, 0.40, BLACK))
    black = nd.gaussian_filter(black.astype(np.float32), 0.9 * s) > 0.5
    lab, n = nd.label(black)
    if n:
        sz = nd.sum(black, lab, np.arange(1, n + 1))
        black = np.isin(lab, 1 + np.nonzero(sz >= 30 * s * s)[0])
    # pinholes of light inside the black read as noise: only real gaps between leaves stay open
    holes = nd.binary_fill_holes(black) & ~black
    lab, n = nd.label(holes)
    if n:
        sz = nd.sum(holes, lab, np.arange(1, n + 1))
        black |= np.isin(lab, 1 + np.nonzero(sz < 40 * s * s)[0])
    sh.fill(place(black))
    # a light outline round the clump, broken where black already closes it
    edge = mass & ~nd.binary_erosion(mass, iterations=max(1, int(0.6 * s)))
    sh.fill(place(edge & ~black))
    return mass


def area_zone(to_page, s, box):
    """The bushes' area (box-local): traced lines there are left out, the bushes are drawn by hand."""
    X0, Y0, X1, Y1 = box
    im = Image.new('L', (X1 - X0, Y1 - Y0), 0)
    ImageDraw.Draw(im).polygon([tuple(np.array(to_page(*p)) - (X0, Y0)) for p in AREA], fill=1)
    return np.asarray(im, bool)
