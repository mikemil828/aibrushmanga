"""Checks for the drawing helpers: Frame, nature, water, prep.ink_contour, tools."""
import numpy as np
import pytest
from scipy import ndimage as nd

from manga import Sheet, Frame, nature, water, prep, tools


def test_frame_maps_points_and_rasters():
    fr = Frame.fit((400, 300), (200, 150), box=(0, 0, 400, 300))
    assert fr.P(10, 20) == (20.0, 40.0)
    m = np.zeros((150, 200), bool); m[50:100, 50:100] = True
    big = fr.upm(m)
    assert big.shape == (300, 400)
    assert big[150, 150] and not big[50, 50]
    assert big.sum() == pytest.approx(4 * m.sum(), rel=0.05)


def test_frame_lay_tones_only_the_mask():
    fr = Frame((400, 300), (200, 150), (0, 0), 2.0)
    sh = Sheet(400, 300, px_per_mm=8)
    m = np.zeros((150, 200), bool); m[20:130, 20:180] = True
    fr.lay(sh, m, 0.4)
    inside = fr.upm(nd.binary_erosion(m, iterations=3))
    assert sh.ink[inside].mean() == pytest.approx(0.4, abs=0.07)
    assert sh.ink[~fr.upm(nd.binary_dilation(m, iterations=2))].max() == 0


def test_frame_poly_and_line_draw():
    fr = Frame((400, 300), (200, 150), (0, 0), 2.0)
    sh = Sheet(400, 300, px_per_mm=8)
    sh.fill(fr.poly([(50, 50), (150, 50), (150, 100), (50, 100)], smooth=0))
    assert sh.ink[150, 200] == 1 and sh.ink[20, 20] == 0
    fr.line(sh, [(10, 130), (100, 125), (190, 130)], pen='g_pen', pressure=0.6, mode='white')
    assert sh.ink[255, 200] == 0 or sh.ink[250:262, 200].min() < 0.5


def test_edge_grass_grows_on_edges_and_more_in_front():
    sh = Sheet(600, 600, px_per_mm=8)
    bank = np.zeros((600, 600), bool); bank[:, 250:350] = True
    n = nature.edge_grass(sh, bank, lambda y: y / 600, rng=np.random.default_rng(1))
    assert n > 50
    top, bottom = sh.ink[:200].sum(), sh.ink[400:].sum()
    assert bottom > 1.5 * top, 'nearer grass should be heavier'
    assert sh.ink[:, 290:310].sum() < 0.2 * sh.ink.sum(), 'blades grow from the edges, not the middle'


def test_depth_below_zero_skips():
    sh = Sheet(300, 300, px_per_mm=8)
    bank = np.zeros((300, 300), bool); bank[:, 100:200] = True
    nature.edge_grass(sh, bank, lambda y: (y - 150) / 150, rng=np.random.default_rng(2))
    nature.tufts(sh, bank, lambda y: (y - 150) / 150, rng=np.random.default_rng(3), prob=(0.01, 0.02))
    assert sh.ink[:120].sum() == 0


def test_seedlings_draw_one_plant_per_blob():
    sh = Sheet(400, 300, px_per_mm=8)
    blobs = np.zeros((300, 400), bool)
    for x in (80, 200, 320):
        blobs[150:180, x - 4:x + 4] = True
    n = nature.seedlings(sh, blobs, lambda y: 1.0)
    assert 9 <= n <= 3 * 8
    for x in (80, 200, 320):
        assert sh.ink[140:185, x - 15:x + 15].sum() > 20
    assert sh.ink[:, 120:160].sum() == 0


def test_fill_under_removes_halos():
    L = np.full((100, 100), 0.9, np.float32)
    obj = np.zeros((100, 100), bool); obj[40:60, 45:55] = True
    L[obj] = 0.1
    Lw, grown = water.fill_under(L, np.ones_like(obj), obj)
    assert Lw[obj].min() > 0.85
    assert grown.sum() > obj.sum()


def test_calm_level_ignores_vertical_streaks():
    L = np.full((100, 200), 0.7, np.float32)
    L[:, ::10] = 0.2                              # rows of dark seedlings
    lv = water.calm_level(L)
    assert lv.std() < 0.02


def test_mirror_flips_and_squashes():
    src = np.zeros((200, 50), bool); src[40:100, 20:30] = True       # a post, base at row 100
    out = water.mirror(src, src_line=100, dst_line=110, squash=0.5, rows=(100, 200))
    ys = np.nonzero(out[:, 25])[0]
    assert ys.min() >= 109 and ys.max() <= 141, 'a 60 px post reflects as about 30 px below the line'


def test_ripples_stay_on_the_mask():
    sh = Sheet(300, 200, px_per_mm=8)
    sh.fill(np.ones((200, 300), bool))
    m = np.zeros((200, 300), bool); m[50:150, 100:200] = True
    n = water.ripples(sh, m, rows=(50, 150), rng=np.random.default_rng(4))
    assert n > 5
    assert sh.ink[:, :90].min() == 1 and sh.ink[m].min() < 0.5


def test_ink_contour_selects_and_splits():
    yy, xx = np.mgrid[:300, :300]
    f = (np.hypot(xx - 150, yy - 150) < 100).astype(np.float32)
    sh = Sheet(300, 300, px_per_mm=8)
    n = prep.ink_contour(sh, nd.gaussian_filter(f, 1), keep=lambda Q: Q[:, 1] < 150, piece=20)
    assert n >= 2
    assert sh.ink[:140].sum() > 0 and sh.ink[160:].sum() == 0


def test_tools_gridcrop_fragment_compare(tmp_path):
    from PIL import Image
    im = Image.new('RGB', (200, 100), 'white')
    g = tools.gridcrop(im, (0, 0, 100, 50), step=10, scale=3)
    assert g.size == (300, 150)
    c = tools.fragment(lambda sh: sh.fill(np.ones((100, 200), bool)), (200, 100), (10, 10, 60, 40), zoom=2)
    assert c.size == (100, 60) and np.asarray(c.convert('L')).mean() < 50
    cmp_ = tools.compare(c, im, (0, 0, 100, 60), (0, 0, 100, 60), height=60)
    assert cmp_.size[1] == 60
