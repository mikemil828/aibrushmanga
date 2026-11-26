"""Automated checks for the manga ink engine.

Run from the repo root:  python3 -m pytest manga/tests -q
They draw small sheets only, so the whole file runs in a few seconds.
"""
import os

import numpy as np
import pytest
from scipy import ndimage as nd

from manga import Sheet, tone, hatch, effects, trace
from manga.pens import get as get_pen
from manga.replay import replay

PPM = 8.0


def blank(w=400, h=300, **kw):
    return Sheet(w, h, px_per_mm=PPM, seed=1, **kw)


# ---------------------------------------------------------------- strokes

def test_stroke_draws_along_its_path_only():
    sh = blank()
    sh.stroke([(50, 150), (350, 150)], pen='g_pen', pressure=0.8, seed=2)
    ink = sh.ink
    assert ink[150, 120:280].min() > 0.9, 'the middle of the line should be solid ink'
    assert ink[:100].max() == 0 and ink[200:].max() == 0, 'no ink far from the line'
    assert sh.stats['strokes'] == 1


def test_stroke_width_matches_pen_at_any_resolution():
    """Widths are in mm, so the same stroke measures the same in mm at 8 and 16 px/mm."""
    widths = []
    for ppm in (8.0, 16.0):
        sh = Sheet(int(50 * ppm), int(30 * ppm), px_per_mm=ppm, seed=1)
        y = 15 * ppm
        sh.stroke([(5 * ppm, y), (45 * ppm, y)], pen='fineliner', pressure=1.0, seed=3)
        col = sh.ink[:, int(25 * ppm)]
        widths.append((col > 0.5).sum() / ppm)
    expect = get_pen('fineliner').width_mm(1.0)
    for w in widths:
        assert w == pytest.approx(expect, abs=0.2)
    assert widths[0] == pytest.approx(widths[1], abs=0.15)


def test_more_pressure_makes_a_wider_line():
    sh = blank()
    sh.stroke([(50, 80), (350, 80)], pen='g_pen', pressure=0.2, seed=4)
    sh.stroke([(50, 220), (350, 220)], pen='g_pen', pressure=1.0, seed=4)
    thin_w = (sh.ink[30:130, 200] > 0.5).sum()
    thick_w = (sh.ink[170:270, 200] > 0.5).sum()
    assert thick_w > 2 * thin_w


def test_taper_thins_the_ends():
    sh = blank()
    sh.stroke([(50, 150), (350, 150)], pen='g_pen', pressure=0.9, seed=5)
    mid = (sh.ink[100:200, 200] > 0.5).sum()
    end = (sh.ink[100:200, 56] > 0.5).sum()
    assert end < mid


def test_white_mode_scrapes_ink_away():
    sh = blank()
    sh.fill(np.ones((300, 400), bool))
    sh.stroke([(50, 150), (350, 150)], pen='white', pressure=1.0, mode='white', seed=6)
    assert sh.ink[150, 200] < 0.1
    assert sh.ink[20, 200] == pytest.approx(1.0)


def test_clip_keeps_ink_inside_the_panel():
    sh = blank()
    m = np.zeros((300, 400), bool)
    m[:, :200] = True
    with sh.clip(m):
        sh.stroke([(20, 150), (380, 150)], pen='marker', pressure=1.0, seed=7)
    assert sh.ink[150, 100] > 0.9
    assert sh.ink[:, 202:].max() == 0


# ---------------------------------------------------------------- screentone

@pytest.mark.parametrize('kind', ['dot', 'line', 'noise'])
@pytest.mark.parametrize('level', [0.15, 0.3, 0.6])
def test_flat_tone_has_the_right_coverage(kind, level):
    sh = blank(600, 600)
    tone(sh, np.ones((600, 600), bool), level=level, lpi=42, kind=kind, seed=8)
    cov = sh.ink[50:550, 50:550].mean()
    assert cov == pytest.approx(level, abs=0.05)


def test_tone_stays_inside_its_mask():
    sh = blank()
    m = np.zeros((300, 400), bool)
    m[100:200, 100:300] = True
    tone(sh, m, level=0.4)
    assert sh.ink[~nd.binary_dilation(m, iterations=1)].max() == 0
    assert sh.ink[m].mean() == pytest.approx(0.4, abs=0.07)


def test_tone_gradient_goes_from_light_to_dark():
    sh = blank(600, 300)
    tone(sh, np.ones((300, 600), bool), level=(0.1, 0.7, (0, 0), (600, 0)))
    left, right = sh.ink[:, 20:120].mean(), sh.ink[:, 480:580].mean()
    assert left < 0.2 and right > 0.55


def test_tone_scratch_removes_tone():
    sh = blank()
    full = np.ones((300, 400), bool)
    tone(sh, full, level=0.5)
    m = np.zeros((300, 400), bool)
    m[100:200, 100:300] = True
    tone(sh, m, level=1.0, mode='white')
    assert sh.ink[110:190, 110:290].max() < 0.05


# ---------------------------------------------------------------- hatching, tracing, layout

def test_hatching_is_darker_where_the_tone_is_darker():
    sh = blank(600, 300)
    t = np.zeros((300, 600), np.float32)
    t[:, :300] = 0.25
    t[:, 300:] = 0.8
    n = hatch(sh, t, np.ones((300, 600), bool), seed=9)
    assert n > 0
    assert sh.ink[20:280, 320:580].mean() > 1.5 * sh.ink[20:280, 20:280].mean()


def test_ink_line_art_draws_a_traced_ring_as_strokes():
    yy, xx = np.mgrid[:300, :400]
    r = np.hypot(xx - 200, yy - 150)
    ring = (r > 80) & (r < 83)
    sh = blank()
    n, _ = trace.ink_line_art(sh, ring, pens=('maru', 'g_pen'), seed=10)
    assert 1 <= n <= 12, 'one ring should become a few long strokes, not many fragments'
    near = nd.binary_dilation(ring, iterations=4)
    assert sh.ink[ring].mean() > 0.5, 'the ring is inked'
    assert sh.ink[~near].max() < 0.05, 'nothing drawn away from the ring'


def test_layout_panels_fit_the_page_without_overlap():
    W, H = 2056, 2912
    rows = effects.layout(W, H, [(1.0, [1, 1]), (1.4, [1, 2, 1])], margin_mm=10, px_per_mm=PPM)
    boxes = []
    for row in rows:
        for p in row:
            xs, ys = [q[0] for q in p], [q[1] for q in p]
            assert min(xs) >= 10 * PPM - 1 and max(xs) <= W - 10 * PPM + 1
            assert min(ys) >= 10 * PPM - 1 and max(ys) <= H - 10 * PPM + 1
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            overlap = min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])
            assert not overlap


def test_lettering_uses_the_bundled_font():
    f = effects._font(40, face='dialogue')
    assert os.path.basename(getattr(f, 'path', '')) == 'ComicNeue-Bold.ttf'
    sh = blank()
    effects.caption(sh, (20, 20), 'HEY!', size_mm=5, face='dialogue')
    assert sh.ink[20:120, 20:250].sum() > 100


# ---------------------------------------------------------------- journal and replay

def _draw_everything(sh):
    sh.stroke([(40, 60), (200, 90), (360, 60)], pen='g_pen', pressure=0.8, seed=11)
    sh.stroke([(40, 120), (360, 140)], pen='brush', pressure=0.9, seed=12)
    m = np.zeros((300, 400), bool)
    m[160:280, 20:200] = True
    with sh.clip(m):
        tone(sh, np.ones((300, 400), bool), level=np.linspace(0.1, 0.6, 400, dtype=np.float32)[None].repeat(300, 0))
        sh.stroke([(0, 220), (400, 220)], pen='white', pressure=0.8, mode='white', seed=13)
    sh.fill([(220, 170), (380, 170), (380, 280), (220, 280)])
    hatch(sh, np.full((300, 400), 0.5, np.float32), [(230, 180), (370, 180), (370, 270), (230, 270)], seed=14)
    effects.border(sh, [(5, 5), (395, 5), (395, 295), (5, 295)], 0.8)
    effects.caption(sh, (230, 20), 'NOW.', size_mm=3.5, face='caption')


def test_replay_matches_the_live_sheet(tmp_path):
    j = str(tmp_path / 'page.jsonl')
    sh = blank(journal=j)
    _draw_everything(sh)
    live = sh.ink.copy()
    sh.close()
    again = replay(j)
    assert again.ink.shape == live.shape
    # the journal keeps stroke points to 0.01 px, so tiny edge differences are expected;
    # they must stay within one grey level of the 8-bit PNG
    assert np.abs(again.ink - live).max() < 1 / 255
    png_diff = np.abs(np.asarray(again.image(), int) - np.asarray(sh.image(), int))
    assert png_diff.max() <= 1
    assert again.stats['strokes'] == sh.stats['strokes']


def test_replay_at_double_scale_matches_when_shrunk(tmp_path):
    j = str(tmp_path / 'page.jsonl')
    sh = blank(journal=j)
    _draw_everything(sh)
    live = sh.ink.copy()
    sh.close()
    big = replay(j, scale=2.0)
    assert big.ink.shape == (600, 800)
    small = big.ink.reshape(300, 2, 400, 2).mean((1, 3))
    # blur both so the finer screen dots at x2 compare as the tone value, not dot for dot
    a, b = nd.gaussian_filter(live, 3), nd.gaussian_filter(small, 3)
    assert np.abs(a - b).mean() < 0.03
