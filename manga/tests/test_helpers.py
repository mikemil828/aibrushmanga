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


def test_tools_pin_crops_round_the_comment(tmp_path):
    from PIL import Image
    page = Image.new('RGB', (1000, 2000), (255, 255, 255))
    c, box = tools.pin(page, 50, 25, size=200)
    assert box == (400, 400, 600, 600) and c.size == (200, 200)
    ref = Image.new('RGB', (500, 1000), (0, 0, 0))
    c2, _ = tools.pin(page, 50, 25, size=200, ref=ref, to_ref=lambda x, y: (x / 2, y / 2), height=100,
                      out=str(tmp_path / 'p.png'))
    assert c2.height == 100 and (tmp_path / 'p.png').exists()


def test_tools_cut_parts_finds_a_sawn_off_finger():
    m = np.zeros((100, 100), bool)
    m[40:90, 40:90] = True                       # palm
    m[60:66, 10:40] = True                       # thumb reaching left
    kept = m.copy(); kept[:, :30] = False        # a cut at x=30 takes the thumb's end
    parts = tools.cut_parts(m, kept, min_px=10)
    assert len(parts) == 1
    assert parts[0]['lost'] == 6 * 20 and parts[0]['bbox'] == (10, 60, 30, 66)
    assert tools.cut_parts(m, m) == []


def test_face_closed_eye_and_smile_ink_where_asked():
    from manga import face
    sh = Sheet(400, 300, px_per_mm=8)
    line = face.closed_eye(sh, (100, 100), (200, 110), to_page=lambda x, y: (x, y), sag=10)
    assert sh.ink[95:125, 100:200].sum() > 100
    assert sh.ink[:80].sum() < 1                 # nothing far above the eye
    assert line[len(line) // 2][1] > 105         # the lid bows downward
    face.closed_smile(sh, [(120, 200), (150, 212), (180, 202)], to_page=lambda x, y: (x, y))
    assert sh.ink[195:230, 110:190].sum() > 30


def _disc(shape, c, r):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < r * r


def test_face_brow_is_solid_tapered_and_measurable():
    from manga import face
    sh = Sheet(400, 200, px_per_mm=8)
    pts = [(100, 100), (150, 90), (200, 88), (250, 95), (300, 110)]
    face.brow(sh, pts, [6, 12, 14, 10, 4])
    assert sh.ink[83:93, 195:205].mean() > 0.9             # solid in the middle
    assert sh.ink[:, 100:104].sum() < sh.ink[:, 198:202].sum() * 0.4   # pointed at the end
    got, wd = face.ribbon_of(sh.ink > 0.5, step=10)
    assert len(got) > 10 and 9 < wd.max() < 18              # the measured ribbon matches what was drawn
    assert abs(got[:, 0].min() - 100) < 12 and abs(got[:, 0].max() - 300) < 12


def test_face_iris_layers_pupil_and_lights():
    from manga import face
    sh = Sheet(300, 300, px_per_mm=8)
    m = _disc((300, 300), (150, 150), 60)
    yy = np.mgrid[:300, :300][0]
    lv = np.where(yy < 130, 0.86, 0.3).astype(np.float32) * m
    pupil = _disc((300, 300), (150, 150), 14)
    light = _disc((300, 300), (125, 120), 9)
    face.iris(sh, m, lv, pupil=pupil, lights=light, seed=1)
    assert sh.ink[100:115, 150:175].mean() > 0.95           # the layer under the lid is solid black
    assert 0.15 < sh.ink[175:195, 110:135].mean() < 0.6     # the lower layer is tone
    assert sh.ink[146:154, 146:154].mean() > 0.95           # pupil
    assert sh.ink[117:123, 122:128].mean() < 0.05           # the catch-light is paper, over everything
    assert sh.ink[:, :80].sum() < 1                         # nothing outside the eye


def test_face_eye_white_shadow_sits_under_the_lash():
    from manga import face
    sh = Sheet(200, 200, px_per_mm=8)
    white = np.zeros((200, 200), bool); white[80:140, 40:160] = True
    lash = np.zeros((200, 200), bool); lash[70:80, 40:160] = True
    s = face.eye_white(sh, white, lash, depth=6)
    assert s[80:86].any() and not s[90:].any()
    assert sh.ink[100:140].sum() < 1


def test_face_open_mouth_black_inside_toned_tongue_paper_teeth():
    from manga import face
    sh = Sheet(300, 300, px_per_mm=8)
    inside = np.zeros((300, 300), bool); inside[120:200, 80:220] = True
    tongue = _disc((300, 300), (150, 190), 30) & inside
    teeth = np.zeros((300, 300), bool); teeth[120:135, 100:200] = True
    face.open_mouth(sh, inside & ~tongue & ~teeth, tongue=tongue, teeth=teeth)
    assert sh.ink[145:155, 90:110].mean() > 0.95
    assert 0.1 < sh.ink[180:195, 140:160].mean() < 0.6
    assert sh.ink[124:131, 120:180].mean() < 0.05


def test_face_open_mouth_shaded_teeth_and_tongue_core():
    from manga import face
    sh = Sheet(300, 300, px_per_mm=8)
    inside = np.zeros((300, 300), bool); inside[120:200, 80:220] = True
    tongue = _disc((300, 300), (150, 190), 30) & inside
    core = _disc((300, 300), (150, 192), 16) & tongue
    teeth = np.zeros((300, 300), bool); teeth[120:135, 100:200] = True
    shade = teeth.copy(); shade[:, 125:] = False
    face.open_mouth(sh, inside & ~tongue & ~teeth, tongue=tongue, teeth=teeth, tongue_level=0.15, tongue_core=core,
                    core_level=0.45, teeth_shade=shade, shade_level=0.3)
    assert sh.ink[124:131, 104:122].mean() > 0.1          # the corner of the teeth is toned
    assert sh.ink[124:131, 140:190].mean() < 0.05         # the rest stays paper
    assert sh.ink[184:198, 142:158].mean() > sh.ink[168:176, 128:136].mean()   # core darker than rim


def test_face_blush_tone_fades_to_its_edge():
    from manga import face
    sh = Sheet(500, 240, px_per_mm=8)
    m = np.zeros((240, 500), bool); m[80:160, 60:440] = True
    face.blush_tone(sh, m, peak=0.3, soft_px=26)
    rows = [sh.ink[y:y + 10, 100:400].mean() for y in (82, 95, 115)]
    assert rows[0] < rows[1] < rows[2]                    # darker toward the middle
    assert abs(sh.ink[82:92, 100:400].mean() - sh.ink[148:158, 100:400].mean()) < 0.05
    assert sh.ink[:70].sum() < 1


def test_face_blush_gloom_and_sweat():
    from manga import face
    sh = Sheet(400, 300, px_per_mm=8)
    face.blush(sh, (100, 150), 50, 25, n=7, seed=1)
    box = sh.ink[115:185, 40:160]
    assert box.sum() > 50 and sh.ink[:, 200:].sum() < 1
    m = np.zeros((300, 400), bool); m[40:120, 220:380] = True
    before = sh.ink.copy()
    face.gloom(sh, m, seed=2)
    assert sh.ink[45:70, 230:370].mean() > sh.ink[95:118, 230:370].mean() > 0.02   # heavier at the top
    assert np.allclose(sh.ink[~m], before[~m])
    face.sweat(sh, (300, 220), 30)
    assert sh.ink[200:245, 285:315].sum() > 10               # an outline
    assert sh.ink[222:228, 302:308].mean() < 0.2             # paper inside


def test_tools_layers_before_after_strip_probe_keep(tmp_path, monkeypatch):
    from manga import tools
    from PIL import Image
    ref = np.zeros((40, 60, 3), np.uint8); ref[10:20, 10:30] = (255, 0, 0)
    drawn = np.full((60, 80, 3), 240, np.uint8); drawn[20:30, 20:40] = 0         # offset (10, 10)
    m = np.zeros((40, 60), bool); m[10:20, 10:30] = True
    c = tools.layers(Image.fromarray(ref), Image.fromarray(drawn), (5, 5, 35, 25), [(m, (0, 0, 255))],
                     offset=(10, 10), zoom=1)
    a = np.asarray(c)
    assert c.size == (90, 20)
    assert tuple(a[10, 15]) == (255, 0, 0) and a[10, 45].max() < 10 and tuple(a[10, 75]) == (0, 0, 255)
    old = Image.fromarray(np.full((50, 50, 3), 255, np.uint8)); new = Image.fromarray(np.zeros((50, 50, 3), np.uint8))
    ba = np.asarray(tools.before_after(old, new, [(0, 0, 20, 10), (10, 10, 30, 30)], width=40, gap=4))
    assert ba.shape == (20 + 4 + 40, 84, 3) and ba[5, 5].min() == 255 and ba[5, 60].max() == 0
    s = tools.strip(new, [(0, 0, 10, 10)] * 5, cols=4)
    assert s.size == (40, 20)
    P = {'red': m, 'none': np.zeros_like(m)}
    assert tools.probe(P, Image.fromarray(ref), [(15, 15), (50, 30)]) == [(15, 15, (255, 0, 0), ['red']),
                                                                          (50, 30, (0, 0, 0), [])]
    monkeypatch.delenv('DUMP', raising=False)
    assert tools.keep(a=m) is None
    p = tools.keep(str(tmp_path / 'd.npz'), a=m)
    assert np.load(p)['a'].sum() == m.sum()


def test_brow_stroke_is_one_thick_to_thin_stroke():
    from manga import face
    m = np.zeros((80, 200), bool)
    m[30:42, 20:70] = True                          # the thick wedge
    m[28:44, 66:72] = True                          # a steep fringe tip lying on its end
    cand = np.zeros_like(m)
    cand[35:37, 70:150] = True                      # the thin drawn tail that continues it
    pts, w, line = face.brow_stroke(m, cand, reach=120)
    assert pts[:, 0].max() > 140                    # it follows the tail: one stroke, not wedge + separate line
    k = int(np.argmax(w))
    assert np.all(np.diff(w[k:]) <= 1e-9)           # and only narrows after its widest point
    assert w.max() < 14 and w[-1] < 4               # the tip did not count as width
    assert line[36, 100] and not line[10, 100]


def test_iris_to_rim_fills_to_the_rim_and_stops_there():
    from manga import face
    yy, xx = np.mgrid[:100, :100]
    r = np.hypot(yy - 50, xx - 50)
    rim = (r > 20) & (r < 24)
    hull = (r < 17) | ((r < 30) & (yy > 70) & (np.abs(xx - 50) < 6))   # short of the rim, and a spill past it
    got = face.iris_to_rim(hull, rim)
    assert got is not None
    assert got[50, 50 + 18] and not got[50 + 27, 50]                   # out to the rim, nothing beyond
    assert face.iris_to_rim(hull, np.zeros_like(rim)) is None or face.iris_to_rim(hull, np.zeros_like(rim)).sum() < 1.6 * hull.sum()


def test_boxes_of_and_run_until(tmp_path):
    from manga import tools
    m = np.zeros((50, 100), bool)
    m[5:10, 5:15] = True; m[30:40, 60:70] = True; m[6:9, 50:60] = True
    bx = tools.boxes_of(m, pad=2)
    assert len(bx) == 3 and bx[0][0] < bx[1][0]
    s = tmp_path / 'draw.py'
    s.write_text("a = 1 + k\n# ---- ink\nraise SystemExit('drawn')\n")
    assert tools.run_until(str(s), '# ---- ink', k=2)['a'] == 3


def test_halo_clears_paper_round_a_figure():
    from manga import effects, tone
    sh = Sheet(200, 200, px_per_mm=8.0)
    tone(sh, np.ones((200, 200), bool), level=0.5, lpi=60)
    fig = np.zeros((200, 200), bool); fig[60:140, 60:140] = True
    ring = effects.halo(sh, fig, width_px=10)
    a = np.asarray(sh.image().convert("L"))
    assert ring[55, 100] and not ring[100, 100] and not ring[40, 100]
    if a is not None:
        assert a[55, 95:105].min() > 200 and a[30:50, 30:50].min() < 120
