"""Test 3: the figure is inked and toned with care; the background lighter and sparser; the hair as a decision."""
import sys, time, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', '..'))
from PIL import Image
from scipy import ndimage as nd
from manga import Sheet, tone, trace

def draw(hair_mode='beta', out='draw3.png', journal=None, bg_len=18, bg_weight=0.9, bg_tone=0.6):
    ref = np.asarray(Image.open('ref.png').convert('RGB'), np.float32) / 255
    H, W = ref.shape[:2]
    d = 1 - ref.mean(2)
    fig, lines, hair = np.load('fig.npy'), np.load('lines.npy'), np.load('hair.npy')
    near_fig = nd.binary_dilation(fig, iterations=2)
    fig_lines, bg_lines = lines & near_fig, lines & ~near_fig
    sh = Sheet(W, H, px_per_mm=8.0, journal=journal)
    # 1. tone. Figure: stock tones a step lighter in the mid range so lines read. Background: paler again.
    lv = trace.area_tones(d, nd.binary_dilation(lines, iterations=1) | (hair if hair_mode == 'beta' else False))
    lv_fig = np.where(lv > 0.6, lv, lv * 0.8)
    lv = np.where(fig, lv_fig, lv * bg_tone)
    if hair_mode != 'tone':
        lv = np.where(hair, 0, lv)
    pool = np.load('pool.npy')
    lv = np.where(pool, 0.0, lv)
    tone(sh, None, level=lv, lpi=65)
    # pool water: one light tone, ripples scraped back to paper where the reference catches the sun
    tone(sh, pool, level=0.28, lpi=65)
    gw = nd.gaussian_filter(ref.mean(2), 1.2)
    rip = pool & (gw - nd.gaussian_filter(gw, 8) > 0.06)
    sh.fill(nd.binary_opening(rip, iterations=1), mode='white')
    blk = nd.gaussian_filter((nd.gaussian_filter(d, 1.0) > 0.80).astype(np.float32), 1.0) > 0.5
    # 2. white edge round the figure's lines only (background lines sit lightly on their tone)
    sh.fill(nd.binary_dilation(fig_lines, iterations=1) & ~blk & ~hair, mode='white')
    # 3. ink. Figure: maru for fine lines, G-pen for contours. Background: maru only, lighter, no short bits.
    n1, _ = trace.ink_line_art(sh, fig_lines & ~hair if hair_mode == 'beta' else fig_lines, seed=3, weight=1.5)
    n2, _ = trace.ink_line_art(sh, bg_lines, pens=('maru', 'maru'), seed=4, weight=bg_weight, min_len_px=bg_len,
                               min_px=1.0)
    n3 = 0
    if hair_mode == 'beta':
        # 4. hair: solid black, gloss scraped out where the reference shines, strands drawn back in white
        sh.fill(hair)
        gh = nd.gaussian_filter(d, 1.5)
        med = np.median(gh[hair])
        gloss = hair & (gh < med - 0.17)
        gloss = nd.binary_opening(gloss, iterations=1)
        sh.fill(gloss, mode='white')
        strands = lines & nd.binary_erosion(hair, iterations=3) & ~gloss
        P = trace.trace(trace.thin(strands))
        rng = np.random.default_rng(5)
        for pts, a, b in P:
            if len(pts) < 2: continue
            from manga import geom
            L = geom.arclen(pts)[-1]
            if L < 25 or rng.random() < 0.4: continue      # a few chosen strands, not every one
            Q = geom.simplify(pts, 0.8)
            sh.stroke(Q, pen='white', pressure=0.08, mode='white', taper=(0.8, 1.6), seed=int(rng.integers(1 << 30)))
            n3 += 1
    # 5. solid blacks (outside the hair, already handled)
    lab, k = nd.label(blk & ~hair)
    if k:
        sz = nd.sum(blk, lab, range(1, k + 1)); sh.fill(np.isin(lab, 1 + np.nonzero(sz >= 30)[0]))
    sh.png(out)
    if journal: sh.close()
    print(hair_mode, 'strokes fig', n1, 'bg', n2, 'hair', n3, flush=True)
    return np.asarray(sh.image().convert('L'))

if __name__ == '__main__':
    modes = sys.argv[1:] or ['beta']
    outs = [draw(m, out=f'draw3_{m}.png') for m in modes]
    crop = np.concatenate([o[100:760, 240:760] for o in outs], 1)
    Image.fromarray(crop).save('hairtest.png')
