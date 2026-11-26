import numpy as np
from PIL import Image
from scipy import ndimage as nd
g = np.asarray(Image.open('../ref.png').convert('L'), np.float32) / 255
d = nd.gaussian_filter(1 - g, 1.5)
lines = np.load('../lines.npy'); rough = np.load('../hair.npy'); mat = np.load('mat.npy')
lab, n = nd.label(~nd.binary_dilation(lines, iterations=1))
idx = np.arange(1, n + 1)
frac = nd.mean(rough, lab, idx); md = nd.mean(d, lab, idx)
keep = (frac > 0.5) & (md > 0.5)
hair = np.concatenate([[False], keep])[lab]
hair = nd.binary_closing(hair, iterations=2)           # close over the strand lines themselves
hair &= nd.binary_dilation(rough, iterations=2)
from PIL import ImageDraw
_f = Image.new('L', (g.shape[1], g.shape[0]), 0)
ImageDraw.Draw(_f).polygon([(2*x, 2*y) for x, y in [(186,175),(215,140),(255,128),(300,140),(322,175),(322,215),(305,255),(295,300),(200,300),(190,240)]], fill=1)
face = np.asarray(_f, bool)
hair |= nd.binary_opening(rough & (d > 0.62) & ~face, iterations=2)
hl, hn = nd.label(nd.binary_fill_holes(hair) & ~hair)
hs = nd.sum(np.ones_like(hl), hl, np.arange(1, hn + 1))
hair |= np.isin(hl, 1 + np.nonzero(hs < 5000)[0])
mat[(mat == 4) & ~hair] = 1          # was hair, now skin (re-classified below if light)
mat[hair & (mat > 0)] = 4
np.save('mat.npy', mat)
rgb = np.stack([g, g, g], 2); rgb[hair] = rgb[hair] * 0.4 + 0.6 * np.array([1, 0.2, 0.2])
Image.fromarray((rgb * 255).astype(np.uint8)).crop((220, 100, 780, 900)).save('hairfix.png')
# the crown's shine was read as skin: above the forehead everything inside the head is hair
crown = np.zeros_like(hair); crown[:265, 330:680] = True
mat[(mat > 0) & (mat != 4) & crown & nd.binary_fill_holes(mat == 4)] = 4
np.save('mat.npy', mat)
# forehead skin showing between the bangs: lighter than any lock there, so it is skin, not hair
gg = nd.gaussian_filter(g, 1.0)
fh = (mat == 4) & face & (gg > 0.38) & (np.arange(g.shape[0])[:, None] > 235)
fh = nd.binary_opening(fh, iterations=1)
mat[fh] = 1
np.save('mat.npy', mat)
