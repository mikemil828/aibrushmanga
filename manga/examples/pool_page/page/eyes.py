"""Eyes and mouth: sclera and teeth are paper, irises get their own tone (materials 5 and 6)."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
g = np.asarray(Image.open('../ref.png').convert('L'), np.float32) / 255
d = nd.gaussian_filter(1 - g, 1.0)
mat = np.load('mat.npy'); H, W = mat.shape
def ell(box):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).ellipse(box, fill=1); return np.asarray(im, bool)
def poly(pts):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon(pts, fill=1); return np.asarray(im, bool)
EYES = [poly([(377, 372), (395, 357), (430, 352), (452, 362), (455, 392), (425, 395), (390, 392)]),
        poly([(535, 380), (555, 370), (590, 372), (612, 388), (612, 412), (580, 414), (545, 405)])]
IRIS = [ell((402, 356, 446, 392)), ell((541, 372, 585, 412))]
MOUTH = poly([(470, 455), (500, 450), (545, 450), (560, 458), (545, 495), (510, 505), (480, 490)])
eye = EYES[0] | EYES[1]
iris = (IRIS[0] | IRIS[1]) & eye
paper = MOUTH & (d < 0.22)          # the eyes themselves are drawn by hand (eyes_hand.py)
mat[paper & (mat > 0)] = 5
np.save('mat.npy', mat)
np.save('blush.npy', np.array([[383, 400, 450, 428], [527, 410, 603, 435]]))
# thin slivers of the scene caught in the figure mask (an umbrella rib across the raised hand): a body part
# is never thinner than ~8 px here, so open the non-hair figure and drop what the opening takes off
from scipy import ndimage as _nd
body = (mat > 0) & (mat != 4)
opened = _nd.binary_opening(body, structure=np.ones((3, 3)), iterations=3)
sliver = body & ~opened
lab, n = _nd.label(sliver)
if n:
    # only slivers that stick out (touch the outside), not thin bits between two body parts
    outside = _nd.binary_dilation(mat == 0, iterations=1)
    touch = _nd.maximum(outside, lab, np.arange(1, n + 1))
    size = _nd.sum(sliver, lab, np.arange(1, n + 1))
    drop = np.concatenate([[False], touch.astype(bool) & (size >= 25)])[lab]
    # only on the side where the scene's umbrella and bushes meet her (raised arm, hand, tail); the ahoge,
    # the hair's edge and the other arm are thin on purpose
    zone = np.zeros_like(drop); zone[:430, 730:] = True
    drop &= zone
    mat[drop] = 0
    np.save('sliver.npy', drop)
np.save('mat.npy', mat)

# the swimsuit's chest diamonds become plain swimsuit
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", ".."))
import eyes_hand as _eh
_sp = Image.new('L', (W, H), 0); ImageDraw.Draw(_sp).polygon(_eh.SUIT_PLAIN, fill=1)
_sp = np.asarray(_sp, bool) & (mat > 0)
mat[_sp & (mat != 4)] = 2
np.save('mat.npy', mat)

# the tail's outer edge as drawn in the reference (its black line): the material map let the dark bushes and the
# umbrella pole above it count as tail, so everything above this arc stops being tail
TAIL_EDGE = [(841.7, 1000), (860, 945), (880, 915), (900, 898.3), (930, 885), (960, 878.3), (990, 876),
             (1020, 877.7), (1050, 882.7), (1075, 890), (1080, 892)]
_tp = Image.new('L', (W, H), 0)
ImageDraw.Draw(_tp).polygon(TAIL_EDGE + [(1080, 700), (841.7, 700)], fill=1)
_above = np.asarray(_tp, bool)
mat[_above & (mat == 3)] = 0
# and the band just under the arc is tail even where the map lost it to the dark background
_tb = Image.new('L', (W, H), 0)
ImageDraw.Draw(_tb).polygon(TAIL_EDGE + [(1080, 905), (1046.7, 905), (1020, 908), (986.7, 920), (960, 936.7),
                                         (940, 966.7), (930, 1000)], fill=1)
mat[np.asarray(_tb, bool) & (mat == 0)] = 3
np.save('mat.npy', mat)

# the waving hand: the umbrella's rim caught under the little finger is not hand, and the middle finger's right side
# had a notch bitten out where the canopy's light line crossed it
_nd2 = __import__('scipy.ndimage', fromlist=['x'])
mat[432:463, 874:916][mat[432:463, 874:916] > 0] = 0
_n = mat[305:352, 806:816]
_n[_n == 0] = 1
# the floor between the fin by her cheek and her palm was read as scales: it is background
_fl = Image.new('L', (W, H), 0)
ImageDraw.Draw(_fl).polygon([(725, 368), (728, 390), (739, 425), (710, 442), (700, 452), (700, 470), (760, 470),
                             (760, 368)], fill=1)
mat[np.asarray(_fl, bool) & (mat == 3)] = 0
# the hand's edge, smoothed at the reference's own scale so each finger keeps an even width
_hr = (slice(250, 470), slice(700, 930))
_hs = _nd2.gaussian_filter((mat[_hr] == 1).astype(np.float32), 2.2) > 0.5
_h = mat[_hr]
_h[_hs & (_h == 0)] = 1
_h[~_hs & (_h == 1)] = 0
# the floor under the waving hand's thumb was read as scales: it is background
mat[465:610, 668:810][mat[465:610, 668:810] == 3] = 0
# the forehead between two bangs, from one bang's black edge to the other's: skin, not hair shine
_ft = Image.new('L', (W, H), 0)
ImageDraw.Draw(_ft).polygon([(457, 241), (448.5, 251), (412, 294), (412, 300), (465, 300), (459.5, 250)], fill=1)
mat[np.asarray(_ft, bool) & (mat == 4)] = 1
# the gap between her left arm and her waist, below the little dash where her side turns, is the pool
# behind her, not body (above the dash it is her side)
_gp = Image.new('L', (W, H), 0)
ImageDraw.Draw(_gp).polygon([(356.5, 1094), (356, 1120), (354, 1160), (355, 1185),
                             (359, 1185), (362, 1170), (365, 1145), (369, 1115), (373, 1094)], fill=1)
mat[np.asarray(_gp, bool)] = 0
# the sky seen between her raised upper arm and her shoulder was read as scales
_ap = Image.new('L', (W, H), 0)
ImageDraw.Draw(_ap).polygon([(660, 560), (805, 540), (820, 575), (840, 615), (860, 655), (876, 696), (860, 698),
                             (800, 652), (730, 611), (660, 611)], fill=1)
mat[np.asarray(_ap, bool) & (mat == 3)] = 0
# the thin wedge of sky between her raised upper arm and forearm was read as scales
_sk = Image.new('L', (W, H), 0)
ImageDraw.Draw(_sk).polygon([(814.8, 563.4), (844.2, 590.1), (876.4, 654.3), (889.7, 698.2), (865.6, 686.4), (841.6, 638.2), (820.2, 590.1)], fill=1)
mat[np.asarray(_sk, bool) & (mat == 3)] = 0
# the pale rim of the scale on her right thigh is scale too: with it left out, the scale got a second outline
_rm = Image.new('L', (W, H), 0)
ImageDraw.Draw(_rm).polygon([(819.0, 1556.1), (851.1, 1515.5), (854.8, 1525.1), (844.1, 1573.2), (824.9, 1613.9), (813.6, 1620.3)], fill=1)
mat[np.asarray(_rm, bool) & (mat == 0)] = 3
# the bottom of that gap, down to where her thighs meet, was read as swimsuit
_gb = Image.new('L', (W, H), 0)
ImageDraw.Draw(_gb).polygon([(354, 1178), (361, 1178), (363, 1190), (353, 1207), (341, 1208), (349, 1195)], fill=1)
mat[np.asarray(_gb, bool) & (mat == 2)] = 0
# the thin line rising behind her ahoge is the umbrella's edge, not skin
mat[0:104, 470:505][mat[0:104, 470:505] == 1] = 0
# the hair round the fin by her right cheek (between the fin and her eye, and under it) was read as scales
_hf = Image.new('L', (W, H), 0)
ImageDraw.Draw(_hf).polygon([(317, 287), (334.4, 343), (368.8, 394.5), (334, 405), (270, 429), (260, 460),
                             (380, 460), (380, 287)], fill=1)
mat[np.asarray(_hf, bool) & (mat == 3)] = 4
np.save('mat.npy', mat)
