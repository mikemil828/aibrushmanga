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
