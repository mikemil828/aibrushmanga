"""Materials of the figure, decided by hand: each one is inked and toned its own way on the page."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as nd
ref = np.asarray(Image.open('../ref.png').convert('RGB'), np.float32) / 255
H, W = ref.shape[:2]
g = ref.mean(2)
d3 = nd.gaussian_filter(1 - g, 3)
fig, hair = np.load('../fig.npy'), np.load('../hair.npy')
def poly(pts):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([(2*x, 2*y) for x, y in pts], fill=1)
    return np.asarray(im, bool)
SUIT = [(205,265),(278,265),(300,310),(335,330),(340,420),(328,500),(345,540),(368,595),(310,700),(290,720),
        (275,720),(255,700),(160,600),(162,560),(150,500),(146,420),(150,330),(200,310)]
FACE = [(186,175),(215,140),(255,128),(300,140),(322,175),(322,215),(305,255),(295,300),(200,300),(190,240)]
light = d3 < 0.30
body = fig & ~hair
suit = body & poly(SUIT) & light
suit = nd.binary_closing(suit, iterations=4) & poly(SUIT) & body
suit = nd.binary_opening(suit, iterations=3)
face = poly(FACE)
scales = body & light & ~suit & ~face
scales = nd.binary_opening(scales, iterations=2)
SIDE = poly([(140,315),(180,315),(188,585),(150,585)]) | poly([(328,515),(368,515),(368,605),(328,605)])
TAIL = [(405,470),(440,435),(500,430),(530,480),(540,560),(535,640),(500,700),(440,705),(425,690),(432,600),(425,520)]
FINS = poly([(115,145),(150,145),(190,190),(185,220),(140,220),(115,190)]) | poly([(318,190),(365,180),(372,222),(330,228)])
scales |= body & (poly(TAIL) | FINS) & ~suit
skin = body & ~suit & ~scales
gap = poly([(276,712),(306,712),(300,960),(282,960)]) & (d3 < 0.35)
skin &= ~gap; scales &= ~gap
mat = np.zeros((H, W), np.uint8)       # 0 bg, 1 skin, 2 suit, 3 scales, 4 hair
mat[skin] = 1; mat[suit] = 2; mat[scales] = 3; mat[hair] = 4
np.save('mat.npy', mat)
if __name__ == '__main__':
    col = np.array([[235, 235, 235], [150, 90, 60], [120, 170, 255], [120, 220, 140], [30, 30, 30]], np.float32)
    rgb = col[mat] * 0.65 + g[..., None] * 255 * 0.35
    Image.fromarray(rgb.astype(np.uint8)).resize((540, 960)).save('mat.png')
