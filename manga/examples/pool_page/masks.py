import sys, numpy as np
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', '..'))
from PIL import Image, ImageDraw
from scipy import ndimage as nd
from manga import prep
ref = np.asarray(Image.open('ref.png').convert('RGB'), np.float32) / 255
H, W = ref.shape[:2]
# figure outline, drawn by hand on the half-size grid view (x2 to base px); the tail is part of her
FIG = [(205,45),(240,55),(280,70),(320,110),(340,160),(350,200),(370,200),(385,150),(400,150),(410,175),(425,145),
       (445,170),(452,240),(470,280),(500,340),(518,385),(505,410),(450,420),(380,385),(352,400),(345,450),(340,520),
       (380,560),(395,600),(410,560),(450,510),(420,480),(450,440),(500,440),(530,480),(540,560),(535,640),(500,700),
       (440,705),(420,750),(400,820),(385,960),(170,960),(160,880),(155,780),(140,700),(110,770),(60,760),(70,700),
       (100,640),(110,560),(110,450),(115,360),(130,320),(145,300),(130,230),(120,200),(150,170),(160,120),(180,80),
       (200,70)]
def poly(pts):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([(2*x, 2*y) for x, y in pts], fill=1)
    return np.asarray(im, bool)
FINS = [(115,145),(150,145),(190,190),(185,220),(140,220),(115,190)]
rough = (poly(FIG) & ~poly([(470,455),(540,455),(540,540),(478,540)])) | poly(FINS) | poly([(330,190),(365,185),(370,215),(330,225)])
lines = prep.xdog_lines(ref, sigma=0.9, p=24, eps=-0.015, phi=80, tidy=12) > 0.5
# snap: every closed area between the lines goes wholly to the figure or wholly to the background
lab, n = nd.label(~nd.binary_dilation(lines, iterations=1))
frac = nd.mean(rough, lab, index=np.arange(1, n + 1))
fig = np.concatenate([[False], frac > 0.5])[lab]
fig = nd.binary_closing(fig, iterations=3) | (lines & nd.binary_dilation(fig, iterations=3) & rough)
hl, hn = nd.label(nd.binary_fill_holes(fig) & ~fig)
hs = nd.sum(np.ones_like(hl), hl, np.arange(1, hn + 1))
fig |= np.isin(hl, 1 + np.nonzero(hs < 2500)[0])
np.save('fig.npy', fig); np.save('lines.npy', lines)
g = ref.mean(2)
show = np.where(fig, g, g * 0.35 + 0.6)
Image.fromarray((show * 255).astype(np.uint8)).resize((540, 960)).save('figmask.png')
HAIR = [(140,150),(150,120),(175,85),(215,65),(265,62),(310,80),(340,120),(352,170),(350,300),(320,310),(300,290),(185,290),
        (165,330),(130,330),(128,230),(145,170)]
FACE = [(186,175),(215,140),(255,128),(300,140),(322,175),(322,215),(305,255),(295,315),(200,315),(190,240)]
BRAIDR = [(345,380),(362,380),(365,430),(345,432)]
dk = nd.gaussian_filter(1 - g, 2) > 0.35
hair = dk & ((poly(HAIR) & ~poly(FACE)) | poly(BRAIDR)) & ~poly(FINS) & ~poly([(330,190),(365,185),(370,215),(330,225)])
band = poly([(180,115),(330,115),(330,205),(180,205)])
dd = nd.gaussian_filter(1 - g, 1.5)
hair |= band & (dd > 0.58) & poly(HAIR)
hair = nd.binary_opening(hair, iterations=1)
# the ahoge: a single strand standing up, drawn as one black shape
hair |= nd.binary_dilation(lines & poly([(190,0),(272,0),(272,62),(190,62)]), iterations=1) | (
    poly([(190,0),(272,0),(272,62),(190,62)]) & (nd.gaussian_filter(1 - g, 1.0) > 0.36))
pool = poly([(0,548),(165,548),(165,765),(0,765)]) & ~fig
np.save('pool.npy', pool)
np.save('hair.npy', hair)
e = hair ^ nd.binary_erosion(hair, iterations=2)
rgb = np.stack([g, g, g], 2); rgb[e] = [1, 0, 0]
Image.fromarray((rgb * 255).astype(np.uint8)).crop((240, 100, 760, 900)).save('hairmask.png')
