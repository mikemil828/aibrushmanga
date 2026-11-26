"""Region map of the rice-field reference (ref px, 1152 x 1728). Writes zones.npz and zones_preview.png.

Zones: sky, cloud, sun, mtn_far (blue ranges + mist), mtn_right (the big green mountain), trees (dark masses),
houses, pine, levee (green banks), water (paddies), path, figure, shadow.
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage as nd

a = np.asarray(Image.open('ref.png'), np.float32) / 255
r, g, b = a[..., 0], a[..., 1], a[..., 2]
L = a.mean(2)
H, W = L.shape
yy, xx = np.mgrid[:H, :W]


def poly(pts):
    im = Image.new('L', (W, H), 0); ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
    return np.asarray(im, bool)


def line_dist(pts, half):
    """Corridor around a polyline; half = per-point half width (interpolated)."""
    im = np.zeros((H, W), np.float32)
    best = np.full((H, W), 1e9, np.float32); hw = np.zeros((H, W), np.float32)
    for (p, q, h0, h1) in zip(pts[:-1], pts[1:], half[:-1], half[1:]):
        p, q = np.array(p, float), np.array(q, float)
        d = q - p; t = np.clip(((xx - p[0]) * d[0] + (yy - p[1]) * d[1]) / max(d @ d, 1e-9), 0, 1)
        dist = np.hypot(xx - (p[0] + t * d[0]), yy - (p[1] + t * d[1]))
        sel = dist < best; best[sel] = dist[sel]; hw[sel] = (h0 + (h1 - h0) * t)[sel]
    return best <= hw


# ---- sky: bright area connected to the top, above the ground
Ls = nd.gaussian_filter(L, 1.2)
br = Ls > 0.69
br[700:] = False
lab, _ = nd.label(br)
sky = lab == lab[5, 5]
sky = nd.binary_fill_holes(sky | (yy < 260))
# the left range and the middle peak, read off the reference (the sunlit slope is as bright as the sky)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from manga import geom
RIDGE_L = [(-5, 456), (80, 480), (160, 492), (250, 502), (350, 530), (450, 570), (520, 600)]
RIDGE_M = [(330, 528), (400, 495), (450, 475), (500, 450), (550, 425), (600, 409), (625, 407), (650, 415),
           (700, 422), (740, 432)]
def _under(R):
    C = geom.catmull(np.array(R, float), 10)
    return poly([tuple(p) for p in C] + [(C[-1][0], 720), (C[0][0], 720)])
sky &= ~_under(RIDGE_L) & ~_under(RIDGE_M)
# clouds: the lilac masses inside the sky (bluer and darker than the sky around them)
lil = nd.gaussian_filter(b - (r + g) / 2, 2)
cloud = sky & (lil > 0.035) & (nd.gaussian_filter(L, 2) < 0.86)
cloud = nd.binary_opening(cloud, iterations=2)
# the sun disc
sun = np.hypot(xx - 336, yy - 486) < 52
sun &= sky

ground = (yy > 672) | ~sky & (yy > 640) & (L > 0.0)
# ---- green banks and seedlings
green = (g > r + 0.03) & (g > b - 0.02)
# ---- the path: brown ground along a hand-placed centre line
centre = [(612, 688), (640, 704), (612, 722), (560, 736), (470, 760), (438, 778), (452, 802), (520, 846),
          (600, 886), (668, 926), (690, 965), (692, 1200), (640, 1282), (580, 1380), (518, 1480), (450, 1600),
          (395, 1728)]
half = [6, 7, 8, 10, 12, 13, 14, 16, 18, 22, 26, 40, 55, 68, 82, 100, 125]
corr = line_dist(centre, half)
brown = (r > g) & (g >= b - 0.02) & (r - b > 0.04)
path = corr & brown & ~green & (yy > 685)
path = nd.binary_closing(path, iterations=2)
path = nd.binary_opening(path, iterations=1) & corr
# around the walker the path is wider than the brown test finds (his shadow and the trodden ground)
AROUND = [(655, 940), (725, 940), (755, 1060), (760, 1200), (745, 1245), (640, 1245), (628, 1150), (630, 1040)]
path |= poly(AROUND) & ~green

# ---- figure: hand outline (hat, haori, hakama, feet), refined by colour
FIG = [(688, 963), (700, 966), (717, 977), (724, 990), (720, 998), (707, 1003), (719, 1010), (731, 1030),
       (742, 1062), (744, 1082), (736, 1094), (734, 1112), (728, 1118), (722, 1112), (721, 1150), (722, 1192),
       (714, 1203), (700, 1207), (697, 1228), (680, 1232), (672, 1214), (661, 1205), (659, 1180), (661, 1120),
       (655, 1118), (648, 1108), (646, 1094), (633, 1086), (636, 1056), (648, 1022), (662, 1008), (675, 1002),
       (660, 999), (651, 991), (656, 980), (676, 967)]
figure = poly(FIG)
SWORDS = [[(670, 1080), (674, 1084), (637, 1141), (632, 1138)], [(705, 1080), (710, 1079), (732, 1146), (727, 1149)]]
for s_ in SWORDS:
    figure |= poly(s_)

# ---- houses, pine (hand boxes refined by darkness)
HOUSES = [[(0, 615), (12, 612), (28, 640), (24, 682), (0, 684)],                                   # cut off left
          [(36, 650), (75, 596), (110, 598), (150, 610), (196, 656), (182, 662), (182, 684), (46, 686), (48, 660)],
          [(248, 648), (262, 618), (318, 617), (334, 646), (300, 664), (252, 664)],                  # far roof
          [(884, 672), (889, 662), (912, 660), (918, 672)],
          [(928, 666), (946, 645), (990, 644), (1001, 665), (995, 676), (932, 676)],
          [(1006, 662), (1024, 637), (1078, 636), (1110, 660), (1104, 678), (1012, 678)]]
houses = np.zeros((H, W), bool)
for h_ in HOUSES:
    houses |= poly(h_)
PINE = [(165, 330), (240, 330), (270, 460), (220, 640), (190, 690), (170, 690), (180, 600), (150, 560), (155, 460)]
pine = poly(PINE) & (L < 0.42)
pine = nd.binary_opening(pine, iterations=1)

# ---- dark tree masses near the horizon
trees = (~sky) & (L < 0.33) & (yy < 690) & ~houses & ~pine & (((xx < 160) & (yy > 540)) | (yy > 590) | ((xx > 940) & (yy > 450) & (L < 0.2)))
trees = nd.binary_opening(trees, iterations=1)

# ---- mountains: everything non-sky above the ground that isn't trees, houses or pine
mtn = (~sky) & (yy < 690) & ~trees & ~houses & ~pine
RIGHT = [(470, 560), (700, 420), (900, 285), (1000, 300), (1152, 340), (1152, 520), (980, 560), (700, 640),
         (470, 640)]
mtn_right = mtn & poly(RIGHT)
mtn_far = mtn & ~mtn_right

# ---- below the horizon
below = (yy >= 672) & ~houses & ~pine & ~trees
levee = below & green & ~path & ~figure
levee = nd.binary_opening(levee, iterations=1)
# far rows near the horizon: clean horizontal strips (smooth along the rows, not across them)
far = (yy < 800)
lf = nd.gaussian_filter(levee.astype(np.float32), (0.8, 7.0)) > 0.5
levee = np.where(far, lf & below & ~path & ~figure, levee)
water = below & ~levee & ~path & ~figure
# shadow of the walker on the right paddy and path: dark, bluish
SHADOW = [(686, 1222), (712, 1226), (745, 1255), (790, 1292), (850, 1322), (900, 1390), (940, 1470), (960, 1530),
          (950, 1558), (972, 1572), (987, 1600), (982, 1632), (958, 1648), (928, 1645), (906, 1626), (900, 1596),
          (908, 1568), (880, 1545), (850, 1500), (820, 1420), (800, 1345), (760, 1305), (722, 1272), (698, 1250)]
shadow = poly(SHADOW) & ~figure

np.savez_compressed('zones.npz', sky=sky & ~cloud & ~sun, cloud=cloud, sun=sun, mtn_far=mtn_far,
                    mtn_right=mtn_right, trees=trees, houses=houses, pine=pine, levee=levee, water=water & ~shadow,
                    path=path, figure=figure, shadow=shadow)
cols = dict(sky=(250, 240, 200), cloud=(200, 170, 230), sun=(255, 255, 255), mtn_far=(130, 140, 200),
            mtn_right=(60, 110, 90), trees=(20, 50, 30), houses=(150, 80, 40), pine=(200, 60, 60),
            levee=(80, 200, 80), water=(150, 200, 250), path=(170, 120, 60), figure=(255, 0, 255),
            shadow=(40, 40, 120))
z = np.load('zones.npz')
o = np.zeros((H, W, 3), np.uint8)
for k, c in cols.items():
    o[z[k]] = c
Image.fromarray(o).resize((576, 864)).save('zones_preview.png')
print({k: int(z[k].sum()) for k in z.files})
