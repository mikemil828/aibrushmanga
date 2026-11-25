# READING VIEW of движок/кисть/подпись.py with names in English. Not runnable; edit the original.
"""Подпись инструмента (п. 9): «кисть, нагрузка → числа», снятые с положенного пробного мазка.

Числа меряются с картинки, а не берутся из пресета, и той же меркой меряется образец:
  шаг_борозд(серое, угол)        шаг борозд поперёк хода, px — длина поперечных профилей на число борозд в них;
  ширина_кромки(профили)          резкость кромки — ширина перехода 10–90 % от фона к краске, px;
  просвечивание(цвет, краска, низ) доля нижнего слоя в цвете следа (0 — кроет, 1 — краски нет).
Пробный мазок: прямой, поперёк холста, длиной 8 ширин, тёмная краска по светлому грунту, нажим 0,85.
Шаг борозд — со светом (как на фото образца), кромка и просвечивание — без света: свет у кромки даёт фаску
(блик и тень), это рельеф, а не переход цвета."""
import numpy as np

from .paint import from_linear, in_linear
from .path import path_with_pressure

_YaRBrushResponseWithT = np.array([0.2126, 0.7152, 0.0722])
GRAtNT = (200, 190, 170)
BrushRAWithBrushA = (70, 60, 45)


def gray(image_linear):
    """Линейная картинка (В, Ш, 3) -> светлота sRGB 0..255 (как у фото)."""
    return (from_linear(image_linear) * 255.0) @ _YaRBrushResponseWithT


def angle_sweep(gray_):
    """Главное направление борозд участка (структурный тензор), радианы, по ходу мазка."""
    gy, gx = np.gradient(np.asarray(gray_, np.float64))
    return 0.5 * np.arctan2(2.0 * (gx * gy).sum(), (gx * gx).sum() - (gy * gy).sum()) + np.pi / 2


def _sample(a, xs, ys):
    """Билинейная выборка a в точках (xs, ys)."""
    In, Sh = a.shape
    xs = np.clip(xs, 0, Sh - 1.001)
    ys = np.clip(ys, 0, In - 1.001)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    return (a[y0, x0] * (1 - fx) * (1 - fy) + a[y0, x0 + 1] * fx * (1 - fy)
            + a[y0 + 1, x0] * (1 - fx) * fy + a[y0 + 1, x0 + 1] * fx * fy)


def transverse(gray_, angle, half=None, spacing_along=2.0):
    """Профили поперёк хода (строка — один профиль, шаг 0,5 px) из середины участка."""
    a = np.asarray(gray_, np.float64)
    In, Sh = a.shape
    half = half if half is not None else 0.45 * min(In, Sh)
    tx, ty = np.cos(angle), np.sin(angle)
    nx, ny = -ty, tx
    cx, cy = (Sh - 1) / 2.0, (In - 1) / 2.0
    s = np.arange(-half, half + 1e-9, spacing_along)
    t = np.arange(-half, half + 1e-9, 0.5)
    xs = cx + s[:, None] * tx + t[None, :] * nx
    ys = cy + s[:, None] * ty + t[None, :] * ny
    return _sample(a, xs, ys)


def _details(profiles):
    """Мелкая деталь профилей (шаг 0,5 px): без плавной части (квадратичная подгонка), шум пикселя сглажен (σ 0,35 px)."""
    p = np.asarray(profiles, np.float64)
    t = np.linspace(-1.0, 1.0, p.shape[1])
    V = np.vstack([np.ones_like(t), t, t * t]).T
    d = p - (np.linalg.lstsq(V, p.T, rcond=None)[0].T @ V.T)
    g = np.exp(-0.5 * (np.arange(-3, 4) * 0.5 / 0.35) ** 2)
    g /= g.sum()
    return np.array([np.convolve(r, g, 'same') for r in d])[:, 3:-3]


def spacing_furrows(gray_=None, angle=None, half=None, profiles=None, threshold=0.3):
    """Шаг борозд поперёк хода, px: длина профилей, делённая на число борозд в них. Борозда — местный минимум мелкой
    детали глубже `порог` её σ (у профиля своя σ: мерка не зависит от контраста). Профили поперёк хода, шаг 0,5 px:
    свои (`профили`) или из середины участка `серое_` по углу хода (None — угол по структурному тензору).
    None — борозд нет."""
    if profiles is None:
        if angle is None:
            angle = angle_sweep(gray_)
        profiles = transverse(gray_, angle, half)
    n, L = 0, 0.0
    for r in _details(profiles):
        s = r.std()
        n += int(((r[1:-1] < r[:-2]) & (r[1:-1] <= r[2:]) & (r[1:-1] < -threshold * s)).sum())
        L += (len(r) - 1) * 0.5
    return L / n if n else None


def width_edges(profiles, bottom=0.1, top=0.9):
    """Ширина перехода, px: профили (строка — от фона к краске, шаг 1 px). Уровень фона — медиана первых 3 px,
    краски — медиана последней трети; переход — от первой доли «низ» до первой доли «верх». Среднее по профилям
    (не медиана: ширины у одного мазка двугорбые — резкий бок и рваный, медиана прыгала между горбами)."""
    sh = []
    for p in np.asarray(profiles, np.float64):
        f, brush = np.median(p[:3]), np.median(p[2 * len(p) // 3:])
        if abs(brush - f) < 8.0:                                   # нет перепада — нечего мерить
            continue
        q = (p - f) / (brush - f)
        def where(level):
            i = int(np.argmax(q >= level))
            if q[i] < level:
                return None
            return i - 1 + (level - q[i - 1]) / max(q[i] - q[i - 1], 1e-9) if i > 0 else 0.0
        a, b = where(bottom), where(top)
        if a is not None and b is not None and b >= a:
            sh.append(b - a)
    return float(np.mean(sh)) if sh else None


def show_through(color_linear, paint_linear, bottom_linear):
    """Доля нижнего слоя в цвете следа: проекция (цвет − краска) на (низ − краска), линейный RGB; среднее."""
    brush, n = np.asarray(paint_linear, np.float64), np.asarray(bottom_linear, np.float64)
    d = n - brush
    t = ((np.asarray(color_linear, np.float64).reshape(-1, 3) - brush) @ d) / float(d @ d)
    return float(np.clip(t, 0.0, 1.0).mean())


def trial_stroke(brush, load=None, seed=1, px_on_mm=None):
    """Положить пробный мазок подписи; ответ — (холст, ось y, x0, x1, ширина кисти px)."""
    from .canvas import Canvas
    brush = brush.in_pixels(px_on_mm).copy()
    brush.wipe()
    W = float(brush.width)
    L = 8.0 * W
    canvas = Canvas(int(L + 4 * W), int(4 * W), ground=GRAtNT, px_on_mm=px_on_mm)
    y = 2.0 * W
    ctrl = np.array([[2 * W, y], [2 * W + L / 2, y], [2 * W + L, y]])
    points = path_with_pressure(ctrl, W, np.random.default_rng(seed), pressure=0.85)
    canvas.stroke(points, BrushRAWithBrushA, brush, layer='проба', seed=seed, load=load)
    return canvas, y, 2 * W, 2 * W + L, W


def _signature_stroke(brush, load, seed, px_on_mm):
    canvas, y, x0, x1, W = trial_stroke(brush, load, seed, px_on_mm)
    xa, xb = int(x0 + 0.2 * (x1 - x0)), int(x0 + 0.8 * (x1 - x0))
    with_light = gray(canvas.image(light=True))
    without = canvas.image(light=False)
    gray_without = gray(without)
    yi = int(round(y))
    # профили поперёк хода — столбцы ±0,42 ширины от оси, на сетке 0,5 px
    h = 0.42 * W
    ys = np.arange(y - h, y + h + 1e-9, 0.5)
    y0 = int(np.floor(ys[0]))
    piece = with_light[y0:int(np.ceil(ys[-1])) + 2, xa:xb]
    profile = np.array([np.interp(ys - y0, np.arange(piece.shape[0]), piece[:, j]) for j in range(piece.shape[1])])
    # кромка: столбцы через 2 px, профиль от фона (за 0,9 ширины от оси) к оси, оба бока
    d = int(0.9 * W)
    top = gray_without[yi - d:yi + 1, xa:xb:2].T
    bottom_ = gray_without[yi:yi + d + 1, xa:xb:2].T[:, ::-1]
    h2 = max(int(0.25 * W), 2)
    return (profile, np.vstack([top, bottom_]),
            show_through(without[yi - h2:yi + h2 + 1, xa:xb], in_linear(BrushRAWithBrushA), in_linear(GRAtNT)))


def signature(brush, load=None, seed=1, px_on_mm=None, strokes=12):
    """Подпись инструмента: {'шаг_борозд', 'кромка', 'просвечивание'} (px, px, доля); с px_на_мм — и в мм.
    Пробных мазков `мазков` (сиды от `сид`), меряется середина каждого (средние 60 % длины): шаг — по ±0,42 ширины
    от оси со светом, кромка — по обоим бокам без света, просвечивание — по ±0,25 ширины от оси без света;
    профили всех мазков — в одну мерку. Кисть не пачкается."""
    profile, kr, pr = [], [], []
    for i in range(strokes):
        a, b, in_ = _signature_stroke(brush, load, int(seed) * 1009 + i, px_on_mm)
        profile.append(a)
        kr.append(b)
        pr.append(in_)
    spacing = spacing_furrows(profiles=np.vstack(profile))
    rim = width_edges(np.vstack(kr))
    response = {'шаг_борозд': spacing, 'кромка': rim, 'просвечивание': float(np.mean(pr))}
    if px_on_mm:
        response['шаг_борозд_мм'] = None if spacing is None else spacing / px_on_mm
        response['кромка_мм'] = None if rim is None else rim / px_on_mm
    return response
