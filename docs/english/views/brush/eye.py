# READING VIEW of движок/кисть/глаз.py with names in English. Not runnable; edit the original.
"""Глаза: взгляд на холст в sRGB 8 бит, цвет в OKLab и светлоте L серой шкалы 0–255, разность с целью, мерки.

Холст только читается: пиксели, сиды и журнал не меняются. Все числа светлоты — L: светлота OKLab, выраженная серой
шкалой sRGB 0–255 (серый (v, v, v) имеет L = v). Цвета на входе и выходе — sRGB 0–255; холст — `х.картинка`
(линейный свет), переведённая в sRGB. Цель (образец), с которой сравнивают, — одно из: картинка (путь к PNG/JPG или
массив (В, Ш, 3) sRGB 0–255 в координатах холста), холст (`Холст` или путь к `.npz` `сохранить`), функция цвета
`f(x, y)` — x, y массивы координат холста, ответ (…, 3) sRGB 0–255.
"""
import math
import os

import numpy as np

from .paint import from_linear

# OKLab (Б. Оттоссон, 2020): линейный sRGB -> LMS -> корень кубический -> Lab
_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_M1i, _M2i = np.linalg.inv(_M1), np.linalg.inv(_M2)
GRAY_OUTSIDE = 128                       # взгляд: место рядом, где образца нет


def _lin(c):
    c = np.asarray(c, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(l):
    l = np.clip(np.asarray(l, np.float64), 0.0, 1.0)
    return 255.0 * np.where(l <= 0.0031308, 12.92 * l, 1.055 * l ** (1 / 2.4) - 0.055)


def oklab(rgb):
    """sRGB 0–255 (…, 3) -> OKLab (…, 3): L 0–1, a, b."""
    return np.cbrt(_lin(rgb) @ _M1.T) @ _M2.T


def from_oklab(lab):
    """OKLab (…, 3) -> sRGB 0–255 (…, 3), float, обрезано в 0–255."""
    return _srgb((np.asarray(lab, np.float64) @ _M2i.T) ** 3 @ _M1i.T)


def _L(lab_L):
    """L OKLab (0–1) -> L серой шкалы 0–255."""
    return _srgb(np.asarray(lab_L, np.float64) ** 3)


def _from_L(L):
    """L серой шкалы 0–255 -> L OKLab 0–1."""
    return np.cbrt(_lin(L))


def lightness(rgb):
    """L серой шкалы: светлота OKLab цвета sRGB 0–255 (…, 3), выраженная серым sRGB 0–255 (серый (v, v, v) — L = v)."""
    return _L(oklab(rgb)[..., 0])


def color(L, C, h):
    """Цвет из L серой шкалы 0–255, хромы C OKLab и тона h (градусы, 0 — к красному, 90 — к жёлтому) -> sRGB.
    Числа — кортеж (r, g, b) целых 0–255 (годится в `мазок`); массивы — float (…, 3). `цвет(246, 0.034, 72)` —
    (255, 243, 222), #fff3de. Вне охвата sRGB — обрезано в 0–255. Обратно — `лцх`."""
    scalar = all(np.ndim(v) == 0 for v in (L, C, h))
    L, C, h = (np.asarray(v, np.float64) for v in (L, C, h))
    lab = np.stack(np.broadcast_arrays(_from_L(L), C * np.cos(np.radians(h)), C * np.sin(np.radians(h))), -1)
    rgb = from_oklab(lab)
    return tuple(int(v) for v in np.round(rgb)) if scalar else rgb


def lch(rgb):
    """sRGB 0–255 -> (L, C, h): L серой шкалы 0–255, хрома OKLab, тон в градусах 0–360. Обратное к `цвет`.
    Для массива (…, 3) — три массива."""
    lab = oklab(rgb)
    L, C = lightness(rgb), np.hypot(lab[..., 1], lab[..., 2])
    h = np.where(C < 1e-6, 0.0, np.degrees(np.arctan2(lab[..., 2], lab[..., 1])) % 360.0)
    if np.ndim(L) == 0:
        return round(float(L), 2), round(float(C), 4), round(float(h), 1)
    return L, C, h


def hex(rgb):
    """sRGB 0–255 -> '#rrggbb'."""
    return '#%02x%02x%02x' % tuple(int(v) for v in np.clip(np.round(np.asarray(rgb, np.float64)), 0, 255))


def difference_color(a, b):
    """Разность двух цветов sRGB 0–255 (или массивов): dL = L(a) − L(b) в серой шкале; da, db — в плоскости a, b
    OKLab; dab — длина (da, db). Числа — словарь {'dL', 'da', 'db', 'dab'}."""
    la, lb = oklab(a), oklab(b)
    dL = lightness(a) - lightness(b)
    da, db = la[..., 1] - lb[..., 1], la[..., 2] - lb[..., 2]
    if np.ndim(dL) == 0:
        return {'dL': round(float(dL), 2), 'da': round(float(da), 4), 'db': round(float(db), 4),
                'dab': round(float(np.hypot(da, db)), 4)}
    return {'dL': dL, 'da': da, 'db': db, 'dab': np.hypot(da, db)}


# ---------------------------------------------------------------- холст и цель в sRGB

def _frame(canvas, region):
    """область (x0, y0, x1, y1) или None -> рамка в координатах холста, обрезанная холстом."""
    gy, gx = canvas._region(region)
    if gy.stop <= gy.start or gx.stop <= gx.start:
        raise ValueError(f'глаз: область {region} мимо холста ({canvas.x0}, {canvas.y0}, {canvas.x0 + canvas.width}, {canvas.y0 + canvas.height})')
    return canvas.x0 + gx.start, canvas.y0 + gy.start, canvas.x0 + gx.stop, canvas.y0 + gy.stop


def _kind(canvas, frame, light, strength=None, surface=None):
    """Холст в рамке -> sRGB 0–255 float64 (не округлено). поверхность — Г-2, ключ вывода `Холст.картинка`."""
    if surface is None:
        return np.clip(from_linear(canvas.image(light=light, strength=strength, region=frame)) * 255.0, 0.0, 255.0)
    return np.clip(from_linear(canvas.image(light=light, strength=strength, region=frame, surface=surface)) * 255.0, 0.0, 255.0)


_BrushESh = {}


def _load_file(path):
    """Картинка (sRGB 0–255 float) или холст по пути; последние два — в памяти, пока файл не менялся."""
    path = os.path.expanduser(str(path))
    option = (os.path.abspath(path), os.path.getmtime(path))
    if option not in _BrushESh:
        if path.endswith('.npz'):
            from .canvas import Canvas
            what = Canvas.load_file(path, cores=1)
        else:
            from PIL import Image
            with Image.open(path) as im:
                what = np.asarray(im.convert('RGB'), np.float64)
        while len(_BrushESh) >= 2:
            _BrushESh.pop(next(iter(_BrushESh)))
        _BrushESh[option] = what
    return _BrushESh[option]


def _target(target, frame, light, strength=None, outside=None):
    """Цель на рамке холста -> sRGB 0–255 float64 (В, Ш, 3). вне=None — цель меньше рамки: ValueError;
    вне=число — недостающее этим серым."""
    from .canvas import Canvas
    x0, y0, x1, y1 = frame
    if isinstance(target, (str, os.PathLike)):
        target = _load_file(target)
    if isinstance(target, Canvas):
        gy, gx = target._region(frame)
        if gy.stop - gy.start == y1 - y0 and gx.stop - gx.start == x1 - x0:
            return _kind(target, frame, light, strength)
        a = None
        if gy.stop > gy.start and gx.stop > gx.start:
            a = _kind(target, (target.x0 + gx.start, target.y0 + gy.start, target.x0 + gx.stop, target.y0 + gy.stop), light, strength)
        shift = (target.x0 + gx.start - x0, target.y0 + gy.start - y0)
        return _complete(a, shift, frame, outside, 'холст-цель')
    if callable(target):
        yy, xx = np.mgrid[y0:y1, x0:x1]
        a = np.asarray(target(xx.astype(np.float64), yy.astype(np.float64)), np.float64)
        if a.shape != (y1 - y0, x1 - x0, 3):
            raise ValueError(f'глаз: функция цели вернула {a.shape}, нужно (В, Ш, 3) = {(y1 - y0, x1 - x0, 3)}: '
                             'f(x, y) получает массивы координат холста и отвечает sRGB 0–255 на каждую точку')
        return a
    a = np.asarray(target)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f'глаз: цель — картинка (В, Ш, 3) sRGB 0–255, холст, путь или функция f(x, y); дано {type(target).__name__} {a.shape}')
    a = a[..., :3].astype(np.float64)
    if a.shape[0] >= y1 and a.shape[1] >= x1:
        return a[y0:y1, x0:x1]
    piece = a[y0:min(y1, a.shape[0]), x0:min(x1, a.shape[1])] if y0 < a.shape[0] and x0 < a.shape[1] else None
    return _complete(piece, (0, 0), frame, outside, f'картинка-цель {a.shape[1]}×{a.shape[0]}')


def _complete(a, shift, frame, outside, what):
    x0, y0, x1, y1 = frame
    if outside is None:
        raise ValueError(f'глаз: {what} не покрывает область {frame}: мерить только там, где цель есть')
    p = np.full((y1 - y0, x1 - x0, 3), float(outside))
    if a is not None:
        p[shift[1]:shift[1] + a.shape[0], shift[0]:shift[0] + a.shape[1]] = a
    return p


def _in_8(a):
    return (np.clip(a, 0, 255) + 0.5).astype(np.uint8)


def _scale(a8, scale):
    if scale == 1:
        return a8
    from PIL import Image
    im = Image.fromarray(a8)
    size = (max(1, int(im.width * scale)), max(1, int(im.height * scale)))
    whole = scale > 1 and float(scale).is_integer()
    return np.asarray(im.resize(size, Image.NEAREST if whole else Image.LANCZOS))


# ---------------------------------------------------------------- Б-1 взгляд

def look(canvas, region=None, scale=1.0, light=True, strength=None, beside=None, path=None, without=None, to=None,
           surface=None):
    """Холст глазами: sRGB 8 бит (uint8 (В, Ш, 3)) области (x0, y0, x1, y1), увеличенный в `масштаб` раз (целое > 1 —
    каждый пиксель квадратом, без сглаживания; иначе Lanczos). свет, сила — как у `картинка`. рядом — образец (цель
    любого вида, см. модуль) или список их: то же место тем же масштабом встаёт справа от холста в один ряд; где
    образца нет — серый 128. путь — записать PNG 8 бит (нужен Pillow). без — слой или список слоёв, которых не видно;
    до — видны только слои ниже этого (Б-5): холст как без них, `х.слои` не меняются (`без_слоёв`). поверхность — Г-2,
    ключ вывода: 'матовое', 'полуматовое', 'лак' (None — умолчание: слабый узкий блик без дымки; образец `рядом` — всегда умолчанием). Ответ — массив ряда."""
    canvas = without_layers(canvas, without, to)
    frame = _frame(canvas, region)
    parts = [_kind(canvas, frame, light, strength, surface)]
    for response in ([] if beside is None else beside if isinstance(beside, (list, tuple)) else [beside]):
        parts.append(_target(response, frame, light, strength, outside=GRAY_OUTSIDE))
    row = _scale(_in_8(np.concatenate(parts, 1)), scale)
    if path is not None:
        from PIL import Image
        Image.fromarray(row).save(os.path.expanduser(str(path)))
    return row


# ---------------------------------------------------------------- Б-2 средний цвет

def mean(canvas, region=None, light=False):
    """Средний цвет области холста: L — среднее L серой шкалы по пикселям, a, b — средние OKLab, из них C, h и rgb.
    свет=False — тон краски без света рельефа (так меряются мерки). Ответ — {'L', 'C', 'h', 'rgb', 'hex'}."""
    a = _kind(canvas, _frame(canvas, region), light)
    lab = oklab(a)
    L = float(_L(lab[..., 0]).mean())
    ma, mb = float(lab[..., 1].mean()), float(lab[..., 2].mean())
    C = math.hypot(ma, mb)
    h = math.degrees(math.atan2(mb, ma)) % 360.0 if C >= 1e-6 else 0.0
    rgb = color(L, C, h)
    return {'L': round(L, 2), 'C': round(C, 4), 'h': round(h, 1), 'rgb': rgb, 'hex': hex(rgb)}


def transition(canvas, region, axis='x', spacing=8, light=False):
    """У-1: спад тона поперёк стыка. Профиль L — среднее L серой шкалы по пикселям поперёк оси (`ось='x'` — по столбцам
    области, профиль слева направо; `'y'` — по строкам, сверху вниз). Концы — медиана L первой и последней десятой доли
    профиля (не меньше 8 px). Переход 10–90 % — от последнего места до середины спада, где тон ещё ближе 10 % к началу,
    до первого места после середины, где он дальше 90 %; середина — место, где тон ближе всего к половине. Ступень —
    наибольшая разность L на `шаг` px вдоль профиля и её место. свет=False — тон краски без света рельефа. Ответ — словарь:
    'L_от', 'L_до', 'от_10', 'до_90' (px холста по оси), 'ширина' (px), 'ступень' (L), 'где' (px холста), 'профиль'
    (numpy, L по оси)."""
    x0, y0, x1, y1 = _frame(canvas, region)
    if axis not in ('x', 'y'):
        raise ValueError("переход: ось — 'x' или 'y'")
    L = lightness(_kind(canvas, (x0, y0, x1, y1), light)).mean(axis=0 if axis == 'x' else 1)
    n, start = len(L), x0 if axis == 'x' else y0
    spacing = int(spacing)
    if n < max(2 * spacing, 16):
        raise ValueError(f'переход: профиль {n} px — мало для шага {spacing}')
    brush = max(8, n // 10)
    a, b = float(np.median(L[:brush])), float(np.median(L[-brush:]))
    t = (L - a) / (b - a) if abs(b - a) > 1e-9 else np.zeros(n)
    with_ = int(np.argmin(np.abs(t - 0.5)))
    to = np.nonzero(t[:with_ + 1] < 0.1)[0]
    after = np.nonzero(t[with_:] > 0.9)[0]
    i10 = int(to[-1]) + 1 if len(to) else 0
    i90 = with_ + int(after[0]) if len(after) else n - 1
    d = np.abs(L[spacing:] - L[:-spacing])
    i = int(np.argmax(d))
    return {'L_от': round(a, 2), 'L_до': round(b, 2), 'от_10': start + i10, 'до_90': start + i90, 'ширина': i90 - i10,
            'ступень': round(float(d[i]), 2), 'где': start + i + spacing // 2, 'профиль': L}


# ---------------------------------------------------------------- Б-6 разность

class Difference:
    """Ответ `х.разность`: `карта` — dL = L холста − L цели (float32 (В, Ш)), `карта_ab` — длина разности в
    плоскости a, b; `сводка` — словарь чисел; `область`. Печать — сводка."""

    def __init__(self, map, map_ab, summary, region):
        self.map, self.map_ab, self.summary, self.region = map, map_ab, summary, region

    def __getitem__(self, option):
        return self.summary[option]

    def __repr__(self):
        return f'Разность({self.summary})'


def _sum_windows(d, w, s):
    """Средние d в квадратах w×w шагом s (левый верх — 0, s, 2s … пока окно внутри): (средние (n, m), y0, x0)."""
    c = np.zeros((d.shape[0] + 1, d.shape[1] + 1))
    c[1:, 1:] = d.cumsum(0).cumsum(1)
    ys, xs = np.arange(0, d.shape[0] - w + 1, s), np.arange(0, d.shape[1] - w + 1, s)
    if not len(ys) or not len(xs):
        return np.zeros((0, 0)), ys, xs
    Y, X = np.meshgrid(ys, xs, indexing='ij')
    return (c[Y + w, X + w] - c[Y, X + w] - c[Y + w, X] + c[Y, X]) / (w * w), ys, xs


def _dl(canvas, target, region, light):
    frame = _frame(canvas, region)
    l, ts = _kind(canvas, frame, light), _target(target, frame, light)
    return l, ts, frame


def difference(canvas, target, region=None, light=False, window=None, worst=5, path=None, scale=10.0):
    """Холст против цели в области: карта dL = L холста − L цели и сводка — среднее, σ, мин, макс dL, средняя dab
    (длина разности в плоскости a, b OKLab) и худшие окна: квадраты стороной `окно` px (без него — длинная сторона
    области / 8) шагом пол-окна, по |среднего dL| наибольшие, не налегающие друг на друга — [(x0, y0, x1, y1, dL)]
    в координатах холста. свет=False — тон краски без света рельефа. путь — PNG 8 бит карты: серый 128 — dL 0,
    светлее — холст светлее цели, ±`шкала` L — белый и чёрный."""
    l, ts, frame = _dl(canvas, target, region, light)
    ol, ots = oklab(l), oklab(ts)
    d = _L(ol[..., 0]) - _L(ots[..., 0])
    dab = np.hypot(ol[..., 1] - ots[..., 1], ol[..., 2] - ots[..., 2])
    x0, y0, x1, y1 = frame
    w = int(window) if window else max(2, int(round(max(d.shape) / 8)))
    w = min(w, *d.shape)
    s = max(1, w // 2)
    sr, ys, xs = _sum_windows(d, w, s)
    worst, occupied = [], []
    for k in np.argsort(-np.abs(sr), axis=None):
        i, j = np.unravel_index(k, sr.shape)
        r = (x0 + int(xs[j]), y0 + int(ys[i]), x0 + int(xs[j]) + w, y0 + int(ys[i]) + w)
        if any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3] for q in occupied):
            continue
        occupied.append(r)
        worst.append(r + (round(float(sr[i, j]), 2),))
        if len(worst) >= worst:
            break
    summary = {'среднее': round(float(d.mean()), 2), 'σ': round(float(d.std()), 2), 'мин': round(float(d.min()), 2),
              'макс': round(float(d.max()), 2), 'dab': round(float(dab.mean()), 4), 'окно': w, 'худшие': worst}
    if path is not None:
        from PIL import Image
        Image.fromarray(_in_8(128.0 + d * (127.0 / scale))).save(os.path.expanduser(str(path)))
    return Difference(d.astype(np.float32), dab.astype(np.float32), summary, frame)


# ---------------------------------------------------------------- Б-7 мерки

def windows(canvas, target, width, n=3, region=None, tolerance=2.0, light=False):
    """Мерка «окно N ширин». Бриф Л4: «Окно 3 ширин» (`удар.окно`): квадрат со стороной в три ширины следа кисти,
    которой написано место (кисть 100 px — окно 300 px), ходит по листу шагом в пол-окна; среднее L листа в каждом
    окне — профиль ± 2 L. Ловит перекос тона, а не рябь.
    Здесь: d = L холста − L цели по пикселям области; квадрат стороной n·ширина px (целое), шаг — половина стороны
    (целое), от левого верха области, пока окно внутри; в каждом окне — среднее d. Ответ — {'мин', 'макс',
    'доля в допуске' (доля окон с |среднее| ≤ допуск), 'окон', 'окно' (сторона), 'худшее' (x0, y0, x1, y1, dL),
    'слово'}. Окно больше короткой стороны области ужимается до неё; 'слово' говорит об этом (иначе None)."""
    l, ts, frame = _dl(canvas, target, region, light)
    d = lightness(l) - lightness(ts)
    w = int(n * width)
    if w < 1:
        raise ValueError(f'окна: окно {w} px — n·ширина от 1 px')
    word = None
    if w > min(d.shape):
        word = f'окно {w} px больше области {d.shape[1]}×{d.shape[0]} px — ужато до {min(d.shape)} px'
        w = min(d.shape)
    s = max(1, w // 2)
    sr, ys, xs = _sum_windows(d, w, s)
    i, j = np.unravel_index(int(np.argmax(np.abs(sr))), sr.shape)
    x0, y0 = frame[0], frame[1]
    return {'слово': word, 'мин': round(float(sr.min()), 1), 'макс': round(float(sr.max()), 1),
            'доля в допуске': round(float(np.mean(np.abs(sr) <= tolerance)), 2), 'окон': int(sr.size), 'окно': w,
            'худшее': (x0 + int(xs[j]), y0 + int(ys[i]), x0 + int(xs[j]) + w, y0 + int(ys[i]) + w, round(float(sr[i, j]), 1))}


def _reduce(a, n):
    k = int(np.ceil(max(a.shape[:2]) / n))
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k, *a.shape[2:]).mean((1, 3)), k


def gray(canvas, target, n=200, region=None, light=False):
    """Мерка «серая N». Бриф Л4: «Серая 200» (`удар.серая`): лист и профиль — в L, уменьшены усреднением так, что
    длинная сторона 200 px (клетка ~12 px, около удара узкой кисти); σ разности — не выше числа строки. Ловит рябь и
    пятна размером в удар.
    Здесь: d = L холста − L цели по пикселям области; клетка k = ⌈длинная сторона области / n⌉ px, область без
    остатка до кратного k; среднее d в каждой клетке; σ этих средних. Ответ — {'σ', 'клетка' (k), 'размер' (В, Ш)
    уменьшенного}."""
    l, ts, frame = _dl(canvas, target, region, light)
    m, k = _reduce(lightness(l) - lightness(ts), n)
    return {'σ': round(float(m.std()), 2), 'клетка': k, 'размер': m.shape}


def darker(canvas, region, background, light=False, to=None):
    """Мерка «объект темнее фона на X». Бриф Л4: стена темнее свода над ней (`лес.стена.темнее`) — видимая стена
    против свода над ней.
    Здесь: X = среднее L фона − среднее L области холста (больше нуля — область темнее). фон — прямоугольник
    (x0, y0, x1, y1) этого же холста или цель (картинка, холст, функция f(x, y)) на той же области. до (У-2) — слой:
    фон-прямоугольник меряется на холсте только со слоями ниже него (как `взгляд(до=…)`: верхний слой — ореол, вуаль —
    в L фона не входит); область — на всём холсте. С фоном-целью до — ValueError. Ответ — {'темнее', 'L', 'L фона'}."""
    frame = _frame(canvas, region)
    L = float(lightness(_kind(canvas, frame, light)).mean())
    if isinstance(background, (tuple, list)) and len(background) == 4 and all(np.ndim(v) == 0 for v in background):
        khf = without_layers(canvas, None, to)
        Lf = float(lightness(_kind(khf, _frame(khf, background), light)).mean())
    else:
        if to is not None:
            raise ValueError('темнее: до — только с фоном-прямоугольником холста (цель слоёв холста не знает)')
        Lf = float(lightness(_target(background, frame, light)).mean())
    return {'темнее': round(Lf - L, 1), 'L': round(L, 2), 'L фона': round(Lf, 2)}


def _blur(a, r):
    """Среднее по квадрату (2r+1)² суммами; за краем — крайнее значение."""
    for axis in (0, 1):
        c = np.cumsum(np.pad(a, [(r + 1, r) if i == axis else (0, 0) for i in (0, 1)], mode='edge'), axis)
        a = (np.take(c, range(2 * r + 1, c.shape[axis]), axis) - np.take(c, range(0, c.shape[axis] - 2 * r - 1), axis)) / (2 * r + 1)
    return a


def furrows(canvas, region=None, width=12, light=True, strength=None):
    """Мерка «борозды, L». Бриф Л4: `удар.борозды` — ворс внутри удара, в dL не входит; не шире 12 px.
    Здесь: L холста со светом рельефа (свет, сила — как у `картинка`) минус его среднее в квадрате (2·ширина + 1) px
    вокруг точки — остаётся то, что уже `ширина` px (ворс, борозды), тон удара и профиль уходят; ответ — {'σ' (σ
    остатка), 'размах' (95-й минус 5-й процентиль остатка)} по области. Число мерки — σ, не размах; свет рельефа в
    нём (без света — `свет=False`: на плоской 100 px σ вдвое меньше)."""
    frame = _frame(canvas, region)
    x0, y0, x1, y1 = frame
    r = int(width)
    field = (max(canvas.x0, x0 - r), max(canvas.y0, y0 - r), min(canvas.x0 + canvas.width, x1 + r), min(canvas.y0 + canvas.height, y1 + r))
    L = lightness(_kind(canvas, field, light, strength))
    response = (L - _blur(L, r))[y0 - field[1]:y1 - field[1], x0 - field[0]:x1 - field[0]]
    p5, p95 = np.percentile(response, [5, 95])
    return {'σ': round(float(response.std()), 2), 'размах': round(float(p95 - p5), 1)}


# ---------------------------------------------------------------- Б-5 без слоя, до слоя

def without_layers(canvas, without=None, to=None):
    """Холст для глаз без слоёв `без` (имя или список имён) и только со слоями ниже слоя `до` (сам `до` не входит) —
    копия полей холста, слои — те же объекты, только читать: `х.слои`, журнал, пиксели не меняются. Без `без` и `до` —
    сам холст. Нет такого слоя — KeyError со списком слоёв."""
    if without is None and to is None:
        return canvas
    canvas.finish_computing()
    names = list(canvas.layers)
    without = [] if without is None else [without] if isinstance(without, str) else list(without)
    for and_ in without + ([] if to is None else [to]):
        if and_ not in names:
            raise KeyError(f'глаз: слоя {and_!r} нет (слои снизу вверх: {names})')
    keep = names[:names.index(to)] if to is not None else names
    from .canvas import Canvas
    in_ = object.__new__(Canvas)
    in_.__dict__.update(canvas.__dict__)
    in_.__dict__.update(_layers={and_: canvas.layers[and_] for and_ in keep if and_ not in without}, _cores=None, _cores=1, journal=None, _snapshots=[],
                     _hints=None)
    return in_


# ---------------------------------------------------------------- К-6 виды холста

from .numbers import _TEN

MM_ON_H = _TEN[0] / 8.0        # мм краски в единице толщины h: 3,2 px высоты на h при 8 px/мм (`числа._ТЕНЬ`) — 0,4 мм
WARMTH_H = 60.0                  # тепло-холод: ось тона OKLab к оранжевому (60°); холодное — к синему (240°)
InAndDY = ('прищур', 'тон', 'насыщенность', 'тепло', 'края', 'толщина', 'сырость', 'направления', 'размеры')
_LAndN8 = _lin(np.arange(256)).astype(np.float32)
_ShRAndFTY = ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans.ttf',
           '/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
           '/Library/Fonts/Arial Unicode.ttf', 'C:/Windows/Fonts/arial.ttf')


class Views:
    """Ответ `х.виды`: `лист` — uint8 (В, Ш, 3) листа видов; `массивы` — словарь массивов клетками (`клетка` px холста
    на клетку, клетка (i, j) — x от x0 + j·клетка, y от y0 + i·клетка); `сводка` — числа; `область`. х['тон'] —
    массив. Печать — сводка."""

    def __init__(self, sheet, arrays, summary, region, cell):
        self.sheet, self.arrays, self.summary, self.region, self.cell = sheet, arrays, summary, region, cell

    def __getitem__(self, option):
        return self.arrays[option]

    def __repr__(self):
        return f'Виды({self.summary})'


def _cells(a, k):
    """Средние a (В, Ш[, …]) в клетках k×k; остаток края (меньше k) отброшен."""
    if k == 1:
        return a.astype(np.float32)
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k, *a.shape[2:]).mean((1, 3), dtype=np.float32)


_ZhAtRNALY = {}


def _strokes_journal(path):
    """Мазки журнала: [(номер строки, слой, ширина px, точки (n, 2))] — по журналу как он есть; мазки слоя, убранного
    строкой "layer_drop", выбывают. Помнит последний журнал, пока файл не менялся."""
    import io
    if isinstance(path, (bytes, bytearray)):             # М-10 (а): журнал до сохранения — байтами
        import hashlib
        option = ('байты', hashlib.sha1(path).hexdigest())
        open = lambda: io.StringIO(bytes(path).decode('utf-8'))
    else:
        path = os.path.expanduser(os.fspath(path))
        option = (os.path.abspath(path), os.path.getmtime(path), os.path.getsize(path))
        open = lambda: open(path, encoding='utf-8')
    if option in _ZhAtRNALY:
        return _ZhAtRNALY[option]
    import json
    brushes, strokes, px_mm = {}, [], None
    with open() as f:
        for number, s in enumerate(f, 1):
            s = s.strip()
            if not s or s.startswith('#'):
                continue
            z = json.loads(s)
            if 'canvas' in z:
                px_mm = z.get('px_per_mm')
            if 'def' in z and isinstance(z.get('brush'), str):
                d = z['def']
                brushes[z['brush']] = (d['ширина_мм'] * px_mm) if d.get('ширина_мм') and px_mm else d.get('ширина')
            if 'layer_drop' in z:
                strokes = [m for m in strokes if m[1] != z['layer_drop']]
            wet = z.get('ctrl', z.get('points', z.get('core')))
            if wet is None or 'tool' in z:
                continue
            t = np.asarray([p[:2] for p in wet], np.float64).reshape(-1, 2)
            sh = z.get('size', brushes.get(z.get('brush')))
            strokes.append((number, z.get('layer', 'основа'), float(sh or 1.0), t))
    _ZhAtRNALY.clear()
    _ZhAtRNALY[option] = strokes
    return strokes


def _thin_out(t, spacing):
    """Точки пути, между которыми не меньше `шаг` px (первая и последняя — всегда)."""
    if len(t) <= 2:
        return t
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t, axis=0).T))])
    i = np.unique(np.searchsorted(d, np.arange(0.0, d[-1], spacing)))
    return t[np.unique(np.concatenate([i, [len(t) - 1]]))]


def _sweeps(strokes, layers, frame, k, n, coverage, threshold=0.3):
    """Мазки на сетке клеток. В клетке — верхний мазок самого верхнего слоя, чья краска кроет клетку не меньше чем на
    `порог` (`покрытие` — {слой: доля клетки}); в слое верхний — позже по журналу. Клетка, где мазки прошли, а краски
    нет ни у одного (тонкая мгла, сухой ход), берёт верхний мазок с наибольшим покрытием. Ответ — (направление °
    0–180 (0 — вдоль x, 90 — вдоль y вниз; NaN — мазка нет), ширина px, длина px, номер строки журнала (0 — нет)) —
    массивы (n_y, n_x), и сколько мазков задели область."""
    from PIL import Image, ImageDraw
    ny, nx = n
    x0, y0, x1, y1 = frame
    by_layer = {with_: [] for with_ in layers}
    for m in strokes:
        if m[1] not in by_layer:
            continue
        t = m[3]
        r = m[2] / 2 + 1
        if t[:, 0].max() + r < x0 or t[:, 0].min() - r > x1 or t[:, 1].max() + r < y0 or t[:, 1].min() - r > y1:
            continue
        by_layer[m[1]].append(m)
    angles, widths, lengths, numbers = [np.nan], [np.nan], [np.nan], [0]
    id = np.zeros((ny, nx), np.int64)
    best = np.full((ny, nx), -1.0, np.float32)       # покрытие слоя, давшего мазок клетке; 2 — слой кроет
    for layer in reversed(layers):
        if not by_layer[layer]:
            continue
        im = Image.new('I', (nx, ny), 0)
        d = ImageDraw.Draw(im)
        for number, _, sh, t in by_layer[layer]:
            t = _thin_out(t, max(2.0 * k, sh / 4))
            dl = float(np.hypot(*np.diff(t, axis=0).T).sum()) if len(t) > 1 else 0.0
            p = (t - (x0, y0)) / k
            w = max(1, int(round(sh / k)))
            if len(t) == 1 or dl < 1e-6:
                angles.append(np.nan); widths.append(sh); lengths.append(dl); numbers.append(number)
                c = p[0]
                d.ellipse((c[0] - w / 2, c[1] - w / 2, c[0] + w / 2, c[1] + w / 2), fill=len(angles) - 1)
                continue
            for a, b in zip(p[:-1], p[1:]):
                angles.append(float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180.0))
                widths.append(sh); lengths.append(dl); numbers.append(number)
                d.line([tuple(a), tuple(b)], fill=len(angles) - 1, width=w)
        and_ = np.asarray(im)
        coverage = np.where(coverage[layer] >= threshold, 2.0, coverage[layer])
        where = (and_ > 0) & (coverage > best)
        id[where], best[where] = and_[where], coverage[where]
    return (np.asarray(angles, np.float32)[id], np.asarray(widths, np.float32)[id], np.asarray(lengths, np.float32)[id],
            np.asarray(numbers, np.int64)[id], sum(len(v) for v in by_layer.values()))


def _shift(a, d, axis):
    """a, сдвинутый на d клеток по оси (за краем — крайнее значение): ответ[i] = a[i + d]."""
    n = a.shape[axis]
    i = np.clip(np.arange(n) + d, 0, n - 1)
    return np.take(a, i, axis)


def _edges(L, threshold):
    """Края по L клеток: перепад — |L(+2) − L(−2)| по той оси, где он больше; жёсткий — больше 60 % перепада приходится
    на один шаг клетки (переход уже ~2 клеток), мягкий — растянут."""
    best, share = np.zeros_like(L), np.zeros_like(L)
    for axis in (0, 1):
        p = np.abs(_shift(L, 2, axis) - _shift(L, -2, axis))
        spacing = np.maximum.reduce([np.abs(_shift(L, d + 1, axis) - _shift(L, d, axis)) for d in (-2, -1, 0, 1)])
        q = np.where(p > 1e-6, spacing / np.maximum(p, 1e-6), 0.0)
        where = p > best
        best, share = np.where(where, p, best), np.where(where, q, share)
    edge = best >= threshold
    return best.astype(np.float32), edge & (share > 0.6), edge & (share <= 0.6)


def _font(size):
    from PIL import ImageFont
    for p in _ShRAndFTY:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _gray(a, lo, hi):
    v = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * 255.0
    return np.repeat(v[..., None], 3, -1)


def _sheet(panels, signatures, path):
    """Панели (В, Ш, 3) одного размера — сеткой по три в ряд, над каждой подпись."""
    from PIL import Image, ImageDraw
    ny, nx = panels[0].shape[:2]
    shp = max(12, min(18, nx // 28))
    header, field = shp + 8, 6
    rows = (len(panels) + 2) // 3
    font = _font(shp)
    count = max([nx] + [int(math.ceil(font.getlength(t))) + 4 for t in signatures])    # подпись не наезжает на соседнюю
    sheet = Image.new('RGB', (3 * count + 4 * field, rows * (ny + header + field) + field), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, (p, t) in enumerate(zip(panels, signatures)):
        x, y = field + (i % 3) * (count + field), field + (i // 3) * (ny + header + field)
        d.text((x + 2, y + 2), t, fill=(0, 0, 0), font=font)
        sheet.paste(Image.fromarray(_in_8(p)), (x, y + header))
    a = np.asarray(sheet)
    if path is not None:
        sheet.save(os.path.expanduser(str(path)))
    return a


def _h(v, n=1):
    return f'{v:.{n}f}'.replace('.', ',')


def legend_thicknesses(path, p16, max_mm, above, what):
    """М-10 (в): `<имя>.легенда.png` рядом с картой толщины `путь` — та же карта 8 бит той же абсолютной шкалой (0 —
    чёрный, `холст.ШКАЛА_ТОЛЩИНЫ_ММ` и толще — белый, одна на все холсты; серый — корень доли шкалы, чтобы тонкое
    0,05–0,2 мм было различимо) и под ней полоса шкалы с подписями в мм,
    доля места толще шкалы и наибольшая толщина. Ответ — путь легенды."""
    from PIL import Image, ImageDraw
    from .canvas import SCALE_THICKNESSES_MM
    in_, sh = p16.shape[:2]
    map = (np.sqrt(p16.astype(np.float64) / 65535.0) * 255.0 + 0.5).astype(np.uint8)   # корень: тонкое различимо
    shp = max(12, min(22, sh // 40))
    font = _font(shp)
    field = 8
    share = above / max(1, in_ * sh)
    t1 = f'толщина, мм — {what}; шкала одна на все холсты: 0 чёрный, {_h(SCALE_THICKNESSES_MM, 1)} и толще белый, серый — корень'
    t2 = f'наибольшая {_h(max_mm, 2)} мм; толще шкалы — {_h(100 * share, 1)} % места'
    W = max(sh, 420, int(math.ceil(max(font.getlength(t1), font.getlength(t2)))) + 2 * field)
    bottom = 3 * shp + 4 * field + 14
    sheet = Image.new('L', (W, in_ + bottom), 255)
    sheet.paste(Image.fromarray(map), (0, 0))
    d = ImageDraw.Draw(sheet)
    x0, x1 = field, W - field
    y = in_ + field
    band = np.tile(np.linspace(0, 255, x1 - x0).astype(np.uint8), (14, 1))
    sheet.paste(Image.fromarray(band), (x0, y))
    d.rectangle((x0 - 1, y - 1, x1, y + 14), outline=0)
    for mm in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, SCALE_THICKNESSES_MM):     # серый — корень доли шкалы
        x = x0 + (x1 - x0 - 1) * math.sqrt(mm / SCALE_THICKNESSES_MM)
        d.line((x, y + 14, x, y + 20), fill=0)
        t = _h(mm, 2 if mm < 0.1 and mm else 1)
        d.text((min(max(x - font.getlength(t) / 2, 0), x1 - font.getlength(t)), y + 21), t, fill=0, font=font)
    d.text((x0, y + 24 + shp), t1, fill=0, font=font)
    d.text((x0, y + 28 + 2 * shp), t2, fill=0, font=font)
    where_to = os.path.splitext(os.fspath(path))[0] + '.легенда.png'
    sheet.save(where_to)
    return where_to


def views(canvas, region=None, journal=None, without=None, to=None, size=500, path=None, light=True, target=None):
    """К-6: холст в девяти разрезах одним вызовом — один лист PNG (три в ряд; первый ряд — массы, тон, толщина:
    М-2) и массивы клетками. Клетка k =
    ⌈длинная сторона области / размер⌉ px холста; разрез — среднее по клетке. Разрезы (ключи `массивы`):
      'прищур' — L, размытый на ~1/60 длинной стороны (глаз прищурен), 'массы' — он же пятью ступенями тона 0–4 (равные
          доли от мин до макс прищура), на листе — серым;
      'тон' — L серой шкалы 0–255; 'насыщенность' — хрома C OKLab; 'тепло' — проекция a, b OKLab на ось тона 60°
          (> 0 — к оранжевому, тёплое; < 0 — к синему, холодное; на листе ±0,06 — полный цвет);
      'края' — перепад L на 4 клетки (|L(+2) − L(−2)|, по оси, где больше), 'жёсткие', 'мягкие' — bool: край — перепад
          ≥ 8 L; жёсткий — больше 60 % перепада на один шаг клетки (переход уже ~2 клеток), иначе мягкий (на листе
          жёсткие белым, мягкие голубым);
      'толщина' — рельеф всех слоёв (`х.рельеф`, им считается свет), мм; 'толщина слоёв' — {слой: толщина слоя, мм}
          (Б-4; единица h — `ММ_НА_H` = 0,4 мм); 'сырость' — наибольшая по слоям влажность × покрытие, 0–1;
      'направление' — угол хода верхнего мазка в клетке, ° 0–180 (0 — вдоль x, 90 — вдоль y вниз, 45 — от левого
          верха к правому низу; NaN — мазка нет), 'ширина мазка', 'длина мазка' (px), 'мазок' — номер строки журнала
          верхнего мазка (0 — нет): по журналу холста (`х.журнал`) или `журнал` — путь .jsonl; путь хода — точки строки
          ("ctrl" / "points" / "core") ломаной, ширина — "size" или кисти; верхний — из самого верхнего слоя, чья
          краска кроет клетку на 0,3 и больше (мгла и сухой ход, что почти не легли, не закрывают нижний), в слое —
          позже по журналу. На листе
          направления — штрихи по клеткам 12 px поверх цвета угла; размеры — светлее шире.
    цель — образец (любой вид цели, см. модуль): десятой панелью его прищур теми же ступенями тона, что у холста,
    ключи 'прищур цели' (вне поля цели NaN), 'массы цели' (вне поля −1), 'есть цель' (bool: клетка целиком в поле
    цели); прищур цели размыт только по своему полю, вне его на листе — серая штриховка «нет цели»; в сводке
    'прищур − цель' — среднее и σ разности прищуров по полю цели, L, 'есть цель' — его доля. Цель мимо области —
    ValueError.
    Вид холста — `взгляд` (свет, сила как у него); без, до — без слоёв и только ниже слоя (Б-5), и в мазках тоже.
    Холст только читается. путь — PNG листа. Ответ — `Виды`: `лист`, `массивы`, `сводка` (клетка, мазков, массы —
    доли ступеней, L p5/p50/p95, C ср, доли тёплого и холодного, доли жёстких и мягких краёв, толщина мм ср/макс,
    доля сырого > 0,5, направления — доли четырёх секторов)."""
    from PIL import Image, ImageDraw
    if journal is None:
        reading = getattr(canvas, '_журнал_для_чтения', None)    # М-10 (а): и журнал, взятый из .npz для чтения
        journal = None if reading is None else reading()
    canvas = without_layers(canvas, without, to)
    frame = _frame(canvas, region)
    x0, y0, x1, y1 = frame
    k = max(1, int(math.ceil(max(x1 - x0, y1 - y0) / max(1, int(size)))))
    kind = look(canvas, frame, light=light)
    linear = _cells(_LAndN8[kind], k)
    ny, nx = linear.shape[:2]
    if ny < 5 or nx < 5:
        raise ValueError(f'виды: область {frame} при клетке {k} px — {nx}×{ny} клеток, нужно от 5×5: размер больше')
    frame_with = (x0, y0, x0 + nx * k, y0 + ny * k)
    lab = oklab(_srgb(linear))
    L = _L(lab[..., 0]).astype(np.float32)
    C = np.hypot(lab[..., 1], lab[..., 2]).astype(np.float32)
    t = (lab[..., 1] * math.cos(math.radians(WARMTH_H)) + lab[..., 2] * math.sin(math.radians(WARMTH_H))).astype(np.float32)
    r = max(1, int(round(max(nx, ny) / 120)))

    def squint(L):
        for _ in range(3):
            L = _blur(L, r)
        return L.astype(np.float32)

    squint = squint(L)
    lo, hi = float(squint.min()), float(squint.max())
    steps = lambda p: np.clip(np.floor((p - lo) / max(hi - lo, 1e-6) * 5), 0, 4).astype(np.int8)
    masses = steps(squint)
    drop, stiff, soft = _edges(L, 8.0)
    gy, gx = canvas._region(frame_with)
    thickness = _cells(canvas.relief(gy.start, gy.stop, gx.start, gx.stop), k) * MM_ON_H
    by_layer, wet, covers = {}, np.zeros((ny, nx), np.float32), {}
    for name, with_ in canvas.layers.items():
        by_layer[name] = _cells(np.asarray(with_.thickness[gy, gx]), k) * MM_ON_H
        p = np.asarray(with_.coverage[gy, gx])
        covers[name] = _cells(p, k)
        wet = np.maximum(wet, _cells(np.asarray(with_.moisture[gy, gx]) * p, k))
    strokes = 0
    if journal is not None:
        angle, width, length, number, strokes = _sweeps(_strokes_journal(journal), list(canvas.layers), frame_with, k, (ny, nx), covers)
    else:
        angle, width, length = (np.full((ny, nx), np.nan, np.float32) for _ in range(3))
        number = np.zeros((ny, nx), np.int64)
    arrays = {'прищур': squint, 'массы': masses, 'тон': L, 'насыщенность': C, 'тепло': t, 'края': drop,
               'жёсткие': stiff, 'мягкие': soft, 'толщина': thickness, 'толщина слоёв': by_layer, 'сырость': wet,
               'направление': angle, 'ширина мазка': width, 'длина мазка': length, 'мазок': number}
    # ---- панели
    p_mass = np.repeat((40.0 + masses * 47.5)[..., None], 3, -1)
    p_tone = _gray(L, 0, 255)
    cmax = max(0.02, float(np.percentile(C, 99)))
    p_us = _gray(C, 0, cmax)
    tn = np.clip(t / 0.06, -1, 1)[..., None]
    p_warm = 128.0 + np.where(tn > 0, tn * (np.array([255, 140, 30]) - 128.0), -tn * (np.array([40, 110, 255]) - 128.0))
    p_kr = np.zeros((ny, nx, 3))
    p_kr[soft] = np.array([90, 170, 255]) * np.clip(drop[soft] / 40.0, 0.35, 1.0)[:, None]
    p_kr[stiff] = 255.0
    tmax = max(0.05, float(np.percentile(thickness, 99.5)))
    p_thick = _gray(thickness, 0, tmax)
    p_wet = 255.0 - np.clip(wet, 0, 1)[..., None] * (255.0 - np.array([30, 80, 200]))
    has = ~np.isnan(angle)
    ug = np.radians(np.nan_to_num(angle))
    p_filling = np.full((ny, nx, 3), 235.0)
    if has.any():
        hue = color(np.full(angle.shape, 200.0), np.full(angle.shape, 0.07), np.degrees(2 * ug))
        p_filling[has] = hue[has]
    im = Image.fromarray(_in_8(p_filling))
    d = ImageDraw.Draw(im)
    s = 12
    c2, s2 = np.where(has, np.cos(2 * ug), 0.0), np.where(has, np.sin(2 * ug), 0.0)
    for i in range(0, ny - s // 2, s):
        for j in range(0, nx - s // 2, s):
            n_ = has[i:i + s, j:j + s].sum()
            if n_ < s * s / 4:
                continue
            a = 0.5 * math.atan2(s2[i:i + s, j:j + s].sum(), c2[i:i + s, j:j + s].sum())
            cx, cy, dx, dy = j + s / 2, i + s / 2, 0.45 * s * math.cos(a), 0.45 * s * math.sin(a)
            d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(0, 0, 0), width=1)
    p_filling = np.asarray(im, np.float64)
    wmax = float(np.nanmax(width)) if has.any() or np.isfinite(width).any() else 1.0
    p_times = np.where(np.isfinite(width)[..., None], _gray(np.log(np.maximum(np.nan_to_num(width, nan=1.0), 1.0)),
                                                         0.0, math.log(max(wmax, 2.0))), np.array([255.0, 230, 230]))
    p5, p50, p95 = (float(v) for v in np.percentile(L, [5, 50, 95]))
    signatures = [f'прищур: массы 5 ступеней, L {_h(lo, 0)}–{_h(hi, 0)}', 'тон L 0–255', f'толщина 0–{_h(tmax, 2)} мм',
               f'насыщенность C 0–{_h(cmax, 3)}', 'тепло-холод ±0,06', 'края: белые жёсткие, голубые мягкие',
               'сырость 0–1 (синее — сырее)',
               'направления мазков' + ('' if journal else ': журнала нет'),
               f'ширина мазка 1–{_h(wmax, 0)} px (log)' if journal else 'размеры мазков: журнала нет']
    panels = [p_mass, p_tone, p_thick, p_us, p_warm, p_kr, p_wet, p_filling, p_times]   # М-2: толщина — в первом ряду
    if target is not None:
        tsa = _target(target, frame_with, light, outside=np.nan)
        given = ~np.isnan(tsa[..., 0])
        covered = _cells(given.astype(np.float32), k)[:ny, :nx] > 0.999      # клетка целиком в поле цели
        if not covered.any():
            raise ValueError(f'виды: цель не покрывает ни одной клетки области {frame_with}: мерить только там, где цель есть')
        ts = _cells(_LAndN8[_in_8(np.where(given[..., None], tsa, GRAY_OUTSIDE))], k)[:ny, :nx]
        m = covered.astype(np.float32)
        pts = squint(np.where(covered, _L(oklab(_srgb(ts))[..., 0]), 0.0)) / np.maximum(squint(m), 1e-6)
        pts = np.where(covered, pts, np.nan).astype(np.float32)          # прищур только по полю цели, вне — NaN
        mts = np.where(covered, steps(np.nan_to_num(pts, nan=lo)), -1).astype(np.int8)
        arrays['прищур цели'], arrays['массы цели'], arrays['есть цель'] = pts, mts, covered
        pmts = np.repeat((40.0 + np.maximum(mts, 0) * 47.5)[..., None], 3, -1)
        yy, xx = np.mgrid[0:ny, 0:nx]
        pmts[~covered] = np.where((((xx + yy) // 4) % 2 == 0)[~covered, None], 150.0, 205.0)   # нет цели — штриховка
        panels.append(pmts)
        signatures.append('прищур цели: те же ступени' + ('' if covered.all() else '; штрих — нет цели'))
    sheet = _sheet(panels, signatures, path)
    # ---- сводка
    total = float(ny * nx)
    nu = angle[has]
    sector = [float(np.mean((nu >= a - 22.5) & (nu < a + 22.5) | ((nu >= a + 157.5) if a == 0 else False))) if nu.size else 0.0
               for a in (0, 45, 90, 135)]
    summary = {'клетка': k, 'клеток': (ny, nx), 'мазков': strokes,
              'массы': [round(float(np.mean(masses == i)), 3) for i in range(5)],
              'L': (round(p5, 1), round(p50, 1), round(p95, 1)), 'C': round(float(C.mean()), 4),
              'тепло': round(float(np.mean(t > 0.01)), 3), 'холод': round(float(np.mean(t < -0.01)), 3),
              'жёсткие': round(float(stiff.sum() / total), 3), 'мягкие': round(float(soft.sum() / total), 3),
              'толщина мм': (round(float(thickness.mean()), 3), round(float(thickness.max()), 3)),
              'сыро': round(float(np.mean(wet > 0.5)), 3),
              'направления': dict(zip(('0°', '45°', '90°', '135°'), (round(v, 3) for v in sector))),
              'с мазком': round(float(has.mean()), 3)}
    if target is not None:
        dp = (squint - arrays['прищур цели'])[covered]
        summary['прищур − цель'] = (round(float(dp.mean()), 2), round(float(dp.std()), 2))
        summary['есть цель'] = round(float(covered.mean()), 3)
    return Views(sheet, arrays, summary, frame_with, k)


# ---------------------------------------------------------------- Б-10 издали и вблизи

def overview(canvas, target, region=None, crops=4, window=300, size=800, path=None, light=True, without=None, to=None):
    """Б-10: издали и вблизи одним листом. Сверху — область целиком, длинная сторона `размер` px, на ней рамками с
    номерами — `кропов` (3–5) мест наибольшего расхождения с целью: квадраты стороной `окно` px холста по наибольшему
    |среднему dL| (`разность`, тон без света, не налегают); ниже — эти места 1:1, «холст | цель», по два в ряд, над
    каждым номер, рамка и dL. свет — у картинок листа (как у `взгляд`); без, до — Б-5. путь — PNG листа. Ответ —
    {'лист' (uint8), 'худшие' [(x0, y0, x1, y1, dL)], 'сводка' (сводка `разность`)}."""
    from PIL import Image, ImageDraw
    canvas = without_layers(canvas, without, to)
    frame = _frame(canvas, region)
    x0, y0, x1, y1 = frame
    r = difference(canvas, target, frame, window=min(int(window), x1 - x0, y1 - y0), worst=int(crops))
    m = min(1.0, size / max(x1 - x0, y1 - y0))
    from_afar = Image.fromarray(look(canvas, frame, scale=m, light=light))
    d = ImageDraw.Draw(from_afar)
    font = _font(16)
    for i, (a, b, c, e, _) in enumerate(r.summary['худшие'], 1):
        q = [(a - x0) * m, (b - y0) * m, (c - x0) * m, (e - y0) * m]
        d.rectangle(q, outline=(255, 0, 0), width=2)
        d.text((q[0] + 3, q[1] + 1), str(i), fill=(255, 0, 0), font=font)
    crops = [Image.fromarray(look(canvas, w[:4], light=light, beside=target)) for w in r.summary['худшие']]
    shk, vk = (max(brush.width for brush in crops), max(brush.height for brush in crops)) if crops else (0, 0)
    field, header = 6, 24
    sh = max(from_afar.width, 2 * shk + field) + 2 * field
    in_ = from_afar.height + 2 * field + ((len(crops) + 1) // 2) * (vk + header + field)
    sheet = Image.new('RGB', (sh, in_), (255, 255, 255))
    sheet.paste(from_afar, (field, field))
    d = ImageDraw.Draw(sheet)
    for i, (brush, w) in enumerate(zip(crops, r.summary['худшие'])):
        x, y = field + (i % 2) * (shk + field), from_afar.height + 2 * field + (i // 2) * (vk + header + field)
        d.text((x + 2, y + 3), f'{i + 1}: {w[:4]}  dL {_h(w[4])}  холст | цель', fill=(0, 0, 0), font=font)
        sheet.paste(brush, (x, y + header))
    if path is not None:
        sheet.save(os.path.expanduser(str(path)))
    return {'лист': np.asarray(sheet), 'худшие': r.summary['худшие'], 'сводка': r.summary}


# ---------------------------------------------------------------- М-6 сетка и укладка образца

AtBrushLADBrushAnd = ('как есть', 'вписать', 'обрезать', 'растянуть')


def lay_out(reference, width, height, laying='как есть'):
    """М-6: образец (путь к картинке или массив (В, Ш, 3) sRGB 0–255) в координатах холста ширина × высота — массив
    (высота, ширина, 3) float64 sRGB и словарь укладки. укладка: 'как есть' — пиксель образца = пиксель холста от угла
    (0, 0), как `рядом` у `взгляд` (где образца нет — серый 128); 'вписать' — целиком, пропорция та же, поля серым по
    центру; 'обрезать' — покрывает холст, пропорция та же, лишнее по краям срезано поровну; 'растянуть' — в размер
    холста, пропорция образца теряется. Словарь: 'укладка', 'масштаб' (sx, sy — px холста на px образца), 'сдвиг' (dx, dy):
    точка образца (u, v) — на холсте (dx + u·sx, dy + v·sy); 'образец' — его ширина и высота. Ответ годится как
    `рядом=` и цель глаз. Холст не нужен и не трогается."""
    from PIL import Image
    if laying not in AtBrushLADBrushAnd:
        raise ValueError(f'укладка {laying!r}: есть {", ".join(repr(at) for at in AtBrushLADBrushAnd)}')
    a = _load_file(reference) if isinstance(reference, (str, os.PathLike)) else np.asarray(reference)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError(f'уложить: образец — картинка (В, Ш, 3) sRGB 0–255 или путь к ней; дано {a.shape}')
    a = np.asarray(a[..., :3], np.float64)
    vo, sho = a.shape[:2]
    width, height = int(width), int(height)
    p = np.full((height, width, 3), float(GRAY_OUTSIDE))
    if laying == 'как есть':
        sx = sy = 1.0
        dx = dy = 0.0
        p[:min(vo, height), :min(sho, width)] = a[:height, :width]
    else:
        if laying == 'растянуть':
            sx, sy = width / sho, height / vo
        else:
            s = (min if laying == 'вписать' else max)(width / sho, height / vo)
            sx = sy = s
        nsh, nv = max(1, int(round(sho * sx))), max(1, int(round(vo * sy)))
        dx, dy = (width - nsh) / 2.0, (height - nv) / 2.0
        im = Image.fromarray(_in_8(a)).resize((nsh, nv), Image.LANCZOS)
        b = np.asarray(im, np.float64)
        x0, y0 = int(math.floor(dx)), int(math.floor(dy))
        # кусок b, что ложится на холст
        bx0, by0 = max(0, -x0), max(0, -y0)
        cx0, cy0 = max(0, x0), max(0, y0)
        w = min(nsh - bx0, width - cx0)
        h = min(nv - by0, height - cy0)
        p[cy0:cy0 + h, cx0:cx0 + w] = b[by0:by0 + h, bx0:bx0 + w]
        dx, dy = float(x0), float(y0)
    return p, {'укладка': laying, 'масштаб': (round(sx, 6), round(sy, 6)), 'сдвиг': (dx, dy), 'образец': (sho, vo)}


def _spacing_grid(length):
    for sh in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000):
        if length / sh <= 12:
            return sh
    return 10000


def _draw_grid(a8, frame, spacing, scale, font):
    """Линии каждые `шаг` px холста — пунктир белый/чёрный (виден на любом тоне), подписи в px холста с обводкой."""
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = frame
    im = Image.fromarray(a8)
    d = ImageDraw.Draw(im)
    In, Sh = a8.shape[:2]
    dashed = (np.arange(max(In, Sh)) // 4) % 2 == 0
    a = np.asarray(im).copy()
    xs = [x for x in range(int(math.ceil(x0 / spacing)) * spacing, x1, spacing)]
    ys = [y for y in range(int(math.ceil(y0 / spacing)) * spacing, y1, spacing)]
    for x in xs:
        with_ = int(round((x - x0) * scale))
        if 0 <= with_ < Sh:
            a[:, with_] = np.where(dashed[:In, None], 255, 0)
    for y in ys:
        with_ = int(round((y - y0) * scale))
        if 0 <= with_ < In:
            a[with_, :] = np.where(dashed[:Sh, None], 255, 0)
    im = Image.fromarray(a)
    d = ImageDraw.Draw(im)
    for x in xs:
        with_ = int(round((x - x0) * scale))
        if 0 <= with_ < Sh:
            d.text((with_ + 3, 2), str(x), fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    for y in ys:
        with_ = int(round((y - y0) * scale))
        if 0 <= with_ < In and with_ > 0:
            d.text((3, with_ + 2), str(y), fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.asarray(im)


def grid(canvas, reference=None, spacing=None, region=None, scale=1.0, laying='как есть', beside=False, path=None, light=True):
    """М-6: сетка в px холста с подписями — поверх холста (образец None) или образца, уложенного на холст (`уложить`:
    'как есть' — как `рядом` у `взгляд`, 'вписать', 'обрезать', 'растянуть'); рядом=True — «холст | образец», обе с
    сеткой. шаг — px холста между линиями (None — круглое, ~6–12 линий на длинную сторону области); область, масштаб, свет
    — как у `взгляд`: линия x — левый край пикселя x холста. Только картинка: холст, журнал и сиды не трогает. Ответ —
    uint8 ряд; путь — PNG; у ответа (`Ряд`) поле `укладка` — словарь `уложить` (или None без образца)."""
    from .canvas import Row
    frame = _frame(canvas, region)
    x0, y0, x1, y1 = frame
    spacing = _spacing_grid(max(x1 - x0, y1 - y0)) if spacing is None else int(spacing)
    if spacing < 1:
        raise ValueError('сетка: шаг — px холста, от 1')
    parts, uk = [], None
    if reference is None or beside:
        parts.append(_kind(canvas, frame, light))
    if reference is not None:
        response, uk = lay_out(reference, canvas.full_width, canvas.full_height, laying)
        parts.append(response[y0:y1, x0:x1])
    font = _font(max(11, min(16, int(12 * max(1.0, scale)))))
    frames = [_draw_grid(_scale(_in_8(h), scale), frame, spacing, scale, font) for h in parts]
    if len(frames) > 1:
        field = np.full((frames[0].shape[0], 6, 3), 255, np.uint8)
        row = np.concatenate([frames[0], field, frames[1]], 1)
    else:
        row = frames[0]
    if path is not None:
        from PIL import Image
        Image.fromarray(row).save(os.path.expanduser(str(path)))
    row = np.ascontiguousarray(row).view(Row)
    row.laying = uk
    row.hints = []
    return row
