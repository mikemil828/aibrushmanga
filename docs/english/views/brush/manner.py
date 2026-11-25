# READING VIEW of движок/кисть/манера.py with names in English. Not runnable; edit the original.
"""К-4: мерка манеры по картинке — у образца живописи и у холста одной линейкой, и разница между ними.

Всё, что меряется у образца, меряется по картинке, и тем же счётом — у холста. Картинка
сначала приводится к рабочему масштабу `РАБОЧИЙ` px/мм (по px/мм картины), поэтому миллиметры у обоих — одни.

- **палитра** — цвет, размытый на 0,5 мм, ступенями OKLab `СТУПЕНЬ` по L, a, b: `'замесов'` — сколько ступеней
  покрывают 90 % площади, `'эфф'` — exp(энтропия) долей, `'главные'` — до 8 ступеней (доля, '#rrggbb');
- **тон в пятне** — σ светлоты L (серая шкала 0–255) вокруг размытого на пятно `ПЯТНО` мм, по пикселю: p10, p50, p90;
  **цвет в пятне** — то же по a, b OKLab (длина отклонения);
- **края** — L клетками 0,5 мм, перепад на 2 мм не меньше `ПОРОГ_КРАЯ` L (`глаз._края`): `'доля'` клеток-краёв,
  `'жёстких'` — доля жёстких среди краёв (больше 60 % перепада на один шаг 0,5 мм), `'мягких'`;
- **направления** — тензор структуры светлоты в окне `окно` мм (места шагом пол-окна): направление хода — поперёк
  перепада, ° 0–180 (0 — вдоль x, 90 — вдоль y вниз), `'порядок'` — когерентность 0–1 (1 — все штрихи окна
  параллельны), сектора по 30° с долями (вес — порядок), `'эфф'`, `'у соседей'` — медиана разности направлений
  соседних мест, °;
- **штрих** — длина и ширина штриха в каждом месте: на скольких мм корреляция фактуры (светлота минус размытая на
  пятно) вдоль направления места и поперёк падает до половины (сдвиги `ЛАГИ` 0,5–16 мм): p10, p50, p90 мм, σ/ср,
  `'эфф'` — exp(энтропия) октав (ступени вдвое, как у `однообразие`), `'на полу'` — доля мест короче первого сдвига
  0,5 мм: где их больше половины, штрих в разницу не идёт («уже 0,5 мм — не различается»);
- **ширина мазка** — пятно-ход поперёк направления места: хорды между краями (перепад светлоты и цвета на ±1 мм от
  `ПОРОГ_КРАЯ`) по линиям через 2 мм поперёк хода; край, по обе стороны которого один замес (`ЗАМЕС`, `ЗАМЕС_L`),
  хорду не рвёт — ходы одного замеса внахлёст меряются одним пятном; хорда — мазок, если её края идут лентой вдоль
  хода (на соседней линии те же края ±0,5 мм) и внутри есть фактура мельче `ФАКТУРА` (не ровнее `ФОН` L: ровный
  грунт, зазор между ходами или заливка — фон, не мазок); верно ~2–15 мм, шкала `ШИРИНА_МАЗКА` 1–20 мм: p10, p50, p90, σ/ср, `'эфф'` октав, `'хорд'`, `'шире'` — доля хорд шире 20 мм,
  `'фон'` — доля площади без фактуры. Мазки на штуки не разбираются;
- **предел** — сколько мм занимают `разбор.ПРЕДЕЛ_PX` px снимка: признак мельче не различается и говорится словом, а не
  числом (растянутый до 4 px/мм снимок своего пикселя не прячет);
- **мазки разбором** (Мн-1, Мн-1б, `разбор.py`) — мазки-кандидаты по лестнице уменьшений снимка: длина, ширина, длина к
  ширине — процентилями обычных мазков, и только если их подтверждает тот же снимок в √2 раз мельче (иначе слово);
  счёт мазков не мельче предела на дм², доли протяжек и ударов, доля площади без мазков; с образцом сравниваются на
  общем пределе;
- **края пятен** — доля прямых, ширина перехода края;
- у холста ещё **мазки** — по журналу (`однообразие`): длина и ширина мм, наклон — тем, кто писал.

Холст только читается. Случайных чисел нет."""
import math
import os

import numpy as np

from . import eye
from . import parsing as R
from . import fast
from .gauge_file import File

RABResponseHAndY = 4.0             # px/мм счёта
WithTAtPEN = 0.02            # ступень замеса в OKLab (та же, что «на глаз» у `однообразие`)
PYaTNResponse = 4.0               # мм: размер пятна для разброса тона внутри
THRESHOLD_EDGES = 8.0          # L: перепад, от которого место — край (как у `виды`)
LAGAnd = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)    # мм: сдвиги корреляции штриха; длиннее последнего — «от 16 мм»
PResponseRYaDResponseBrush = 0.15            # места с меньшей когерентностью в «у соседей» не идут
WIDTH_STROKE = (1.0, 20.0)  # мм: шкала ширины мазка по картинке
LAndNAndAnd = 2.0               # мм: шаг линий поперёк хода
FResponseN = 1.0                 # L: фактура (светлота минус размытая на 1 мм) ровнее — фон без мазка (доля 'фон')
FABrushTAtRA = 0.5             # мм: середина хорды — фон, если фактура мельче ФАКТУРА ровнее ФОН; мерой 1 мм край хода
                          # заходит в зазор грунта 2–3 мм между ходами, и зазор шёл мазком
RAZMYIn, PEREPAD = 0.5, 1.0  # мм: края хорд — по цвету, размытому на РАЗМЫВ, перепад на ±ПЕРЕПАД
ZAMEWith = 255 * WithTAtPEN     # a, b ×255: по обе стороны края цвет ближе — один замес, край хорду не рвёт
MIX_L = 2 * THRESHOLD_EDGES  # L: и светлота ближе (полоса набора, борозда ворса внутри хода)
FEW_CHORDS = 20            # хорд меньше — ширина мазка в разницу не идёт
FLOOR_CHORDS = 2.0           # мм: уже хорды не читаются (края — перепад на ±1 мм)
ON_HALF_CHORDS = 0.1        # доля хорд у пола, от которой у снимка ширина по хордам не называется (p10 — уже пол)
SPACING_EDGES = 0.5            # мм: шаг клетки жёстких краёв; предел снимка грубее — жёсткий край не различается
PResponseDTInERZhDENResponse = 0.2        # p50 длины и ширины мазков на снимке в √2 раз мельче — в этой доле: размер подтверждён картинкой
ON_HALF = 0.5             # доля мест на первом сдвиге, от которой штрих не различается


class Gauge:
    """Мерка одной картинки: `сводка` — словарь разделов ('палитра', 'тон в пятне', 'цвет в пятне', 'края',
    'направления', 'штрих', у холста ещё 'мазки'); `м['края']` — то же, что `м.сводка['края']`; `массивы` — карты
    рабочего масштаба для листа; `слова` — строкой (print); `откуда` — {'угол' меренного куска в px картинки, 'сторона' px картины,
    'предел', 'окно'}."""
    __slots__ = ('сводка', 'массивы', 'слова', 'что', 'px_на_мм', 'размер', 'откуда', 'файл')

    def album(self, path=None, places=None, cores=None):
        """Мн-3: самые непохожие места картинки ×2, под каждым — его числа и адрес в px; один PNG. Подробно — `кисть.альбом`."""
        from .album import album
        return album(self, None, path, places, cores)

    def __getitem__(self, option):
        return self.summary[option]

    def stroke(self, x, y):
        """Форма мазка разбора в точке (x, y) px меренного куска (у образца с областью — от её угла): `разбор.форма` — словарь
        с 'слово'; нет мазка — None."""
        r = self.arrays['разбор']
        S = r['метки']
        j = int(S[int(y), int(x)]) if 0 <= y < S.shape[0] and 0 <= x < S.shape[1] else 0
        return R.shape(r, j - 1) if j else None

    def __repr__(self):
        return self.words


class _Arrays(dict):
    """Карты мерки; 'картинка' и 'тон в пятне' мерки, прочитанной из файла, считаются из снимка, когда их спросят."""

    def __missing__(self, option):
        if option == 'картинка':
            self[option] = _brush_working(self['снимок'], self['px_на_мм'])
        elif option == 'тон в пятне':
            self[option] = _tone_in_patch(eye._L(fast.oklab(self['картинка'].astype(np.uint8))[..., 0]))[0]
        else:
            raise KeyError(option)
        return self[option]


class Manner:
    """Ответ `х.манера`: `холст`, `образец` — `Мерка` (образца может не быть), `разница` — {признак: {'холст',
    'образец', 'во сколько'}}, `слова` — отчёт строкой; `сводка` — сводка холста (`м['направления']`); `лист` — PNG-массив
    «картинка | направления | края и хорды ширины мазка | мазки разбором» для холста и образца; `места` — с образцом: клетки холста, где
    он пестрее или однообразнее образца (`места_манеры`), худшие первыми; без образца — []; `перевод` — с образцом: мазок
    образца в долях его стороны, переведённый в мм этого холста, и кисти пресетов рядом (`перевод_манеры`)."""
    __slots__ = ('холст', 'образец', 'разница', 'слова', 'лист', 'места', 'перевод')

    def album(self, path=None, places=None, cores=None):
        """Мн-3: самые непохожие места образца ×2 и те же места холста рядом, под каждым — его числа и адрес; один PNG. Без
        образца — места холста. Подробно — `кисть.альбом`."""
        from .album import album
        if self.reference is None:
            return album(self.canvas, None, path, places, cores)
        return album(self.reference, self.canvas, path, places, cores)

    @property
    def summary(self):
        return self.canvas.summary

    def __getitem__(self, option):
        return self.canvas.summary[option]

    def __repr__(self):
        return self.words


# ---------------------------------------------------------------- счёт

def _gauss(a, σ):
    from scipy import ndimage
    return ndimage.gaussian_filter(a, σ, mode='reflect') if σ > 0 else a


class _Blur:
    """Размытая светлота: одна σ считается один раз."""
    def __init__(self, L):
        self.L, self.was = L, {}

    def __call__(self, σ):
        if σ not in self.was:
            self.was[σ] = _gauss(self.L, σ)
        return self.was[σ]


def _brush_working(rgb, px_on_mm):
    """sRGB 0–255 (В, Ш, 3) при px_на_мм -> sRGB float рабочего масштаба."""
    from PIL import Image
    k = RABResponseHAndY / float(px_on_mm)
    a8 = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    if abs(k - 1.0) < 1e-6:
        return a8.astype(np.float64)
    sh, in_ = max(8, int(round(a8.shape[1] * k))), max(8, int(round(a8.shape[0] * k)))
    return np.asarray(Image.fromarray(a8).resize((sh, in_), Image.LANCZOS), np.float64)


def _eff(shares):
    p = np.asarray(shares, np.float64)
    p = p[p > 0] / p.sum()
    return float(math.exp(-(p * np.log(p)).sum())) if p.size else 0.0


def _pts(v, n=2):
    v = np.asarray(v, np.float64).ravel()
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    sr = float(v.mean())
    return {'p10': round(float(p10), n), 'p50': round(float(p50), n), 'p90': round(float(p90), n), 'ср': round(sr, n),
            'σ/ср': round(float(v.std()) / sr, 3) if sr > 1e-9 else None}


def _palette(rgb, lab):
    smoothed = np.stack([_gauss(lab[..., i], 0.5 * RABResponseHAndY) for i in range(3)], -1)
    option = np.floor(smoothed / WithTAtPEN).astype(np.int64).reshape(-1, 3)
    bottom = option.min(0)
    row = option.max(0) - bottom + 1
    sample = ((option[:, 0] - bottom[0]) * row[1] + option[:, 1] - bottom[1]) * row[2] + option[:, 2] - bottom[2]   # порядок — по L, a, b
    n = np.bincount(sample)
    has = np.flatnonzero(n)
    n = n[has]
    order = np.argsort(-n, kind='stable')
    shares = n[order] / n.sum()
    mix = int(np.searchsorted(np.cumsum(shares), 0.9) + 1)
    main = []
    sums = [np.bincount(sample, rgb[..., i].ravel())[has] for i in range(3)]      # целые: порядок суммы не важен
    for j in order[:8]:
        main.append((round(float(n[j] / n.sum()), 3), eye.hex(np.array([with_[j] for with_ in sums]) / n[j])))
    return {'замесов': mix, 'ступеней': int(len(n)), 'эфф': round(_eff(n), 1), 'главные': main}


def _directions(L, window_px, blurred=None):
    Lg = (blurred or _Blur(L))(0.5 * RABResponseHAndY)
    gy, gx = np.gradient(Lg)
    σ = window_px / 2.0
    spacing = max(1, int(window_px // 2))
    grid = fast.Grid(gx, σ, spacing // 2, spacing)
    Jxx, Jyy, Jxy = (grid.blur(v) for v in (gx * gx, gy * gy, gx * gy))
    drop = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    θ = (drop + 90.0) % 180.0                                       # ход — поперёк перепада
    with_ = np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / np.maximum(Jxx + Jyy, 1e-9)
    brush = np.floor(((θ + 15.0) % 180.0) / 30.0).astype(int)
    weights = np.bincount(brush.ravel(), weights=with_.ravel(), minlength=6)
    shares = weights / max(weights.sum(), 1e-12)
    main = [(int(i) * 30, round(float(shares[i]), 3)) for i in np.argsort(-shares, kind='stable')[:3] if shares[i] > 0]
    z = (with_ * np.exp(2j * np.radians(θ))).sum() / max(with_.sum(), 1e-12)
    R = float(abs(z))
    diff = []
    for a, b, ca, cb in ((θ[:, 2:], θ[:, :-2], with_[:, 2:], with_[:, :-2]), (θ[2:], θ[:-2], with_[2:], with_[:-2])):   # через окно
        d = np.abs(a - b) % 180.0
        d = np.minimum(d, 180.0 - d)
        diff.append(d[(ca > PResponseRYaDResponseBrush) & (cb > PResponseRYaDResponseBrush)])
    diff = np.concatenate(diff)
    summary = {'главные': main, 'эфф': round(_eff(weights), 2), 'порядок': round(float(np.median(with_)), 3),
              'σ': round(float(np.degrees(math.sqrt(max(-2 * math.log(max(R, 1e-12)), 0.0))) / 2), 1),
              'у соседей': round(float(np.median(diff)), 1) if diff.size else None, 'мест': int(θ.size)}
    return summary, θ, with_, spacing


def _streak(L, θ, window_px, spacing, limit=0.0, blurred=None):
    """Длина и ширина штриха в каждом месте: расстояние, на котором корреляция фактуры (светлота минус размытая на
    пятно) вдоль направления места и поперёк него падает до половины; сдвиги — `ЛАГИ` мм по 6 направлениям через 30°."""
    blurred = blurred or _Blur(L)
    h = blurred(0.25 * RABResponseHAndY) - blurred(PYaTNResponse * RABResponseHAndY / 2.0)
    σ = window_px / 2.0
    grid = fast.Grid(h, σ, spacing // 2, spacing)
    E = np.maximum(grid.shift(0, 0), 1e-9)
    C = np.ones((6, len(LAGAnd)) + E.shape)
    along_i = np.round(θ / 30.0).astype(int) % 6
    across_i = (along_i + 3) % 6
    for i in range(6):
        u = math.radians(30 * i)
        waiting = [along_i == i, across_i == i]                         # места, где корреляция ещё не упала до половины
        for j, d in enumerate(LAGAnd):
            nodes = waiting[0] | waiting[1]
            if not nodes.any():
                break
            dx, dy = int(round(d * RABResponseHAndY * math.cos(u))), int(round(d * RABResponseHAndY * math.sin(u)))
            C[i, j] = grid.shift(dy, dx, nodes) / E
            waiting = [m & (C[i, j] >= 0.5) for m in waiting]
    lag = np.log(np.asarray(LAGAnd))

    def to_half(c):                                    # c (лаги, …) -> мм, где корреляция падает до 0,5
        below = c < 0.5
        first = np.where(below.any(0), np.argmax(below, 0), len(LAGAnd))
        response = np.full(c.shape[1:], LAGAnd[-1] * 1.0)
        for j in range(len(LAGAnd)):
            where = first == j
            if j == 0:
                response[where] = LAGAnd[0]                     # короче первого сдвига — «до 0,5 мм»
            else:
                a, b = c[j - 1][where], c[j][where]
                t = np.clip((a - 0.5) / np.maximum(a - b, 1e-9), 0, 1)
                response[where] = np.exp(lag[j - 1] + t * (lag[j] - lag[j - 1]))
        return response
    dl = to_half(np.take_along_axis(C, along_i[None, None], 0)[0])
    width = to_half(np.take_along_axis(C, across_i[None, None], 0)[0])
    log2 = lambda v: np.floor(np.log2(v) + 1e-9).astype(int).ravel()
    with_ = {'длина': _pts(dl), 'ширина': _pts(width)}
    for brush, v in (('длина', dl), ('ширина', width)):
        with_[brush]['эфф'] = round(_eff(np.unique(log2(v), return_counts=True)[1]), 2)
        with_[brush]['на полу'] = round(float((v <= max(LAGAnd[0], limit) + 1e-9).mean()), 3)
        with_[brush]['пол'] = max(LAGAnd[0], round(limit, 2))
        if limit > LAGAnd[0] and with_[brush]['на полу'] > ON_HALF:               # пиксель снимка, а не штрих: числа нет
            with_[brush].update({and_: None for and_ in ('p10', 'p50', 'p90', 'ср', 'σ/ср', 'эфф')})
    with_['до'] = LAGAnd[-1]
    return with_, dl, width


def _width_stroke(L, lab, θ, cohort, spacing, limit=0.0, blurred=None):
    """Ширина мазка по картинке: хорды между краями по линиям поперёк хода места (через `ЛИНИИ` мм, 6 направлений
    через 30°). Край — местный максимум перепада |c(t + 1 мм) − c(t − 1 мм)| светлоты L и a, b OKLab ×255 от
    `ПОРОГ_КРАЯ` (как у «края»), по цвету, размытому на `РАЗМЫВ`. Края, по обе стороны которых один замес (`скоро.замесы`, порог `ЗАМЕС_L`),
    хорду не рвут: ходы одного замеса внахлёст меряются одним пятном. Хорда идёт в счёт, если место у её середины — с ходом этого направления (порядок
    от `ПОРЯДОК`), её края есть и на соседней линии (±0,5 мм — лента вдоль хода) и фактура мельче `ФАКТУРА` в середине хорды
    не ровнее `ФОН`."""
    from scipy import ndimage
    In, Sh = L.shape
    blurred = blurred or _Blur(L)
    channel = [blurred(RAZMYIn * RABResponseHAndY)] + [_gauss(lab[..., i] * 255.0, RAZMYIn * RABResponseHAndY) for i in (1, 2)]
    texture, fine = (np.sqrt(_gauss((blurred(0.25 * RABResponseHAndY) - blurred(r * RABResponseHAndY)) ** 2, r / 2 * RABResponseHAndY))
                for r in (1.0, FABrushTAtRA))
    k = int(round(PEREPAD * RABResponseHAndY))
    extra = int(round(0.5 * RABResponseHAndY))
    lo, hi = WIDTH_STROKE
    R = math.hypot(In, Sh)
    t = np.arange(-R / 2, R / 2, 1.0)
    of = np.arange(-R / 2, R / 2, LAndNAndAnd * RABResponseHAndY)
    width, wider, chords = [], 0, []
    for i in range(6):
        u = math.radians(30 * i + 90)                                  # линия — поперёк хода 30·i°
        ux, uy = math.cos(u), math.sin(u)
        row = of[np.abs(of) <= 0.5 * (In * abs(ux) + Sh * abs(uy)) + 1.0, None]   # прочие линии картинки не задевают
        X = Sh / 2 + row * -uy + t[None] * ux
        Y = In / 2 + row * ux + t[None] * uy
        inside = (X >= 0) & (X <= Sh - 1) & (Y >= 0) & (Y <= In - 1)
        usable = np.zeros_like(inside)
        usable[:, k:-k] = inside[:, 2 * k:] & inside[:, :-2 * k]
        g = np.flatnonzero(usable)
        square, profile = 0.0, []
        for c in channel:
            p = ndimage.map_coordinates(c, [Y.ravel(), X.ravel()], order=1, mode='nearest')
            profile.append(p)
            square = square + (p[g + k] - p[g - k]) ** 2
        drop = np.zeros(X.shape)
        drop.ravel()[g] = np.sqrt(square)
        peak = (drop >= THRESHOLD_EDGES) & (drop >= ndimage.maximum_filter1d(drop, 2 * k + 1, axis=1))
        peak[:, 1:] &= ~peak[:, :-1]                                     # из подряд — первый
        row, e = np.nonzero(peak)
        n = X.shape[1]
        live = fast.mixes(profile, row, e, k, n, ZAMEWith, MIX_L)
        row, e = row[live], e[live]
        r, a, b = row[:-1], e[:-1], e[1:]
        da = (row[1:] == r) & ((b - a) / RABResponseHAndY >= lo)
        r, a, b = r[da], a[da], b[da]
        m = (a + b) // 2
        gi = np.clip(np.rint((Y[r, m] - spacing // 2) / spacing).astype(np.int64), 0, θ.shape[0] - 1)
        gj = np.clip(np.rint((X[r, m] - spacing // 2) / spacing).astype(np.int64), 0, θ.shape[1] - 1)
        da = (np.rint(θ[gi, gj] / 30.0).astype(np.int64) % 6 == i) & (cohort[gi, gj] >= PResponseRYaDResponseBrush)
        r, a, b = r[da], a[da], b[da]
        beside = np.zeros((peak.shape[0] + 2, n), bool)                  # где на линии край ближе `доп`
        beside[1:-1] = ndimage.maximum_filter1d(peak.astype(np.uint8), 2 * extra + 1, axis=1, mode='constant') > 0
        da = (beside[r, a] & beside[r, b]) | (beside[r + 2, a] & beside[r + 2, b])      # строка q — в `рядом` строкой q + 1
        r, a, b = r[da], a[da], b[da]
        if not len(r):
            continue
        dl = b - a > 2 * k + 2
        with0, with1 = np.where(dl, a + k, a), np.where(dl, b - k + 1, b + 1)
        whose = np.repeat(np.arange(len(r)), with1 - with0)
        number = r[whose] * n + with0[whose] + np.arange(len(whose)) - np.r_[0, np.cumsum(with1 - with0)[:-1]][whose]
        middle = ndimage.map_coordinates(fine, [Y.ravel()[number], X.ravel()[number]], order=1, mode='nearest')
        da = fast.medians(middle, whose, len(r)) >= FResponseN
        r, a, b = r[da], a[da], b[da]
        w = (b - a) / RABResponseHAndY
        wider += int((w > hi).sum())
        da = w <= hi
        w, r, a, b = w[da], r[da], a[da], b[da]
        width += list(w)
        chords += list(zip(w, X[r, a], Y[r, a], X[r, b], Y[r, b]))
    width = np.asarray(width)
    with_ = _pts(width) if width.size else {'p10': None, 'p50': None, 'p90': None, 'ср': None, 'σ/ср': None}
    with_['эфф'] = round(_eff(np.unique(np.floor(np.log2(width) + 1e-9), return_counts=True)[1]), 2) if width.size else None
    with_['хорд'] = int(width.size)
    with_['шире'] = round(wider / max(width.size + wider, 1), 3)
    with_['фон'] = round(float((texture < FResponseN).mean()), 3)
    with_['шкала'] = WIDTH_STROKE
    floor = max(FLOOR_CHORDS, limit)
    with_['пол'] = round(floor, 2)
    with_['на полу'] = round(float((width <= floor + 0.25).mean()), 3) if width.size else 0.0
    if limit > LAGAnd[0] and with_['на полу'] > ON_HALF_CHORDS:                 # низ шкалы и пиксель снимка, а не мазок
        with_.update({and_: None for and_ in ('p10', 'p50', 'p90', 'ср', 'σ/ср', 'эфф')})
    return with_, chords


def _share(v, w, q):
    """Взвешенный квантиль: у мазков с весом `w` доля `q` веса лежит не выше ответа."""
    i = np.argsort(v, kind='stable')
    c = np.cumsum(w[i])
    return float(v[i][min(int(np.searchsorted(c, q * c[-1])), len(i) - 1)])


def _pts_in(v, w, n=2):
    return {and_: round(_share(v, w, q), n) for and_, q in (('p10', 0.1), ('p50', 0.5), ('p90', 0.9))}


def _integers(r):
    """Номера обычных мазков разбора и их медиана ширины по площади: целых (без полей), не у́же её половины. Мало — None."""
    response = r['области']
    ts = np.flatnonzero(~response['у края'] & (response['вес'] > 0) & r['мазки'])
    if len(ts) < R.FEW_STROKES:
        return None
    return ts[response['ширина'][ts] >= 0.5 * _share(response['ширина'][ts], (response['вес'] * response['площадь'])[ts], 0.5)]


def _sizes(r, px, side):
    """Обычный мазок разбора: длина, ширина, длина к ширине — процентилями по числу целых мазков (без полей) не у́же половины
    ширины, на которую приходится половина площади мазков: мельче — клочки, и сколько их, решает предел снимка, а не картина.
    'в долях' — длина и ширина в долях длинной стороны картины (`сторона` px), 'в ширинах' — в ширинах кисти (p50 ширины):
    от масштаба мм не зависят. 'размах' — на сколько медиана может стоять иначе из-за малого числа мазков (σ ранга медианы
    при их весах), доля. Мало мазков — None."""
    g = _integers(r)
    if g is None:
        return None
    d, sh, weight, pl = (r['области'][and_][g] for and_ in ('длина', 'ширина', 'вес', 'площадь'))
    range = 0.0
    for v, w in ((d, weight), (sh, weight), (sh, weight * pl)):             # и порог W/2 стоит на медиане по площади
        σ = 0.5 * math.sqrt(float((w * w).sum())) / float(w.sum())
        m = _share(v, w, 0.5)
        range = max(range, *(abs(_share(v, w, min(max(0.5 + z * σ, 0.0), 1.0)) / m - 1.0) for z in (-1, 1)))
    W = _share(sh, weight, 0.5)
    return {'длина': _pts_in(d / px, weight), 'ширина': _pts_in(sh / px, weight), 'длина к ширине': _pts_in(d / sh, weight), 'размах': range,
            'в долях': {'длина': _pts_in(d / side, weight, 5), 'ширина': _pts_in(sh / side, weight, 5)},
            'в ширинах': {'длина': _pts_in(d / W, weight), 'ширина': _pts_in(sh / W, weight)}}


RAZMERY = ('длина', 'ширина', 'длина к ширине', 'в долях', 'в ширинах')
SHAPE_CLOSE = 0.1        # ширины: p50 формы на снимке в √2 раз мельче — в этом или в `ПОДТВЕРЖДЕНО` доле: форма подтверждена


def _shape(f):
    """Форма обычного мазка по замерам `Р.формы`: вход и сход — длина и тупость p10–p50–p90 (ширин кисти), доли концов по
    именам (`рука.торец`), прядей на рваном, изгиб. Мазков с видными концами меньше `Р.МАЛО_МАЗКОВ` — None."""
    from . import hand
    f = [z for z in f if z['W'] >= hand.InAndDNResponse and z['длина к ширине'] >= hand.BrushResponseRResponseTResponseBrush]
    if len(f) < R.FEW_STROKES:
        return None
    weight = np.array([z['вес'] for z in f])
    with_ = {'мазков': len(f), 'изгиб': _pts_in(np.array([z['изгиб'] for z in f]), weight, 3)}
    for and_ in ('вход', 'сход'):
        with_[and_] = {brush: _pts_in(np.array([z[and_][brush] for z in f]), weight, 3) for brush in ('длина', 'тупость')}
        with_[and_]['сужен'] = round(float((weight * [hand.narrowed(z[and_]) for z in f]).sum() / weight.sum()), 3)
        t = np.array([z['W'] >= hand.VISIBLE_BUTT_END for z in f])    # торцы — только у мазков, где зуб шире предела снимка
        with_[and_]['торцы'] = None
        if t.sum() >= R.FEW_STROKES:
            names = np.array([hand.butt_end(z[and_], z['W']) for z in f])
            with_[and_]['торцы'] = {n: round(float((weight * t * (names == n)).sum() / (weight * t).sum()), 3)
                             for n in ('прямой', 'круглый', 'сужен', 'рваный')}
    with_['торцы видны у'] = int(sum(z['W'] >= hand.VISIBLE_BUTT_END for z in f))
    return with_


def _shape_diverged(a, b):
    """Что в форме на снимке мельче вышло другим: строка «сход 0,3 вместо 0,5 ширины» или ''."""
    no = []
    for and_, brush, word in (('вход', 'длина', 'вход'), ('сход', 'длина', 'сход'),
                        ('вход', 'тупость', 'тупость входа'), ('сход', 'тупость', 'тупость схода')):
        v, w = a[and_][brush]['p50'], b[and_][brush]['p50']
        if abs(v - w) > max(SHAPE_CLOSE, PResponseDTInERZhDENResponse * v):
            no.append(f'{word} {_h(w, 2)} вместо {_h(v, 2)} ширины')
    return ', '.join(no)


def _why(size, size2, range, prev):
    """Чем размер не подтвердился на снимке в √2 раз мельче — только то, что на самом деле разошлось. Про предел — когда
    ширина мазка у него (полоса самого мелкого уровня разбора) и ушла вместе с ним."""
    if size2 is None:
        return f'на снимке в 1,4 раза мельче целых мазков меньше {R.FEW_STROKES}'
    went = {and_: size2[and_]['p50'] / size[and_]['p50'] - 1.0 for and_ in ('ширина', 'длина')}
    not_ = [and_ for and_ in went if abs(went[and_]) > PResponseDTInERZhDENResponse]
    than = ', '.join(f'{and_} {_h(size2[and_]["p50"])} вместо {_h(size[and_]["p50"])} мм' for and_ in not_ or
                    [max(went, key=lambda and_: abs(went[and_]))])
    response = f'на снимке в 1,4 раза мельче {than}'
    if not not_:                                                    # по отдельности в доле, не сошлась сумма ошибок
        response += (f' ({_h(100 * max(abs(v) for v in went.values()), 0)} %), и мазков мало — медиана гуляет на '
                  f'{_h(100 * range, 0)} %; вместе больше {_h(100 * PResponseDTInERZhDENResponse, 0)} %')
    elif range > 0.5 * PResponseDTInERZhDENResponse:
        response += f'; мазков мало — медиана гуляет на {_h(100 * range, 0)} %'
    if went['ширина'] > PResponseDTInERZhDENResponse and size['ширина']['p50'] <= prev * R.PResponseLResponseWithA[1] / R.PResponseLResponseWithA[0]:
        response = f'ширина у предела снимка ({_h(prev)} мм) и уходит вместе с ним: {response}'
    return response


def _analysis(snapshot, px_on_mm, limit=None, side=None):
    """Мазки-кандидаты и края пятен снимка: (сводка мазков, сводка краёв, разбор). `предел` мм грубее предела снимка —
    лестница разбора кончается на нём (у двух картинок разного разрешения — один счёт). `сторона` px — длинная сторона всей
    картины, если снимок — её кусок. Размер — число, только если его подтверждает сама картинка: тот же разбор снимка в √2
    раз мельче (предел — во столько же раз грубее) даёт p50 длины и ширины в `ПОДТВЕРЖДЕНО` и медиана не держится на горстке
    областей; иначе — слово с тем, что не сошлось. Форма — так же, своим подтверждением (`_форма`)."""
    px = float(px_on_mm)
    prev_px = max(float(limit or 0.0) * px, R.LIMIT_PX)
    prev = prev_px / px
    side = float(side or max(snapshot.shape[:2]))
    r = R.parse(snapshot, prev_px)
    r['px_на_мм'] = px
    response = r['области']
    r['мазки'] = R.normal_width(response)[1]
    everything = float(snapshot.shape[0] * snapshot.shape[1])
    weight = response['вес'] * r['мазки']
    number = float(weight.sum())
    with_ = {'мазков': int(round(number)), 'предел': round(prev, 3), 'слово': None, 'сторона': round(side / px, 1),
         'на дм²': round(number / (everything / (px * 100.0) ** 2), 1),
         'без мазков': round(float(np.clip(1.0 - (weight * response['площадь']).sum() / everything, 0.0, 1.0)), 3),
         'поля': round(float(np.clip((response['вес'] * ~r['мазки'] * response['площадь']).sum() / everything, 0.0, 1.0)), 3)}
    wipe = float(weight[response['вид'] == 'протяжка'].sum())
    with_['доли'] = {'протяжек': round(wipe / max(number, 1e-9), 3), 'ударов': round(1.0 - wipe / max(number, 1e-9), 3)}
    with_.update({and_: None for and_ in RAZMERY + ('форма',)})
    size = _sizes(r, px, side)
    integers = _integers(r)
    r['формы'] = [] if integers is None else R.shape(r, integers)
    shape = _shape(r['формы'])
    r2, size2 = None, None
    if (size is not None and size['размах'] <= PResponseDTInERZhDENResponse) or shape is not None:
        # снимок мельче и предел его грубее: размер, который идёт от предела, уйдёт вместе с ним
        finer = R._brush_level(r['картинка'], math.sqrt(2.0))
        r2 = R.parse(finer, prev_px) if min(finer.shape[:2]) >= R.SIDE_MIN else None
        if r2 is not None:
            r2['мазки'] = R.normal_width(r2['области'])[1]
            size2 = _sizes(r2, px / math.sqrt(2.0), side / math.sqrt(2.0))
    if size is None:
        with_['слово'] = (f'целых мазков {int((~response["у края"] & (response["вес"] > 0) & r["мазки"]).sum())} — мало, размеры не называются; мазок уже '
                      f'{_h(prev)} мм или ближе к соседу, чем ступень краски, не различается')
    elif size['размах'] > PResponseDTInERZhDENResponse:
        with_['слово'] = (f'размер держится на горстке крупных областей (медиана гуляет на {size["размах"] * 100:.0f} %): мазков, '
                      'по которым назвать размер, мало — размеры не числом')
    else:
        went = None if size2 is None else max(abs(size2[and_]['p50'] / size[and_]['p50'] - 1.0) for and_ in ('длина', 'ширина'))
        if went is not None:                                      # две независимые ошибки: от предела и от малого числа мазков
            went = math.hypot(went, size['размах'])
        with_['подтверждено'] = None if went is None else round(went, 3)
        if went is None or went > PResponseDTInERZhDENResponse:
            with_['слово'] = f'размер мазка не подтвердился: {_why(size, size2, size["размах"], prev)} — размеры не числом'
        else:
            with_.update({and_: size[and_] for and_ in RAZMERY})
    with_['слово формы'] = None
    if shape is None:
        with_['слово формы'] = f'мазков с видными концами меньше {R.FEW_STROKES}'
    else:
        ts2 = _integers(r2) if r2 is not None else None
        f2 = _shape(R.shape(r2, ts2)) if ts2 is not None else None
        diverged = 'мазков с видными концами мало' if f2 is None else _shape_diverged(shape, f2)
        if diverged:
            with_['слово формы'] = f'не подтвердилась: на снимке в 1,4 раза мельче {diverged}'
        else:
            with_['форма'] = shape
    if prev > R.LIMIT_PX / px * (1 + 1e-6):                             # края — на снимке, уменьшенном до общего предела
        snapshot = R._brush_level(r['картинка'], prev_px / R.LIMIT_PX)
        px = R.LIMIT_PX / prev
    brush = R.edges_patches(snapshot)
    p = brush['переход']
    edge = {'прямых': None if brush['прямых'] is None else round(brush['прямых'], 3), 'границ': round(brush['границ'], 4),
            'переход': None, 'на полу': None, 'пол': round(prev, 3)}
    if p.size:
        edge['на полу'] = round(float((p <= R.LIMIT_PX).mean()), 3)
        if edge['на полу'] <= ON_HALF:
            edge['переход'] = round(float(np.median(p)) / px, 2)
    return with_, edge, r


def _tone_in_patch(L, blurred=None):
    """Разброс светлоты вокруг размытого на пятно: (карта, p10–p50–p90)."""
    σp = PYaTNResponse * RABResponseHAndY / 2.0
    blurred = blurred or _Blur(L)
    Lf, Lc = blurred(0.25 * RABResponseHAndY), blurred(σp)
    tone = np.sqrt(np.maximum(_gauss((Lf - Lc) ** 2, σp), 0.0))
    return tone, {brush: v for brush, v in _pts(tone).items() if brush != 'σ/ср'}


def _edges_places(L, px_on_mm):
    """Края по клеткам 0,5 мм: (доли, карта жёстких, карта мягких); снимок грубее `ШАГ_КРАЯ` жёстких от мягких не различает."""
    _, stiff, soft = eye._edges(eye._cells(L, 2), THRESHOLD_EDGES)
    edges = stiff | soft
    with_ = {'доля': round(float(edges.mean()), 3), 'жёстких': round(float(stiff.sum() / max(edges.sum(), 1)), 3),
         'мягких': round(float(soft.sum() / max(edges.sum(), 1)), 3)}
    if R.LIMIT_PX / float(px_on_mm) > SPACING_EDGES:
        with_.update({'жёстких': None, 'мягких': None})
    return with_, stiff, soft


def gauge(image, px_on_mm=None, window=8.0, what='картинка', region=None, light=True, limit=None,
          side=None, file=True):
    """К-4: мерка манеры картинки — sRGB 0–255 (В, Ш, 3), путь к PNG/JPG, `Холст` или путь к `.npz`; `px_на_мм` —
    сколько px на мм картины (у холста — его, у картинки без него — 8 и слово об этом в отчёте), `область` — кусок
    (x0, y0, x1, y1) в px картины, `свет` — холст со светом рельефа (как на снимке картины), `окно` — мм, окно
    направлений и штриха; `предел` — мм: мазки разбором не мельче этого (сравнить два снимка разного разрешения —
    с пределом грубого, `м['предел']`); `сторона` — px длинной стороны всей картины, если картинка — её вырезка
    (с `область` и у холста — сама): от неё доли мазка. Ответ — `Мерка`; годится в `х.манера(образец=…)` и `х.однообразие(образец=…)`.
    Мерка картинки, данной путём, пишется рядом с ней файлом и в другой раз читается из него (`мерка_файл`); `файлом=False` — нет."""
    from .canvas import Canvas
    from .hints import function_taken
    function_taken('мерка')
    file = None
    if isinstance(image, (str, bytes)) or hasattr(image, '__fspath__'):
        path = os.fsdecode(image)
        image = eye._load_file(path)
        file = File(path) if file and not isinstance(image, Canvas) else None
    said, angle = '', (0, 0)
    if isinstance(image, Canvas):
        angle = eye._frame(image, region)[:2]
        px_on_mm = px_on_mm or image.px_on_mm or 8.0
        side = side or max(image.width, image.height)
        rgb = eye._kind(image, eye._frame(image, region), light)
    else:
        rgb = np.asarray(image, np.float64)
        side = side or max(rgb.shape[:2])
        if region is not None:
            x0, y0, x1, y1 = (int(v) for v in region)
            rgb, angle = rgb[y0:y1, x0:x1], (x0, y0)
        if not px_on_mm:
            px_on_mm, said = 8.0, ' (масштаб не дан — взято 8 px/мм)'
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f'мерка: картинка — (В, Ш, 3) sRGB 0–255, а пришло {rgb.shape}')
    snapshot = np.clip(np.round(rgb[..., :3]), 0, 255).astype(np.uint8)
    limit_mm = R.LIMIT_PX / float(px_on_mm)
    args = [float(px_on_mm), float(window), None if region is None else [int(v) for v in region],
              None if limit is None else float(limit), float(side)]
    m = file and _from_file(file, args, snapshot, float(px_on_mm), what, said)
    if m:
        return m
    rgb = _brush_working(rgb[..., :3], px_on_mm)
    window_px = window * RABResponseHAndY
    if min(rgb.shape[:2]) < window_px:
        raise ValueError(f'мерка: картинка {rgb.shape[1] / RABResponseHAndY:.0f}×{rgb.shape[0] / RABResponseHAndY:.0f} мм меньше окна '
                         f'{window} мм — окно= меньше или кусок больше')
    lab = fast.oklab(rgb.astype(np.uint8))                       # `_к_рабочему` — целые
    L = eye._L(lab[..., 0])
    with_ = {'палитра': _palette(rgb, lab)}
    σp = PYaTNResponse * RABResponseHAndY / 2.0
    blurred = _Blur(L)
    tone, with_['тон в пятне'] = _tone_in_patch(L, blurred)
    ab = 0.0
    for i in (1, 2):
        ab = ab + (_gauss(lab[..., i], 0.25 * RABResponseHAndY) - _gauss(lab[..., i], σp)) ** 2
    ab = np.sqrt(np.maximum(_gauss(ab, σp), 0.0))
    with_['цвет в пятне'] = {brush: v for brush, v in _pts(ab, 4).items() if brush != 'σ/ср'}
    with_['края'], stiff, soft = _edges_places(L, px_on_mm)
    with_['предел'] = round(limit_mm, 3)
    with_['направления'], θ, cohort, spacing = _directions(L, window_px, blurred)
    with_['штрих'], dl, width = _streak(L, θ, window_px, spacing, limit_mm, blurred)
    with_['ширина мазка'], chords = _width_stroke(L, lab, θ, cohort, spacing, limit_mm, blurred)
    with_['мазки разбором'], with_['края пятен'], parsing = _analysis(snapshot, px_on_mm, limit, side)
    m = Gauge()
    m.file = file
    m.summary, m.what, m.px_on_mm, m.size = with_, what, float(px_on_mm), (rgb.shape[1] / RABResponseHAndY, rgb.shape[0] / RABResponseHAndY)
    m.from_where = {'угол': tuple(int(v) for v in angle), 'сторона': float(side), 'предел': limit, 'окно': float(window)}
    m.arrays = _Arrays({'картинка': rgb, 'направление': θ, 'порядок': cohort, 'шаг': spacing, 'жёсткие': stiff, 'мягкие': soft,
                 'тон в пятне': tone, 'длина': dl, 'ширина': width, 'хорды': chords, 'снимок': snapshot, 'разбор': parsing, 'px_на_мм': float(px_on_mm)})
    if file:
        itself = ('картинка', 'тон в пятне', 'снимок', 'px_на_мм', 'хорды', 'разбор')
        file.write('мерка', args, {
            'сводка': with_, 'размер': m.size, 'откуда': m.from_where, 'массивы': {and_: v for and_, v in m.arrays.items() if and_ not in itself},
            'хорды': np.array(chords, np.float64).reshape(-1, 5), 'разбор': {and_: v for and_, v in parsing.items() if and_ != 'картинка'}})
    m.words = _words_gauges(m) + said
    return m


def _from_file(file, args, snapshot, px_on_mm, what, said):
    """Мерка из файла рядом с картинкой или None."""
    d = file.take('мерка', args)
    try:
        if d is None or d['разбор']['метки'].shape != snapshot.shape[:2]:
            return None
        m = Gauge()
        m.file, m.what, m.px_on_mm = file, what, px_on_mm
        m.summary, m.size, m.from_where = d['сводка'], d['размер'], d['откуда']
        m.arrays = _Arrays(d['массивы'], chords=[tuple(canvas) for canvas in d['хорды']], snapshot=snapshot,
                             parsing=dict(d['разбор'], image=snapshot), px_on_mm=px_on_mm)
        m.words = _words_gauges(m) + said
        return m
    except (KeyError, TypeError, AttributeError):                  # не наш файл
        return None


def _parsing_on_limit(m, limit, side):
    """Сводки мазков и краёв пятен мерки `м` на другом пределе; у мерки с файлом — тоже файлом."""
    file, snapshot = getattr(m, 'файл', None), m.arrays['снимок']
    x, y = m.from_where['угол']
    args = [m.px_on_mm, [x, y, x + snapshot.shape[1], y + snapshot.shape[0]], float(limit), float(side)]
    response = file and file.take('мазки', args)
    if not response:
        response = _analysis(snapshot, m.px_on_mm, limit, side)[:2]
        if file:
            file.write('мазки', args, response)
    return response


# ---------------------------------------------------------------- слова и разница

def _h(v, n=1):
    if v is None:
        return '—'
    v = 0.0 if abs(v) < 0.5 * 10 ** -n else float(v)                  # без «-0»
    return f'{v:.{n}f}'.replace('.', ',')


def _words_streak(brush, v):
    if v.get('на полу', 0) > ON_HALF:
        return f'{brush} уже {_h(v.get("пол", LAGAnd[0]))} мм в {_h(100 * v["на полу"], 0)} % мест — не различается'
    return f'{brush} {_h(v["p10"])}–{_h(v["p90"])} мм (σ/ср {_h(v["σ/ср"], 2)}, эфф {_h(v["эфф"])})'


def _words_stroke(in_):
    background = f', фон без фактуры {_h(100 * in_["фон"], 0)} % места' if in_['фон'] >= 0.05 else ''
    if not in_['хорд']:
        return f'ширина мазка по картинке: лент между краями нет{background}'
    if in_['p50'] is None:
        return (f'ширина мазка по хордам: {_h(100 * in_["на полу"], 0)} % хорд не шире {_h(in_["пол"])} мм — снимок слишком мелкий, '
                f'не различается{background}')
    few = f' — мало, в разницу не идёт' if in_['хорд'] < FEW_CHORDS else ''
    wider = f', шире {_h(in_["шкала"][1], 0)} мм — {_h(100 * in_["шире"], 0)} % хорд' if in_['шире'] >= 0.05 else ''
    return (f'ширина пятна-хода по хордам {_h(in_["p10"])}–{_h(in_["p50"])}–{_h(in_["p90"])} мм (p10–p50–p90, σ/ср '
            f'{_h(in_["σ/ср"], 2)}, хорд {in_["хорд"]}{few}{wider}{background})')


def _words_analysis(r, brush):
    d, tr = r['доли'], lambda v, n=1: f'{_h(v["p10"], n)}–{_h(v["p50"], n)}–{_h(v["p90"], n)}'
    count = (f'не мельче {_h(r["предел"])} мм — {r["мазков"]}, на дм² {_h(r["на дм²"], 0)}; протяжек {_h(100 * d["протяжек"], 0)} %, '
            f'ударов {_h(100 * d["ударов"], 0)} %; без мазков {_h(100 * r["без мазков"], 0)} % площади, из них поля (шире '
            f'{_h(R.PResponseLE, 0)} ширин мазка) — {_h(100 * r["поля"], 0)} %')
    if r['слово']:
        mz = f'мазки разбором: {r["слово"]}; {count}'
    else:
        mz = (f'мазки разбором: длина {tr(r["длина"])} мм, ширина {tr(r["ширина"])} мм (p10–p50–p90), '
              f'длина к ширине {_h(r["длина к ширине"]["p50"])}; в ширинах кисти (p50 ширины) длина {tr(r["в ширинах"]["длина"])}, ширина '
              f'{tr(r["в ширинах"]["ширина"], 2)}; от длинной стороны картины {_h(r["сторона"], 0)} мм длина '
              f'{_h(100 * r["в долях"]["длина"]["p50"], 2)} %, ширина {_h(100 * r["в долях"]["ширина"]["p50"], 2)} %; {count}')
    mz += f'; {_words_shape(r)}'
    if brush['прямых'] is None:
        return mz + '; краёв пятен нет'
    per = (f'переход {_h(brush["переход"])} мм' if brush['переход'] is not None else
           f'переход у {_h(100 * brush["на полу"], 0)} % не шире {_h(brush["пол"])} мм — резче не различается')
    return f'{mz}; края пятен: прямых {_h(100 * brush["прямых"], 0)} %, {per}'


def _words_shape(r):
    f = r['форма']
    if f is None:
        return f'форма мазка: {r["слово формы"]}'
    t = []
    for and_ in ('вход', 'сход'):
        brush = f[and_]
        names = ('' if brush['торцы'] is None else
                 ' (' + ', '.join(f'{n} {_h(100 * v, 0)} %' for n, v in brush['торцы'].items() if v >= 0.005) + ')')
        t.append(f'{and_} {_h(brush["длина"]["p50"])} (p90 {_h(brush["длина"]["p90"])}), тупость {_h(brush["тупость"]["p50"], 2)}, сужен у '
                 f'{_h(100 * brush["сужен"], 0)} %{names}')
    visible = (f'; торцы по именам — у {f["торцы видны у"]} мазков' if f['вход']['торцы'] is not None else
             '; рваный торец от ровного не различается: мазки уже 12 px снимка')
    return (f'форма мазка в ширинах кисти ({f["мазков"]} мазков; вход — где больше сплошной краски, рука по картинке не видна): '
            f'{"; ".join(t)}; изгиб {_h(f["изгиб"]["p50"], 2)} длины{visible}')


def _words_gauges(m):
    with_ = m.summary
    p, t, ts, brush, n, sh = (with_[k] for k in ('палитра', 'тон в пятне', 'цвет в пятне', 'края', 'направления', 'штрих'))
    parts = [f'{m.what} {_h(m.size[0], 0)}×{_h(m.size[1], 0)} мм',
             f'замесов {p["замесов"]} (90 % площади; ступеней {p["ступеней"]}, эфф {_h(p["эфф"])})',
             f'тон в пятне σ {_h(t["p50"])} L (p10–p90 {_h(t["p10"])}–{_h(t["p90"])}), цвет в пятне {_h(ts["p50"], 3)}',
             f'края {_h(100 * brush["доля"], 0)} % места, ' + (f'жёстких {_h(100 * brush["жёстких"], 0)} %' if brush['жёстких'] is not None else
                                                        f'жёсткие от мягких не различаются: снимок мельче {_h(SPACING_EDGES)} мм не видит'),
             'направления ' + ', '.join(f'{g}°±15 — {_h(100 * d, 0)} %' for g, d in n['главные'][:2])
             + f' (эфф {_h(n["эфф"])}, порядок {_h(n["порядок"], 2)}, у соседей {_h(n["у соседей"], 0)}°)',
             'штрих фактуры: ' + ', '.join(_words_streak(brush, sh[brush]) for brush in ('длина', 'ширина')),
             _words_analysis(with_['мазки разбором'], with_['края пятен']), _words_stroke(with_['ширина мазка'])]
    if 'мазки' in with_:
        mz = with_['мазки']
        if mz.get('мазков'):
            parts.append(f'мазки журнала {mz["мазков"]}: длина {_h(mz["длина"]["p10"])}–{_h(mz["длина"]["p90"])} мм '
                         f'(σ/ср {_h(mz["длина"]["σ/ср"], 2)}), ширина {_h(mz["ширина"]["p10"])}–{_h(mz["ширина"]["p90"])}'
                         f' мм (σ/ср {_h(mz["ширина"]["σ/ср"], 2)}), наклон σ {_h(mz["наклон"]["σ"], 0)}°')
    return '; '.join(parts)


# признак: (раздел, ключ, что больше — шире, единица, знаков)
PRAndZNABrushAnd = {'замесов': ('палитра', 'замесов', 'палитра', '', 0),
            'тон в пятне': ('тон в пятне', 'p50', 'разброс тона в пятне', ' L', 1),
            'цвет в пятне': ('цвет в пятне', 'p50', 'разброс цвета в пятне', '', 3),
            'края': ('края', 'доля', 'доля краёв', '', 3),
            'жёстких': ('края', 'жёстких', 'доля жёстких краёв', '', 2),
            'наклон': ('направления', 'эфф', 'разброс наклона (эфф секторов 30°)', '', 2),
            'у соседей': ('направления', 'у соседей', 'расхождение наклона соседних мест', '°', 0),
            'порядок': ('направления', 'порядок', 'параллельность штрихов в окне', '', 2),
            'длина штриха': ('штрих', ('длина', 'p50'), 'длина штриха', ' мм', 1),
            'разброс длины': ('штрих', ('длина', 'σ/ср'), 'разброс длины штриха (σ/ср)', '', 2),
            'варианты длины': ('штрих', ('длина', 'эфф'), 'варианты длины штриха (эфф октав)', '', 2),
            'ширина штриха': ('штрих', ('ширина', 'p50'), 'ширина штриха', ' мм', 1),
            'разброс ширины': ('штрих', ('ширина', 'σ/ср'), 'разброс ширины штриха (σ/ср)', '', 2),
            'ширина мазка': ('ширина мазка', 'p50', 'ширина мазка (p50)', ' мм', 1),
            'разброс ширины мазка': ('ширина мазка', 'σ/ср', 'разброс ширины мазка (σ/ср)', '', 2),
            'мазков на дм²': ('мазки разбором', 'на дм²', 'мазков на дм²', '', 0),
            'длина мазка разбором': ('мазки разбором', ('длина', 'p50'), 'длина мазка', ' мм', 1),
            'ширина мазка разбором': ('мазки разбором', ('ширина', 'p50'), 'ширина мазка разбором', ' мм', 1),
            'длина к ширине': ('мазки разбором', ('длина к ширине', 'p50'), 'длина мазка к ширине', '', 1),
            'длина мазка в долях': ('мазки разбором', ('в долях', 'длина', 'p50'), 'длина мазка от стороны картины', '', 4),
            'ширина мазка в долях': ('мазки разбором', ('в долях', 'ширина', 'p50'), 'ширина мазка от стороны картины', '', 4),
            'сход мазка': ('мазки разбором', ('форма', 'сход', 'длина', 'p50'), 'сход мазка', ' ширины', 2),
            'тупость схода': ('мазки разбором', ('форма', 'сход', 'тупость', 'p50'), 'тупость схода', '', 2),
            'протяжек': ('мазки разбором', ('доли', 'протяжек'), 'доля протяжек', '', 2),
            'без мазков': ('мазки разбором', 'без мазков', 'площадь без мазков', '', 2),
            'прямых краёв': ('края пятен', 'прямых', 'доля прямых краёв пятен', '', 2),
            'переход края': ('края пятен', 'переход', 'переход края пятен', ' мм', 1)}
ON_SHARED = ('мазки разбором', 'края пятен')   # признаки, что сравниваются на общем разрешении двух картинок
VO_HOW_MUCH = 1.5          # от этого отношения отчёт говорит «уже» или «шире»


def _take(d, option):
    for brush in (option if isinstance(option, tuple) else (option,)):
        d = d.get(brush) if isinstance(d, dict) else None
    return d


def _shares_on_shared(canvas, response, shared, limit):
    """Сводки мазков для сравнения в долях стороны: предел — общий в долях, а не в мм (у картины 1 м и холста 25 см он в мм
    разный). Грубая картинка — её мерка; тонкая, если у грубой размер числом, — разбором с тем же пределом в долях."""
    st = [m['мазки разбором']['сторона'] for m in (canvas, response)]
    relative = [m['предел'] / with_ for m, with_ in zip((canvas, response), st)]
    and_ = int(relative[1] > relative[0])                                      # грубая
    response = [m['мазки разбором'] for m in (canvas, response)]
    if response[and_]['слово'] is None and relative[1 - and_] < 0.99 * relative[and_]:
        m = (canvas, response)[1 - and_]
        p = relative[and_] * st[1 - and_]
        response[1 - and_] = (shared[1 - and_]['мазки разбором'] if abs(p - limit) < 0.01 * limit else
                        _parsing_on_limit(m, p, st[1 - and_] * m.px_on_mm)[0])
    return response


def _difference(canvas, response):
    """Штрих, у которого на любой стороне больше `НА_ПОЛУ` мест на первом сдвиге, — 'во сколько' None и 'на полу';
    ширина мазка, где у любой стороны хорд меньше `МАЛО_ХОРД`, — 'во сколько' None и 'хорд'. Мазки разбором и края пятен —
    на общем пределе (грубом из двух): картинка резче меряется уменьшенной, и её числа здесь другие, чем в её мерке."""
    r = {}
    limit = max(canvas['предел'], response['предел'])
    shared = [m.summary if abs(m['предел'] - limit) < 0.01 * limit else
             dict(zip(ON_SHARED, _parsing_on_limit(m, limit, m['мазки разбором']['сторона'] * m.px_on_mm)))
             for m in (canvas, response)]
    shares = _shares_on_shared(canvas, response, shared, limit)
    for name, (section, option, *_ ) in PRAndZNABrushAnd.items():
        a, b = (_take((shared if section in ON_SHARED else m.summary)[section], option) for m, shared in zip((canvas, response), shared))
        if 'в долях' in option:
            a, b = (_take(d, option) for d in shares)
        k = (b / a) if a not in (None, 0) and b is not None else None
        r[name] = {'холст': a, 'образец': b, 'во сколько': round(k, 2) if k is not None else None}
        if section == 'ширина мазка':
            few = min(m.summary[section]['хорд'] for m in (canvas, response))
            if few < FEW_CHORDS:
                r[name].update({'во сколько': None, 'хорд': few})
        if section == 'штрих':
            floor = max(_take(m.summary['штрих'], (option[0], 'на полу')) or 0.0 for m in (canvas, response))
            if floor > ON_HALF:
                r[name].update({'во сколько': None, 'на полу': floor})
    return r


def _words_difference(r):
    already, wider = [], []
    for name, (_, _, word, ed, n) in PRAndZNABrushAnd.items():
        v = r[name]
        k = v['во сколько']
        if k is None:
            continue
        t = f'{word}: холст {_h(v["холст"], n)}{ed}, образец {_h(v["образец"], n)}{ed}'
        if k >= VO_HOW_MUCH:
            already.append(f'{t} — у образца в {_h(k)} раза больше')
        elif k == 0:
            wider.append(f'{t} — у образца нет вовсе')
        elif k <= 1 / VO_HOW_MUCH:
            wider.append(f'{t} — у холста в {_h(1 / k)} раза больше')
    parts = []
    if already:
        parts.append('у холста меньше, чем у образца: ' + '; '.join(already))
    if wider:
        parts.append('у холста больше, чем у образца: ' + '; '.join(wider))
    if not parts:
        parts.append(f'по всем признакам холст и образец ближе, чем в {_h(VO_HOW_MUCH)} раза')
    return '. '.join(parts)


# ---------------------------------------------------------------- холст

def _strokes(canvas, journal, frame, object, stage, layer):
    from . import monotony as Response
    path = journal if journal is not None else (canvas._journal_for_reading() if hasattr(canvas, '_журнал_для_чтения') else None)   # М-10 (а)
    if path is None:
        return None
    all = Response.strokes_journal(path)
    zone = Response.select(all, object, stage, layer, None if (object or stage or layer) else frame)
    if not zone:
        return {'мазков': 0}
    response = Response.report(zone)
    mm = 1.0 / float(getattr(canvas, 'px_на_мм', None) or 8.0)
    def in_mm(v):
        return {brush: (round(z * mm, 2) if brush in ('ср', 'σ', 'p10', 'p50', 'p90') and z is not None else z)
                for brush, z in v.items()}
    return {'мазков': len(zone), 'длина': in_mm(response['длина']), 'ширина': in_mm(response['ширина']), 'наклон': response['наклон']}


def _brushes():
    """Ширина мм кистей `пресеты/*.json` (px пресета — при 8 px/мм): {имя файла: мм}."""
    import json
    import os
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'пресеты')
    response = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith('.json'):
            with open(os.path.join(folder, f), encoding='utf-8') as p:
                response[f[:-5]] = json.load(p)['ширина'] / 8.0
    return response


def translation_manner(mo, side_mm, px_on_mm):
    """Мн-2: мазок образца (`Мерка` мо) на холсте с длинной стороной `сторона_мм`: 'длина' и 'ширина' мм p10–p50–p90 — те же
    доли стороны, что у образца; 'ширина px' p50; 'кисти' — {пресет: {'мм' его ширина, 'ширин' — сколько его ширин в p50
    ширины мазка}}. Только перевод чисел: кисть выбирает тот, кто пишет. Размер образца словом — {'слово'}."""
    r = mo['мазки разбором']
    response = {'слово': r['слово'], 'сторона': round(float(side_mm), 1), 'сторона образца': r['сторона']}
    if r['слово'] is not None:
        return response
    for and_ in ('длина', 'ширина'):
        response[and_] = {brush: round(v * side_mm, 2) for brush, v in r['в долях'][and_].items()}
    w = r['в долях']['ширина']['p50'] * side_mm
    response['ширина px'] = round(w * px_on_mm, 1)
    response['кисти'] = {brush: {'мм': round(mm, 2), 'ширин': round(w / mm, 2)} for brush, mm in _brushes().items()}
    return response


def _words_translation(p):
    if p['слово'] is not None:
        return 'перевод в мм холста: размер мазка образца словом — переводить нечего (кроп образца с мазками или снимок крупнее)'
    tr = lambda v: f'{_h(v["p10"], 2)}–{_h(v["p50"], 2)}–{_h(v["p90"], 2)}'
    brushes = ', '.join(f'{brush} {_h(v["мм"], 2)} мм — {_h(v["ширин"], 2)}' for brush, v in p['кисти'].items())
    return (f'мазок образца в долях стороны на этом холсте ({_h(p["сторона"], 0)} мм; образец {_h(p["сторона образца"], 0)} мм): ширина '
            f'{tr(p["ширина"])} мм, длина {tr(p["длина"])} мм; p50 ширины — {_h(p["ширина px"])} px холста; в ширинах кистей '
            f'пресетов: {brushes}')


def manner(canvas, region=None, reference=None, region_sample=None, px_on_mm_sample=None, object=None, stage=None,
           layer=None, journal=None, light=True, window=8.0, path=None, without=None, to=None):
    """К-4: мерка манеры холста в `область` (картинкой, со светом рельефа по умолчанию — как на снимке картины) и, если
    дан `образец` (путь к картинке, массив sRGB, `Холст` или `.npz`, или готовая `Мерка`), — его мерка той же линейкой и
    разница. `px_на_мм_образца` — сколько px на мм у картины образца (без него — как у холста, и отчёт это говорит);
    `область_образца` — кусок образца в его px. Мазки холста — по журналу (`объект`, `этап`, `слой` или центром в
    области). `путь` — PNG «картинка | направления | края | мазки» холста и образца. Холст только читается."""
    from .canvas import Canvas
    canvas = eye.without_layers(canvas, without, to)
    frame = eye._frame(canvas, region)
    px = float(getattr(canvas, 'px_на_мм', None) or 8.0)
    mkh = gauge(eye._kind(canvas, frame, light), px, window, 'холст', side=max(canvas.width, canvas.height))
    mkh.from_where['угол'] = tuple(int(v) for v in frame[:2])
    mkh.summary['мазки'] = _strokes(canvas, journal, frame, object, stage, layer) or {'мазков': None}
    mkh.words = _words_gauges(mkh)
    mo, said = reference, ''
    if reference is not None and not isinstance(reference, Gauge):
        path = isinstance(reference, (str, bytes)) or hasattr(reference, '__fspath__')
        pxo = px_on_mm_sample
        if not pxo and not isinstance(eye._load_file(os.fsdecode(reference)) if path else reference, Canvas):
            pxo, said = px, f' (масштаб образца не дан — взят как у холста, {_h(px, 1)} px/мм; px_на_мм_образца=)'
        mo = gauge(reference, pxo, window, 'образец', region_sample, light)
        mo.words += said
    m = Manner()
    m.canvas, m.reference = mkh, mo
    m.difference = _difference(mkh, mo) if mo is not None else {}
    m.places, sl = places_manner(mkh, mo, frame, px) if mo is not None else ([], '')
    m.translation = translation_manner(mo, max(canvas.width, canvas.height) / px, px) if mo is not None else {}
    m.words = mkh.words + ('.\n' + mo.words + '.\n' + _words_translation(m.translation) + '.\n' + sl + '.\n' + _words_difference(m.difference) if mo is not None else '')
    m.sheet = _sheet([mkh] + ([mo] if mo is not None else []), path)
    return m


# ---------------------------------------------------------------- по клеткам (зеркало с адресом)

CELL_MANNER = 50.0      # мм: клетка холста; у образца меньше клетки — клетка по его меньшей стороне
PLACES_MANNER = 8           # худших клеток в словах


def _by_cell(m, cell_mm):
    """Клетки мерки (рабочий масштаб): [(x0, y0, x1, y1 в мм от угла картинки, тон в пятне p50, замесов)]; клетка у
    края меньше половины — не идёт."""
    rgb, tone = m.arrays['картинка'], m.arrays['тон в пятне']
    in_, sh = tone.shape
    k = max(8, int(round(cell_mm * RABResponseHAndY)))
    lab = eye.oklab(rgb)
    result = []
    for y0 in range(0, in_, k):
        for x0 in range(0, sh, k):
            y1, x1 = min(in_, y0 + k), min(sh, x0 + k)
            if (y1 - y0) * (x1 - x0) < 0.5 * k * k:
                continue
            result.append((x0 / RABResponseHAndY, y0 / RABResponseHAndY, x1 / RABResponseHAndY, y1 / RABResponseHAndY, float(np.median(tone[y0:y1, x0:x1])),
                         _palette(rgb[y0:y1, x0:x1], lab[y0:y1, x0:x1])['замесов']))
    return result


def places_manner(mkh, mo, frame, px_on_mm, cell=CELL_MANNER):
    """Холст по клеткам `клетка` мм против образца той же клеткой (медиана его клеток): разброс тона в пятне (p50 клетки)
    и замесы (90 % площади клетки). Место — клетка, где холст пестрее или однообразнее образца в `ВО_СКОЛЬКО` раз и
    больше по любому из двух. Ответ — (места, слова): места — словари 'рамка' (px холста), 'тон в пятне' и 'замесов' —
    (холст, образец), 'во сколько' — образец / холст по каждому, 'слово' ('однообразнее' / 'пестрее'); худшие первыми."""
    kl = min(cell, mo.size[0], mo.size[1])
    response = _by_cell(mo, kl)
    if not response:
        return [], f'по клеткам: образец {_h(mo.size[0], 0)}×{_h(mo.size[1], 0)} мм — клеток нет'
    tone_response, measure_response = float(np.median([c[4] for c in response])), float(np.median([c[5] for c in response]))
    places = []
    for x0, y0, x1, y1, tone, measure in _by_cell(mkh, kl):
        kt = tone_response / max(tone, 1e-3)
        kz = measure_response / max(measure, 1)
        brush = max((kt, kz, 1 / max(kt, 1e-9), 1 / max(kz, 1e-9)))
        if brush < VO_HOW_MUCH:
            continue
        one = max(kt, kz) >= VO_HOW_MUCH
        motley = min(kt, kz) <= 1 / VO_HOW_MUCH
        word = 'однообразнее и пестрее' if one and motley else 'однообразнее' if one else 'пестрее'
        places.append({'рамка': (int(round(frame[0] + x0 * px_on_mm)), int(round(frame[1] + y0 * px_on_mm)),
                                int(round(frame[0] + x1 * px_on_mm)), int(round(frame[1] + y1 * px_on_mm))),
                      'тон в пятне': (round(tone, 2), round(tone_response, 2)), 'замесов': (measure, round(measure_response, 1)),
                      'во сколько': (round(kt, 2), round(kz, 2)), 'слово': word, '_к': brush})
    places.sort(key=lambda d: -d['_к'])
    for d in places:
        del d['_к']
    n_response = sum(1 for d in places if 'однообразнее' in d['слово'])
    n_p = sum(1 for d in places if 'пестрее' in d['слово'])
    h = [f'по клеткам {_h(kl, 0)} мм (образец — медиана своих клеток: тон в пятне {_h(tone_response)} L, замесов {_h(measure_response, 0)}): '
         f'однообразнее образца в {_h(VO_HOW_MUCH)} раза и больше — {n_response}, пестрее — {n_p}']
    for i, d in enumerate(places[:PLACES_MANNER], 1):
        kt, kz = d['во сколько']
        times = lambda k: f'в {_h(1 / k)} раза больше' if k < 1 else f'в {_h(k)} раза меньше'
        h.append(f'  {i}. рамка {d["рамка"]}: {d["слово"]} — тон в пятне {_h(d["тон в пятне"][0])} L ({times(kt)}), '
                 f'замесов {d["замесов"][0]} ({times(kz)})')
    return places, '\n'.join(h)


# ---------------------------------------------------------------- лист

PANEL = (420, 340)


def _fit_in(a, frame=PANEL):
    from PIL import Image
    sh, in_ = frame
    k = min(sh / a.shape[1], in_ / a.shape[0])
    im = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8))
    im = im.resize((max(1, int(a.shape[1] * k)), max(1, int(a.shape[0] * k))), Image.LANCZOS)
    field = Image.new('RGB', (sh, in_), (128, 128, 128))
    field.paste(im, (0, 0))
    return field, k


def _sheet(gauges, path):
    from PIL import ImageDraw
    panels, signatures = [], []
    for m in gauges:
        a = m.arrays
        map, k = _fit_in(a['картинка'])
        panels.append(np.asarray(map, np.float64))
        signatures.append(f'{m.what}: картинка')
        shadow, _ = _fit_in(a['картинка'] * 0.35 + 80)
        d = ImageDraw.Draw(shadow)
        θ, with_, spacing = a['направление'], a['порядок'], a['шаг']
        r = max(1, math.ceil(max(θ.shape) / 32))                        # не гуще 32 штрихов по длинной стороне
        for i in range(0, θ.shape[0], r):
            for j in range(0, θ.shape[1], r):
                cx, cy = (spacing // 2 + j * spacing) * k, (spacing // 2 + i * spacing) * k
                r = 0.5 * r * spacing * k * (0.3 + 0.7 * min(1.0, float(with_[i, j]) / 0.6))
                dx, dy = r * math.cos(math.radians(θ[i, j])), r * math.sin(math.radians(θ[i, j]))
                d.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=(255, 230, 120), width=1)
        panels.append(np.asarray(shadow, np.float64))
        signatures.append(f'{m.what}: направления (длина — порядок)')
        brush = np.zeros(a['жёсткие'].shape + (3,))
        brush[a['мягкие']] = (40, 70, 115)                              # края приглушены: поверх — хорды
        brush[a['жёсткие']] = (115, 115, 115)
        kr, _ = _fit_in(brush)
        d = ImageDraw.Draw(kr)
        lo, hi = WIDTH_STROKE
        for w, x0, y0, x1, y1 in a.get('хорды', ()):              # хорды ширины мазка: жёлтые узкие — пурпурные широкие
            t = min(1.0, math.log(w / lo) / math.log(hi / lo))
            d.line([(x0 * k, y0 * k), (x1 * k, y1 * k)], fill=(255, int(230 * (1 - t)), int(255 * t)), width=1)
        panels.append(np.asarray(kr, np.float64))
        signatures.append(f'{m.what}: края (светлее — жёсткие), хорды мазка')
        r = a['разбор']
        S = np.r_[0, np.where(r['мазки'], np.arange(len(r['мазки'])) + 1, 0)][r['метки']]   # без полей; мелкие уровни
                                                                # большого снимка — только в плитках
        b = np.zeros(S.shape, bool)
        b[1:] |= S[1:] != S[:-1]
        b[:, 1:] |= S[:, 1:] != S[:, :-1]
        brush = np.array(r['картинка'], np.uint8)
        brush[b] = (255, 0, 255)
        panels.append(np.asarray(_fit_in(brush)[0], np.float64))
        signatures.append(f'{m.what}: мазки разбором')
    return eye._sheet(panels, signatures, path)
