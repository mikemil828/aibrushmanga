"""Мн-4: счёт мерки без питоньих циклов — те же числа бит в бит (`тесты/test_скорость_мерки_Мн4`): суммы идут в том же порядке."""
import math

import numpy as np
from scipy import ndimage

from . import глаз

_ЯДРА = {}
_ЛИН = []


def oklab(rgb):
    """`глаз.oklab`: целые (uint8) — таблицей. Не в `глаз.py`: его текст — в хеше таблицы фактуры (`фактура.хеш`)."""
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.size > 256):
        return глаз.oklab(rgb)
    if not _ЛИН:
        _ЛИН.append(глаз._lin(np.arange(256.0)))
    return np.cbrt(_ЛИН[0][rgb] @ глаз._M1.T) @ глаз._M2.T


def _ядро(σ):
    """Ядро `ndimage.gaussian_filter` (truncate 4)."""
    if σ not in _ЯДРА:
        r = int(4.0 * float(σ) + 0.5)
        e = np.zeros(2 * r + 1)
        e[r] = 1.0
        _ЯДРА[σ] = ndimage.gaussian_filter1d(e, σ, mode='constant')
    return _ЯДРА[σ]


class Сетка:
    """Гауссиана σ, нужная только в узлах сетки `нач::шаг`: `размыть(a)` — `ndimage.gaussian_filter(a, σ, mode='reflect')
    [нач::шаг, нач::шаг]`, по первой оси считаются только строки сетки. `сдвиг(dy, dx)` — то же для `h · np.roll(h, (−dy, −dx))`
    (`h` — массив, данный при создании); `узлы` — маска узлов, где ответ нужен (в прочих — что угодно)."""
    ПОЛОСА, РЯДОВ = 128, 32                                       # столбцов и строк сетки за раз: окно — в кэше

    def __init__(self, h, σ, нач, шаг):
        self.h, self.σ, self.нач, self.шаг = h, σ, нач, шаг

    def размыть(self, a):
        return self._счёт(lambda x0, x1, y0, y1, куда: np.copyto(куда, a[y0:y1, x0:x1]), None)

    def сдвиг(self, dy, dx, узлы=None):
        h = self.h
        В, Ш = h.shape
        dy %= В

        def полоса(x0, x1, y0, y1, куда):
            j = (x0 + dx) % Ш
            к = min(x1 - x0, Ш - j)                               # дальше края — с начала строки
            м = min(max(В - dy, y0), y1)                          # строки, что сдвиг не заворачивает
            for c0, c1, s0 in ((0, к, j), (к, x1 - x0, 0)):
                for b0, b1, t0 in ((y0, м, dy), (м, y1, dy - В)):
                    if c1 > c0 and b1 > b0:
                        np.multiply(h[b0:b1, x0 + c0:x0 + c1], h[b0 + t0:b1 + t0, s0:s0 + c1 - c0], out=куда[b0 - y0:b1 - y0, c0:c1])
        return self._счёт(полоса, узлы)

    def _счёт(self, полоса, узлы):
        w = _ядро(self.σ)
        r = len(w) // 2
        В, Ш = self.h.shape
        нач, шаг = self.нач, self.шаг
        n, m = len(range(нач, В, шаг)), len(range(нач, Ш, шаг))
        if r >= min(В, Ш):                                        # ядро шире картинки: отражение — не раз
            a = np.empty((В, Ш))
            полоса(0, Ш, 0, В, a)
            return ndimage.gaussian_filter(a, self.σ, mode='reflect')[нач::шаг, нач::шаг]
        узлы = np.ones((n, m), bool) if узлы is None else узлы
        ряды = np.flatnonzero(узлы.any(1))
        куски, i = [], 0                                          # строки сетки подряд, не больше `РЯДОВ`
        while i < len(ряды):
            k = i + 1
            while k < len(ряды) and ряды[k] - ряды[k - 1] == 1 and k - i < self.РЯДОВ:
                k += 1
            куски.append((int(ряды[i]), int(ряды[k - 1]) + 1))
            i = k
        где = [np.flatnonzero(узлы[i0:i1].any(0)) * шаг + нач for i0, i1 in куски]
        out = np.zeros((n, Ш))
        буф, t_, o_ = (np.empty((k, self.ПОЛОСА)) for k in (В + 2 * r, self.РЯДОВ, self.РЯДОВ))
        for x in range(0, Ш, self.ПОЛОСА):
            x1 = min(x + self.ПОЛОСА, Ш)
            p, готово = буф[:, :x1 - x], 0                        # строки `p` ниже `готово` посчитаны
            for (i0, i1), столбцы in zip(куски, где):
                if not ((столбцы >= x - r) & (столбцы < x1 + r)).any():
                    continue                                      # ни один нужный узел этих столбцов не видит
                k = i1 - i0
                c = нач + r + i0 * шаг
                от, до = c - r, c + (k - 1) * шаг + r + 1         # строки окна в `p`: картинка — с r-й
                y0, y1 = max(от - r, 0), min(до - r, В)            # строки картинки; отражение берёт первые и последние r
                if до > r + В:
                    y0 = min(y0, В - r)
                if от < r:
                    y1 = max(y1, r)
                y0 = max(y0, готово)
                if y1 > y0:
                    полоса(x, x1, y0, y1, p[r + y0:r + y1])
                    готово = y1
                if от < r:
                    p[:r] = p[r:2 * r][::-1]
                if до > r + В:
                    p[r + В:] = p[В:r + В][::-1]
                t, o = t_[:k, :x1 - x], o_[:k, :x1 - x]
                к = slice(c, c + (k - 1) * шаг + 1, шаг)
                np.multiply(p[к], w[r], out=o)
                for j in range(r, 0, -1):
                    np.add(p[к.start - j:к.stop - j:шаг], p[к.start + j:к.stop + j:шаг], out=t)
                    t *= w[r - j]
                    o += t
                out[i0:i1, x:x1] = o
        return ndimage.correlate1d(out, w, 1, mode='reflect')[:, нач::шаг]


_КОРНИ = np.sqrt(np.arange(40001.0))


def расстояния(до):
    """`ndimage.distance_transform_edt(~до)`: расстояние до ближнего px `до` — по столбцам, потом по строкам, целыми."""
    В, Ш = до.shape
    g = np.where(до, np.uint16(0), np.uint16(200))
    т = np.empty((В, Ш), np.uint16)
    for y, от in [(y, y - 1) for y in range(1, В)] + [(y, y + 1) for y in range(В - 2, -1, -1)]:
        np.minimum(g[y], np.add(g[от], 1, out=т[0]), out=g[y])
    g *= g
    d = g.copy()
    for dx in range(1, min(7, Ш)):
        np.minimum(d[:, :-dx], np.add(g[:, dx:], np.uint16(dx * dx), out=т[:, dx:]), out=d[:, :-dx])
        np.minimum(d[:, dx:], np.add(g[:, :-dx], np.uint16(dx * dx), out=т[:, dx:]), out=d[:, dx:])
    dx = min(7, Ш)
    пл, гп = d.ravel(), g.ravel()
    где = np.flatnonzero(пл > 36)                                 # дальние — немногие: списком
    v, x = пл[где].astype(np.int64), где % Ш
    while len(где):
        if dx >= 150:                                             # до `до` столбцов далеко: общим счётом
            return ndimage.distance_transform_edt(~до)
        for e, да in ((-dx, x >= dx), (dx, x < Ш - dx)):
            np.minimum(v, np.where(да, гп[где + np.where(да, e, 0)].astype(np.int64) + dx * dx, v), out=v)
        пл[где] = v
        dx += 1
        ещё = v > dx * dx
        где, v, x = где[ещё], v[ещё], x[ещё]
    return _КОРНИ[d]


def близко(до):
    """Расстояние до ближнего px `до`, но не больше 4: дальше — ровно 4 (px сетки до 4 — окном 9×9 целыми по байту)."""
    В, Ш = до.shape
    g = np.where(до, np.uint8(0), np.uint8(16))
    т = np.empty((В, Ш), np.uint8)

    def шире(d, a, ось):
        """d = min(d, a, сдвинутое на k = 1…4 по оси, + k²)."""
        for k in range(1, min(5, a.shape[ось])):
            дал, бл = (slice(k, None), slice(None, -k))
            if ось == 0:
                т_ = np.add(a, np.uint8(k * k), out=т)
                np.minimum(d[бл], т_[дал], out=d[бл])
                np.minimum(d[дал], т_[бл], out=d[дал])
            else:
                т_ = np.add(a, np.uint8(k * k), out=т)
                np.minimum(d[:, бл], т_[:, дал], out=d[:, бл])
                np.minimum(d[:, дал], т_[:, бл], out=d[:, дал])
        return d
    g = шире(g.copy(), g, 0)
    d = шире(g.copy(), g, 1)
    return _КОРНИ[:17][d]


def по_ключу(ключ):
    """`np.argsort(ключ, kind='stable')` целых ≥ 0: ключ и место — одним числом, числа сортируются быстрее мест."""
    n = len(ключ)
    if not n:
        return np.zeros(0, np.int64)
    б = int(n - 1).bit_length()
    if int(ключ.max()).bit_length() + б > 62:
        return np.argsort(ключ, kind='stable')
    з = (ключ.astype(np.int64) << б) | np.arange(n)
    з.sort()
    return з & ((1 << б) - 1)


def по_ключу_и_числу(ключ, v):
    """`np.lexsort((v, ключ))`: сперва по целому ключу, при равных — по числу, при равных — по месту."""
    n = len(ключ)
    if not n:
        return np.zeros(0, np.int64)
    б, бк = int(n - 1).bit_length(), int(ключ.max()).bit_length()
    бч = min(62 - б - бк, 40)
    lo, hi = float(v.min()), float(v.max())
    if бч < 8 or not np.isfinite(hi - lo):
        return np.lexsort((v, ключ))
    м = (2.0 ** бч - 1) / (hi - lo) if hi > lo else 0.0
    ступень = ((v - lo) * м).astype(np.int64)                      # не убывает по v: порядок ступеней — порядок чисел
    з = (ключ.astype(np.int64) << (б + бч)) | (ступень << б) | np.arange(n)
    з.sort()
    и = з & ((1 << б) - 1)
    # внутри одной ступени числа могут стоять не по порядку: такие куски — точной сортировкой
    к = з >> б
    один = к[1:] == к[:-1]
    vs = v[и]
    if (один & (vs[1:] < vs[:-1])).any():
        н = np.r_[False, один] | np.r_[один, False]
        г = np.flatnonzero(н)
        и[г] = и[г][np.lexsort((и[г], vs[г], к[г]))]
    return и


def уникальные(ключ, обратно=True):
    """`np.unique(ключ, return_inverse=True)` целых ≥ 0: (ключи, номер ключа у каждого)."""
    и = по_ключу(ключ)
    к = ключ[и]
    нов = np.r_[True, к[1:] != к[:-1]] if len(к) else np.zeros(0, bool)
    обр = np.empty(len(к), np.int64)
    обр[и] = np.cumsum(нов) - 1
    return к[нов], обр


def уникальные_счёт(ключ):
    """`np.unique(ключ, return_counts=True)`: сортировка самих чисел."""
    к = np.sort(ключ)
    if not len(к):
        return к, np.zeros(0, np.int64)
    г = np.flatnonzero(np.r_[True, к[1:] != к[:-1]])
    return к[г], np.diff(np.r_[г, len(к)])


def медианы(v, метки, N):
    """`ndimage.median(v, метки, np.arange(N))`, когда у каждой метки есть px."""
    v, м = v.ravel(), метки.ravel()
    и = по_ключу_и_числу(м, v)
    n = np.bincount(м, minlength=N)
    н = np.r_[0, np.cumsum(n)[:-1]]
    s = v[и]
    lo, hi = н + (n - 1) // 2, н + n // 2
    return (s[lo] + s[hi]) / 2.0


# ---------------------------------------------------------------- хорды

def _суммы(проф, н, к):
    """Средние строк `н`:`к` (по каналам): та же последовательная сумма, что у `.mean(0)`."""
    дл = к - н
    чей = np.repeat(np.arange(len(н)), дл)
    где = np.repeat(н - np.r_[0, np.cumsum(дл)[:-1]], дл) + np.arange(len(чей))
    return np.stack([np.bincount(чей, c[где], len(н)) for c in проф], 1) / дл[:, None]


def замесы(проф, строка, e, k, n, ЗАМЕС, ЗАМЕС_L):
    """Края пятен одного замеса у всех линий разом. `проф` — каналы линий (по (строк·n)), края — (`строка`, `e`) по порядку.
    Из краёв линии по одному, начиная с самых близких, уходят те, по обе стороны которых средние участков ближе `ЗАМЕС`
    по цвету и `ЗАМЕС_L` по светлоте; последний край линии остаётся. Ответ — маска оставшихся."""
    всего = len(e)
    жив = np.ones(всего, bool)
    if not всего:
        return жив
    перв = np.flatnonzero(np.r_[True, строка[1:] != строка[:-1]])
    счёт = np.diff(np.r_[перв, всего])
    перв, счёт = перв[счёт > 1], счёт[счёт > 1]                    # у линии с одним краем сливать нечего
    if not len(перв):
        return жив
    Л, М = len(перв), int(счёт.max())
    стр = строка[перв]
    # участок j (0…счёт) — между краем j − 1 и краем j; край 0 и край счёт — концы линии
    где = np.arange(М)[None] < счёт[:, None]
    гр = np.zeros((Л, М + 2), np.int64)
    гр[:, 1:-1][где] = e[np.repeat(перв, счёт) + (np.arange(int(счёт.sum())) - np.repeat(np.r_[0, np.cumsum(счёт)[:-1]], счёт))]
    гр[np.arange(Л), счёт + 1] = n - 1
    ср = np.zeros((Л, М + 1, 3))

    def средние(л, a, b):
        """Средние участков линий `л` от края a до края b (px): у длинных — без полей `k`."""
        дл = b - a > 2 * k + 2
        н = np.where(дл, a + k, (a + b) // 2)
        кн = np.where(дл, b - k + 1, (a + b) // 2 + 1)
        б = стр[л] * n
        return _суммы(проф, б + н, б + кн)

    уч = np.arange(М + 1)[None] <= счёт[:, None]
    л, j = np.nonzero(уч)
    ср[л, j] = средние(л, гр[л, j], гр[л, j + 1])
    цена = np.full((Л, М), np.inf)

    def ценить(л, i, a, b):
        """Цена краёв i (0…) линий `л` между участками a и b: сумма долей порогов, у разных замесов — бесконечность."""
        d = ср[л, a] - ср[л, b]
        ab = np.array(list(map(math.hypot, d[:, 1].tolist(), d[:, 2].tolist())))
        dl = np.abs(d[:, 0])
        цена[л, i] = np.where((ab < ЗАМЕС) & (dl < ЗАМЕС_L), ab / ЗАМЕС + dl / ЗАМЕС_L, np.inf)

    л, i = np.nonzero(где)
    ценить(л, i, i, i + 1)
    # участок назван своим левым краем; соседи края — цепочкой
    назад = np.tile(np.arange(-1, М - 1), (Л, 1))                 # живой край перед краем i (−1 — начало линии)
    вперёд = np.tile(np.arange(1, М + 1), (Л, 1))                 # живой край за ним (счёт — конец)
    осталось = счёт.copy()
    идут = np.arange(Л)
    while len(идут):
        i = цена[идут].argmin(1)
        да = np.isfinite(цена[идут, i])
        идут, i = идут[да], i[да]
        if not len(идут):
            break
        л = идут
        цена[л, i] = np.inf
        пр, сл = назад[л, i], вперёд[л, i]
        жив[перв[л] + i] = False
        q = пр >= 0
        вперёд[л[q], пр[q]] = сл[q]
        q2 = сл < счёт[л]
        назад[л[q2], сл[q2]] = пр[q2]
        осталось[л] -= 1
        # участок слева от края i (его номер — пр + 1) забрал участок справа (номер i + 1)
        л_ = л
        ср[л_, пр + 1] = средние(л_, гр[л_, пр + 1], гр[л_, сл + 1])
        if q.any():
            ценить(л[q], пр[q], _левый(назад, л[q], пр[q]), пр[q] + 1)
        if q2.any():
            ценить(л[q2], сл[q2], пр[q2] + 1, сл[q2] + 1)
        идут = идут[осталось[идут] > 1]
    return жив


def _левый(назад, л, i):
    """Номер участка слева от края i: он назван живым краем перед ним."""
    return назад[л, i] + 1
