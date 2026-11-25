# READING VIEW of движок/кисть/скоро.py with names in English. Not runnable; edit the original.
"""Мн-4: счёт мерки без питоньих циклов — те же числа бит в бит (`тесты/test_скорость_мерки_Мн4`): суммы идут в том же порядке."""
import math

import numpy as np
from scipy import ndimage

from . import eye

_YaDRA = {}
_LAndN = []


def oklab(rgb):
    """`глаз.oklab`: целые (uint8) — таблицей. Не в `глаз.py`: его текст — в хеше таблицы фактуры (`фактура.хеш`)."""
    if not (isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.size > 256):
        return eye.oklab(rgb)
    if not _LAndN:
        _LAndN.append(eye._lin(np.arange(256.0)))
    return np.cbrt(_LAndN[0][rgb] @ eye._M1.T) @ eye._M2.T


def _core(σ):
    """Ядро `ndimage.gaussian_filter` (truncate 4)."""
    if σ not in _YaDRA:
        r = int(4.0 * float(σ) + 0.5)
        e = np.zeros(2 * r + 1)
        e[r] = 1.0
        _YaDRA[σ] = ndimage.gaussian_filter1d(e, σ, mode='constant')
    return _YaDRA[σ]


class Grid:
    """Гауссиана σ, нужная только в узлах сетки `нач::шаг`: `размыть(a)` — `ndimage.gaussian_filter(a, σ, mode='reflect')
    [нач::шаг, нач::шаг]`, по первой оси считаются только строки сетки. `сдвиг(dy, dx)` — то же для `h · np.roll(h, (−dy, −dx))`
    (`h` — массив, данный при создании); `узлы` — маска узлов, где ответ нужен (в прочих — что угодно)."""
    PResponseLResponseWithA, RYaDResponseIn = 128, 32                                       # столбцов и строк сетки за раз: окно — в кэше

    def __init__(self, h, σ, start, spacing):
        self.h, self.σ, self.start, self.spacing = h, σ, start, spacing

    def blur(self, a):
        return self._count(lambda x0, x1, y0, y1, where_to: np.copyto(where_to, a[y0:y1, x0:x1]), None)

    def shift(self, dy, dx, nodes=None):
        h = self.h
        In, Sh = h.shape
        dy %= In

        def band(x0, x1, y0, y1, where_to):
            j = (x0 + dx) % Sh
            brush = min(x1 - x0, Sh - j)                               # дальше края — с начала строки
            m = min(max(In - dy, y0), y1)                          # строки, что сдвиг не заворачивает
            for c0, c1, s0 in ((0, brush, j), (brush, x1 - x0, 0)):
                for b0, b1, t0 in ((y0, m, dy), (m, y1, dy - In)):
                    if c1 > c0 and b1 > b0:
                        np.multiply(h[b0:b1, x0 + c0:x0 + c1], h[b0 + t0:b1 + t0, s0:s0 + c1 - c0], out=where_to[b0 - y0:b1 - y0, c0:c1])
        return self._count(band, nodes)

    def _count(self, band, nodes):
        w = _core(self.σ)
        r = len(w) // 2
        In, Sh = self.h.shape
        start, spacing = self.start, self.spacing
        n, m = len(range(start, In, spacing)), len(range(start, Sh, spacing))
        if r >= min(In, Sh):                                        # ядро шире картинки: отражение — не раз
            a = np.empty((In, Sh))
            band(0, Sh, 0, In, a)
            return ndimage.gaussian_filter(a, self.σ, mode='reflect')[start::spacing, start::spacing]
        nodes = np.ones((n, m), bool) if nodes is None else nodes
        rows = np.flatnonzero(nodes.any(1))
        pieces, i = [], 0                                          # строки сетки подряд, не больше `РЯДОВ`
        while i < len(rows):
            k = i + 1
            while k < len(rows) and rows[k] - rows[k - 1] == 1 and k - i < self.RYaDResponseIn:
                k += 1
            pieces.append((int(rows[i]), int(rows[k - 1]) + 1))
            i = k
        where = [np.flatnonzero(nodes[i0:i1].any(0)) * spacing + start for i0, i1 in pieces]
        out = np.zeros((n, Sh))
        buffer, t_, o_ = (np.empty((k, self.PResponseLResponseWithA)) for k in (In + 2 * r, self.RYaDResponseIn, self.RYaDResponseIn))
        for x in range(0, Sh, self.PResponseLResponseWithA):
            x1 = min(x + self.PResponseLResponseWithA, Sh)
            p, ready = buffer[:, :x1 - x], 0                        # строки `p` ниже `готово` посчитаны
            for (i0, i1), columns in zip(pieces, where):
                if not ((columns >= x - r) & (columns < x1 + r)).any():
                    continue                                      # ни один нужный узел этих столбцов не видит
                k = i1 - i0
                c = start + r + i0 * spacing
                from_, to = c - r, c + (k - 1) * spacing + r + 1         # строки окна в `p`: картинка — с r-й
                y0, y1 = max(from_ - r, 0), min(to - r, In)            # строки картинки; отражение берёт первые и последние r
                if to > r + In:
                    y0 = min(y0, In - r)
                if from_ < r:
                    y1 = max(y1, r)
                y0 = max(y0, ready)
                if y1 > y0:
                    band(x, x1, y0, y1, p[r + y0:r + y1])
                    ready = y1
                if from_ < r:
                    p[:r] = p[r:2 * r][::-1]
                if to > r + In:
                    p[r + In:] = p[In:r + In][::-1]
                t, o = t_[:k, :x1 - x], o_[:k, :x1 - x]
                brush = slice(c, c + (k - 1) * spacing + 1, spacing)
                np.multiply(p[brush], w[r], out=o)
                for j in range(r, 0, -1):
                    np.add(p[brush.start - j:brush.stop - j:spacing], p[brush.start + j:brush.stop + j:spacing], out=t)
                    t *= w[r - j]
                    o += t
                out[i0:i1, x:x1] = o
        return ndimage.correlate1d(out, w, 1, mode='reflect')[:, start::spacing]


_BrushResponseRNAnd = np.sqrt(np.arange(40001.0))


def distances(to):
    """`ndimage.distance_transform_edt(~до)`: расстояние до ближнего px `до` — по столбцам, потом по строкам, целыми."""
    In, Sh = to.shape
    g = np.where(to, np.uint16(0), np.uint16(200))
    t = np.empty((In, Sh), np.uint16)
    for y, from_ in [(y, y - 1) for y in range(1, In)] + [(y, y + 1) for y in range(In - 2, -1, -1)]:
        np.minimum(g[y], np.add(g[from_], 1, out=t[0]), out=g[y])
    g *= g
    d = g.copy()
    for dx in range(1, min(7, Sh)):
        np.minimum(d[:, :-dx], np.add(g[:, dx:], np.uint16(dx * dx), out=t[:, dx:]), out=d[:, :-dx])
        np.minimum(d[:, dx:], np.add(g[:, :-dx], np.uint16(dx * dx), out=t[:, dx:]), out=d[:, dx:])
    dx = min(7, Sh)
    pl, gp = d.ravel(), g.ravel()
    where = np.flatnonzero(pl > 36)                                 # дальние — немногие: списком
    v, x = pl[where].astype(np.int64), where % Sh
    while len(where):
        if dx >= 150:                                             # до `до` столбцов далеко: общим счётом
            return ndimage.distance_transform_edt(~to)
        for e, da in ((-dx, x >= dx), (dx, x < Sh - dx)):
            np.minimum(v, np.where(da, gp[where + np.where(da, e, 0)].astype(np.int64) + dx * dx, v), out=v)
        pl[where] = v
        dx += 1
        more = v > dx * dx
        where, v, x = where[more], v[more], x[more]
    return _BrushResponseRNAnd[d]


def close(to):
    """Расстояние до ближнего px `до`, но не больше 4: дальше — ровно 4 (px сетки до 4 — окном 9×9 целыми по байту)."""
    In, Sh = to.shape
    g = np.where(to, np.uint8(0), np.uint8(16))
    t = np.empty((In, Sh), np.uint8)

    def wider(d, a, axis):
        """d = min(d, a, сдвинутое на k = 1…4 по оси, + k²)."""
        for k in range(1, min(5, a.shape[axis])):
            gave, bl = (slice(k, None), slice(None, -k))
            if axis == 0:
                t_ = np.add(a, np.uint8(k * k), out=t)
                np.minimum(d[bl], t_[gave], out=d[bl])
                np.minimum(d[gave], t_[bl], out=d[gave])
            else:
                t_ = np.add(a, np.uint8(k * k), out=t)
                np.minimum(d[:, bl], t_[:, gave], out=d[:, bl])
                np.minimum(d[:, gave], t_[:, bl], out=d[:, gave])
        return d
    g = wider(g.copy(), g, 0)
    d = wider(g.copy(), g, 1)
    return _BrushResponseRNAnd[:17][d]


def by_key(option):
    """`np.argsort(ключ, kind='stable')` целых ≥ 0: ключ и место — одним числом, числа сортируются быстрее мест."""
    n = len(option)
    if not n:
        return np.zeros(0, np.int64)
    b = int(n - 1).bit_length()
    if int(option.max()).bit_length() + b > 62:
        return np.argsort(option, kind='stable')
    z = (option.astype(np.int64) << b) | np.arange(n)
    z.sort()
    return z & ((1 << b) - 1)


def by_key_and_numbers(option, v):
    """`np.lexsort((v, ключ))`: сперва по целому ключу, при равных — по числу, при равных — по месту."""
    n = len(option)
    if not n:
        return np.zeros(0, np.int64)
    b, bk = int(n - 1).bit_length(), int(option.max()).bit_length()
    bch = min(62 - b - bk, 40)
    lo, hi = float(v.min()), float(v.max())
    if bch < 8 or not np.isfinite(hi - lo):
        return np.lexsort((v, option))
    m = (2.0 ** bch - 1) / (hi - lo) if hi > lo else 0.0
    step = ((v - lo) * m).astype(np.int64)                      # не убывает по v: порядок ступеней — порядок чисел
    z = (option.astype(np.int64) << (b + bch)) | (step << b) | np.arange(n)
    z.sort()
    and_ = z & ((1 << b) - 1)
    # внутри одной ступени числа могут стоять не по порядку: такие куски — точной сортировкой
    brush = z >> b
    one = brush[1:] == brush[:-1]
    vs = v[and_]
    if (one & (vs[1:] < vs[:-1])).any():
        n = np.r_[False, one] | np.r_[one, False]
        g = np.flatnonzero(n)
        and_[g] = and_[g][np.lexsort((and_[g], vs[g], brush[g]))]
    return and_


def unique(option, back=True):
    """`np.unique(ключ, return_inverse=True)` целых ≥ 0: (ключи, номер ключа у каждого)."""
    and_ = by_key(option)
    brush = option[and_]
    new = np.r_[True, brush[1:] != brush[:-1]] if len(brush) else np.zeros(0, bool)
    sample = np.empty(len(brush), np.int64)
    sample[and_] = np.cumsum(new) - 1
    return brush[new], sample


def unique_count(option):
    """`np.unique(ключ, return_counts=True)`: сортировка самих чисел."""
    brush = np.sort(option)
    if not len(brush):
        return brush, np.zeros(0, np.int64)
    g = np.flatnonzero(np.r_[True, brush[1:] != brush[:-1]])
    return brush[g], np.diff(np.r_[g, len(brush)])


def medians(v, labels, N):
    """`ndimage.median(v, метки, np.arange(N))`, когда у каждой метки есть px."""
    v, m = v.ravel(), labels.ravel()
    and_ = by_key_and_numbers(m, v)
    n = np.bincount(m, minlength=N)
    n = np.r_[0, np.cumsum(n)[:-1]]
    s = v[and_]
    lo, hi = n + (n - 1) // 2, n + n // 2
    return (s[lo] + s[hi]) / 2.0


# ---------------------------------------------------------------- хорды

def _sums(profile, n, brush):
    """Средние строк `н`:`к` (по каналам): та же последовательная сумма, что у `.mean(0)`."""
    dl = brush - n
    whose = np.repeat(np.arange(len(n)), dl)
    where = np.repeat(n - np.r_[0, np.cumsum(dl)[:-1]], dl) + np.arange(len(whose))
    return np.stack([np.bincount(whose, c[where], len(n)) for c in profile], 1) / dl[:, None]


def mixes(profile, row, e, k, n, ZAMEWith, MIX_L):
    """Края пятен одного замеса у всех линий разом. `проф` — каналы линий (по (строк·n)), края — (`строка`, `e`) по порядку.
    Из краёв линии по одному, начиная с самых близких, уходят те, по обе стороны которых средние участков ближе `ЗАМЕС`
    по цвету и `ЗАМЕС_L` по светлоте; последний край линии остаётся. Ответ — маска оставшихся."""
    total = len(e)
    live = np.ones(total, bool)
    if not total:
        return live
    first = np.flatnonzero(np.r_[True, row[1:] != row[:-1]])
    count = np.diff(np.r_[first, total])
    first, count = first[count > 1], count[count > 1]                    # у линии с одним краем сливать нечего
    if not len(first):
        return live
    L, M = len(first), int(count.max())
    line = row[first]
    # участок j (0…счёт) — между краем j − 1 и краем j; край 0 и край счёт — концы линии
    where = np.arange(M)[None] < count[:, None]
    gr = np.zeros((L, M + 2), np.int64)
    gr[:, 1:-1][where] = e[np.repeat(first, count) + (np.arange(int(count.sum())) - np.repeat(np.r_[0, np.cumsum(count)[:-1]], count))]
    gr[np.arange(L), count + 1] = n - 1
    sr = np.zeros((L, M + 1, 3))

    def mean(l, a, b):
        """Средние участков линий `л` от края a до края b (px): у длинных — без полей `k`."""
        dl = b - a > 2 * k + 2
        n = np.where(dl, a + k, (a + b) // 2)
        kn = np.where(dl, b - k + 1, (a + b) // 2 + 1)
        b = line[l] * n
        return _sums(profile, b + n, b + kn)

    uch = np.arange(M + 1)[None] <= count[:, None]
    l, j = np.nonzero(uch)
    sr[l, j] = mean(l, gr[l, j], gr[l, j + 1])
    price = np.full((L, M), np.inf)

    def value(l, i, a, b):
        """Цена краёв i (0…) линий `л` между участками a и b: сумма долей порогов, у разных замесов — бесконечность."""
        d = sr[l, a] - sr[l, b]
        ab = np.array(list(map(math.hypot, d[:, 1].tolist(), d[:, 2].tolist())))
        dl = np.abs(d[:, 0])
        price[l, i] = np.where((ab < ZAMEWith) & (dl < MIX_L), ab / ZAMEWith + dl / MIX_L, np.inf)

    l, i = np.nonzero(where)
    value(l, i, i, i + 1)
    # участок назван своим левым краем; соседи края — цепочкой
    back = np.tile(np.arange(-1, M - 1), (L, 1))                 # живой край перед краем i (−1 — начало линии)
    forward = np.tile(np.arange(1, M + 1), (L, 1))                 # живой край за ним (счёт — конец)
    left = count.copy()
    go = np.arange(L)
    while len(go):
        i = price[go].argmin(1)
        da = np.isfinite(price[go, i])
        go, i = go[da], i[da]
        if not len(go):
            break
        l = go
        price[l, i] = np.inf
        pr, sl = back[l, i], forward[l, i]
        live[first[l] + i] = False
        q = pr >= 0
        forward[l[q], pr[q]] = sl[q]
        q2 = sl < count[l]
        back[l[q2], sl[q2]] = pr[q2]
        left[l] -= 1
        # участок слева от края i (его номер — пр + 1) забрал участок справа (номер i + 1)
        l_ = l
        sr[l_, pr + 1] = mean(l_, gr[l_, pr + 1], gr[l_, sl + 1])
        if q.any():
            value(l[q], pr[q], _left(back, l[q], pr[q]), pr[q] + 1)
        if q2.any():
            value(l[q2], sl[q2], pr[q2] + 1, sl[q2] + 1)
        go = go[left[go] > 1]
    return live


def _left(back, l, i):
    """Номер участка слева от края i: он назван живым краем перед ним."""
    return back[l, i] + 1
