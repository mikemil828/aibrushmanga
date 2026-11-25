# READING VIEW of движок/кисть/правка.py with names in English. Not runnable; edit the original.
"""М-3: правка высохшей краски — `размочить` и `прижать` (в журнале — строка {"wet": …} и строка с "tool": "нож").

Высохшее ни кисть, ни нож без краски, ни тряпка не берут (Ф-3: сухое не мнётся). У живописца на это два приёма.

**Размочить** — масло (или растворитель) поверх сухого: верх плёнки набухает и снова берётся кистью, тряпкой и ножом.
Масло входит сверху и неглубоко: за раз сырым становится не больше `_РАЗМОЧ[0]` толщины (h) от поверхности — у тонкого
письма это весь слой, у пастозного вала только кожа, тело под ней стоит твёрдым. Влажность слоя — доля его толщины,
что сыра и лежит сверху (так её читают нож и снятие): размоченное — `доля` × достигнутая глубина. Повторный вызов
глубже не идёт (масло стоит на той же коже), слой выше забирает глубину на свою толщину. Схват (`схватить`) снимается
в ту же меру: корка размягчается на достигнутой доле толщины.

**Прижать** — ход ножа плашмя по сухому: нож срезает то, что выше его плоскости, и оставляет ровную площадку.
Плоскость — не линия под рукой, а лезвие: оно лежит на холсте подошвой (`_ПРИЖИМ[1]` мм по ходу × ширина кромки).
Опора лезвия — краска под подошвой, а не грунт рядом (М-3-2): точка от четверти самой высокой под подошвой держит
лезвие в полную меру, тоньше — в меру своей толщины, голый грунт не держит и вниз лезвие не тянет. Слабый нажим — нож
едет по вершинам (среднее высот верхних 15 % опоры) и ничего не берёт; с нажимом лезвие садится к телу краски под
собой (медиана высот опоры): нажим 0,6 — полпути от вершин к телу, от 0,9 — до тела. Тело мазка нож не срывает: ниже
этой медианы лезвие не идёт, сколько ни дави, — и там, где кромка косо сходит с вала и большей частью висит над
грунтом, оно стоит на вале. Тонкий слой рядом с валом опора слабая: вал на ровном слое срезается до него, вал на
голом грунте — до своей середины. `глубина` (мм) — осторожный ход: ниже самой высокой точки под подошвой больше чем
на столько лезвие за ход не садится. Слой, названный в `слой`, под целой краской другого слоя не режется. Всё, что выше плоскости, — сухое и сырое — уходит
на нож и с холста: сухая крошка валика не даёт. Цвет остаётся цветом слоя (срез открывает нутро того же мазка), где
слой срезан насквозь — открывается то, что под ним; грунт и зерно нож не режет. Случайных чисел нет."""
import numpy as np

from .fast import clamp
from .removal import _step, _remove

MM_ON_H = 0.4                # мм краски в единице толщины h (`холст.ММ_НА_H`)
_RAZMResponseH = (0.25,)            # на столько h от поверхности масло размягчает сухое (0,1 мм — кожа плёнки)
_PRAndZhAndM = (None, 4.0, 0.3, 0.9, 0.03, 0.15, 0.02, 0.25)   # [0] глубина по умолчанию, мм: ниже самой высокой точки под подошвой нож за ход не идёт (None — предела нет: глубину ведёт нажим); [1] длина подошвы ножа по ходу, мм; [2] нажим, до которого нож едет по вершинам; [3] нажим, с которого лезвие сидит на теле; [4] нажим, ниже которого нож холста не касается (как `_НОЖ[7]`); [5] доля высот под подошвой, чьё среднее — «вершины»; [6] h, тоньше которого срез не считается следом; [7] доля самой высокой точки под подошвой, от которой краска держит лезвие в полную меру (тоньше — в меру толщины, голый грунт не держит)
_BrushAtWithResponseBrush = 256


def re_wet(canvas, share, layer=None, region=None):
    """Вернуть сухому верху краски влажность (шапка модуля). доля 0–1; слой — имя или None (все: глубина считается
    от поверхности, верхние слои её забирают); область — (x0, y0, x1, y1) или None. Ответ: площадь, где влажность
    выросла, px."""
    share = float(clamp(share, 0.0, 1.0))
    if share <= 0.0:
        return 0
    gy, gx = canvas._region(region)
    names = list(canvas.layers)
    if layer is not None and layer not in names:
        return 0
    grew = 0
    for y0 in range(gy.start, gy.stop, _BrushAtWithResponseBrush):
        for x0 in range(gx.start, gx.stop, _BrushAtWithResponseBrush):
            where = (slice(y0, min(y0 + _BrushAtWithResponseBrush, gy.stop)), slice(x0, min(x0 + _BrushAtWithResponseBrush, gx.stop)))
            above = None                                           # толщина слоёв над текущим
            for name in names[::-1]:
                with_ = canvas.layers[name]
                h = np.asarray(with_.thickness[where], np.float64)
                if not h.any():
                    continue
                if layer is None or name == layer:
                    depth = _RAZMResponseH[0] if above is None else np.maximum(_RAZMResponseH[0] - above, 0.0)
                    reached = np.minimum(h, depth)               # толщина слоя, до которой дошло масло
                    w = np.asarray(with_.moisture[where], np.float64)
                    wet = np.maximum(h * w, share * reached)
                    w2 = np.where(h > 1e-9, clamp(wet / np.maximum(h, 1e-9), 0.0, 1.0), w)
                    gentler = share * np.where(h > 1e-9, reached / np.maximum(h, 1e-9), 0.0)
                    skh = np.asarray(with_.degree_of_set[where], np.float64)
                    skh2 = skh * (1.0 - gentler)
                    became = (w2 > w + 1e-7)
                    if became.any() or (skh2 < skh - 1e-7).any():
                        grew += int(became.sum())
                        with_.moisture[where] = w2.astype(np.float32)
                        with_.degree_of_set[where] = skh2.astype(np.float32)
                above = h if above is None else above + h
    return grew


class Pressed:
    """Ответ `х.прижать`: `прямоугольник` (x0, y0, x1, y1) хода ножа; `под_ножом` — px под лезвием; `срезано` — px, где
    нож снял краску; `на_нож` — объём срезанного, h·px² (ушёл с холста), `мм3` — он же в мм³; `доля` — срезанное к
    краске, лежавшей под ножом; `глубже` — самый глубокий срез, мм; `сырого` — доля сырой краски в срезанном;
    `почему` — слово, если нож ничего не снял."""
    __slots__ = ('прямоугольник', 'под_ножом', 'срезано', 'на_нож', 'мм3', 'доля', 'глубже', 'сырого', 'почему', '_клетки')

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.get(name))

    def __bool__(self):
        return bool(self.cut_off)

    def __repr__(self):
        if not self.cut_off:
            return f'Прижато(ничего не срезано: {self.why})'
        return (f'Прижато(срезано {self.cut_off} px из {self.under_knife} под ножом, на нож {self.on_knife:.1f} h·px² '
                f'({self.mm3:.2f} мм³, {self.share:.0%} краски под ножом), глубже всего {self.deeper:.2f} мм'
                + (f', сырого в срезанном {self.wet:.0%}' if self.wet else '') + ')')


def _plane(S, p, depth_h, px_on_mm):
    """Высота лезвия на каждом шаге хода (N,). S (N, nu) — толщина краски под кромкой по шагам (шаг 0,5 px),
    p (N,) — нажим. Подошва — окно шагов длиной `_ПРИЖИМ[1]` мм вокруг кромки. Опора лезвия — краска под подошвой:
    голый грунт рядом не держит и не тянет вниз. Тело — высота, ниже
    которой лежит половина опоры (медиана высот с весом точки: от `_ПРИЖИМ[7]` самой высокой — 1, тоньше — в меру
    толщины), вершины — среднее высот верхних `_ПРИЖИМ[5]` опоры. Где нож оторван (нажим ниже `_ПРИЖИМ[4]`) или под подошвой краски нет — +inf.
    Окна читаются через px (каждый второй шаг и каждая вторая точка кромки) и считаются кусками шагов: память хода —
    десятки МБ при любой длине."""
    N, nu = S.shape
    floor = max(1, int(round(0.5 * _PRAndZhAndM[1] * px_on_mm / 0.5)))
    floor += floor % 2                                                 # чётная: окно через шаг симметрично вокруг кромки
    touches = p >= _PRAndZhAndM[4]
    Sk = np.where(touches[:, None], S, 0.0)[:, ::2]               # под оторванным ножом краска в опору не идёт
    Sk = np.pad(Sk, ((floor, floor), (0, 0)))
    m = (floor + 1) * Sk.shape[1]                                    # точек в окне
    top, body, peak = np.full(N, np.inf), np.full(N, np.inf), np.full(N, np.inf)
    piece = max(1, int(4e6 // m))
    for i0 in range(0, N, piece):
        i1 = min(i0 + piece, N)
        windows = np.lib.stride_tricks.sliding_window_view(Sk[i0:i1 + 2 * floor], 2 * floor + 1, axis=0)[:, :, ::2].reshape(i1 - i0, m)
        row = np.sort(windows, axis=1)
        holds = clamp(row / np.maximum(_PRAndZhAndM[7] * row[:, -1:], 1e-12), 0.0, 1.0)   # вес точки в опоре: от четверти самой высокой — полный, тоньше — в меру толщины, грунт — ноль
        gathers = np.cumsum(holds, axis=1)
        total = gathers[:, -1]
        has = row[:, -1] > 1e-9
        j = (gathers >= 0.5 * total[:, None]).argmax(axis=1)
        weight = np.where(gathers > (1.0 - _PRAndZhAndM[5]) * total[:, None], holds, 0.0)
        body[i0:i1] = np.where(has, row[np.arange(i1 - i0), j], np.inf)
        top[i0:i1] = np.where(has, (weight * row).sum(axis=1) / np.maximum(weight.sum(axis=1), 1e-12), np.inf)
        peak[i0:i1] = np.where(has, row[:, -1], np.inf)
    strength = clamp((p - _PRAndZhAndM[2]) / (_PRAndZhAndM[3] - _PRAndZhAndM[2]), 0.0, 1.0)
    with np.errstate(invalid='ignore'):
        z = np.maximum(body + (top - body) * (1.0 - strength), np.where(strength > 0.0, peak - depth_h * np.maximum(strength, 1e-9), np.inf))   # глубже `глубина` под самой высокой точкой под подошвой лезвие за ход не садится
    return np.where(touches & (strength > 0.0) & np.isfinite(body), z, np.inf)


def press(canvas, points, width, layer=None, depth=None):
    """Ход ножа плашмя по лежащей краске: срезать всё выше плоскости лезвия (шапка модуля). точки — (x, y, нажим, …);
    ширина — длина кромки, px; слой — имя (режется только он и только там, где над ним нет краски другого слоя) или None (сверху вниз все); глубина — мм, не глубже
    стольких под самой высокой точкой под ножом за ход (None — предела нет, глубину ведёт нажим). Ответ — `Прижато`."""
    if not width > 0:
        raise ValueError('прижать: ширина ножа — больше нуля')
    depth_mm = _PRAndZhAndM[0] if depth is None else float(depth)
    if depth_mm is not None and not depth_mm > 0:
        raise ValueError('прижать: глубина — мм, больше нуля')
    pm = float(canvas.px_on_mm or 8.0)
    path, d = _step(points, 0.5)
    x, y, p = path[:, 0] - canvas.x0, path[:, 1] - canvas.y0, path[:, 2]
    N = len(path)
    W = float(width)
    r = W / 2 + 2.0
    bx0, by0 = max(int(np.floor(x.min() - r)), 0), max(int(np.floor(y.min() - r)), 0)
    bx1, by1 = min(int(np.ceil(x.max() + r)) + 1, canvas.width), min(int(np.ceil(y.max() + r)) + 1, canvas.height)
    empty = lambda why: Pressed(rectangle=(bx0 + canvas.x0, by0 + canvas.y0, bx1 + canvas.x0, by1 + canvas.y0), under_knife=0, cut_off=0,   # noqa: E731
                                   on_knife=0.0, mm3=0.0, share=0.0, deeper=0.0, wet=0.0, why=why)
    if bx1 <= bx0 or by1 <= by0:
        return empty('ход мимо холста')
    if not (p >= _PRAndZhAndM[4]).any():
        return empty('нож холста не коснулся (нажим 0)')
    where = (slice(by0, by1), slice(bx0, bx1))
    bh, bw = by1 - by0, bx1 - bx0
    names = list(canvas.layers)
    is_cut = [name for name in names if layer is None or name == layer]
    if not is_cut:
        return empty(f'слоя {layer!r} нет' if layer is not None else 'краски на холсте нет')
    surface = np.asarray(canvas.relief(by0, by1, bx0, bx1, full=True), np.float64)
    # --- кромка по шагам: какие пиксели под ней и на какой высоте она идёт
    nu = max(2, int(np.ceil(W / 0.5)))
    uu = (np.arange(nu) + 0.5) * (W / nu) - W / 2
    ax, ay = -d[:, 1], d[:, 0]
    ix = np.floor(x[:, None] + uu[None, :] * ax[:, None] - bx0).astype(np.intp)
    iy = np.floor(y[:, None] + uu[None, :] * ay[:, None] - by0).astype(np.intp)
    ok = (ix >= 0) & (ix < bw) & (iy >= 0) & (iy < bh)
    S = np.where(ok, surface[clamp(iy, 0, bh - 1), clamp(ix, 0, bw - 1)], np.nan)
    S = np.where(np.isnan(S), 0.0, S)                              # за краем холста краски нет
    z = _plane(S, p, np.inf if depth_mm is None else depth_mm / MM_ON_H, pm)
    Z = np.full(bh * bw, np.inf)
    q = (iy * bw + ix)[ok]
    np.minimum.at(Z, q, np.broadcast_to(z[:, None], (N, nu))[ok])
    under = np.zeros(bh * bw, bool)
    under[q[np.broadcast_to((p >= _PRAndZhAndM[4])[:, None], (N, nu))[ok]]] = True
    Z, under = Z.reshape(bh, bw), under.reshape(bh, bw)
    the_excess = np.where(np.isfinite(Z), np.maximum(surface - Z, 0.0), 0.0)
    if not (the_excess > 1e-9).any():
        response = empty('нож едет по вершинам: выше плоскости лезвия ничего нет (нажим от 0,3 сажает лезвие ниже; '
                  'ровное место срезать нечем)' if the_excess.size and under.any() else 'под ножом пусто')
        response.under_knife = int(under.sum())
        return response
    # --- срез: сверху вниз по слоям, не больше толщины слоя
    canvas._to_records((bx0 + canvas.x0, by0 + canvas.y0, bx1 + canvas.x0, by1 + canvas.y0))
    lay_there = 0.0
    removed_total, wet = np.zeros((bh, bw)), 0.0
    remainder = the_excess.copy()
    above = np.zeros((bh, bw))                                     # толщина слоёв над текущим, которые не режутся
    for name in names[::-1]:
        with_ = canvas.layers[name]
        h = np.asarray(with_.thickness[where], np.float64)
        if name not in is_cut:
            above = above + h
            continue
        lay_there += float(h[under].sum())
        remove = np.where(above > 1e-6, 0.0, np.minimum(remainder, h))   # под целой краской другого слоя нож не достаёт: там слой не режется
        if (remove > 0.0).any():
            wet += float(np.minimum(remove, h * np.asarray(with_.moisture[where], np.float64)).sum())
            if with_.transmittance is not None:                             # не кроющая плёнка тоньше — светлее
                share = np.where(h > 1e-6, clamp((h - remove) / np.maximum(h, 1e-6), 0.0, 1.0), 1.0)
                with_.transmittance[where] = np.power(np.asarray(with_.transmittance[where], np.float64), share[..., None]).astype(np.float32)
            _remove(with_, where, remove.astype(np.float32))
            removed_total += remove
            remainder = remainder - remove
    volume = float(removed_total.sum())
    trace = removed_total > _PRAndZhAndM[6]
    if volume <= 0.0:
        response = empty('выше плоскости лезвия краски слоя нет — там грунт или другой слой')
        response.under_knife = int(under.sum())
        return response
    ys, xs = np.nonzero(trace)                                    # зеркалу С18: срезано px по клеткам 32 px (`подсказки.КЛЕТКА`)
    numbers, how_much = np.unique(((ys + by0 + canvas.y0) // 32) * 100000 + (xs + bx0 + canvas.x0) // 32, return_counts=True)
    return Pressed(_cells=dict(zip(numbers.tolist(), how_much.tolist())),
                   rectangle=(bx0 + canvas.x0, by0 + canvas.y0, bx1 + canvas.x0, by1 + canvas.y0), under_knife=int(under.sum()),
                   cut_off=int(trace.sum()), on_knife=volume, mm3=volume * MM_ON_H / (pm * pm), share=volume / max(lay_there, 1e-12),
                   deeper=float(removed_total.max()) * MM_ON_H, wet=wet / volume, why=None)
