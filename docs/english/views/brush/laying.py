# READING VIEW of движок/кисть/укладка.py with names in English. Not runnable; edit the original.
"""Укладка на холст: перенос краски по гребням, растр, запись в слой."""
import numpy as np

from .fast import clamp, surround, by_samples, by_ridge, between_steps

from .paint import in_ks, from_ks, PResponseLResponseWith, transmittance_film
from .numbers import (
    _A_PRESSURE, _LIFT_GR, _JAB_TUFT, _FURROW_SPREAD, _BResponseBrush, _SIDE_CHARGE, _SIDE_ONE, _SIDE_SHIFT, _LENGTHS_LAID_FLAT, _DResponseLAnd, _DREYF, _DYShAndT, _BEHIND_EDGE, _BrushResponseNHAndBrush,
    _BrushResponseRZAndN, _EDGE_ONE, _EDGE_NARROW, _BrushRResponseMBrushA, _BrushRAtG2, _SOFT_RUB_IN, _MYaGBrushResponseWithT, _ResponseBResponseYMA, _LAG_GR, _PEREGRAtZ,
    _PLAShMYa, _PResponseDAtShBrushA, _PAD_0, _PAD_SWEEP, _PResponseLE, _THRESHOLD_RUB_IN, _LAYER_PRESSURE, _WithCanvasInAT, _TAAt, _TONE_SWEEP, _AtGResponseL,
    _SWEEP_RUB_IN, _ShBrush, _ShTRAndCanvas, _STREAK_ONE, _YaDRResponse)
from . import numbers as H
from .tool import scale_thicknesses
from .path import _path, reserve_fringe
from .strands import _edge, _window_by, _landing, _makeup, _noise, opens_hand
from .charge import _charge, _layout, _streak
from .tiles import Tiles
from .furrow import furrows, uneven_reserve, profile_furrows, tone_furrows
from .growing_together import Streams, Bank, Gap, Edge
from .grain import by_grain
from .soft import Soft, separate
from . import soft as M
from .knife import sweep_knife


def _blur(a, times=1):
    """Размытие [1 2 1]/4 по обеим осям, раз проходов. За краем — ноль."""
    for _ in range(times):
        b = a * 0.5
        b[1:] += a[:-1] * 0.25
        b[:-1] += a[1:] * 0.25
        a = b * 0.5
        a[:, 1:] += b[:, :-1] * 0.25
        a[:, :-1] += b[:, 1:] * 0.25
    return a


def _box(a, r):
    """Среднее по квадрату (2r+1)^2 через накопленные суммы. За краем — ноль."""
    k = 2 * r + 1
    c = np.cumsum(surround(a, ((r + 1, r), (0, 0))), axis=0)
    a = (c[k:] - c[:-k]) / k
    c = np.cumsum(surround(a, ((0, 0), (r + 1, r))), axis=1)
    return (c[:, k:] - c[:, :-k]) / k


# ------------------------------------------------------------------------- растр
def _spatter2(xs, ys, bw, bh, weights, piece=1 << 20, at_once=True):
    """Разложить отсчёты (xs, ys — в пикселях прямоугольника мазка) на сетку вдвое мельче
    ядром 3x3 (квадратичный B-сплайн) и сложить 2x2. Ответ — список полей (bh, bw).
    Отсчёты идут кусками по `кусок` (v4, 1д): ядро 3x3 одного куска держит 9 × кусок чисел, а не 9 × все отсчёты мазка
    (спираль ореола R 440 — 24,7 млн отсчётов, 1,66 ГБ на одно поле). Складывает `np.add.at` по порядку отсчётов — те же
    суммы в том же порядке, что `np.bincount` всех разом: бит в бит."""
    fw, fh = 2 * bw, 2 * bh
    shifts = np.array([-fw - 1, -fw, -fw + 1, -1, 0, 1, fw - 1, fw, fw + 1], dtype=np.intp)
    if at_once and len(xs) <= piece and len(weights) > 1 and bw > 1 and _at_once():
        return _spatter_at_once(xs, ys, bw, bh, weights, shifts)
    fields = [np.zeros(fw * fh) for _ in weights]
    for a in range(0, len(xs), piece):
        xr = xs[a:a + piece] * 2.0 - 0.5
        yr = ys[a:a + piece] * 2.0 - 0.5
        ix = np.rint(xr).astype(np.intp)
        iy = np.rint(yr).astype(np.intp)
        usable = (ix >= 1) & (ix < fw - 1) & (iy >= 1) & (iy < fh - 1)
        vk = [w[a:a + piece] for w in weights]
        if not usable.all():
            ix, iy, xr, yr = ix[usable], iy[usable], xr[usable], yr[usable]
            vk = [w[usable] for w in vk]
        fx = xr - ix
        fy = yr - iy
        wx = np.stack([0.5 * (0.5 - fx) ** 2, 0.75 - fx * fx, 0.5 * (0.5 + fx) ** 2], axis=1)
        wy = np.stack([0.5 * (0.5 - fy) ** 2, 0.75 - fy * fy, 0.5 * (0.5 + fy) ** 2], axis=1)
        w9 = (wy[:, :, None] * wx[:, None, :]).reshape(-1, 9)
        idx = ((iy * fw + ix)[:, None] + shifts[None, :]).ravel()
        del xr, yr, ix, iy, fx, fy, wx, wy
        for i, w in enumerate(vk):
            if a == 0 and len(xs) <= piece:              # Е-3: один кусок — `np.bincount` по нулям: те же суммы в том же порядке
                fields[i] = np.bincount(idx, (w9 * w[:, None]).ravel(), fw * fh)
            else:
                np.add.at(fields[i], idx, (w9 * w[:, None]).ravel())
    return [_add_up_2x2(f, bh, bw) for f in fields]


def _usable_samples(xs, ys, fw, fh, weights):
    """Отсчёты внутри сетки `_брызги2` с их долями ядра 3x3 — те же числа, что в её цикле (один кусок)."""
    xr = xs * 2.0 - 0.5
    yr = ys * 2.0 - 0.5
    ix = np.rint(xr).astype(np.intp)
    iy = np.rint(yr).astype(np.intp)
    usable = (ix >= 1) & (ix < fw - 1) & (iy >= 1) & (iy < fh - 1)
    if not usable.all():
        ix, iy, xr, yr = ix[usable], iy[usable], xr[usable], yr[usable]
        weights = [w[usable] for w in weights]
    fx = xr - ix
    fy = yr - iy
    wx = (0.5 * (0.5 - fx) ** 2, 0.75 - fx * fx, 0.5 * (0.5 + fx) ** 2)
    wy = (0.5 * (0.5 - fy) ** 2, 0.75 - fy * fy, 0.5 * (0.5 + fy) ** 2)
    w9 = np.empty((len(ix), 9))                       # `(wy[:, :, None] * wx[:, None, :]).reshape(-1, 9)` по столбцам
    for a in range(3):
        for b in range(3):
            np.multiply(wy[a], wx[b], out=w9[:, 3 * a + b])
    return ix, iy, w9, weights


def _spatter_at_once(xs, ys, bw, bh, weights, shifts):
    """Е-2: `_брызги2` одним кусом — все поля за один проход. Отсчёт — столбец разреженной матрицы с девятью ячейками
    сетки (CSC готова без сортировки), `csc_matvecs` идёт по отсчётам по порядку и в каждую ячейку прибавляет
    `доля ядра × вес` всех полей разом, с нуля — те же произведения и те же суммы в том же порядке, что `np.bincount`
    каждого поля: бит в бит (сверка — `_разом`). Восемь проходов `bincount` и восемь умножений 9 × N — один проход."""
    fw, fh = 2 * bw, 2 * bh
    ix, iy, w9, weights = _usable_samples(xs, ys, fw, fh, weights)
    n, K, M = len(ix), len(weights), fw * fh
    idx = ((iy * fw + ix)[:, None] + shifts[None, :]).ravel()
    Wn = np.empty((n, K))
    for i, w in enumerate(weights):
        Wn[:, i] = w
    Y = np.zeros((M, K))
    _WithP.csc_matvecs(M, n, K, np.arange(0, 9 * n + 1, 9, dtype=np.intp), idx, w9.ravel(), Wn.ravel(), Y.ravel())
    g = Y.reshape(bh, 2, bw, 2, K)                         # bw > 1: `_сложить_2x2` поэлементно
    with_ = (g[:, 0, :, 0] + g[:, 0, :, 1]) + (g[:, 1, :, 0] + g[:, 1, :, 1])
    return [np.ascontiguousarray(with_[..., i]) for i in range(K)]


_WithP, _SP_VALID = None, None


def _at_once():
    """Можно ли `_брызги_разом`: есть `scipy.sparse._sparsetools` и на пробе она даёт те же байты, что `np.bincount`
    (сборка с FMA или иным порядком сумм — нет, и тогда считает прежний путь)."""
    global _WithP, _SP_VALID
    if _SP_VALID is None:
        try:
            from scipy.sparse import _sparsetools as sp
            _WithP = sp
            t = np.arange(3000.0)                              # проба без случайных чисел: дробные места, разброс порядков
            xs, ys = 6.0 + 7.0 * np.sin(t * 0.7311), 4.0 + 5.0 * np.cos(t * 1.1173)
            weights = [np.sin(t * (0.37 + 0.21 * j)) * 10.0 ** np.round(8.0 * np.sin(t * (0.53 + j))) for j in range(3)]
            weights[2][::7] = 0.0
            fw, fh = 24, 16
            shift = np.array([-fw - 1, -fw, -fw + 1, -1, 0, 1, fw - 1, fw, fw + 1], dtype=np.intp)
            a = _spatter_at_once(xs, ys, 12, 8, weights, shift)
            b = _spatter2(xs, ys, 12, 8, weights, at_once=False)
            _SP_VALID = all(x.tobytes() == y.tobytes() for x, y in zip(a, b))
        except Exception:
            _SP_VALID = False
    return _SP_VALID


def _add_up_2x2(f, bh, bw):
    """`f.reshape(bh, 2, bw, 2).sum(axis=(1, 3))` бит в бит (Е-3): numpy при bw > 1 складывает пары строк, (a + b) + (c + d);
    при bw = 1 — своим порядком, там — им же."""
    if bw == 1:
        return f.reshape(bh, 2, bw, 2).sum(axis=(1, 3))
    g = f.reshape(bh, 2, bw, 2)
    return (g[:, 0, :, 0] + g[:, 0, :, 1]) + (g[:, 1, :, 0] + g[:, 1, :, 1])


def _beside_with_film(film):
    """Пиксели не дальше 2 px от плёнки (маска) — вынесено из `положить_мазок` как было."""
    bh, bw = film.shape
    dense = surround(film, 2)
    beside = np.zeros((bh, bw), dtype=bool)
    for dy, dx in _BrushRAtG2:
        beside |= dense[2 + dy:2 + dy + bh, 2 + dx:2 + dx + bw]
    return beside


def _wet_below(lower, take, wet_h, Co):
    """Сырая краска, с которой мазок мешается: своя (сыр_h, цвет Cо) и нижних слоёв, если просили (`мешать_ниже`);
    схватившаяся плёнка подвижна лишь отчасти. Ответ — (толщина, цвет смеси по толщине, [(слой, толщина, сырая,
    цвет)] нижних). Вынесено из `положить_мазок` (Ф-1: функция на потолке строк)."""
    wet_c, bottom = Co, []
    for ns in lower:
        hn, wn = take(ns.thickness), take(ns.moisture)
        sn = hn * wn * (1.0 - _WithCanvasInAT * take(ns.degree_of_set))
        if sn.max() > 1e-4:
            bottom.append((ns, hn, sn, take(ns.color, True)))
    if bottom:
        sum = wet_h.copy()
        tsv = (Co if Co is not None else np.zeros(wet_h.shape + (3,))) * wet_h[..., None]
        for _, _, sn, cn in bottom:
            sum += sn
            tsv += cn * sn[..., None]
        wet_c = tsv / np.maximum(sum, 1e-9)[..., None]
        wet_h = sum
    return wet_h, wet_c, bottom


def _color_without_coverage(C, A, h, Co, ho, ts, V, eps):
    """Ф-1: плёнка мутности 0 не кроет (покрытие 0), но лежит краской своего цвета: её подбирает и с ней мешается
    следующий ход. Где покрытия нет, а толщина есть, цвет — смесь лежащей и новой краски по толщине; там, где покрытие
    есть, цвет прежний."""
    empty = (A <= eps) & (h > eps)
    if not empty.any():
        return C
    C = C.copy()
    vl = ho[empty] if Co is not None else 0.0
    lies = Co[empty] if Co is not None else 0.0
    C[empty] = (lies * np.asarray(vl)[..., None] + ts[empty] * V[empty][:, None]) / np.maximum(np.asarray(vl) + V[empty], eps)[:, None]
    return C


def _on_gap(p, canvas, local, color, thickness, turbidity, Co):
    """П. 10, инструмент 3: не кроющая краска — мутная доля плёнки кроет своим цветом (покрытие), остальная темнит лежащее
    на просвет: и краску своего слоя (Cо — ответ), и то, что под слоем (п — массив пропуска слоя). Сколько — решает
    толщина плёнки в этом месте."""
    T = transmittance_film(color, thickness, turbidity)
    p[canvas] = p[canvas] * T[local]
    return None if Co is None else Co * T


def _gap_behind_thickness(p, canvas, local, was, became):
    """П. 10, инструмент 3 (3а-2): кисть сняла или разровняла сырую не кроющую краску слоя — её пропуск идёт за толщиной:
    где слой стал тоньше, нижнее светлеет, где толще — темнеет (пропуск — степень толщины, `краска.пропуск_плёнки`)."""
    share = np.where(was > 1e-6, clamp(became / np.maximum(was, 1e-6), 0.0, 2.0), 1.0)
    p[canvas] = np.power(p[canvas], share[local][..., None])


def _drains(V, bottom, liquid=1.0):
    """П. 10, инструмент 3: жидкая (не кроющая) краска после кисти оплывает. Своё (3а-2): за ворсом она смыкается — щель
    в волосок шириной затягивается, борозды собственных волосков выравниваются (чем жиже, тем ровнее), широкие полосы
    хода остаются. Лежащее: с гребней рельефа `низ` стекает в его ямки — в ямке слой толще, на гребне тоньше. Ровняется
    толщина краски, краски столько же; за свой след не течёт."""
    total = V.sum()
    has = ((V > 0.0) | (_blur((V > 0.0) * 1.0) >= 0.5)) * 1.0
    even = _blur(V, H._InAndD[5]) / np.maximum(_blur(has, H._InAndD[5]), 1e-6)
    V = V + (even - V) * has * liquid
    top = (bottom + V) * has
    even = _blur(top, H._InAndD[4]) / np.maximum(_blur(has, H._InAndD[4]), 1e-6)
    V2 = np.maximum(V + (even - top) * has * (H._InAndD[3] * liquid), 0.0)
    return V2 * (total / max(V2.sum(), 1e-12))


def _slump(V, color, bottom, liquid):
    """`_стекает` для плёнки мазка: в затянутую щель краска пришла от соседей — и цвет их (`цвет` правится на месте)."""
    V2 = _drains(V, bottom, liquid)
    closed_over = (V2 > 0.0) & ~(V > 0.0)
    if closed_over.any():
        color[closed_over] = (np.stack([_blur(color[..., i] * V) for i in range(3)], axis=-1)[closed_over]
                          / np.maximum(_blur(V)[closed_over], 1e-9)[:, None])
    return V2


def _frame(a, field=0):
    """К4: срезы наименьшего прямоугольника с маской `a`, раздвинутого на `поле` px (в пределах массива); None — маска пуста."""
    ys, xs = np.flatnonzero(a.any(axis=1)), np.flatnonzero(a.any(axis=0))
    if not len(ys):
        return None
    return (slice(max(int(ys[0]) - field, 0), int(ys[-1]) + 1 + field), slice(max(int(xs[0]) - field, 0), int(xs[-1]) + 1 + field))


def _circle(a, R):
    """К4: маска `a`, раздвинутая кругом радиуса `R` px (пиксель в ответе, если в круге вокруг него есть пиксель маски).
    За краем — пусто. Считается только вокруг маски: по строке — расстояние до ближнего её пикселя, по столбцу — хорды
    круга."""
    result = np.zeros(a.shape, dtype=bool)
    Ri = int(R)
    frame = _frame(a, Ri)
    if frame is None:
        return result
    a = a[frame]
    H, Wd = a.shape
    far = min(Ri + 1, 255)
    if Ri:
        x = np.arange(Wd, dtype=np.int32)
        d = np.minimum(x - np.maximum.accumulate(np.where(a, x, -Wd - Ri - 2), axis=1),
                       np.minimum.accumulate(np.where(a, x, 2 * Wd + Ri + 2)[:, ::-1], axis=1)[:, ::-1] - x)
        d = np.minimum(d, far).astype(np.uint8 if Ri < 255 else np.int32)
    else:
        d = np.where(a, 0, 1).astype(np.uint8)
    b = np.full((H + 2 * Ri, Wd), far, dtype=d.dtype)
    b[Ri:Ri + H] = d
    part = result[frame]
    for dy in range(-Ri, Ri + 1):
        part |= b[Ri + dy:Ri + dy + H] <= int((R * R - dy * dy) ** 0.5)
    return result


def _meniscus(film, R):
    """К4: плёнка текучей краски (маска), сомкнутая мениском радиуса `R` px: без плёнки остаётся только то место, куда
    встаёт круг такого радиуса, не задев её. Щель и дыра уже 2R затягиваются целиком; клин между двумя кромками залит до
    вогнутой дуги радиуса R; за общую касательную двух кромок и за наружную кромку краска не идёт."""
    m = int(R) + 1
    frame = _frame(film, m)
    result = np.zeros(film.shape, dtype=bool)
    if frame is None:
        return result
    a = surround(film[frame], m)
    result[frame] = ~_circle(~_circle(a, R), R)[m:-m, m:-m]
    return result


def _pour(grain, where, passes=64):
    """К4: связные куски маски `где`, в которых есть хоть один пиксель `зерно` (от него весь кусок): проходы по строкам и
    столбцам — отрезок маски, задетый зерном, берётся весь; не больше `проходов` пар проходов."""
    already = grain & where
    if not already.any():
        return already
    H, Wd = where.shape
    segments = []
    for axis in (1, 0):
        start = np.cumsum(~where, axis=axis, dtype=np.int64)
        start += (np.arange(H, dtype=np.int64)[:, None] * (Wd + 1)) if axis == 1 else (np.arange(Wd, dtype=np.int64)[None, :] * (H + 1))
        segments.append((start, int(start.max()) + 1))
    was = int(already.sum())
    for _ in range(passes):
        for start, total in segments:
            already = where & (np.bincount(start[already], minlength=total) > 0)[start]
        became = int(already.sum())
        if became == was:
            break
        was = became
    return already


def _seam(own, foreign, trace_sv, trace_h, W):
    """К4: место между плёнкой своего текучего хода и лежащей сырой текучей плёнкой, которое затягивает краска. Решает не
    пиксель, а шов целиком: вдоль зоны, где обе кромки рядом, краска смыкается, если следы двух кистей (где ворс доставал
    до холста) хоть где-то сошлись вплотную, — от этого места мениск идёт по всему шву и заливает его карманы до ширины
    двух радиусов `_МЕНИСК`; где кромки расходятся клином, залив кончается вогнутой дугой. Шов, где следы не сошлись нигде
    (зазор оставлен рукой), остаётся зазором с кромками, как легли."""
    R = max(H._MENAndWithBrush[0] * W, H._MENAndWithBrush[1])
    both = own | foreign
    zone = _circle(own, R + 1.0) & _circle(foreign, R + 1.0)
    encounter = _circle(trace_sv | own, 1.5) & _circle(trace_h | foreign, 1.5)
    if not (encounter & zone).any():
        return None
    seam = _pour(encounter, zone)
    slit = _meniscus(both, R) & ~both
    slit = _pour(seam, slit)                  # карман шва берётся весь, и там, где он отошёл от одной из кромок
    # дыра, замкнутая краской со всех сторон (выемка кромки одного хода против кромки другого), — не клин: наружу она не
    # открыта, краска идёт в неё с обоих берегов на длину выравнивания `_РОВНО`
    holes = _holes(both | slit, max(H._RResponseInNResponse * W, R))
    if holes is not None:
        slit |= _pour(_circle(foreign, 1.5), _pour(_circle(own | slit, 1.5), holes))     # дыра шва: у неё оба берега
    return slit if slit.any() else None


def _holes(film, R):
    """К4: дыры плёнки текучей краски, замкнутые ею со всех сторон (наружу, к грунту за кромкой, не открыты), — та их
    часть, куда не встаёт круг радиуса `R` px; None — таких нет."""
    background = ~film
    edge = np.zeros_like(background)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    holes = background & ~_pour(edge, background)
    if not holes.any():
        return None
    frame = _frame(holes, 2 * (int(R) + 1))        # мениск — только вокруг дыр
    holes[frame] &= _meniscus(film[frame], R)
    return holes if holes.any() else None


def _level_seam(V2, has, slit, beside, foreign, r, W, share):
    """К4: текучая краска двух берегов расходится по шву до общего уровня. Своя плёнка (`V2`, маска `есть` — уже со швом
    `щель`) и лежащая (`рядом`, маска `чужая`, где её не накрыл свой ход) у шва — один слой: он ровняется теми же тремя
    средними, что своя плёнка, на длине не меньше поперечника мениска (шов такой ширины краска проходит с двух берегов), на
    `доля`. Провала в шве не остаётся: уровень в нём — общий с берегами, берега у шва на столько же ниже; краски в слое
    столько же. Считается только в досягаемости шва — круг в полторы такие длины вокруг него, к краю сила сходит на нет. Ответ: (своя плёнка, сколько отдала лежащая — None, если нисколько)."""
    rsh = max(r, int(np.ceil(2.0 * max(H._MENAndWithBrush[0] * W, H._MENAndWithBrush[1]))))
    sh = _frame(slit, 6 * rsh)
    e, V = has[sh] > 0.0, V2[sh]
    h = foreign[sh] & ~e
    def three_times(a):
        return _box(_box(_box(a, rsh), rsh), rsh)
    # досюда краска шва дотекает: круг вокруг шва, а не квадрат, и сходит на нет плавно — границы по осям холста нет
    rk = max(1, rsh // 2)
    strength = _box(_box(_box(_circle(slit[sh], 1.5 * rsh) * 1.0, rk), rk), rk) * (e | h)
    U = np.where(e, V, beside[sh] * h)
    Un = U + (three_times(U) / np.maximum(three_times((e | h) * 1.0), 1e-6) - U) * strength * share
    Un = Un - strength * U * ((Un.sum() - U.sum()) / max((strength * U).sum(), 1e-12))     # краски столько же: остаток — туда же, по той же силе
    Un = np.maximum(Un, 0.0)
    V2 = V2.copy()
    V2[sh] = np.where(e, Un, V)
    went = np.zeros(V2.shape)
    went[sh] = np.where(h, U - Un, 0.0)
    return V2, went


def _level(V, color, W, share, tau, beside=None, ts_beside=None, touches=None, trace=None, paints=None):
    """П. 15: слой текучей кроющей краски за ворсом ровняется на длине в `_РОВНО` ширины кисти (не меньше 2 px) —
    три средних по квадрату; краски столько же, за свою плёнку она не течёт. Пигмент течёт с ней: цвет сходится к среднему
    по той же длине (`цвет` правится на месте). Дыра в своей плёнке уже `_РОВНО_ЩЕЛЬ` затягивается (`_мениск`). К4: с
    лежащей сырой текучей плёнкой своего слоя (`рядом` — её толщина, `ц_рядом` — цвет, `след` — след текучей кисти в слое,
    `трогает` — след своего ворса) ход смыкается швом целиком (`_шов`) — своя краска затекает в него при выравнивании,
    пигмент в шве — обеих сторон; кромка у грунта — где легла."""
    r = max(2, int(round(H._RResponseInNResponse * W)))
    own = V > 0.69 * tau                         # плёнка, что кроет хотя бы наполовину; редкая кайма тоньше лежит как легла
    n = max(2, int(round(H._EVEN_SLIT * W)))       # дыра в своей плёнке не шире этого затягивается; кромка плёнки — где была
    has = _meniscus(own, 0.5 * n)
    holes = _holes(has, max(H._RResponseInNResponse * W, H._MENAndWithBrush[1]))       # дыра внутри своей плёнки: краска идёт в неё со всех сторон на длину выравнивания
    if holes is not None: has |= holes
    foreign = ((beside > 0.69 * tau) & (trace > 0.5)) if beside is not None and trace is not None else None   # текучая с текучей: без следа текучей кисти лежащий ход — не берег
    slit = _seam(own, foreign, touches > H._MENAndWithBrush[2], trace > H._MENAndWithBrush[2], W) if foreign is not None and foreign.any() else None
    if slit is not None:
        has = has | slit
        n = max(n, int(np.ceil(max(H._MENAndWithBrush[0] * W, H._MENAndWithBrush[1]))))
    has = has * 1.0
    def three_times(a):
        return _box(_box(_box(a, r), r), r)
    weight = np.maximum(three_times(has), 1e-6)
    V2 = V + (three_times(V * has) / weight - V) * has * share
    V2 = np.where(has > 0.0, V2 * ((V * has).sum() / max((V2 * has).sum(), 1e-12)), V)
    went = None
    if slit is not None: V2, went = _level_seam(V2, has, slit, beside, foreign, r, W, share)
    if has.any():
        # краска стекается вместе с пигментом: тон прядей и чужая краска в щели сходятся к общему (в пигментном состоянии);
        # считается только в рамке плёнки и только по ней
        rts = max(r, int(round(H._EVEN_TONE * W)))      # размешанная с маслом краска одного тона: тон сходится по всей ширине кисти
        rm = _frame(has > 0.0, rts)
        where, weight_ts, tsv = has[rm] > 0.0, (V * has)[rm], color[rm]
        if paints is not None:
            _tone_charge(tsv, where, weight_ts, V2[rm], paints, rts, r, share)
        else:
            ks = np.zeros((2 * PResponseLResponseWith,) + where.shape)
            ks[:, where] = (in_ks(tsv[where]) * weight_ts[where][:, None]).T
            ts = from_ks(_box_color(_box_color(ks, rts), rts)[:, where].T / np.maximum(_box(_box(weight_ts, rts), rts)[where], 1e-9)[:, None])
            # на месте осталась доля своей краски, остальное притекло с общим тоном: в затянутую дыру (своей краски не было,
            # цвета у пикселя нет) краска приходит со своим пигментом целиком
            left = (V[rm][where] * (1.0 - share))[:, None]
            arrived = np.maximum(V2[rm][where][:, None] - left, 0.0)
            tsv[where] = (tsv[where] * left + ts * arrived) / np.maximum(left + arrived, 1e-12)
        if slit is not None:
            # в шов краска пришла с обоих берегов: пигмент — своей плёнки и соседней по толщине у берегов (у каждого берега —
            # его краска возле шва, смешиваются два берега пигментом)
            sh = _frame(slit, n)
            shch, in_with, in_h, tsv = slit[sh], V2[sh] * own[sh], beside[sh] * foreign[sh], color[sh]
            m_with, m_h = _box(in_with, n)[shch][:, None], _box(in_h, n)[shch][:, None]
            ts_with = _box_color(np.moveaxis(tsv * in_with[..., None], -1, 0), n)[:, shch].T / np.maximum(m_with, 1e-9)
            ts_h = _box_color(np.moveaxis(ts_beside[sh] * in_h[..., None], -1, 0), n)[:, shch].T / np.maximum(m_h, 1e-9)
            tsv[shch] = from_ks((in_ks(ts_with) * m_with + in_ks(ts_h) * m_h) / np.maximum(m_with + m_h, 1e-9))
    return V2, went


def _shares_paints(ts, paints):
    """К7а: доли красок набора в цвете пикселя (N, 3) -> (N, n), в сумме 1. Растр кладёт краски в пиксель в линейном
    цвете, и цвет пикселя — их взвешенная сумма: доли — та смесь красок набора, что ближе всего к цвету пикселя
    (наименьшие квадраты, отрицательных долей нет). Порога «чья краска» нет: пиксель на стыке полос несёт обе."""
    brush = np.asarray(paints, np.float64)
    u = (ts - brush[-1]) @ np.linalg.pinv((brush[:-1] - brush[-1]).T).T
    w = np.maximum(np.concatenate([u, 1.0 - u.sum(1, keepdims=True)], axis=1), 0.0)
    return w / w.sum(1, keepdims=True)


def _tone_charge(tsv, where, V, V2, paints, rts, r, share):
    """К7, К7а: пигмент текучей плёнки набора в несколько красок (`цв` правится на месте; `где` — плёнка, `V` — краска до
    выравнивания, `V2` — после). Плёнка ровняет толщину, а пигмент идёт только с краской. Пиксель несёт доли красок набора
    (`_доли_красок`), и краска каждой полосы размешана с маслом сама по себе: её тон (отличие от набранного цвета) сходится
    к общему на длине `rц` по её же краске, доли в пикселе — как легли. Что лежало, лежит на месте; в борозду и щель
    (`V2` > `V`) притекает излишек соседей (`V` > `V2`) со своими долями красок: он расходится по плёнке кругами (три средних
    подряд, радиус от шага к шагу вдвое больше, в сумме не дальше длины выравнивания `r`), и место, куда краска прибывает, берёт её из дошедшего — ближний
    берег успевает первым. На стыке полос сходятся обе краски и мешаются пигментом."""
    has, n, brush = where & (V > 0.0), len(paints), np.asarray(paints, np.float64)
    w = np.zeros((n,) + where.shape)
    w[:, has] = _shares_paints(tsv[has], brush).T
    δ = np.zeros(tsv.shape)
    δ[has] = tsv[has] - w[:, has].T @ brush
    a = np.zeros((4 * n,) + where.shape)                 # на краску: её тон на её толщину (3) и толщина
    for q in range(n):
        a[4 * q:4 * q + 3], a[4 * q + 3] = np.moveaxis(δ, -1, 0) * (w[q] * V), w[q] * V
    b = _box_color(_box_color(a, rts), rts)
    def tone(q, m):
        return brush[q] + (b[4 * q:4 * q + 3, m] / np.maximum(b[4 * q + 3, m], 1e-12)).T
    own = np.zeros(tsv.shape)
    own[has] = tsv[has] - δ[has] * share + sum((tone(q, has) - brush[q]) * w[q][has][:, None] for q in range(n)) * share
    left, arrived = np.minimum(V, V2) * has, np.maximum(V2 - V * has, 0.0) * where
    excess, waits, reached = w * (np.maximum(V - V2, 0.0) * has), arrived.copy(), np.zeros(w.shape)
    path2, ρ = 0.0, 1
    while True:
        path2 += ρ * (ρ + 1.0)                          # три средних радиуса ρ разносят краску на √(ρ(ρ+1)) px; шаги складываются квадратами
        if waits.max() <= 1e-12 or path2 > max(r * r, 2.0): break
        excess = _box_color(_box_color(_box_color(excess, ρ), ρ), ρ) * where
        total = excess.sum(0)
        taken = np.minimum(waits, total) / np.maximum(total, 1e-300)
        reached += excess * taken
        excess *= 1.0 - taken
        waits -= taken * total
        ρ *= 2
    reached += w * waits                                   # чего излишек соседей не покрыл — своя же краска этого места
    waits = waits * ~has                                 # голое место дальше длины выравнивания: краска та, что дошла, а нет её — та, что вокруг на длине тона
    if (waits > 1e-12).any():
        around = np.where(reached.sum(0) > 0.0, reached, b[3::4])
        reached += around / np.maximum(around.sum(0), 1e-300) * waits
    arrived = reached.sum(0)
    tsv[has] = own[has]
    m = where & (arrived > 0.0)                           # пигментом мешается только то место, куда краска пришла
    ks = in_ks(clamp(own[m], 0.0, 1.0)) * left[m][:, None]
    for q in range(n):
        ks += in_ks(clamp(tone(q, m), 0.0, 1.0)) * reached[q][m][:, None]
    tsv[m] = from_ks(ks / (left + arrived)[m][:, None])


def _box_color(a, r):
    """`_коробка` для поля (каналы, H, W) — все каналы разом, те же числа."""
    k = 2 * r + 1
    C, H, Wd = a.shape
    b = np.zeros((C, H + k, Wd))
    b[:, r + 1:r + 1 + H] = a
    c = np.cumsum(b, axis=1)
    a = (c[:, k:] - c[:, :-k]) / k
    b = np.zeros((C, H, Wd + k))
    b[:, :, r + 1:r + 1 + Wd] = a
    c = np.cumsum(b, axis=2)
    return (c[:, :, k:] - c[:, :, :-k]) / k


def _relief_film(V, thickly, furrow):
    """Жидкая краска оплывает (п. 5): мелкий рельеф борозд после кисти сглаживается, густая держит гребень. Оплывает масло,
    а не пигмент (5в): рельеф ровнее, а просвет нижнего слоя на дне борозды и её тон — как легли. За кромку следа краска
    не течёт: сглаживание идёт только по закрашенному."""
    if not (thickly < _FURROW_SPREAD[1] and furrow is not None):
        return V
    has_V = (V > 0.0) * 1.0
    return V + (_blur(V, 2) / np.maximum(_blur(has_V, 2), 1e-6) - V) * has_V * (_FURROW_SPREAD[0] * (1.0 - thickly / _FURROW_SPREAD[1]))


def _slope_edges(has, thinner):
    """v4, 5в: доля толщины плёнки у её края (маска `есть`) — smoothstep по расстоянию до края внутрь плёнки на длине
    `Ч._КРОМКА_ММ` в мм; None — склон короче пикселя или края нет."""
    m = float(thinner or 0.0)
    L = (H._RIM_MM[0] + (H._RIM_MM[1] - H._RIM_MM[0]) * m ** H._RIM_MM[2]) * 8.0 * H._MM
    if L <= 1.0 + 1e-9 or not has.any() or has.all():
        return None
    from scipy.ndimage import distance_transform_edt
    t = clamp((distance_transform_edt(has) - 0.5) / L, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _rim(V_r, thinner, underfill=None):
    """v4, 5в: край плёнки хода — не обрыв в пиксель: рельеф плёнки у края спадает к нулю на длине `Ч._КРОМКА_ММ` в мм
    (густая — резко, жидкая — мягко, по разбавителю), smoothstep по расстоянию до края внутрь плёнки: ступенька высотой с
    плёнку становится склоном той же высоты. Масло края уходит в ход, за кромку не течёт; пигмент (цвет и покрытие)
    лежит как лёг. v4, 5д: и по сырой краске своего слоя — тем же склоном (`_срослась` обрыв в 1 px не снимала: светлая
    нить по контуру каждого хода с разбавителем, `проверка/ДИАГНОЗ_гребни_свода.md`). v4, 5д-2: по сырой склоном спадает
    только то, что встаёт над прежней поверхностью: `недолив` — сколько сырой краски под ходом убыло (снято кистью,
    вытеснено, ушло в шов); столько плёнка доливает в уровень без склона — иначе у края по сырому ложбина, и жидкая
    заливка рядами шьёт каждый ряд швом. По сухому `недолив` 0 — как в 5в. Короче пикселя — как было."""
    slope = _slope_edges(V_r > 0.0, thinner)
    if slope is None:
        return V_r
    V2 = V_r * slope
    return V2 if underfill is None else V2 + (V_r - V2) * clamp(underfill / np.maximum(V_r, 1e-9), 0.0, 1.0)


def _mix_in(V, mixed_in, ts_new, Co, eps):
    """Плёнка хода с вмешанной на месте лежащей сырой краской: (толщина, что красит, её цвет) — вынесено из
    `положить_мазок` как было (5в)."""
    M = V if mixed_in is None else V + mixed_in
    ts_top = ts_new
    if mixed_in is not None:
        blend = mixed_in > 1e-5
        if blend.any():
            ts_top = ts_new.copy()
            share_new = V[blend] / (M[blend] + eps)
            ts_top[blend] = from_ks(in_ks(Co[blend]) * (1.0 - share_new)[:, None] + in_ks(ts_new[blend]) * share_new[:, None])
    return M, ts_top


def _flowing(V, color, W, even, tau, bottom, beside=None, ts_beside=None, touches=None, trace=None, paints=None):
    """П. 15, поле мазка `ровно`: текучая кроющая краска (набрана густо, с маслом) за ворсом смыкается в ровный слой —
    щели и борозды своих волосков и полосы прядей затягиваются краской соседей (`_ровнять`); за свою плёнку не течёт —
    край как лёг. Рельеф слоёв под своим (`низ` — та его доля, что видна сквозь верхний слой, борозды подмалёвка) она
    заливает: в ямке слой толще, на гребне тоньше. Ответ: (пигмент плёнки, её рельеф)."""
    (V, went), V_r = _level(V, color, W, even, tau, beside, ts_beside, touches, trace, paints), None
    V_r = V
    if np.ndim(bottom):
        has = (V > 0.0) * 1.0
        V_r = np.maximum(V - (bottom - _blur(bottom * has, 6) / np.maximum(_blur(has, 6), 1e-6)) * has * even, 0.3 * V)
    return V, V_r, went


def _close(V_r, bottom, dry, flowing, thinner=0.0):
    """Ф-2: текучая плёнка хода (`V_р` — её рельеф) по сухой лежащей краске затекает в низины её рельефа: `низ` —
    поверхность под плёнкой (лежащая краска своего слоя и слоёв ниже), её никто не трогает. Течёт сама плёнка: от места,
    где её верх (`низ` + плёнка) выше, к соседу, где он ниже, — и тем охотнее, чем она толще (`_СМЫК[0]`, куб): плёнка в
    волосок держится за рельеф, толстая расходится в уровень. Отдаёт место только то, что на нём есть: с гребня выше
    плёнки она сходит вся, и гребень стоит как стоял — толстый мазок рельеф держит, рябь мелкого письма тонет. Краски
    столько же, за свой след не течёт; склон края плёнки (`_кромка`, по разбавителю) — её равновесная форма: к краю
    плёнка не стекает и стенкой у него не встаёт. `сух` — доля сухого в лежащем под плёнкой (по сырому своего слоя — как
    было); `текуче` — 0–1 (`ровно`, разбавитель): столько времени у плёнки до схватывания. Пигмент (цвет, покрытие)
    лежит как лёг."""
    V0, steps, dt, power, ceiling = H._WithMYBrush
    n = int(min(np.ceil(steps * float(flowing) * H._MM ** 2), ceiling))
    has = V_r > 0.0
    frame = _frame(has & (dry > 0.0), 2)
    if n <= 0 or frame is None:
        return V_r
    slope = _slope_edges(has, thinner)
    V, b, m = V_r[frame].copy(), bottom[frame], has[frame]
    e = None if slope is None else np.maximum(slope[frame], 0.05)
    by_x, by_y = m[:, 1:] & m[:, :-1], m[1:] & m[:-1]      # течёт только по своему следу
    for _ in range(n):
        s = b + V if e is None else b + V / e              # верх плёнки, как если бы склона края не было
        mobile = np.minimum(V / V0, 1.0) ** power * dt
        if e is not None: mobile = mobile * e
        dx, dy = (s[:, :-1] - s[:, 1:]) * by_x, (s[:-1] - s[1:]) * by_y      # > 0 — течёт вправо, вниз
        right, left = np.maximum(dx, 0.0) * mobile[:, :-1], np.maximum(-dx, 0.0) * mobile[:, 1:]
        down, up = np.maximum(dy, 0.0) * mobile[:-1], np.maximum(-dy, 0.0) * mobile[1:]
        went = np.zeros_like(V)
        went[:, :-1] += right; went[:, 1:] += left; went[:-1] += down; went[1:] += up
        share = np.minimum(V / np.maximum(went, 1e-12), 1.0)                 # больше, чем есть, место не отдаёт
        right, left, down, up = right * share[:, :-1], left * share[:, 1:], down * share[:-1], up * share[1:]
        V[:, :-1] += left - right; V[:, 1:] += right - left
        V[:-1] += up - down; V[1:] += down - up
    result = V_r.copy()
    result[frame] = V_r[frame] + (np.maximum(V, 0.0) - V_r[frame]) * clamp(dry[frame], 0.0, 1.0)
    return result * (V_r.sum() / max(float(result.sum()), 1e-12))


def _trace_flowing(layer, canvas, even, trace, alpha):
    """К4: след текучей кисти в слое (`Слой.текучая`) — по нему следующий текучий ход найдёт этот: где ворс хода с `ровно`
    доставал до холста и где легла его краска. Не текучая краска, легшая поверх, след стирает на своё покрытие."""
    if even:
        if layer.flowing is None: layer.flowing = np.zeros(layer.thickness.shape, np.float32) if isinstance(layer.thickness, np.ndarray) else Tiles(layer.thickness.shape)
        layer.flowing[canvas] = np.maximum(layer.flowing[canvas], trace)
    else:
        layer.flowing[canvas] = layer.flowing[canvas] * (1.0 - alpha)


def _grew_together(h, w, V, even):
    """П. 15: с лежащей рядом сырой краской своего слоя текучая срастается — ступенька на шве двух ходов оплывает."""
    seam = (_blur((V > 0.0) * 1.0, H._EVEN_SEAM) > 0.02) * (h > 1e-9) * clamp(w * 2.0 - 1.0, 0.0, 1.0)
    return h + (_blur(h * seam, H._EVEN_SEAM) / np.maximum(_blur(seam, H._EVEN_SEAM), 1e-6) - h) * seam * even


def _covers(M, tau, turbidity):
    """П. 10, инструмент 3: толщина, которой не кроющая плёнка кроет своим цветом, — её мутная доля; в плёнке тоньше
    кроющей толщины `тау` она кроет слабее своей доли: зёрна пигмента в один слой не смыкаются; в толстом месте (комок
    сухой кисти, пастозный слой) смыкаются, и краска кроет (3а-2)."""
    return M * turbidity * clamp(M / tau, 0.0, 1.0) ** (H._InAndD[6] * (1.0 - turbidity))


def _levels_wet(h, w, even, brush_even):
    """Ф-3: с какой силой кисть ровняет лежащую краску своего слоя под собой (поле) и где та стоит сухой (доля, поле).
    Двигается только сырое. Бугор кисть срезает по его влажности `w` (у корки `схватить` — по подвижной доле): высохший
    стоит. Ложбину и голое место рядом заливает то, что кисть согнала с краски вокруг, — по влажности краски в окне
    ровняния (взвешена толщиной): среди сухой заливать нечем. По сырой краске (`w` = 1) — числа те же, что до Ф-3."""
    wet = _box(_box(h * w, 8), 8)
    w_around = clamp(np.divide(wet, even, out=np.zeros_like(even), where=even > 0.0), 0.0, 1.0)
    return (H._RResponseInNYaET[0] + H._RResponseInNYaET[1] * w) * np.where(brush_even > 0.0, w_around, w), np.where(h > 1e-9, 1.0 - w, 0.0)


def _carries_wetness(h_to, h_after, w, skh, C, A, wiped=False):
    """«Мастихин», первая строка: краска, которую ход согнал с одного места на другое, несёт влажность и схват места,
    откуда пришла. Откуда — краска, убывшая в том же окне, в котором кисть ровняет (две коробки по 8 px): её влажность
    и схват, взвешенные убывшей толщиной. Раньше прибывшее писалось с влажностью места прибытия — на голом грунте 0:
    сырая краска, согнанная на грунт рядом, числилась сухой, а цвета у неё не было вовсе (цвет пикселя без покрытия — что
    осталось в массиве: следующий ход мешал в свою плёнку его). Цвет прибывшее несёт туда, где покрытия не было; где
    лежит своя краска с покрытием, цвет её прежний. Где ничего не прибыло — как было. Ответ: влажность, схват, цвет.
    З-1: у вытертой кисти (`вытерта`) цвет прибывшее несёт и туда, где покрытия почти нет (не больше `_БЕЗ_ЦВЕТА`:
    толщина без пигмента — оплывший рельеф прежних ходов, цвет в массиве случайный, до чёрного), и туда, в окне
    которого ничего не убыло (кисть свела остаток объёма по всей сырой краске под собой) — средний цвет убывшего по
    всему ходу, как влажность и схват: без цвета (чёрной, цветом пустого места слоя) плёнка не ложится. Прочие кисти —
    как было."""
    arrived = np.maximum(h_after - h_to, 0.0)
    if not (arrived > 0.0).any():
        return w, skh, C
    went = np.maximum(h_to - h_after, 0.0)
    ob = _box(_box(went, 8), 8)
    total = float(went.sum())
    w0, with0 = (float((went * w).sum()) / total, float((went * skh).sum()) / total) if total > 1e-12 else (1.0, 0.0)
    has = ob > 1e-9 * max(float(ob.max()), 1e-30)
    w_source = np.where(has, _box(_box(went * w, 8), 8) / np.where(has, ob, 1.0), w0)
    with_source = np.where(has, _box(_box(went * skh, 8), 8) / np.where(has, ob, 1.0), with0)
    left = h_to - went
    denominator = np.maximum(h_after, 1e-12)
    bare = (arrived > 0.0) & (A <= 1e-6) & has
    if C is not None and bare.any():
        C = C.copy()
        C[bare] = np.moveaxis(_box_color(_box_color(np.moveaxis(C * went[..., None], -1, 0), 8), 8), 0, -1)[bare] / ob[bare][:, None]
    if wiped and C is not None and total > 1e-12:
        almost = (arrived > 0.0) & (A <= H._WITHOUT_COLORS) & ~bare
        if almost.any():
            source = np.where(has[..., None], np.moveaxis(_box_color(_box_color(np.moveaxis(C * went[..., None], -1, 0), 8), 8), 0, -1) / np.where(has, ob, 1.0)[..., None], (C * went[..., None]).sum(axis=(0, 1)) / total)
            C = C.copy() if not bare.any() else C
            C[almost] = source[almost]
    return (np.where(arrived > 0.0, clamp((w * left + w_source * arrived) / denominator, 0.0, 1.0), w),
            np.where(arrived > 0.0, clamp((skh * left + with_source * arrived) / denominator, 0.0, 1.0), skh), C)


def _knife(layer, path, brush, W, H, clean, mt, tau, paints, touch, grain, under, take, ho, Ao, wo, similar, Co, local, canvas, frame, writing, opacity, turbidity, transmittance, measurement, why, X, Y):
    """«Мастихин»: ход ножа (`нож.ход_ножа`) — в слой. Нож срезает сырое выше кромки, тянет его валом по ходу, кладёт в
    пустое под кромкой, выдавливает валики за концы кромки и оставляет ком, где оторван; твёрдого (зерно, слои ниже,
    высохшее своего слоя) не мнёт. Здесь — только запись: покрытие, цвет, толщина, влажность и схват слоя."""
    bx0, by0, bw, bh = frame
    bottom = np.zeros((bh, bw))
    if grain is not None:
        bottom = bottom + grain(bx0, by0, bx0 + bw, by0 + bh)[0]
    for ns in under:
        bottom = bottom + take(ns.thickness)
    r = sweep_knife(path, W, H, clean, mt, paints, brush.touch if touch is None else touch, bottom, ho, wo, similar, Co, frame, widths=brush.reserve, field=_PResponseLE - 1.0)
    if r is None or (r['снято'] <= 0.0 and not r['V'].any()):
        return _empty(why, 'нажим' if r is None and not clean else 'чистая' if clean else 'пусто', X, Y)
    eps = 1e-9
    coverage_from = lambda h: 1.0 - np.exp(-np.maximum(h, 0.0) / tau)   # noqa: E731
    V, ts, h1 = r['V'], r['ц'], r['h']
    # где сырую плёнку срезали — открывается то, что под ней (как у кисти)
    A1 = Ao * clamp(coverage_from(h1) / np.maximum(coverage_from(ho), 1e-6), 0.0, 1.0) ** 0.7
    C1 = np.zeros((bh, bw, 3)) if Co is None else Co + (r['ц_мешано'] - Co) * r['мешано'][..., None]
    alpha = coverage_from(V if turbidity >= 1.0 else _covers(V, tau, turbidity)) * opacity
    C_wet = C1
    if turbidity < 1.0: C1 = _on_gap(transmittance(), canvas, local, ts, V * opacity, turbidity, C1)
    A_new = alpha + A1 * (1.0 - alpha)
    C_new = (C1 * (A1 * (1.0 - alpha))[..., None] + ts * alpha[..., None]) / np.maximum(A_new, eps)[..., None]
    h_new = h1 + V
    w_new = np.where(h_new > eps, (r['w'] * h1 + r['wV'] * V) / np.maximum(h_new, eps), 0.0)
    skh_new = np.where(h_new > eps, (r['сх'] * h1 + r['схV'] * V) / np.maximum(h_new, eps), 0.0)
    if layer.transmittance is not None: _gap_behind_thickness(layer.transmittance, canvas, local, ho, h1)
    layer.color[canvas] = _color_without_coverage(C_new, A_new, h_new, C_wet, h1, ts, V, eps)[local]
    layer.coverage[canvas] = A_new[local]
    layer.thickness[canvas] = np.maximum(h_new[local], 0.0); layer.moisture[canvas] = clamp(w_new[local], 0.0, 1.0); layer.degree_of_set[canvas] = clamp(skh_new[local], 0.0, 1.0)
    if layer.flowing is not None: _trace_flowing(layer, canvas, 0.0, np.maximum(r['касание'], V > 0.0)[local], alpha[local])
    brush._charged = r['запас']
    if measurement is not None: measurement(dict(alpha=alpha[local], touch=r['касание'][local], laid=float(V[local].sum()), wet=None if r['сырое'] is None else r['сырое'][local], removed=r['снято'], on_knife=r['на_ноже'], x=path['x'], y=path['y'], W=W, angle=(writing[0], writing[1])))
    return writing


# ------------------------------------------------------------------------- мазок
def _narrow(W):
    """П. 10: во сколько раз кисть уже 16 px (`_ОСТРАЯ[0]`) (1 — не уже): на столько мельче её кромка по сырому — иначе
    кисть 3–6 px вся «кромка» и своей краски по сырому не даёт."""
    return min(W / H._ResponseWithTRAYa[0], 1.0)


def _noise2(rng, a, b, spacing):
    """Н-2: гладкое поле (len(a), len(b)) на плоскости (a, b) — узлы через ~`шаг` тех же единиц на решётке, повёрнутой
    на случайный угол (без осей: края пятен не ложатся прямой ни вдоль хода, ни поперёк), среднее 0, разброс ~1."""
    spacing = max(float(spacing), 1e-3)
    f = rng.uniform(0.0, np.pi)
    A, B = np.asarray(a, float)[:, None], np.asarray(b, float)[None, :]
    if not A.size or not B.size:
        return np.zeros((A.shape[0], B.shape[1]))             # кисть в один гребень: поля между гребнями нет (ведущих 0)
    u = (A * np.cos(f) - B * np.sin(f)) / spacing
    v = (A * np.sin(f) + B * np.cos(f)) / spacing
    u, v = u - u.min() + rng.random(), v - v.min() + rng.random()
    nodes = rng.standard_normal((int(u.max()) + 2, int(v.max()) + 2))
    iu, iv = u.astype(np.intp), v.astype(np.intp)
    fu, fv = u - iu, v - iv
    fu, fv = fu * fu * (3.0 - 2.0 * fu), fv * fv * (3.0 - 2.0 * fv)
    return ((nodes[iu, iv] * (1.0 - fv) + nodes[iu, iv + 1] * fv) * (1.0 - fu)
            + (nodes[iu + 1, iv] * (1.0 - fv) + nodes[iu + 1, iv + 1] * fv) * fu) * 1.6


def _threshold_tuft(with_, brush, path, W, sharp, ss):
    """Н-2: порог касания гребней (N, nr) круглой кисти, поставленной на месте. Пучок в сечении — круг волосков: по ходу
    отпечатка (`путь._отпечаток_пучка`, нажим на нём тот, что дала рука) идут не одни и те же волоски, а ряды пучка.
    Волосок касается, если достаёт до холста на своём расстоянии ρ от оси пучка в круге (порог `_А_НАЖИМ`·ρ^1,15, как у
    гребня поперёк), и если он в пучке: за круг раскрытого пучка выходят только кончики — у каждого своя длина.
    Длины — поле по кругу (прядь — пятно, волосок — зерно), а не по гребню на весь отпечаток: у хода прядь одна и та же
    по всей длине, у тычка прямой полосой она легла бы гранью и щелью. Случай — свой поток от сида: прежние не сдвигаются."""
    from scipy.special import ndtr
    rng = np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=tuple(ss.spawn_key) + (0x4E2,)))
    response = _pointed(W, sharp)
    A = _A_PRESSURE * (1.0 - H._ResponseWithTRAYa[5] * response)
    pressure = float(path['p'].max())
    reaches = min(max((pressure - 0.04) / A, 0.0) ** (1.0 / 1.15), 1.0)
    L = max(float(path['L']), 1e-6) / reserve_fringe(W)   # длина круга пучка (`путь._отпечаток_пучка`: путь с запасом на бахрому)
    along = path['s'] - 0.5 * float(path['L'])                                         # px от середины отпечатка
    d = along / (0.5 * L) * reaches                                     # в радиусах пучка
    ρ = np.sqrt(d[:, None] ** 2 + with_.distance[None, :] ** 2)
    spacing = max(float(np.median(2.0 * with_.half_width)), 1.0)
    width_strands = max(5.0 * H._MM, W * (0.30 - 0.12 * brush.scraps))
    by_sweep = along * (0.5 * W) / (0.5 * L) * max(reaches, 1e-3)    # ряды пучка — в px его сечения, как гребни поперёк
    strand = ndtr(_noise2(rng, by_sweep, with_.center, _JAB_TUFT[0] * width_strands))
    hair = ndtr(_noise2(rng, by_sweep, with_.center, _JAB_TUFT[1] * spacing))
    strength = brush.scraps
    threshold = A * ρ ** 1.15 + brush.spread_lengths * (_JAB_TUFT[4] * (strand ** 1.2) * (0.25 + 0.75 * strength) + 0.14 * hair ** 1.5)
    threshold = threshold - threshold.min() + 0.04
    # где кончается пучок: у каждого волоска свой кончик — бахрома. Гребень касается, если внутренний его край в пучке;
    # выборки профиля за кругом краски не кладут (`маска`): иначе крайний гребень клал прямую грань длиной √(R·ширина гребня)
    tip = _noise2(rng, by_sweep, with_.u, spacing)
    edge_m = 1.0 + _JAB_TUFT[2] * tip                         # (N, nm)
    r_vn = clamp((np.abs(with_.center) - with_.half_width) / (0.5 * W), 0.0, None)
    ρ_vn = np.sqrt(d[:, None] ** 2 + r_vn[None, :] ** 2)
    edge_g = np.stack([edge_m[:, with_.ridge == q].min(axis=1) for q in range(with_.nr)], axis=1)
    threshold = threshold + 4.0 * clamp((ρ_vn - edge_g) / 0.04, 0.0, 1.0)
    ρ_m = np.sqrt(d[:, None] ** 2 + (with_.u / (0.5 * W))[None, :] ** 2)
    # борозда волоска — чёрточка: по ходу отпечатка идут разные волоски, у каждого кончик давит по-своему (N, nv)
    leading = (with_.center[1:] + with_.center[:-1]) * 0.5
    dashes = clamp(0.5 + _noise2(rng, by_sweep, leading, _JAB_TUFT[3] * spacing), 0.0, 1.0)
    # ноша волосков: у хода прядь несёт одно и то же весь ход (`пряди._состав`, `нагр`), у тычка ряды пучка — разные
    # волоски: ноша идёт полем по кругу с тем же разбросом (прядь, волосок), иначе гребни ложатся полосами через пятно
    carry = np.exp(0.30 * strength * _noise2(rng, by_sweep, with_.u, _JAB_TUFT[0] * width_strands) + 0.14 * _noise2(rng, by_sweep, with_.u, spacing))
    return threshold, _Tuft(clamp((edge_m - ρ_m) / 0.03 + 0.5, 0.0, 1.0), dashes, carry / carry.mean())


class _WithoutLayer:
    """К-14б: слой для сухого счёта линий (`линии_хода`): сырого нет, холст без краёв."""
    _wet = False


def lines_sweep(points, brush, seed, load=None, touch=None, hand=False):
    """К-14б: линии, которыми ляжет ход кистью из пучков (`Кисть.просветы` > 0), — по одной на пучок, что коснётся
    холста: [длина касания px, ширина пучка px, нажим пучка сверх его порога касания (средний по линии)]. Тот же счёт
    пути, состава кисти и касания, что у `положить_мазок` с этими точками, кистью (в px холста) и сидом, до первого
    чтения холста: ничего не кладёт и не меняет. Сколько краски хватит пучку, здесь не видно: линия — где он касается.
    У кисти со сплошным ворсом — пустой список. Журнал пишет их в строку хода полем "lines", их читает зеркало
    однообразия (`однообразие.py`)."""
    if getattr(brush, 'просветы', 0.0) <= 0 or brush.shape == 'мастихин':
        return []
    response = []
    lay_stroke(_WithoutLayer, points, [(np.full(3, 0.5), 1.0)], brush, seed, load=load, touch=touch, window=(0, 0, 10 ** 9, 10 ** 9), hand=hand, lines=response)
    return response


def _lines_tuft(lines, with_, touch, reserve_g, ds):
    """К-14б: линии хода по пучкам (`линии_хода`): пучок касается, где касается хоть один его гребень."""
    for j in range(with_.ns):
        g = np.nonzero((with_.strand == j) & (with_.has > 0))[0]
        if not len(g):
            continue
        lies = touch[:, g].max(axis=1) > 0.5
        if lies.sum() * ds < 1.0:
            continue
        lines.append([round(float(lies.sum() * ds), 1), round(float(2.0 * with_.half_width[g].sum()), 1),
                      round(float(reserve_g[:, g].max(axis=1)[lies].mean()), 2)])
    return None


def _imprint_tuft(with_, path, reserve_g, T, ss):
    """К-14б: тычок кистью из пучков (`пряди._пучки`) — каждый пучок кладёт свой отпечаток, а не все разом полосу в длину
    пути. Путь тычка — толщина отпечатка `Т` (`толщина_тычка`) и расход кончиков под нажимом (`путь._отпечаток_пучка`).
    К-14б-2: пучок, что коснулся холста, ложится всей своей толщиной `Т`, как ложится плоская (клок — пятно, а не
    крапина: до К-14б-2 толщина в длину отпечатка не шла); сверх неё — расход кончиков этого пучка: чем больше у него
    нажима сверх порога касания (длинный пучок, середина веера), тем сильнее его согнуло и тем длиннее он лёг. У веера
    в один ряд толщины нет — чёрточка из одного расхода, как была. Место отпечатка в толщину кисти — своё: пучки торчат
    вперёд и назад вразнобой (свой поток случая). Ответ — множитель касания (N, nr)."""
    rng = np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=tuple(ss.spawn_key) + (7714,)))
    s, L = path['s'], path['L']
    weight = np.maximum(with_.has, 1e-9)
    reserve = np.bincount(with_.strand, weights=reserve_g.max(axis=0) * weight, minlength=with_.ns) / np.bincount(with_.strand, weights=weight, minlength=with_.ns)
    share = clamp(reserve / max(float(reserve.max()), 1e-6), 0.0, 1.0)
    consumption = np.maximum((L - T) * share, 1.5)
    length = T + consumption
    place = 0.5 * L + (rng.random(with_.ns) - 0.5) * np.maximum(L - length, 0.0) * (T / max(L, 1e-6) if T > 0 else 0.0)
    # волоски пучка — не одной длины: у гребня расход свой, короче, чем у пучка, и не точно на его месте (отпечаток пучка — не брусок)
    length_g = T + consumption[with_.strand] * (0.55 + 0.45 * rng.random(with_.nr))
    place_g = place[with_.strand] + (rng.random(with_.nr) - 0.5) * (length[with_.strand] - length_g)
    return _window_by(s[:, None], (place_g - 0.5 * length_g)[None, :], (place_g + 0.5 * length_g)[None, :], 1.5)


_NO_HAIR = 1e3     # К-14б: порог касания гребня в просвете между пучками (`пряди._пучки`): нажимом его не достать


def _from_lower(threshold, with_):
    """Пороги касания гребней (N, nr) — от самого низкого на этом шаге пути (он касается при нажиме 0,04). К-14б: в
    просвете между пучками волоса нет — холста там ничто не касается, каким бы ни был нажим."""
    if with_.has is not None:
        threshold = threshold + _NO_HAIR * (1.0 - with_.has)[None, :]
    return threshold - threshold.min(axis=1, keepdims=True) + 0.04


def _touch_ridges(threshold, with_, brush, path, p, p_r, laid_flat, lies, ss):
    """Касание гребней (N, nr) и их запас — нажим гребня сверх порога касания. На кончике кисть касается по порогам
    `порог` и нажиму `p_р`; плашмя (`лежит` — где рука держит её всей шириной) место пряди порога не даёт, а разница
    длин сказывается только у самого холста. К-14б-2: тычок кистью из пучков — отпечатки пучков (`_отпечаток_пучков`)."""
    touch = clamp((reserve_g := p_r[:, None] - threshold) / _MYaGBrushResponseWithT, 0.0, 1.0)
    touch = touch * touch * (3.0 - 2.0 * touch)
    if laid_flat > 0:
        threshold_pl = _from_lower(with_.threshold[None, :] * _LENGTHS_LAID_FLAT, with_)
        touch_pl = clamp((p[:, None] - threshold_pl) / _MYaGBrushResponseWithT, 0.0, 1.0)
        touch = touch + (touch_pl * touch_pl * (3.0 - 2.0 * touch_pl) - touch) * lies[:, None]
        reserve_g = reserve_g + (p[:, None] - threshold_pl - reserve_g) * lies[:, None]
    if with_.has is not None and path['на_месте']:
        touch = touch * _imprint_tuft(with_, path, reserve_g, thickness_jab(brush), ss)
    return touch, reserve_g


class _Tuft:
    """Н-2: круглая кисть, поставленная на месте (`_порог_пучка`), — что её круг пучка меняет в укладке мазка."""

    def __init__(ya, in_tuft, dashes, carry):
        ya.in_tuft, ya._dashes, ya.carry = in_tuft, dashes, carry

    @staticmethod
    def for_(threshold, with_, brush, path, W, sharp, ss):
        """Порог касания и пучок тычка круглой; у хода и у плоской — порог как был и None."""
        return _threshold_tuft(with_, brush, path, W, sharp, ss) if path['на_месте'] and brush.shape == 'круглая' else (threshold, None)

    @staticmethod
    def open(tack, leaving):
        """Пучок, придавленный на месте, раскрыт — пряди к осям не собираются, вбок не уходят, бок не поджат."""
        return np.ones_like(tack), np.zeros_like(leaving), None

    def dashes(ya, furrow):
        """У тычка борозда — чёрточка своего волоска, а не линия через всё пятно."""
        if furrow is None:
            return furrow
        deeper = furrow[0] * ya._dashes
        deeper[:, [0, -1]] = furrow[0][:, [0, -1]]                   # крайние волоски кромку ведут, как было
        return (deeper,) + tuple(furrow[1:])

    @staticmethod
    def even(uneven):
        """Долг-ф: запас гребня вдоль хода (`борозда.неровный_запас`) — свойство пряди, что идёт по холсту весь ход; у тычка
        по отпечатку идут ряды пучка — разные волоски, и их неровная ноша уже полем по кругу (`ноша`). Запас гребня тычку
        не нужен: бедный гребень клал щель через всю точку (узкая круглая 5–6 px — кольцо или «буква» с дыркой до грунта)."""
        return np.ones_like(uneven)

    def mask(ya, mask):
        """За кругом пучка волосков нет; ноша — полем по кругу."""
        return mask * ya.in_tuft * ya.carry


def _leaving_short(leaving, path, p, W):
    """Н-2: короткий ход — кисть поставили и потянули: вбок прядь уводит трение, пока её тащат, а не с первого
    касания (иначе соседние пряди вставали врозь на полпикселя-пиксель, и по ходу в ширину шла щель до грунта,
    3 из 30); от `_КОРОТКИЙ[0]` ширин — как было, бит в бит."""
    if (shorter := _short_sweep(path, W)) > 0:
        leaving = leaving * (1.0 - shorter * np.exp(-np.maximum(path['s'] - path['s'][int(np.argmax(p > 0.02))], 0.0) / W))[:, None]
    return leaving


def _short_sweep(path, W):
    """Н-2: насколько ход короткий (`путь._короткий`): 1 — не длиннее `_КОРОТКИЙ[1]` ширин, 0 — от `_КОРОТКИЙ[0]`
    (там всё как было, бит в бит); тычок — не ход (0)."""
    if path['на_месте']:
        return 0.0
    return float(clamp((H._BrushResponseRResponseTBrushAndY[0] * W - path['L']) / ((H._BrushResponseRResponseTBrushAndY[0] - H._BrushResponseRResponseTBrushAndY[1]) * W), 0.0, 1.0))


def _pointed(W, sharp=1.0):
    """П. 10: насколько кисть — узкая с острым кончиком: 1 до 12 px, к 16 px сходит на нет (шире — как было).
    П. 12: `остр` — свойство кисти `Кисть.остриё`: 0 — щетина в обойме, и узкая лежит плашмя всей шириной."""
    return float(clamp((H._ResponseWithTRAYa[0] - W) / H._ResponseWithTRAYa[1], 0.0, 1.0)) * sharp


def _belly(W, round, p_r=None):
    """П. 10: нажим, с которого след узкой кисти в одну краску идёт во всю ширину. Плоская и круглая до 6 px —
    `_ДРОЖЬ[4]`; пучок круглой кисти толще раскрывается до брюшка раньше (к 9 px — при `_КРУГЛАЯ[0]`), дальше его
    держит обойма: след круглой 9–12 px при нажиме листа выходит названной ширины, а не в две трети.
    С `p_р` — ответ другой: нажим (N,), по которому расходится ворс (у круглой — в счёте раннего брюшка, не выше 0,85)."""
    full = H._DRResponseZh[4] - (H._DRResponseZh[4] - H._BrushRAtGLAYa[0]) * float(clamp((W - H._BrushRAtGLAYa[1]) / H._BrushRAtGLAYa[2], 0.0, 1.0)) * bool(round)
    if p_r is None:
        return full
    return np.maximum(p_r, np.minimum(p_r * H._DRResponseZh[4] / full, 0.85)) if round else p_r


def _carry_belly(with_, W, live):
    """П. 10: круглая узкая кисть в одну краску, раскрытая до брюшка (`_брюшко`: с 6 px, к 9 px — целиком), кладёт
    краем следа бок брюшка, а краска у круглой кисти — в брюшке: крайние гребни несут не `0,55` от середины, как у
    пучка на кончике, а `_КРУГЛАЯ[3]`. Иначе у следа 9–12 px плёнка у краёв, едва запас тронут, тоньше видимой, и тело
    выходит уже названного (круглая 12 — как круглая 9). Ответ — нагрузка гребней (nr,), в среднем 1."""
    share = live * float(clamp((W - H._BrushRAtGLAYa[1]) / H._BrushRAtGLAYa[2], 0.0, 1.0))
    if share <= 0:
        return with_.load
    dome = np.sqrt(clamp(1.0 - with_.distance * with_.distance, 0.0, 1.0))
    a = 0.55 + (H._BrushRAtGLAYa[3] - 0.55) * share
    load = with_.load * (a + (1.0 - a) * dome) / (0.55 + 0.45 * dome)
    return load / np.average(load, weights=with_.half_width)


def _settled(gap, with_, live, p, ds, W, rng):
    """П. 10: щель между прядями узкой кисти в одну краску (`Просвет`, правится здесь: модель узкой кисти).
    Где кисть села каплей, краска держит пряди вместе, щель закрыта; каплю стягивает ходом за `_КАПЛЯ_УЗК[9]` ширины:
    корень следа плоской 9–12 px не расходится двумя зубцами. Дальше ворс собран в остриё и пряди то расходятся, то
    сходятся сами — щель открыта кусками в `_ЩЕЛЬ_УЗК[0..1]` ширины хода и закрыта между ними (свой поток случая);
    плёнкой поперёк открытой щели краска у узкой кисти не встаёт: перемычки `Просвет.затекла` через каждые 5–14 px
    читались «лесенкой» между двумя прядями (круглая 9 px по сухому)."""
    at_landing = live * np.exp(-np.cumsum(ds * (p > 0.03)) / (H._DROP_NARROW[9] * W))
    Shch = H._SLIT_NARROW
    n = gap.open.shape[1]
    z = clamp((_noise(rng, len(p), n, 0.5 * (Shch[0] + Shch[1]) * max(W, Shch[2]) / ds) - Shch[3]) / Shch[4] + 0.5, 0.0, 1.0)
    parted = 1.0 - live * (1.0 - z * z * (3.0 - 2.0 * z))                       # (N, щелей и боков)
    gap.seeped = gap.seeped * (1.0 - live)
    open = gap.open * parted * (1.0 - at_landing)[:, None]
    # края у открытой щели кладут тоньше — тот же счёт, что в `Просвет`, с новой `открыта`
    pr_m, ns, opening_up = with_.strand[with_.ridge], with_.ns, max(H._GROWING_TOGETHER_GAP[2], 1e-6)
    at_p = clamp(1.0 - (gap.right[pr_m] - with_.u) / opening_up, 0.0, 1.0)
    at_l = clamp(1.0 - (with_.u - gap.left[pr_m]) / opening_up, 0.0, 1.0)
    shch_p, shch_l = np.where(pr_m < ns - 1, pr_m, ns), np.where(pr_m > 0, pr_m - 1, ns - 1)
    gap.open = open
    gap.less_thick = clamp(1.0 - open[:, shch_p] * at_p[None, :] - open[:, shch_l] * at_l[None, :], 0.0, 1.0)


def _point(with_, W, p_r, distance, palette_knife, live=False, round=False, sharp=1.0, merged=0.0):
    """Порог касания гребней (N, nr) и множитель ширины следа (N,) — с острым кончиком узкой кисти (п. 10).
    Ворс узкой кисти собран в остриё: пряди одной длины и отходят от холста по гребню от краёв к середине, а не прядью
    разом (кончик травинки 9–12 px выходил уступом); на слабом нажиме след сходится к оси, а не идёт двумя-тремя
    иглами на половине ширины; сужает его сбор ворса, а не уход гребней по одному. Широкая кисть — как было."""
    response = 0.0 if palette_knife else _pointed(W, sharp)
    if response <= 0 and merged <= 0:
        return _A_PRESSURE * distance ** 1.15 + with_.threshold[None, :], 1.0
    sr = np.bincount(with_.strand, weights=with_.threshold, minlength=with_.ns) / np.maximum(np.bincount(with_.strand, minlength=with_.ns), 1)
    if response <= 0:      # п. 16: текучая краска слепила пряди кисти на ребре (`_ось_подъёма`) — короткая прядь не отходит раньше соседей
        return _A_PRESSURE * distance ** 1.15 + (with_.threshold - merged * (sr - sr.min())[with_.strand])[None, :], 1.0
    z = clamp(p_r / H._ResponseWithTRAYa[3], 0.0, 1.0)
    exit = (1.0 - z) ** 2
    if live:
        # кисть в одну краску: остриё мягкое, след ширится с нажимом на всём ходе, а не только у самого схода —
        # ширина травинки идёт за рукой (иначе тело следа — лента по линейке, «проволока»)
        exit = (1.0 - clamp(p_r / _belly(W, round), 0.0, 1.0)) ** H._DRResponseZh[3]
    return (_A_PRESSURE * (1.0 - response * H._ResponseWithTRAYa[5]) * distance ** 1.15 + (with_.threshold - response * H._ResponseWithTRAYa[2] * sr[with_.strand])[None, :],
            1.0 - response * (1.0 - H._ResponseWithTRAYa[4]) * exit)


def _tremor(rng, p, W, ds, thickly, no, sharp=1.0):
    """П. 10: узкую кисть с тонкой краской рука держит на весу, кончик мягкий — нажим вдоль хода то и дело слабеет
    (выше заданного не идёт); с ним идут ширина следа (`_остриё`) и отдача краски. Только кисть в одну краску; густая
    краска держит ворс обоймой, широкая кисть и набор — как было (свой поток случая, прежние не сдвигаются).
    Ответ: насколько кисть живая (0 — нет), нажим и нажим руки, каким он пришёл (по нему меряется вход, `_посадка`)."""
    live = 0.0 if (no or len(p) < 3) else _pointed(W, sharp) * (1.0 - thickly)
    if live <= 0:
        return 0.0, p, p
    noise = _noise(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]), len(p), 1, max(H._DRResponseZh[1] * W, H._DRResponseZh[2]) / ds)[:, 0]
    # капля посадки: полная кисть садится каплей и касается холста через неё — там, где она села, след ложится как под
    # нажимом, какой бы лёгкой ни была рука, и сходит к нажиму руки за ширину-две хода (обратный ход начинается пятном)
    Brush = H._DROP_NARROW
    drop = min(Brush[0], Brush[7] * float(p.max())) * np.exp(-np.cumsum(ds * (p > 0.03)) / max(Brush[1] * W, Brush[2])) * clamp(p / 0.03, 0.0, 1.0)
    # севшая каплей кисть обратно в остриё не собирается, пока рука ещё набирает нажим: след от пятна посадки идёт
    # в тело без перехвата (у кисти 6–9 px капля сходила раньше, чем рука набирала нажим, — «спичка»)
    holds = Brush[8] * np.maximum.accumulate(drop) * (np.arange(len(p)) <= int(np.argmax(p)))
    return live, np.maximum(p * (1.0 - live * H._DRResponseZh[0] * clamp(noise + 0.5, 0.0, 2.0)), live * np.maximum(drop, holds)), p


def _wider_point(W):
    """П. 10: насколько узкая кисть шире острия `_КАПЛЯ_УЗК[10]` px (0 — остриё, 1 — вдвое шире и больше): остриё 3 px
    кладёт как было (по сырому не тонет), у кисти шире краска к концу хода идёт из запаса скупее и плёнка тоньшает."""
    return float(clamp(W / H._DROP_NARROW[10] - 1.0, 0.0, 1.0))


def _saturated(W, charge, full, belly=0.0):
    """П. 10 (v3.17): насколько кончик узкой кисти сыт (nr,): 1, пока в брюшке больше `_КАПЛЯ_УЗК[13]` набранного — краску
    к кончику тянет из брюшка, и короткое касание полной кисти (ход в 2–5 ширин) кладёт её до конца; дальше — по остатку
    в степени `_КАПЛЯ_УЗК[5]`: беднеет длинный ход. (Степень остатка шла с первого пикселя: касание в ширину кисти теряло
    седьмую часть краски, в три ширины — больше трети.) Остриё 3 px (`_шире_острия` = 0) — по остатку, как было.
    3б (v3.31): `брюшком` — насколько круглая кисть раскрыта до брюшка (`_раскрыта`). Она кладёт самим брюшком: часть
    запаса, из которого кормится кончик, уже на холсте, и сытая фаза короче — порог идёт к 1 на долю `_КАПЛЯ_УЗК[14]`:
    след 9–12 px по сухому беднеет к концу длинного хода, а не держит плёнку до кончика (покрытие конца к началу было
    0,73–0,78 при пороге 0,75)."""
    sh = _wider_point(W)
    threshold = H._DROP_NARROW[13] + (1.0 - H._DROP_NARROW[13]) * H._DROP_NARROW[14] * belly
    return clamp(charge / (full * (1.0 - (1.0 - threshold) * sh)), 0.0, 1.0) ** (1.0 + (H._DROP_NARROW[5] - 1.0) * sh)


def _opened(brush, W, live):
    """3б: доля, на которую круглая узкая кисть раскрыта до брюшка (0 — плоская и пучок до `_КРУГЛАЯ[1]` px, 1 — с 9 px;
    та же, что в `_ноша_брюшка`)."""
    return live * float(clamp((W - H._BrushRAtGLAYa[1]) / H._BrushRAtGLAYa[2], 0.0, 1.0)) * (brush.shape == 'круглая')


def _saturated_gives_back(live, W, p):
    """П. 10 (v3.17): сытый кончик отдаёт краску сам — капля смачивает холст, какой бы лёгкой ни была рука, как у щетины
    (корень из нажима); круче за нажимом (`_КАПЛЯ_УЗК[4]`) краска идёт, когда брюшко пустеет и её надо выжимать.
    Ответ — показатель (Nc,): отдача гребня делится на exp(показатель · сытость) (`_сыт`). Иначе лёгкое касание полной
    кисти (нажим 0,45) клало плёнку в 0,56 полной вместо 0,75 — светлые касания копии тускнели."""
    return H._DROP_NARROW[4] * live * _wider_point(W) * np.log(clamp(p, 0.03, None) / 0.8)


def _narrow_paint(live, W, p=None, charge=None, full=None, belly=0.0):
    """П. 10, краска узкой кисти в одну краску (живая > 0; иначе — как было, бит в бит). По тому, что дано:
    p — отдача от нажима: мягкий кончик отдаёт краску с нажимом круче, чем щетина широкой (тон идёт за рукой);
    заряд — сколько своей краски идёт из запаса в каплю: у узкой кисти — по мере запаса (плёнка тоньшает вдоль
    хода постепенно, а не держится до последней четверти запаса);
    ни то ни другое — во сколько раз короче запас: плёнка узкой кисти толще закона ширины (`_ПОЛ_ПЛЁНКИ`), а запас
    острия шире `_КАПЛЯ_УЗК[10]` px в px хода не растёт."""
    if p is not None:
        return np.sqrt(p / 0.8) if live <= 0 else (p / 0.8) ** (0.5 + H._DROP_NARROW[4] * live)
    if charge is not None:
        was = np.minimum(charge * 2.5, 1.0)
        # остаток запаса — против того, с чем гребень начал (`полный`), а не против среднего набора: набор гуляет на ±20 %,
        # и в кубе бедно набранная кисть клала с первого пикселя треть плёнки — след 9–12 px выходил уже названного
        # п. 10 (v3.17): краску к кончику тянет из брюшка, и пока в брюшке больше `_КАПЛЯ_УЗК[13]` набранного, кончик сыт —
        # короткое касание полной кисти (ход в 2–5 ширин) кладёт краску до конца; беднеет длинный ход (куб остатка шёл
        # с первого пикселя: касание в ширину кисти теряло седьмую часть краски, в три ширины — больше трети)
        return was if live <= 0 else was + live * (_saturated(W, charge, full, belly) - was)
    if live <= 0:
        return 1.0
    # запас острия в px хода не растёт с шириной (`_КАПЛЯ_УЗК[10]`): след 9–12 px тоньшает к концу хода, как след 3 px,
    # а не держит плёнку до кончика (по сухому читался наклейкой)
    Wo = min(W, H._DROP_NARROW[10])
    return 1.0 - live * (1.0 - min(Wo / H._FLOOR_FILM, 1.0) ** H._DROP_NARROW[6] * (Wo / W) ** 0.5)


def _strands_paints(paints, layout, with_, Nc, spacing_brush, rng, W):
    """П. 10: несколько красок на узкой кисти с острым концом (след до 12 px). Ворс собран в остриё, и краски лежат на
    нём не полосами по линейке, а прядями: граница двух красок идёт по ворсу и гуляет вдоль хода на `_ТРИ_УЗК[0]` ширины
    (σ; длина волны `_ТРИ_УЗК[1]` ширин, не короче `_ТРИ_УЗК[2]` px) — полоса то шире, то уже, местами сходит на нет, и
    соседние краски встречаются. Гребень, через который идёт граница, несёт обе краски по доле своей ширины.
    Ответ: доли красок на гребнях (Nc, nr, n), в сумме 1; порядок красок поперёк кисти — как у `_раскладка`."""
    n = len(paints)
    shares = np.array([d for _, d in paints])
    order = rng.permutation(n) if layout == 'полосы' else np.arange(n)[::-1]
    T = H._THREE_NARROW
    bounds = np.cumsum(shares[order])[:-1][None, :] + T[0] * _noise(rng, Nc, n - 1, max(T[1] * W, T[2]) / spacing_brush) \
        + T[3] * rng.standard_normal(n - 1)[None, :]
    bounds = np.sort(clamp(bounds, 0.0, 1.0), axis=1)
    edges = np.concatenate([np.zeros((Nc, 1)), bounds, np.ones((Nc, 1))], axis=1)            # (Nc, n + 1)
    left, right = (with_.center - with_.half_width) / W + 0.5, (with_.center + with_.half_width) / W + 0.5
    shared = clamp(np.minimum(right[None, :, None], edges[:, None, 1:]) - np.maximum(left[None, :, None], edges[:, None, :-1]), 0.0, None)
    weight = np.zeros((Nc, with_.nr, n))
    weight[:, :, order] = shared / np.maximum(shared.sum(axis=2, keepdims=True), 1e-9)
    return weight


def _narrow_charge(rng, paints, palette_knife, W, layout, thickly, sharp=1.0):
    """П. 10: свой поток случая узкой кисти (пряди, щель, краски; прежние потоки не сдвигаются) и узкий ли это набор —
    несколько красок на кисти с острым концом (до 12 px): та же узкая кисть, что в одну краску (форма следа, капля,
    запас, щель), краски лежат прядями ворса (`_пряди_красок`). Только тонкая краска (густо = 0): густой набор, и у узкой
    кисти, — как был (п. 2в–2д: его держит `test_узкая_кисть_и_боковые_полосы`), как и набор широкой кисти."""
    return (np.random.default_rng(np.random.SeedSequence(rng.bit_generator.seed_seq.entropy, spawn_key=(41,))),
            len(paints) > 1 and not palette_knife and thickly <= 0.0 and _pointed(W, sharp) >= 1.0 and layout in ('полосы', 'ряд'))


def _strands_narrow(with_, leaving, live, rng, N, W, ds, round):
    """П. 10: ворс узкой кисти собран в остриё — прядь от пряди на весь ход не отходит (шум в 260 px на ходе в сотню —
    постоянный сдвиг до 1,5 px: у следа 9 px выходила щель-рельса на весь ход и нахлёст с другой стороны); пряди
    расходятся и сходятся на длине в 2–4 ширины, на доли пикселя. У круглой — и ноша брюшка (`_ноша_брюшка`, в `с.нагр`).
    Ответ — уход прядей вбок (N, ns)."""
    if live <= 0:
        return leaving
    if round:
        with_.load = _carry_belly(with_, W, live)
    Shch = H._SLIT_NARROW
    return leaving * (1.0 - live) + live * Shch[5] * _noise(rng, N, with_.ns, 0.5 * (Shch[0] + Shch[1]) * max(W, Shch[2]) / ds)


def _share_drop(ks_drops, k, spacing_brush, W, travelled):
    """П. 10, узкий набор: соседние гребни делятся каплей на кончике — краски местами смешиваются (на месте).
    Мешает их ход: севшая кисть несёт краски, как набрала, пряди трутся друг о друга, когда их потащило, — в полную
    силу к `_ТРИ_УЗК[5]` ширинам пути от касания (v3.17: короткое светлое касание в 2–3 краски тускнело с первой
    ширины хода — светлая прядь делилась с тёмной соседкой, ещё не сдвинувшись)."""
    d = (ks_drops[1:] - ks_drops[:-1]) * ((H._THREE_NARROW[4] * spacing_brush / W) * float(clamp(travelled / (H._THREE_NARROW[5] * W), 0.0, 1.0))
                                          * np.minimum(k[:-1], k[1:])[:, None])
    ks_drops[:-1] += d
    ks_drops[1:] -= d


def _ferrule(breaks, charge, capacity, spacing):
    """v3.23: заряд гребней после шага переноса — у полупустой кисти краска запаса расходится по соседям (`зерно.Рвёт.делят`)."""
    return charge if breaks is None else breaks.share(charge, capacity, spacing)


def _in_ferrule(charge, charge0, capacity, width, spacing):
    """Долг-ф: ворс живой краски (`загрузка`) в одной обойме — краска запаса расходится по соседним гребням, как у скупо
    набранной (`_Срыв.делят`, та же доля `_СРЫВ[9]` разницы на px пути): прижатый гребень отдаёт и пустеет, соседний
    неприжатый подаёт ему свою, и кисть пустеет вся, а не серединой (были «рельсы»: края, что не касаются или касаются
    слабее, держали краску, середина пустела — две нити по краям и пустая середина). Ровняется полнота гребня — доля
    его полного запаса (`заряд / заряд0`: полная кисть держит в ворсе поровну на единицу ширины), а не заряд в единицах
    хода; объём краски гребня — заряд · ёмкость · ширина гребня, его сумма по кисти не меняется."""
    if len(charge) < 2:
        return charge
    volume, full = capacity * width, charge0 * capacity * width
    share = charge / np.maximum(charge0, 1e-30)
    flow = min(H._WithRYIn[9] * spacing, 0.33) * (share[:-1] - share[1:]) * np.minimum(full[:-1], full[1:])
    paint = charge * volume
    paint[:-1] -= flow
    paint[1:] += flow
    return np.maximum(paint, 0.0) / np.maximum(volume, 1e-30)


def _dry(brush, load):
    """П. 10, инструмент 2: кисть набрана полупустой — нагрузка 'сухая' (не мастихин)."""
    return (brush.load if load is None else load) == 'сухая' and brush.shape != 'мастихин'


class _Breakaway:
    """П. 17, поле мазка `запас` (в журнале "reserve"): сухой срыв. Кисть набрана скупо — краски в ворсе на `запас` ширин
    кисти хода, и кончик кормится из ворса в меру того, что в нём осталось (у полной кисти кончик сыт, пока в запасе
    есть четверть набранного, а запаса хватает на десяток ширин: на ходе в 3–5 ширин он не кончается). Капля на кончике
    тоньшает вдоль всего хода; гребень кладёт, пока капля выше порога, — порог у каждого волоска свой: бугорок лежащей
    краски и зерна холста ближе, ложбина дальше, на богатой прядке краски больше (`зерно.по_зерну`). Начало хода
    плотное, дальше след рвётся на острова вдоль хода, к концу они реже и уже. Что легло, то из запаса убыло: рваный
    след тратит краску медленнее сплошного, хвост хода длинный."""
    __slots__ = ('ширин',)

    def __init__(self, widths):
        self.widths = float(widths)

    def charge(self, rng, nr):
        """Заряд гребней (nr,): набрано, сколько сказано, — от мазка к мазку кисть целиком не гуляет."""
        z = np.exp(0.10 * rng.standard_normal(nr))
        return H._WithRYIn[0] * z / z.mean()

    def reserve(self, with_, W):
        """На сколько px хода хватает единицы заряда гребня (nr,): от пряди к пряди запас ходит, в сумме по кисти — один."""
        z = np.sqrt(with_.reserve)
        return self.widths * H._WithRYIn[2] * W * z / np.average(z, weights=with_.half_width)

    @staticmethod
    def evener(uneven, half_width):
        """Запас гребней вдоль хода (Nc, nr): краски в кисти набрано, сколько сказано, — гребни богаче и беднее вокруг
        среднего по кисти, вся кисть разом от места к месту не богатеет и не беднеет."""
        return uneven + 1.0 - (uneven * half_width).sum(axis=1, keepdims=True) / half_width.sum()

    @staticmethod
    def share(charge, capacity, spacing):
        """Ворс в одной обойме: краска запаса расходится по соседним гребням (доля `_СРЫВ[9]` разницы на px пути) — кисть
        пустеет вся, а не гребень за гребнем рельсами; сумма заряд·ёмкость не меняется."""
        if len(charge) < 2:
            return charge
        flow = min(H._WithRYIn[9] * spacing, 0.33) * (charge[:-1] - charge[1:]) * np.minimum(capacity[:-1], capacity[1:])
        paint = charge * capacity
        paint[:-1] -= flow
        paint[1:] += flow
        return paint / capacity

    @staticmethod
    def drops(drop, ks_drops, full, spacing):
        """Краска на кончиках тоже общая: с бедной прядки, которая до холста не достаёт, она переходит на соседние
        кончики, что кладут (доля `_СРЫВ[12]` разницы на px пути), и несёт свой цвет. Иначе краска бедных прядок
        остаётся на кисти, а богатые пустеют — след гаснет разом. Ответ: капля и её краска (nr,), (nr, 24)."""
        if len(drop) < 2:
            return drop, ks_drops
        flow = min(H._WithRYIn[12] * spacing, 0.33) * (drop[:-1] / full[:-1] - drop[1:] / full[1:]) * np.minimum(full[:-1], full[1:])
        right, left = np.maximum(flow, 0.0), np.maximum(-flow, 0.0)
        new = drop.copy()
        new[:-1] += left - right
        new[1:] += right - left
        ks = ks_drops * (drop - np.append(right, 0.0) - np.insert(left, 0, 0.0))[:, None]
        ks[1:] += ks_drops[:-1] * right[:, None]
        ks[:-1] += ks_drops[1:] * left[:, None]
        return new, np.where((new > 1e-12)[:, None], ks / np.maximum(new, 1e-12)[:, None], ks_drops)

    @staticmethod
    def feeds(remainder, share_drops):
        """Сколько своей краски идёт из ворса на кончик (nr,): в меру остатка запаса; полный кончик не переливается."""
        return clamp(remainder, 0.0, 1.0) ** H._WithRYIn[1] * clamp(2.2 - 1.2 * share_drops, 0.0, 1.0)


def _breakaway(reserve, no):
    """П. 17: `_Срыв` или None (поля нет; мастихин, вытертая и полупустая кисть — как были)."""
    if reserve is None or no:
        return None
    if not float(reserve) > 0.0:
        raise ValueError('запас: больше нуля — на сколько ширин кисти хватает краски')
    return _Breakaway(reserve)


def _axis_lift(with_, merged):
    """П. 16: место, от которого идёт порог касания гребня у кисти на ребре. У щетины это ось пряди: прядь отходит от
    холста разом, и бок сходящего следа идёт уступами в ширину пряди (6–10 px). Текучая краска (поле `ровно`) заливает
    ворс и слепляет пряди в одну обойму: гребни отходят по одному, по своему месту, — бок клина плавный. `слита` — 0–1:
    доля текучести у хода на ребре; 0 — как было (тот же массив)."""
    return with_.axis_strands[with_.strand] + (with_.center - with_.axis_strands[with_.strand]) * merged if merged > 0 else with_.axis_strands[with_.strand]


def _stood(brush, palette_knife, laid_flat, speed):
    """П. 16: плоская кисть на ребре, которую остановили. Пряди отстают вдоль хода, пока их тащат (трение у каждой
    своё); рука тормозит и встаёт — ворс выпрямляется, кончики сходятся к линии кисти, и кисть отрывается разом:
    конец следа — торец той ширины, что касалась, а не хвост прядей на 20–30 px. Остановлена ли кисть — по скорости
    руки в конце хода (`_ТОРЕЦ[0]`: так кончается только «остановка»; подъём, отрыв, смахивание идут быстрее);
    выпрямляется ворс на всём тормозном пути, а не в последних пикселях: кончик, догоняющий кисть рывком, рвал бы след.
    Плашмя пряди держит обойма (`_ОБОЙМА`) — там как было. Ответ — множитель отставания прядей (N,) или 1.0."""
    if palette_knife or brush.shape != 'плоская' or laid_flat >= 1.0 or speed[-1] >= H._TResponseRETs[0]:
        return 1.0
    z = clamp((H._TResponseRETs[1] - speed) / (H._TResponseRETs[1] - H._TResponseRETs[2]), 0.0, 1.0) * (np.arange(len(speed)) >= int(np.argmax(speed)))
    return 1.0 - H._TResponseRETs[3] * (1.0 - laid_flat) * z * z * (3.0 - 2.0 * z)


def _agrees(p, share, fast):
    """Насколько руку поднимают (N,), 0…1: после наибольшего нажима, когда нажим упал ниже `_КОНЦЫ[5]` набранного
    (тело хода дышит нажимом на десятую — это не подъём); сорванная кисть (`быстро`) не поднята."""
    return (np.arange(len(p)) > int(np.argmax(p))) * clamp((H._BrushResponseNTsY[5] - share) / H._BrushResponseNTsY[6], 0.0, 1.0) * (1.0 - fast)


def _melts(with_, comes_off, lifted, comes_off_hand, reserve, p_r, touch, thickly):
    """П. 18б: хвост хода, с которого рука сходит. Гребень — пучок волосков разной длины. Пока рука снимает нажим
    (`сходит_рука`), они отходят от холста не разом, а по одному, на `_ХВОСТ[0]` нажима от первого до последнего:
    гребень кладёт краску всё уже (касается доля волосков) и всё тоньше (слабо прижатый волосок отдаёт меньше,
    `_ХВОСТ[2]`) — прядь кончается нитью, а не торцом. Самые длинные волоски распрямляются вслед за рукой: их порог
    касания (и добавка крайних волосков пряди, `_сходит`) идёт к нулю вместе с нажимом (`поднята`, степень
    `_ХВОСТ[1]`), и последний держится, пока рука не сняла нажим совсем, — след доходит до конца нажима, а не
    обрывается на пороге. Дальние гребни отходят как раньше: кисть собирается к кончику. Плёнка сходит на нет у
    каждого гребня в своём месте, по его порогу; случайных чисел нет. Вне схода руки — всё как было, бит в бит.
    Ответ: касание гребней (N, nr); сход (N, nr) — доля волосков гребня на холсте, ею сужается его след; сколько
    краски отдаёт волосок против тела (N, nr); где хвост (N, 1)."""
    u, v = lifted[:, None], comes_off_hand[:, None]
    brush_zero = (1.0 - u) ** H._CanvasInResponseWithT[1]
    above = p_r[:, None] - reserve - 0.04                                      # порог гребня сверх самого длинного (у него 0,04): кисть собирается к кончику как раньше
    share_in = clamp((p_r[:, None] - above - (0.04 + comes_off[0]) * brush_zero) / ((_MYaGBrushResponseWithT + H._CanvasInResponseWithT[0] * v) * brush_zero + 1e-9), 0.0, 1.0)
    u = v                                                                 # хвост — весь сход руки
    became = share_in * share_in * (3.0 - 2.0 * share_in)
    touch_canvas = np.where(u > 0, np.maximum(touch, became), touch)
    full = np.maximum(touch_canvas ** (1.0 - 0.75 * thickly), 1e-6)              # так гребень заполнен в теле (`заполнен`)
    # след гребня уже одной выборки профиля не бывает: нить тоньше неё кладёт краски во столько же раз меньше, а не пропадает
    thread = np.full(with_.nr, np.inf); np.minimum.at(thread, with_.ridge, with_.x); thread = clamp((thread + 0.1) / 1.2, 0.0, 1.0)[None, :]
    return touch_canvas, np.minimum(np.where(became > 0, np.maximum(became, thread), 0.0) / full, 1.0), np.where(u > 0, (1.0 - u * (1.0 - share_in ** H._CanvasInResponseWithT[2])) * np.minimum(became / thread, 1.0), 1.0), u > 0


def _comes_off(brush, with_, palette_knife, clean, W, p, share, fast, merged, sharp):
    """П. 18: как прядь уходит с холста, когда руку поднимают. Прядь — пучок: его крайние волоски отходят первыми, и
    след пряди сужается к её оси и сходит на нет, а не обрублен поперёк во всю прядь (лесенка сходящего хода). Последние,
    самые длинные волоски касаются холста, пока рука не сняла нажим совсем: порог и ширина перехода касания идут к нулю
    вместе с нажимом. Кисть ставят так же: прядь касается сначала осью и раскрывается, пока рука набирает нажим.
    Сорванная кисть (`быстро`) и текучая краска, слепившая пряди (`слита`), — как было. Ответ: (добавка к порогу
    (N, nr), пол порога (N,), мягкость (N,)) или None — не щетина плоской кисти, кисть уже `_ОСТРАЯ[0]` px, вытертая; не зовётся
    для мягкой кисти и не кроющей краски (она оплывает сама) — их концы как были."""
    if palette_knife or clean or brush.shape != 'плоская' or W < H._ResponseWithTRAYa[0]:
        return None
    strength = 1.0 - merged
    if strength <= 0:
        return None
    N = len(p)
    agrees = _agrees(p, share, fast) * strength
    left = np.full(with_.ns, np.inf); right = np.full(with_.ns, -np.inf)
    np.minimum.at(left, with_.strand, with_.center - with_.half_width); np.maximum.at(right, with_.strand, with_.center + with_.half_width)
    from_axes = clamp(np.abs(with_.center - with_.axis_strands[with_.strand]) / np.maximum(0.5 * (right - left)[with_.strand], 0.5), 0.0, 1.0)
    z = 1.0 - agrees * (1.0 - clamp(share / H._BrushResponseNTsY[1], 0.0, 1.0))
    charged = int(np.argmax(p >= H._BrushResponseNTsY[4] * float(p.max())))
    enters = clamp(1.0 - p / (H._BrushResponseNTsY[4] * float(p.max())), 0.0, 1.0) * (np.arange(N) <= charged) * strength
    return H._BrushResponseNTsY[0] * from_axes[None, :] * np.maximum(agrees, enters)[:, None], H._BrushResponseNTsY[2] + (1.0 - H._BrushResponseNTsY[2]) * z, H._BrushResponseNTsY[3] + (1.0 - H._BrushResponseNTsY[3]) * z


def _ends(brush, with_, s, p, reserve, reserve_g, lies, W, thickly, laid_flat, speed, share, fast, merged, palette_knife, clean, seed, even=0.0):
    """П. 18: концы следа множителем укладки. Сколько краски кисть отдала и что у неё осталось, считается как раньше
    (тело хода не меняется); здесь — только где у концов хода волосок лежит на холсте: сход прядей (`_сходит`: по
    запасу нажима `зап` гребня на ребре; плашмя пряди держит обойма, и ход сходит как раньше — углами) и пятно
    касания (`_пятно`).
    Ответ: сход прядей по гребням (N, nr) — он сужает след гребня к его оси (гребень сходит остриём, а не гаснет во
    всю ширину); концы пятна (их кладёт `_лёг`); насколько кисть остановлена под нажимом. (None, None, 0.0) — как были."""
    comes_off = _comes_off(brush, with_, palette_knife, clean, W, p, share, fast, merged, None)
    if comes_off is None:
        if merged >= 1.0 and _comes_off(brush, with_, palette_knife, clean, W, p, share, fast, 0.0, None) is not None:
            # Д1, строка 6 (долг 16): текучая краска слепила пряди на ребре — врозь они не сходят, но прижатый ворс
            # лежит пятном касания, как и при `слита` чуть меньше 1: конец — пятно (овал по нажиму гребней), а не обрез пути
            return (np.ones((len(p), with_.nr)),) + _patch(with_, s, p, reserve_g, W, thickly, laid_flat, speed, merged / max(1.0 - laid_flat, 1e-6) if laid_flat < 1.0 else 0.0, even)
        return None, None, 0.0
    was, became = clamp(reserve / _MYaGBrushResponseWithT, 0.0, 1.0), clamp((reserve + 0.04 * (1.0 - comes_off[1][:, None]) - comes_off[0]) / (_MYaGBrushResponseWithT * comes_off[2][:, None]), 0.0, 1.0)
    was, became = was * was * (3.0 - 2.0 * was), became * became * (3.0 - 2.0 * became)
    share_brush = np.where(was > 1e-6, np.minimum(became / np.maximum(was, 1e-6), 1.0), 1.0)
    return (1.0 + (share_brush - 1.0) * (1.0 - laid_flat) * (1.0 - lies[:, None]),) + _patch(with_, s, p, reserve_g, W, thickly, laid_flat, speed, merged / max(1.0 - laid_flat, 1e-6) if laid_flat < 1.0 else 0.0, even)


def _tail(brush, with_, p, p_r, reserve, lies, W, thickly, laid_flat, speed, share, fast, merged, palette_knife, clean, dry, exit_g, touch, holds=None):
    """П. 18б: хвост сходящего хода поверх концов п. 18 (`_тает`). Хвост тянется, пока кисть идёт: вставшую кисть
    (`_ТОРЕЦ`) поднимают на месте, её конец — пятно и торец, как были; сорванная, текучая, узкая с острым кончиком
    (п. 10) и сухая (п. 17: у неё хвост кладёт волосок) — как были. Хвост на ребре идёт долей `1 − плашмя`; у кисти плашмя (п. 18в,
    `держит` — `_держит`) хвост — утоньшением плёнки (`_Плёнка`): гребни лежат до конца нажима. Ответ: сход прядей (N, nr), касание на укладке
    (N, nr), `_Плёнка` — сколько краски отдаёт волосок — или None."""
    comes_off = None if exit_g is None or dry else _comes_off(brush, with_, palette_knife, clean, W, p, share, fast, merged, None)
    strength = 1.0 - _pointed(W, brush.point)
    if comes_off is None or strength <= 0:
        return exit_g, touch, None
    if holds is not None:
        touch = _Film.touch(with_, p, touch, holds)
    goes = (np.arange(len(p)) > int(np.argmax(p))) * clamp((speed - H._TResponseRETs[2]) / (2.0 * H._TResponseRETs[0] - H._TResponseRETs[2]), 0.0, 1.0) * (1.0 - fast) * (1.0 - merged) * (1.0 - lies) * strength * (1.0 - laid_flat)
    touch_canvas, melts, thin, tail = _melts(with_, comes_off, goes * (1.0 - clamp(share / H._BrushResponseNTsY[5], 0.0, 1.0)), goes * clamp((H._CanvasInResponseWithT[3] - share) / H._CanvasInResponseWithT[4], 0.0, 1.0), reserve, p_r, touch, thickly)
    return np.where(tail, melts, exit_g), touch_canvas, _Film(thin, holds, share, thickly)


def _spread(brush, p, torn, lies, belly, exit_response, holds=None):
    """Ширина следа в долях ширины кисти (N,): при обычном нажиме 0,85 след — в ширину кисти; придавленная кисть
    расходится веером шире; на слабом нажиме ворс собирается к оси (сорванная кисть отходит плашмя). П. 18в, `держит`
    (`_держит`): пока рука сходит с кисти плашмя на ходу, обойма держит ворс в ширину — к оси он не собирается и на
    кончик кисть не встаёт."""
    collect = np.maximum(clamp(p / 0.15, 0.0, 1.0), torn)
    if holds is not None:
        collect, lies = np.maximum(collect, holds * H._PLYoNBrushA[9]), np.maximum(lies, holds * H._PLYoNBrushA[9])
    spread = brush.splaying + max(_PLAShMYa - brush.splaying, 0.0) * lies
    return (spread + (1.0 - spread) * (belly / 0.85) ** 0.6 * (1.0 + 0.9 * clamp(p - 0.85, 0.0, 0.15))) * (0.3 + 0.7 * collect * collect * (3.0 - 2.0 * collect)) * exit_response


def _holds(brush, p, W, laid_flat, speed, share, torn, even, no):
    """П. 18в: насколько рука сходит с кисти, лежащей плашмя (N,), 0…1, или None — хвоста плашмя нет. Рука снимает
    нажим на ходу (ниже `_ПЛЁНКА[0]` набранного, в полную силу — ещё на `_ПЛЁНКА[1]` ниже): обойма идёт над холстом
    как шла, ворс распрямляется вслед за рукой, и гребни остаются на холсте всей шириной, пока нажим не снят совсем, —
    кисть не встаёт на кончик и не сужает след углами (`пряди._посадка`), она кладёт всё более тонкую плёнку
    (`_Плёнка`). Вставшая (`_ТОРЕЦ`), сорванная, текучая (`ровно`), узкая с острым кончиком, а с `нет` — мастихин,
    вытертая, сухая, скупо набранная, мягкая и не кроющая — как были."""
    if no or laid_flat <= 0 or even > 0 or brush.shape != 'плоская' or W < H._ResponseWithTRAYa[0] or _pointed(W, brush.point) >= 1.0:
        return None
    exit = np.arange(len(p)) > int(np.argmax(p))
    # п. 18г: сорванная и смахнутая кисть хвоста плашмя не кладёт вовсе — ворс за рукой распрямиться не успевает. Жест
    # один на весь сход: нажим сорван хоть где-то (`сорвана` — срыв до доли густоты) или рука на сходе ускоряется
    # против хода тела (`_ПЛЁНКА[13]`, целиком — ещё на `[14]`), — кисть отходит, как отходила без хвоста
    body = speed[share >= 0.97]
    acceleration = clamp((float(speed[exit].max(initial=0.0)) / max(float(np.median(body)) if len(body) else 1.0, 1e-6) - H._PLYoNBrushA[13]) / H._PLYoNBrushA[14], 0.0, 1.0)
    strength = (1.0 - float(torn[exit].max(initial=0.0))) * (1.0 - acceleration * acceleration * (3.0 - 2.0 * acceleration))
    goes = exit * clamp((speed - H._TResponseRETs[2]) / (2.0 * H._TResponseRETs[0] - H._TResponseRETs[2]), 0.0, 1.0) * strength * (1.0 - _pointed(W, brush.point)) * laid_flat
    z = clamp((H._PLYoNBrushA[0] - share) / H._PLYoNBrushA[1], 0.0, 1.0)
    return goes * z * z * (3.0 - 2.0 * z) if goes.any() else None


class _Film:
    """П. 18в: хвост кисти плашмя — утоньшением плёнки, а не сужением пряди. Пока рука сходит (`_держит`), гребень лежит
    на холсте, но прижат всё слабее и краски отдаёт всё меньше. П. 18д: плёнка сначала тоньшает и светлеет, до грунта
    сходит только у конца и там, где на волоске краски нет (`тоньше`): толщина идёт к нулю плавно, без порога. Неровность —
    от запаса гребня (`борозда.неровный_запас`) и рельефа под кистью; по зерну холста её рвёт ещё и `зерно.по_зерну`.
    Гребень к оси не стягивается. Случайных чисел здесь нет. `ребро` — что отдаёт волосок в хвосте на ребре (`_тает`) или None."""
    __slots__ = ('ребро', 'держит', 'нажим', 'степень', 'мягко')

    def __init__(self, edge, holds, share, thickly):
        self.edge, self.holds = edge, holds
        self.pressure = None if holds is None else clamp(share / H._PLYoNBrushA[0], 0.0, 1.0)       # нажим в долях того, с которого рука сходит
        self.power = H._PLYoNBrushA[2] + (H._PLYoNBrushA[17] - H._PLYoNBrushA[2]) * _Film.thick(thickly)   # слабо прижатый волосок оставляет мазню одной толщины, сколько бы краски ни нёс: густая плёнка тоньшает против своего тела круче
        self.soft = H._PLYoNBrushA[5] + (H._PLYoNBrushA[18] - H._PLYoNBrushA[5]) * _Film.thick(thickly)       # густая плёнка до грунта почти не рвётся: в бедном месте она тоньше и светлее

    @staticmethod
    def thick(thickly):
        """Насколько краска держит ворс обоймой и в хвосте (0…1): в полную меру — пастозная, до густоты `_ПЛЁНКА[12]`
        (нагрузка около 0,6) — нет: её пряди собираются к своим осям, а плёнка тоньшает только у конца нажима."""
        return float(clamp((thickly - H._PLYoNBrushA[12]) / (1.0 - H._PLYoNBrushA[12]), 0.0, 1.0))

    @staticmethod
    def touch(with_, p, touch, holds):
        """Касание гребней на укладке (N, nr): пока рука сходит, гребень лежит, как лежал плашмя, — отходит по своей
        длине (`_ДЛИНЫ_ПЛАШМЯ`), когда нажим снят почти совсем."""
        threshold = with_.threshold * _LENGTHS_LAID_FLAT
        z = clamp((p[:, None] - (threshold - threshold.min())[None, :] * H._PLYoNBrushA[6] - H._PLYoNBrushA[7]) / H._PLYoNBrushA[8], 0.0, 1.0)
        return touch + (np.maximum(z * z * (3.0 - 2.0 * z), touch) - touch) * holds[:, None]

    @staticmethod
    def strands(tack, holds):
        """Сбор прядей к своим осям (N, ns): густая краска держит пряди вместе — пока рука сходит, прядь в язык не
        стягивается (`держит` — с долей густоты: тонкая краска пряди не держит, они собираются, как раньше)."""
        return 1.0 - (1.0 - tack) * (1.0 - H._PLYoNBrushA[10] * holds)[:, None]

    def less_thick(self, with_, uneven, thickness, i0, i1, f0, f1, spacing_brush, under):
        """Множитель толщины плёнки по выборкам профиля (N, nm) или 1.0. П. 18д: толщина = нажим^`степень` · z^`_ПЛЁНКА[11]`,
        z = (запас волоска − уровень) / `мягко` + 1 в пределах 0…1. Уровень растёт со снятым нажимом — `_ПЛЁНКА[4]`·снято^`[3]` — и с
        путём, пройденным на сходящем нажиме (в полную силу — за `[19]` ширин кисти: кончик ворса отдаёт свою краску не разом;
        быстро снятая кисть плёнку порвать не успевает). Пока уровень ниже запаса на `мягко`, плёнка цела; дальше она
        тоньшает (степень `[11]`: широкая полоса тонкой светлой плёнки) и только при z = 0 сходит до грунта — кромка мягкая.
        Запас — по месту волоска между осями соседних гребней, мелкая неровность (против среднего на ширину кисти вдоль
        хода) весит в `[15]` раз больше: пятна идут и поперёк хода, и вдоль. `под(ж)` — рельеф лежащей краски под выборкой (в
        толщинах плёнки) или None (зерно холста и рельеф по волоску у сухой кисти и с полем "ridges" считает
        `зерно.по_зерну`), вес `[16]` растёт со снятым нажимом."""
        g, W = with_.ridge, with_.W
        m = 1.0 if self.edge is None else self.edge[:, g]
        if self.holds is None:
            return m
        zh = np.nonzero(self.holds > 0)[0]
        if not len(zh):
            return m
        v, t = self.holds[zh, None], self.pressure[zh, None]
        removed = 1.0 - t
        q = uneven * thickness
        n = max(int(round(W / spacing_brush)), 2)
        sum = np.cumsum(surround(q, ((n // 2 + 1, n - n // 2 - 1), (0, 0)), edge=True), axis=0)
        q = q + H._PLYoNBrushA[15] * (q - (sum[n:] - sum[:-n]) / n)                        # запас гребня в этом месте хода (Nc, nr)
        q = q[i0[zh]] * f0[zh] + q[i1[zh]] * f1[zh]
        # комок краски лежит на волосках гребня и его соседа: между осями гребней запас — по месту волоска
        j = clamp(np.searchsorted(with_.center, with_.u) - 1, 0, with_.nr - 2)
        w = clamp((with_.u - with_.center[j]) / np.maximum(with_.center[j + 1] - with_.center[j], 1e-6), 0.0, 1.0)
        q = q[:, j] * (1.0 - w) + q[:, j + 1] * w
        if under is not None:
            q = q + H._PLYoNBrushA[16] * removed * under(zh)
        path = (i0[zh] + f1[zh].reshape(-1)) * spacing_brush
        z = clamp((q - H._PLYoNBrushA[4] * removed ** H._PLYoNBrushA[3] * clamp((path - path[0]) / (W * H._PLYoNBrushA[19]), 0.0, 1.0)[:, None]) / self.soft + 1.0, 0.0, 1.0)
        thin = np.ones((len(self.holds), len(g)))
        thin[zh] = 1.0 - v * (1.0 - t ** self.power * (z * z * z if H._PLYoNBrushA[11] == 3.0 else z ** H._PLYoNBrushA[11]))   # куб — умножением: степень с дробным показателем втрое дольше
        return m * thin


def _reached(stood, laid_flat, speed):
    """П. 18: кисть плашмя, которую остановили под нажимом, доходит до последней точки пути. Пока кисть тащат, ворс
    согнут назад и кончики отстают от обоймы; рука тормозит и давит — согнутый ворс ложится на холст всей длиной от
    кончика до обоймы, и след кончается там, где встала кисть, а не там, куда дотянулись отставшие кончики.
    Поднятая и сорванная кисть ворса не кладёт (`встала` = 0) — её конец как был; на ребре то же делает `_встал`.
    Ответ — множитель отставания прядей (N,) или 1.0."""
    if stood <= 0 or laid_flat <= 0:
        return 1.0
    z = clamp((H._TResponseRETs[1] - speed) / (H._TResponseRETs[1] - H._TResponseRETs[2]), 0.0, 1.0) * (np.arange(len(speed)) >= int(np.argmax(speed)))
    return 1.0 - H._TResponseRETs[3] * stood * laid_flat * z * z * (3.0 - 2.0 * z)


def _rank(a):
    """Место числа в своём ряду от −1 (наибольшее) до 1 (наименьшее); ряд из равных чисел — нули."""
    if len(a) < 2 or float(np.ptp(a)) <= 0:
        return np.zeros(len(a))
    return 1.0 - 2.0 * np.argsort(np.argsort(a, kind='stable'), kind='stable') / (len(a) - 1.0)


def _strands_patch(with_, thickly, laid_flat, merged, even=0.0):
    """П. 18а: чем пряди в пятне касания разнятся. Случайных чисел здесь нет — всё из того, чем кисть уже набрана:
    прядь с меньшим порогом касания длиннее и ложится дальше, гребень внутри пряди — по своему порогу; у полупустой
    кисти гребень ещё и лежит по своему отставанию в пряди (на посадке и на остановке — в разные стороны). Гружёную
    кисть плашмя краска заливает, и кромка ровная; на ребре кончики прядей краску держат врозь, и в кромке посадки
    пряди читаются (`_ПЯТНО[15]`); текучая краска (`слита`) пряди слепляет. Ответ: множители избытка нажима гребня — посадка (nr,), остановка (nr,); у остановки гружёной кисти — сила, с которой пряди лежат врозь (`_ПЯТНО[16]`), и ранг пряди (nr,)."""
    sr = np.bincount(with_.strand, weights=with_.threshold, minlength=with_.ns) / np.maximum(np.bincount(with_.strand, minlength=with_.ns), 1)
    strand, ridge = _rank(sr)[with_.strand], 0.5 * _rank(with_.threshold - sr[with_.strand])
    own = _rank(with_.lag_gr)
    dry = 1.0 - H._PYaTNResponse[6] * thickly
    settled = 1.0 - H._PYaTNResponse[6] * thickly * (1.0 - H._PYaTNResponse[15] * (1.0 - laid_flat * laid_flat) * (1.0 - merged) ** 2)
    # Д1, строка 5: и остановленная гружёная кисть держит пряди врозь — кромка остановки идёт по прядям, а не ровным
    # овалом, который растр режет равными ступенями: прядь длиннее ложится дальше овала, короче — ближе (на сколько —
    # `врозь` на ранг пряди; середину отсчёта ставит `_пятно`: площадь пятна та же); текучая краска (`ровно`) пряди слепляет и плашмя
    apart = H._PYaTNResponse[4] * H._PYaTNResponse[6] * thickly * H._PYaTNResponse[16] * (1.0 - even) ** 2
    return (1.0 + H._PYaTNResponse[4] * (settled * strand + dry * ridge)) * (1.0 + H._PYaTNResponse[5] * dry * own), (1.0 + H._PYaTNResponse[4] * dry * (strand + ridge)) * (1.0 - H._PYaTNResponse[5] * dry * own), (apart, strand)


def _settled(s, ia, window, reserve_g, where, lags):
    """П. 18а: откуда меряется пятно посадки, когда рука набирает нажим по ходу. Гребни касаются холста по одному,
    каждый по своему порогу, и позже, чем легла бы кромка пятна от первого коснувшегося волоска: начало следа выходило
    веером прядей, а текучая краска затягивала веер в прямой срез. Кисть села там, где её ворс уже лежит пятном:
    кромка пятна проходит по касанию того гребня, что коснулся позже всех. При быстром входе все касаются сразу, и
    место прежнее. Ответ — место на пути, px."""
    oblique = reserve_g[window][:, where] > 0
    later = (s[window][np.argmax(oblique, axis=0)] - s[ia] - lags[where])[oblique.any(axis=0)]
    return float(s[ia]) + (max(float(later.max()), 0.0) if len(later) else 0.0)


def _fan(with_, s, p, reserve_g, ia, window, pm, strength, strand):
    """Д1б, строка 3: посадка гружёной кисти ребром — веером по нажиму руки. Кисть, которую ставят, касается
    холста сперва серединой: ком краски на кончиках держит её горкой, и в ширину ворс расходится по мере того, как рука
    дожимает: от первого касания до 85 % набранного нажима — на долю полуширины `(доля пути нажима)^_ПЯТНО[17]`;
    легший ворс лежит.
    Гребень ложится, когда раскрытие дошло до его места на кисти; длинная прядь (меньший порог) — раньше, короткая —
    позже (`_ПЯТНО[18]` на ранг пряди): кромка посадки — веер прядей. Длина посадки — путь, на котором рука набирает
    нажим: удар кладёт кисть разом, медленный набор — длинным языком. Ответ: на сколько px отстаёт гребень (nr,)
    от первой точки касания."""
    r = p[window] / max(pm, 1e-6)
    sh = np.maximum.accumulate(clamp((r - r[ia]) / max(0.85 - r[ia], 1e-6), 0.0, 1.0) ** H._PYaTNResponse[17])
    needed = clamp(np.abs(with_.center / with_.half_width) * strength * (1.0 - H._PYaTNResponse[18] * strand), 0.0, 1.0)
    i = np.minimum(np.searchsorted(sh, needed, side='left'), len(sh) - 1)
    oblique = reserve_g[window] > 0                                       # и не раньше, чем гребень коснулся по своему порогу
    itself = np.where(oblique.any(axis=0), s[window][np.argmax(oblique, axis=0)], s[window][-1])
    return np.maximum(np.maximum(s[window][i], itself) - s[ia], 0.0)


def _patch(with_, s, p, reserve_g, W, thickly, laid_flat, speed, merged, even=0.0):
    """П. 18: концы следа — пятно касания согнутого ворса, а не обрез пути. Прижатый волосок лежит на холсте не точкой:
    чем больше у гребня нажима сверх порога касания, тем длиннее он лёг вдоль хода (длина — по избытку нажима).
    У середины касания избыток наибольший, к краям сходит на нет (на ребре — купол порогов, плашмя — купол длин ворса
    у углов обоймы): пятно кисти, которую поставили и которую остановили под нажимом, — овал, и концы следа круглые.
    Середина пятна доходит до первой и последней точки пути, края отстают. Гружёную кисть краска заливает, и пятно
    сплошное; у полупустой жёсткой каждая прядь лежит по-своему — край посадки рваный, пальцами прядей
    (`_пряди_пятна`). Поднятая кисть (нажим у конца снят) пятна не оставляет: её конец — сход прядей. Ответ: концы —
    у каждого (0 — посадка, 1 — остановка; место на пути; на сколько px отстаёт кромка у гребня (nr,) и у места на
    кисти (nm,)) — на холст их кладёт `_лёг`; насколько кисть остановлена под нажимом (0…1)."""
    N, nr = reserve_g.shape
    has = np.nonzero((p > 0.02) & (reserve_g > 0).any(axis=1))[0]   # концы следа — где ворс коснулся и где отошёл: рука набирает нажим не с первой точки пути
    if not len(has):
        return [], 0.0
    ia, ib = int(has[0]), int(has[-1])
    sweep = float(s[ib] - s[ia])
    ends, stood = [], 0.0
    if sweep < 2.0:
        return ends, stood
    t = with_.center / with_.half_width
    strands = _strands_patch(with_, thickly, laid_flat, merged, even)
    charged = max(float(s[int(np.argmax(p >= 0.85 * float(p.max())))] - s[ia]), 0.0)   # п. 18а: рука набирает нажим по ходу — кисть садится на всей этой длине
    for end, window in ((0, s <= s[ia] + min(max(0.5 * W, charged), sweep / 3.0)), (1, s >= s[ib] - min(H._PYaTNResponse[3] * W, sweep / 4.0))):
        pm = float(p[window].max())
        e = reserve_g[window].max(axis=0) - (H._PYaTNResponse[2] * laid_flat + H._PYaTNResponse[12] * thickly * (1.0 - laid_flat)) * pm * t * t   # на ребре: ком краски на кончиках гружёной кисти — горкой к середине, углы касаются позже
        where = np.nonzero(e > 0)[0]
        if not len(where):
            continue
        half_width = 0.5 * float((with_.center + with_.half_width)[where].max() - (with_.center - with_.half_width)[where].min())
        # под нажимом ли кисть у этого конца: поднятая рукой пятна не оставляет, сорванная — оставляет
        if end == 0:
            # удар: рука набрала нажим на первых трёх пикселях — кисть легла разом, торец плоский (плашка, п. 2д);
            # кисть, которую ставят (вход от `_ПЯТНО[9]` px), садится пятном
            entry = float(s[int(np.argmax(p >= 0.85 * pm))] - s[ia])
            strength = float(clamp((entry - H._PYaTNResponse[8]) / (H._PYaTNResponse[9] - H._PYaTNResponse[8]), 0.0, 1.0))
        else:
            # пятно оставляет кисть, которую остановили под нажимом («остановка»: скорость руки в конце ниже `_ТОРЕЦ[0]`);
            # сорванная кончает обрывом, поднятая и остановленная на излёте нажима (клин, п. 16) — сходом прядей и торцом
            z = float(clamp((pm / max(float(p.max()), 1e-6) - H._PYaTNResponse[10]) / H._PYaTNResponse[11], 0.0, 1.0))
            strength = z * z * (3.0 - 2.0 * z) * (speed[-1] < H._TResponseRETs[0])
        z = float(clamp((pm - H._PYaTNResponse[13]) / H._PYaTNResponse[14], 0.0, 1.0))   # лёгкая рука ворса не гнёт: кисть касается кончиками, торец как был (п. 2а)
        if end == 1: stood = strength * z * z * (3.0 - 2.0 * z)
        dl = min((H._PYaTNResponse[0] - (H._PYaTNResponse[0] - H._PYaTNResponse[1]) * laid_flat) * half_width, 0.3 * sweep) * strength * z * z * (3.0 - 2.0 * z)
        if dl < 0.5:
            continue
        e = clamp(e * strands[end], 0.0, None)
        oval = lambda c=None: dl * (1.0 - clamp(e * (1.0 if c is None else 1.0 + strands[2][0] * (strands[2][1] - c)) / max(float(e[where].max()), 1e-6), 0.0, 1.0) ** H._PYaTNResponse[7])
        lags = even = oval()
        if end == 0 and thickly * (1.0 - laid_flat) > 0.0:
            lags = even = np.maximum(lags, _fan(with_, s, p, reserve_g, ia, window, pm, thickly * (1.0 - laid_flat), strands[2][1]))
        if end == 1 and strands[2][0] > 0.0:
            # Д1, строка 5: пряди врозь вокруг такой середины ранга, при которой площадь пятна та же, что у ровного овала
            # (кромка в среднем там же: краски в конце хода столько же) — делением пополам, без случайных чисел
            lo, hi = 0.0, 1.0
            for _ in range(24):
                c = 0.5 * (lo + hi)
                lo, hi = (c, hi) if float(((oval(c) - even) * with_.half_width).sum()) < 0.0 else (lo, c)
            lags = oval(0.5 * (lo + hi))
        # гружёную кисть краска заливает между гребнями: кромка пятна идёт по месту на кисти, а не ступенями гребней
        by_place = lambda response: response[with_.ridge] + thickly * (np.interp(with_.u, with_.center, response) - response[with_.ridge])
        ends.append((end, _settled(s, ia, window, reserve_g, where, lags) if end == 0 else float(s[ib]), lags, by_place(lags)))
        if lags is not even:                                                        # Д1, строка 5: ровный овал той же остановки — по нему меряется краска наплыва (`_вал_конца`): пряди врозь двигают кромку, а краски у пряди столько же
            ends[-1] += (even, by_place(even))
    return ends, stood


def _patch(mask, out, patch_m, patch_m0, s, W):
    """П. 18: пятно касания у концов (`_лёг`) — множителем укладки (N, nm). Д1, строка 5: если кромка остановки идёт
    по прядям (`пятно_м0` — то же пятно ровным овалом), краски у пряди столько же, сколько при овале: прядь, вставшая
    раньше, кладёт своё на более коротком конце гуще, ушедшая дальше — тоньше (в окне конца хода)."""
    if patch_m is None:
        return mask
    diff = np.nonzero((patch_m != patch_m0).any(axis=1))[0] if patch_m0 is not None else []
    if len(diff):
        ok = s >= s[diff[0]] - H._PYaTNResponse[3] * W
        was_by, became = (out[ok] * mask[ok] * patch_m0[ok]).sum(axis=0), (out[ok] * mask[ok] * patch_m[ok]).sum(axis=0)
        mask[ok] *= np.where(became > 0.0, was_by / np.maximum(became, 1e-300), 1.0)[None, :]
    return mask * patch_m


def _by_oval(swell, oval):
    """Д1, строка 5: краска наплыва по местам на кисти (N, nm) — столько, сколько её при ровном овале остановки
    (`овалом`, (N, nm) или None): пряди врозь двигают кромку, а не убавляют наплыв."""
    return swell if oval is None else swell * (oval.sum(axis=0) / np.maximum(swell.sum(axis=0), 1e-300))[None, :] * (swell.sum(axis=0) > 0)


def _along(lag_g, drags, ferrule, limit):
    """Насколько кончик гребня лежит впереди кисти вдоль хода (N, nr), px (у отставшей пряди — меньше нуля); густая
    краска у кисти плашмя держит пряди обоймой — не дальше `предел`."""
    vd = lag_g[None, :] * drags[:, None]
    return vd + (limit * np.tanh(vd / limit) - vd) * ferrule if ferrule > 0 else vd


def _along_brush_hair(lag, tip, drags, ferrule, limit):
    """Насколько кончики лежат впереди кисти вдоль ворса (N, len(отст)), px: отставание пряди (`тащит` — (N,), с
    наклоном кисти) и место кончика; густая краска у кисти плашмя держит пряди обоймой — не дальше `предел`."""
    vd = lag[None, :] * drags[:, None] + tip[None, :]
    if ferrule > 0:
        vd = vd + (limit * np.tanh(vd / limit) - vd) * ferrule
    return vd


def _ahead(brush, with_, path, p, W, laid_flat, thickly, palette_knife, speed):
    """К8б: насколько кончик гребня лежит впереди кисти вдоль хода (N, nr), px, — тем же счётом, что `на_холст` в
    `уложить` (отставание пряди набирается на первой ширине пути от касания), без остановки в конце: посадке нужен
    только вход. Ответ: это и сколько кисть уже тащат (N,), 0…1."""
    touched = path['s'][int(np.argmax(p > 0.02))]
    drags = (1.0 - np.exp(-np.maximum(path['s'] - touched, 0.0) / W)) * _stood(brush, palette_knife, laid_flat, speed)
    ferrule = 0.0 if (palette_knife or brush.shape == 'круглая') else laid_flat * thickly
    along_brush = 1.0 + 2.0 * np.sin(np.radians(clamp(path['наклон'], 0, 85)))
    return _along(with_.lag + (_LAG_GR[0] + _LAG_GR[1] * W) * ferrule * with_.lag_gr, along_brush * drags, ferrule, _ResponseBResponseYMA * W), drags


def _laid(ends, with_, s, lag_g, drags, ferrule, limit):
    """П. 18: пятно касания (`_пятно`) на холсте. Кончики прядей отстают от кисти вдоль хода (`отст_г`, `тащит`), и
    место гребня на холсте — не место кисти на пути: кромка пятна меряется от первой и последней точки пути по
    холсту, туда, где лёг кончик. Ответ: лежит ли гребень (N, nr) и место на кисти (N, nm); (None, None) — концы как были."""
    if ends is None:
        return None, None
    vd = _along(lag_g, drags, ferrule, limit)
    lies, lies_m = np.ones(vd.shape), np.ones((len(s), len(with_.u)))
    for end, s0, lags, by_place in (q[:4] for q in ends):
        for to_whom, lag, back in ((lies, lags, -vd), (lies_m, by_place, -vd[:, with_.ridge])):   # вд — вперёд по ходу, у отставшей пряди меньше нуля
            z = clamp((((s - s0)[:, None] - back) if end == 0 else ((s0 - s)[:, None] + back)) / 1.5 - lag[None, :] / 1.5 + 0.5, 0.0, 1.0)
            to_whom *= z * z * (3.0 - 2.0 * z)
    return lies, lies_m


def _swell(s, p, speed, W, thickly):
    """П. 18: наплыв на остановке. Гружёная кисть гонит перед собой валик краски, выжатой из ворса; когда её
    останавливают (скорость руки в конце хода ниже `_ТОРЕЦ[0]` — так кончается «остановка»), валик остаётся там, где
    кисть встала: на длине `_НАПЛЫВ[1]` ширины у конца краски больше, чем в теле, и лежит она по пятну касания
    (`_пятно`). Валик набран всем ходом, а не кончиком: он не беднеет вместе с каплей. Тонкая краска валика не держит.
    Ответ: сколько толщин полной плёнки ложится сверх следа (N,)."""
    if thickly <= 0 or speed[-1] >= H._TResponseRETs[0]:
        return np.zeros(len(p))
    has = np.nonzero(p > 0.02)[0]
    z = clamp(1.0 - (s[has[-1]] - s) / max(min(H._NAPLYIn[1] * W, 0.35 * float(s[has[-1]] - s[has[0]])), 1e-6), 0.0, 1.0)
    return H._NAPLYIn[0] * thickly * z * z * (3.0 - 2.0 * z) * (p > 0.02)


def _zamknut(rim, window, side):
    """Д1, строка 4: жгут валика — сплошной по наружному краю следа: щель в крайней пряди короче окна (2·`окно` + 1
    шагов пути, `_ВАЛИК[10]` ширины кисти) он перекрывает по её штрихам, а не ныряет в каждую. Кромка «замкнута» вдоль
    хода: самое наружное в окне, потом самое внутреннее в окне; где след сужается надолго (поднятая прядь, скруглённая
    остановка, сход), замкнутая кромка — сама кромка, жгут за след не выходит. `кромка` — номер крайней выборки по
    рядам пути (у ряда без следа — середина кисти), `бок` — 0 или 1."""
    e = rim * (1 if side == 0 else -1)                                                  # меньше — наружу
    for f in (np.min, np.max):
        e = f(np.lib.stride_tricks.sliding_window_view(surround(e, window, edge=True), 2 * window + 1), axis=1)
    return e * (1.0 if side == 0 else -1.0)


def _bead_sides(D, with_, brush, p, spread, Out, Hg, uneven, i0, i1, f0, f1, ds, W, thickly, no):
    """П. 19: валик по бокам. Прижатая гружёная кисть выдавливает краску из-под ворса к краям; она ложится гребнем у
    кромки следа с обеих сторон: крутой склон наружу, пологий — к телу (п. 19а: вал шириной ~0,12 ширины кисти, вершина у самой кромки). Краска берётся из тела: сколько легло в гребни,
    столько не легло под ворсом (`D` — краска по выборкам профиля, (N, nm); сумма та же). Доля — с густотой и нажимом (`_ВАЛИК[1..2]`: выше 0,6 — там же, где густая кисть кладёт слой тоньше, `слой_от_нажима`; лёгкая рука и сход не выдавливают), у
    пустеющей кисти сходит на нет; тонкая краска (густо = 0) гребня не держит. У каждого бока своя доля — по запасу его
    крайней пряди (`борозда.неровный_запас`); выдавленное ложится не там же, а на длине `_ВАЛИК[3]` ширины хода: гребень
    идёт то выше, то ниже и не по линейке — по кромке следа, какой она легла (поджатая прядь, рваный край, сход).
    Случайных чисел нет. Узкая с острым кончиком, круглая, мастихин, а с `нет` — мягкая, вытертая, сухая, скупо
    набранная, не кроющая и текучая — как были (тот же массив)."""
    In = H._InALAndBrush
    strength = 0.0 if no or brush.shape != 'плоская' else thickly * (1.0 - _pointed(W, brush.point)) * In[0] * float(clamp((100.0 / W) ** 0.5, 1.0, 1.12))   # выдавливает кромка ворса: у узкой кисти её доля больше (п. 19а: ненамного — тело не тоньше 0,7 прежнего)
    if strength <= 0 or with_.nm < 8 or with_.nr < 3:
        return D
    full = clamp(((Out / Hg[None, :]).mean(axis=1) - In[8]) / In[9], 0.0, 1.0)
    presses = clamp((p - In[1]) / In[2], 0.0, 1.2) * (full[i0] * f0[:, 0] + full[i1] * f1[:, 0])
    row = D.sum(axis=1)
    live = (row > 0) & (presses > 0)
    if not live.any():
        return D
    threshold, m = (0.25 * row / np.maximum(np.count_nonzero(D, axis=1), 1))[:, None], with_.nm // 2                                   # кромка следа — крайняя выборка, где краски не меньше четверти средней по ряду; ищется от своего бока до середины
    rim = np.argmax(D[:, :m] > threshold, axis=1), with_.nm - 1 - np.argmax(D[:, :with_.nm - m - 1:-1] > threshold, axis=1)
    spacing = (W / with_.nm) * spread                                                              # px между выборками (N,)
    n = max(int(round(In[3] * W / ds)), 1)
    core = np.hanning(n + 2)[1:-1]; core = core / core.sum()
    body = 0.25 * float(np.median((row / np.maximum(np.count_nonzero(D, axis=1), 1))[live]))               # Д1, строка 4: кромка для замыкания — где краски не меньше четверти средней по телу хода: тающий ряд (скруглённая остановка, сход) наружу её не выносит
    rim_t, full = (np.argmax(D[:, :m] > body, axis=1), with_.nm - 1 - np.argmax(D[:, :with_.nm - m - 1:-1] > body, axis=1)), ((D[:, :m] > body).any(axis=1), (D[:, with_.nm - m:] > body).any(axis=1))
    window_kr = max(int(round(0.5 * In[10] * W / ds)), 1) if In[10] > 0 else None                                   # Д1, строка 4: полуокно замыкания кромки, в шагах пути
    place, steepness, width = max(In[5] * W, 1.0), max(In[6] * W, 0.6), max(In[7] * W, 1.0)
    zh = np.nonzero(live)[0]                                                               # гребень считается только в полосе у кромки: дальше 3 его ширин краски нет
    k = np.arange(-1, min(int(np.ceil((place + 3.0 * width) / (0.5 * W / with_.nm))) + 2, with_.nm))[None, :]   # с запасом на собранный ворс; от хода не зависит
    share, ridges = np.zeros(len(p)), []
    for side, gr in ((0, 0), (1, with_.nr - 1)):
        reserve = uneven[:, gr] / max(float(uneven[:, gr].mean()), 1e-6)
        reserve = clamp(reserve[i0] * f0[:, 0] + reserve[i1] * f1[:, 0], 0.0, 2.0) ** In[4]
        part = 0.5 * strength * presses * reserve * live                                         # доля краски ряда — в этот бок: сколько легло в гребень, столько не легло под ворсом
        share += part
        laid = np.convolve(part * row, core)                                            # выдавленное ложится позади, на длине памяти гребня
        laid, behind_end = laid[:len(p)][zh], float(laid[len(p):].sum())
        laid[-min(len(zh), n):] += behind_end / min(len(zh), n)                             # что кисть не успела оставить, остаётся у конца следа
        # гребень — валик краски вдоль бока: за каждым зубцом рваной кромки он не прыгает, идёт по ней плавно
        e = rim[side] if window_kr is None else _zamknut(np.where(live & full[side], rim_t[side], m if side == 0 else with_.nm - 1 - m), window_kr, side)
        kr = (np.convolve(e * live, core, 'same') / np.maximum(np.convolve(live * 1.0, core, 'same'), 1e-9))[zh]
        jj = (np.floor(kr).astype(np.intp)[:, None] + k) if side == 0 else (np.ceil(kr).astype(np.intp)[:, None] - k)   # выборки от кромки внутрь (ж, K)
        d = ((jj - kr[:, None]) if side == 0 else (kr[:, None] - jj)) * spacing[zh, None] - place
        usable = (jj >= 0) & (jj < with_.nm)
        jj = np.where(usable, jj, 0)
        # за каждым зубцом рваной кромки гребень не прыгает; с `_ВАЛИК[10]` (Д1, строка 4) он ложится и в щель крайней
        # пряди, а крайняя выборка берёт его той долей, какой её полоса лежит внутри кромки
        edge = (d > -place - 0.5 * spacing[zh, None]) * (D[zh[:, None], jj] > 0) if window_kr is None else clamp((d + place) / spacing[zh, None] + 1.0, 0.0, 1.0)
        weight = np.where(d < 0, np.exp(-(d / steepness) ** 2), np.exp(-(d / width) ** 2)) * edge * usable
        ridges.append((jj, weight * (laid / np.maximum(weight.sum(axis=1), 1e-9))[:, None]))
    D *= (1.0 - np.minimum(share, 0.5))[:, None]
    for jj, ridge in ridges:
        np.add.at(D, (zh[:, None], jj), ridge)                                          # вне следа вес — ноль, а место у них общее: складывать, не присваивать
    return D


def _window(a, ri, rj):
    """Среднее по прямоугольнику (2ri+1)×(2rj+1) через накопленные суммы. За краем — ноль."""
    if ri > 0:
        k = 2 * ri + 1
        c = np.cumsum(surround(a, ((ri + 1, ri), (0, 0))), axis=0)
        a = (c[k:] - c[:-k]) / k
    if rj > 0:
        k = 2 * rj + 1
        c = np.cumsum(surround(a, ((0, 0), (rj + 1, rj))), axis=1)
        a = (c[:, k:] - c[:, :-k]) / k
    return a


def _stolb(V, x, y, W):
    """П. 21: во сколько толщин тела (`_тело`, в квадрате конца хода с полем; x, y — где кончилось тело следа) стоит
    самое высокое место свежей плёнки мазка. Порог «есть ли ком» — `_КОМ[9]`."""
    r = int(np.ceil(1.5 * H._BrushResponseM[5] * W))
    p = V[max(int(y) - r, 0):int(y) + r + 2:2, max(int(x) - r, 0):int(x) + r + 2:2]                      # через пиксель: это только порог
    p = p[p > 0.0]
    return float(V.max() / _body(p)) if len(p) else 0.0


def _zone_comma(lump, D, Xl, Yl, u, bh, bw, W, V):
    """П. 21: где кисть, уходя с холста, оставила ком — пиксели прямоугольника мазка (маска): последняя `_КОМ[5]` ширины
    следа (по холсту, от места, где кончилось его тело, и всё, что дальше) и ряды пути, где лёг сгусток срыва (`ком`,
    (N, nm)). Тело хода в зону не входит. Пряди — следы самих волосков (`Xл`, `Yл` — где шёл волосок на каждом шаге,
    `u` — его место поперёк кисти, px): на кривом конце прядь гнётся вместе с бороздой. Ответ: None или (y0, x0, маска,
    прядь, путь) — окно конца хода в прямоугольнике мазка, зона в нём и для каждого пикселя окна — номер пряди (-1 — волосок
    здесь не шёл) и путь вдоль неё, px (ноль — где кончилось тело следа)."""
    paints = D.sum(axis=1) + lump.sum(axis=1)
    if not paints.any():
        return None
    last = int(np.nonzero(paints >= 0.25 * np.median(paints[paints > 0.0]))[0][-1])           # где кончается тело следа: дальше — только усы прядей
    sgustki = lump.any(axis=1)
    if _stolb(V, Xl[last].mean(), Yl[last].mean(), W) <= H._BrushResponseM[9]:
        return None                                                                          # столба у конца нет (кисть поднялась): оседать нечему, счёта нет
    xm, ym = Xl.mean(axis=1), Yl.mean(axis=1)
    to = np.hypot(xm - xm[last], ym - ym[last])                                              # по холсту, не по пути: на смахивании пряди отстают и ложатся короче пути
    rows = (sgustki | (np.arange(len(paints)) > last) | (to <= H._BrushResponseM[5] * W)) & (paints > 0.0)
    iy, ix, r = np.rint(Yl[rows]).astype(np.intp), np.rint(Xl[rows]).astype(np.intp), int(np.ceil(H._BrushResponseM[5] * W))
    y0, y1 = max(min(int(iy.min()) - 2, int(ym[last]) - r), 0), min(max(int(iy.max()) + 3, int(ym[last]) + r + 2), bh)   # окно конца хода: счёт идёт только в нём
    x0, x1 = max(min(int(ix.min()) - 2, int(xm[last]) - r), 0), min(max(int(ix.max()) + 3, int(xm[last]) + r + 2), bw)
    if y1 <= y0 or x1 <= x0:
        return None
    h, w = y1 - y0, x1 - x0
    zone = np.zeros((h, w), dtype=bool)
    zone[clamp(iy - y0, 0, h - 1), clamp(ix - x0, 0, w - 1)] = True
    zone = (_neighbors(_neighbors(zone * 1.0)) > 0.0) | ((np.arange(x0, x1)[None, :] - xm[last]) ** 2 + (np.arange(y0, y1)[:, None] - ym[last]) ** 2 <= (H._BrushResponseM[5] * W) ** 2)   # отсчёт ложится ядром в пиксель с лишним
    # пряди окна: волосок красит пиксели своим номером и своим путём
    own = np.nonzero(rows | (to <= 1.25 * H._BrushResponseM[5] * W))[0]                                # ряды пути у зоны: по ним меряется и толщина пряди рядом с комом
    with_ = int(own[0])                                                                         # путь волоска копится подряд — от первого такого ряда
    through = max(int(max(len(u) - 1, 1) / max(u.max() - u.min(), 1e-6) + 1e-6), 1)   # волоски гуще пикселя — берётся каждый такой: шаг поперёк не шире 1 px, пропуски закрывают соседи
    Xo, Yo, u = Xl[with_:, ::through], Yl[with_:, ::through], u[::through]
    path = np.concatenate([np.zeros((1, Xo.shape[1])), np.cumsum(np.hypot(np.diff(Xo, axis=0), np.diff(Yo, axis=0)), axis=0)])
    back = np.nonzero(to[:last + 1] <= 0.5 * H._BrushResponseM[5] * W)[0][0]
    length = float(to[back])
    forward = 0.0 if length <= 1.0 else ((Xl[last, ::through] - xm[last]) * (xm[last] - xm[back]) + (Yl[last, ::through] - ym[last]) * (ym[last] - ym[back])) / length
    path = path - path[max(last - with_, 0)][None, :] + forward                                    # ноль пути — поперёк хода у конца тела: на прямом ходу путь — это x
    if len(own) < len(Xo):
        Xo, Yo, path = Xo[own - with_], Yo[own - with_], path[own - with_]
    jy, jx = np.rint(Yo).astype(np.intp) - y0, np.rint(Xo).astype(np.intp) - x0
    here = (jy >= 0) & (jy < h) & (jx >= 0) & (jx < w)
    where = jy[here] * w + jx[here]
    S = np.stack([np.bincount(where, weights=q, minlength=h * w) for q in (None, np.broadcast_to(u[None, :], Xo.shape)[here], path[here])], axis=-1).reshape(h, w, 3)   # сколько отсчётов легло в пиксель, сумма их мест поперёк кисти и пути
    holes = S[..., 0] == 0.0                                                                  # между отсчётами: прядь и путь — от соседей
    if holes.any():
        S = np.where(holes[..., None], _neighbors(S), S)
    has = S[..., 0] > 0.0
    divide = np.where(has, S[..., 0], 1.0)
    strand = np.where(has, np.rint(S[..., 1] / divide - u.min()), -1.0).astype(np.intp)
    return y0, x0, zone, strand, S[..., 2] / divide


def _accumulated(a, axis):
    """Накопленная сумма вдоль оси с нулём впереди — для `_сумма_вдоль`."""
    c = np.zeros(a.shape[:axis] + (a.shape[axis] + 1,) + a.shape[axis + 1:])
    np.cumsum(a, axis=axis, out=c[(slice(None),) * axis + (slice(1, None),)])
    return c


def _sum_along(c, k0, k1, axis):
    """Сумма соседей от k0 до k1 (сдвиги, включительно) вдоль оси массива по его накопленной сумме `c` (`_накоплено`). За краем — ноль."""
    n = c.shape[axis] - 1
    i = np.arange(n)
    return np.take(c, clamp(i + k1 + 1, 0, n), axis=axis) - np.take(c, clamp(i + k0, 0, n), axis=axis)


def _mean(a, R):
    """Среднее по квадрату (2R+1)², за краем — ноль. `a` — (h, w)."""
    return _sum_along(_accumulated(_sum_along(_accumulated(a, 0), -R, R, 0), 1), -R, R, 1) / (2 * R + 1) ** 2


def _neighbors(a):
    """Сумма по квадрату 3×3 (за краем — ноль) по двум первым осям: счёт срезами, без накопленных сумм."""
    b = np.zeros((a.shape[0] + 2, a.shape[1] + 2) + a.shape[2:])
    b[1:-1, 1:-1] = a
    b = b[:-2] + b[1:-1] + b[2:]
    return b[:, :-2] + b[:, 1:-1] + b[:, 2:]


def _body(V):
    """Толщина тела хода: медиана плёнки по краске (половина краски лежит тоньше), ком срезан на трёх медианах по
    пикселям — тонкая дымка у кромки и сам ком её не сдвигают."""
    p = V[V > 0.0]
    p = np.sort(np.minimum(p, 3.0 * np.median(p)))
    c = np.cumsum(p)
    return float(p[np.searchsorted(c, 0.5 * c[-1])])


def _osel(V, color, W, zone, one=False):
    """П. 21: ком краски у конца хода оседает. Густая краска столбом не стоит: где свежая плёнка у конца хода (`зона`,
    `_зона_кома`) легла комом (выше `_КОМ[8]` толщин тела: вал остановки — `_вал_конца`, сгусток отрыва и срыва — `Сгус`,
    краска собранных в конце прядей), ком оплывает пологим валом вокруг места, где лёг. Счёт идёт по прядям — следам
    волосков (`_зона_кома`: у каждого пикселя номер пряди и путь вдоль неё; на кривом конце прядь гнётся с бороздой): всё,
    что у кома выше `_КОМ[6]` толщин своей пряди (борозды прядей идут сквозь вал), ложится
    вдоль пряди валом с плавным спадом от места кома — `_КОМ[2]` толщин тела на ширину кисти: гребень там, где стоял ком,
    к телу высота сходит без ступени. Вбок краска расходится только по соседним прядям своей полосы набора (не дальше
    `_КОМ[12]` ширины кисти; чему там места нет — вдвое и вчетверо дальше): чужую полосу ком не красит. Выше `_КОМ[0]`
    толщин тела (по плёнке без борозд — среднее на `_КОМ[3]` ширины кисти; гребень борозды — не выше `_КОМ[10]` этого)
    вал не растёт — большой ком ложится шире; тонкая плёнка кома не держит — слой на ней не толще `_КОМ[1]` её толщин,
    на дымку у кромки (тоньше `_КОМ[7]` тела), на голый грунт и из зоны конца краска не течёт. Чему места не нашлось —
    остаётся в своей пряди. Краски столько же (сумма та же), цвет идёт с краской. Тело меряется у конца хода (окно зоны
    с полем в половину `_КОМ[5]` ширины). Счёт — один проход: высота вала пряди находится делением пополам (`_КОМ[4]`
    раз). Случайных чисел нет. Ответ: V, цвет (те же массивы, если кома нет)."""
    Brush = H._BrushResponseM
    if zone is None:
        return V, color
    V0, color0, (zy, zx, mask, P, Q), field = V, color, zone, int(round(0.5 * Brush[5] * W))
    h, w = mask.shape
    oy, ox = max(zy - field, 0), max(zx - field, 0)                                            # счёт — в окне конца хода с полем: тело меряется рядом с концом, начало длинного хода конца не меняет
    V = V0[oy:zy + h + field, ox:zx + w + field]
    near = V[V > 0.0]
    if not len(near):
        return V0, color0
    body = _body(near)
    v, in_strands = V0[zy:zy + h, zx:zx + w], P >= 0
    mask = mask & in_strands
    lump = mask & (v > Brush[8] * body)
    if not lump.any():                                                                        # ком — что стоит столбом; гребни борозд и валик бока у конца хода — не ком
        return V0, color0
    ts = None if one else color0[zy:zy + h, zx:zx + w]                                         # ход в одну краску цвета не меняет — ком ложится краской своей плёнки
    n, p = int(P.max()) + 1, np.where(in_strands, P, 0)

    def by_strands(a, where=None):
        """Сумма по пикселям каждой пряди."""
        return np.bincount(P[in_strands] if where is None else p[where], weights=a[in_strands] if where is None else a[where], minlength=n)

    r = max(int(round(Brush[3] * W)), 2)
    at_comma = _mean(lump * 1.0, r) > 0.5 / (2 * r + 1) ** 2
    even = (v >= Brush[7] * body) & ~at_comma & in_strands
    count = by_strands(even * 1.0)
    strand = np.where(count > 0, by_strands(v * even) / np.maximum(count, 1), body)              # толщина пряди у конца хода без кома: борозда — прядь тоньше, гребень — толще
    base = (Brush[6] * np.minimum(strand, body))[p]
    above = mask & at_comma & (v > base)                                                       # оседает ком и то, что выше базы вплотную к нему
    film = np.where(above, base, v)
    Sl = v - film                                                                          # ком: что выше базы своей пряди
    smooth = _mean(film, r)
    Me = clamp(np.minimum(Brush[0] * body - smooth - Brush[10] * (film - smooth), Brush[1] * smooth), 0.0, None) * (mask & (v >= Brush[7] * body))   # столько вала плёнка здесь держит
    E, vmestit, where = by_strands(Sl, above), by_strands(Me), by_strands(Sl * Q, above)             # ком пряди: сколько краски, сколько места, где лежит (с весом краски)
    place = Me > 0.0
    pm, qm, me = p[place], Q[place], Me[place]                                               # пиксели, куда ляжет вал: прядь, путь, сколько держит
    if ts is None:
        kr, band = np.zeros((n, 0)), np.zeros(n, dtype=np.intp)
    else:
        kr = np.stack([by_strands(ts[..., k] * Sl, above) for k in range(3)], axis=1)           # краска кома пряди по каналам
        tone = np.stack([np.bincount(pm, weights=ts[..., k][place], minlength=n) for k in range(3)], axis=1) / np.maximum(np.bincount(pm, minlength=n), 1)[:, None]   # цвет плёнки пряди там, куда ляжет вал
        band = np.cumsum(np.concatenate([[0], np.abs(np.diff(tone, axis=0)).max(axis=1) > Brush[11]]))        # полоса набора: соседние пряди одной краски

    def sideways(what, receiver, rb):
        """Краска прядей расходится поровну по прядям своей полосы набора не дальше rb; некуда — остаётся в своей."""
        rb = min(rb, n - 1)                                                                  # у узкой кисти прядей меньше, чем шагов вбок
        how_much, slices = np.zeros(n), [(slice(max(d, 0), n - max(-d, 0)), slice(max(-d, 0), n - max(d, 0))) for d in range(-rb, rb + 1)]
        for a, b in slices:                                                                   # прядь б отдаёт пряди а
            how_much[b] += receiver[a] & (band[a] == band[b])
        result = [q * (how_much == 0.0).reshape((n,) + (1,) * (q.ndim - 1)) for q in what]
        for a, b in slices:
            share = (receiver[a] & (band[a] == band[b])) / np.maximum(how_much[b], 1.0)
            for q, r in zip(what, result):
                r[a] += q[b] * share.reshape((-1,) + (1,) * (q.ndim - 1))
        return result

    Hm, c = Brush[0] * body, Brush[13] * body

    def wedge(A, from_, top):
        """Вал пряди: высота спадает от места кома ровным уклоном, подошва скруглена (излома у тела нет), не выше места (`верх`)."""
        z = np.maximum(A - from_, 0.0)
        return np.minimum(np.where(z < c, z * z / (2.0 * c), z - 0.5 * c), top)

    bank, rest, rb = np.zeros(len(pm)), E, max(int(round(Brush[12] * W)), 1)
    E, goes = np.zeros(n), (rest, where, kr)
    where, kr = np.zeros(n), np.zeros(kr.shape)
    for circle in range(3):                                                                # вбок: сперва на соседние пряди; чему в них места нет — дальше
        e2, g2, brush2 = sideways(goes, vmestit > E, rb << circle)
        E, where, kr = E + e2, where + g2, kr + brush2
        has = (E > 0.0)[pm]
        pe = pm[has]
        from_ = np.abs(qm[has] - (where / np.maximum(E, 1e-300))[pe]) * (Brush[2] * body / W)        # спад вала от места кома
        top = np.maximum(me[has] - Brush[14] * from_, 0.0)                                         # столько держит место: потолок спадает от места кома
        lo, hi = np.zeros(n), np.full(n, Hm + c + (from_.max() if len(from_) else 0.0))
        for _ in range(int(Brush[4])):                                                           # высота вала пряди — делением пополам
            A = 0.5 * (lo + hi)
            few = np.bincount(pe, weights=wedge(A[pe], from_, top), minlength=n) < E
            lo, hi = np.where(few, A, lo), np.where(few, hi, A)
        kl = wedge(hi[pe], from_, top)
        s = np.bincount(pe, weights=kl, minlength=n)
        bank[has] = kl * np.minimum(E / np.maximum(s, 1e-300), 1.0)[pe]
        rest = np.where(E - s > 1e-6 * body, E - s, 0.0)
        if not rest.any() or circle == 2:
            break
        share = rest / np.maximum(E, 1e-300)                                                                       # лишнее уходит дальше вбок со своей долей краски
        goes = (rest, where * share, kr * share[:, None])
        E, where, kr = E - rest, where - goes[1], kr - goes[2]
    laid = np.bincount(pm, weights=bank, minlength=n)
    bank *= (E / np.maximum(laid, 1e-300))[pm]                                                 # ровно объём кома; чему места не нашлось — тем же валом выше
    layer = np.zeros((h, w))
    layer[place] = bank
    empty = ((laid <= 0.0) & (E > 0.0))[p] & above                                             # пряди без места вовсе: ком остаётся где лёг
    if empty.any():
        layer[empty] = (Sl * (E / np.maximum(by_strands(Sl, above), 1e-300))[p])[empty]
    touched = (layer > 0.0) | above
    V0 = V0.copy()
    V0[zy:zy + h, zx:zx + w] = np.where(touched, film + layer, v)
    if ts is not None:
        tsn = (ts * film[..., None] + (kr / np.maximum(E, 1e-300)[:, None])[p] * layer[..., None]) / np.maximum(film + layer, 1e-12)[..., None]
        color0 = color0.copy()
        color0[zy:zy + h, zx:zx + w] = np.where(touched[..., None], tsn, ts)
    return V0, color0


def _bank_end(swell, ends, with_, s, vd, width, ds, W):
    """П. 19: наплыв на остановке читается светом. Краски в наплыве столько же (`_наплыв`), но лежит она не пологой
    горкой на полширины кисти, а валом поперёк хода у кромки остановки: кисть гнала валик перед собой и оставила его
    там, где встала. `наплыв` — краска наплыва по выборкам профиля, как она легла бы горкой, (N, nm). У каждого места на
    кисти вся его краска собирается к кромке пятна касания (`_пятно`: кромка — овал, вал идёт по нему): гребень в
    `_ВАЛ_КОНЦА[0]` ширины от кромки, наружу — круто (`[2]` этой длины), к телу — полого. Объём по каждому месту на кисти тот же."""
    end = [q for q in (ends or []) if q[0] == 1]
    if not end or not swell.any():
        return swell
    _, s0, _, by_place = end[0][:4]
    l = max(H._BANK_END[0] * W, H._BANK_END[1])
    zh = np.nonzero(swell.any(axis=1))[0]                                                    # ряды пути, где наплыв есть
    d = ((s0 - s[zh])[:, None] - vd[zh][:, with_.ridge] - by_place[None, :]) / l - 1.0           # от гребня вала внутрь, в его длинах
    weight = np.exp(-(d / np.where(d < 0, H._BANK_END[2], 1.0)) ** 2) * (swell[zh] > 0)
    bank = np.zeros(swell.shape)
    bank[zh] = weight * (swell.sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-9))[None, :]
    return bank


def _solid(xs, ys, weights, X, Y, D, ii, jj):
    """Д1б, строка 7: кончик гребня кладёт краску на всём пути, который прошёл за шаг. Отсчёт мазка — точка в конце
    шага пути; у тела хода шаг 0,5 px, и точки сливаются. Когда кончики догоняют остановленную кисть (`_дошла`,
    `_встал`), кончик за шаг уходит по холсту на 1–1,6 px, и точки ложились поперечными строками — лесенка у кромок
    торца. Отсчёт, чей кончик ушёл дальше `_СПЛОШЬ[0]` px от места на прошлом шаге (X, Y — места кончиков (N, nm),
    D — краска, ii, jj — свои отсчёты), делится поровну на отсчёты через `_СПЛОШЬ[1]` px назад по пройденному отрезку;
    сам он остаётся на своём месте (len(ii) первых отсчётов — свои, за ними дописанные плёнкой щелей и валом — не трогаются). Новые отсчёты — в конец: где делить нечего,
    поля складываются в прежнем порядке, бит в бит."""
    prev = np.maximum(ii - 1, 0); went = (ii > 0) & (D[prev, jj] > 0.0)                      # откуда кончик пришёл за шаг
    n0, dx, dy = len(ii), np.where(went, by_samples(X, ii, jj) - X[prev, jj], 0.0), np.where(went, by_samples(Y, ii, jj) - Y[prev, jj], 0.0)
    k = np.where(np.hypot(dx, dy) > H._WithPLResponseSh[0], np.ceil(np.hypot(dx, dy) / H._WithPLResponseSh[1]), 1.0)
    if (k <= 1.0).all():
        return xs, ys, weights
    ki = k.astype(np.intp)
    who = np.repeat(np.arange(n0), ki - 1)
    t = (np.arange(len(who)) - np.repeat(np.cumsum(ki - 1) - (ki - 1), ki - 1) + 1.0) / k[who]
    share = np.concatenate([1.0 / k, np.ones(len(xs) - n0)])
    return (np.concatenate([xs, xs[who] - dx[who] * t]), np.concatenate([ys, ys[who] - dy[who] * t]),
            [np.concatenate([w * share, w[who] / k[who]]) for w in weights])


def _empty(why, what, X=None, Y=None):
    """v4, 1б: где мазок кончился без следа — в словарь `почему` (`положить_мазок`); X, Y — места кончиков: прямоугольник
    следа кисти. Ответ — None (мазок без следа)."""
    if why is not None:
        why['что'] = what
        if X is not None:
            why['след'] = (int(np.floor(X.min())), int(np.floor(Y.min())), int(np.ceil(X.max())) + 1, int(np.ceil(Y.max())) + 1)
    return None


def _flow(seed):
    """Потоки случая мазка из его сида, в прежнем порядке: главный; свой (прежние мазки те же); п. 2в — набор красок, углы;
    п. 2г — бок, торец, кромка; п. 2д — посадка кисти, узкая кисть; п. 4 — борозды, срез конца."""
    rng = np.random.default_rng(seed)
    return (rng,) + tuple(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]) for _ in range(5))


def _joints(numbers0, with_, rng_b, thickly, paints, charge, p_brush, spacing_brush, W):
    """Стыки несмешанных густых красок на кисти (вынесено из `положить_мазок` как было): (номера гребней у стыка, щель,
    полуширина щели px), и доля капли, что прижатая кисть отдаёт соседу за шаг переноса (Nc,)."""
    joint0 = np.nonzero(numbers0[:-1] != numbers0[1:])[0]
    slit = clamp(rng_b.random(len(joint0)) * 1.7 - 0.6, 0.0, 1.0) * thickly       # у каждого стыка своя; у трети нет
    floor_slit = 0.7 + 1.1 * rng_b.random(len(joint0))                               # px
    if charge:
        # п. 2г: щель узкую полосу не съедает — она не шире шестой части меньшей из двух полос у стыка
        # (у полосы в 6–7 px, зажатой двумя щелями, от краски оставалась нитка)
        width_p = np.array([2.0 * with_.half_width[numbers0 == q].sum() for q in range(paints)])
        floor_slit = np.minimum(floor_slit, np.maximum(0.17 * np.minimum(width_p[numbers0[joint0]], width_p[numbers0[joint0 + 1]]), 0.7))
    return (joint0, slit, floor_slit), thickly * clamp((p_brush - 0.55) / 0.3, 0.0, 1.0) * (spacing_brush / (1.5 * W))   # доля капли соседу за шаг


def _full_brush(brush, H, Hg, reserve_px, drop0, with_, W, live, sparingly, breaks):
    """Д-1-2: сколько краски в полной кисти — объём на единицу ширины гребня: (в ворсе, на кончике (nr,)). От хода не
    зависит: без случайного набора гребней (`заряд0`), без их неровного запаса и нагрузки (`с.запас`, `с.нагр`) и по
    толщине плёнки самой кисти, а не нагрузки хода. Доли `Загрузка` — доли этого объёма: та же полная кисть — та же
    краска, каким бы ходом её ни клали (долю множили на ёмкость хода с его случайным набором: мягкая 6 мм на ходе
    40 мм держала 7197, на ходе 240 мм — 3701, и краска бралась из ничего). H, Hг — плёнка хода и её гребней, запас_px —
    на сколько px хватает единицы заряда гребня, капля0 — капля, с которой садится свежая кисть (nr,)."""
    Hk = brush.thickness_film(); Hk = Hk if Hk > 0.0 else H
    if sparingly is not None:
        brush_hair = H._WithRYIn[0] * sparingly.widths * H._WithRYIn[2] * W
    elif breaks is not None and breaks.dry:
        brush_hair = _PEREGRAtZ * breaks.charge * brush.reserve * 32.0 * H._WithAtCanvasAYa[7]
    else:
        brush_hair = _PEREGRAtZ * (1.0 if breaks is None else breaks.charge) * brush.reserve * W * (32.0 * H._MM / W) ** 0.5 * _narrow_paint(live, W)
    return Hk * brush_hair, drop0 * (Hk / np.maximum(Hg, 1e-12))


def _with_loading(loading, place, full_brush, drop0, capacity):
    """Д-1: ход начинает с того, что на кисти: в ворсе — доля краски полной кисти (`_полная_кисть`) по месту гребня
    поперёк кисти (`место` — номер корзины, (nr,)), на кончике — доля её капли. Пока кисть в воздухе, краска из ворса
    подходит к кончику: капля доливается до того, сколько держит кончик этого гребня с таким запасом (`капля0`; полная —
    пока в ворсе больше `_ЗАГРУЗКА[0]` полного), и столько же уходит из ворса (ёмкость (nr,) — объём единицы заряда
    гребня, в тех же единицах, что капля). Ответ: (заряд, заряд полной кисти, капля) — заряд в единицах этого хода,
    объём тот же (Д-1-2)."""
    brush_hair0, tip0 = full_brush
    reserve = loading.reserve[place]
    brush_hair, drop = brush_hair0 * reserve, tip0 * loading.drop[place]
    top_up = np.minimum(np.maximum(drop0 * clamp(reserve / H._ZAGRAtZBrushA[0], 0.0, 1.0) - drop, 0.0), brush_hair)
    capacity = np.maximum(capacity, 1e-12)
    return (brush_hair - top_up) / capacity, brush_hair0 / capacity, drop + top_up


def _in_loading(place, reserve, drop, weight=None):
    """Д-1: что осталось в ворсе и на кончиках после хода — по местам поперёк кисти (`_КОРЗИН`), как `Кисть._запомнить`:
    запас, капля (nr,) — доли краски полной кисти (`_полная_кисть`; на кончике гребня, набранного гуще, — и больше 1);
    пустое место берёт у ближнего. Ответ: (запас, капля) по корзинам."""
    if weight is not None:      # К-14б: кисть из пучков — в просвете волоса нет: остаток считается по пучкам (иначе нетронутый «запас» просветов держал бы кисть полной)
        has = np.bincount(place, weight, _BrushResponseRZAndN)
        nearest = np.abs(np.arange(_BrushResponseRZAndN)[:, None] - np.nonzero(has)[0][None, :]).argmin(axis=1)
        return tuple((np.bincount(place, a * weight, _BrushResponseRZAndN)[has > 0] / has[has > 0])[nearest] for a in (clamp(reserve, 0.0, 1.0), clamp(drop, 0.0, 2.0)))
    has = np.bincount(place, None, _BrushResponseRZAndN).astype(np.float64)
    nearest = np.abs(np.arange(_BrushResponseRZAndN)[:, None] - np.nonzero(has)[0][None, :]).argmin(axis=1)
    return tuple((np.bincount(place, a, _BrushResponseRZAndN)[has > 0] / has[has > 0])[nearest] for a in (clamp(reserve, 0.0, 1.0), clamp(drop, 0.0, 2.0)))


def _razrovnyat(ho, wo, similar, Co, Ao, wo_mobile, touches, palette_knife, clean, by_hair):
    """Кисть разравнивает рельеф под собой — только сырое (Ф-3): высохшая краска твёрдая и под ворсом стоит. 5а: вытертая
    кисть (`чистая`) ровняет рельеф, краски не прибавляя; З-1: остаток объёма — по лежащей сырой краске, не по пустому
    месту слоя. «Мастихин», первая строка: согнанное на соседнее место сырое остаётся сырым — в записи слоя; сам ход
    считает подвижное (`wо_подв`) как до него: его подбор и края уже сочтены по прежнему. `по_волосу` —
    `мягкая.по_волосу`. Ответ: толщина, влажность, схват и цвет слоя под ходом."""
    even = _box(_box(ho, 8), 8)
    brush_even = even * by_hair - ho
    strength, stands_still_dry = (0.95, 0.0) if palette_knife else _levels_wet(ho, wo_mobile, even, brush_even)
    h_after = ho + M.even(clean, brush_even * (touches * strength), touches * (1.0 - stands_still_dry), ho * wo_mobile, lambda a: _box(_box(a, 8), 8))
    return (h_after,) + _carries_wetness(ho, h_after, wo, similar, Co, Ao, clean)


def thickness_tuft(brush):
    """Д-7: толщина пучка плоской кисти, px холста: поле `пучок_мм` (мм; на холсте без `px_на_мм` — 8 px/мм); None —
    по ширине кисти: обойма щетинной плоской сжата до `_ПУЧОК[0]` ширины, не тоньше `_ПУЧОК[1]` и не толще `_ПУЧОК[2]` мм.
    0 — толщины нет (как до Д-7); у круглой и мастихина её нет."""
    if brush.shape != 'плоская':
        return 0.0
    mm = 8.0 * H._MM
    if brush.tuft_mm is None:
        return float(min(clamp(H._PAtHResponseBrush[0] * float(brush.width), H._PAtHResponseBrush[1] * mm, H._PAtHResponseBrush[2] * mm), float(brush.width)))
    return max(float(brush.tuft_mm), 0.0) * mm


def thickness_jab(brush):
    """К-14б-2: сколько кисть из пучков (`Кисть.просветы` > 0), поставленная на месте, кладёт своей толщиной, px по ходу
    кисти: пучок обычной для своей ширины толщины (`толщина_пучка` без `пучок_мм`) — столько же, сколько плоская той же
    ширины (`_ТЫЧОК[0]` ширины: обойма та же, ворс тот же, только стоит клоками); тоньше или толще — во столько же раз;
    веер в один ряд (`пучок_мм` 0) — 0: его отпечаток — один расход кончиков."""
    T = thickness_tuft(brush)
    if T <= 0.0:
        return 0.0
    normal = float(min(clamp(H._PAtHResponseBrush[0] * float(brush.width), H._PAtHResponseBrush[1] * 8.0 * H._MM, H._PAtHResponseBrush[2] * 8.0 * H._MM), float(brush.width)))
    return H._TYHResponseBrush[0] * float(brush.width) * T / normal


def _edge(path, brush, touch):
    """Д-7: плоская с толщиной пучка, взятая не плашмя (`касание` < 1), — та же кисть, повёрнутая в пальцах ребром к
    ходу на (1 − касание)·90°: поперёк хода идёт её толщина, ворс лежит по ходу (азимут — ход, если рука не дала
    своего), холста она касается как плашмя. Касание на месте (тычок) — отпечаток пучка, кисть стоит всем торцом — `касание` у тычка 1.
    Ответ: (путь, касание, толщина пучка px)."""
    T = thickness_tuft(brush)
    brush0 = float(clamp(brush.touch if touch is None else touch, 0.0, 1.0)) if brush.shape == 'плоская' else 1.0
    if T <= 0.0 or brush0 >= 1.0:
        return path, touch, T
    if path['на_месте']:
        return path, 1.0, T   # тычок: пучок стоит на холсте всем торцом, как его ни держи, — отпечаток один, куда он смотрит — `угол`
    path = dict(path, angle=path['угол'] + (1.0 - brush0) * np.pi / 2, azimuth=path['ход'] if path['азимут'] is None else path['азимут'])
    return path, 1.0, T


def _rows_tuft(T, with_, rng, p, path, angle, W):
    """Д-7: волоски лежат в толщину пучка рядами, а не одной линией: у каждого своё место поперёк толщины, v ∈ ±Т/2
    (по одному на слой толщины — пучок набит ровно; свой поток: прежние не сдвигаются). Под нажимом и наклоном пучок
    расплющивается в толщину (`_ПУЧОК[3:6]`). На след идёт доля толщины поперёк хода — sin(угол кисти − ход), и только
    там, где ряды не идут след в след за задним рядом: из поперечника следа W·|cos| + Т·|sin| на толщину приходится
    Т·|sin|; пока эта доля мала (`_ПУЧОК[6]`), борозду режет задний ряд — плашмя и на дуге, где кисть отстала от хода,
    след как был бит в бит; к доле `_ПУЧОК[7]` ряды кладут каждый свой след — на ребре вся толщина. Ответ: (места
    волосков u, их v px, доля поперёк хода (N,), орт поперёк хода x, y (N,)) или None — толщины нет."""
    if T <= 0.0:
        return None
    ss = rng.bit_generator.seed_seq
    rng_t = np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=tuple(ss.spawn_key) + (7701,)))
    wider = (H._PAtHResponseBrush[3] + H._PAtHResponseBrush[4] * clamp(p / 0.85, 0.0, 1.2)) * (1.0 + H._PAtHResponseBrush[5] * np.sin(np.radians(clamp(path['наклон'], 0, 85))))
    v = ((rng_t.permutation(with_.nm) + rng_t.random(with_.nm)) / with_.nm - 0.5) * T
    sn, cs = np.sin(angle - path['ход']), np.cos(angle - path['ход'])
    own = T * np.abs(sn) / np.maximum(W * np.abs(cs) + T * np.abs(sn), 1e-9)
    own = clamp((own - H._PAtHResponseBrush[6]) / (H._PAtHResponseBrush[7] - H._PAtHResponseBrush[6]), 0.0, 1.0)
    own = own * own * (3.0 - 2.0 * own)
    if not own.any():
        return None
    return with_.u, v, sn * own * wider, -np.sin(path['ход']), np.cos(path['ход'])


def lay_stroke(layer, points, paints, brush, seed, opacity=1.0, layout='полосы', load=None,
                   lower=(), touch=None, window=None, before_writing=None, measurement=None, grain=None, turbidity=1.0, under=(), transmittance=None, bands=None, neighboring=(), by_ridges=None, even=None, even_bottom=((), 0.0), reserve=None, release=None, fibers=None, hand=False, why=None, thinner=0.0, loading=None, lines=None):
    """Посчитать мазок и вписать его в слой.

    краски — список (линейный RGB (3,), доля). нижние — слои под этим, с сырой краской которых
    мазок тоже мешается (по умолчанию — только свой слой). окно (п. 9) — (x0, y0, ширина, высота) холста, если слой — участок с углом (x0, y0). зерно (п. 5б) — `Холст._зерно_под`: рельеф грунта под мазком, или None — грунт гладкий. полосы (п. 13) — (ширина полосы набора, длина куска краски в ней), px холста, любое — None: как было. мутность, под, пропуск (п. 10, инструмент 3) — вид краски (`краска.Краска`; 1 — кроющая, как было), слои под этим (их рельеф) и функция, дающая массив пропуска слоя. разбавитель (Г-1) — 0–1: у не кроющей краски тоже идёт `ровно` (нагрузку, мутность, ровно и запас жидкой краски даёт `краска.разбавить`). запас (п. 17) — на сколько ширин кисти хватает краски (сухой срыв, `_срыв`); None — как у кисти. перед_записью (11б) — зовётся до записи с прямоугольником холста, вне которого мазок ни в какой слой не пишет. Ответ — записанный прямоугольник холста (x0, y0, x1, y1) или None. почему (v4, 1б) — словарь: при ответе None в него пишется, где мазок кончился без следа (`"что"`: `вне` — след кисти весь за краем, `"след"` — его прямоугольник (x0, y0, x1, y1) в px холста; `нажим` — ни один гребень не коснулся; `чистая` — нагрузка 0 и сырого под кистью нет; `пусто` — касания были, краски не легло); на пиксели не влияет. замер (v4, 3г, К-1) — функция: зовётся после записи со словарём полей следа для ответа мазка (`ответ.Ответ`); на пиксели не влияет. загрузка (Д-1) — живая краска кисти (`инструмент.Загрузка`) или None: ход начинает с её запаса и капли, что осталось — в `к._расход`."""
    rng, rng_b, rng_in, rng_g, rng_d, rng_e = _flow(seed)
    ox, oy, Sh, In = window if window is not None else (0, 0, layer.coverage.shape[1], layer.coverage.shape[0])
    path = _path(points, brush, rng)
    path, touch, t_tuft = _edge(path, brush, touch)   # Д-7: у пучка плоской есть толщина; `касание` < 1 — кисть повёрнута ребром к ходу
    x, y, p, speed, angle, ds, N, L = path['x'], path['y'], path['p'], path['скор'], path['угол'], path['ds'], path['N'], path['L']
    W = float(brush.width)
    reserve_fields = W * 0.9 + 2 * _PResponseLE
    if (x.max() < -reserve_fields or y.max() < -reserve_fields or x.min() > Sh + reserve_fields
            or y.min() > In + reserve_fields or p.max() <= 0.0):
        return _empty(why, 'нажим 0' if p.max() <= 0.0 else 'вне')
    palette_knife = brush.shape == 'мастихин'
    with_ = _makeup(brush, rng)
    nr, ns, nm = with_.nr, with_.ns, with_.nm
    H, clean = M.clean(brush, load)                        # толщина плёнки полного мазка; п. 10, инструмент 5 (5а): нагрузка 0 — кисть вытерта, своей краски нет (`мягкая.чистая`)
    even = float(even) if even and (turbidity >= 1.0 or thinner) and not clean and not _dry(brush, load) else 0.0; sparingly = _breakaway(reserve, palette_knife or clean or _dry(brush, load))   # п. 15: текучая кроющая краска; п. 17: краски набрано на `запас` ширин кисти
    mt = scale_thicknesses(W)
    thickly = float(clamp((H / mt - 0.35) / 0.5, 0.0, 1.0))  # 0 — тонкая краска, 1 — пастозная
    tau = _TAAt * mt / max(brush.hiding_power, 1e-3)
    sharp = float(brush.point); rng_uz, narrow_charge = _narrow_charge(rng, paints, palette_knife, W, layout, thickly, sharp)   # п. 10; п. 12: щетина в обойме (`остриё` 0) острого кончика не знает и узкой
    live, p, p_hand = _tremor(rng, p, W, ds, thickly, palette_knife or (len(paints) != 1 and not narrow_charge) or _dry(brush, load), sharp)   # п. 10: нажим узкой кисти гуляет; полупустая каплей не садится
    if path['на_месте'] and brush.shape == 'круглая':
        p = p_hand   # касание-2: поставленную на месте кисть рука по холсту не ведёт — дрожи хода нет, а капля посадки садится тем же пятном, что пучок: нажим по хорде (`путь._отпечаток_пучка`) остаётся как дан, иначе капля держала все гребни на всей длине отпечатка и узкая круглая ставила брусок
    # густые несмешанные краски на кисти (п. 2в): набраны с палитры рядом и не размешаны
    charge, soft = len(paints) > 1 and thickly > 0 and not palette_knife and layout in ('полосы', 'ряд') and not narrow_charge, Soft.for_(brush, with_, rng, p, ds, N, W, M.wet_under(brush, layer, x, y, angle, with_.center, ox, oy, tau), release)   # п. 10, инструмент 4: мягкая кисть (`Кисть.кайма`); у щетины — None, и всё как было; 4б: по сырой плёнке соседнего мазка короткий волос достаёт
    paints_r = paints
    if charge:
        # доли из вызова — в среднем: с палитры кисть набирает каждый раз чуть по-своему, до ±30 %
        # (узкая кисть — меньше: у неё на полосу три-четыре гребня, и полоса пропала бы целиком)
        d = np.array([d for _, d in paints]) * np.exp(clamp(_DResponseLAnd * min(W / 40.0, 1.0) * rng_in.standard_normal(len(paints)), -0.3, 0.3))
        paints_r = [(c, q / d.sum()) for (c, _), q in zip(paints, d)]

    # --- касание гребней
    # сколько кисти касается холста, решает не нажим, а то, как кисть поставлена: плашмя — всей шириной
    # и при лёгкой руке; на ребре — узко, и ширина растёт с нажимом. Нажим решает, насколько она втирает
    laid_flat = float(clamp(brush.touch if touch is None else touch, 0.0, 1.0)) if brush.shape == 'плоская' else 0.0
    # поставленная плашмя кисть лежит всей шириной, пока рука держит свой нажим, каким бы лёгким он ни был;
    # когда руку поднимают (нажим падает против набранного в этом мазке), кисть встаёт на кончик:
    # дальние и короткие пряди отходят первыми, след сужается — как у кисти на ребре
    share_pressure = p / np.maximum(np.maximum.accumulate(p), 1e-6)
    lies = clamp((share_pressure - 0.5) / 0.4, 0.0, 1.0)
    lies = lies * lies * (3.0 - 2.0 * lies)
    # сорванная кисть на кончик встать не успевает: нажим ушёл быстрее, чем за ширину кисти хода, —
    # она отходит от холста плашмя, вся разом, и след кончается обрывом, а не клином
    falloff = np.maximum(-np.gradient(p, ds), 0.0) * W / np.maximum(np.maximum.accumulate(p), 1e-6)
    fast = clamp((falloff - 0.7) / 0.6, 0.0, 1.0)
    fast = fast * fast * (3.0 - 2.0 * fast)
    max = 0.0
    for i in range(N):                                        # начатый срыв держится до конца этого спада
        max = max(max, fast[i]) if share_pressure[i] < 0.97 else fast[i]
        fast[i] = max
    torn, fast = fast, fast * thickly                  # тонкая краска обойму не держит: её конец прежний
    lies = laid_flat * np.maximum(lies, fast)                       # (N,)
    p_r = p + (0.85 * share_pressure ** 1.1 - p) * laid_flat                      # нажим «на кончике»: доля от своего полного
    if brush.shape == 'круглая':
        center = np.zeros(N)
        distance = np.broadcast_to(with_.distance[None, :], (N, nr))
    elif palette_knife:
        center = np.zeros(N)
        tilt = rng.uniform(-1, 1)
        distance = np.broadcast_to(((0.5 + 0.5 * tilt * with_.center / with_.half_width) * 0.25)[None, :], (N, nr))
    else:
        # кисть встаёт не точно серединой; место первого касания вдоль мазка не ползёт: рука кисть не перекатывает
        e0, e1 = rng.uniform(-0.7, 0.7, 2)
        center = np.full(N, e0)
        # порог идёт больше от места пряди, чем от места гребня: прядь отрывается почти разом, хвост — прядью, а не иглой
        place = M.place_lift(brush, clean, with_.center, ((0.35 + 0.65 * _pointed(W, sharp)) * with_.center + 0.65 * (1.0 - _pointed(W, sharp)) * _axis_lift(with_, even * (1.0 - laid_flat)))) / with_.half_width   # 5а-2: вытертый мягкий пучок отходит от холста куполом, не прядями обоймы
        distance = np.abs(place[None, :] - center[:, None]) / (1.0 + np.abs(center))[:, None]
    threshold, exit_response = _point(with_, W, p_r, distance, palette_knife, live > 0, brush.shape == 'круглая', sharp, even * (1.0 - laid_flat))   # п. 10: у узкой кисти — острый кончик
    if thickly > 0 and not palette_knife:
        # 4б: на медленном подъёме гребень, полный густой краски, держится за холст дольше соседа — концы гребней
        # стоят вразнобой, а не лесенкой от края к середине
        agrees = (np.arange(N) > int(np.argmax(p))) & (share_pressure < 0.97)
        rng_zh = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])      # свой поток: прежние не сдвигаются
        threshold = threshold + (_LIFT_GR * thickly * (rng_zh.random(nr) - 0.5))[None, :] * agrees[:, None]
    threshold = _from_lower(threshold, with_)
    threshold, tuft = _Tuft.for_(threshold, with_, brush, path, W, sharp, rng.bit_generator.seed_seq)   # Н-2: у тычка круглой пятно касания — круг пучка, а не стопка прядей
    touch, reserve_g = _touch_ridges(threshold, with_, brush, path, p, p_r, laid_flat, lies, rng.bit_generator.seed_seq)   # (N, nr); запас: нажим гребня сверх порога касания (6а-2: отрыв по сырому)
    # посадка и подъём кисти (п. 2д): плоская кисть, которую кладут плашмя, сначала касается серединой обоймы,
    # углы встают, когда нажим набран и щетину потащило; при подъёме углы уходят первыми. Кисть редко стоит ровно:
    # одна сторона встаёт раньше другой. Удар с коротким входом кладёт кисть разом — торец плоский
    landing_m = holds = None
    if laid_flat > 0 and nr > 2 and soft is None:   # мягкий пучок обоймой не встаёт: его вход — купол и нажим волоска (`мягкая.py`)
        landing_g, landing_m = _landing(with_, path, p, rng_d, W, rng_e, thickly, p_hand if live > 0 else None, holds := _holds(brush, p, W, laid_flat, speed, share_pressure, torn, even, palette_knife or clean or sparingly is not None or _dry(brush, load) or turbidity < 1.0), hand, _ahead(brush, with_, path, p, W, laid_flat, thickly, palette_knife, speed) if opens_hand(p, hand) else None)   # п. 18в; К8.1: `рука` — ширина посадки по самому нажиму; К8б — и где лёг кончик
        if landing_m is not None:
            touch = touch * (1.0 + (landing_g - 1.0) * laid_flat)
    exit_g, ends, stood = _ends(brush, with_, path['s'], p, p_r[:, None] - threshold, reserve_g, lies, W, thickly, laid_flat, speed, share_pressure, fast, even * (1.0 - laid_flat), palette_knife, clean, seed, even) if soft is None and turbidity >= 1.0 else (None, None, 0.0)   # п. 18: концы следа — пятно касания и сход прядей; множители укладки, перенос краски прежний
    angle_m, touch = None, (touch if soft is None else soft.touch(touch, reserve_g)); exit_g, touch_at, thin_g = _tail(brush, with_, p, p_r, p_r[:, None] - threshold, lies, W, thickly, laid_flat, speed, share_pressure, fast, even * (1.0 - laid_flat), palette_knife, clean, _dry(brush, load) or sparingly is not None, exit_g, touch, holds if landing_m is not None else None)   # п. 18б: касание на укладке — с хвостом сходящего хода; перенос краски идёт по `кас`, как раньше   # мягкая кисть: короткий волос края достаёт до холста слабее или не достаёт
    if lines is not None: return _lines_tuft(lines, with_, touch_at, reserve_g, ds)   # К-14б: сухой счёт — только где какой пучок касается холста; холст и кисть не тронуты
    tip = np.zeros(nr)
    # густая кисть плашмя (п. 2г): бок, торец, углы, тонкое место. П. 2д: кисть в одну краску живёт так же — чем гуще
    # краска, тем полнее (пастозная — целиком, до нагрузки 0,55 — как раньше); смесь у кромки — только у набора
    one_thick = len(paints) == 1 and not palette_knife
    lives = 1.0 if charge else (float(clamp((thickly - 0.4) / 0.4, 0.0, 1.0)) if one_thick else 0.0)
    thick = lives > 0 and laid_flat > 0 and nr > 4
    if thick:
        # углы входа скруглены (п. 2в, 2г): у самого края краски меньше, и кисть подходит к холсту серединой —
        # крайние полтора-два пикселя следа встают на 1–4 px позже; считается по месту на кисти, а не по гребню:
        # у каждого удара оба угла, середина торца не тронута
        edge_sl = np.full(2, with_.half_width - _YaDRResponse / 2)            # где кончается след: левый и правый край
        for kr, st in ((0, 0), (nr - 1, 1)):
            if 2.0 * with_.half_width[kr] < 2.0:                           # крайний гребень уже двух пикселей краски не кладёт
                edge_sl[st] -= 2.0 * with_.half_width[kr]
        st_m = (with_.u > 0).astype(np.intp)
        from_kr = edge_sl[st_m] - np.abs(with_.u)
        back_m = rng_g.uniform(*_AtGResponseL[:2], 2)[st_m] * clamp((_AtGResponseL[3] - from_kr) / (_AtGResponseL[3] - _AtGResponseL[2]), 0.0, 1.0) \
            * thickly * lives * laid_flat * float(clamp(W / 24.0, 1.0, 1.7)) ** 0.4     # у широкой кисти слой толще и угол заплывает
        travelled = path['s'] - path['s'][int(np.argmax(p > 0.02))]
        z = clamp(travelled[:, None] - back_m[None, :] + 0.5, 0.0, 1.0)
        angle_m = np.where(back_m[None, :] > 0.05, z * z * (3.0 - 2.0 * z), 1.0)       # (N, nm)
        # торец — кончики щетины, а не линейка (п. 2г): гребни стоят вдоль хода неровно на 1–2 px ещё до того,
        # как их потащили
        # (вперёд из торца кончик не торчит: большинство стоит в ряд, часть отстаёт; у углов — своя задержка)
        tip = _BrushResponseNHAndBrush * (1.0 if charge else 1.0 + 0.25 * float(clamp((W - 24.0) / 16.0, 0.0, 1.0))) * thickly * lives * laid_flat * clamp(rng_g.uniform(-2.0, 1.0, nr), 0.0, None) ** 0.7 \
            * (with_.half_width - np.abs(with_.center) > 2.5)
    if brush.fan > 0 and not palette_knife: tip = tip - float(brush.fan) * W * (1.0 - clamp(with_.distance, 0.0, 1.0) ** 2)   # К-14б: ворс веером — кончики стоят дугой: середина кисти длиннее краёв и тянется за кистью дальше позади
    if not touch.any():
        return _empty(why, 'нажим')

    # --- место гребней и выборки профиля на холсте
    ax, ay, tx, ty = -np.sin(angle), np.cos(angle), np.cos(angle), np.sin(angle)   # широкая ось кисти; по ходу
    # при обычном нажиме 0,85 след — в ширину кисти; придавленная кисть расходится веером шире
    # «касание и короткий ход»: круглая кисть, поставленная на месте (`путь._отпечаток_пучка`), раскрыта нажимом руки на
    # всём отпечатке — по его длине идёт только пятно касания (порог гребней), а ворс вбок за ним не ходит: иначе крайние
    # пряди чертили дуги от оси к краю и обратно («усы» вместо пятна)
    p_width, p_width_r, exit_sh = (np.full(N, p.max()), np.full(N, p_r.max()), (np.full(N, exit_response[int(np.argmax(p))]) if isinstance(exit_response, np.ndarray) else exit_response)) if path['на_месте'] and brush.shape == 'круглая' else (p, p_r, exit_response)
    spread0 = spread = _spread(brush, p_width, fast * laid_flat, lies, _belly(W, live > 0 and brush.shape == 'круглая', p_width_r), exit_sh)
    if holds is not None: spread = _spread(brush, p_width, fast * laid_flat, lies, _belly(W, live > 0 and brush.shape == 'круглая', p_width_r), exit_sh, holds)   # п. 18в: рука сходит с кисти плашмя — ворс к оси не собирается
    axis = center * with_.half_width
    sides = None
    if palette_knife:
        spread = np.ones(N)
        tack = np.ones((N, ns))
        leaving = np.zeros((N, ns))
    else:
        # пряди: при слабом нажиме собираются к своей оси (просветы открываются), при сильном смыкаются;
        # медленно дышат и уходят вбок вдоль пути — не больше пикселя на сотню
        breathes = _noise(rng, N, ns, 140.0 / ds)
        # полная кисть плашмя: густая краска держит пряди вместе, просветы открывает только пустеющая
        tack = 1.0 - with_.sticks[None, :] * (1.0 - 0.85 * thickly * lies)[:, None] \
            * clamp(1.15 - (p_width_r[:, None] / 0.85) ** 1.5 + 0.22 * breathes, 0.0, 1.0)
        # кромка мазка идёт за крайними прядями: крайняя прядь свободна с одной стороны, трение по холсту
        # то поджимает её к соседке (краска там ложится гуще), то отпускает — на пиксель-два на длине
        # в ширину кисти. Кисть встаёт на своё место, бок уходит только с ходом; у каждого бока своя жизнь
        with_sweep = 1.0 - np.exp(-np.maximum(path['s'] - path['s'][int(np.argmax(p > 0.02))], 0.0) / (0.3 * W))
        # тонкая краска кромку держит сухим следом и просветами, как раньше; поджатая прядь видна у густой
        range = _BResponseBrush * (_SIDE_CHARGE if charge else 1.0 + (_SIDE_ONE - 1.0) * lives * float(clamp((W - 16.0) / 8.0, 0.0, 1.0))) * thickly * (0.6 + 2.0 * with_.sticks[[0, ns - 1]] / max(brush.scraps, 1e-3) * min(brush.scraps, 1.0))
        # наружу за свою линию кисть не выходит: шире всего она там, где встала
        sides = clamp(_noise(rng_b, N, 2, float(clamp(0.45 * W, 7.0, 26.0)) / ds) + _SIDE_SHIFT, 0.0, None) \
            * range[None, :] * ((with_sweep ** 2 if charge else with_sweep) * (1.0 - fast))[:, None]   # сорванная кисть отходит плашмя, во всю ширину
        # (у набора в несколько красок бок гуляет сильнее, но у самого входа кисть ещё стоит во всю ширину: иначе
        # крайняя строка следа у входа пропадала и начиналась позже)
        leaving = _leaving_short(_noise(rng, N, ns, 260.0 / ds) * 0.5, path, p, W)
        tack, leaving = M.tuft(soft, tack, _strands_narrow(with_, leaving, live, rng_uz, N, W, ds, brush.shape == 'круглая'))   # п. 10; 4а-2: мягкий волос на пряди со щелями не расходится
        if tuft is not None: tack, leaving, sides = tuft.open(tack, leaving)
    drift = None
    if charge:
        # пряди внутри густой кисти сходятся и расходятся, как крайняя у бока: граница пряди уходит вбок
        # на 1–3 px на длине в ширину кисти, плавно; гребни между границами раздвигаются и сжимаются
        # (поперечное место — гладкая функция места на кисти: пряди не перехлёстываются). Края кисти на месте
        nodes = (with_.center + with_.half_width)[np.nonzero(with_.strand[1:] != with_.strand[:-1])[0]]
        nodes = nodes[np.abs(nodes) < with_.half_width - 2.0]
        if len(nodes):
            anchors = np.concatenate([[-with_.half_width], nodes, [with_.half_width]])
            K = len(anchors)
            position = np.broadcast_to(anchors, (N, K)).copy()
            position[:, 1:-1] += _noise(rng_in, N, K - 2, 0.8 * W / ds) * (_DREYF * thickly * min(W / 40.0, 1.0)) * with_sweep[:, None]
            gap = np.diff(anchors)
            for q in range(1, K - 1):
                position[:, q] = clamp(position[:, q], position[:, q - 1] + 0.65 * gap[q - 1], position[:, q - 1] + 1.5 * gap[q - 1])
            for q in range(K - 2, 0, -1):
                position[:, q] = clamp(position[:, q], position[:, q + 1] - 1.5 * gap[q], position[:, q + 1] - 0.65 * gap[q])
            drift = (anchors, position - anchors[None, :])
        # след густой кисти дышит по ширине на 5–10 %: лента постоянной ширины — штамп
        # (у входа и у отрыва — нет: там ширину держит торец, а сужение читалось бы скруглённым углом)
        from_end = clamp(np.minimum(path['s'] - path['s'][int(np.argmax(p > 0.02))], path['L'] - path['s']) / W - 0.5, 0.0, 1.0)
        # (п. 2г: только внутрь — шире всего кисть там, где встала; иначе у бока вырастала строка, которой нет у входа)
        spread = spread * (1.0 - 1.6 * _DYShAndT * clamp(-_noise(rng_in, N, 1, W / ds)[:, 0], 0.0, None) * from_end)
    along_brush = 1.0 + 2.0 * np.sin(np.radians(clamp(path['наклон'], 0, 85)))
    if path['азимут'] is not None:
        qx, qy = np.cos(path['азимут']), np.sin(path['азимут'])
    else:
        qx, qy = tx, ty

    # кисть встаёт на холст ровным торцом: пряди расходятся вдоль хода, только когда их потащили
    # (трение у каждой своё) — отставание набирается на первой ширине пути от касания
    touched = path['s'][int(np.argmax(p > 0.02))]
    drags = (1.0 - np.exp(-np.maximum(path['s'] - touched, 0.0) / W)) * _stood(brush, palette_knife, laid_flat, speed) * _reached(stood, laid_flat, speed)   # п. 16: остановленная на ребре кисть встаёт
    if palette_knife or brush.shape == 'круглая':
        drags = np.ones(N)
    if path['на_месте'] and brush.shape == 'круглая':
        drags = np.zeros(N)                                   # «касание и короткий ход»: поставленную на месте кисть не тащили — пряди вдоль хода не разошлись
    # густая краска у кисти плашмя держит пряди обоймой: вдоль хода они расходятся не дальше её доли ширины —
    # обрыв сорванной кисти не длиннее трети ширины, но у каждой пряди своё место в нём
    ferrule = 0.0 if (palette_knife or brush.shape == 'круглая') else laid_flat * thickly
    limit = _ResponseBResponseYMA * W

    if sides is not None:
        # крайняя прядь поджимается к своей внутренней границе — к соседке; у кисти в одну-две пряди — к середине; 11г: и не дальше неё — иначе узкая прядь выворачивалась и клала отрицательную краску (цвет слоя 10⁸, README)
        anchor_l, anchor_p = min(float((with_.center + with_.half_width)[with_.strand == 0].max()), 0.0), max(float((with_.center - with_.half_width)[with_.strand == ns - 1].min()), 0.0)
        sides = np.minimum(sides, np.array([anchor_l + with_.half_width, with_.half_width - anchor_p])[None, :])

    along_canvas = lambda lag, tip: _along_brush_hair(lag, tip, along_brush * drags, ferrule, limit)   # noqa: E731
    row_v = _rows_tuft(t_tuft, with_, rng, p, path, angle, W)   # Д-7: место волоска в толщину пучка
    def on_canvas(u, strand, lag, tip, was=False):
        """Поперечные места u (их пряди, их отставания, места кончиков) -> координаты (N, len(u))."""
        axis_p = with_.axis_strands[strand]
        side = axis_p[None, :] + (u - axis_p)[None, :] * (tack0 if was else tack)[:, strand]
        if drift is not None:
            anchors, shift = drift
            j = clamp(np.searchsorted(anchors, u) - 1, 0, len(anchors) - 2)
            t = clamp((u - anchors[j]) / (anchors[j + 1] - anchors[j]), 0.0, 1.0)
            side = side + shift[:, j] * (1.0 - t)[None, :] + shift[:, j + 1] * t[None, :]
        side = axis[:, None] + (side - axis[:, None]) * (spread0 if was else spread)[:, None] + leaving[:, strand]
        if sides is not None:
            side = side + sides[:, :1] * clamp((anchor_l - u) / (anchor_l + with_.half_width), 0.0, 1.0)[None, :] \
                - sides[:, 1:] * clamp((u - anchor_p) / (with_.half_width - anchor_p), 0.0, 1.0)[None, :]
        vd = along_canvas(lag, tip)   # отставание вдоль ворса; его же берёт перемычка (`Просвет.в_растр`)
        if row_v is not None:
            across = row_v[1][clamp(np.searchsorted(row_v[0], u), 0, len(row_v[0]) - 1)][None, :] * row_v[2][:, None]
            return M.behind_brush(soft, x[:, None] + side * ax[:, None] + across * row_v[3][:, None], y[:, None] + side * ay[:, None] + across * row_v[4][:, None], vd, qx, qy)
        return M.behind_brush(soft, x[:, None] + side * ax[:, None], y[:, None] + side * ay[:, None], vd, qx, qy)   # К5: с "load_along" кончик идёт за своим корнем

    strand_m = with_.strand[with_.ridge]
    # у густой кисти плашмя гребни внутри пряди рвутся с холста каждый в своём месте
    lag_g = with_.lag + (_LAG_GR[0] + _LAG_GR[1] * W) * ferrule * with_.lag_gr
    (patch_g, patch_m), (patch_g0, patch_m0) = (_laid(q, with_, path['s'], lag_g, along_brush * drags, ferrule, limit) for q in (ends, [q[:2] + q[4:] if len(q) > 4 else q for q in ends] if ends and any(len(q) > 4 for q in ends) else None))   # п. 18: пятно касания у концов — по холсту; Д1, строка 5: и оно же ровным овалом (мерка краски конца хода), если кромка остановки идёт по прядям
    gap, tack0, tack = M.without_slit(soft, Gap(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]), with_, tack, leaving, 1.0 if (palette_knife or tuft is not None) else thickly, N, ds, W)), tack, (tack if holds is None else _Film.strands(tack, holds * _Film.thick(thickly)))   # 6а, свой поток; 4а-2: у мягкой кисти щелей между прядями нет; п. 18в: пока рука сходит, пряди густой краски к своим осям не собираются (щели считаны как были)
    if opens_hand(p, hand) and isinstance(gap, Gap): gap.both = True   # К8б: плёнка щели — от обеих прядей
    X, Y = separate(soft, *on_canvas(with_.u, strand_m, lag_g[with_.ridge], tip[with_.ridge]), ax, ay, with_.ridge)   # мягкая кисть: крайний волос водит вбок
    if live > 0 and gap.has: _settled(gap, with_, live, p, ds, W, rng_uz)   # п. 10: у посадки щель закрыта, дальше пряди расходятся и сходятся
    # --- прямоугольник мазка (с полями), обрезанный холстом с запасом
    field = _PResponseLE + (0 if palette_knife else int(np.ceil(H._GROWING_TOGETHER_BANK[1] + H._GROWING_TOGETHER_BANK[2] * W)))   # 6а: за кромкой следа — место валу сырой краски
    bx0, by0 = max(int(np.floor(X.min())) - field, -_BEHIND_EDGE), max(int(np.floor(Y.min())) - field, -_BEHIND_EDGE)
    bx1 = min(int(np.ceil(X.max())) + field + 1, Sh + _BEHIND_EDGE)
    by1 = min(int(np.ceil(Y.max())) + field + 1, In + _BEHIND_EDGE)
    bw, bh = bx1 - bx0, by1 - by0; (why if why is not None else {})['рамка'] = M.with_axis(brush, (bx0, by0, bx1, by1), x, y)   # 2б: вне её мазок холста не читает и не пишет (`Холст.прямоугольник_мазка`, `ядра.py`)
    if bw < 3 or bh < 3:
        return _empty(why, 'вне', X, Y)
    cx0, cy0, cx1, cy1 = max(bx0, ox), max(by0, oy), min(bx1, ox + layer.coverage.shape[1]), min(by1, oy + layer.coverage.shape[0])
    if cx1 <= cx0 or cy1 <= cy0:
        return _empty(why, 'вне', X, Y)
    local, canvas = (slice(cy0 - by0, cy1 - by0), slice(cx0 - bx0, cx1 - bx0)), (slice(cy0 - oy, cy1 - oy), slice(cx0 - ox, cx1 - ox))
    if before_writing is not None: before_writing((cx0, cy0, cx1, cy1))     # 11б: снимок холста копирует, что тут лежит

    def take(massiv, tryokh=False):
        a = np.zeros((bh, bw, 3) if tryokh else (bh, bw))
        a[local] = massiv[canvas]
        return a

    ho, Ao, wo = take(layer.thickness), take(layer.coverage), take(layer.moisture)
    similar = take(layer.degree_of_set)
    has_paint = bool(Ao.any() or ho.any())   # Ф-1: плёнка мутности 0 кладёт толщину без покрытия — это тоже лежащая краска
    Co = take(layer.color, True) if has_paint else None
    if palette_knife:   # «мастихин»: нож — не ворс, счёт свой (`нож.py`)
        return _knife(layer, path, brush, W, H, clean, mt, tau, paints, touch, grain, under, take, ho, Ao, wo, similar, Co, local, canvas, (bx0, by0, bw, bh), (cx0, cy0, cx1, cy1), opacity, turbidity, transmittance, measurement, why, X, Y)
    # сырая краска, с которой мазок мешается: своя, а если просили — и нижних слоёв;
    # схватившаяся плёнка подвижна лишь отчасти
    wo_mobile = wo * (1.0 - _WithCanvasInAT * similar)
    wet_h = ho * wo_mobile
    wet_h, wet_c, bottom = _wet_below(lower, take, wet_h, Co)
    has_wet = bool(wet_h.max() > 1e-4)

    # --- перенос краски по гребням, шаг 2 px
    ic = np.arange(0, N, _ShBrush)
    Nc = len(ic)
    spacing_brush = _ShBrush * ds
    v0 = 2.0
    from_speed = ((1.0 + 1.0 / v0) / (1.0 + speed[ic] / v0)) ** (0.35 + 0.5 * brush.viscosity)
    stands_still = clamp(0.25 / speed[ic] - 0.5, 0.0, 6.0)          # кисть почти стоит: краска стекает на место
    # нажим у полной густой кисти: лёгкая рука оставляет краску толще, валиком; плотная прижимает слой
    # тоньше, а лишнее выдавливает к бокам (ширину следа нажим не меняет — она за касанием)
    layer_from_pressure = 1.0 + thickly * clamp(_LAYER_PRESSURE * (0.6 - p[ic]), -0.28, 0.30)
    # тонкая краска идёт с волоска тем охотнее, чем сильнее он прижат; густая сходит с кисти сама,
    # а нажим её только раздавливает
    release = (_narrow_paint(live, W, p=p[ic]) * (1.0 - thickly) + 0.85 * layer_from_pressure * thickly) * from_speed * (1.0 + 0.35 * stands_still) * (0.0 if clean else 1.0)   # 5а: из вытертого ворса на кончик не приходит ничего
    # запас меряется в ширинах хода, каким бы ни был нажим: на лёгком ходе полной кисти его хватает так же
    consumption = np.sqrt(p[ic] / 0.8) * from_speed * (1.0 + 0.35 * stands_still)
    touch_brush = touch[ic]
    Xg, Yg = separate(soft, *on_canvas(with_.center, with_.strand, lag_g, tip), ax, ay)
    Xg, Yg = Xg[ic] - bx0, Yg[ic] - by0
    ixg, iyg = np.floor(Xg).astype(np.intp), np.floor(Yg).astype(np.intp)
    inside = (ixg >= 0) & (ixg < bw) & (iyg >= 0) & (iyg < bh)
    ixg, iyg = np.where(inside, ixg, 0), np.where(inside, iyg, 0)
    uneven, thickness = M.reserve(soft, *uneven_reserve(with_, brush, rng, Nc, spacing_brush, W, palette_knife, _dry(brush, load)))       # запас гребней вдоль хода (`борозда.py`); 4а-2: у мягкого волоса комок запаса — на нескольких соседних волосках
    if tuft is not None and not _dry(brush, load): uneven = tuft.even(uneven)   # Долг-ф: у тычка по отпечатку идут разные волоски — запас гребня вдоль хода щелью через точку не ложится
    steepness = 1.0 / (0.16 if palette_knife else 0.30)
    addition0 = np.broadcast_to(-(0.14 + brush.dryness + 0.10 * speed[ic] / (speed[ic] + v0))[:, None], (Nc, nr))
    # рельеф лежащей краски: полусухая кисть цепляет гребни и проскакивает впадины
    drop = clamp((ho - _blur(ho, 3)) / mt, -0.6, 0.6) if has_paint else None
    relief = np.where(inside, drop[iyg, ixg], 0.0) * 1.1 if has_paint and not (_dry(brush, load) or by_ridges) else np.zeros((Nc, nr))   # у полупустой кисти (и с полем "ridges", п. 13) рельеф лежащего — по волоску (`зерно.по_зерну`)
    breaks, drop, uneven = by_grain(grain, _dry(brush, load), rng, with_, X, Y, (bx0, by0, bx1, by1), ho, mt, H, p, ic, ds, steepness, drop, uneven, by_ridges, lambda: [(ho, wo_mobile)] + [(take(ns.thickness), take(ns.moisture)) for ns in neighboring], sparingly is not None, fibers); uneven = sparingly.evener(uneven, with_.half_width) if sparingly is not None else uneven   # п. 13: рельеф краски соседних слоёв; п. 5б и п. 10, инструмент 2: зерно холста и нагрузка 'сухая' (`зерно.py`)
    furrow = M.furrows(soft, furrows(with_, brush, path, p, touch, thickly, rng_e, X - bx0, Y - by0, drop))   # жизнь борозд вдоль хода (п. 4); 4а-2: борозда мягкого волоска мельче
    if tuft is not None: furrow = tuft.dashes(furrow)
    # подбор сырой краски: доля растёт с толщиной и влажностью лежащего и с нажимом
    takes = in_wet = None
    from_edges = (1.0 - with_.distance) * W / 2                         # px от гребня до края кисти
    if has_wet and brush.pickup > 0:
        wet_g = np.where(inside, wet_h[iyg, ixg], 0.0)
        if wet_g.max() > 1e-4:
            share = clamp(brush.pickup * 1.7 * H._GROWING_TOGETHER_PULLS[1] * (0.4 + 0.6 * p[ic])[:, None] * wet_g / (wet_g + 0.05 * mt), 0.0, 0.92)
            takes = share * wet_g                              # толщина, которую гребень снимает под собой
            ks_canvas = in_ks(wet_c[iyg, ixg])                  # (Nc, nr, 24)
            if not palette_knife:
                # кромка на сырой краске: крайние гребни своей краски почти не дают, а перекладывают
                # лежащую — борозды идут по соседу, цвет переходит зубцами. Глубина перехода гуляет вдоль пути
                # глубина — в долях ширины (2–15 px у кисти 16): иначе широкая кисть втирает слабее узкой
                depth = (2.0 + 13.0 * clamp(0.55 + 0.7 * _noise(rng, Nc, ns, 70.0 / spacing_brush)[:, with_.strand]
                                              + 0.25 * rng.standard_normal(nr)[None, :], 0.0, 1.0)) * max(W / 16.0, 1.0) * Streams.rim(thickly)
                at_edges = clamp((depth * _narrow(W) - from_edges[None, :]) / (3.0 * _narrow(W)) + 0.5, 0.0, 1.0)   # п. 10: у узкой кисти кромка мельче вместе с шириной
                in_wet = 1.0 - at_edges * clamp(wet_g / (0.06 * mt), 0.0, 1.0)
    # втирание: нажим с поправкой на скорость (быстрая рука касается короче, стоящая — давит дольше)
    pressure_vt = p[ic] * (0.8 + 0.4 / (1.0 + speed[ic] / v0))
    at_edges_vt = 1.0 - 0.3 * clamp((1.5 - from_edges) / 1.5, 0.0, 1.0)   # у самого края кисти подушка тоньше
    by_wet, by_canvas, full = np.zeros(nr), np.zeros(nr), (1.0 if clean or brush.charged(load) * mt < H * (1.0 + 1e-9) else brush.charged(load) * mt / H)   # сколько гребень уже прошёл по сырому и касаясь холста, px; п. 13в: подушка — по тому, насколько кисть набрана (`Кисть.набрана`), а не по плёнке, которую она кладёт: у узкой кисти с густой краской плёнка тоньше, а набрана она так же полно
    RubIn, exchange = (np.ones((Nc, nr)), M.Exchange.for_(soft, H._GROWING_TOGETHER_BANK[3] * mt / max(H, 1e-9), Nc, nr, mt, ic, wet_g, ks_canvas, spacing_brush, W, clean)) if takes is not None else (None, None)   # 4в-2: почти пустая мягкая кисть в толстом сыром слое меняется с ним краской
    bank = Bank(Nc, with_, H._GROWING_TOGETHER_BANK[3] * mt / max(H, 1e-9), W / spacing_brush) if takes is not None and not palette_knife else None
    streams = Streams.for_(rng, with_, furrow, N, ds, W, thickly, bank, ic)
    edge = None if streams is None else Edge.for_(has_paint, rng, with_, N, Nc, ds, W, thickly, spacing_brush, touch_brush, wet_g / mt, share_pressure, reserve_g, ic)   # 6а-2: края по сырому (свой поток случая)
    # краски гребней
    shares_gr = _strands_paints(paints_r, layout, with_, Nc, spacing_brush, rng_uz, W) if narrow_charge else None   # п. 10: краски прядями ворса
    numbers = shares_gr.argmax(axis=2) if narrow_charge else _layout(paints_r, layout, with_, Nc, spacing_brush, rng, thickly if layout in ('полосы', 'ряд') else 0.0, bands[0] if charge and bands else None)
    tone_gr = with_.tone
    if charge:
        # краски набора взяты с палитры чистыми: тон пряди гуляет меньше, чем у размешанной смеси (её пестрота —
        # от недомеса), и дальше ±8 % не уходит — иначе треть полосы выходила другой краской, чем просили
        lt = 0.6 * np.log(with_.tone)
        sr = lt.mean(axis=1, keepdims=True)
        tone_gr = np.exp(lt + clamp(sr, -0.08, 0.08) - sr)
    ks_paints = np.stack([in_ks(clamp(c[None, :] * tone_gr, 0.0, 1.0)) for c, _ in paints])   # (n, nr, 24)
    one, gr = len(paints) == 1, np.arange(nr)
    # несмешанные густые краски на кисти: полосы лежат каждая на своих прядях. На стыке двух красок кисть
    # набрана беднее (макали порознь) — под волоском стыка борозда шире и глубже, местами до лежащего слоя:
    # тёмная щель между полосами. Краски мешаются только на стыке и только когда кисть прижата
    joints, overflowed = _joints(numbers[0], with_, rng_b, thickly, len(paints), charge, p[ic], spacing_brush, W) if not one and thickly > 0 and not palette_knife and nr > 1 else (None, None)
    ks_own = ks_paints[0] if one else (np.einsum('cnq,qnk->cnk', shares_gr, ks_paints) if narrow_charge else None)   # узкий набор: (Nc, nr, 24) — что гребень несёт из запаса
    vein = thin = by_place = None                                           # прожилка и тонкое место в часть гребня (п. 2д)
    if charge and nr > 2:
        vein, ks_vein, break_off, thin, by_place = _charge(numbers[0], with_, path, ic, p, rng_in, W, len(paints),
                                                          [d for _, d in paints], rng_d, *((bands[1], np.cumsum(consumption * touch_brush.max(axis=1)) * spacing_brush) if bands else ()))
        ks_vein = ks_paints[ks_vein, gr]                     # (nr, 24)
    sc_brush = path['s'][ic] - path['s'][int(np.argmax(p > 0.02))]   # px пути от касания, на шаге переноса
    # бок живой (п. 2г): крайний гребень реже и беднее краской — его след рвётся, кромка отступает и возвращается
    edge_breaks = 1.0 - lives * (1.0 - _edge(with_, sc_brush, rng_g, W, *((1.0, 1.0) if charge else _EDGE_ONE[2:]), narrow=charge and W < 20.0)) if thick else None
    edge_places = None
    if thick and not charge:
        # тонкое место штрихом — и у одной краски (п. 2д): реже, чем у набора, — одна краска набрана ровнее
        thin = 1.0 - _ShTRAndCanvas[2] * lives * _streak(with_, sc_brush, rng_g, W, _STREAK_ONE) * (1.0 - _short_sweep(path, W))   # Н-2: короткий ход кладёт сытая кисть — запас гребня опустеть не успел (штрих в 9–15 px на ходу в полширины ложился щелью поперёк следа)
        # у одной краски крайняя прядь рвётся не всем гребнем, а крайними 2–3 px следа (кромка отступает и
        # возвращается; остальной гребень кладёт) — считается по месту на кисти, при укладке
        edge_places = (edge_breaks, clamp((rng_g.uniform(*_EDGE_ONE[:2], 2)[st_m] * float(clamp((W - 8.0) / 16.0, 0.3, 1.0)) - from_kr) / 0.6 + 0.5, 0.0, 1.0))
        edge_breaks = None
    elif thick and charge and W < 20.0:
        # узкая кисть в несколько красок (п. 2д): крайний гребень — пятая часть кисти, целиком он не рвётся;
        # кромка отступает на 1,3–2,3 px, как у одной краски
        edge_places = (edge_breaks, clamp((rng_d.uniform(*_EDGE_NARROW, 2)[st_m] - from_kr) / 0.6 + 0.5, 0.0, 1.0))
        edge_breaks = None
    # удар связан с лежащей краской (п. 2г): где кисть шла с нажимом от 0,5, подушка своей краски у края продавлена,
    # и крайний гребень (2–3 px бока) кладёт смесь с лежащим — кусками, не каймой: одна сторона удара теряется
    # чаще другой. На отрыве кончики, достававшие до лежащего, тянут его в конец удара. До 0,45 кромка чистая
    catches = None
    tsv_kr = None
    if takes is not None and thick and charge:
        strength = clamp((p[ic] - _BrushRResponseMBrushA[0]) / _BrushRResponseMBrushA[1], 0.0, 1.0)
        if strength.any():
            st_g = (with_.center > 0).astype(np.intp)
            teryaetsya = rng_g.permutation([0.35, -0.25])            # одна сторона достаёт до лежащего чаще
            pieces = clamp((_noise(rng_g, Nc, 2, float(clamp(0.45 * W, 8.0, 22.0)) / spacing_brush) + teryaetsya[None, :]) / 0.2 + 0.5, 0.0, 1.0)
            pieces = pieces * pieces * (3.0 - 2.0 * pieces)
            at_lift_off = clamp(1.0 - (float(sc_brush[-1]) - sc_brush) / (0.3 * W), 0.0, 1.0) ** 2 * float(strength.max())
            wet = clamp(wet_g / (0.06 * mt), 0.0, 1.0)
            # смесь у кромки считается по месту на кисти (крайние 3 px следа), а не по гребню: гребень шире
            catches_kr = _BrushRResponseMBrushA[2] * thickly * strength[:, None] * pieces[:, st_g] * wet
            catches = _BrushRResponseMBrushA[2] * thickly * 0.8 * at_lift_off[:, None] * rng_g.random(nr)[None, :] ** 1.5 * wet
            tsv_kr = np.empty((Nc, nr, 2 * PResponseLResponseWith))
    Hg = H * with_.load                                          # толщина плёнки по гребням
    lambda_ = max(brush.carry_out * W, 4.0)
    sliv = 1.0 - np.exp(-spacing_brush / lambda_)
    # капля сытой кисти; у краёв кисть держит меньше
    full = Hg * lambda_ * (1.0 if palette_knife else H.carry_at_edges(from_edges)); full = full * float(clamp(H._WithRYIn[10] * sparingly.widths, H._WithRYIn[11], 1.0)) if sparingly is not None else full   # п. 17: у скупо набранной кисти и на кончике краски меньше — садится она плотно, а хватает её ненадолго
    drop = full * (1.0 + live * H._DROP_NARROW[3]) if breaks is None else breaks.drop(full)   # п. 10: узкая кисть садится каплей; нагрузка 'сухая' — с тем, что осталось
    ks_drops = (ks_own[0] if narrow_charge else ks_paints[numbers[0], gr]).copy()
    if brush._remainder is not None:
        # грязная кисть: на кончиках осталось подобранное прошлым мазком
        place = clamp(((with_.center / W + 0.5) * _BrushResponseRZAndN).astype(np.intp), 0, _BrushResponseRZAndN - 1)
        ks_rest, share_rest = brush._remainder
        g = (0.85 * share_rest[place])[:, None]
        ks_drops = ks_drops * (1.0 - g) + ks_rest[place] * g
    charge = charge0 = _PEREGRAtZ * np.exp(0.10 * rng.standard_normal(nr) + 0.18 * rng.standard_normal() * (breaks is None or not breaks.dry)) * (1.0 if breaks is None else breaks.charge); charge = charge0 = sparingly.charge(np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(171,))), nr) if sparingly is not None else charge0   # полупустая кисть отжата: набрано, сколько сказано; п. 17: свой поток случая
    reserve_px = sparingly.reserve(with_, W) if sparingly is not None else brush.reserve * (with_.reserve * W * (32.0 * H._MM / W) ** 0.5 * _narrow_paint(live, W) if breaks is None or not breaks.dry else breaks.reserve(with_.reserve, with_.half_width) * 32.0 * H._WithAtCanvasAYa[7])   # полупустая кисть набрана кончиком: на гребне краски поровну у узкой и широкой
    place_g, drop0 = clamp(((with_.center / W + 0.5) * _BrushResponseRZAndN).astype(np.intp), 0, _BrushResponseRZAndN - 1), drop; zhk = None if loading is None or clean else _full_brush(brush, H, Hg, reserve_px, drop, with_, W, live, sparingly, breaks); charge, charge0, drop = (charge, charge0, drop) if zhk is None else _with_loading(loading, place_g, zhk, drop, Hg * reserve_px)   # Д-1: живая краска — ход начинает с остатка; Д-1-2: в объёме полной кисти, а не в долях случайного набора хода
    # срыв: кисть уходит с холста под нажимом — капля остаётся сгустком
    dp = np.gradient(p, ds)[ic]
    breakaway = np.maximum(clamp(-dp / 0.06 - 0.35, 0.0, 1.0), fast[ic]) * (0.25 + 0.75 * brush.viscosity)
    dkas = np.diff(touch_brush, axis=0, prepend=touch_brush[:1])
    dkas[-1] -= touch_brush[-1]                                     # оборванный мазок: отрыв в последней точке
    agrees = np.maximum(-dkas, 0.0)
    if p[-1] > 0.05:
        breakaway = breakaway.copy()
        breakaway[-1] = max(breakaway[-1], 0.25 + 0.75 * brush.viscosity)

    Out, light, belly = np.empty((Nc, nr)), (_saturated_gives_back(live, W, p[ic]) if live > 0 else None), _opened(brush, W, live)                                # толщина, которую гребень кладёт (на единицу касания)
    Saturated = np.empty((Nc, nr))                                 # насколько гребень кладёт сплошь (0 — пропуск)
    Clot, (Buildup, Buildup0) = np.zeros((Nc, nr)), ((None if g is None else _swell(path['s'], p, speed, W, thickly)[ic][:, None] * (touch_brush * g[ic]) ** 3 * (Hg * spacing_brush)[None, :]) for g in (patch_g, patch_g0))   # сгусток на отрыве: объём на единицу ширины; п. 18: и наплыв остановленной гружёной кисти (п. 19: кладёт `_вал_конца`); Д1, строка 5: `Напл0` — наплыв при ровном овале: столько краски у пряди и остаётся
    Tsv, release = np.empty((Nc, nr, 2 * PResponseLResponseWith)), M.released(soft, ic)
    for c in range(Nc):
        k = touch_brush[c]; ko = k * release[c]                          # К5: отпущенная кисть ("load_along") краску с ворса не тратит — подача, расход запаса и сход капли идут той же отдачей
        # своя краска из запаса в каплю
        # пока кончик полон подобранной краской, своя из запаса не идёт
        has = _narrow_paint(live, W, charge=charge, full=charge0, belly=belly) * clamp(2.2 - 1.2 * drop / full, 0.0, 1.0) if breaks is None or not breaks.dry else breaks.feeds(drop / full, charge / charge0, Hg * (release[c] * spacing_brush) * k / full, spacing_brush); has = sparingly.feeds(charge / charge0, drop / full) if sparingly is not None else has   # п. 10, инструмент 2: ворс полупустой кисти каплю не наливает; п. 17: кончик кормится в меру остатка запаса
        if takes is not None:
            # кончик идёт по подушке своей краски: до лежащей достаёт, когда нажим её продавил
            # подушка считается от набранного и от пути в ширинах, а не от капли: капля узкой и широкой
            # кисти пустеет по-разному (у узкой почти все гребни крайние), и порог уходил за шириной
            pad = Hg * full * (_PAD_0 - _PAD_SWEEP * clamp(by_canvas / (5.0 * max(W, H._ResponseWithTRAYa[0])), 0.0, 1.0)) / (0.9 * mt)
            # волоски держит одна обойма: кисть лежит на своей краске вся, бедная прядь — на подушке соседей
            shared = float((pad * k).sum() / max(k.sum(), 1e-6))
            pad = clamp((0.25 * pad + 0.75 * shared) * at_edges_vt, 0.0, 1.2)
            threshold_vt = M.rubs_in(soft, _THRESHOLD_RUB_IN + _PResponseDAtShBrushA * pad - _SWEEP_RUB_IN * clamp(by_wet / (10.0 * max(W, H._ResponseWithTRAYa[0])), 0.0, 1.0), _PResponseDAtShBrushA * pad, wet_g[c] / mt)   # п. 10: подушку узкой кисти ход выдавливает за те же px, что у кисти 16; 4в: полная мягкая кисть ложится своей краской в лежащий сырой слой и под лёгкой рукой
            vt = clamp((pressure_vt[c] - threshold_vt) / _SOFT_RUB_IN, 0.0, 1.0)
            vt = vt * vt * (3.0 - 2.0 * vt)
            RubIn[c] = vt
            by_wet += spacing_brush * k * (wet_g[c] > 0.02 * mt)
            by_canvas += spacing_brush * k
            takes[c] *= release[c] * (vt if edge is None else vt * edge.takes[c])   # К5: отпущенная кисть и не снимает; 6а-2: севший в сырое кончик снимает её не сразу
            if in_wet is not None:
                has = has * (1.0 - (1.0 - in_wet[c]) * vt)
        own = Hg * (release[c] * spacing_brush) * ko * has * (np.exp(-light[c] * _saturated(W, charge, charge0, belly)) if live > 0 else 1.0)   # п. 10: сытый кончик отдаёт краску и под лёгкой рукой
        if edge_breaks is not None:
            own = own * edge_breaks[c]                        # бедный гребень у края: где не кладёт, там и не набирает
        if zhk is not None and breaks is None: own = own * clamp(charge / np.maximum(charge0 * H._ZAGRAtZBrushA[0], 1e-12), 0.0, 1.0)   # Долг-ф: кончик кормится из ворса в меру остатка — тем же законом, что доливается в воздухе (`_с_загрузки`): полно, пока в ворсе больше `_ЗАГРУЗКА[0]` полного, дальше — меньше; иначе ворс, что делится по всей кисти (`_в_обойме`), отдавал до дна разом и исход обрывался
        if zhk is not None: own = np.minimum(own, charge * (Hg * reserve_px))   # Д-1-2: из ворса на кончик — не больше, чем в ворсе есть
        ks_with = M.Exchange.feed(exchange, c, ks_own if one else (ks_own[c] if narrow_charge else ks_paints[numbers[c], gr]), ks_drops, k * spacing_brush, own)   # 4в-3: в залитом лежащей краской пучке на кончик приходит она
        charge = _ferrule(breaks, np.maximum(charge - ((consumption[c] * spacing_brush) * ko * has * M.Exchange.consumption(exchange) / reserve_px if zhk is None else own / np.maximum(Hg * reserve_px, 1e-12)), 0.0), Hg * reserve_px, spacing_brush); charge = _Breakaway.share(charge, Hg * reserve_px, spacing_brush) if sparingly is not None else _in_ferrule(charge, charge0, Hg * reserve_px, 2.0 * with_.half_width * (1.0 if with_.has is None else with_.has), spacing_brush) if zhk is not None and breaks is None else charge   # v3.23: у полупустой кисти краска запаса расходится по соседним гребням; Долг-ф: и у живой — ворс в одной обойме (`_в_обойме`), кисть пустеет вся, а не серединой (рельсы)
        if takes is not None:
            takes[c] /= np.inf if clean else 1.0 + 0.3 * drop / full   # сытая кисть берёт чуть меньше; 5а: вытертая с холста не уносит — только меняется краской со слоем (`Обмен`)
            if streams is not None:                             # 6а-1: в струе лежащая остаётся; ношей — сколько сойдёт в струи; в запасе — и подобранное
                takes[c], ks_with = streams.holds(c, takes[c], bank.carry, Hg * spacing_brush), bank.carry.from_reserve(own, ks_with, Hg * reserve_px * charge)
            foreign = takes[c] * k * spacing_brush
            new = drop + own + foreign
            if bank is not None:                               # 6а (`срастание.py`): лишнее подобранное — валом за кромку; часть — ношей на кончик
                new, foreign = bank.spill(c, new, full, foreign, ks_canvas[c], None if streams is None else streams.place)
            ks_new = (ks_drops * drop[:, None] + ks_with * own[:, None]
                        + ks_canvas[c] * foreign[:, None]) / np.maximum(new, 1e-12)[:, None]
        else:
            new = drop + own
            ks_new = (ks_drops * drop[:, None] + ks_with * own[:, None]) / np.maximum(new, 1e-12)[:, None]
        # пустой кончик краску не меняет: иначе у отошедшего гребня она считалась «никакой» (белой),
        # и конец узкого удара кончался белой точкой
        ks_drops = np.where((new > 1e-9)[:, None], ks_new, ks_drops)
        drop, ks_drops = M.share(soft, new, *((ks_drops, full) if exchange is None else exchange.spacing(c, wet_g[c], p[ic[c]], k * spacing_brush, new, ks_drops, ks_canvas[c], full, own))); drop, ks_drops = sparingly.drops(drop, ks_drops, full, spacing_brush) if sparingly is not None else (drop, ks_drops)   # 4а-2: соседние мягкие волоски делят каплю; 4в-2: в сыром слое — и лежащую
        if narrow_charge and nr > 1: _share_drop(ks_drops, k, spacing_brush, W, sc_brush[c])
        if joints is not None and overflowed[c] > 0:
            # прижатая кисть: соседние гребни разных красок делятся каплей — кромка полос втирается на гребень-два
            l = np.nonzero(numbers[c][:-1] != numbers[c][1:])[0]
            if len(l):
                g = overflowed[c] * np.minimum(k[l], k[l + 1])[:, None]
                a, b = ks_drops[l].copy(), ks_drops[l + 1].copy()
                ks_drops[l] = a + (b - a) * g
                ks_drops[l + 1] = b + (a - b) * g
        # сухая кисть: пока капля полна, кладёт сплошь; пустея — только там, где запас гуще
        # полная густая кисть кладёт сплошь: неровный запас и рельеф лежащего рвут след,
        # только когда запас на исходе
        breaks = 1.0 - 0.8 * thickly * clamp(charge / 0.6, 0.0, 1.0)
        m = (drop / full) * (1.0 + (uneven[c] - 1.0) * breaks)
        m += addition0[c] * (1.0 - live * _wider_point(W) * H._DROP_NARROW[12]) + relief[c] * breaks   # п. 10: мягкий кончик, пустея, проскакивает позже
        m = clamp(m * steepness, 0.0, 1.0) if breaks is None else breaks.ridge(c, m, breaks, drop / full)   # п. 5б: гребень кладёт на бугорки нитей под собой
        if vein is not None:
            m = m * break_off[c]                                  # запас полосы кончился или прядь сорвалась
        if edge_breaks is not None:
            m = m * edge_breaks[c]
        Saturated[c] = m = m * 0.0 if clean else m if loading is None else m * clamp(drop / (H._ZAGRAtZBrushA[2] * full), 0.0, 1.0)   # 5а: своей плёнки вытертая кисть не кладёт; Д-1: и гребень с пустым кончиком (полупустую кисть по зерну кормили бугорки грунта и без краски)
        # пустея, гребень кладёт не тоньше, а уже и реже: сухой след — резкие линии, а не дымка (п. 10: у узкой кисти в одну краску пол толщины ниже, `_КАПЛЯ_УЗК[11]`: плёнка к концу хода тоньшает, иначе след по сухому — наклейка)
        Out[c] = Hg * clamp(drop / full, 0.45 * (1.0 - live * _wider_point(W) * H._DROP_NARROW[11]), 1.6) * (thickness[c] * layer_from_pressure[c]) * (1.0 if breaks is None else breaks.thicker)
        if thin is not None:
            Out[c] *= thin[c]
        agrees_now = Out[c] * (spacing_brush * ko * m)
        if zhk is not None and breaks is None:                  # Д-1-2: след кладёт на холст больше, чем сходит с кончика (`заполнен` ниже: касание и корень сытости, а не их произведение) — рвущийся след брал краску из ничего; у живой краски разницу отдаёт ворс, а пустой ворс — кончик
            past = Out[c] * (spacing_brush * release[c] * k ** (1.0 - 0.75 * thickly) * np.sqrt(m)) - agrees_now; from_brush_hair = np.minimum(past, charge * (Hg * reserve_px)); charge, agrees_now = np.maximum(charge - from_brush_hair / np.maximum(Hg * reserve_px, 1e-12), 0.0), agrees_now + (past - from_brush_hair)   # Р-1: ворс пустеет до нуля, не ниже — у почти пустого гребня округление давало заряд −1e-49, с ним отрицательную каплю и вес ноши в струе, а деление на него — цвет 1e221 (белый прямоугольник)
        agrees_now = np.minimum(agrees_now, drop)
        Tsv[c] = ks_drops
        if bank is not None and bank.carry is not None:        # 6а: подобранное сходит с кончика первым (тонкая краска по сырому — в струи, 6а-1)
            Tsv[c], agrees_now = bank.fade_out(c, agrees_now, ks_drops, streams, RubIn[c] * clamp(wet_g[c] / (0.06 * mt), 0.0, 1.0))
        drop, ks_drops = (drop - agrees_now, ks_drops) if streams is None else bank.carry.in_drop(drop - agrees_now, ks_drops)   # 6а-1: несошедшая в струи ноша мешается в каплю
        if breakaway[c] > 0 and not clean:
            sg = drop * (agrees[c] * breakaway[c]) * (1.0 if breaks is None else breaks.hangs(drop / full))   # на полупустом кончике капля не висит
            Clot[c] += sg
            drop = drop - sg * release[c]
        if vein is not None:
            # прожилка: гребень несёт поверх своей краски мазок соседней — кладёт её, в каплю не мешая
            g = vein[c][:, None]
            Tsv[c] = Tsv[c] * (1.0 - g) + ks_vein * g
        if catches is not None:
            g = catches[c][:, None]
            Tsv[c] = Tsv[c] * (1.0 - g) + ks_canvas[c] * g
            g = catches_kr[c][:, None]
            tsv_kr[c] = Tsv[c] * (1.0 - g) + ks_canvas[c] * g
    # что осталось на кончиках — в память кисти (следующий мазок начнётся с этого)
    brush._charged = 0.0 if clean else float(((Hg * reserve_px * charge0 + Hg * lambda_ * H.carry_at_edges(from_edges)) * 2.0 * with_.half_width * (1.0 if with_.has is None else with_.has)).sum()); M.Exchange.remember(exchange, brush, with_.center / W, ks_drops); M.Exchange.balance(exchange, brush, Hg * reserve_px, charge0, charge, Hg * lambda_ * H.carry_at_edges(from_edges), 2.0 * with_.half_width)   # 4в-4: баланс краски кисти

    if zhk is not None: brush._consumption = _in_loading(place_g, charge / np.maximum(charge0, 1e-12), drop / np.maximum(zhk[1], 1e-12), with_.has)   # Д-1: с этим начнётся следующий ход; Д-1-2: убыль ворса — объём, пришедший на кончик
    # --- с шага переноса на шаг пути (линейно) и на выборку профиля
    if_ = np.arange(N) / _ShBrush
    i0 = np.minimum(if_.astype(np.intp), Nc - 1)
    i1 = np.minimum(i0 + 1, Nc - 1)
    f1 = (if_ - i0)[:, None]
    f0 = 1.0 - f1
    bead_m = np.ones(nm)
    if thickly > 0 and not palette_knife and brush.bead > 0:
        # краску выдавливает на одну сторону: на внешнюю сторону дуги, а на прямом ходе — куда наклонена кисть
        curvature = float(np.mean(np.gradient(path['ход'], ds)))
        sign = -np.sign(curvature) if abs(curvature) * W > 0.05 else (1.0 if rng.random() < 0.5 else -1.0)
        to_edges = (W / 2 - sign * with_.u)
        bead_m = 1.0 + brush.bead * 1.6 * thickly * np.exp(-to_edges / 1.6)
    elif palette_knife:
        to_edges = W / 2 - np.abs(with_.u)
        bead_m = 1.0 + brush.bead * 2.0 * np.exp(-to_edges / 1.5)
    if thickly > 0 and not palette_knife and brush.bead > 0:
        # выдавленное нажимом уходит в валик — на ту же сторону (обводки с двух сторон нет)
        vydavleno = thickly * clamp(_LAYER_PRESSURE * (p - 0.3), 0.0, 0.5)
        bead_m = bead_m[None, :] + vydavleno[:, None] * (2.0 * brush.bead * np.exp(-to_edges / 1.6))[None, :]
    else:
        bead_m = bead_m[None, :]
    g = with_.ridge
    saturated = (Saturated[i0] * f0 + Saturated[i1] * f1)
    # (N, nm): гребень сужается, когда едва касается или пустеет; густая краска касается холста раньше
    # волоска — едва коснувшаяся полная кисть кладёт гребень почти во всю ширину, а не серую дымку
    filled = ((touch_at ** (1.0 - 0.75 * thickly) * np.sqrt(saturated))[:, g] if breaks is None else breaks.filled(touch_at ** (1.0 - 0.75 * thickly), Saturated, i0, i1, f0, f1, g)) * (1.0 if exit_g is None else exit_g[:, g])   # п. 5б: по выборкам профиля
    mask = M.mask(soft, (touch_at ** (1.0 - 0.75 * thickly))[:, g], filled, with_.x, lambda: clamp((filled * 1.2 - with_.x[None, :]) * 10.0, 0.0, 1.0) if breaks is None else breaks.ridge(filled, with_.x) if not breaks.dry else breaks.hair(filled, (touch_at ** (1.0 - 0.75 * thickly))[:, g], with_.x, X, Y, ds, p, drags, W))   # пятно касания у концов множится ниже, после `вых`; п. 18: пятно касания у концов — на укладке, перенос краски в теле прежний; п. 10, инструмент 2: у полупустой кисти кладёт волосок, а не гребень; 4а-2: бедный мягкий гребень кладёт тоньше, а не уже
    out = (Out[i0] * f0 + Out[i1] * f1)[:, g]
    mask = _patch(mask, out, patch_m, patch_m0, path['s'], W)
    if tuft is not None: mask = tuft.mask(mask)
    if with_.has is not None: mask = mask * with_.has[g][None, :]   # К-14б: где волоса нет, краска не ложится
    width = W / nm; path_m = M.by_path(soft, X, Y, with_.u, ds)      # К5, "load_along": площадь следа отсчёта — по пути волоска; без поля — 1
    # лёгкая кисть борозду не продавливает до дна: глубина борозд идёт с нажимом
    fine = clamp(0.45 + 0.65 * p / 0.85, 0.0, 1.0)[:, None]
    profile = profile_furrows(with_, furrow, fine)
    # след кладётся ядром в пиксель с лишним: чтобы он вышел в ширину кисти, самый край кисти краски не кладёт
    # (у густой: тонкая плёнка у края сходит на нет сама)
    edge_m = np.ones(nm) if palette_knife else 1.0 - thickly * (1.0 - clamp((W / 2 - _YaDRResponse / 2 - np.abs(with_.u)) / 0.5 + 0.5, 0.0, 1.0))
    if thickly > 0 and not palette_knife and nr > 2:
        # крайний гребень уже двух пикселей густой краски не несёт: за бороздой крайнего волоска нитки-обводки нет
        for kr in (0, nr - 1):
            if 2.0 * with_.half_width[kr] < 2.0:
                edge_m = edge_m * (1.0 - thickly * (g == kr))
    if angle_m is not None:
        mask = mask * angle_m
    if landing_m is not None:
        mask = mask * (1.0 + ((landing_m if breaks is None or not breaks.dry else breaks.irregular(landing_m, p, ds, W)) - 1.0) * laid_flat)   # v3.21: у полупустой кисти посадка и подъём — по волоску
    if edge_places is not None:
        breaks_m = 1.0 - (1.0 - (edge_places[0][i0] * f0 + edge_places[0][i1] * f1)[:, g]) * edge_places[1][None, :]
        mask = mask * breaks_m
    D = out * mask * profile * (bead_m * (width * ds)) * edge_m[None, :] * (1.0 if thin_g is None else thin_g.less_thick(with_, uneven, thickness, i0, i1, f0, f1, spacing_brush, None if drop is None or breaks is not None else lambda zh: drop[clamp(np.floor(Y[zh] - by0).astype(np.intp), 0, drop.shape[0] - 1), clamp(np.floor(X[zh] - bx0).astype(np.intp), 0, drop.shape[1] - 1)])) * (exit_response[:, None] if live > 0 else 1.0) * (1.0 if soft is None else soft.lays[:, g] * path_m) * (1.0 if breaks is None else breaks.with_ridge())   # п. 13: гребень лежащей краски снимает с волоска полупустой кисти больше; мягкая кисть: слабо прижатый волос оставляет плёнку тоньше; п. 10: собранный к оси ворс кладёт краски по своей ширине, а не гуще
    # борозда цветом (п. 5, 5в): тон идёт за живой глубиной борозды — у тонкой краски. Пастозная кроет телом, её гребень и дно одного тона
    tone_furrow = (tone_furrows(with_, furrow, profile, D / (H * width * ds), 1.0 - np.exp(-D / (tau * width * ds)), brush.spread_tone * (1.0 - thickly), ds)
               if furrow is not None and thickly < 1.0 else None)
    if by_place is not None:
        D = D * (1.0 - _ShTRAndCanvas[2] * (by_place[2][i0] * f0 + by_place[2][i1] * f1))
    if joints is not None:
        for ks, shch, floor in zip(*joints):
            if shch > 0:
                t = clamp(np.abs(with_.u - (with_.center[ks] + with_.half_width[ks])) / floor, 0.0, 1.0)
                D = D * (1.0 - min(1.7 * shch, 1.0) * (1.0 - t ** 4))[None, :]
    if sides is not None:
        # поджатая прядь краску не копит: что не легло, осталось на кисти — обводки с двух сторон нет
        D = D * (1.0 - (sides[:, :1] / (anchor_l + with_.half_width)) * (with_.u < anchor_l)[None, :]
                 - (sides[:, 1:] / (with_.half_width - anchor_p)) * (with_.u > anchor_p)[None, :])
    D = _bead_sides(gap.open(D, with_, width, tau * ds, None if breaks is None else breaks.touched), with_, brush, p, spread, Out, Hg, uneven, i0, i1, f0, f1, ds, W, thickly, soft is not None or clean or sparingly is not None or _dry(brush, load) or turbidity < 1.0)   # п. 19: валик по бокам (текучая краска его ровняет сама, `_текучая`)
    clot = Clot[i0][:, g] * (width / _ShBrush) * (1.0 if patch_m is None else patch_m)                       # сгусток — на шаги своего отрезка поровну
    if Buildup is not None and Buildup.any(): clot = clot + _bank_end(_by_oval(Buildup[i0][:, g] * (width / _ShBrush) * patch_m, None if patch_m0 is None else Buildup0[i0][:, g] * (width / _ShBrush) * patch_m0), ends, with_, path['s'], _along(lag_g, along_brush * drags, ferrule, limit), width, ds, W)   # п. 19: наплыв — валом у кромки остановки
    if clot.any():
        D += M.released(soft, clot) * (0.6 + 0.4 * with_.profile)[None, :]   # К5: сгусток и наплыв сходят с кисти той же отдачей
    Xl, Yl = X - bx0, Y - by0
    comma_no = not H._LUMP_ON or clean or sparingly is not None or _dry(brush, load) or breaks is not None or brush.dryness > 0 or palette_knife or (soft is not None and (soft.release is not None or (thickly <= 0.0 and has_wet)))             # п. 21, Д1: ком оседает и у не кроющей краски (пропуск считается ниже по осевшей плёнке — идёт с краской), и у мягкой кисти; полупустая кисть кома не несёт — её рельеф и есть след по зерну; мастихин — как был. Мягкая кисть с тонкой краской по сырому (Дв105): густо она кладёт не свою краску, а снятую сырую — жилами по прядям, в десятки толщин своей плёнки, посреди короткого хода; столба над сырым слоем это не ставит, а оседание уводило бы лежащую краску из пятна (ореол: тон ядра 0,925 → 0,891); К5, "load_along": рука отпускает кисть по ходу — кома на конце нет, а самое толстое место хода, который ложится сам на себя (спираль), — не конец
    ii, jj = np.nonzero((D > 0.0) | (touch_at[:, g] > 0.0))
    if len(ii) == 0 or (clean and takes is None):           # 5а: вытертая кисть по сухому и по голому грунту не меняет ничего
        return _empty(why, 'чистая' if clean else 'пусто')
    xs, ys = by_samples(Xl, ii, jj), by_samples(Yl, ii, jj)
    d = by_samples(D, ii, jj)
    colors = from_ks(Tsv)                                         # (Nc, nr, 3)
    if thin is not None and charge:
        # краска набора не размешана и вдоль волоска: тон гребня плавно гуляет по ходу (без точек: тонкое место — в `_набор`)
        colors = clamp(colors * np.exp(_TONE_SWEEP * _noise(rng_in, Nc, nr, max(0.4 * W, 8.0) / spacing_brush))[:, :, None], 0.0, 1.0)
    ts = by_ridge(colors, i0, g, ii, jj) * f0[ii] + by_ridge(colors, i1, g, ii, jj) * f1[ii]
    if tone_furrow is not None:
        ts = clamp(ts * by_samples(tone_furrow, ii, jj), 0.0, 1.0)
    if by_place is not None:
        # прожилка в часть гребня: поверх своей краски гребень несёт нитку соседней
        g = (by_place[0][i0[ii], jj] * f0[ii, 0] + by_place[0][i1[ii], jj] * f1[ii, 0])[:, None]
        ts_veins = np.stack([clamp(c[None, :] * tone_gr, 0.0, 1.0) for c, _ in paints])[by_place[1][jj], g[jj]]
        ts = ts * (1.0 - g) + ts_veins * g
    if catches is not None:
        zone = clamp((_BrushRResponseMBrushA[3] - from_kr) / 0.8, 0.0, 1.0)[jj][:, None]      # крайние 3 px следа
        colors_kr = from_ks(tsv_kr)
        ts = ts * (1.0 - zone) + (by_ridge(colors_kr, i0, g, ii, jj) * f0[ii] + by_ridge(colors_kr, i1, g, ii, jj) * f1[ii]) * zone
    touch_m = clamp((touch_at[:, g] * 1.2 - with_.x[None, :]) * 4.0, 0.0, 1.0) * edge_m[None, :] * (1.0 if patch_m is None else patch_m * exit_g[:, g])
    if with_.has is not None: touch_m = touch_m * with_.has[g][None, :]   # К-14б: и лежащего там ничто не трогает
    if angle_m is not None:
        touch_m = touch_m * angle_m
    if landing_m is not None:
        touch_m = touch_m * (1.0 + (landing_m - 1.0) * laid_flat)
    if edge_places is not None:
        touch_m = touch_m * breaks_m
    touch_m = M.released(soft, touch_m)[ii, jj]; pl_m = width * ds if np.ndim(path_m) == 0 else (width * ds) * by_samples(path_m, ii, jj)   # К5: и касание, и съём сырого — на ту же площадь
    d, ts, edge, edge_in, exchange_m = M.Exchange.in_trace(exchange, ts, lambda: (lambda tskh: by_ridge(tskh, i0, g, ii, jj) * f0[ii] + by_ridge(tskh, i1, g, ii, jj) * f1[ii])(from_ks(ks_canvas)), (d, ts, None, []) if streams is None else streams.in_film(d, ts, i0, i1, f0, f1, g, ii, jj, None if edge is None else edge.brush_film(touch_m, RubIn, _pointed(W, brush.point))), i0, i1, f0, f1, g, ii, jj, touch_m * pl_m * M.sweep_hair(soft, clean, profile, ii, jj))   # Г-3: разворошённое вытертым мягким волосом ложится по его бороздкам; 6а-1: в струе своей краски нет; 6а-2: у входа, бока и отрыва по сырому своя плёнка меняется местами с лежащей (`Край`)
    weights = [d, d * ts[:, 0], d * ts[:, 1], d * ts[:, 2], touch_m * pl_m]
    if takes is not None:
        removed = between_steps(takes, i0, i1, f0, f1, g, ii, jj) * touch_m * with_.removal[jj] * pl_m
        weights.append((removed if streams is None else streams.removed(removed, i0, i1, f0, f1, g, ii, jj, edge)) + exchange_m)   # в струе лежащая не снята, ноша ложится взамен неё; у бока съём — по ноше волоска
        weights += [between_steps(RubIn, i0, i1, f0, f1, g, ii, jj) * touch_m * pl_m] + edge_in + M.plough(soft, clean, touch_m * pl_m * M.sweep_hair(soft, clean, profile, ii, jj))   # Г-3: поле бороздок вытертого мягкого волоса — последним
    if bank is not None:
        xs, ys, weights = bank.in_raster(xs, ys, weights, with_, W, on_canvas, lag_g, i0, bx0, by0)
    xs, ys, weights = gap.in_raster(xs, ys, weights, with_, D, on_canvas, lag_g, tip, colors, i0, ax, ay, spread, bx0, by0, None if holds is None else spread0, along_canvas, qx * ax + qy * ay); xs, ys, weights = _solid(xs, ys, weights, X, Y, D, ii, jj)   # Д1б, строка 7: след кончика сплошной
    fields = _spatter2(xs, ys, bw, bh, weights)
    V, eps = (fields[0] if breaks is None else breaks.reached(fields[0])), 1e-9   # К6: в ложбину под лёгкой рукой краска не ложится
    ts_new = np.stack(fields[1:4], axis=-1) / np.maximum(fields[0], eps)[..., None]
    touch = clamp(fields[4], 0.0, 1.0)
    # втирание по пикселям: где кисть лежащего не достала, оно не тронуто
    rubbing_in = clamp(fields[6] / np.maximum(fields[4], 1e-6), 0.0, 1.0) if takes is not None else None
    touches = touch if rubbing_in is None else touch * rubbing_in

    # --- сырая краска под мазком: часть снята кистью (она уже в каплях), часть вмешана на месте
    def coverage_from(h):
        return 1.0 - np.exp(-np.maximum(h, 0.0) / tau)

    # на голом месте краска, не собравшаяся в плёнку (едва заметные пиксели дальше 2 px от закрашенного),
    # не остаётся: за кромкой покрытие падает до нуля не дальше 2 px
    if soft is None: V = np.where(_beside_with_film(coverage_from(V) >= 0.25) | (Ao > 0.02), V, 0.0)   # у мягкой кисти тонкая плёнка края и есть след
    V, ts_new = (V, ts_new) if comma_no else _osel(V, ts_new, W, _zone_comma(clot, D, Xl, Yl, with_.u, bh, bw, W, V), one)                  # п. 21: ком у конца хода оседает по плёнке своего следа
    V_r = _relief_film(V, thickly, furrow)                       # рельеф плёнки; V — её пигмент: он красит и кроет

    if turbidity < 1.0 and thickly < 1.0 and not _dry(brush, load):
        V = V_r = _slump(V, ts_new, ho + sum(take(ns.thickness) for ns in under), 1.0 - thickly)     # п. 10, инструмент 3: густая и полупустая кисть не текут
    ho_original = ho                                                       # v4, 5д-2: поверхность сырой под ходом до него
    if even: V, V_r, went_h = _flowing(V, ts_new, W, even, tau, sum(take(ns.thickness) for ns in even_bottom[0]) * even_bottom[1], ho * wo_mobile * (Ao > 0.5) if has_paint else None, Co, touch, None if layer.flowing is None else take(layer.flowing), [c for c, _ in paints] if charge else None); ho = ho if went_h is None else ho - went_h   # К4: лежащая текучая плёнка отдала краску в шов; рядом — сырая краска своего слоя, что кроет (толщина без покрытия — оплывший рельеф, не плёнка)
    if has_paint: ho, wo, similar, Co = _razrovnyat(ho, wo, similar, Co, Ao, wo_mobile, touches, palette_knife, clean, M.by_hair(soft, clean, fields, fields[4]))   # кисть разравнивает сырой рельеф под собой (Ф-3, 5а, З-1)
    mixed_in = None
    if takes is not None:
        asked = fields[5]
        dostupno = wet_h if not bottom else ho * wo_mobile
        remove = np.minimum(asked, dostupno * 0.9)
        remainder_requests = asked - remove
        was = coverage_from(ho)
        h_before = ho
        ho, V, V_r = M.reduced(clean, ho, ho - remove if streams is None else streams.level(ho - remove, streams.under_brush(touches) * wo_mobile, _blur), remove, asked, V, V_r, touches * wo_mobile)   # 6а-1: струя — не гребень
        if clean: wo, similar, Co = _carries_wetness(h_before, ho, wo, similar, Co, Ao, True)   # З-1: остаток объёма, сведённый вытертой кистью по сырой краске под собой, несёт цвет места, откуда ушёл
        # где сырую плёнку сняли до дна — в борозде открывается то, что под ней
        Ao = Ao * clamp(coverage_from(ho) / np.maximum(was, 1e-6), 0.0, 1.0) ** 0.7
        for ns, hn, sn, cn in bottom:
            more = np.minimum(remainder_requests, sn * 0.9)
            remainder_requests = remainder_requests - more
            ns.thickness[canvas] = (hn - more)[local]
    V, V_r = (V, V_r) if takes is None else M.Exchange.not_from_nothing(exchange, clean, remainder_requests, fields[0], V, V_r); ts_new, ho = (ts_new, ho) if edge is None or clean else edge.on_place(fields[7], V, ho, wo_mobile, ts_new, Co)   # 6а-2: края по сырому — лежащая краска на месте; v4, 2а: чего с лежащего снять не вышло, то в след не легло
    if even and has_paint: ho = ho - np.minimum(ho * wo, V_r) * (even * clamp(V_r / np.maximum(V_r.max(), eps) * 4.0, 0.0, 1.0))   # п. 15: лежащую сырую краску своего слоя текучий ход вытесняет — второй ход внахлёст ложится в уровень с первым; v4, 5д: по плёнке `V_р`, не по пигменту
    V_r = _rim(V_r, thinner, np.maximum(ho_original - ho, 0.0) * wo if has_paint else None)              # v4, 5в: край плёнки спадает на длине в мм; 5д-2: убыль сырой под ходом плёнка доливает в уровень
    if has_wet and has_paint:
        mixed_in = np.minimum(ho, 0.3 * mt) * wo_mobile * (0.10 + 0.4 * brush.pickup) * touches
    if layer.transmittance is not None: _gap_behind_thickness(layer.transmittance, canvas, local, take(layer.thickness), ho)      # п. 10, инструмент 3 (3а-2)
    M, ts_top = _mix_in(V, mixed_in, ts_new, Co, eps)
    # --- толщина красит: тонкая плёнка просвечивает, густая кроет
    alpha = coverage_from(M if turbidity >= 1.0 else _covers(M, tau, turbidity)) * opacity
    co_wet = Co   # Ф-1: цвет лежащей краски до просвета — им красится плёнка без покрытия
    if turbidity < 1.0: Co = _on_gap(transmittance(), canvas, local, ts_top, M * opacity, turbidity, Co)     # п. 10, инструмент 3

    # --- запись в слой
    A_new = alpha + Ao * (1.0 - alpha)
    if even and not _dry(brush, load):      # Ф-2: текучая плёнка затекает в низины сухой лежащей краски — своего слоя и слоёв ниже
        below_h, dry_r = None, np.zeros((bh, bw))
        for ns in even_bottom[0]:                # как их сложит `Холст.рельеф`: под краской слоя выше нижнее ослаблено
            t_n = take(ns.thickness)
            below_h = t_n if below_h is None else below_h * (1.0 - (1.0 - even_bottom[1]) * take(ns.coverage)) + t_n
            dry_r = np.where(t_n > eps, 1.0 - take(ns.moisture), dry_r)
        if has_paint: dry_r = np.where(ho_original > eps, 1.0 - wo, dry_r)      # по толщине до хода: кисть ровняла её и за край лежащего
        if (dry_r > 1e-3).any():               # сухого под ходом нет (голый грунт, сырое) — как было
            V_r = _close(V_r, ho if below_h is None else ho + below_h * (1.0 - (1.0 - even_bottom[1]) * A_new), dry_r, even, thinner)
    if has_paint:
        C_new = (Co * (Ao * (1.0 - alpha))[..., None] + ts_top * alpha[..., None]) / np.maximum(A_new, eps)[..., None]
        h_new = ho + V_r
        w_new = np.where(h_new > eps, (wo * ho + V_r) / np.maximum(h_new, eps), 0.0)
        if even: h_new = _grew_together(h_new, w_new, V, even)       # п. 15
        # свежая краска не схвачена: схват слоя разбавляется её толщиной
        layer.degree_of_set[canvas] = np.where(h_new > eps, similar * ho / np.maximum(h_new, eps), 0.0)[local]
    else:
        C_new = np.where((alpha > 0)[..., None], ts_top, 0.0)
        h_new = V_r
        w_new = (V_r > eps) * 1.0
    layer.color[canvas] = _color_without_coverage(C_new, A_new, h_new, co_wet, ho, ts_top, V_r, eps)[local]
    layer.coverage[canvas] = A_new[local]
    layer.thickness[canvas] = np.maximum(h_new[local], 0.0); layer.moisture[canvas] = clamp(w_new[local], 0.0, 1.0)
    if even or layer.flowing is not None: _trace_flowing(layer, canvas, even, np.maximum(touch, V > 0.0)[local], alpha[local])   # К4
    if measurement is not None: measurement(dict(alpha=alpha[local], touch=touch[local], laid=float(V[local].sum()), wet=wet_h[local] if has_wet else None, removed=float((asked - remainder_requests)[local].sum()) if takes is not None else 0.0, x=x, y=y, W=W, angle=(cx0, cy0)))   # 3г (К-1): поля следа — ответу мазка (`ответ.Ответ`); слой и кисть уже записаны, здесь только чтение; снято — со своего слоя и нижних
    return (cx0, cy0, cx1, cy1)
