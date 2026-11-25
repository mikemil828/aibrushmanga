# READING VIEW of движок/кисть/нож.py with names in English. Not runnable; edit the original.
"""Мастихин (задание v4.86): нож — жёсткая гладкая кромка, а не ворс.

Счёт идёт по пикселям холста в порядке, в котором кромка ножа до них доходит: полоса хода в 1 px (`ряд`) — это
отрезок кромки, поделённый поперёк на места шириной около 1 px. Случайных чисел нет.

- **Опора.** Нож твёрдого не мнёт. Твёрдое — зерно грунта, слои ниже и высохшая доля краски своего слоя; сырая
  (подвижная) доля лежит на нём. Кромка встаёт на вершины твёрдого под собой (среднее верхних 15 % по кромке, самое
  высокое на длине подошвы `_НОЖ[17]` мм по ходу), прогибается к местному рельефу не больше `_НОЖ[6]` и стоит над опорой на зазор: зазор — толщина плёнки ножа
  (`нагрузка`; у ножа без краски — `_НОЖ[0]`) × множитель от нажима (`_зазор`): 0,7 — единица, слабее — толще,
  нажим 1 — только след `_НОЖ[1]`.
- **Срез.** Сырое выше кромки нож снимает: при нажиме до `_НОЖ[9]` оно идёт в вал перед кромкой (нож тянет краску по
  ходу), выше — на плоскость ножа и с холста уходит (срезать).
- **Вал** перед кромкой — краска на месте кромки: объём, цвет (К-М), влажность и схват. Из него краска ложится туда,
  где под кромкой пусто: не глубже `досяг` — зазор и то, что продавил нажим (`_НОЖ[4]`, `_НОЖ[5]`): слабый нажим садит
  краску на вершины, сильный снимает её с вершин и оставляет в низинах. Вал кормит запас ножа (`нагрузка`), пока
  он есть. Вдоль кромки вал растекается (`_НОЖ[10]`), с концов кромки выдавливается наружу — валик у бока хода
  (`_НОЖ[11]`). Где нож оторван, вал остаётся на холсте комом. Валик и ком стоят откосом `_НОЖ[13]`.
- **Плёнка под кромкой** мешается с валом (`_НОЖ[12]`): цвет одного места нож затягивает в соседнее.
- **Ребро** (`касание` 0): кромка шириной `_НОЖ[14]` мм — те же правила, след — линия с валиками.
- **Нож с краской плашмя** (Н-1): краска на ноже лежит горкой (`_НОЖ[20]`) — у концов кромки её меньше; лезвие к концам
  отогнуто от холста, нажим его распрямляет (`_НОЖ[27]`, `_НОЖ[28]`). Вал встаёт на холст там, где он, дожатый нажимом
  (`_НОЖ[24]`), достаёт до холста (`_НОЖ[21]`): нож садится серединой, след расходится на ширину с нажимом; край следа —
  там, где краски хватает до холста: над низиной зерна тонкий вал висит, на нити садится. Запас — полосами по местам
  кромки: у концов он кончается раньше. `касание` в ответе — где вал доставал до холста. Нож без краски считается как был.
- **Тело хода, начало и хвост** (Н-1-2). Ненажатая горка стоит под лезвием сама и вершиной достаёт до холста с первого
  касания (дальше оседает под нажимом): след начинается там, где нож коснулся холста, а не где он дожат. Место кромки,
  где краска на ноже кончается, держит вал тоньше (как весь нож — `_НОЖ[19]`), а не обрывом: край пустеющего ножа
  сходит без уступов. Край краски под ножом — крайние места кромки, чей вал стоит на холсте (при слабом нажиме он
  ближе к середине, чем конец лезвия). Из-под него паста выдавливается за бок по нажиму и зазору над холстом
  (`_НОЖ[23]`) и собирается комком: комок держится за бок ножа и едет с ним; холст стирает с него краску вершинами
  нитей (`_НОЖ[25]`; над низиной комок дна не касается) — валик у бока рвётся по зерну; комок, что перерос
  удерживаемое ножом (`_НОЖ[30]`; на высокой нити — раньше), остаётся на холсте целиком, тянется за ножом на три своих
  размера и стоит откосом кома — край выпирает; на ноже остаётся его доля `_НОЖ[31]`. Валик стоит откосом `_НОЖ[26]`.

Объём краски в слое ход без краски не меняет (кроме среза на нож при нажиме выше `_НОЖ[9]`); сдвинутая краска несёт
влажность и схват места, откуда пришла.
"""
import numpy as np

from .fast import clamp
from .paint import in_ks, from_ks, PResponseLResponseWith
from .numbers import _WithCanvasInAT
from . import numbers as H


def _gap(p):
    """Множитель зазора под кромкой от нажима (N,): 1 при `_НОЖ[2]`, 0 при нажиме 1, не больше `_НОЖ[3]`."""
    return clamp((np.maximum(1.0 - p, 0.0) / (1.0 - H._NResponseZh[2])) ** 1.2, 0.0, H._NResponseZh[3])


def _step(a):
    a = clamp(a, 0.0, 1.0)
    return a * a * (3.0 - 2.0 * a)


def _slope(ob, x, y, nx, ny, slope, bw, bh, fields, limit):
    """Краска, выдавленная за кромку, встаёт откосом: объём `об` (n,) из точки (x, y) в сторону (nx, ny) ложится
    треугольником длиной sqrt(2·об/уклон) (сечение на 1 px кромки; не длиннее `предел` px — рамка мазка), выборками через 0,5 px. `поля` — (n, k) величины
    на единицу объёма. Ответ: поле объёма (bh, bw) и поля величин × объём (bh, bw, k)."""
    V, E = np.zeros(bh * bw), np.zeros((bh * bw, fields.shape[1]))
    has = ob > 1e-12
    if not has.any():
        return V.reshape(bh, bw), E.reshape(bh, bw, -1)
    ob, x, y, nx, ny, fields = ob[has], x[has], y[has], nx[has], ny[has], fields[has]
    length = clamp(np.sqrt(2.0 * ob / slope), 1.0, max(limit, 1.0))
    K = int(np.ceil(float(length.max()) / 0.5)) + 1
    r = (np.arange(K) + 0.5) * 0.5
    weight = np.maximum(1.0 - r[None, :] / length[:, None], 0.0)
    weight = weight / np.maximum(weight.sum(axis=1, keepdims=True), 1e-12) * ob[:, None]
    ix = np.floor(x[:, None] + nx[:, None] * r[None, :]).astype(np.intp)
    iy = np.floor(y[:, None] + ny[:, None] * r[None, :]).astype(np.intp)
    ok = (ix >= 0) & (ix < bw) & (iy >= 0) & (iy < bh) & (weight > 0.0)
    q = (iy * bw + ix)[ok]
    V += np.bincount(q, weights=weight[ok], minlength=bh * bw)
    for k in range(fields.shape[1]):
        E[:, k] += np.bincount(q, weights=(weight * fields[:, k:k + 1])[ok], minlength=bh * bw)
    return V.reshape(bh, bw), E.reshape(bh, bw, -1)


def _smooth(a):
    """[1 2 1]/4 по обеим осям, объём тот же (за край рамки не уходит)."""
    for axis in (0, 1):
        b = 0.5 * a
        with_ = [slice(None)] * a.ndim
        l, p = list(with_), list(with_)
        l[axis], p[axis] = slice(1, None), slice(None, -1)
        b[tuple(l)] += 0.25 * a[tuple(p)]
        b[tuple(p)] += 0.25 * a[tuple(l)]
        brush0, brush1 = list(with_), list(with_)
        brush0[axis], brush1[axis] = slice(0, 1), slice(-1, None)
        b[tuple(brush0)] += 0.25 * a[tuple(brush0)]
        b[tuple(brush1)] += 0.25 * a[tuple(brush1)]
        a = b
    return a


def sweep_knife(path, W, H, clean, mt, paints, touch, hard_bottom, ho, wo, similar, Co, frame, widths=5.0, field=7.0):
    """Ход ножа в рамке мазка — `_ход` (доводы и ответ — там). Рамка мазка обрезана у края холста (`_ЗА_КРАЕМ`), а нож
    тратит краску и несёт вал и там, где шёл за ней: счёт идёт в рамке всего следа (за рамкой мазка — голый гладкий
    грунт), ответ — вырезка рамки мазка. Так ход от края холста тот же, что на большом холсте. След длиннее `_НОЖ[18]`
    площадей рамки (и 4 млн px) считается в рамке мазка как есть."""
    bx0, by0, bw, bh = frame
    x, y = path['x'], path['y']
    r = W / 2 + field + 2.0
    vx0, vy0 = min(bx0, int(np.floor(x.min() - r))), min(by0, int(np.floor(y.min() - r)))
    vx1, vy1 = max(bx0 + bw, int(np.ceil(x.max() + r)) + 1), max(by0 + bh, int(np.ceil(y.max() + r)) + 1)
    vw, vh = vx1 - vx0, vy1 - vy0
    if (vw, vh) == (bw, bh) or (vw * vh > H._NResponseZh[18] * bw * bh and vw * vh > 4_000_000):
        return _sweep(path, W, H, clean, mt, paints, touch, hard_bottom, ho, wo, similar, Co, frame, widths, field)
    cutout = (slice(by0 - vy0, by0 - vy0 + bh), slice(bx0 - vx0, bx0 - vx0 + bw))

    def wider(a):
        if a is None:
            return None
        b = np.zeros((vh, vw) + a.shape[2:])
        b[cutout] = a
        return b

    r = _sweep(path, W, H, clean, mt, paints, touch, wider(hard_bottom), wider(ho), wider(wo), wider(similar), wider(Co), (vx0, vy0, vw, vh), widths, field)
    if r is None:
        return None
    return {k: (v[cutout] if isinstance(v, np.ndarray) else v) for k, v in r.items()}


def _sweep(path, W, H, clean, mt, paints, touch, hard_bottom, ho, wo, similar, Co, frame, widths=5.0, field=7.0):
    """Ход ножа в рамке мазка (bx0, by0, bw, bh). путь — `путь._путь`; H — толщина плёнки ножа (h), чистая — нож без
    краски; мт — масштаб толщины; краски — [(линейный RGB, доля)]; касание — 1 плашмя, 0 ребро; ширин — на сколько ширин ножа набрано краски (`Кисть.запас`); поле — дальше стольких px от кромки валик и ком не идут; твёрдое_низ (bh, bw) —
    зерно грунта и слои ниже; hо, wо, схо (bh, bw), Cо (bh, bw, 3) или None — лежащая краска своего слоя.

    Ответ — None (нож холста не коснулся) или словарь: `h`, `w`, `сх` — лежащая краска после среза (до новой);
    `V`, `ц`, `wV`, `схV` — что легло из вала (толщина, цвет, влажность, схват); `мешано` — доля цвета вала в
    оставшейся плёнке и `ц_мешано` — его цвет; `касание` (bh, bw) — где шла кромка; `снято` — объём, срезанный с
    места; `на_ноже` — объём, ушедший с холста на нож; `сырое` — подвижная краска до хода."""
    bx0, by0, bw, bh = frame
    N = H._NResponseZh
    mm = H._MM
    x, y, p, angle, N = path['x'], path['y'], path['p'], path['угол'], path['N']
    ax, ay, tx, ty = -np.sin(angle), np.cos(angle), np.cos(angle), np.sin(angle)
    laid_flat = float(clamp(1.0 if touch is None else touch, 0.0, 1.0))
    We = max(W * laid_flat + N[14] * 8.0 * mm * (1.0 - laid_flat), 1.0)
    nb = max(1, int(round(We)))
    width = We / nb
    # --- какой пиксель когда и каким местом кромки пройден (первое касание)
    nu = max(2, int(np.ceil(We / 0.5)))
    uu = (np.arange(nu) + 0.5) * (We / nu) - We / 2
    ix = np.floor(x[:, None] + uu[None, :] * ax[:, None] - bx0).astype(np.intp)
    iy = np.floor(y[:, None] + uu[None, :] * ay[:, None] - by0).astype(np.intp)
    ok = (ix >= 0) & (ix < bw) & (iy >= 0) & (iy < bh) & (p >= N[7])[:, None]
    if not ok.any():
        return None
    q = (iy * bw + ix)[ok]
    spacing = np.broadcast_to(np.arange(N)[:, None], (N, nu))[ok]
    first = np.full(bh * bw, N, dtype=np.intp)
    np.minimum.at(first, q, spacing)
    pixel = np.nonzero(first < N)[0]
    row_p = first[pixel] // 2
    order = np.argsort(row_p, kind='stable')
    pixel, row_p = pixel[order], row_p[order]
    mp = first[pixel]                                           # доля пикселя под кромкой: у бока хода — часть (по месту его центра поперёк кромки)
    u_ts = (pixel % bw + 0.5 + bx0 - x[mp]) * ax[mp] + (pixel // bw + 0.5 + by0 - y[mp]) * ay[mp]
    covered = clamp(We / 2 - np.abs(u_ts) + 0.5, 0.0, 1.0)
    place = clamp(np.floor((u_ts + We / 2) / width).astype(np.intp), 0, nb - 1)
    rows, starts = np.unique(row_p, return_index=True)
    ends = np.append(starts[1:], len(pixel))
    n = len(pixel)

    # --- твёрдое и сырое
    w_mobile = (wo * (1.0 - _WithCanvasInAT * similar)).ravel()
    h_pl, w_pl, skh_pl = ho.ravel(), wo.ravel(), similar.ravel()
    wet_field = h_pl * w_mobile
    T = (hard_bottom.ravel() + h_pl - wet_field)[pixel]
    wet = wet_field[pixel]
    has_wet = bool(wet.max() > 1e-4) if n else False
    if clean and not has_wet:
        return None
    K = 2 * PResponseLResponseWith
    ks_lying = in_ks(Co.reshape(-1, 3)[pixel]) if Co is not None else np.zeros((n, K))
    w_lying, skh_lying = w_pl[pixel], skh_pl[pixel]

    # --- нож: зазор, запас, вал
    He = N[0] * mt if clean else H
    trace = N[1] * mt
    bank_full = N[8] * He * (8.0 * mm * N[15]) * width          # объём вала на место кромки: плёнка в `_НОЖ[8]` толщин на длине `_НОЖ[15]` мм
    length_roll = 8.0 * mm * N[15]
    E = np.zeros((nb, K + 3))                                 # вал: объём, влажность·объём, схват·объём, К-М·объём
    reserve = np.zeros(nb)
    ks_reserve = np.zeros((nb, K))
    reserve0 = max(float(H * float(N[16]) * float(widths) * W * width), 1e-12)
    mound = np.ones(nb)                                       # краска на ноже — горкой: к бокам кромки её меньше
    if not clean:
        t_brush = (np.arange(nb) + 0.5) / nb * 2.0 - 1.0
        mound = 1.0 - N[20] * laid_flat * t_brush * t_brush             # на ребре горки нет: краска — на самом ребре
        mound /= mound.mean()
        reserve[:] = reserve0 * mound
        shares = np.array([max(float(d), 0.0) for _, d in paints], dtype=np.float64)
        boundary = np.cumsum(shares / max(shares.sum(), 1e-12))
        whose = np.minimum(np.searchsorted(boundary, (np.arange(nb) + 0.5) / nb), len(paints) - 1)
        ks_reserve = np.stack([in_ks(np.asarray(c, dtype=np.float64)[None, :])[0] for c, _ in paints])[whose]
    slope = N[13] * mt / mm
    reach_pressure = N[4] * mt * clamp((p - N[5]) / (1.0 - N[5]) / 0.857, 0.0, 1.0) ** 2   # 0,857: полный досяг — при нажиме 0,9
    on_plane = _step((p - N[9]) / 0.1)
    contact = _step(p / 0.1)                                # едва касаясь, нож кладёт и срезает долей
    gap = He * _gap(p)                                     # под кромкой — плёнка ножа; срезая сырое, нож оставляет ещё и след

    cut_off = np.zeros(n)
    laid = np.zeros(n)
    ks_laid = np.zeros((n, K))
    w_laid, skh_laid = np.zeros(n), np.zeros(n)
    mixed = np.zeros(n)
    ks_mixed = np.zeros((n, K))
    on_knife = 0.0
    nR = int(rows[-1]) + 1
    side = np.zeros((2, nR, 6))                                # валик у концов кромки: объём, w·объём, сх·объём, RGB·объём
    taken_reserve = 0.0
    settled_last = np.zeros(nb)
    bead, share_bead, dropped = np.zeros((2, 6)), 1.0 / max(N[25] * 8.0 * mm, 1.0), np.zeros(nb, dtype=bool)
    stood = np.ones((len(rows), nb), dtype=bool)             # где вал ножа с краской доставал до холста: там нож холста и касался
    bulge = np.zeros(nb) if clean else N[27] * He * laid_flat * clamp((np.abs(t_brush) - N[28]) / (1.0 - N[28]), 0.0, 1.0) ** 2
    lumps = []                                               # (шаг пути, вал мест кромки, оставленный там комом)
    p0 = 0.0 if clean else N[21] * N[3] / (N[8] * float(mound.max()))   # ненажатый нож: горка стоит под лезвием сама и вершиной как раз достаёт до холста
    exit_b = [[], []]                                        # (ряд, комок 6), оставленный у бока целиком
    B_prev = N[30] * He * (8.0 * mm) ** 2                    # комок у бока, который ещё держит бок лезвия
    Tv = None if clean else hard_bottom.ravel() + h_pl - wet_field     # твёрдое: об него за краем краски стирается комок у бока
    edge_u = np.full((2, nR), np.nan)                        # где поперёк кромки шёл край краски под ножом (нож с краской)
    edges = None
    # опора: нож лежит на вершинах твёрдого не только вдоль кромки, но и подошвой по ходу (`_НОЖ[17]` мм) — в борозду
    # поперёк хода кромка не проваливается
    z_row = np.empty(len(rows))
    for n_r, (i0, i1) in enumerate(zip(starts, ends)):
        Tb = T[i0:i1]
        k = max(1, int(0.15 * len(Tb)))
        z_row[n_r] = float(np.partition(Tb, len(Tb) - k)[len(Tb) - k:].mean())
    half_sole = max(1, int(round(0.5 * N[17] * 8.0 * mm)))
    z_support = z_row.copy()
    for shift in range(1, half_sole + 1):
        z_support[shift:] = np.maximum(z_support[shift:], z_row[:-shift])
        z_support[:-shift] = np.maximum(z_support[:-shift], z_row[shift:])
    if len(z_support) > 2:                                        # на опору нож всходит и сходит с неё полого — на той же длине
        core = np.ones(2 * half_sole + 1) / (2 * half_sole + 1)
        z_support = np.convolve(np.pad(z_support, half_sole, mode='edge'), core, mode='valid')
    n_r = -1
    for b, i0, i1 in zip(rows, starts, ends):
        m = min(2 * int(b), N - 1)
        j = place[i0:i1]
        pb = float(p[m])
        pressed = laid_flat * max(min(pb / N[24], 1.0), p0 * (1.0 + min(pb / N[24], 1.0))) + (1.0 - laid_flat)                 # слабым нажимом горка краски до холста не дожата: касается её середина
        if not clean:                                        # запас ножа кормит вал
            ur = clamp(reserve / (N[19] * reserve0 * mound), 0.0, 1.0) ** 0.7   # место кромки, где краска на ноже кончается, держит вал тоньше — не обрывом
            reached_by = (N[8] * He * mound * ur) * pressed >= N[21] * (gap[m] + bulge * (1.0 - pb))   # вал встаёт там, где полный он до холста дожат
            stood[n_r + 1] = reached_by | (settled_last > 0.2)
            iya = np.nonzero(stood[n_r + 1])[0]                # край краски под ножом — крайние места кромки, чей вал стоит на холсте
            edges = (int(iya[0]), int(iya[-1])) if len(iya) else None
            need = np.minimum(np.maximum(bank_full * mound * ur - E[:, 0], 0.0), reserve) * (reached_by | (settled_last > 0.2))   # вал, что уже стоит на холсте, кормится, пока нож его не бросил
            if need.any():
                reserve -= need
                E[:, 0] += need
                E[:, 1] += need
                E[:, 3:] += ks_reserve * need[:, None]
                taken_reserve += float(need.sum())
        Tb, sb = T[i0:i1], wet[i0:i1]
        settled = np.zeros(nb)
        n_r += 1
        z0 = float(z_support[n_r])
        Tm = np.full(nb, -np.inf)
        np.maximum.at(Tm, j, Tb)
        if nb > 2:
            Tm = np.maximum(Tm, np.maximum(np.concatenate([Tm[1:], Tm[-1:]]), np.concatenate([Tm[:1], Tm[:-1]])))
        saturated = np.ones(nb) if clean else np.full(nb, float(clamp(reserve.sum() / (N[19] * reserve0 * nb), 0.0, 1.0)) ** 0.7)    # пустеющий нож кладёт тоньше: зазор держит краска на ноже
        gap_r = (gap[m] + bulge * (1.0 - pb)) * saturated       # лезвие к концам кромки отогнуто от холста; нажим его распрямляет
        rim = (np.maximum(Tm, z0 - N[6] * mt * pb) + gap_r)[j]
        # срез сырого выше кромки
        share_p = covered[i0:i1] * contact[m]
        the_excess = clamp(Tb + sb - rim - trace, 0.0, sb) * share_p
        if the_excess.any():
            cut_off[i0:i1] = the_excess
            in_bank = the_excess * (1.0 - on_plane[m])
            on_knife += float((the_excess - in_bank).sum())
            sb = sb - the_excess
            if in_bank.any():
                np.add.at(E, j, np.concatenate([in_bank[:, None], (in_bank * w_lying[i0:i1])[:, None], (in_bank * skh_lying[i0:i1])[:, None],
                                                ks_lying[i0:i1] * in_bank[:, None]], axis=1))
        ob = E[:, 0]
        has_bank = ob > 1e-12
        tsv_roll = E[:, 3:] / np.where(has_bank, ob, 1.0)[:, None]
        # оставшаяся под кромкой сырая плёнка мешается с валом (объём у обоих тот же)
        if has_bank.any() and sb.any():
            share = N[12] * clamp(ob / max(bank_full, 1e-12), 0.0, 1.0)[j] * contact[m] * (sb > 1e-9)
            exchange = share * np.minimum(sb, ob[j] / np.maximum(np.bincount(j, minlength=nb)[j], 1))
            if exchange.any():
                mixed[i0:i1] = exchange / np.maximum(h_pl[pixel[i0:i1]] - the_excess, 1e-12)
                ks_mixed[i0:i1] = tsv_roll[j]
                np.add.at(E, (j, slice(3, None)), (ks_lying[i0:i1] - tsv_roll[j]) * exchange[:, None])
                tsv_roll = E[:, 3:] / np.where(has_bank, ob, 1.0)[:, None]
        # вал ложится в пустое под кромкой — не глубже досяга
        depth = rim - Tb - sb
        reach = (gap_r + reach_pressure[m])[j]
        asked = np.maximum(depth, 0.0) * clamp(1.0 - (depth - reach) / max(0.2 * gap[m], 1e-9), 0.0, 1.0) * share_p   # глубже досяга краска с кромки до дна не достаёт: низина пуста
        if asked.any() and has_bank.any():
            film = (ob * min(1.0 / (length_roll * width), 1.0))[j]   # краска вала у этого места — плёнкой на длине вала
            lays = np.minimum(asked, film * share_p)       # с вала на px холста сходит не больше его доли: пустеющий нож кладёт тоньше, а не обрывом
            if not clean:
                # краска садится там, где вал достаёт до холста: тонкий вал касается только вершин (зазор над ними
                # мал), над низиной висит — край пустеющего ножа рвётся по зерну, а не бледнеет ровной полосой
                lays = lays * _step((film / np.maximum(depth, 1e-12) - N[21]) / (1.0 - N[21])) * (reached_by | (settled_last > 0.2))[j]
            total = np.bincount(j, weights=lays, minlength=nb)
            give = np.minimum(total, ob)
            lays = lays * (give / np.maximum(total, 1e-12))[j]
            laid[i0:i1] = lays
            settled = np.bincount(j, weights=lays, minlength=nb) / np.maximum(gap_r * width, 1e-12)   # где кромка краску не кладёт (висит над низиной), там и за конец её не выдавливает
            ks_laid[i0:i1] = tsv_roll[j]
            w_roll, skh_roll = (E[:, 1] / np.where(has_bank, ob, 1.0)), (E[:, 2] / np.where(has_bank, ob, 1.0))
            w_laid[i0:i1], skh_laid[i0:i1] = w_roll[j], skh_roll[j]
            E *= (1.0 - give / np.where(has_bank, ob, 1.0))[:, None]
        if not clean:                                        # место кромки, что клало краску и перестало доставать до холста (нож
            settled_last = 0.7 * settled_last + 0.3 * np.minimum(settled, 1.0)   # поднимают), оставляет свой вал на холсте комом
            threshold_brush = 0.7 * N[21] * (gap[m] + bulge * (1.0 - pb))
            dropped = (E[:, 0] > 1e-12) & (((settled_last > 0.2) & ((N[8] * He * mound * ur) * pressed < threshold_brush)) | ((E[:, 0] * pressed / (length_roll * width) < threshold_brush) & (reserve < 0.02 * reserve0 * mound) & reached_by))
            if dropped.any():
                lumps.append((m, np.where(dropped[:, None], E, 0.0)))
                E[dropped] = 0.0
                settled_last[dropped] = 0.0
        # вал растекается вдоль кромки и выдавливается за её концы
        if nb > 1 and clean:
            flow = N[10] * (E[:-1] - E[1:])
            E[:-1] -= flow
            E[1:] += flow
        elif nb > 1:                                          # паста течёт вдоль кромки, пока перепад выше её предела текучести, и тем
            diff = E[:-1] - E[1:]                              # живее, чем сильнее нажим
            a = np.abs(diff[:, :1])
            flow = min(N[29] * (pb / N[2]) ** 2, 0.45) * np.maximum(a - N[22] * bank_full, 0.0) / np.maximum(a, 1e-30) * diff
            E[:-1] -= flow
            E[1:] += flow
        for side, edge in ((0, 0), (1, nb - 1)):
            if clean:
                excess = max(float(E[edge, 0]) - 0.75 * bank_full, 0.0)
                d = N[11] * excess / float(E[edge, 0]) if excess > 0.0 else 0.0
                if excess > 0.0:
                    went = E[edge] * d
                    E[edge] -= went
                    side[side, int(b), :3] += went[:3]
                    side[side, int(b), 3:] += from_ks((went[3:] / max(went[0], 1e-30))[None, :])[0] * went[0]
                continue
            if edges is not None:
                # из-под края краски паста уходит за бок по зазору: поток — как куб зазора над холстом у края (над
                # нитью зазор мал — почти не течёт, над ямкой течёт) и квадрат нажима
                edge = edges[side]
                u_brush = (edge + side) * width - We / 2
                edge_u[side, int(b)] = u_brush
                kr = j == edge
                zz = float(np.mean(np.maximum(depth[kr], 0.0))) / max(float(gap[m]), 1e-12) if kr.any() else 0.0
                excess = float(E[edge, 0]) * min(zz, 2.0) ** 3 * float(min(settled[edge], 1.0))
                d = min(N[23] * (pb / N[2]) ** 2 * excess / max(float(E[edge, 0]), 1e-30), 0.9)
                if excess > 0.0:
                    went = E[edge] * d
                    E[edge] -= went
                    bead[side, :3] += went[:3]
                    bead[side, 3:] += from_ks((went[3:] / max(went[0], 1e-30))[None, :])[0] * went[0]
            if bead[side, 0] > 1e-15:
                # выдавленное за бок — комок пасты: держится за бок ножа и едет с ним по холсту. Холст стирает с него
                # краску вершинами нитей (над низиной комок дна не касается) — валик у бока рвётся по зерну; комок, что
                # перерос удерживаемое ножом, остаётся на холсте целиком и тянется за ножом, — край выпирает
                if edges is None or n_r == len(rows) - 1:
                    exit_b[side].append((int(b), bead[side].copy()))
                    bead[side] = 0.0
                else:
                    sign = 2.0 * side - 1.0
                    qx = min(max(int(np.floor(x[m] + (u_brush + sign) * ax[m] - bx0)), 0), bw - 1)
                    qy = min(max(int(np.floor(y[m] + (u_brush + sign) * ay[m] - by0)), 0), bh - 1)
                    thread = float(clamp(1.0 - (z0 - Tv[qy * bw + qx]) / (N[6] * mt), 0.0, 1.0))
                    trace_b = bead[side] * (share_bead * thread)
                    side[side, int(b)] += trace_b
                    bead[side] -= trace_b
                    if bead[side, 0] * thread * thread > B_prev:
                        exit_b[side].append((int(b), bead[side] * (1.0 - N[31])))
                        bead[side] *= N[31]

    V = np.zeros(bh * bw)
    V[pixel] = laid
    # --- валики у боков хода: откосом наружу от конца кромки
    mb = np.minimum(2 * np.arange(nR), N - 1)
    Vv, Ev = np.zeros((bh, bw)), np.zeros((bh, bw, 5))
    inward = min(mm, We / 8)                                  # валик стоит на краю плёнки: откос начинается чуть внутри кромки (у ребра — у самой кромки)
    for side, sign in ((0, -1.0), (1, 1.0)):
        ob = side[side, :, 0]
        u_b = sign * (We / 2 - inward)                         # откуда наружу встаёт валик: у ножа без краски — конец кромки
        if not clean:                                        # у ножа с краской — край краски под ним (где его не было — как в соседнем ряду)
            u_b = edge_u[side]
            has_u = np.nonzero(~np.isnan(u_b))[0]
            u_b = (u_b[has_u[np.searchsorted(has_u, np.arange(nR)).clip(0, len(has_u) - 1)]] if len(has_u) else np.full(nR, sign * We / 2)) - sign * inward
        v, e = _slope(ob, x[mb] + u_b * ax[mb] - bx0, y[mb] + u_b * ay[mb] - by0, sign * ax[mb], sign * ay[mb],
                      slope if clean else slope * N[26], bw, bh, side[side, :, 1:] / np.maximum(ob, 1e-30)[:, None], field)
        Vv, Ev = Vv + v, Ev + e
        if exit_b[side]:                                   # комки: каждый лежит на длине своего размера, откосом кома
            lump_b = np.zeros((nR, 6))
            for row_brush, B in exit_b[side]:
                size = (float(B[0]) / slope) ** (1.0 / 3.0)       # комок тянется за ножом: толст, где сидел, и сходит на нет по ходу на длине в три своих размера
                rows_brush = row_brush + np.arange(-max(1, int(round(0.5 * size))), max(2, int(round(3.0 * size))) + 1)
                weight = np.maximum(1.0 - np.where(rows_brush < row_brush, (row_brush - rows_brush) / (0.5 * size + 1.0), (rows_brush - row_brush) / (3.0 * size + 1.0)) ** 2, 0.0) ** 2
                ok_brush = (rows_brush >= 0) & (rows_brush < nR)
                lump_b[rows_brush[ok_brush]] += (weight / weight.sum())[ok_brush, None] * B[None, :]
            ob = lump_b[:, 0]
            v, e = _slope(ob, x[mb] + u_b * ax[mb] - bx0, y[mb] + u_b * ay[mb] - by0, sign * ax[mb], sign * ay[mb],
                          slope, bw, bh, lump_b[:, 1:] / np.maximum(ob, 1e-30)[:, None], field)
            Vv, Ev = Vv + v, Ev + e
    # --- ком: где нож оторван, вал остаётся на холсте
    on_canvas = np.ones(nb, dtype=bool) if clean else settled_last > 0.2      # вал мест, что холста не касались, и бусины у боков нож с краской унёс на себе
    carried_off = 0.0 if clean else float(E[~on_canvas, 0].sum() + bead[:, 0].sum())
    lumps.append((min(2 * int(rows[-1]) + 1, N - 1), E * on_canvas[:, None]))
    uk = (np.arange(nb) + 0.5) * width - We / 2
    shares = max(1, int(round(width / 0.5)))
    slope_brush = slope if clean else 2.0 * bank_full * float(mound.max()) / width / field ** 2 / shares   # ком — вал, как стоял перед кромкой: где его больше, там длиннее
    for mk, Ek in lumps:
        if Ek[:, 0].sum() <= 1e-12:
            continue
        ob = Ek[:, 0] / width
        fields = np.concatenate([Ek[:, 1:3], from_ks(Ek[:, 3:] / np.maximum(Ek[:, :1], 1e-30)) * Ek[:, :1]], axis=1) / np.maximum(Ek[:, :1], 1e-30)
        for share_m in ((np.arange(shares) + 0.5) / shares - 0.5):
            v, e = _slope(ob * width / shares, x[mk] + (uk + share_m * width) * ax[mk] - bx0, y[mk] + (uk + share_m * width) * ay[mk] - by0,
                          np.full(nb, tx[mk]), np.full(nb, ty[mk]), slope_brush, bw, bh, fields, field)
            Vv, Ev = Vv + v, Ev + e
    if Vv.any():
        for _ in range(max(1, int(round(mm * mm)))):            # откос оплывает на длине в мм, а не в px
            Vv, Ev = _smooth(Vv), _smooth(Ev)
        if not clean:                                        # паста стоит тупым краем: тоньше следа ножа не растекается — подошва откоса подбирается в него
            thin = Vv < trace
            if thin.any() and (~thin).any():
                k = float(Vv.sum()) / max(float(Vv[~thin].sum()), 1e-30)
                Ev = np.where(thin[:, :, None], 0.0, Ev * k)
                Vv = np.where(thin, 0.0, Vv * k)
    # --- поля ответа
    h1 = h_pl.copy()
    h1[pixel] -= cut_off
    share_wet = np.ones(bh * bw)                               # что осталось от сырой доли: срезано только подвижное
    share_wet[pixel] = np.where(wet > 1e-12, 1.0 - cut_off / np.maximum(wet, 1e-12), 1.0)
    w1 = np.where(h1 > 1e-12, w_pl * h_pl * share_wet / np.maximum(h1, 1e-12), 0.0)
    skh1 = np.where(h1 > 1e-12, skh_pl, 0.0)
    ts = np.zeros((bh * bw, 3))
    ts[pixel] = from_ks(ks_laid)
    wV, SkhV = np.zeros(bh * bw), np.zeros(bh * bw)
    wV[pixel], SkhV[pixel] = w_laid, skh_laid
    vv_pl, ev_pl = Vv.ravel(), Ev.reshape(-1, 5)
    everything = V + vv_pl
    has = everything > 1e-12
    denominator = np.where(has, everything, 1.0)
    ts = (ts * V[:, None] + ev_pl[:, 2:]) / denominator[:, None]
    wV = (wV * V + ev_pl[:, 0]) / denominator
    SkhV = (SkhV * V + ev_pl[:, 1]) / denominator
    share_m = np.zeros(bh * bw)
    ts_m = np.zeros((bh * bw, 3))
    share_m[pixel] = clamp(mixed, 0.0, 1.0)
    ts_m[pixel] = from_ks(ks_mixed)
    touch = np.zeros(bh * bw)
    touch[pixel] = contact[np.minimum(2 * row_p, N - 1)] * covered * stood[np.searchsorted(rows, row_p), place]
    kind = lambda a: a.reshape(bh, bw)   # noqa: E731
    return dict(h=kind(h1), w=kind(clamp(w1, 0.0, 1.0)), skh=kind(clamp(skh1, 0.0, 1.0)), V=kind(everything), ts=ts.reshape(bh, bw, 3), wV=kind(wV), SkhV=kind(SkhV),
                mixed=kind(share_m), ts_mixed=ts_m.reshape(bh, bw, 3), touch=kind(touch), removed=float(cut_off.sum()), on_knife=on_knife,
                wet=kind(wet_field) if has_wet else None, reserve=float(reserve.sum()) + carried_off)
