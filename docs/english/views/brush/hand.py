# READING VIEW of движок/кисть/рука.py with names in English. Not runnable; edit the original.
"""Мн-2: форма следа одного мазка — числами в ширинах его кисти: вход, сход, торцы, пряди, изгиб.

Перенос `архив/анализатор/рука.py` (шаг 2в-5) на мазок картины: там маска мазка готова (скан на белом), здесь её край
ищется по снимку (`след`). Все длины — в ширинах кисти W (медиана ширины среза по середине мазка), поэтому мазок 3 мм
картины 655 мм и мазок 1 мм той же картины, названной 250 мм, — одни числа. Миллиметров здесь нет.

1. `след` — сколько краски мазка в каждом px снимка, 0–1: доля на отрезке «краска соседа — краска мазка», обе краски
   берутся рядом с этим px. Край мазка — уровень 0,5 — стоит точнее px. Решается только полоса у края; что внутри и
   что снаружи, знает разбор.
2. `замер` — ось по серединам срезов тела, за ним — по касательной; хвост, загнутый вбок, — по дуге до кончика (`_дуга`):
   по касательной его мерили бы поперёк него самого. След выпрямляется вдоль оси; по выпрямленному: ширина по ходу, где
   она набирает и теряет 0,8 тела (вход и сход), тупость кончика, кромка торца и её рваность, пряди, изгиб.

Куда шла рука, по картинке не узнать: вход — конец, где больше сплошной краски, сход — где он прядями, у́же или острее."""
import math

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ShAG = 1.0 / 24.0       # W: шаг выпрямления
WithEREDAndNA = 0.6         # W — медиана ширины по средним 60 % длины (концы в тело не входят)
TELResponse = 0.8             # вход и сход — где ширина меньше 0,8 тела
ShAndRE = 1.25            # срез у конца шире медианы больше — не тело, а загнутый хвост вдоль среза
InGLAtB = 0.5           # W: дуга хвоста начинается внутри тела — касательная там чистая
BEND_PX = 0.25        # px: точнее кромка на снимке с шумом не видна
ZAGAndB, BEND_EXPLICIT = 0.03, 0.07  # W: кромка хвоста пошла наружу на столько — загнут, если торец не прямой; на столько — загнут. Чистые: срез
#                        наискось до 0,04, клин 1,6 W, загнутый на 40°, — 0,04–0,12 (от 8 px)
WithGLADAndT = 0.1         # W: окно ширины по ходу — зубец пряди не обрывает тело
DLAndNNYY = 5.0          # W: длиннее — ось многочленом 3-й степени, короче — 2-й
ZONE_BUTT_ENDS = 0.6       # W: кромка торца ищется от кончика внутрь на длину конца + столько
RResponseInNYaT = 0.2          # W: рваность — отклонение кромки торца от неё же, сглаженной этим окном
ShchEL = 0.05            # W: щель поперёк хода шире делит пряди
TAtPResponseWithT = 0.25         # W: тупость — средняя ширина на таком куске от кончика, считая от среза (срез наискось — тоже срез)
WithBrushResponseWith = 1.5             # тангенс: срез наискось не круче (56°); круче — это уже сход клином
ResponseTRResponseWithTResponseBrush = 1.25        # px: отросток тоньше — зуб растра, хода не загибает
ZONE_ENTRY = 1.5       # W: вход — конец, где на таком куске больше краски (пряди и сход длиной до 1,5 W)
FRResponseNT = 0.15           # W: кончик — середина следа, самого далёкого от тела, с таким допуском
LAtHShE = 0.75          # дуга принимается, если середина следа ушла от неё во столько раз меньше, чем от касательной
ShPResponseRA = 0.25           # W: отросток тоньше половины этого (1 px у 8 px) тупости не меняет
PRYaDAnd = 0.3            # W: пряди считаются от кончика внутрь на max(столько, длина конца), не дальше 1,5 W
ResponseTWithHYoTResponseIn = 24          # px на W: мазок вдвое крупнее и больше меряется клетками k×k px
EDGE_OUTWARD, EDGE_INWARD = 0.35, 0.2   # W: полоса у границы разбора, в которой край решает снимок
BrushResponseNTRAWithT = 5.0         # единицы разбора: краски по сторонам ближе — края на снимке нет, он как у разбора


def _paint_beside(lab, where, σ):
    """Краска пикселей `где`, какой она видна из каждого места: среднее с весом гаусса `σ`; далеко от всех — ближайшая."""
    in_ = ndimage.gaussian_filter(where.astype(np.float64), σ, mode='constant')
    brush = np.stack([ndimage.gaussian_filter(lab[..., j] * where, σ, mode='constant') for j in range(3)], -1)
    good = in_ > 1e-4
    brush = brush / np.maximum(in_, 1e-12)[..., None]
    if not good.all() and where.any():
        iy, ix = ndimage.distance_transform_edt(~where, return_distances=False, return_indices=True)
        brush = np.where(good[..., None], brush, lab[iy, ix])
    return brush


def trace(lab, mask, width):
    """Доля краски мазка в каждом px, 0–1. `lab` — снимок в единицах разбора, `маска` — область разбора, `ширина` px."""
    da = ndimage.distance_transform_edt(mask)
    no = ndimage.distance_transform_edt(~mask)
    inside = da > EDGE_INWARD * width
    outside = no > EDGE_OUTWARD * width
    if inside.sum() < 4 or not outside.any():
        return mask.astype(np.float64)
    own = _paint_beside(lab, inside, max(0.5 * width, 1.0))
    foreign = _paint_beside(lab, outside, max(0.2 * width, 0.7))
    d = own - foreign
    dd = (d * d).sum(-1)
    share = np.clip(((lab - foreign) * d).sum(-1) / np.maximum(dd, 1e-9), 0.0, 1.0)
    share = np.where(dd >= BrushResponseNTRAWithT ** 2, share, mask)
    share[inside] = 1.0
    share[outside] = 0.0
    return share


def _piece(a):
    """Самый большой связный кусок следа (пряди, отошедшие до 0,5 доли, — с ним); остальное — соседи."""
    labels, n = ndimage.label(a > 0.5, np.ones((3, 3)))
    if n <= 1:
        return a
    brush = 1 + int(np.argmax(ndimage.sum(a, labels, range(1, n + 1))))
    iy, ix = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
    return np.where(labels[iy, ix] == brush, a, 0.0)


def _between(a, share):
    """Процентиль упорядоченного ряда, как у `np.percentile`."""
    m = share * (len(a) - 1)
    i = min(int(m), len(a) - 2)
    return a[i] + (m - i) * (a[i + 1] - a[i])


def _axis(yx, W0, gl, across):
    ts = yx.mean(0)
    s, t = (yx - ts) @ gl, (yx - ts) @ across
    L = s.max() - s.min()
    power = 3 if L > DLAndNNYY * W0 else 2
    n = int(np.clip(L / (0.25 * W0), 6, 80))
    brush = np.clip(((s - s.min()) / max(L, 1e-9) * n).astype(int), 0, n - 1)
    section = []
    for i in range(n):
        tt = t[brush == i]
        if len(tt) >= 2:
            tt = np.sort(tt)
            lo, hi = _between(tt, 0.02), _between(tt, 0.98)
            section.append((s.min() + (i + 0.5) * L / n, 0.5 * (lo + hi), hi - lo, math.sqrt(len(tt))))
    section = np.array(section).reshape(-1, 4)
    # середины срезов конца, срезанного наискось, уходят к его углу: ось тела — по телу, за ним — по касательной или `_дуга`.
    # Срез много шире тела у самого конца — это хвост, загнутый вдоль среза: в тело он не входит
    body = np.zeros(len(section), bool)
    if len(section):
        m = np.median(section[:, 2])
        body = section[:, 2] >= TELResponse * m
        narrow = np.flatnonzero(body & (section[:, 2] <= ShAndRE * m))
        if len(narrow):
            body[:narrow[0]] = False
            body[narrow[-1] + 1:] = False
    if body.sum() > power + 1:
        section = section[body]
    if len(section) <= power + 1:
        return ts, np.zeros(1), s.min(), s.max(), s.min(), s.max()
    return ts, np.polyfit(section[:, 0], section[:, 1], power, w=section[:, 3]), s.min(), s.max(), section[0, 0], section[-1, 0]


def _arc(yx, b, h, W0):
    """Продолжение оси за телом: `б` — точка оси внутри конца тела, `h` — ход в ней (наружу). Ход с постоянным загибом —
    дуга от `б` по касательной `h` через кончик (самое далёкое от `б` за телом); точки с шагом `ШАГ` и за кончик на 0,75 W."""
    d = yx - b
    d = d[d @ h > 0]
    if not len(d):
        return None
    r = np.linalg.norm(d, axis=1)
    e = d[r >= r.max() - max(FRResponseNT * W0, 0.75)].mean(0)
    n = np.array([-h[1], h[0]])
    x, y = float(e @ h), float(e @ n)
    α = math.atan2(y, x)                                         # хорда к касательной; вся дуга загнута на 2α
    if abs(α) < 1e-3:
        return None
    R = math.hypot(x, y) / (2 * math.sin(α))
    φ = np.arange(0.0, 1.0, ShAG * W0 / (2 * α * R)) * 2 * α
    t = math.cos(2 * α) * h + math.sin(2 * α) * n
    return np.r_[b + np.outer(R * np.sin(φ), h) + np.outer(R * (1 - np.cos(φ)), n),
                 b + e + np.outer(np.arange(0, int(0.75 / ShAG)) * ShAG * W0, t)]


def _straighten(a, yx, axis, W0, bent=(False, False)):
    ts, coef, s0, s1, with0, with1, gl, across = axis
    spacing = ShAG * W0
    ss = np.arange(s0 - 0.75 * W0, s1 + 0.75 * W0, spacing / 4)
    ss = np.clip(ss, with0, with1)                                     # за телом ось — по касательной
    pr = np.polyval(np.polyder(coef), ss) if len(coef) > 1 else 0.0 * ss
    point = ts + np.outer(ss, gl) + np.outer(np.polyval(coef, ss) + pr * (ss - ss), across)
    parts, from_, to = [], 0, len(point)
    for end in (0, 1):                                         # загнутый хвост: вместо касательной — дуга до кончика
        if not bent[end]:
            continue
        with_ = max(with1 - InGLAtB * W0, 0.5 * (with0 + with1)) if end else min(with0 + InGLAtB * W0, 0.5 * (with0 + with1))
        i = int(np.clip(np.searchsorted(ss, with_), 1, len(ss) - 2))
        h = point[i + 1] - point[i - 1]
        h = (h if end else -h) / max(float(np.linalg.norm(h)), 1e-9)
        d = _arc(yx, point[i], h, W0)
        if d is not None:
            parts.append((end, d))
            from_, to = (from_, i) if end else (i + 1, to)
    point = np.concatenate([d[::-1] for brush, d in parts if not brush] + [point[from_:to]] + [d for brush, d in parts if brush])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(point, axis=0), axis=1))]
    u = np.arange(0.0, arc[-1], spacing)
    axes = np.stack([np.interp(u, arc, point[:, 0]), np.interp(u, arc, point[:, 1])], 1)
    touch = np.gradient(axes, axis=0)
    touch /= np.maximum(np.linalg.norm(touch, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-touch[:, 1], touch[:, 0]], 1)
    v = np.arange(-1.6 * W0, 1.6 * W0 + spacing / 2, spacing)
    Y = axes[None, :, 0] + v[:, None] * normal[None, :, 0]
    X = axes[None, :, 1] + v[:, None] * normal[None, :, 1]
    gray = ndimage.map_coordinates(a, [Y, X], order=1, cval=0.0)
    done = gray > 0.5
    if parts:                                                    # срезы загнутой оси пересекаются: px — у ближайшей точки оси
        vv, uu = np.nonzero(done & (np.abs(v) > 0.5 * W0)[:, None])
        done[vv, uu] = cKDTree(axes).query(np.stack([Y[vv, uu], X[vv, uu]], 1))[0] >= np.abs(v[vv]) - 0.1 * W0
    return done, gray, axes, normal, spacing, np.stack([Y, X], -1)


def _circle(d):
    n = max(1, int(round(d)) // 2)
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return y * y + x * x <= max(0.25 * d * d, 1.0)


def _bent(gray, top, bottom, tip, inward, edge, W, lines):
    """Насколько загнут хвост у `кончик`а, W: на сколько кромка одной стороны, отойдя от тела внутрь или нет, идёт наружу
    там, где противоположная — внутрь. У среза наискось длинная кромка прямая, у клина обе идут внутрь: у них 0."""
    st = np.arange(edge + inward * int(W), tip - inward, -inward)  # от тела (на W вглубь) до кончика
    st = st[(st >= 0) & (st < len(top))]
    st = st[bottom[st] >= top[st]]
    if len(st) < 2:
        return 0.0
    # кромки точнее шага: уровень 0,5 между соседними строками
    in_, n = np.clip(top[st], 1, None), np.clip(bottom[st], None, gray.shape[0] - 2)
    in_ = in_ - np.clip((gray[in_, st] - 0.5) / np.maximum(gray[in_, st] - gray[in_ - 1, st], 1e-6), 0.0, 1.0)
    n = n + np.clip((gray[n, st] - 0.5) / np.maximum(gray[n, st] - gray[n + 1, st], 1e-6), 0.0, 1.0)
    at = np.abs(st - edge) <= 0.5 * W
    in0, n0 = (np.median(in_[at]), np.median(n[at])) if at.any() else lines
    response = 0.0
    for own, foreign in ((n - n0, in_ - in0), (in0 - in_, n0 - n)):
        outward = own - np.minimum(np.minimum.accumulate(own), 0.0)
        response = max(response, float(np.minimum(outward, foreign).max()) / W)
    return response


def _past_middles(top, bottom, end, W, row):
    """Насколько середина следа на последней W хода в среднем отходит от середины тела, W."""
    st = end[0] + end[1] * np.arange(max(1, int(round(W))))
    st = st[(st >= 0) & (st < len(top))]
    st = st[bottom[st] >= top[st]]
    return float(np.abs(0.5 * (top[st] + bottom[st]) - 0.5 * (row[0] + row[1])).mean() / W) if len(st) else 0.0


def _strand(columns, slit):
    """Прядей в каждом столбце: полос краски, между которыми щель не у́же `щель`."""
    t = columns.T
    d = np.diff(np.pad(t.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    (kn, start), (_, end) = np.nonzero(d == 1), np.nonzero(d == -1)
    divides = (kn[1:] == kn[:-1]) & (start[1:] - end[:-1] >= slit)
    return t.any(1) + np.bincount(kn[1:][divides], minlength=len(t))


def _butt_end(done, tip, inward, length, W, lines):
    """Кромка торца — где в каждой строке тела от кончика внутрь впервые краска; числа в W."""
    depth = int(round((length + ZONE_BUTT_ENDS) * W))
    st = tip + inward * np.arange(depth)
    st = st[(st >= 0) & (st < done.shape[1])]
    brush = done[lines[0]:lines[1] + 1][:, st]
    e = np.where(brush.any(1), brush.argmax(1), len(st)) / W
    if len(e) < 3:
        return {'рваность': 0.0, 'скос': 0.0, 'тупость': 0.0}
    v = (np.arange(len(e)) - 0.5 * (len(e) - 1)) / W
    n = float(np.clip((v @ (e - e.mean())) / (v @ v), -WithBrushResponseWith, WithBrushResponseWith))
    rest = e - n * v
    smooth = ndimage.uniform_filter1d(rest, max(3, int(round(RResponseInNYaT * W)) | 1), mode='nearest')
    sweep = ndimage.median_filter(rest, int(round(ShPResponseRA * W)) | 1, mode='nearest')   # шпора — не кончик
    to = np.clip((TAtPResponseWithT - (sweep - sweep.min())) / TAtPResponseWithT, 0.0, 1.0)   # средняя ширина кончика до глубины ТУПОСТЬ от среза
    return {'рваность': float(np.sqrt(np.mean((rest - smooth) ** 2))), 'скос': float(np.degrees(np.arctan(abs(n)))),
            'тупость': float(to.sum() / W)}


def _on_paint(yx, done, axes, normal, spacing, u, inward, W):
    """Кончик: середина краски в последних 0,1 W хода — в ближайший px следа."""
    st = u + inward * np.arange(max(1, int(round(0.1 * W))))
    st = st[(st >= 0) & (st < done.shape[1])]
    vv, i = np.nonzero(done[:, st])
    v = (vv - (done.shape[0] - 1) / 2) * spacing
    t = (axes[st[i]] + v[:, None] * normal[st[i]]).mean(0)
    return tuple((yx[np.argmin(((yx - t) ** 2).sum(1))] - 1.0).tolist())


def measurement(a, width):
    """Форма следа: `а` — доля краски 0–1 (или маска), `ширина` — примерная ширина, px. Ответ — словарь: 'W' px,
    'длина к ширине', 'вход', 'сход' — {'длина', 'тупость', 'рваность', 'скос', 'пряди', 'сплошность',
    'полнота' — средняя ширина последней W хода,
    'краски' — краска на последних `ЗОНА_ВХОДА`, W²}, 'разброс ширины',
    'сплошность', 'изгиб', 'поворот'. Меньше 12 px краски — None."""
    a = np.asarray(a, np.float64)
    a = _piece(np.pad(a, 1))
    if (a > 0.5).sum() < 12:
        return None
    W0 = max(float(width), 2.0)
    yx = np.argwhere(a > 0.5).astype(np.float64)
    with_, in_ = np.linalg.eigh(np.cov(yx.T))
    axis = _axis(yx, W0, in_[:, 1], in_[:, 0]) + (in_[:, 1], in_[:, 0])
    bent, yx_, past, pass_ = None, yx, None, 1
    while True:
        done, gray, axes, normal, spacing, place = _straighten(a, yx, axis, W0, bent or (False, False))
        u_ = np.flatnonzero(done.any(0))
        if len(u_) < 6:
            return None
        u0, u1 = int(u_[0]), int(u_[-1])
        has = done.any(0)
        top, bottom = done.argmax(0), np.where(has, done.shape[0] - 1 - done[::-1].argmax(0), -1)
        width = np.where(has, bottom - top + 1, 0).astype(np.float64)
        solid = np.where(has, done.sum(0) / np.maximum(width, 1.0), 0.0)
        n = u1 - u0 + 1
        field = int(0.5 * (1 - WithEREDAndNA) * n)
        W = max(float(np.median(width[u0 + field:u1 + 1 - field])), 4.0)
        smooth = ndimage.uniform_filter1d(width, max(1, int(round(WithGLADAndT * W))), mode='constant')
        body = np.flatnonzero(smooth[u0:u1 + 1] >= TELResponse * W) + u0
        t0, t1 = (int(body[0]), int(body[-1])) if len(body) else (u0 + field, u1 - field)
        t = np.arange(t0, t1 + 1)
        t = t[has[t]]
        rows = []
        for edge in (t0, t1):                                    # тело у этого конца: кисть могла увести след от оси
            at = t[np.abs(t - edge) <= max(W, 3)]
            rows.append((int(np.median(top[at])), int(np.median(bottom[at]))))
        ends_u = ((u0, 1), (u1, -1))
        now = [_past_middles(top, bottom, at, W, row) for at, row in zip(ends_u, rows)]
        if bent is not None:                                   # дуга верна, если идёт серединой следа, — не к углу прямого конца
            true = tuple(z and with_ < LAtHShE * m for z, with_, m in zip(bent, now, past))
            if true == bent or pass_ > 2:
                break
            bent, pass_ = true, pass_ + 1
            if not any(bent):
                done, gray, axes, normal, spacing, top, bottom, has, width, solid, n, W, t0, t1, t, rows, u0, u1 = straight_sweep
                break
            continue
        past = now
        straight_sweep = done, gray, axes, normal, spacing, top, bottom, has, width, solid, n, W, t0, t1, t, rows, u0, u1
        butt_ends = [_butt_end(done, *sh, W, row) for sh, row in zip(((u0, 1, (t0 - u0) / W), (u1, -1, (u1 - t1) / W)), rows)]
        thick = ndimage.binary_opening(done, _circle(ResponseTRResponseWithTResponseBrush / spacing))  # отросток в px ход не загибает
        has_ = np.flatnonzero(thick.any(0))
        if len(has_) < 6:
            break
        in_, n_ = thick.argmax(0), np.where(thick.any(0), thick.shape[0] - 1 - thick[::-1].argmax(0), -1)
        outward = [_bent(gray, in_, n_, has_[0], 1, t0, W, rows[0]), _bent(gray, in_, n_, has_[-1], -1, t1, W, rows[1])]
        yx = place[thick]
        bent = tuple(n > BEND_EXPLICIT or (n > max(ZAGAndB, BEND_PX / (W * spacing)) and t_['тупость'] < PRYaMResponseY) for n, t_ in zip(outward, butt_ends))
        if not any(bent):
            break
    ends = []
    for tip, inward, length, row in ((u0, 1, (t0 - u0) / W, rows[0]), (u1, -1, (u1 - t1) / W, rows[1])):
        brush = butt_ends[inward < 0] if not bent or not any(bent) else _butt_end(done, tip, inward, length, W, row)
        zone = tip + inward * np.arange(max(1, int(round(min(1.5, max(PRYaDAnd, length)) * W))))
        zone = zone[(zone >= 0) & (zone < len(width)) & (has[np.clip(zone, 0, len(width) - 1)])]
        brush['пряди'] = int(np.percentile(_strand(done[:, zone], max(1, int(round(ShchEL * W)))), 90))
        brush['сплошность'] = float(solid[zone].mean())
        brush['длина'] = float(length)
        st = tip + inward * np.arange(max(1, int(round(W))))
        brush['полнота'] = float(np.minimum(width[st[(st >= 0) & (st < len(width))]], W).sum() / W / len(st))
        st = tip + inward * np.arange(max(1, int(round(ZONE_ENTRY * W))))
        brush['краски'] = float(np.minimum(done[:, st[(st >= 0) & (st < len(width))]].sum(0), W).sum() / W ** 2)
        ends.append(brush)
    # вход — где больше сплошной краски на последних 1,5 W: кисть, уходя, сходит длинно, остро, прядями
    a, b = ends
    straight = a['краски'] * a['сплошность'] >= b['краски'] * b['сплошность']
    tch = axes[u0:u1 + 1]
    canvas = tch[-1] - tch[0]
    dl = max(float(np.linalg.norm(canvas)), 1e-9)
    ends_px = (u0, u1) if straight else (u1, u0)
    return {'W': W * spacing, 'длина к ширине': n / W, 'вход': a if straight else b, 'сход': b if straight else a,
            'разброс ширины': float(np.std(width[t]) / W), 'сплошность': float(solid[t].mean()),
            'изгиб': float(np.abs((tch - tch[0]) @ np.array([-canvas[1], canvas[0]]) / dl).max() / dl),
            'поворот': float(np.degrees(np.arccos(np.clip(normal[u0] @ normal[u1], -1.0, 1.0)))),
            'концы': [_on_paint(yx_, done, axes, normal, spacing, u, 1 if u == u0 else -1, W) for u in ends_px]}


# ---------------------------------------------------------------- слова

InAndDNResponse = 5.0            # px: у́же — концы не различаются (сужение на 1 W — меньше двух пределов снимка)
VISIBLE_BUTT_END = 12.0     # px: у́же — рваный торец от суженного, прямой от круглого не отличить (зуб 1/4 W — 3 px, предел снимка)
WithAtZhEN, PRYaMResponseY = 0.4, 0.7      # тупость: клин 0,1–0,3, полукруг 0,5–0,65, срез 0,7–1
PResponseLResponseInAndNA = 0.5         # полнота (средняя ширина последней W хода): клин 1,5 W — 0,33, срез 45° — 0,5, полукруг — 0,9
WithPLResponseShNResponseY = 0.9         # узкий (полнота меньше половины) и щелей меньше десятой — сплошной хвост, а не рваный торец
RInANYY = 0.08          # W: рваность ровных концов (срез, полукруг, клин) с растром и шумом — до 0,07
BrushResponseRResponseTResponseBrush = 1.5          # длина к ширине: короче — концов не назвать


def narrowed(brush):
    """Конец сужен: остёр кончик или последняя ширина хода наполовину пуста (длинный хвост с тупым кончиком); срез наискось — нет."""
    return brush['тупость'] < WithAtZhEN or (brush['полнота'] < PResponseLResponseInAndNA and brush['тупость'] < PRYaMResponseY)


def butt_end(brush, W):
    """Имя конца: 'рваный', 'сужен' (клином или со скошенными углами), 'круглый', 'прямой'; у мазка у́же `ВИДНО_ТОРЕЦ` px — только 'сужен' или 'тупой'; у́же
    `ВИДНО` — None."""
    if W < InAndDNResponse:
        return None
    if W < VISIBLE_BUTT_END:
        return 'сужен' if narrowed(brush) else 'тупой'
    if brush['рваность'] > RInANYY and brush['пряди'] >= 2 and not (brush['полнота'] < PResponseLResponseInAndNA and brush['сплошность'] > WithPLResponseShNResponseY):
        return 'рваный'
    return 'сужен' if narrowed(brush) else 'круглый' if brush['тупость'] < PRYaMResponseY else 'прямой'


def words(z):
    """Форма одного мазка строкой."""
    h = lambda v, n=1: f'{v:.{n}f}'.replace('.', ',')
    if z['длина к ширине'] < BrushResponseRResponseTResponseBrush or z['W'] < InAndDNResponse:
        return (f'удар {h(z["длина к ширине"])} ширины' if z['W'] >= InAndDNResponse else f'шириной {h(z["W"])} px') + ' — концы не различаются'
    t = [butt_end(z[and_], z['W']) for and_ in ('вход', 'сход')]
    brush = [f'{and_} {t[i]} (тупость {h(z[and_]["тупость"], 2)}), {h(z[and_]["длина"])} ширины' + (f', прядей {z[and_]["пряди"]}' if z['W'] >= VISIBLE_BUTT_END and z[and_]['пряди'] > 1 else '')
         + (f', срез наискось {h(z[and_]["скос"], 0)}°' if t[i] == 'прямой' and z[and_]['скос'] >= 20 else '')
         for i, and_ in enumerate(('вход', 'сход'))]
    return f'длина {h(z["длина к ширине"])} ширины; {brush[0]}; {brush[1]}; изгиб {h(z["изгиб"], 2)} длины'
