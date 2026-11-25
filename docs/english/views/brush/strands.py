# READING VIEW of движок/кисть/пряди.py with names in English. Not runnable; edit the original.
"""Пряди и контур: состав кисти (ведущие, гребни, пряди), шум, кромка следа, посадка и подъём."""
import numpy as np

from .fast import clamp

from . import numbers as H
from .numbers import _GAP_MIN, _PResponseLBResponseRResponseZDY, _PResponsePERYoBrush, _PResponseWithADBrushA, _RAWithBrushRYTA, _OPENED_SCATTERED, _WithREZ


# --------------------------------------------------------------------------- шум
def _noise(rng, n, m, scale):
    """Гладкий непериодический шум (n, m): случайные узлы через ~масштаб отсчётов,
    у каждого столбца своя фаза и свой шаг узлов. Среднее 0, разброс ~1."""
    scale = max(float(scale), 1.0)
    steps = scale * (0.75 + 0.5 * rng.random(m))
    k = int(n / (scale * 0.75)) + 3
    nodes = rng.standard_normal((k, m))
    t = np.arange(n)[:, None] / steps[None, :] + rng.random(m)[None, :]
    i = t.astype(np.intp)
    f = t - i
    f = f * f * (3.0 - 2.0 * f)
    columns = np.arange(m)[None, :]
    return nodes[i, columns] * (1.0 - f) + nodes[i + 1, columns] * f


# ------------------------------------------------------------------ состав кисти
class _Makeup:
    """Кисть на один мазок: ведущие волоски, гребни между ними, пряди, выборка профиля."""
    __slots__ = ('nr', 'ns', 'nm', 'центр', 'полуш', 'прядь', 'ось_пряди', 'нагр', 'запас', 'порог',
                 'расст', 'отст', 'липнет', 'u', 'гребень', 'x', 'профиль', 'съём', 'полуширина',
                 'тон', 'W', 'отст_гр', 'вед', 'd_вед', 'дно_в', 'полб', 'давит', 'край_проф', 'есть')


def _tufts(brush, rng, W, spacing, mm):
    """К-14б: ворс пучками (`Кисть.просветы` > 0) — веерная и растрёпанная кисть. Пучок — несколько гребней ворса
    подряд (`в_пучке` в среднем; шаг гребней — как у сплошной кисти), за ним просвет: волоса там нет. На просветы идёт
    доля `просветы` ширины кисти; пучки и просветы неровные — просвет гуляет тем сильнее, чем больше кисть живёт
    клочками (`клочки`: у веерной ворс разложен ровно, у растрёпанной торчит как попало). Пучок — прядь кисти: у него
    своя длина, нагрузка, запас, тон, жёсткость и отставание, просвет поделён между соседними прядями пополам.
    Ответ: места ведущих волосков (границы гребней без краёв кисти), пуст ли гребень (nr,), прядь гребня (nr,)."""
    g = float(clamp(brush.gaps, 0.0, 0.9))
    in_ = max(float(brush.in_tuft), 1.0)
    sigma = brush.unevenness_spacing
    sigma_p = 0.25 + 0.6 * float(clamp(brush.scraps, 0.0, 1.0))
    gap_sr = in_ * spacing * g / (1.0 - g)
    widths, empty, strand = [], [], []
    first = max(gap_sr, 1.5 * mm) * rng.random()            # кисть встаёт не с пучка у самого края
    if first >= 1.0 * mm:
        widths.append(first); empty.append(True); strand.append(0)
    sum, j = sum(widths), 0
    while sum < W:
        n = max(1, int(round(in_ * np.exp(0.45 * rng.standard_normal()))))
        gaps = clamp(spacing * np.exp(sigma * rng.standard_normal(n) - 0.5 * sigma * sigma), _GAP_MIN, 2.2 * spacing)
        pr = float(clamp(gap_sr * np.exp(sigma_p * rng.standard_normal() - 0.5 * sigma_p * sigma_p), 1.5 * mm, max(3.0 * gap_sr, 1.5 * mm)))
        widths += list(gaps) + [0.5 * pr, 0.5 * pr]
        empty += [False] * n + [True, True]
        strand += [j] * (n + 1) + [j + 1]
        sum += float(gaps.sum()) + pr
        j += 1
    edges = np.cumsum(widths)
    nr = int(np.searchsorted(edges, W - 0.9 * mm)) + 1             # последний гребень кончается краем кисти
    empty, strand = np.array(empty[:nr]), np.array(strand[:nr], dtype=np.intp)
    if nr > 1 and empty[-1] and strand[-1] != strand[-2]:
        strand[-1] = strand[-2]                                    # пряди из одного просвета нет
    return edges[:nr - 1] - W / 2, empty, strand


def _makeup(brush, rng):
    W = float(brush.width)
    mm = H._MM                     # Д-8а: px ниже — px холста 8 px/мм (`мм.в_мм`)
    with_ = _Makeup()
    with_.W = W
    with_.half_width = W / 2
    palette_knife = brush.shape == 'мастихин'
    # --- ведущие волоски: шаг борозд
    if palette_knife:
        spacing = 3.0
        leading = np.arange(spacing, W - spacing * 0.5, spacing) - W / 2
    else:
        spacing = float(clamp(brush.spacing, 1.5 * mm, max(1.5 * mm, W / 2.5)))
        sigma = brush.unevenness_spacing
        m = int(W / spacing * 2.0) + 8
        gaps = clamp(spacing * np.exp(sigma * rng.standard_normal(m) - 0.5 * sigma * sigma), _GAP_MIN, 2.2 * spacing)
        position = np.cumsum(gaps) - gaps[0] * rng.random() * 0.8 + 0.6 * mm
        leading = position[position < W - 0.9 * mm] - W / 2
    tufts = None
    if not palette_knife and brush.gaps > 0:
        leading, *tufts = _tufts(brush, rng, W, spacing, mm)
    bounds = np.concatenate([[-W / 2], leading, [W / 2]])
    with_.center = 0.5 * (bounds[1:] + bounds[:-1])
    with_.half_width = 0.5 * (bounds[1:] - bounds[:-1])
    nr = len(with_.center)
    with_.nr = nr
    # --- пряди: соседние гребни по 3–7
    width_strands = max(5.0 * mm, W * (0.30 - 0.12 * brush.scraps))
    strand = np.zeros(nr, dtype=np.intp)
    i, number = 0, 0
    while i < nr and tufts is None:
        k = max(1, int(round(width_strands / spacing * np.exp(0.35 * rng.standard_normal()))))
        strand[i:i + k] = number
        i += k
        number += 1
    ns = number
    with_.has = None
    if tufts is not None:                                         # К-14б: пучок — прядь; в просвете волоса нет
        strand, ns, with_.has = tufts[1], int(tufts[1][-1]) + 1, 1.0 - tufts[0]
    with_.ns, with_.strand = ns, strand
    weight = with_.half_width
    with_.axis_strands = np.bincount(strand, weights=with_.center * weight, minlength=ns) / np.bincount(strand, weights=weight, minlength=ns)
    strength = 0.0 if palette_knife else brush.scraps
    # нагрузка: прядь набрала больше или меньше, гребень внутри пряди — ещё чуть по-своему
    load = np.exp(0.30 * strength * rng.standard_normal(ns))[strand] * np.exp(0.14 * rng.standard_normal(nr))
    radians = np.abs(with_.center) / (W / 2)
    if not palette_knife:
        load *= 1.0 - 0.35 * radians ** 6                         # к краям кисть несёт меньше: обводки нет
    if brush.shape == 'круглая':
        load *= 0.55 + 0.45 * np.sqrt(clamp(1.0 - radians * radians, 0.0, 1.0))
    with_.load = load / np.average(load, weights=with_.half_width)
    with_.reserve = np.exp(0.40 * strength * rng.standard_normal(ns))[strand] * np.exp(0.25 * rng.standard_normal(nr))
    # длина: короткая прядь касается позже и отрывается раньше; внутри пряди — небольшой разброс
    shortness = (rng.random(ns) ** 1.2)[strand] * (0.25 + 0.75 * strength) + 0.14 * rng.random(nr) ** 1.5
    with_.threshold = brush.spread_lengths * shortness
    with_.distance = radians
    # отставание пряди вдоль хода: кисть — не линейка, пряди стоят вразнобой
    lag_pr = (rng.standard_normal(ns) * (0.07 + 0.22 * strength) * W)[strand]
    with_.lag_gr = rng.standard_normal(nr)                       # место гребня вдоль хода внутри пряди, в долях разброса
    lag = lag_pr + (0.6 * mm + 0.02 * W) * with_.lag_gr
    with_.lag = lag - 0.08 * W
    with_.sticks = (0.08 + 0.32 * rng.random(ns)) * strength         # насколько прядь собирается к своей оси при слабом нажиме
    if palette_knife:
        with_.threshold = np.zeros(nr)
        with_.lag = np.zeros(nr)
        with_.lag_gr = np.zeros(nr)
        with_.reserve = np.exp(0.10 * rng.standard_normal(nr))
    # тон: недомешанная краска — у пряди и гребня свой сдвиг светлоты и чуть цвета
    # п. 12: полосы недомеса на палитре шире узкой кисти — щетина в обойме (`остриё` < 1) уже 16 px набирает краску одного
    # тона, и её две-три пряди расходятся слабее (куб ширины: у 12 px — 0,42 прежнего). Иначе каждый ход короткой серии
    # делился на тёмную и светлую половину, и серия одной краской читалась частоколом. Шире 16 px и остриё — как было
    t = brush.spread_tone * (1.0 - (1.0 - brush.point) * (1.0 - min(brush.width / 16.0, 1.0) ** 3))
    lightness = 0.09 * rng.standard_normal(ns)[strand] + 0.085 * rng.standard_normal(nr)
    hue = 0.028 * rng.standard_normal((ns, 3))[strand] + 0.020 * rng.standard_normal((nr, 3))
    with_.tone = np.exp(t * (lightness[:, None] + hue))
    # 11а: среднее тона по кисти (вес — сколько гребень несёт). У кисти в 5–7 гребней оно гуляет от мазка к мазку на 7 % (σ),
    # до 17 %, красный сам по себе: мазок выходит светлее и рыжее и краски, и подложки. `тон_кисти=0` — недомес только делит
    # набранную краску между гребнями, среднее по кисти — сама краска; 1 (умолчание) — как было, бит в бит
    if brush.tone_brushes != 1.0:
        with_.tone = with_.tone / np.average(with_.tone, axis=0, weights=with_.load * with_.half_width)[None, :] ** (1.0 - brush.tone_brushes)
    # --- выборка профиля поперёк кисти
    nm = max(3, int(round(W / _PResponsePERYoBrush)))
    u = (np.arange(nm) + 0.5) * (W / nm) - W / 2
    ridge = clamp(np.searchsorted(leading, u), 0, nr - 1)
    with_.nm, with_.u, with_.ridge = nm, u, ridge
    with_.x = clamp(np.abs(u - with_.center[ridge]) / with_.half_width[ridge], 0.0, 1.0)
    with_.leading = None                                              # борозд нет: мастихин или кисть в один гребень
    if palette_knife or len(leading) == 0:
        with_.profile = np.ones(nm)
        with_.removal = np.ones(nm)
    else:
        # под кончиком ведущего — борозда: дно тем ниже, чем сильнее он давит
        nearest = clamp(np.searchsorted(leading, u), 1, len(leading) - 1) if len(leading) > 1 else np.zeros(nm, np.intp)
        if len(leading) > 1:
            left = leading[nearest - 1]
            right = leading[nearest]
            j = np.where(np.abs(u - left) < np.abs(u - right), nearest - 1, nearest)
        else:
            j = nearest
        d = np.abs(u - leading[j])
        presses = np.exp(0.6 * brush.spread_stiffness * rng.standard_normal(len(leading)))
        if tufts is not None:                                     # К-14б: пучок жёстче или мягче соседнего весь — его волоски давят вместе
            presses = presses * np.exp(0.5 * brush.spread_stiffness * rng.standard_normal(ns))[strand[:len(leading)]]
        depth = (0.80 + 0.20 * brush.viscosity) / (1.0 + 0.06 * (max(int(brush.rows), 1) - 1))
        bottom = clamp(1.0 - depth * presses, 0.06, 0.9)[j]
        half_furrow = _PResponseLBResponseRResponseZDY + 0.35 * mm * rng.random(len(leading))
        t = clamp(d / half_furrow[j], 0.0, 1.0)
        with_.profile = bottom + (1.0 - bottom) * t * t * (3.0 - 2.0 * t)
        with_.removal = 1.35 - 0.7 * t                               # сырую краску кончик снимает сильнее, чем бока
        # к самому краю кисти плёнка сходит на нет: крайний гребень не должен обводить мазок
        with_.edge_profile = 0.45 + 0.55 * clamp((W / 2 - np.abs(u)) / (1.3 * mm), 0.0, 1.0)
        with_.profile = with_.profile * with_.edge_profile
        # для живой борозды (п. 4, `борозда.py`): чей волосок у места, как далеко, дно и полуширина его борозды
        with_.leading, with_.d_leading, with_.half_furrow, with_.presses = j, d, half_furrow, presses
        with_.bottom_in = clamp(1.0 - depth * presses, 0.06, 0.9)
    return with_


def _window_by(sc, a, b, rim=1.5):
    """Гладкое окно 0..1 по пути: от a до b, с краями по кромке px."""
    l = clamp((sc - a) / rim + 0.5, 0.0, 1.0)
    r = clamp((b - sc) / rim + 0.5, 0.0, 1.0)
    return l * l * (3.0 - 2.0 * l) * r * r * (3.0 - 2.0 * r)


def _edge(with_, sc, rng, W, long=1.0, more_often=1.0, narrow=False):
    """Бок живой (версия 3, п. 2г). Крайние пряди кисти реже и беднее краской: след крайнего гребня рвётся,
    кромка отступает внутрь на его ширину (1–3 px) и возвращается; иногда вместе с ним отходит и соседний.
    У входа кисть полна — первую треть ширины хода край держит. Ответ (Nc, nr): 1 — гребень кладёт, 0 — нет."""
    Nc, nr = len(sc), with_.nr
    holds = np.ones((Nc, nr))
    L = float(sc[-1])
    for row in (np.arange(nr), np.arange(nr)[::-1]):
        i = 1 if 2.0 * with_.half_width[row[0]] < 2.0 else 0             # гребень уже двух пикселей густой краски не кладёт
        r1, r2 = row[i], row[i + 1]
        s = W * rng.uniform(0.3, 0.6)
        if narrow:
            s += rng.uniform(0.0, 1.2 * W)                    # у узкой кисти бока рвутся не в одном месте
        while s < L:
            dl = W * rng.uniform(0.12, 0.32) * long
            if narrow:
                # узкая кисть в несколько красок (п. 2д): прядь, отошедшая от холста, возвращается не быстрее чем за 5–9 px
                dl = max(dl, rng.uniform(4.0, 7.0) * min((20.0 - W) / 4.0, 1.0))
            holds[:, r1] = np.minimum(holds[:, r1], 1.0 - _window_by(sc, s, s + dl, 3.0))
            if rng.random() < 0.4:
                holds[:, r2] = np.minimum(holds[:, r2], 1.0 - _window_by(sc, s + dl * rng.uniform(0.15, 0.4), s + dl * rng.uniform(0.6, 0.85), 3.0))
            s += dl + W * rng.uniform(0.3, 0.8) / more_often
    return holds


def opens_hand(p, hand):
    """К8б: кисть раскрывает рука — нажим ведёт рука ("pressure_along") и где-то на ходу был ниже `_РАСКРЫТА`
    (кисть касалась не всей шириной). Иначе посадка — как без поля."""
    return bool(hand) and bool(((np.maximum.accumulate(p) < _RAWithBrushRYTA) & (p > 0.02)).any())


def _laid_on_place(z, s, forward, drags, shorter):
    """К8б: касание z (N, n) места или гребня кисти, который кладут на холст раскрытием нажимом. Кончик пряди отстаёт
    от кисти (`вперёд`, (N, n), px вдоль хода: впереди — больше нуля), потому что его тащат по холсту; волосок, который
    ещё не касался, никто не тащил — он ложится туда, где стоит кисть, и только потом отстаёт. Волосок короче соседних
    (`короче`, (n,), px) согнутая тащимая кисть доносит до холста на столько позже, насколько её уже тащат (`тащит`,
    (N,)); у первого касания кисть стоит прямо — там все встают вместе. Ответ — то же касание, вставшее на столько
    раньше или позже по пути, чтобы кончик лёг там, где до него дошло раскрытие, и ещё позже на свою длину. Подъём —
    как был."""
    growth = np.maximum.accumulate(z, axis=0)
    has = growth > 0.5
    became = z.copy()
    for m in np.nonzero(has.any(axis=0))[0]:
        i = int(np.argmax(has[:, m]))
        d = 0.0
        for _ in range(3):                         # кончик лёг в s_i: место встало там, где s + вперёд(s) = s_i
            j = int(clamp(np.searchsorted(s, s[i] - d), 0, len(s) - 1))
            d = float(forward[j, m])
        d -= shorter[m] * float(drags[i])
        r = np.interp(s + d, s, growth[:, m])
        became[:, m] = np.where(growth[:, m] > 1e-9, r * z[:, m] / np.maximum(growth[:, m], 1e-9), r)
    return became


def _landing(with_, path, p, rng, W, rng_brush=None, thickly=0.0, p_hand=None, holds=None, hand=False, forward=None):
    """Посадка и подъём плоской кисти плашмя (версия 3, п. 2д): какая часть ширины касается холста, пока нажим
    набирают и пока его снимают. Кисть встаёт серединой обоймы — от половины до четырёх пятых ширины, одна
    сторона раньше другой (как — от сида); углы встают, когда нажим набран, а щетину протащило на `_ПОСАДКА[2..3]`
    ширины: в ширину она расходится с ходом. Пряди встают не по линейке: у каждой своя задержка. Вход короче
    пятой части ширины — удар: кисть ложится разом, торец плоский. При подъёме углы уходят первыми, тем же
    порядком; сорванная кисть (`быстро`) отходит плашмя. Пока рука держит набранный нажим, кисть лежит всей шириной.
    Конец (п. 4, `rng_к`): кисть уходит с холста с креном — срез густой краски идёт под углом к ходу и чуть дугой
    (`_срез`); на медленном подъёме гребень держится за холст по-своему, и концы гребней стоят вразнобой.
    Ответ: (N, nr) — касается ли гребень, (N, nm) — касается ли место на кисти; (None, None) — кисть легла разом
    и ушла разом."""
    s = path['s']
    ds = path['ds']
    i0 = int(np.argmax(p > 0.02))
    growth = np.gradient(p, ds)
    # п. 10 (`p_руки`): у узкой кисти в одну краску нажим у посадки держит капля (`укладка._дрожь`), а вход меряется
    # по руке — иначе капля читалась набранным нажимом, каждый вход выходил «ударом», и корень следа срезан прямо
    pv = p if p_hand is None else p_hand
    growth_in = growth if p_hand is None else np.gradient(pv, ds)
    stands_still = np.nonzero((growth_in[i0:] * W < 0.15 * pv[i0:]) & (pv[i0:] > 0.02))[0]
    i1 = i0 + int(stands_still[0]) if len(stands_still) else len(p) - 1          # нажим набран
    if path['L'] < H._BrushResponseRResponseTBrushAndY[0] * W and not path['на_месте']:
        # Н-2: короткий ход (`путь._короткий`) — рука ставит кисть, а `'остановка'` дальше давит её до конца хода: рост
        # нажима не стихал, набранным читался нажим у самого конца, вход — в полхода, и угол кисти не вставал (след 0,54–0,8
        # ширины у 5 из 30). Нажим набран, где рост упал вчетверо против самого быстрого на входе
        fast = np.maximum.accumulate(growth_in[i0:])
        knee = np.nonzero((growth_in[i0:] < 0.25 * fast) & (pv[i0:] > 0.02) & (np.arange(len(p) - i0) > 0))[0]
        if len(knee):
            i1 = min(i1, i0 + int(knee[0]))
    # длина входа: путь, на котором рука набрала 85 % нажима (дальше нажим тела только дышит)
    entry = float(s[int(np.argmax(pv >= 0.85 * pv[i1]))]) - (0.0 if p_hand is None else float(s[i0]))
    soft, sharp = max(0.26 * W, 3.8), max(0.15 * W, 2.8)
    strike = float(clamp((soft - entry) / (soft - sharp), 0.0, 1.0))
    strike = strike * strike * (3.0 - 2.0 * strike)
    w0 = rng.uniform(_PResponseWithADBrushA[0], _PResponseWithADBrushA[1])
    tilt = rng.uniform(-1.0, 1.0) * _PResponseWithADBrushA[4]
    c0 = clamp(np.array([w0 + tilt, w0 - tilt]), 0.2, 1.0)        # доля полуширины, что встала сразу: слева и справа
    sweep = W * rng.uniform(_PResponseWithADBrushA[2], _PResponseWithADBrushA[3])
    delay = _PResponseWithADBrushA[5] * rng.random(with_.ns)[with_.strand] + _PResponseWithADBrushA[6] * rng.random(with_.nr)
    opens = opens_hand(p, hand)
    if opens:
        # К8б: кисть раскрывают нажимом — до места доходит раскрытие от середины к краю, одно на всю обойму; ни прядь,
        # ни гребень не ждут сверх него своей очереди по нажиму (задержка по нажиму пускала крайнюю прядь раньше
        # внутренней, и та вставала позже бруском во всю свою ширину). Волоски короче или длиннее соседних, и край
        # пряди входит зубцами, — это по пути (`_лёг_на_месте`, `_РАСКРЫТА_ВРАЗБЕГ`)
        delay = np.zeros(with_.nr)
    g = np.ones(len(p))
    heel = np.ones(len(p))
    if strike < 1.0:
        # (п. 10: севшая каплей кисть набрана против нажима капли, а не руки — пятно обратного хода не игла)
        t = np.where(np.arange(len(p)) <= i1, p / max(float(p[i1] if p_hand is None else min(p[i1], p[i0:i1 + 1].min())), 1e-6), 1.0)
        charge = clamp((t - 0.1) / 0.85, 0.0, 1.0)
        soft = np.minimum(charge, clamp((s - s[i0]) / sweep, 0.0, 1.0))
        g = strike + (1.0 - strike) * soft
        # самое первое касание — пятно у середины: до своей доли ширины оно расходится, пока набирается первая
        # половина нажима (перёд следа круглый, а не срезанный)
        z = clamp((t - 0.1) / _PResponseWithADBrushA[9], 0.0, 1.0)
        heel = strike + (1.0 - strike) * (_PResponseWithADBrushA[8] + (1.0 - _PResponseWithADBrushA[8]) * z * (2.0 - z))
        if p_hand is not None:
            # п. 10: узкая кисть — с острым концом: севшая каплей, она касается сначала остриём, и пятно расходится до
            # своей доли ширины, пока кисть протащило на треть-половину ширины (по нажиму капли оно вставало разом: у
            # кисти, севшей одной стороной во всю полуширину, корень следа 12 px выходил прямым срезом, бруском)
            z = clamp((s - s[i0]) / (W * (0.3 + 0.2 * abs(tilt) / _PResponseWithADBrushA[4])), 0.0, 1.0)
            heel = strike + (1.0 - strike) * (0.35 + 0.65 * z * (2.0 - z))     # остриё в капле — пятно в треть своей доли ширины, не игла
    if hand:
        # К8.1: нажим ведёт рука ("pressure_along") — ширина касания растёт с самим нажимом, а не с долей набранного:
        # слабо поставленная кисть касается серединой обоймы и раскрывается, пока на неё давят, в долю ширины по
        # доле нажима от `_РАСКРЫТА`, до всей ширины; раскрытая на ходу не сжимается (снимает её подъём ниже). Удар слабой рукой — тоже узкий
        t_r = np.maximum.accumulate(p) / _RAWithBrushRYTA
        g = np.minimum(g, clamp(t_r, 0.0, 1.0))
        heel = np.minimum(heel, clamp(t_r, _PResponseWithADBrushA[8], 1.0))
    share = p / np.maximum(np.maximum.accumulate(p), 1e-6)
    # сорванная кисть (нажим ушёл быстрее, чем за треть ширины хода) отходит плашмя: углы уйти первыми не успевают
    falloff = np.maximum(-growth, 0.0) * W / np.maximum(np.maximum.accumulate(p), 1e-6)
    breakaway = clamp((falloff - _PResponseWithADBrushA[10]) / (_PResponseWithADBrushA[11] - _PResponseWithADBrushA[10]), 0.0, 1.0)
    max = 0.0
    for i in range(len(p)):
        max = max(max, breakaway[i]) if share[i] < 0.97 else breakaway[i]
        breakaway[i] = max
    g = np.minimum(g, np.maximum(clamp((share - 0.12) / 0.78, 0.0, 1.0), breakaway))
    cut_g = cut_m = None
    irregular = np.zeros(with_.nr)
    if rng_brush is not None:
        cut_g, cut_m = _cut(with_, s, p, rng_brush, W, thickly, tilt)
        # гребень, полный краски, липнет к холсту дольше соседа; бедный уходит раньше — не лесенкой от угла к середине
        irregular = _WithREZ[4] * (rng_brush.random(with_.ns)[with_.strand] - 0.5 + 0.8 * (rng_brush.random(with_.nr) - 0.5))
    if g.min() >= 1.0 and heel.min() >= 1.0:
        return cut_g, cut_m
    agrees = (np.arange(len(p)) > int(np.argmax(p))) & (share < 0.97)        # рука снимает нажим

    def touches(place, side, back, shift):
        g_m = clamp((g[:, None] - back[None, :]) / (1.0 - back[None, :]), 0.0, 1.0)
        # углы встают последними: край касания идёт к краю кисти сначала медленно, под конец быстро
        # (К8б: кисть, которую раскрывает рука, раскрывается в долю ширины по доле нажима — край касания идёт к краю
        # кисти ровно, без рывка под конец: рывок клал крайнюю прядь разом)
        g_m = g_m if opens else g_m ** _PResponseWithADBrushA[7]
        to = with_.half_width * (c0[side][None, :] * heel[:, None] + (1.0 - c0[side][None, :]) * g_m) + 1.0 * g_m
        to_opened = to
        to = to + with_.half_width * shift[None, :] * (agrees * (1.0 - breakaway) * (1.0 - g))[:, None]
        if opens:
            # К8б: кисть, недожатая до всей ширины, на подъёме уходит вразнобой, но шире, чем её раскрыли, не ложится
            # (иначе недожатый край вставал бруском, когда руку уже снимали)
            to = np.minimum(to, np.maximum.accumulate(to_opened, axis=0))
        z = clamp((to - place[None, :]) / 0.8 + 0.5, 0.0, 1.0)
        return z * z * (3.0 - 2.0 * z)

    by_ridges = touches(np.maximum(np.abs(with_.center) - with_.half_width, 0.0), (with_.center > 0).astype(np.intp), delay, irregular)
    if holds is not None:
        # п. 18в (`укладка._держит`): рука сходит с кисти на ходу — углы на укладке не уходят первыми, кисть лежит всей
        # шириной до конца нажима; перенос краски (по гребням) — как был
        # К8б: кисть, которую раскрывают нажимом, лежит шириной, до какой её раскрыли, а не всей: недожатые края не
        # ложатся разом, когда рука её отпускает (вставали бруском на подъёме позже соседней пряди)
        g = g + ((np.maximum.accumulate(g) if opens else 1.0) - g) * holds * agrees
    by_places = touches(np.abs(with_.u), (with_.u > 0).astype(np.intp), delay[with_.ridge], irregular[with_.ridge])
    if opens and forward is not None:
        # К8б: волоски обоймы не одной длины: у гребня своя, у волоска в нём — ещё своя (свой поток, от сида)
        rng_in = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
        shorter_g = _OPENED_SCATTERED[0] * W * rng_in.random(with_.nr)
        shorter_m = shorter_g[with_.ridge] + _OPENED_SCATTERED[1] * W * rng_in.random(with_.nm) ** 2
        vd, drags = forward
        by_ridges = _laid_on_place(by_ridges, s, vd, drags, shorter_g)
        by_places = _laid_on_place(by_places, s, vd[:, with_.ridge], drags, shorter_m)
    if cut_g is not None:
        by_ridges, by_places = by_ridges * cut_g, by_places * cut_m
    return by_ridges, by_places


def _cut(with_, s, p, rng, W, thickly, tilt):
    """Срез конца удара (версия 3, п. 4). Кисть уходит с холста не ровно: одна сторона отрывается раньше другой
    (чаще та, что позже встала, — рука держит тот же крен), середина обоймы отстаёт или опережает. Конец следа
    густой кисти — линия под углом к ходу и чуть дугой, у каждого удара своя, а не поперечный срез буквой D.
    Тонкая краска обойму не держит — её конец прежний. Ответ: (N, nr) и (N, nm) — лежит ли ещё гребень и место;
    (None, None) — срез не нужен."""
    if thickly <= 0.0:
        return None, None
    has = np.nonzero(p > 0.02)[0]
    if not len(has):                                          # Р-1: рука нигде не нажала выше 0,02 — кисть не вставала, срезать нечего (было: IndexError, мазок падал без слов)
        return None, None
    s_brush = float(s[has[-1]])                                   # где кисть ушла с холста
    sweep = s_brush - float(s[has[0]])
    side = -np.sign(tilt) if rng.random() < 0.7 else np.sign(tilt)     # раньше уходит сторона, что встала позже
    bevel = min(W * rng.uniform(_WithREZ[0], _WithREZ[1]), _WithREZ[3] * sweep) * thickly * (side if side != 0 else 1.0)
    arc = min(W * rng.uniform(-_WithREZ[2], _WithREZ[2]), _WithREZ[3] * sweep) * thickly

    after = np.arange(len(p)) >= int(np.argmax(p))
    half = np.nonzero(after & (p <= 0.5 * float(p.max())))[0]
    exit = _WithREZ[5] * (s_brush - float(s[half[0]])) if len(half) else 0.0
    t_ = np.array([-1.0, 0.0, 1.0])
    max = (np.abs(bevel) * 0.5 * (1.0 + np.sign(bevel) * t_) + arc * (1.0 - t_ * t_))
    max = float(max.max() - max.min())

    def lies(u):
        t = u / with_.half_width                                  # -1..1 поперёк кисти
        earlier = np.abs(bevel) * 0.5 * (1.0 + np.sign(bevel) * t) + arc * (1.0 - t * t)
        earlier = earlier - earlier.min()                        # дальше всех кисть лежит до конца хода
        # 4б: скос считается от места, где рука сняла половину нажима, а не от последней точки пути: иначе он
        # прятался в сходе нажима, и конец читался прежним торцом. Сторона, что лежит до конца, не тронута
        earlier = earlier + exit * earlier / max(float(max), 1e-6)
        z = clamp(s_brush - earlier[None, :] - s[:, None] + 1.5, 0.0, 1.0)     # где кисть лежит до конца, срез ничего не трогает
        return z * z * (3.0 - 2.0 * z)

    return lies(with_.center), lies(with_.u)
