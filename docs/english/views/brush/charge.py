# READING VIEW of движок/кисть/набор.py with names in English. Not runnable; edit the original.
"""Набор краски: раскладка красок по гребням, неровный набор, прожилки, тонкие места."""
import numpy as np

from .fast import clamp

from .numbers import _VEIN_PLACES, _ShTRAndCanvas
from .path import _window
from .strands import _window_by, _noise


def _layout(paints, layout, with_, Nc, ds_brush, rng, thickly=0.0, width_bands=None):
    """Какая краска на каком гребне в каждой точке пути: номера (Nc, nr).

    густо — насколько краска густая (0..1): густая с гребня на гребень по кисти не переползает,
    граница полос стоит на своём волоске весь мазок; у жидкой она плывёт вдоль хода."""
    n = len(paints)
    nr = with_.nr
    if n == 1:
        return np.zeros((Nc, nr), dtype=np.intp)
    shares = np.array([d for _, d in paints])
    W = with_.W
    if layout == 'смесь':
        # волоски несут случайную из красок по долям; соседние чаще одну
        numbers = np.empty(nr, dtype=np.intp)
        numbers[0] = rng.choice(n, p=shares)
        for k in range(1, nr):
            numbers[k] = numbers[k - 1] if rng.random() < 0.55 else rng.choice(n, p=shares)
        return np.broadcast_to(numbers, (Nc, nr))
    if layout == 'кончик':
        # первая краска — на всей кисти; остальные набраны углом: только с одного края
        sign = 1.0 if rng.random() < 0.5 else -1.0
        place = (sign * with_.center / W + 0.5)                    # 0 — «тот» край, 1 — дальний
        edge = np.cumsum(shares[:0:-1])[::-1] if n > 1 else np.zeros(0)
        sweep = _noise(rng, Nc, n - 1, 90.0 / ds_brush) * 0.035
        bounds = np.concatenate([edge[None, :] + sweep + 0.03 * rng.standard_normal(n - 1)[None, :],
                                  np.zeros((Nc, 1))], axis=1)   # убывают: [доля 2..n, доля 3..n, ..., 0]
        numbers = np.zeros((Nc, nr), dtype=np.intp)
        for q in range(1, n):
            numbers[place[None, :] < bounds[:, q - 1:q]] = q
        return numbers
    if layout not in ('полосы', 'ряд'):
        raise ValueError("раскладка: 'полосы', 'ряд', 'смесь' или 'кончик'")
    # полосы поперёк кисти: в случайном порядке ('полосы') или в заданном, слева направо по ходу ('ряд');
    # границы неровные и плывут вдоль мазка
    if width_bands:
        # п. 13: кисть набрана не по разу каждой краской, а мелко — краски лежат на ворсе полосками заданной ширины
        # (px холста), чередуясь; ширина полоски гуляет, у краски с меньшей долей полоски уже. Граница стоит на своём
        # гребне весь мазок (густая краска по ворсу не ползёт); рвётся полоска вдоль хода — в `_набор` (`кусок`)
        numbers = np.empty(nr, dtype=np.intp)
        q = int(rng.choice(n, p=shares / shares.sum())) if layout == 'полосы' else n - 1
        edge = -W / 2
        to = edge + width_bands * n * shares[q] / shares.sum() * float(np.exp(0.35 * rng.standard_normal())) * rng.uniform(0.3, 1.0)
        for k in np.argsort(with_.center):
            while with_.center[k] > to:
                q = (q + (1 + int(rng.integers(0, n - 1)) if layout == 'полосы' and n > 2 else 1)) % n if layout == 'полосы' else (q - 1) % n
                to += max(width_bands * n * shares[q] / shares.sum() * float(np.exp(0.35 * rng.standard_normal())), 1.0)
            numbers[k] = q
        return np.broadcast_to(numbers, (Nc, nr))
    order = rng.permutation(n) if layout == 'полосы' else np.arange(n)[::-1]
    bounds = np.cumsum(shares[order])[:-1]
    bounds = bounds[None, :] + _noise(rng, Nc, n - 1, 110.0 / ds_brush) * (0.05 * (1.0 - thickly)) \
        + 0.03 * rng.standard_normal(n - 1)[None, :]
    place = with_.center / W + 0.5 + 0.02 * rng.standard_normal(nr)
    band = (place[None, :, None] > bounds[:, None, :]).sum(axis=2)
    return order[band]


def _exit_by(sc, a, b, rim):
    """К7б: доля хода sc, где краски пряди нет, — от a до b (b = inf — до конца хода), со сходом на длине кромка (px)
    на каждом краю, серединой там, где был бы срез. Разность двух гладких ступеней: площадь — ровно b − a, и разрыв
    короче кромки до полной соседней краски не доходит, а редеет. a, b — числа или по гребням (nr,): ответ (Nc, nr)."""
    h = 0.5 * max(float(rim), 1e-6)
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    s = np.asarray(sc, dtype=np.float64).reshape((-1,) + (1,) * max(a.ndim, b.ndim))

    def step(x):                                           # гладкая ступень 0→1 на [−h, h]
        t = clamp((x + h) / (2.0 * h), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    return step(s - a) - step(s - b)


def _streak(with_, sc, rng, W, threshold, inside_brush=None):
    """Тонкое место гребня (п. 2г): запас пустеет и возвращается на длине 0,35 ширины (не короче 9 px) — штрих вдоль
    хода, 0..1 по (Nc, nr). Сытая кисть у самого входа кроет: первые 3–6 px тонкого места нет. Крайние гребни —
    отдельно (`_край`)."""
    Nc = len(sc)
    spacing_brush = max(float(sc[1] - sc[0]), 1e-3) if Nc > 1 else 2.0
    if inside_brush is None:
        inside_brush = with_.distance * W / 2 < W / 2 - 3.0
    streak = clamp((_noise(rng, Nc, with_.nr, max(0.35 * W, 9.0) / spacing_brush) - threshold) / _ShTRAndCanvas[1], 0.0, 1.0)
    return streak * streak * (3.0 - 2.0 * streak) * inside_brush[None, :] * clamp((sc - 3.0) / 3.0, 0.0, 1.0)[:, None]


def _charge(band, with_, path, ic, p, rng, W, n, shares, rng_d, piece=None, consumption=None):
    """Как густые несмешанные краски набраны на кисть (версия 3, п. 2в). Краски взяты с палитры рядом и не
    размешаны: у прядей одной полосы запас разный, узкая полоса кончается раньше удара или рвётся (её место
    занимает соседняя краска или борозда до лежащего слоя); в прядях есть гребни соседней краски — прожилки;
    слой гребня вдоль хода то толще, то тоньше. На дуге снаружи слой тоньше и рвётся раньше, внутри гуще.
    П. 2г: прожилка — в главной полосе (краска с наибольшей долей в вызове, от 45 %) и уже у входа; тонкое
    место — штрих. доли — доли красок из вызова.

    полоса — номер краски на каждом гребне в начале мазка. Ответ, всё (Nc, nr), кроме соседа:
    жилка — насколько гребень кладёт соседнюю краску вместо своей; сосед — её номер (nr,);
    обрыв — 1, где гребень кладёт, 0 — где его краска кончилась и под ним борозда; тонко — множитель слоя.
    К7б: прядь кончается не срезом поперёк хода, а сходом: краски на волосе становится меньше на той длине, на которой
    пустеющий слой тоньшает (конец запаса — 0,5 ширины, разрыв — край `редеет`, 0,3 ширины; комок — 0,3 его длины), и
    соседняя краска (или борозда) входит по мере того, как своя тает. Сход стоит серединой там, где был срез (`_сход_по`):
    соседней краски в куске столько же, у каждого гребня свой край — по его запасу."""
    nr, ns = with_.nr, with_.ns
    Nc = len(ic)
    s0 = path['s'][int(np.argmax(p > 0.02))]
    sc = path['s'][ic] - s0                                   # px пути от касания
    L = max(float(sc[-1]), 1.0)
    spacing_brush = max(float(sc[1] - sc[0]), 1e-3) if Nc > 1 else 2.0
    # соседние краски слева и справа: прожилка — чаще ближней
    left = np.full(nr, -1)
    right = np.full(nr, -1)
    dl = np.full(nr, 1e9)
    dp = np.full(nr, 1e9)
    for r in range(nr):
        i = np.nonzero(band[:r] != band[r])[0]
        if len(i):
            left[r], dl[r] = band[i[-1]], with_.center[r] - with_.center[i[-1]]
        i = np.nonzero(band[r + 1:] != band[r])[0]
        if len(i):
            right[r], dp[r] = band[r + 1 + i[0]], with_.center[r + 1 + i[0]] - with_.center[r]
    vl = np.where(left >= 0, np.exp(-dl / (0.3 * W)), 0.0)
    vp = np.where(right >= 0, np.exp(-dp / (0.3 * W)), 0.0)
    neighbor = np.where(rng.random(nr) * (vl + vp) < vl, left, right)
    neighbor = np.where(neighbor >= 0, neighbor, band)
    width_floor = np.array([2.0 * with_.half_width[band == q].sum() for q in range(n)]) / W
    narrow = width_floor[band] < 0.15
    # дуга: + — внутренняя сторона (гребень сдвинут к центру поворота)
    ds = path['ds']
    kr = _window(np.gradient(path['ход'], ds), int(0.5 * W / ds))
    kr_sr = float(np.mean(kr[p > 0.1])) if (p > 0.1).any() else 0.0
    inward = kr[ic][:, None] * with_.center[None, :]
    # слой гребня вдоль хода: запас на прядях лежит неровно
    # у входа кисть сыта и кроет; неровность запаса проступает, когда краска с гребней сходит
    exit = (0.6 + 0.4 * clamp(sc / (3.0 * W), 0.0, 1.0))[:, None]
    thin = 1.0 + exit * (0.40 * _noise(rng, Nc, ns, W / spacing_brush)[:, with_.strand] + 0.28 * _noise(rng, Nc, nr, 0.4 * W / spacing_brush))
    thin *= clamp(1.0 + 1.0 * inward, 0.55, 1.45)
    # на сколько хватает краски полосы у каждой пряди, px
    u = rng.random(ns)[with_.strand]
    length = np.where(narrow, W * (0.6 + 1.6 * u), W * (3.0 + 6.0 * u)) * np.exp(0.15 * rng.standard_normal(nr))
    length *= clamp(1.0 + 2.5 * kr_sr * with_.center, 0.35, 1.3)
    # К7б: своя краска гребня сходит на той же длине, на которой пустеющий слой тоньшает (0,5 ширины), серединой — у конца запаса
    transmittance = _exit_by(sc, length, np.inf, 0.5 * W)
    thin *= clamp((length[None, :] - sc[:, None]) / (0.5 * W), 0.3, 1.0)     # пустея, слой тоньше
    # прядь рвётся: узкая полоса где угодно, широкая — когда сытость у входа прошла (больше ширины хода)
    thins_out = np.zeros((Nc, nr))
    for q in range(ns):
        own = np.nonzero(with_.strand == q)[0]
        narrow = narrow[own].any()
        with_how = 0.15 * L if narrow else max(0.15 * L, 1.2 * W)
        if with_how >= 0.8 * L:
            continue
        for _ in range(rng.poisson((0.8 * L - with_how) / (2.0 * W) * (0.7 if narrow else 0.3))):
            a = rng.uniform(with_how, 0.8 * L)
            b = min(a + W * rng.uniform(0.15, 0.6), L - 0.3 * W)
            for r in own:                                    # гребни пряди рвутся не по линейке
                ar, br = a + rng.normal(0.0, 0.12 * W), b + rng.normal(0.0, 0.12 * W)
                transmittance[:, r] = np.maximum(transmittance[:, r], _exit_by(sc, ar, br, 0.3 * W))   # К7б: сходит на краю `редеет`
                thins_out[:, r] = np.maximum(thins_out[:, r], _window_by(sc, ar - 0.35 * W, br + 0.15 * W, 0.3 * W))
    thin *= 1.0 - 0.75 * thins_out                              # к разрыву слой тоньше: сквозь него лежащее
    # где краски нет: у одних прядей соседняя полоса занимает место, у других открывается борозда
    furrow = (rng.random(ns) < 0.4)[with_.strand]
    break_off = 1.0 - transmittance * furrow[None, :]
    vein = transmittance * (~furrow)[None, :]
    if piece:
        # п. 13б: у кисти, набранной комками, запас пряди кончается не разом на весь остаток хода (нить соседней краски
        # или борозда до конца удара), а комками: за своей длиной комки пряди мельче и реже (ниже)
        break_off, vein = np.ones((Nc, nr)), np.zeros((Nc, nr))
        # п. 13: краска полосы лежит на ворсе комками, а не ровным слоем по всей длине волоса: комок сошёл на холст —
        # гребень кладёт то, что под ним (краску, набранную раньше, — соседнюю) или ничего (борозда до лежащего слоя),
        # пока по волосу не подошёл следующий. Комок — на всех гребнях своей полосы (п. 13б), у каждого гребня свой край:
        # кусок краски в следе рваный. Длина куска `кусок` (px холста) — в среднем, при обычном нажиме: комок сходит
        # по отданной краске (`расход` — что кисть отдала к шагу, px пути обычной руки), лёгкая рука тянет его дольше
        sweep = sc if consumption is None else consumption
        Lkh = max(float(sweep[-1]), 1.0)
        rim = 0.3 * piece                                  # К7б: комок сходит на трети своей длины, а не срезом
        r = 0
        while r < nr:
            # п. 13б: комок сидит на всей ширине своей полосы — на всех её гребнях подряд
            together = 1
            while r + together < nr and band[r + together] == band[r]:
                together += 1
            # комки полосы вдоль волоса: начало и длина каждого; длина — сколько в комке краски (в среднем `кусок`)
            lumps = []
            a = -piece * rng.uniform(0.0, 1.5)
            lasts = float(np.interp(float(np.mean(length[r:r + together])), sc, sweep))   # где запас полосы на исходе
            while a < Lkh:
                few = float(clamp((a - lasts) / (4.0 * piece), 0.0, 1.0))
                has = piece * float(np.exp(0.5 * rng.standard_normal())) * (1.0 - 0.45 * few)
                lumps.append((a, has, rng.random() < 0.5))
                a += has + piece * rng.uniform(0.25, 1.1) * float(np.exp(0.4 * rng.standard_normal())) * (1.0 + 0.8 * few)
            # край комка у каждого гребня свой, спереди и сзади: комок — капля, к бокам полосы короче, лежит на ворсе
            # косо, и каждый гребень несёт свою долю — больше или меньше соседа; крайнему гребню достаётся не всякий комок
            nk = len(lumps)
            across = (with_.center[r:r + together] - with_.center[r:r + together].mean()) / max(with_.half_width[r:r + together].sum(), 1e-6)   # −1…1
            shorter = 0.5 * (1.0 - np.sqrt(clamp(1.0 - 0.8 * across ** 2, 0.0, 1.0)))[None, :] * np.array([e for _, e, _ in lumps])[:, None]
            oblique = rng.normal(0.0, 0.18 * piece, (nk, 2, 1)) * across[None, None, :]
            before = shorter + oblique[:, 0] + rng.normal(0.0, 0.10 * piece, (nk, together))
            back = -shorter + oblique[:, 1] + rng.normal(0.0, 0.10 * piece, (nk, together))
            past = np.zeros((nk, together), dtype=bool)
            if together > 2:
                past[:, [0, -1]] = rng.random((nk, 2)) < 0.2
            for i, j in enumerate(range(r, r + together)):
                end_prev = -1e9
                for k, (a, has, slit) in enumerate(lumps):
                    if past[k, i]:
                        continue
                    start = a + before[k, i]
                    end = max(a + has + back[k, i], start + 0.35 * piece)
                    if start - end_prev > 0.0 and end_prev > -1e8:
                        w = _exit_by(sweep, end_prev, start, rim)
                        if slit:
                            break_off[:, j] = break_off[:, j] * (1.0 - w)
                        else:
                            vein[:, j] = np.maximum(vein[:, j], w)
                    end_prev = max(end, end_prev)
                if end_prev > -1e8 and end_prev < Lkh:
                    w = _exit_by(sweep, end_prev, np.inf, rim)
                    if lumps[-1][2]:
                        break_off[:, j] = break_off[:, j] * (1.0 - w)
                    else:
                        vein[:, j] = np.maximum(vein[:, j], w)
            r += together
    # прожилки (п. 2г): краска на кисти не размешана ещё до касания — гребень соседней краски кладёт её с первого
    # пикселя, а не когда своя сошла. У одних прядей примеси почти нет, у других до половины гребней. Прожилка
    # тянется вдоль хода (не короче четырёх своих ширин), начало и конец сходят на нет — на четверти её длины (К7б: не 3 px); потом гребень опять свой
    # в полосе уже трёх гребней чужой гребень или тонкое место — уже разрыв полосы (он считается выше)
    ridges = np.array([np.count_nonzero(band == q) for q in range(n)])[band]
    wide = ridges >= 3
    inside_brush = (with_.distance * W / 2 < W / 2 - 3.0) & wide        # не у самого края кисти: край — в `_край`
    # прожилки — в главной полосе (краска, которой в вызове больше всех и не меньше 45 % кисти): она набрана
    # поверх прежних красок, и те в ней остались. Прожилка — в гребень (до 3,6 px), не у самого края кисти
    main = (band == int(np.argmax(shares))) & (max(shares) >= 0.45)
    usable = (neighbor != band) & (2.0 * with_.half_width <= 3.6) & inside_brush & main
    # краски набраны рядом: чужого больше у границы полос, к середине широкой полосы — меньше
    # (на втором-третьем гребне от неё; сам пограничный гребень чужим — это уже ступенька границы)
    to_gr = np.minimum(dl, dp)
    at_bounds = np.where(to_gr < 3.5, 0.04, np.exp(-np.maximum(to_gr - 7.0, 0.0) / (0.12 * W)))
    neighbor = np.where(dl < dp, left, right)
    neighbor = np.where(neighbor >= 0, neighbor, band)
    # у одних прядей примеси почти нет, у других до половины гребней; какие гребни — решает место у границы
    admixture = 0.6 * rng.random(ns) ** 1.5
    how_much = sum(int(np.floor(admixture[q] * np.count_nonzero((with_.strand == q) & usable) + rng.random())) for q in range(ns))
    own = np.nonzero(usable)[0]
    veins = []
    if len(own):
        weight = at_bounds[own] / at_bounds[own].sum()
        # в главной полосе чужой гребень есть всегда: чистой полосы с палитры не набрать
        n_pr = min(how_much, len(own))
        veins = [int(r) for r in rng.choice(own, size=max(n_pr, 1), replace=False, p=weight)]
        if piece:
            veins = []                                         # п. 13б: чужую краску в полосе несут комки, нитью во весь ход она не идёт
    else:
        n_pr = 0
    for i, r in enumerate(veins):
        sh = 2.0 * with_.half_width[r]
        # гребень набрал чужую краску до касания: прожилка чаще идёт от самого входа
        with_entry = i >= n_pr or rng.random() < 0.65
        a = -rng.uniform(0.0, 0.2 * W) if with_entry else rng.uniform(0.0, 0.9 * W)
        while a < L:
            dl_zh = max(4.0 * sh, W * (rng.uniform(0.6, 1.4) if with_entry else rng.uniform(0.3, 1.0)))
            vein[:, r] = np.maximum(vein[:, r], _exit_by(sc, a, a + dl_zh, 0.25 * dl_zh))   # К7б: чужая на гребне кончается сходом
            a += dl_zh + W * rng.uniform(0.6, 2.5)
            with_entry = False
    # тонкое место (п. 2г): запас гребня пустеет и возвращается на длине хода, а не в одном пикселе — слой
    # сходит почти на нет штрихом вдоль хода, и сквозь него видно лежащее. Крайние гребни — отдельно (`_край`)
    thin = clamp(thin, 0.08, 1.5) * (1.0 - _ShTRAndCanvas[2] * _streak(with_, sc, rng, W, _ShTRAndCanvas[0], inside_brush))
    return clamp(vein, 0.0, 1.0), neighbor, break_off, thin, None if piece else _veins_by_place(band, with_, sc, rng_d, W, n, main, wide)


def _veins_by_place(band, with_, sc, rng, W, n, main, wide):
    """Прожилка и тонкое место в часть гребня (версия 3, п. 2д). У узкой кисти на полосу два-три гребня, и чужой
    гребень целиком — уже разрыв полосы; в боковой полосе широкой кисти то же. Здесь прожилка соседней краски
    занимает 1,2–2,6 px поперёк (часть гребня), идёт вдоль хода как у широкой кисти (чаще от самого входа) и лежит
    в ближней к соседке трети полосы, не вплотную к границе; тонкое место — штрих в 1–1,8 px поперёк. Берутся
    полосы шире 5 px, кроме главной полосы широкой кисти (её прожилки — целым гребнем, в `_набор`); крайние 2 px
    кисти — дело `_край`. Ответ: (прожилка (Nc, nm), номер её краски (nm,), тонкое место (Nc, nm)) или None."""
    Nc, nm = len(sc), with_.nm
    L = max(float(sc[-1]), 1.0)
    veins = np.zeros((Nc, nm))
    whose = np.zeros(nm, dtype=np.intp)
    thin = np.zeros((Nc, nm))
    has = False

    def across(u0, sh):
        z = clamp((with_.u - u0) / 0.5 + 0.5, 0.0, 1.0) * clamp((u0 + sh - with_.u) / 0.5 + 0.5, 0.0, 1.0)
        return z * z * (3.0 - 2.0 * z)

    for q in range(n):
        own = np.nonzero(band == q)[0]
        if len(own) == 0:
            continue
        left, right = float((with_.center - with_.half_width)[own[0]]), float((with_.center + with_.half_width)[own[-1]])
        a, b = max(left, -W / 2 + 2.0), min(right, W / 2 - 2.0)
        if right - left <= 5.0 or b - a < 2.5:
            continue
        if main[own].any() and wide[own].all() and len(own) >= 4:
            continue                                          # главная полоса широкой кисти: прожилки целым гребнем
        neighbors = [(int(band[own[0] - 1]), -1)] if own[0] > 0 else []
        neighbors += [(int(band[own[-1] + 1]), 1)] if own[-1] < len(band) - 1 else []
        if neighbors and (main[own].any() or rng.random() < _VEIN_PLACES[0]):
            kr, side = neighbors[int(rng.integers(len(neighbors)))]
            sh = float(clamp(rng.uniform(0.15, 0.25) * (right - left), 1.0, 2.6))
            indent = rng.uniform(1.0, max(0.4 * (b - a) - sh, 1.0 + 1e-6))
            u0 = (left + indent) if side < 0 else (right - indent - sh)
            u0 = float(clamp(u0, a, max(b - sh, a)))
            window_u = across(u0, sh)
            # в главной полосе прожилка чаще идёт от самого входа (как у широкой кисти), в боковой — короче и реже от входа
            with_entry = rng.random() < (0.65 if main[own].any() else 0.3)
            x = -rng.uniform(0.0, 0.2 * W) if with_entry else rng.uniform(0.0, 0.9 * W)
            while x < L:
                dl = max(4.0 * sh, W * (rng.uniform(0.6, 1.4) if with_entry and main[own].any() else rng.uniform(0.3, 0.9)))
                veins = np.maximum(veins, _exit_by(sc, x, x + dl, 0.25 * dl)[:, None] * window_u[None, :])   # К7б: сходом
                x += dl + W * rng.uniform(0.6, 2.5)
                with_entry = False
            whose[window_u > 0] = kr
            has = True
        if not (wide[own].all() and len(own) >= 3):
            # тонкое место: у полосы в два-три гребня — часть гребня по ширине и часть хода
            x = rng.uniform(4.0, 4.0 + _VEIN_PLACES[1] * W)
            while x < L:
                sh = rng.uniform(1.0, 1.8)
                dl = max(0.35 * W, 6.0) * rng.uniform(0.7, 1.5)
                u0 = rng.uniform(a, max(b - sh, a + 1e-6))
                thin = np.maximum(thin, _window_by(sc, x, x + dl, 2.0)[:, None] * across(u0, sh)[None, :])
                x += dl + rng.uniform(0.5, 1.0) * _VEIN_PLACES[1] * W
                has = True
    return (veins, whose, thin) if has else None
