# READING VIEW of движок/кисть/борозда.py with names in English. Not runnable; edit the original.
"""Живая борозда (версия 3, пункт 4): глубина и наполнение борозды меняются вдоль хода, место её — нет.

Борозда — след кончика ведущего волоска. Ровной лентой она выходит, когда волосок всю дорогу давит одинаково
и несёт одинаково. Здесь у каждого ведущего волоска вдоль хода своя жизнь, и вся она — из кисти и из того, что
лежит на холсте, а не шум поверх картинки:

  запас кончика   краска на кончике кончается и возобновляется из запаса выше по волоску. Под полным кончиком
                  краска затекает за волоском — борозда мелкая и узкая; под пустым волосок скребёт — глубокая
                  и широкая, у жёсткого волоска до дна своей плёнки. Расход идёт с нажимом и касанием, поэтому
                  счёт ведётся не в px пути, а в израсходованной краске (`_запас_кончика`);
  проскок         волосок цепляется, гнётся и срывается вперёд: 1,5–4,5 px он холста не касается (борозда на этом
                  месте закрыта — короткий срыв), а приземлившись, оставляет каплю — светлую точку под светом.
                  Чаще в густой краске и под лёгкой рукой; на гребне лежащей краски — всегда (`_проскок`);
  слипание        соседние волоски слипаются и расходятся: гребень между ними проседает, две борозды читаются
                  одной широкой. Прижатая щетина расходится — под нажимом слипаний нет (`_слипание`).

`положить_мазок` зовёт `неровный_запас` (запас гребней, перенесён из укладки как был), `борозды` (один раз на мазок)
и `профиль_борозд` (поперечный профиль плёнки на каждом шаге пути). Числа — `числа.py`, блок «пункт 4».
"""
import numpy as np

from .fast import clamp, surround

from .numbers import (_FURROW_DEPTH, _FURROW_RIDGE, _FURROW_LIVE, _FURROW_RESERVE, _FURROW_DROP, _FURROW_FINE, _FURROW_LIFT, _FURROW_SKIP, _FURROW_STRAND,
                    _FURROW_STICKING, _FURROW_TONE, _FURROW_TONE_WEIGHT, _FURROW_WIDER, _WithAtCanvasAYa)
from .strands import _noise


def uneven_reserve(with_, brush, rng, Nc, spacing_brush, W, palette_knife, half_empty=False):
    """Неровный запас гребней: шум вытянут вдоль хода и свой у каждого гребня — сухая кисть рвётся линиями.
    Ответ: неровн (Nc, nr) — запас гребня вдоль хода; густота (Nc, nr) — гребень то гуще, то тоньше: краска
    на волоске лежит неровно."""
    nr, ns = with_.nr, with_.ns
    if palette_knife:
        nk = max(3, int(W / 9.0) + 2)
        base = _noise(rng, Nc, nk, 60.0 / spacing_brush) * 0.38 + _noise(rng, Nc, nk, 20.0 / spacing_brush) * 0.28
        t = (with_.center / W + 0.5) * (nk - 1)
        i0 = clamp(t.astype(np.intp), 0, nk - 2)
        f = t - i0
        f = f * f * (3 - 2 * f)
        uneven = 1.0 + base[:, i0] * (1 - f)[None, :] + base[:, i0 + 1] * f[None, :]
        return uneven, np.ones((Nc, nr))
    uneven = (1.0 + 0.30 * _noise(rng, Nc, nr, 170.0 / spacing_brush) + 0.26 * _noise(rng, Nc, nr, 55.0 / spacing_brush)
              + 0.14 * _noise(rng, Nc, nr, 16.0 / spacing_brush)
              + 0.20 * brush.scraps * _noise(rng, Nc, ns, 90.0 / spacing_brush)[:, with_.strand])
    if brush.unevenness_reserve != 1.0:      # К8.3: мягкий длинный волос (лайнер) держит краску вдоль хода ровнее щетины
        uneven = 1.0 + (uneven - 1.0) * float(brush.unevenness_reserve)
    thickness = clamp(1.0 + 0.16 * _noise(rng, Nc, nr, 38.0 / spacing_brush) + 0.12 * _noise(rng, Nc, nr, 9.0 / spacing_brush), 0.4, 1.7)
    if half_empty:
        # п. 10, инструмент 2: на полупустом ворсе краска лежит комками в несколько волосков, а не нитью вдоль каждого
        # гребня — разрыв следа выходит пятном разного размера, вытянутым по ходу, а не щелью-рельсой на весь ход
        rng_with = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])

        def lumps(length, passes):
            sh = _noise(rng_with, Nc, nr, max(length, 2.0 * spacing_brush) / spacing_brush)
            for _ in range(passes):
                q = surround(sh, ((0, 0), (1, 1)), edge=True)
                sh = 0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]
            return sh / max(float(sh.std()), 1e-6)

        uneven = (1.0 + _WithAtCanvasAYa[3] * lumps(0.9 * W, 2 * _WithAtCanvasAYa[6]) + _WithAtCanvasAYa[4] * lumps(0.3 * W, _WithAtCanvasAYa[6]) + _WithAtCanvasAYa[5] * lumps(5.0, 0)
                  + 0.20 * brush.scraps * _noise(rng_with, Nc, ns, 90.0 / spacing_brush)[:, with_.strand])
    return uneven, thickness


def _segments(q, lengths):
    """Счёт q (N, n), растущий вдоль хода, разбит у каждого столбца на отрезки своей длины (K, n).
    Ответ: номер отрезка (N, n) и сколько пройдено от его начала (в единицах q)."""
    K, n = lengths.shape
    ends = np.cumsum(lengths, axis=0)
    large = float(max(ends[-1].max(), q.max())) * 2.0 + 1.0
    shift = np.arange(n) * large
    k = np.searchsorted((ends + shift[None, :]).T.ravel(), (q + shift[None, :]).ravel()).reshape(q.shape) - np.arange(n) * K
    k = clamp(k, 0, K - 1)
    columns = np.arange(n)[None, :]
    start = np.where(k > 0, ends[np.maximum(k - 1, 0), columns], 0.0)
    return k, q - start


def _reserve_tip(with_, consumption_in, ds, live, live_sr, fine, rng):
    """Полнота кончика каждого ведущего волоска вдоль хода, 0..1 (N, nv). Кончик пустеет, пока кладёт, и наполняется
    из запаса выше по волоску; на сколько его хватает и до какой доли он пустеет — у каждого раза своё. Тонкая
    краска под прижатой кистью стекает на кончик плавно, густая под лёгкой рукой сходит комком — разом.
    Половина запаса — общая у волосков пряди (`_БОР_ПРЯДЬ`): прядь набрана одним комком."""
    N, nv = consumption_in.shape
    suffices = (_FURROW_RESERVE[0] + (_FURROW_RESERVE[1] - _FURROW_RESERVE[0]) * live) * fine         # (N,) px
    fills = _FURROW_RESERVE[5] + (_FURROW_RESERVE[6] - _FURROW_RESERVE[5]) * live_sr

    def wave(consumption, reserve):
        n = consumption.shape[1]
        q = np.cumsum(consumption / suffices[:, None], axis=0) * ds / reserve[None, :] ** 0.5   # в «запасах кончика»
        K = int(q.max() * 3.2) + 4
        lengths = np.exp(clamp(_FURROW_RESERVE[2] * rng.standard_normal((K, n)), -1.0, 1.0))
        to = rng.uniform(_FURROW_RESERVE[3], _FURROW_RESERVE[4], (K, n))
        k, from_starts = _segments(q + lengths[0] * rng.uniform(0.0, 1.0, n), lengths)      # кончики встают на холст разными
        columns = np.arange(n)[None, :]
        t = clamp(from_starts / lengths[k, columns], 0.0, 1.0)
        was = np.where(k > 0, to[np.maximum(k - 1, 0), columns], to[K - 1, columns])
        z = clamp(t / fills, 0.0, 1.0)
        grows = was + (1.0 - was) * z * z * (3.0 - 2.0 * z)
        falls = 1.0 - (1.0 - to[k, columns]) * clamp((t - fills) / (1.0 - fills), 0.0, 1.0)
        return np.where(t < fills, grows, falls)

    own = wave(consumption_in, 0.5 * (with_.reserve[:-1] + with_.reserve[1:]))
    strand_in = with_.strand[:-1]
    consumption_p = np.stack([consumption_in[:, strand_in == q].mean(axis=1) if (strand_in == q).any() else np.zeros(N) for q in range(with_.ns)], axis=1)
    shared = wave(consumption_p, np.ones(with_.ns))[:, strand_in]
    return _FURROW_STRAND * shared + (1.0 - _FURROW_STRAND) * own


def _skip(with_, consumption_in, ds, live, fine, rng, on_ridge):
    """Проскок волоска (N, nv): 1 — волосок в воздухе, борозды под ним нет; и капля (N, nv), которую он оставляет.
    Цепляется волосок тем чаще, чем гуще краска и легче рука (прижатый волосок идёт ровнее); жёсткий — чаще мягкого.
    На гребне лежащей краски (`на_гребне`, N×nv, 0..1) волосок взлетает всегда.
    Каждый проскок свой (4б): волосок отходит от холста не разом — борозда мелеет и сужается на 0,6–2,2 px и так же
    возвращается, поэтому у срыва нет стенки поперёк хода (под косым светом стенки соседних борозд складывались
    в штриховку в одну сторону). Капля — то перед срывом (краска, что волосок нагнал, зацепившись), то за ним
    (сошла с волоска, когда он сел), то её нет; величина у каждой своя. `мелк` — мелкость зерна: у тонкой краски
    срыв короче и капля меньше."""
    N, nv = consumption_in.shape
    frequency = np.maximum(_FURROW_SKIP[0] / 100.0 * live ** _FURROW_SKIP[1], 1e-6)                     # (N,) раз на px
    q = np.cumsum(frequency[:, None] * (consumption_in > 0.05) * clamp(with_.presses, 0.6, 1.6)[None, :], axis=0) * ds
    K = int(q.max() * 2.5) + 4
    lengths = np.exp(clamp(0.7 * rng.standard_normal((K, nv)), -1.5, 1.5) - 0.245)
    lengths[0] *= rng.uniform(0.2, 1.0, nv)
    k, from_starts = _segments(q, lengths)
    px = from_starts / frequency[:, None]                                                 # px пути от начала отрезка
    columns = np.arange(nv)[None, :]

    def own(from_, to):
        return rng.uniform(from_, to, (K, nv))[k, columns]

    length = own(_FURROW_SKIP[2], _FURROW_SKIP[3]) * fine
    up = own(_FURROW_LIFT[0], _FURROW_LIFT[1]) * fine
    down = own(_FURROW_LIFT[0], _FURROW_LIFT[1]) * fine
    z = clamp(px / up, 0.0, 1.0) * clamp((length + down - px) / down, 0.0, 1.0)
    # волосок отходит от холста не всегда целиком: чаще борозда только мелеет
    in_air = (k > 0) * z * z * (3.0 - 2.0 * z) * own(_FURROW_LIFT[5], 1.0)
    # капля: у каждого проскока своя — перед срывом, за ним или никакой
    lot = own(0.0, 1.0)
    size = own(0.0, 1.0) ** 1.5 * 2.5 * fine
    patch = own(_FURROW_DROP[0], _FURROW_DROP[1]) * fine                                 # длина капли, px
    slope = _FURROW_DROP[2] * fine                               # пологие скаты: комок, а не ступенька с тенью
    before = clamp(px / slope, 0.0, 1.0) * clamp((patch - px) / slope, 0.0, 1.0)
    behind = clamp((px - length - 0.5 * down) / slope, 0.0, 1.0) * clamp((length + 0.5 * down + patch - px) / slope, 0.0, 1.0)
    before, behind = before * before * (3.0 - 2.0 * before), behind * behind * (3.0 - 2.0 * behind)
    settled = (k > 0) * size * np.where(lot < _FURROW_LIFT[2], before, np.where(lot < _FURROW_LIFT[3], behind, 0.0))
    if on_ridge is not None:
        # гребень лежащей краски поднимает волосок; он тянется в воздухе ещё пару px за гребнем
        tail = max(int(round(_FURROW_RIDGE[1] / ds)), 1)
        pulls = on_ridge.copy()
        for i in range(1, tail + 1):
            pulls[i:] = np.maximum(pulls[i:], on_ridge[:-i] * (1.0 - i / (tail + 1.0)))
        in_air = np.maximum(in_air, pulls)
    return in_air, settled * _FURROW_SKIP[4]


def _sticking(with_, brush, live, N, ds, W, rng):
    """Слипание соседних волосков (N, nr): 1 — два волоска по бокам гребня слиплись, гребень между ними просел.
    Слипаются волоски одной пряди, в густой краске при лёгкой руке; прижатая щетина расходится. Крайние гребни
    не трогаются."""
    threshold = _FURROW_STICKING[0] + _FURROW_STICKING[1] * (1.0 - live) + 0.6 * (1.0 - min(brush.scraps / 0.6, 1.0))
    sticking = clamp((_noise(rng, N, with_.nr, _FURROW_STICKING[2] * W / ds) - threshold[:, None]) / 0.3, 0.0, 1.0)
    inside = np.zeros(with_.nr)
    inside[1:-1] = (with_.strand[1:-1] == with_.strand[:-2]) & (with_.strand[1:-1] == with_.strand[2:])
    sticking = sticking * inside[None, :]
    sticking[:, 1:] *= 1.0 - (sticking[:, :-1] > sticking[:, 1:])          # два соседних гребня разом не проседают
    return sticking * sticking * (3.0 - 2.0 * sticking)


def furrows(with_, brush, path, p, touch, thickly, rng, X, Y, drop):
    """Жизнь борозд одного мазка. кас — касание гребней (N, nr); X, Y — места выборок профиля в прямоугольнике
    мазка (N, nm); перепад — рельеф лежащей краски в толщинах плёнки (или None).
    Ответ: None (у кисти борозд нет) или (глубже, шире, капля, слип): первые три — (N, nv) по ведущим волоскам,
    слип — (N, nr) по гребням."""
    if with_.leading is None or with_.nr < 3:
        return None
    ds, N = path['ds'], len(p)
    touch_in = np.minimum(touch[:, :-1], touch[:, 1:])                # волосок стоит между двумя гребнями
    consumption_in = np.sqrt(clamp(p, 0.0, None) / 0.8)[:, None] * touch_in
    # живость: тонкая краска под прижатой кистью даёт ровную борозду с плавной переменой глубины;
    # густая под лёгкой рукой — рваную: комки, проскоки, слипшиеся волоски
    firmly = clamp((p - 0.3) / 0.55, 0.0, 1.0)
    live = (_FURROW_LIVE[0] + (1.0 - _FURROW_LIVE[0]) * thickly) * (1.0 - _FURROW_LIVE[1] * firmly * (1.0 - _FURROW_LIVE[2] * thickly))
    weight = consumption_in.sum(axis=1)
    live_sr = float((live * weight).sum() / max(weight.sum(), 1e-9))
    # мелкость зерна (4б): тонкая краска живёт так же, как густая, но мельче — на кончике её меньше, запас кончается
    # раньше, срыв короче, капля меньше
    fine = _FURROW_FINE + (1.0 - _FURROW_FINE) * thickly
    fullness = _reserve_tip(with_, consumption_in, ds, live, live_sr, fine, rng)
    on_ridge = None
    if drop is not None:
        nv = with_.nr - 1
        m = np.array([int(np.argmin(np.where(with_.leading == v, with_.d_leading, np.inf))) for v in range(nv)])   # выборка у волоска
        ix, iy = np.floor(X[:, m]).astype(np.intp), np.floor(Y[:, m]).astype(np.intp)
        usable = (ix >= 0) & (ix < drop.shape[1]) & (iy >= 0) & (iy < drop.shape[0])
        above = np.where(usable, drop[np.where(usable, iy, 0), np.where(usable, ix, 0)], 0.0)
        on_ridge = clamp((above - _FURROW_RIDGE[0]) / _FURROW_RIDGE[0], 0.0, 1.0) * live[:, None] ** 0.5
    in_air, drop = _skip(with_, consumption_in, ds, live, fine, rng, on_ridge)
    sticking = _sticking(with_, brush, live, N, ds, with_.W, rng)
    # глубина: под полным кончиком мелко, под пустым глубоко; жёсткий волосок пустым скребёт сильнее мягкого
    empty = 1.0 - fullness
    fine = (_FURROW_DEPTH[0] + (_FURROW_DEPTH[1] - _FURROW_DEPTH[0]) * live)[:, None]
    deep = (_FURROW_DEPTH[2] + (_FURROW_DEPTH[3] - _FURROW_DEPTH[2]) * live)[:, None] * clamp(with_.presses, 0.75, 1.25)[None, :]
    deeper = fine + (deep - fine) * empty
    wider = _FURROW_WIDER[0] + ((_FURROW_WIDER[1] + (_FURROW_WIDER[2] - _FURROW_WIDER[1]) * live)[:, None] - _FURROW_WIDER[0]) * empty
    deeper = deeper * (1.0 - in_air)
    wider = wider * (1.0 - _FURROW_LIFT[4] * in_air)            # отходя от холста, волосок касается его всё уже
    # крайние волоски живут по-старому: кромку следа ведут бок и край (`пряди.py`, `_край`), а не борозда
    for v in (0, -1):
        deeper[:, v], wider[:, v], drop[:, v] = 1.0, 1.0, 0.0
    return deeper, wider, drop, sticking


def profile_furrows(with_, furrow, fine):
    """Поперечный профиль плёнки на каждом шаге пути (N, nm): 1 — гребень, меньше — борозда.
    мелко (N, 1) — лёгкая кисть борозду до дна не продавливает. Без `бор` — прежний ровный профиль."""
    if furrow is None:
        return 1.0 - (1.0 - with_.profile[None, :]) * fine
    deeper, wider, drop, sticking = furrow
    j = with_.leading
    t = clamp(with_.d_leading[None, :] / (with_.half_furrow[None, :] * wider)[:, j], 0.0, 1.0)
    wall = 1.0 - t * t * (3.0 - 2.0 * t)                     # 1 — под кончиком волоска, 0 — на гребне
    depth = clamp((1.0 - with_.bottom_in)[None, :] * deeper, 0.0, 1.0 - _FURROW_DEPTH[4])[:, j]
    base = 1.0 - depth * wall
    # слипшиеся волоски: гребень между ними проседает к дну их борозд
    sagged = sticking[:, with_.ridge] * _FURROW_STICKING[3]
    base = base * (1.0 - sagged) + np.minimum(base, 1.0 - depth) * sagged
    profile = 1.0 - (1.0 - base * with_.edge_profile[None, :]) * fine
    # капля шире борозды: комок ложится и на бока соседних гребней — точка, а не чёрточка в борозде
    t = clamp(with_.d_leading[None, :] / (with_.half_furrow[None, :] * _FURROW_LIFT[6])[:, j], 0.0, 1.0)
    return profile + (drop[:, j] * (1.0 - t * t * (3.0 - 2.0 * t))) * with_.edge_profile[None, :]


def tone_furrows(with_, furrow, profile, mass, visible, undermixing, ds):
    """Борозда цветом (версия 3, пункты 5 и 5в): тон идёт за живой глубиной борозды (N, nm, 3), множитель к цвету
    гребня. Краска на кисти недомешана (`недомес` — `разброс_тона` кисти): плотная светлая её часть сходит с полного
    кончика, жидкая тёмная остаётся там, где волосок скребёт пустым. Тёмное — двух родов: место в два-три волоска,
    где кончики пусты (пятно длиной в запас кончика, а не лента во весь ход), и дно глубокой борозды (тонкая линия,
    что рвётся там, где кончик наполнился). `бор` — жизнь борозд (`борозды`), `профиль` (N, nm) — их поперечный
    профиль; `масса` и `видно` (N, nm) — что мазок кладёт и его покрытие: средний цвет каждого сечения мазка
    на глаз не меняется (тон не темнит тонкий конец мазка и тонкие плёнки целиком)."""
    if with_.leading is None or furrow is None or undermixing <= 0.0:
        return None
    scrapes = furrow[0]
    for _ in range(_FURROW_TONE_WEIGHT[0]):
        q = surround(scrapes, ((0, 0), (1, 1)), edge=True)
        scrapes = 0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]
    # пятно не длиннее запаса кончика: медленная часть (волосок пустеет на всём ходу, период запаса) вычтена —
    # иначе под светом вдоль длинного мазка встаёт период (`test_нет_периода_вдоль_мазка`)
    L = int(_FURROW_TONE_WEIGHT[2] / ds) | 1
    q = np.cumsum(surround(scrapes, ((L // 2 + 1, L // 2), (0, 0)), edge=True), axis=0)
    scrapes = scrapes - (q[L:] - q[:-L]) / L
    dark = _FURROW_TONE[0] * scrapes[:, with_.leading] + _FURROW_TONE[1] * clamp(1.0 - profile - _FURROW_TONE[2], 0.0, 1.0)
    weight = mass ** _FURROW_TONE_WEIGHT[1] * visible ** (1.0 - _FURROW_TONE_WEIGHT[1])
    mean = (dark * weight).sum(axis=1, keepdims=True) / np.maximum(weight.sum(axis=1, keepdims=True), 1e-9)
    shift = -undermixing * clamp(dark - mean, -_FURROW_TONE[3], _FURROW_TONE[3])
    return np.exp(shift[:, :, None] * np.array(_FURROW_TONE[4])[None, None, :])
