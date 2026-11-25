# READING VIEW of движок/кисть/путь.py with names in English. Not runnable; edit the original.
"""Путь и жесты: точки руки -> сплайн -> равный шаг; нажим, скорость, угол кисти."""
import hashlib

import numpy as np

from .fast import clamp, surround

from .numbers import _SPACING_PATH, _ShBrush, _TYHResponseBrush, _JAB_CONSUMPTION, _JAB_CIRCLE, _JAB_TUFT, _JAB_SHARP, _A_PRESSURE, _BrushResponseRResponseTBrushAndY, _ResponseWithTRAYa


# ------------------------------------------------------- компактная строка (А-3)
class Sweep(np.ndarray):
    """Точки ядра от `жест`, `путь_с_нажимом`, `тычок` — обычный массив, который помнит, из чего он построен: опорные
    точки, конец, нажим, вход, ширина, сид или состояние генератора. Журнал холста пишет тогда строку "ctrl" (опорные
    точки), а не сотни точек "core": повтор строит по ней те же точки бит в бит. Точки поправили (на месте или новым
    массивом) — память не годится, и журнал пишет "core", как раньше."""

    def __array_finalize__(self, obj):
        self.sweep = None          # срез, копия и любая арифметика — уже другие точки

    def row(self):
        """Поля строки журнала вместо "core" или None — точки уже не те, из которых строились."""
        sweep = getattr(self, 'ход', None)
        if sweep is None or _imprint(self) != sweep[1]:
            return None
        return dict(sweep[0])


def _imprint(a):
    a = np.ascontiguousarray(a, dtype=np.float64)
    return (a.shape, hashlib.blake2b(a.tobytes(), digest_size=16).digest())


def _generator(rng, seed):
    """rng и сид вызова -> (генератор, поля строки журнала, по которым повтор возьмёт тот же). Сид — как у повтора по
    строке с "seed": `default_rng([сид, 7])`; генератор PCG64 — его состояние до вызова ("path_rng")."""
    if seed is not None:
        if rng is not None:
            raise ValueError('путь: rng или сид — что-то одно')
        return np.random.default_rng([int(seed), 7]), {'path_seed': int(seed)}
    if rng is None:
        return np.random.default_rng(0), {'path_rng': None}
    bg = getattr(rng, 'bit_generator', None)
    if isinstance(bg, np.random.PCG64):
        with_ = bg.state
        return rng, {'path_rng': [with_['state']['state'], with_['state']['inc'], with_['has_uint32'], with_['uinteger']]}
    return rng, None                 # другой генератор: строки "ctrl" не будет, журнал пишет "core"


def generator_sweep(field):
    """Повтор: "path_rng" строки -> генератор PCG64 в том же состоянии (None — `default_rng(0)`, как без rng)."""
    if field is None:
        return np.random.default_rng(0)
    g = np.random.Generator(np.random.PCG64())
    g.bit_generator.state = {'bit_generator': 'PCG64', 'state': {'state': int(field[0]), 'inc': int(field[1])},
                             'has_uint32': int(field[2]), 'uinteger': int(field[3])}
    return g


def _with_sweep(points, fields):
    t = np.ascontiguousarray(points, dtype=np.float64).view(Sweep)
    t.sweep = None if fields is None else (fields, _imprint(t))
    return t


# -------------------------------------------------------------------------- путь
def _hermite(t, P, tt):
    n = len(t)
    m = np.empty_like(P)
    if n > 2:
        m[1:-1] = (P[2:] - P[:-2]) / (t[2:] - t[:-2])[:, None]
    m[0] = (P[1] - P[0]) / (t[1] - t[0])
    m[-1] = (P[-1] - P[-2]) / (t[-1] - t[-2])
    i = clamp(np.searchsorted(t, tt, side='right') - 1, 0, n - 2)
    h = (t[i + 1] - t[i])[:, None]
    s = ((tt - t[i]) / h[:, 0])[:, None]
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * P[i] + (s3 - 2 * s2 + s) * h * m[i]
            + (-2 * s3 + 3 * s2) * P[i + 1] + (s3 - s2) * h * m[i + 1])


def _window(a, half_width):
    """Сглаживание окном Ханна, края держатся за крайние значения."""
    k = int(half_width)
    if k < 1 or len(a) < 3:
        return a
    w = np.hanning(2 * k + 3)[1:-1]
    w /= w.sum()
    return np.convolve(surround(a, k, edge=True), w, mode='valid')


def _axial(ctrl, spacing=2.0):
    """Опорные точки -> плотная ломаная по сплайну и длина дуги в её точках."""
    P = np.asarray(ctrl, dtype=np.float64)[:, :2]
    if len(P) > 1:
        P = P[np.concatenate([[True], np.hypot(*np.diff(P, axis=0).T) > 1e-6])]
    if len(P) == 1:
        return P, np.zeros(1)
    chords = np.hypot(*np.diff(P, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(np.sqrt(chords))])
    n = max(8, int(chords.sum() / spacing) + 2)
    Q = _hermite(t, P, np.linspace(0.0, t[-1], n))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    return Q, s


def _short(L, W, length, own, share):
    """Вход или сход короткого хода («касание и короткий ход»). Длины входа и схода нажима взяты с длинного хода и на
    коротком съедали его весь (0,32 и 0,5 длины): нажим доходил до названного в одной точке, углы плоской кисти не
    вставали (`пряди._посадка`: вход длиннее «удара»), и ход в ширину кисти выходил в три четверти ширины, веретеном.
    Короткий ход рука кладёт иначе: ставит кисть, тянет и снимает. Ход не длиннее `_КОРОТКИЙ[1]` ширин — вход и сход
    по `доля` ширины кисти (`_КОРОТКИЙ[2]` и `[3]`); до `_КОРОТКИЙ[0]` ширин — плавно к прежним; от них и дальше —
    как было, бит в бит. `своя` — длину дал вызов (`вход=`, `сход=`): она остаётся."""
    W = float(W)
    if own or W <= 0 or L >= _BrushResponseRResponseTBrushAndY[0] * W:
        return length
    brush = min((_BrushResponseRResponseTBrushAndY[0] - L / W) / (_BrushResponseRResponseTBrushAndY[0] - _BrushResponseRResponseTBrushAndY[1]), 1.0)
    return length + (min(length, share * W) - length) * brush * brush


def path_with_pressure(ctrl, size, rng=None, pressure=0.85, speed=1.0, entry=None, exit=None, seed=None):
    """По опорным точкам [[x, y], ...] — точки (x, y, нажим, dt_мс) с нажимом «вход — тело — сход».

    размер — ширина кисти, px: от неё берутся длины входа и схода, если они не заданы.
    Заданные вход и сход берутся как есть (короткий сход — быстрый отрыв).
    сид (А-3) — вместо rng: те же точки, что повтор строит по строке {"ctrl", "size", "seed": сид}. С сидом или с
    генератором numpy (`default_rng`) журнал холста пишет мазок строкой "ctrl" (опорные точки), а не точками ядра."""
    rng, fields = _generator(rng, seed)
    fields = _fields_sweep(ctrl, size, pressure, entry, None, fields) if speed == 1.0 and exit is None else None
    return _with_sweep(_path_with_pressure(ctrl, size, rng, pressure, speed, entry, exit), fields)


def _fields_sweep(ctrl, size, pressure, entry, end, fields):
    """Поля строки "ctrl" — те, по которым повтор зовёт `жест` / `путь_с_нажимом` (`кисть/повтор.py`)."""
    if fields is None:
        return None
    P = np.asarray(ctrl, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] < 2 or not (np.hypot(*np.diff(P[:, :2], axis=0).T) > 1e-6).any():
        return None                  # на месте: повтор кладёт по одной точке тычок (`тычок`), не этот путь
    d = {'ctrl': P[:, :2].tolist(), 'size': float(size)}
    if end is not None:
        d['end'] = end
    if float(pressure) != 0.85:
        d['pressure'] = float(pressure)
    if entry is not None:
        d['entry'] = float(entry)
    d.update(fields)
    return d


def _path_with_pressure(ctrl, size, rng, pressure=0.85, speed=1.0, entry=None, exit=None):
    Q, s = _axial(ctrl)
    if len(Q) == 1:
        return np.array([[Q[0, 0], Q[0, 1], pressure, 30.0]])
    L = s[-1]
    vkh = max((0.5 + 1.2 * rng.random()) * size, 6.0) if entry is None else float(entry)
    skh = max((1.4 + 1.8 * rng.random()) * size, 14.0) if exit is None else float(exit)
    vkh = _short(L, size, min(vkh, 0.32 * L), entry is not None, _BrushResponseRResponseTBrushAndY[2])
    skh = _short(L, size, min(skh, 0.5 * L), exit is not None, _BrushResponseRResponseTBrushAndY[3])
    a = clamp(s / max(vkh, 1e-6), 0.0, 1.0)
    b = clamp((L - s) / max(skh, 1e-6), 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    b = b ** 1.2
    body = 1.0 + 0.07 * np.sin(s / max(L, 1.0) * (2.0 + 3.0 * rng.random()) + 6.28 * rng.random())
    p = clamp(pressure * a * b * body, 0.0, 1.0)
    v = speed * (0.55 + 0.45 * np.minimum(a, b))          # на входе и сходе рука медленнее
    dt = np.concatenate([[0.0], np.diff(s) / v[1:]])
    return np.column_stack([Q[:, 0], Q[:, 1], p, dt])


ZhEWithTY = ('подъём', 'отрыв', 'остановка', 'смахивание')


def gesture(ctrl, size, kind='подъём', rng=None, pressure=0.85, speed=1.0, entry=None, seed=None):
    """Точки мазка с одним из четырёх концов. Конец задаётся жестом руки, а не формой следа:

    'подъём'      — кисть медленно поднимают на ходу: нажим сходит на двух-четырёх ширинах;
    'отрыв'       — кисть срывают: нажим падает на четверти ширины, рука ускоряется;
    'остановка'   — кисть останавливают, придавливают и поднимают на месте;
    'смахивание'  — конец уводят вбок дугой, ускоряясь и отпуская.
    сид (А-3) — вместо rng: те же точки, что повтор строит по строке {"ctrl", "size", "end": вид, "seed": сид}. С сидом
    или с генератором numpy журнал холста пишет мазок строкой "ctrl", а не точками ядра."""
    if kind not in ZhEWithTY:
        raise ValueError(f'жест: один из {ZhEWithTY}')
    rng, fields = _generator(rng, seed)
    fields = _fields_sweep(ctrl, size, pressure, entry, kind, fields) if speed == 1.0 else None
    return _with_sweep(_gesture(ctrl, size, kind, rng, pressure, speed, entry), fields)


def _gesture(ctrl, size, kind, rng, pressure=0.85, speed=1.0, entry=None):
    W = float(size)
    if kind == 'смахивание':
        P = np.asarray(ctrl, dtype=np.float64)[:, :2]
        d = P[-1] - P[-2]
        d /= max(np.hypot(*d), 1e-9)
        side = np.array([-d[1], d[0]]) * (1.0 if rng.random() < 0.5 else -1.0)
        r = (0.9 + 0.6 * rng.random()) * W
        ctrl = np.vstack([P, P[-1] + d * 0.75 * r + side * 0.3 * r, P[-1] + d * 1.05 * r + side * 1.05 * r])
    Q, s = _axial(ctrl, spacing=1.0)
    L = s[-1]
    vkh = max((0.5 + 1.2 * rng.random()) * W, 6.0) if entry is None else float(entry)
    a = clamp(s / max(_short(L, W, min(vkh, 0.32 * L), entry is not None, _BrushResponseRResponseTBrushAndY[2]), 1e-6), 0.0, 1.0)
    a = a * a * (3 - 2 * a)
    body = 1.0 + 0.07 * np.sin(s / max(L, 1.0) * (2.0 + 3.0 * rng.random()) + 6.28 * rng.random())
    d = L - s                                               # до конца пути
    v = np.full(len(s), float(speed))
    if kind == 'подъём':
        skh = _short(L, W, min((2.0 + 2.0 * rng.random()) * W, 0.5 * L), False, _BrushResponseRResponseTBrushAndY[3])
        b = clamp(d / skh, 0.0, 1.0) ** 1.2
        v *= 0.6 + 0.4 * clamp(d / skh, 0, 1)
    elif kind == 'отрыв':
        skh = max(0.25 * W, 3.0)
        b = clamp(d / skh, 0.0, 1.0)
        v *= 1.0 + 1.5 * (1.0 - clamp(d / (1.5 * W), 0, 1))
    elif kind == 'остановка':
        brake = min(0.6 * W, 0.4 * L)
        b = clamp(d / 2.0, 0.0, 1.0)                      # подъём на последних двух пикселях
        pressed = 1.0 + 0.3 * (1.0 - clamp(d / brake, 0, 1)) ** 2
        b = b * pressed
        v *= 0.06 + 0.94 * clamp(d / brake, 0, 1) ** 1.5
    else:
        skh = min(1.6 * r, 0.5 * L)
        b = clamp(d / skh, 0.0, 1.0) ** 1.1
        v *= 1.0 + 2.0 * (1.0 - clamp(d / skh, 0, 1))
    p = clamp(pressure * a * b * body, 0.0, 1.0)
    dt = np.concatenate([[0.0], np.diff(s) / v[1:]])
    return np.column_stack([Q[:, 0], Q[:, 1], p, dt])


def profile_pressure(profile, where='pressure_along'):
    """Проверить профиль нажима по ходу — [[доля пути, множитель], …]: доли 0–1 строго по возрастанию, множители
    не меньше 0. Ответ — массив (n, 2) или None (профиля нет). Кривой — ValueError с `где` в начале."""
    if profile is None:
        return None
    try:
        p = np.asarray(profile, dtype=np.float64)
    except (TypeError, ValueError):
        p = None
    if p is None or p.ndim != 2 or p.shape[1] != 2 or len(p) < 1 or not np.isfinite(p).all():
        raise ValueError(f'{where}: должен быть списком пар [доля пути, множитель]')
    if (p[:, 0] < 0).any() or (p[:, 0] > 1).any():
        raise ValueError(f'{where}: доля пути вне 0–1')
    if (np.diff(p[:, 0]) <= 0).any():
        raise ValueError(f'{where}: доли пути не по возрастанию')
    if (p[:, 1] < 0).any():
        raise ValueError(f'{where}: множитель нажима меньше 0')
    return p


def pressure_along(points, profile):
    """Нажим по ходу (п. 11д, поле журнала "pressure_along"): нажим точек мазка (x, y, нажим, …) множится на профиль
    [(доля пути, множитель), …] по доле пройденного пути (длина ломаной точек), между узлами по прямой (np.interp).
    Счёт тот же, что у анализатора (`подбор.вдоль`). Профиля нет, у точек нет нажима или точек меньше двух — точки
    как были (тот же объект). Профиль проверяется (`профиль_нажима`)."""
    p = profile_pressure(profile)
    t = np.array(points, np.float64)
    if p is None or t.ndim != 2 or t.shape[1] < 3 or len(t) < 2:
        return points
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t[:, :2], axis=0).T))])
    t[:, 2] *= np.interp(s / max(s[-1], 1e-9), p[:, 0], p[:, 1])
    return t


def segments_touch(points, profile):
    """Отрыв по ходу (п. 11ж, поле журнала "pressure_along"): где множитель профиля 0 — кисть над холстом.
    Ответ — список кусков точек (x, y, нажим, dt_мс, …): каждый от отрыва до отрыва, нажим на концах куска,
    стоящих у нуля профиля, — 0, время на месте отрыва — по ходу руки. Нулей в профиле нет (или точек меньше двух) —
    None: мазок один, как был. Точки — уже с профилем (`нажим_вдоль`); места нулей берутся из самого профиля,
    а не из точек: ноль на одном узле, между точками пути, отрывает тоже."""
    p = profile_pressure(profile)
    t = np.asarray(points, np.float64)
    if p is None or t.ndim != 2 or t.shape[1] < 3 or len(t) < 2 or not (p[:, 1] == 0).any():
        return None
    # нулевые участки профиля по доле пути: [a, b] (a == b — ноль на одном узле); за крайними узлами — их множитель
    zeros = []
    i = 0
    while i < len(p):
        if p[i, 1] == 0:
            j = i
            while j + 1 < len(p) and p[j + 1, 1] == 0:
                j += 1
            a = 0.0 if i == 0 else p[i, 0]
            b = 1.0 if j == len(p) - 1 else p[j, 0]
            zeros.append((a, b))
            i = j + 1
        else:
            i += 1
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t[:, :2], axis=0).T))])
    d = s / max(s[-1], 1e-9)
    time = np.cumsum(t[:, 3]) if t.shape[1] >= 4 else None
    pieces = []
    start = 0.0
    for a, b in [*zeros, (1.0, 1.0)]:
        if a > start:
            inside = (d > start) & (d < a)
            def point(share):
                return np.array([np.interp(share, d, t[:, k]) for k in range(t.shape[1])])
            piece = np.vstack([point(start), t[inside], point(a)])
            if start > 0:
                piece[0, 2] = 0.0                    # кисть входит заново от нуля
            if a < 1.0:
                piece[-1, 2] = 0.0                   # и уходит в ноль у отрыва
            if time is not None:
                t_piece = np.interp(np.concatenate([[start], d[inside], [a]]), d, time)
                piece[:, 3] = np.concatenate([[t[0, 3] if start == 0 else 0.0], np.diff(t_piece)])
            if (piece[:, 2] > 0).any():
                pieces.append(piece)
        start = max(start, b)
    return pieces


def jab(x, y, pressure=0.85, angle=None, ms=30.0):
    """Тычок — касание на месте (З-10): точки ядра одной точки, те же, что повтор кладёт по строке
    `{"ctrl": [[x, y]], "pressure": нажим, "angle": угол}`. угол — направление кисти, градусы от оси x (y вниз), широкая
    сторона плоской кисти поперёк него; None — кисть стоит как стоит (то же, что `[(x, y, нажим, мс)]`). Без нажима и
    угла тычок — просто `[(x, y)]`, как `{"ctrl": [[x, y]]}` и `{"points": [[x, y]]}`: нажим строит ядро."""
    fields = None
    if float(ms) == 30.0:            # журнал: строка {"ctrl": [[x, y]], "pressure", "angle"} — повтор кладёт этот тычок
        fields = {'ctrl': [[float(x), float(y)]]}
        if float(pressure) != 0.85:
            fields['pressure'] = float(pressure)
        if angle is not None:
            fields['angle'] = float(angle)
    if angle is None:
        return _with_sweep([[float(x), float(y), float(pressure), float(ms)]], fields)
    at = float(angle)
    return _with_sweep([[float(x), float(y), float(pressure), float(ms), 0.0, np.radians(at), at + 90.0]], fields)


def polyline(points, spacing=0.5):
    """К2, сырые точки руки: путь как есть. Точки (x, y, нажим[, dt_мс, наклон °, азимут рад, поворот °]) — те же, но
    каждый отрезок длиннее `шаг` px разбит на равные части не длиннее его: нажим и углы — по прямой между точками
    (азимут и поворот — кратчайшей дугой, как их читает ядро), время отрезка делится поровну. Ядро (`_путь`) ведёт
    сплайн по своим точкам; по точкам через полпикселя он идёт по самой ломаной, и угол руки остаётся углом, а не
    дугой. Точки и так не реже шага (и одна точка) — те же (тот же объект). Точка, повторённая на месте, остаётся:
    стояние кисти. Входа и схода нажима здесь нет: какие дала рука."""
    t = np.asarray(points, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 3 or len(t) == 0:
        raise ValueError('ломаная: нужен список точек (x, y, нажим, ...)')
    if not np.isfinite(t).all():
        raise ValueError('ломаная: есть нечисла')
    if len(t) == 1:
        return points
    chords = np.hypot(*np.diff(t[:, :2], axis=0).T)
    m = np.maximum(np.ceil(chords / float(spacing) - 1e-9), 1.0)
    if (m == 1).all():
        return points
    if m.sum() > 4_000_000:
        raise ValueError('ломаная: путь длиннее 2 млн px')
    m = m.astype(np.int64)
    t = t.copy()
    if t.shape[1] >= 6:
        t[:, 5] = np.unwrap(t[:, 5])
    if t.shape[1] >= 7:
        t[:, 6] = np.degrees(np.unwrap(np.radians(t[:, 6])))
    start = np.concatenate([[0], np.cumsum(m)])
    i = np.repeat(np.arange(len(m)), m)                       # отрезок новой точки
    j = np.arange(start[-1]) - start[i] + 1                       # её номер в отрезке, 1…m
    new = t[i] + (t[i + 1] - t[i]) * (j / m[i])[:, None]
    new[j == m[i]] = t[1:]                                    # точки руки — сами, без округления
    if t.shape[1] >= 4:
        new[:, 3] = t[i + 1, 3] / m[i]
    return np.vstack([t[:1], new])


def reserve_fringe(W):
    """Н-2: во сколько раз путь тычка круглой длиннее круга её пучка — бахроме кончиков (`_ТЫЧОК_ПУЧОК[2]`, три σ)
    есть где лечь, и конец пути круг не режет прямой. Узкая кисть до 16 px — без запаса: бахрома у неё меньше пикселя,
    а капля посадки ложится на всю длину пути (с запасом пятно 8–10 px выходило длиннее ширины на 13–25 %); к 24 px — полный."""
    return 1.0 + 3.0 * _JAB_TUFT[2] * float(clamp((float(W) - 16.0) / 8.0, 0.0, 1.0))


def _imprint_tuft(point, brush, along):
    """Путь тычка: кисть, поставленная на месте, оставляет отпечаток пучка — кончики проходят его длину по ходу кисти.
    Плоская (и мягкая, и узкая): пучок встаёт всей кромкой разом и разом уходит — нажим на всей длине отпечатка тот,
    что дала рука (вход короче `пряди._посадка` «удара»: углы обоймы встают сразу); длина — толщина пучка и расход
    кончиков под нажимом (`_ТЫЧОК[0]`, `_ТЫЧОК_РАСХОД`). Круглая: пятно касания растёт от кончика к брюшку, пока кисть
    давят; нажим на всём отпечатке тот, что дала рука, а касается тот волосок пучка, что достаёт до холста на своём
    расстоянии от оси пучка в круге сечения (Н-2, `укладка._порог_пучка`); длина — ширина следа при этом нажиме
    (`укладка._расп`: распластывание кисти). Ответ — точки ядра (n, столбцов как у точки), нажим на концах 0."""
    W, pressure = float(brush.width), float(clamp(point[2], 0.0, 1.0))
    sharp = float(clamp((_ResponseWithTRAYa[0] - W) / _ResponseWithTRAYa[1], 0.0, 1.0)) * float(getattr(brush, 'остриё', 1.0))   # как `укладка._острая`: узкая кисть с острым кончиком
    if brush.shape == 'круглая':
        spread = float(getattr(brush, 'распластывание', 0.4))
        wider = spread + (1.0 - spread) * (pressure / 0.85) ** 0.6 * (1.0 + 0.9 * float(clamp(pressure - 0.85, 0.0, 0.15)))
        # касание-2: у узкой кисти с острым кончиком порог крайних гребней ниже (`укладка._остриё`, `_ОСТРАЯ[5]`) — нажим по
        # хорде считается от него же, иначе все гребни вставали на всей длине отпечатка и пятно выходило бруском
        A = _A_PRESSURE * (1.0 - _ResponseWithTRAYa[5] * sharp)
        reaches = min(max((pressure - 0.04) / A, 0.0) ** (1.0 / 1.15), 1.0)
        length = max(_JAB_CIRCLE[0] * reaches * wider * W, 1.6) * reserve_fringe(W)   # Н-2: с запасом на бахрому — круг пучка не режет конец пути
    else:
        # + 1 px: нажим встаёт и уходит на крайних полпикселя пути. Касание-2: у узкой кисти с острым кончиком —
        # короче на `_ТЫЧОК_ОСТР` px: её ворс собран, кончики встают разом (с целым пикселем отпечаток узкой плоской
        # 12 px выходил квадратом 9 × 10 px, и угол тычка по пятну не читался; без него — короче 0,6 ширины, К8.2)
        length = max((_TYHResponseBrush[0] + _JAB_CONSUMPTION * pressure) * W + 1.0 - _JAB_SHARP * sharp, 1.6)
        if getattr(brush, 'просветы', 0.0) > 0:
            # К-14б: кисть из пучков (веерная, растрёпанная) — толщина отпечатка идёт от толщины её пучка
            # (`укладка.толщина_тычка`, К-14б-2: пучок обычной толщины — как плоская той же ширины; у веера в один ряд,
            # `пучок_мм` 0, — 0: отпечаток — один расход кончиков под нажимом)
            from .laying import thickness_jab
            length = max(thickness_jab(brush) + _JAB_CONSUMPTION * pressure * W + 1.0, 1.6)
    n = max(int(np.ceil(length / _SPACING_PATH)) + 1, 5)
    share = np.linspace(-1.0, 1.0, n)
    p = np.full(n, pressure)   # Н-2: круглая — нажим руки на всём отпечатке; пятно касания — круг пучка (`укладка._порог_пучка`)
    p[0] = p[-1] = 0.0
    t = np.repeat(np.asarray(point, dtype=np.float64)[None, :], n, axis=0)
    t[:, :2] += 0.5 * length * share[:, None] * along[None, :]
    t[:, 2] = p
    t[:, 3] = 30.0 / n
    return t


def _path(points, brush, rng):
    t = np.asarray(points, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 2 or len(t) == 0:
        raise ValueError('точки мазка: нужен список (x, y, нажим, dt_мс, ...)')
    if not np.isfinite(t).all():
        raise ValueError('точки мазка: есть нечисла')
    if t.shape[1] == 2:
        t = _path_with_pressure(t, brush.width, rng)
    if t.shape[1] == 3:
        t = np.column_stack([t, np.zeros(len(t))])
    steps = np.hypot(np.diff(t[:, 0]), np.diff(t[:, 1]))
    # стояние на месте: время повторной точки уходит следующему шагу, нажим берётся наибольший
    hold = np.concatenate([[True], steps > 1e-6])
    if not hold.all():
        t = t.copy()
        number = np.cumsum(hold) - 1
        time = np.bincount(number, weights=t[:, 3])
        pressures = np.zeros(number[-1] + 1)
        np.maximum.at(pressures, number, t[:, 2])
        t = t[hold]
        t[:, 3] = time
        t[:, 2] = pressures
    has_rotation = t.shape[1] >= 7
    on_place = len(t) == 1
    if on_place:
        # касание на месте — отпечаток пучка: ряд кончиков проходит его длину по ходу кисти (п. 10, К8.2), у кисти любой
        # ширины («касание и короткий ход»: шире 16 px отпечаток сходил на нет, и круглая 22 px ставила штрих 21×7 px)
        angle = np.radians(t[0, 6]) - np.pi / 2 if has_rotation else rng.uniform(0, 2 * np.pi)
        t = _imprint_tuft(t[0], brush, np.array([np.cos(angle), np.sin(angle)]))
    P = t[:, :2]
    chords = np.hypot(*np.diff(P, axis=0).T)
    t = np.concatenate([[0.0], np.cumsum(np.sqrt(chords))])
    n = max(16, int(chords.sum() * 4.0) + 1)
    tt = np.linspace(0.0, t[-1], n)
    Q = _hermite(t, P, tt)
    S = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
    L = float(S[-1])
    N = max(3, int(np.ceil(L / _SPACING_PATH - 1e-6)) + 1)       # 1e-6: длина ровно в шаг не должна зависеть от округления
    N += (-(N - 1)) % _ShBrush                                    # число шагов кратно шагу переноса
    s = np.linspace(0.0, L, N)
    ds = L / (N - 1)
    x = np.interp(s, S, Q[:, 0])
    y = np.interp(s, S, Q[:, 1])
    S_nodes = np.interp(t, tt, S)
    p = clamp(_window(np.interp(s, S_nodes, t[:, 2]), int(1.0 / ds)), 0.0, 1.0)
    time = np.cumsum(np.concatenate([[0.0], np.maximum(t[1:, 3], 0.0)]))
    if time[-1] > 0:
        dtds = np.gradient(np.interp(s, S_nodes, time), ds)
        speed = _window(1.0 / np.maximum(dtds, 0.02), int(3.0 / ds))
    else:
        speed = np.ones(N)
    speed = clamp(speed, 0.02, 50.0)
    dx = _window(np.gradient(x, ds), int(1.5 / ds))
    dy = _window(np.gradient(y, ds), int(1.5 / ds))
    sweep = np.unwrap(np.arctan2(dy, dx))
    slant = np.interp(s, S_nodes, t[:, 4]) if t.shape[1] >= 5 else np.zeros(N)
    azimuth = np.interp(s, S_nodes, np.unwrap(t[:, 5])) if t.shape[1] >= 6 else None
    if has_rotation:
        angle = np.interp(s, S_nodes, np.unwrap(np.radians(t[:, 6]))) - np.pi / 2
        if np.cos(angle[0] - sweep[0]) < 0:
            angle = angle + np.pi
    elif brush.shape == 'круглая':
        angle = sweep
    else:
        # плоская кисть доворачивается за ходом с отставанием
        lambda_ = max(brush.lag * brush.width, 1e-3)
        share = 1.0 - np.exp(-ds / lambda_)
        sweep_sp = sweep.tolist()
        at = sweep_sp[min(len(sweep_sp) - 1, int(2.0 / ds))]
        angle = []
        for f in sweep_sp:
            at += (f - at) * share
            angle.append(at)
        angle = np.array(angle)
    return dict(x=x, y=y, p=p, speed=speed, angle=angle, sweep=sweep, slant=slant, azimuth=azimuth,
                ds=ds, L=L, s=s, N=N, on_place=on_place)
