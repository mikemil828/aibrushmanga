# READING VIEW of движок/кисть/зерно.py with names in English. Not runnable; edit the original.
"""Зерно холста (версия 3, пункт 5б): рельеф грунта — полотняное переплетение нитей.

Не картинка и не текстура поверх: карта высоты, которая считается из нитей. Нить основы идёт сверху вниз, нить утка —
слева направо; на каждом перекрестье одна лежит поверх другой, через раз. Нити не по линейке: шаг гуляет (нить
то ближе к соседке, то дальше), у каждой своя толщина, и вдоль нити она меняется (утолщения пряжи). Грунт залил
впадины не до конца: остались бугорки нитей и ямки между ними.

`Зерно.высота(x0, y0, x1, y1)` — высота грунта в единицах толщины h (0 — дно ямки, `глубина` — верх самой толстой
нити) в прямоугольнике холста. Считается от координат холста: у участка и у холста целиком — одно и то же.
Кто им пользуется: `Холст.рельеф` (свет по зерну), `укладка.положить_мазок` (полупустая кисть кладёт на бугорки,
волосок проскакивает на гребне нити — тем же путём, что на гребне лежащей краски). Числа — `числа._ЗЕРНО`.
"""
import numpy as np

from .fast import clamp

from .numbers import _InResponseLResponseBrushNA, _ZERNResponse, _PEREGRAtZ, _WithRYIn, _WithAtCanvasAYa, _DRY_RIDGE
from .strands import _noise

_NAndTEY = 4096     # свойства нитей повторяются через столько нитей (холст в 2,5 м при шаге 0,6 мм)


class Grain:
    __slots__ = ('шаг', 'глубина', 'сид', '_волны', '_нити')

    def __init__(self, spacing_px, depth, seed=0):
        """шаг_px — шаг нити в px; глубина — в единицах h (верх нити над дном ямки)."""
        self.spacing, self.depth, self.seed = float(spacing_px), float(depth), int(seed)
        if self.spacing < 2.0:
            raise ValueError(f'зерно: шаг нити {self.spacing:.2f} px — мельче 2 px на холсте его нет')
        rng = np.random.default_rng([5_061, self.seed])
        # шаг гуляет: поперёк нитей — на 7–30 шагов, вдоль нити — на 40–120 (нить почти прямая)
        self._wave = [(rng.uniform(7.0, 30.0, 4), rng.uniform(40.0, 120.0, 4), rng.uniform(0.0, 6.2832, (2, 4)))
                       for _ in range(2)]
        # у каждой нити своя толщина; вдоль нити — утолщения пряжи: своя длина волны и фаза
        self._threads = [(np.exp(_ZERNResponse[2] * clamp(rng.standard_normal(_NAndTEY), -2.0, 2.0)), rng.uniform(6.0, 22.0, _NAndTEY),
                       rng.uniform(0.0, 6.2832, _NAndTEY), rng.uniform(-1.0, 1.0, _NAndTEY)) for _ in range(2)]

    def height(self, x0, y0, x1, y1):
        """Высота грунта (y1 - y0, x1 - x0), float32, единицы h."""
        x = (np.arange(x0, x1, dtype=np.float64) + 0.5)[None, :] / self.spacing
        y = (np.arange(y0, y1, dtype=np.float64) + 0.5)[:, None] / self.spacing
        place = []
        for (across, along, phases), (a, b) in zip(self._wave, ((x, y), (y, x))):
            shift = 0.0
            for k in range(4):
                shift = shift + np.sin(6.2832 * a / across[k] + phases[0, k]) * np.sin(6.2832 * b / along[k] + phases[1, k] + 1.3 * k)
            place.append(a + _ZERNResponse[3] * shift)
        U, V = place
        iu, iv = np.floor(U).astype(np.intp), np.floor(V).astype(np.intp)
        fu, fv = U - iu - 0.5, V - iv - 0.5
        beyond = np.where((iu + iv) % 2 == 0, 1.0, -1.0)      # основа поверх утка — через раз
        z = None
        for (thickness, wave, phase, weight), i, across, along, length, sign in (
                (self._threads[0], iu % _NAndTEY, fu, fv, V, beyond), (self._threads[1], iv % _NAndTEY, fv, fu, U, -beyond)):
            t = thickness[i] * (1.0 + _ZERNResponse[4] * weight[i] * np.sin(6.2832 * length / wave[i] + phase[i]))
            t = clamp(np.abs(across) / (0.5 * clamp(_ZERNResponse[5] * t, 0.5, 1.0)), 0.0, 1.0)
            thread = np.cos(1.5708 * t) ** 2 * t * (0.5 + 0.5 * sign * np.cos(3.1416 * along))
            z = thread if z is None else np.maximum(z, thread)
        return (clamp(z / (1.0 + 2.0 * _ZERNResponse[2]), 0.0, 1.0) * self.depth).astype(np.float32)

    def flooded(self, height, thickness):
        """Зерно под лежащей краской: тонкая плёнка идёт по нитям, толстая заливает ямки. высота и толщина — h."""
        return height * np.exp(-thickness / (_ZERNResponse[6] * self.depth))


def _blur(a, times):
    """Размытие [1 2 1]/4 по обеим осям, раз проходов; за краем — крайнее значение."""
    for _ in range(times):
        q = np.pad(a, 1, mode='edge')
        a = 0.25 * q[:-2, 1:-1] + 0.5 * q[1:-1, 1:-1] + 0.25 * q[2:, 1:-1]
        q = np.pad(a, 1, mode='edge')
        a = 0.25 * q[1:-1, :-2] + 0.5 * q[1:-1, 1:-1] + 0.25 * q[1:-1, 2:]
    return a


def _blur0(a, times):
    """Как `укладка._размыть`: [1 2 1]/4 по обеим осям, за краем — ноль."""
    for _ in range(times):
        q = np.pad(a, 1)
        a = 0.25 * q[:-2, 1:-1] + 0.5 * q[1:-1, 1:-1] + 0.25 * q[2:, 1:-1]
        q = np.pad(a, 1)
        a = 0.25 * q[1:-1, :-2] + 0.5 * q[1:-1, 1:-1] + 0.25 * q[1:-1, 2:]
    return a


def _top(hr, mt):
    """К6, вторая доделка: до чего достаёт ворс лёгкой руки на лежащем рельефе hр — по холсту, а не по пути волоска.
    Верх — место не ниже своего окружения: кончик волоска не тоньше полутора пикселей (высота под ним — средняя,
    `_СУХАЯ_ГРЕБЕНЬ[10]` проходов [1 2 1]/4), лежит он на том, что вокруг (среднее на ±5 px, `[7]` проходов), и кладёт,
    где высота под ним не ниже этого среднего на `[8]` местного размаха рельефа (четыре σ высоты вокруг; не мельче
    `[12]` плёнки); переход «достаёт — не достаёт» — `[14]` размаха. Прежде верхом был самый высокий гребень на ±8 px пути:
    из двух-трёх гребней в окне краску брал один, который — менялось от волоска к волоску, и кусок на гребне был
    порезан поперёк. Теперь каждый гребень, что выше своего окружения, — верх, во всю свою длину. По ровному (размах
    мельче `[12]` плёнки) ворс лежит всюду. Ответ: достаёт (≤ 1; ниже нуля — не достаёт, тем глубже) и верх (0–1)."""
    G = _DRY_RIDGE
    around = _blur(hr, int(G[7]))
    range = 4.0 * np.sqrt(_blur((hr - around) ** 2, 2 * int(G[7])))       # от ложбины до верха — четыре σ
    share = clamp(((_blur(hr, int(G[10])) - around) / np.maximum(range, G[12] * mt) - G[8]) / G[14], -8.0, 1.0)
    even = clamp(2.0 - range / (G[12] * mt), 0.0, 1.0)
    return share + even * (1.0 - share), clamp(share, 0.0, 1.0) * (1.0 - even)


def _pulls(a, falloff):
    """Рельеф под волоском вдоль хода (N, nm): с бугорка волосок сходит не обрывом — краску тянет за бугорок."""
    slope = falloff * np.arange(len(a))[:, None]
    return np.maximum.accumulate(a + slope, axis=0) - slope


def _threads(a):
    """Волосок — нить в 1–1,5 px: соседние выборки профиля (N, nm) идут вместе; размах — единица."""
    q = np.pad(a, ((0, 0), (1, 1)), mode='edge')
    a = 0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]
    return a / max(float(a.std()), 1e-6)


class _Strands:
    """Ворс полупустой кисти склеен краской в прядки разной толщины (v3.22): соседние волоски в прядке — одной длины,
    с одной краской, встают и уходят вместе; до грунта достаёт или не достаёт прядка целиком. Прядка сходится к острию:
    волоски у её края короче середины — между прядками просвет, и шире он у тех, что едва достают.
    `номер` (nm,) — прядка выборки профиля, `число` — сколько их, `край` (nm,) — место в прядке: 0 — середина, 1 — край."""
    __slots__ = ('номер', 'число', 'край')

    def __init__(self, rng, nm, spacing, wider=1.0):
        widths = np.maximum(wider * _WithAtCanvasAYa[25] * (nm * spacing / 24.0) ** 0.5 * np.exp(_WithAtCanvasAYa[26] * rng.standard_normal(nm + 1)), 1.5) / spacing     # в выборках профиля
        edges = np.cumsum(widths) - rng.random() * widths[0]
        i = np.arange(nm) + 0.5
        self.number = np.searchsorted(edges, i)
        self.number = int(self.number[-1]) + 1
        self.edge = clamp(np.abs(i - (edges[self.number] - 0.5 * widths[self.number])) / (0.5 * widths[self.number]), 0.0, 1.0)

    def own(self, rng, n=1, scale=None):
        """Свойство прядки по выборкам (n, nm), разброс ~1: одно на прядку; масштаб — гуляет вдоль хода (в отсчётах пути)."""
        v = rng.standard_normal((1, self.number)) if scale is None else _noise(rng, n, self.number, scale)
        return v[:, self.number]


class Breaks:
    """Что рвёт след гребня мельче самого гребня — по выборкам профиля (N, nm), в долях полной капли: бугорки и ямки
    нитей грунта (п. 5б) и, у полупустой кисти, краска на отдельных волосках (п. 10, инструмент 2). Закон тот же, что
    у рельефа лежащей краски в `положить_мазок`: гребень кладёт там, где капля с добавкой выше порога."""

    def __init__(self, by_samples, ic, with_, steepness, dry, rng, strands=None, on_brush_hair=None):
        self.strands = strands
        # v3.24: на_ворсе (N, nm) — та часть по_выборкам, что от краски на волосках полупустой кисти: её капля не заливает
        self.ms, self.ks, self.breaks_with = on_brush_hair, (None if on_brush_hair is None else on_brush_hair[ic]), (None if on_brush_hair is None else np.zeros((len(ic), with_.nr)))
        self.m, self.brush, self.g, self.nr, self.steepness, self.dry, self.rng = by_samples, by_samples[ic], with_.ridge, with_.nr, steepness, dry, rng
        self.number = np.maximum(np.bincount(with_.ridge, minlength=with_.nr), 1)
        self.to, self.breaks, self.lays = (np.empty((len(ic), with_.nr)) for _ in range(3))
        self.by_ridges, self.on_ridge, self.breakaway, self.fibers, self.thicker, self.top, self.middle, self.reaches, self.touched = False, None, False, 0.0, 1.0, None, None, None, None
        self.charge = _WithAtCanvasAYa[2] / _PEREGRAtZ if dry else 1.0      # полупустая кисть набрана не полной

    def drop(self, full):
        """Капля на кончиках, когда кисть села: у полной кисти — полная, у полупустой — что осталось, у гребня своё."""
        if not self.dry:
            return full.copy()
        charged = np.exp(0.25 * self.rng.standard_normal(self.nr))
        if self.middle is not None:
            charged = charged ** _InResponseLResponseBrushNA[13] * self.middle      # К6: расчёсанная кисть набрана ровнее, и больше — серединой обоймы
        self.charged = _WithAtCanvasAYa[1] * charged / charged.mean()     # кисть набрана одна: гребни богаче и беднее вокруг её среднего
        return full * self.charged

    def feeds(self, share_drops, remainder, behind_spacing, spacing):
        """Полупустая кисть: сколько краски идёт из ворса на кончик (nr,). Ворс отжат — каплю он не наливает: кончик
        мокр не больше самого ворса, и подаётся только то, что легло. На посадке кисть не богаче, чем в ходе; пустеет
        она по тому, что положила, а не по пройденному пути. остаток — заряд гребня в долях набранного."""
        # v3.28: к редкому гребню краска идёт по смоченным волоскам ровно (`по_зерну`) — его кончик сохнет позже: когда в
        # ворсе осталась доля `_СУХАЯ[39]` от той, с которой сохнут соседи
        target = self.charged * np.minimum(remainder / (_WithAtCanvasAYa[17] * (1.0 if self.jerks is None else np.where(self.jerks < 1.0, _WithAtCanvasAYa[39], 1.0))), 1.0)      # кончик сохнет, когда ворс отдал половину
        # v3.24: за шаг переноса ворс подаёт не больше, чем кончику недостаёт (за_шаг — сколько он подал бы, в долях полной
        # капли). У кисти 12 px капля мала против шага, и подача её переливала: кончик полупустой кисти шёл с полной
        # каплей — первые 50–110 px хода ложились лентой полной кисти («плита», проверка Приёмки 71)
        # v3.24: краска идёт из отжатого ворса на кончик не разом: за px пути — доля `_СУХАЯ[33]` того, чего кончику недостаёт.
        # Гребень, что кладёт сплошь, отдаёт быстрее, чем ворс подаёт, — его кончик сохнет до просветов за первые px, а не
        # когда опустеет запас всей кисти; гребень, что кладёт прядками, ворс успевает кормить. шаг — px пути
        return clamp((target - share_drops) * min(_WithAtCanvasAYa[33] * spacing, 1.0) / np.maximum(behind_spacing, 0.08), 0.0, 1.0) * (remainder > 0.0)

    def feeds(self, uneven):
        """v3.28: запас гребней вдоль хода (Nc, nr, вокруг единицы) у полупустой кисти: к редкому гребню краска подходит
        без рывков — комки запаса у него слабее, `_СУХАЯ[37]` прежнего (какие гребни — `по_зерну`)."""
        return uneven if self.jerks is None else 1.0 + (uneven - 1.0) * self.jerks[None, :]

    @staticmethod
    def reserve(by_ridges, half_width):
        """v3.23: запас полупустой кисти. Кисть отжата о палитру вся: краски в ней набрано, сколько сказано, — от пряди
        к пряди запас ходит, но в сумме по кисти он один. Иначе у кисти в четыре пряди запас целиком гулял от мазка к
        мазку вдвое, и ход, где у холста оказались богатые пряди, к концу не редел."""
        return by_ridges / np.average(by_ridges, weights=half_width)

    def share(self, charge, capacity, spacing):
        """v3.23: ворс в одной обойме — краска расходится по соседним гребням, а не по всей кисти разом: что сосед отдал,
        то у него убыло. Гребень, что кладёт сплошь, пустеет сам и тянет из ближних; до дальнего края широкой кисти его
        расход не доходит. заряд (nr,) — в долях запаса, ёмкость (nr,) — сколько краски в полном запасе гребня, шаг — px
        пути; сумма заряд·ёмкость не меняется."""
        if not self.dry or len(charge) < 2:
            return charge
        flow = min(_WithAtCanvasAYa[32] * spacing, 0.33) * (charge[:-1] - charge[1:]) * np.minimum(capacity[:-1], capacity[1:])
        paint = charge * capacity
        paint[:-1] -= flow
        paint[1:] += flow
        return paint / capacity

    def ridge(self, c, m, breaks, share_drops):
        """Шаг переноса c: m (nr,) — сытость гребня до порога; ответ — доля ширины гребня, что кладёт краску."""
        dry, breaks = breaks, breaks * (1.0 if self.by_ridges else clamp(_ZERNResponse[8] - share_drops, _ZERNResponse[9], 1.0))       # полная капля затекает и в ямку; п. 13, "ridges": лёгкая рука каплю в ложбину не вдавливает
        self.to[c], self.breaks[c] = m, breaks
        addition = self.brush[c] * breaks[self.g]
        if self.ks is not None:
            # v3.24: в ямку грунта капля затекает, а пустую прядку не наполняет: краска на ворсе своя у каждой прядки, и
            # богатый гребень кладёт теми же прядками с просветами, а не лентой (плита в начале хода 12 px)
            self.breaks_with[c] = dry - breaks
            addition = addition + self.ks[c] * self.breaks_with[c][self.g]
        self.lays[c] = np.bincount(self.g, clamp((m[self.g] + addition) * self.steepness, 0.0, 1.0), self.nr) / self.number
        return self.lays[c].copy()

    def with_ridge(self):
        """П. 13: во сколько раз толще кладёт выборка профиля (N, nm) полупустой кисти на гребне лежащей краски."""
        return 1.0 if self.on_ridge is None else 1.0 + _DRY_RIDGE[0] * self.on_ridge

    def hangs(self, share_drops):
        """Сгусток на отрыве: на полупустом кончике капля не висит — краска держится в ворсе."""
        return clamp((share_drops - _WithAtCanvasAYa[14]) / max(1.0 - _WithAtCanvasAYa[14], 1e-6), 0.0, 1.0) if self.dry else 1.0

    def hair(self, filled, touch_m, x, X, Y, ds, p, bend, W):
        """Полупустая кисть: что кладёт выборка профиля (N, nm) — волосок, а не гребень целиком. Гребень касается холста
        по-прежнему (кас_м, x — место выборки в гребне); волосок кладёт краску, пока идёт по холсту: где кончик стоит
        (посадка — ворс ещё не потянуло), слой не копится.
        v3.21: волоски разной длины. Севшая под лёгкой рукой кисть стоит на кончиках самых длинных — до грунта достают
        не все; остальные ложатся, когда ворс потащило и согнуло (изгиб (N,): 0 — кисть только села, 1 — её протащило
        на ширину; считается на длину ворса `_СУХАЯ[24]`) или когда руку прижали (p). Волосок у порога касается то и дело: вдоль хода его держит то сосед, то
        бугорок краски. Отпечаток севшей кисти и её подъём рваны, как тело хода: кромка посадки и отрыва — по волоску."""
        goes = clamp(np.hypot(np.gradient(X, axis=0), np.gradient(Y, axis=0)) / ds, 0.0, 1.0)
        With = _WithAtCanvasAYa
        # v3.22: длина — у прядки, а не у волоска: соседи в прядке достают до грунта вместе; к краю прядки волоски короче
        p = self.strands
        length = With[27] * p.own(self.rng) + With[28] * (p.edge ** 2 - 0.4)[None, :] + With[29] * _threads(self.rng.standard_normal((1, x.shape[0])))
        # рука держит кисть по её же ворсу: среднее и разброс длин — у кисти свои, сколько прядок ни вышло
        self.length = (length - length.mean()) / max(float(length.std()), 1e-6)          # короче среднего — больше нуля
        at_threshold = (With[20] + self.fibers * (_InResponseLResponseBrushNA[2] - With[20])) / ds      # К6: расчёсанный ворс идёт ровно — волосок у порога касается и уходит на длине дорожки, а не то и дело
        undershoot = With[19] * (0.8 * self.length + 0.6 * (With[27] * p.own(self.rng, len(p), at_threshold)
                                                     + With[29] * _threads(_noise(self.rng, len(p), x.shape[0], at_threshold))))
        bend = 1.0 - (1.0 - bend) ** (W / With[24])                # ворс сгибает ход в длину ворса, а не в ширину кисти
        z = clamp((p[:, None] * (1.0 + With[21] * bend[:, None]) - With[18] - undershoot) / With[22] + 0.5, 0.0, 1.0)
        return clamp((touch_m * 1.2 - x[None, :]) * 10.0, 0.0, 1.0) * filled * goes * (z * z * (3.0 - 2.0 * z))

    def irregular(self, landing_m, p, ds, W):
        """Посадка и подъём плоской кисти (`пряди._посадка`, (N, nm)) по волоску: короткий волосок встаёт позже соседа и
        уходит раньше — на долю ширины хода; кромка посадки не линия по обойме."""
        k = np.rint(_WithAtCanvasAYa[23] * W * clamp(0.5 + 0.5 * self.length[0], 0.0, 1.5) / ds).astype(np.intp)
        i = np.arange(len(p))[:, None]
        i = np.where(i <= int(np.argmax(p)), i - k[None, :], i + k[None, :])
        return landing_m[clamp(i, 0, len(p) - 1), np.arange(len(k))[None, :]]

    def filled(self, touch, Saturated, i0, i1, f0, f1, g):
        """После переноса: насколько выборка профиля заполнена (N, nm) — краска на бугорке нити и на полном волоске,
        ямка и пустой волосок под тем же гребнем пропущены. Сыт — что гребни положили (с обрывами набора и края)."""
        factor = Saturated / np.maximum(self.lays, 1e-9)
        addition = self.m * (self.breaks[i0] * f0 + self.breaks[i1] * f1)[:, g]
        if self.ms is not None:
            addition = addition + self.ms * (self.breaks_with[i0] * f0 + self.breaks_with[i1] * f1)[:, g]
        saturated = clamp(((self.to[i0] * f0 + self.to[i1] * f1)[:, g] + addition) * self.steepness * (_WithAtCanvasAYa[34] + self.fibers * (_InResponseLResponseBrushNA[9] - _WithAtCanvasAYa[34]) if self.dry else 1.0), 0.0, 1.0)
        if self.dry:
            # полупустая кисть: волосок кладёт столько, сколько на нём краски сверх порога, — тон в пятне идёт от плотного
            # к тонкому, пятно начинается и кончается у каждого волоска в своём месте (гребень целиком не гаснет)
            # v3.26: толщина растёт с краской волоска полого (`_СУХАЯ[34]`), а не насыщается у самого порога: тон в клочке
            # и вдоль хода ходит, плоской ленты одним тоном нет
            saturated = clamp((saturated - _WithAtCanvasAYa[15]) / (1.0 - _WithAtCanvasAYa[15]), 0.0, 1.0)      # дымки под порогом волосок не оставляет
            return saturated * (2.0 - saturated) * clamp(factor[i0] * f0 + factor[i1] * f1, 0.0, 1.0)[:, g]
        response = touch[:, g] * np.sqrt(saturated * clamp(factor[i0] * f0 + factor[i1] * f1, 0.0, 1.0)[:, g])
        if self.top is not None:
            # К6, "ridges": верх лежащего гребня поднят к ворсу — его задевают и те волоски гребня кисти, что до ровного
            # места под лёгкой рукой не достают: кусок на гребне сплошной, полосами своих прядей кисть его не режет
            self.touched = np.where(touch[:, g] > 0.0, self.top * clamp(4.0 * saturated, 0.0, 1.0), 0.0)
            response = np.maximum(response, self.touched)
        return response

    def ridge(self, filled, x):
        """Доля выборки профиля (N, nm), что кладёт, у кисти с краской: бедный гребень кисти кладёт серединой (x — место
        выборки в гребне). К6, "ridges": верх лежащего гребня задевает каждый волосок, что над ним прошёл, — и у края
        гребня кисти: кусок на верху сплошной во всю длину верха, а не штрихами по серединам прядей кисти."""
        mask = clamp((filled * 1.2 - x[None, :]) * 10.0, 0.0, 1.0)
        return mask if self.top is None else np.maximum(mask, clamp(self.touched * _DRY_RIDGE[11], 0.0, 1.0))

    def reached(self, V):
        """К6, "ridges": плёнка следа на холсте (рамка мазка) — только там, докуда ворс лёгкой руки достаёт (`_верхи`)."""
        return V if self.reaches is None else V * self.reaches


def _at_edges(u, W, apart=False):
    """К6: 0 — середина обоймы, 1 — самый край. Краёв у кисти два, и меряются они по-разному.
    Где краски меньше — `_ВОЛОКНА[5]` ширины кисти с каждой стороны: кисть отжата о палитру, и середина обоймы
    держит краску у кисти любой ширины.
    Где ворс расходится (врозь) — та же доля у кисти 48 px, а с шириной идёт с её корнем, как само волокно
    (`_ВОЛОКНА[10]`): волокон в этом краю у узкой кисти столько же, сколько у широкой, — кисть 24 px рвёт край
    тремя-четырьмя волокнами, а не одной кромкой. Но в середине обоймы, где краска держится, ворс сомкнут и врозь не
    идёт, сколько бы волокон ни насчитал край: расходиться он начинает за серединой, на ширине одного волокна. Без
    этого у кисти 24 px краем была вся ширина, и середина хода выходила полой (вторая доделка К6)."""
    from_middles = np.abs(u) - (0.5 - _InResponseLResponseBrushNA[5]) * W
    if not apart:
        return clamp(from_middles / (_InResponseLResponseBrushNA[5] * W), 0.0, 1.0)
    fiber = _InResponseLResponseBrushNA[10] * (W / 48.0) ** 0.5
    zone = min(_InResponseLResponseBrushNA[5] * 48.0 * (W / 48.0) ** 0.5, 0.5 * W)
    return clamp((np.abs(u) - (0.5 * W - zone)) / zone, 0.0, 1.0) * clamp(from_middles / fiber, 0.0, 1.0)


def _fibers(f, hairs, short, rng, with_, n, ds):
    """К6, поле "fibers" (f, 0–1): краска на ворсе полупустой кисти расчёсана вдоль волоска. Кисть отжата о палитру и
    протянута: в середине обоймы ворс сомкнут и держит краску — тело хода плотное; к краям кисти ворс расходится, и
    краски на нём меньше — там кладут отдельные прядки и волоски. На волоске краска лежит не короткими комками, а
    длинной дорожкой (`_ВОЛОКНА[2]` px и длиннее): волосок кладёт нить, пока на нём краска есть, и кончается в своём
    месте — просветы между нитями вытянуты по ходу. Короткие комки остаются долей `_ВОЛОКНА[0]`. Свой поток случая:
    без поля ворс тот же."""
    In = _InResponseLResponseBrushNA
    rng = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0])
    k = max(int(round(In[10] * (with_.W / 48.0) ** 0.5 * with_.nm / with_.W)), 1)       # волокно — пучок волосков в `_ВОЛОКНА[10]` px у кисти 48 px, идёт с корнем из ширины, как прядка

    def tufts(a):
        q = np.cumsum(np.pad(a, ((0, 0), (k, k - 1)), mode='wrap'), axis=1)
        a = (q[:, k:] - q[:, :-k])[:, :a.shape[1]]
        return a / max(float(a.std()), 1e-6)
    track = tufts(_noise(rng, n, with_.nm, In[2] / ds))
    own = tufts(rng.standard_normal((1, with_.nm)))
    apart = _at_edges(with_.u, with_.W, True)[None, :]
    body = In[3] - (In[3] + In[4]) * _at_edges(with_.u, with_.W)[None, :] ** In[6]
    return hairs - f * (1.0 - In[0]) * short + f * (In[1] * (1.0 + In[14] * apart) * track + In[7] * own * (In[8] + (1.0 - In[8]) * apart) + body - In[16] * _without_brush_hair(rng, with_))


def _without_brush_hair(rng, with_):
    """К6, третья доделка: где между разошедшимися волокнами ворса нет — по выборкам профиля (1, nm): 1 — щель,
    меньше нуля — волокно, собравшее краску. У края обоймы ворс расходится не поодиночке: краска стягивает волоски в
    волокна, и между волокнами остаётся место, где ворса нет вовсе, — краска туда не ложится, сколько бы её ни несли
    соседи. Без этого два соседних волокна с краской сливались в полосу, и край кисти 24 px шёл кромкой.
    Волокно стоит в своей клетке: ширина клетки — волокно `_ВОЛОКНА[10]` и щель при нём, от клетки к клетке ходит.
    В середине обоймы (где краска держится, `_у_края`) ворс сомкнут: у клетки, что задевает середину хотя бы боком,
    щели нет. За серединой щель раскрывается на ширине одного волокна до доли клетки `_ВОЛОКНА[15]`: ворс либо
    сомкнут, либо разошёлся. Стянутые волоски несут свою краску в волокно: оно богаче на `_ВОЛОКНА[17]` полной капли
    на единицу пустой доли — за сплошным телом хода кладут отдельные нити, а не кромка."""
    In = _InResponseLResponseBrushNA
    fiber = In[10] * (with_.W / 48.0) ** 0.5
    widths = fiber / (1.0 - In[15]) * np.exp(0.25 * rng.standard_normal(int(2.0 * with_.W / fiber) + 4))
    edges = np.cumsum(widths) - rng.random() * widths[0] - 0.5 * with_.W
    i = np.minimum(np.searchsorted(edges, with_.u), len(edges) - 1)
    axis = edges[i] - 0.5 * widths[i]
    empty = In[15] * clamp((np.abs(axis) - 0.5 * widths[i] - (0.5 - In[5]) * with_.W) / fiber, 0.0, 1.0)     # одна на волокно
    from_sides = (0.5 * widths[i] * (1.0 - empty) - np.abs(with_.u - axis)) / (with_.W / with_.nm)      # в выборках профиля: меньше нуля — щель
    slit = clamp(0.5 - from_sides, 0.0, 1.0) * clamp(empty * widths[i] / (with_.W / with_.nm), 0.0, 1.0)      # щель уже выборки — её доля; сомкнутый ворс — без щели
    return (slit - In[17] / In[16] * empty)[None, :]


def by_grain(grain, half_empty, rng, with_, X, Y, frame, ho, mt, H, p, ic, ds, steepness, drop, uneven, by_ridges=None, neighboring=None, breakaway=False, fibers=None):
    """Для `положить_мазок`: (Рвёт или None, перепад, неровн — запас гребней вдоль хода, у полупустой кисти с ровно
    кормленными гребнями, `Рвёт.кормит`). зерно — `Холст._зерно_под` или None; рамка — прямоугольник мазка
    на холсте; hо — толщина краски своего слоя в нём; перепад — рельеф лежащей краски в толщинах плёнки (или None):
    к нему добавляется зерно — волосок на гребне нити проскакивает, как на гребне краски (`борозда._проскок`)."""
    m = jerks = on_ridge = top = reaches = None
    tacky = hr = None
    if (half_empty or by_ridges or breakaway) and neighboring is not None:
        # п. 13: полупустая кисть (и кисть с краской при "ridges") идёт по рельефу всей лежащей краски — и других слоёв
        layers = neighboring()                                     # (толщина, сырость) своего слоя и остальных
        hr = sum(h for h, _ in layers)
        if float(hr.max()) > 1e-6:
            tacky = sum(h * w for h, w in layers) / np.maximum(hr, 1e-9)
            if float((hr - layers[0][0]).max()) > 1e-6:
                drop = clamp((hr - _blur0(hr, 3)) / mt, -0.6, 0.6)
    if half_empty and drop is not None:
        # п. 10, инструмент 2: полупустая кисть ловит гребни лежащей краски и пропускает впадины — по волоску и по шагу
        # пути, а не по гребню кисти на шаге переноса (борозды в 2–3 px шаг переноса в 2 px не различает)
        ix = clamp(np.floor(X - frame[0]).astype(np.intp), 0, drop.shape[1] - 1)
        iy = clamp(np.floor(Y - frame[1]).astype(np.intp), 0, drop.shape[0] - 1)
        m = _pulls(drop[iy, ix] * _WithAtCanvasAYa[16], 0.6 * _WithAtCanvasAYa[16] * ds / _ZERNResponse[10])
        # п. 13: гребень сырой лежащей краски снимает краску с волоска — на гребне слой толще, куском; в ложбину волосок не достаёт
        on_ridge = clamp(_pulls(drop[iy, ix], 0.6 * ds / _ZERNResponse[10]) / 0.6, 0.0, 1.0) * tacky[iy, ix]   # сырая краска липкая: тянет с волоска; сухой гребень только цепляет, как было
    elif by_ridges and drop is not None:
        # п. 13, поле "ridges": кисть с краской лёгкой рукой идёт по гребням лежащей краски — волосок в ложбину не достаёт,
        # чем легче рука; прижатая кисть кроет и ложбины, как без поля
        ix = clamp(np.floor(X - frame[0]).astype(np.intp), 0, drop.shape[1] - 1)
        iy = clamp(np.floor(Y - frame[1]).astype(np.intp), 0, drop.shape[0] - 1)
        # ворс лежит на гребнях вокруг: где лежащая краска неровная, до мест вровень со средним волосок тоже не достаёт
        light = float(by_ridges) * clamp((_DRY_RIDGE[2] - p) / _DRY_RIDGE[3], 0.0, 1.0)
        hangs = _DRY_RIDGE[5] * clamp(_blur(np.abs(drop), 6) / _DRY_RIDGE[6], 0.0, 1.0)
        m = _pulls(drop[iy, ix], 0.6 * ds / _DRY_RIDGE[4]) - hangs[iy, ix] * light[:, None]
        if hr is not None:
            # К6: волосок лёгкой руки лежит на верхах гребней, что у него на пути (на `_СУХАЯ_ГРЕБЕНЬ[7]` px), и между ними
            # не провисает: краску он оставляет на верхах, ниже не достаёт — борозда между гребнями пуста, а не затянута
            # плёнкой с дырами. Верх сырого гребня снимает краску с волоска куском (как у полупустой)
            G = _DRY_RIDGE
            # вторая доделка К6: верх — место не ниже своего окружения, по холсту (`_верхи`); порог один у всех прядей
            to, top = _top(hr, mt)
            m = _pulls(np.minimum(to[iy, ix], G[13]), 0.6 * ds / G[4]) * light[:, None] + m * (1.0 - light[:, None])
            top = top[iy, ix] * light[:, None]
            # куда ворс не достаёт, краска не ложится и с соседнего отсчёта следа: ложбина рядом с верхом пуста. Рука
            # тяжелее — ворс достаёт ниже (порог уходит вниз), прижатая кисть кроет всё, как без поля
            hand, trace = np.zeros(hr.shape), np.zeros(hr.shape)
            hand[iy, ix], trace[iy, ix] = light[:, None], 1.0
            hand = _blur(hand, 2) / np.maximum(_blur(trace, 2), 1e-6)
            reaches = clamp(to + (1.0 - hand) / np.maximum(hand, 1e-6), 0.0, 1.0)
            on_ridge = top * tacky[iy, ix] * _DRY_RIDGE[9]
        m = m * (1.1 + (_DRY_RIDGE[1] - 1.1) * light)[:, None]
    elif breakaway and drop is not None:
        # п. 17: кисть, у которой краска на исходе, кладёт на гребни лежащей краски (любых слоёв) и пропускает ложбины —
        # по волоску и по шагу пути; пока капля полна, она кроет и ложбины (порог — `Рвёт.гребень`)
        ix = clamp(np.floor(X - frame[0]).astype(np.intp), 0, drop.shape[1] - 1)
        iy = clamp(np.floor(Y - frame[1]).astype(np.intp), 0, drop.shape[0] - 1)
        m = _pulls(drop[iy, ix] * _WithRYIn[3], 0.6 * _WithRYIn[3] * ds / _ZERNResponse[10])
    if grain is not None:
        g, depth, flooded = grain(*frame)
        g = flooded(g, ho)
        g = g - _blur(g, 6)                                # бугорок и ямка — против среднего вокруг
        if float(np.abs(g).max()) > 1e-4 * depth:
            drop = clamp(g / mt, -0.6, 0.6) + (0.0 if drop is None else drop)
            ix = clamp(np.floor(X - frame[0]).astype(np.intp), 0, g.shape[1] - 1)
            iy = clamp(np.floor(Y - frame[1]).astype(np.intp), 0, g.shape[0] - 1)
            # в долях глубины зерна: лёгкая рука до ямок не достаёт; плёнка толще зерна ямки заливает
            # волосок по зерну идёт чертой: с бугорка нити он сходит не обрывом — краску тянет за бугорок, в ямке слой
            # тоньше или пропуск (у бедного волоска), на следующем бугорке снова плотнее
            under = _pulls(g[iy, ix], depth * ds / _ZERNResponse[10]) * (_ZERNResponse[7] / depth * min(depth / max(H, 1e-9), 1.0) * clamp(_ZERNResponse[1] - p, 0.3, 1.0))[:, None]
            m = under if m is None else m + under
    rng_with = np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]) if half_empty else None
    if half_empty:
        # на полупустом ворсе краска держится отдельных волосков — один кладёт, сосед уже пуст; нажим выжимает её
        # из ворса: средняя рука кроет плотнее лёгкой
        # v3.22: краска склеивает волоски в прядки — она у прядки общая, от волоска к волоску внутри прядки ходит слабее
        # v3.26: краска лежит на ворсе короткими комками (`_СУХАЯ[10]` px) больше у волоска, чем у прядки (`[30]`, `[31]`),
        # постоянная доля волоска мала (`[8]`): бока клочка не прямые, длинный ход — зерно из коротких штрихов по ходу
        strands = _Strands(rng_with, with_.nm, with_.W / with_.nm)
        a, b = _WithAtCanvasAYa[30], _WithAtCanvasAYa[31]
        own = a * strands.own(rng_with) + b * rng_with.standard_normal((1, with_.nm))
        # v3.28: редкий гребень ворс кормит ровно — краска идёт к его кончику по смоченным волоскам середины гребня плёнкой,
        # а не комками: запас такого гребня вдоль хода не скачет (`Рвёт.кормит`), кончик его сохнет позже соседей
        # (`Рвёт.подаёт`), и на волосках его середины (`_СУХАЯ[38]` px от оси гребня) коротких комков нет — волосок кладёт
        # тонкую нить, пока гребню хватает краски; боковым волоскам гребня краски достаётся меньше (`_СУХАЯ[40]`).
        # Ровно кормит ворс гребень из `1/_СУХАЯ[36]` (у кисти 24 px — один-два, бывает ни одного); у края кисти (ближе
        # `_СУХАЯ[41]` px) ворс торчит врозь — те гребни он ровно не кормит. Ровной дорожкой краска ложится, когда ворс
        # потащило и расчесало (на длине ворса `_СУХАЯ[24]`): севшая кисть и короткое касание кладут ещё комками.
        # Свой счётчик случая: остальной ворс тот же, что без нитей
        rng_n = np.random.default_rng(rng_with.bit_generator.seed_seq.spawn(1)[0])
        even = (rng_n.random(with_.nr) < _WithAtCanvasAYa[36]) & (np.abs(with_.center) < 0.5 * with_.W - _WithAtCanvasAYa[41])
        jerks = np.where(even, _WithAtCanvasAYa[37], 1.0)
        combing = (1.0 - np.exp(-ds * np.arange(len(p)) / _WithAtCanvasAYa[24]))[:, None]
        at_axes = clamp((_WithAtCanvasAYa[38] - with_.x * with_.half_width[with_.ridge]) / 0.5 + 0.5, 0.0, 1.0)[None, :]
        jerks_m = 1.0 - combing * (1.0 - jerks[with_.ridge])[None, :] * at_axes
        past = -_WithAtCanvasAYa[40] * combing * even[with_.ridge][None, :] * (1.0 - at_axes)
        lumps = jerks_m * _WithAtCanvasAYa[9] * (a * strands.own(rng_with, len(p), _WithAtCanvasAYa[10] / ds) + b * _threads(_noise(rng_with, len(p), with_.nm, _WithAtCanvasAYa[10] / ds))) / np.hypot(a, b)
        hairs = (_WithAtCanvasAYa[8] * (own - own.mean()) / max(float(own.std()), 1e-6)
                   + lumps
                   + _WithAtCanvasAYa[12] * (a * strands.own(rng_with, len(p), _WithAtCanvasAYa[13] / ds) + b * _threads(_noise(rng_with, len(p), with_.nm, _WithAtCanvasAYa[13] / ds))) / np.hypot(a, b)
                   + (_WithAtCanvasAYa[11] * (p - 0.45))[:, None]
                   + _WithAtCanvasAYa[35] + past)       # v3.26: краску несёт больше волосков — покрытие берётся числом мест, а не толщиной
        if fibers:
            hairs = _fibers(float(fibers), hairs, lumps, rng_n, with_, len(p), ds)
        m = hairs if m is None else m + hairs
    if breakaway:
        # п. 17: на ворсе краска лежит комками — прядка в несколько волосков (`_Прядки`) несёт её то больше, то меньше,
        # на длине `_СРЫВ[5]` ширины кисти и короче (`_СРЫВ[7]`); пока кончик полон, это не видно, на исходе запаса
        # богатая прядка ещё кладёт, бедная уже нет: след распадается на острова вдоль хода шириной в прядку-две
        rng_brush = np.random.default_rng(np.random.SeedSequence(rng.bit_generator.seed_seq.entropy, spawn_key=(17,)))
        strands = _Strands(rng_brush, with_.nm, with_.W / with_.nm, _WithRYIn[8])
        lumps = (_WithRYIn[4] * strands.own(rng_brush, len(p), max(_WithRYIn[5] * with_.W, 8.0) / ds)
                 + _WithRYIn[6] * strands.own(rng_brush, len(p), max(_WithRYIn[7] * with_.W, 4.0) / ds))
        lumps -= lumps.mean(axis=1, keepdims=True)           # кисть набрана одна: богатые и бедные прядки — вокруг её среднего
        m = lumps if m is None else m + lumps
    if m is None:
        return None, drop, uneven
    if breakaway:
        breaks = Breaks(m, ic, with_, steepness, False, None)
        breaks.jerks, breaks.by_ridges, breaks.breakaway = None, True, True
        return breaks, drop, uneven
    breaks = Breaks(m, ic, with_, steepness, half_empty, rng_with, strands if half_empty else None, hairs if half_empty else None)
    breaks.jerks, breaks.on_ridge, breaks.by_ridges = jerks, on_ridge, bool(by_ridges) and not half_empty
    breaks.top = top
    breaks.reaches = reaches
    breaks.fibers = float(fibers) if fibers and half_empty else 0.0
    breaks.thicker = 1.0 + breaks.fibers * (_InResponseLResponseBrushNA[11] - 1.0)
    if breaks.fibers:
        at_edges = _at_edges(with_.center, with_.W)
        breaks.middle = 1.0 + breaks.fibers * _InResponseLResponseBrushNA[12] * (0.5 - at_edges)
    return breaks, drop, breaks.feeds(uneven)
