# READING VIEW of движок/кисть/хватит.py with names in English. Not runnable; edit the original.
"""О-1 (Д-1, К-13 баланса): на сколько мм пути хватит того, что на кисти, — `к.хватит(…)`, до кладки.

Живая краска (`к.набрать(цвет, количество)`) тратится от хода к ходу; модель узнавала, когда добирать, только пробой.
Здесь проба — на своём стенде в памяти, копией кисти: кисть модели, её холст и журнал не трогаются.

Почему стенд, а не таблица. Путь кисти зависит не только от пресета и ширины: от нагрузки, нажима, того, сколько на
кисти и что на кончиках. Таблица на каждую кисть не снимется; серия ходов стенда — 0,05–0,2 с на ход. От px холста путь
в мм не зависит (Р-1: запас ворса — в мм; `плоская` 6 мм — 174, 178 и 179 мм при 4, 8 и 16 px/мм), кроме кисти, узкой
на этом холсте (уже 24 px: её физика держит px — `круглая` 3 мм проходит 51 мм при 4 px/мм и 132 при 8 и 16).

Стенд: чистый гладкий грунт `ГРУНТ`, масштаб — холста модели (`холст=` или `px_на_мм=`); ходы прямо по `ход_мм`
(умолчание 40 мм) один за другим без добора, каждый своим рядом (сырое прошлого хода под следующим не лежит); сид
серии — один на все её ходы (`СИДЫ`, 12 серий). Мерка — `о.закрыто` хода. Порог — `доля` (умолчание 0,5) от закрытого
первым ходом той же кисти полной свежей загрузкой (`набрать(…, 1)`) той же нагрузкой и нажимом. «Хватит на N мм» —
путь до места, где закрыто ходов переходит порог: закрыто хода стоит в его середине, между серединами — линейно; до
середины первого — от закрытого свежей кистью. Ответ — среднее серий, разброс — их σ.
"""
import math
import warnings

import numpy as np

GRAtNT = (235, 232, 225)
WithAndDY = tuple(7 * s + 3 for s in range(12))     # сид серии (тот же, что у стенда Приёмки 277 `путь12с.py`)
DResponseLYa = 0.5
SWEEP_MM = 40.0
SWEEPS_MM = (5.0, 400.0)
TO_MM = 2000.0                                  # дальше не меряется: «хватит больше 2000 мм»
FIELD_MM = 4.0                                   # от края стенда до хода, мм
RYaDResponseIn = 12                                      # рядов на одном холсте стенда (дальше — новый)


def _px(brush, canvas, px_on_mm):
    if canvas is not None and getattr(canvas, 'px_на_мм', None):
        return float(canvas.px_on_mm)
    if px_on_mm:
        return float(px_on_mm)
    raise ValueError('хватит: дай холст= (или px_на_мм=) — стенд кладёт ходы в px холста, а кисть, узкая на нём '
                     '(уже 24 px), держит px: `круглая` 3 мм полной загрузкой проходит 51 мм при 4 px/мм и 132 при 8')


def _width_px(brush, px):
    return float(brush.width_mm) * px if brush.width_mm is not None else float(brush.width)


class _TestBench:
    """Ряды ходов одной серии: холст за холстом по `РЯДОВ` рядов, кисть — своя копия."""

    def __init__(self, brush, px, sweep_mm, pressure, load, thinner, seed):
        self.brush, self.px, self.L, self.pressure, self.seed = brush, px, float(sweep_mm), pressure, int(seed)
        self.d = {}
        if load is not None:
            self.d['нагрузка'] = load
        if thinner:
            self.d['разбавитель'] = float(thinner)
        self.sh = _width_px(brush, px)
        self.spacing = max(self.sh * 1.6, 3.0 * px)
        self.canvas, self.row = None, RYaDResponseIn

    def sweep(self):
        from .canvas import Canvas
        from .path import path_with_pressure
        if self.row >= RYaDResponseIn:
            field = FIELD_MM * self.px + self.sh
            self.canvas = Canvas(int(round(self.L * self.px + 2 * field)), int(round(RYaDResponseIn * self.spacing + 2 * field)), ground=GRAtNT,
                           px_on_mm=self.px, cores=1, hints=False)
            self.field, self.row = field, 0
        y = self.field + (self.row + 0.5) * self.spacing
        self.row += 1
        t = [(self.field, y), (self.field + self.L * self.px, y)]
        if self.pressure is not None:
            t = path_with_pressure(t, self.sh, pressure=float(self.pressure), seed=self.seed)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            response = self.canvas.stroke(t, None, self.brush, seed=self.seed, **self.d)
        return 0.0 if response is None else float(response.covered)


def _fresh(brush):
    """Та же кисть с той же краской полной свежей загрузкой, без подобранного на кончиках."""
    from .tool import Loading
    from .numbers import _BrushResponseRZAndN
    n = brush.with_()
    kr = brush._paint
    n._paint = Loading(tuple(kr), np.ones(_BrushResponseRZAndN), np.ones(_BrushResponseRZAndN))
    return n


def _path(z, threshold, L, z0):
    """мм до места, где закрыто переходит порог: з — закрыто ходов по порядку (последний — ниже порога или нет)."""
    xs = [0.0] + [(i + 0.5) * L for i in range(len(z))]
    ys = [z0] + list(z)
    for i in range(1, len(ys)):
        if ys[i] < threshold:
            a, b = ys[i - 1], ys[i]
            t = (a - threshold) / (a - b) if a > b else 0.0
            return xs[i - 1] + min(max(t, 0.0), 1.0) * (xs[i] - xs[i - 1]), True
    return None, False


def lasts(brush, pressure=None, share=DResponseLYa, load=None, sweep_mm=SWEEP_MM, thinner=None,
           canvas=None, px_on_mm=None):
    """О-1: `к.хватит(…)` — на сколько мм пути хватит живой краски, что сейчас на кисти, — до кладки; кисть, холст и
    журнал не трогаются. Ходы по `ход_мм` этой кистью, этим нажимом (None — как ход точками (x, y)) и нагрузкой (None —
    кисти) без добора, пока `закрыто` хода не упадёт ниже `доля` закрытого свежей полной кистью той же нагрузки. Масштаб
    — холста (`холст=` или `px_на_мм=`, обязателен: стенд — в px холста). Ответ — словарь: 'мм', 'ходов', 'разброс_мм',
    'закрыто_свежей', 'порог', 'закрыто_ходов' (среднее серий по ходам), 'осталось', 'больше' (если за `ДО_ММ` не
    кончилась), … Справочник «Paint on the brush»."""
    from .tool import Loading
    kr = brush._paint
    if not isinstance(kr, Loading):
        raise ValueError('хватит: краска на кисти не живая — ' + (
            'кисть не набрана: `к.набрать(цвет, количество)`' if kr is None else
            'набрана без количества (или ход с цветом): кисть полна на каждом ходе, хватит её всегда; '
            'живая — `к.набрать(цвет, 1)`'))
    if brush.shape == 'мастихин':
        raise ValueError('хватит: мастихин краску не тратит ворсом — путь не меряется')
    px = _px(brush, canvas, px_on_mm)
    L = float(sweep_mm)
    if not (SWEEPS_MM[0] <= L <= SWEEPS_MM[1]):
        raise ValueError(f'хватит: ход_мм {L:g} — от {SWEEPS_MM[0]:g} до {SWEEPS_MM[1]:g}; ближайшее — '
                         f'{min(max(L, SWEEPS_MM[0]), SWEEPS_MM[1]):g}')
    d = float(share)
    if not 0.0 < d < 1.0:
        raise ValueError(f'хватит: доля {share} — от 0 до 1, не включая (0,5 — вдвое беднее свежей кисти)')
    if pressure is not None and not (0.05 <= float(pressure) <= 1.0):
        raise ValueError(f'хватит: нажим {float(pressure):g} — от 0,05 до 1; None — как ход точками (x, y)')
    if thinner is not None and not (0.0 <= float(thinner) <= 1.0):
        raise ValueError(f'хватит: разбавитель {float(thinner):g} — от 0 до 1')
    to = int(math.ceil(TO_MM / L))
    series = []                                  # (стенд, закрыто свежей, закрыто ходов, мм или None)
    for seed in WithAndDY:
        z0 = _TestBench(_fresh(brush), px, L, pressure, load, thinner, seed).sweep()
        st, z, n = _TestBench(brush.copy(), px, L, pressure, load, thinner, seed), [], None
        while n is None and len(z) < to:
            z.append(st.sweep())
            n = _path(z, d * z0, L, z0)[0]
        series.append([st, z0, z, n])
    long = max(len(with_[2]) for with_ in series)     # закрыто по ходам — у всех серий до одного хода
    for with_ in series:
        while len(with_[2]) < long:
            with_[2].append(with_[0].sweep())
    reached = sum(with_[3] is not None for with_ in series)
    mm = [to * L if with_[3] is None else with_[3] for with_ in series]
    fresh, rows = [with_[1] for with_ in series], [with_[2] for with_ in series]
    m = np.asarray(mm)
    by_sweep = [round(float(v), 4) for v in np.mean(np.asarray(rows), axis=0)]
    with np.errstate(all='ignore'):
        response = {'мм': round(float(m.mean()), 1), 'ходов': round(float(m.mean()) / L, 2), 'разброс_мм': round(float(m.std()), 1),
             'закрыто_свежей': round(float(np.mean(fresh)), 4), 'порог': round(d * float(np.mean(fresh)), 4), 'доля': d,
             'закрыто_ходов': by_sweep, 'осталось': brush.left, 'ход_мм': L, 'px_на_мм': px,
             'нажим': None if pressure is None else float(pressure), 'сидов': len(WithAndDY)}
    if reached < len(WithAndDY):
        response['больше'] = (f'в {len(WithAndDY) - reached} сериях из {len(WithAndDY)} закрыто не упало ниже порога за {TO_MM:g} мм: '
                       f'хватит больше {TO_MM:g} мм')
    return response


class Load(float):
    """О-1: ответ `нагрузка_для` у кисти с живой краской — то же число (им и кладут) с полями остатка."""
    lasts = lasts_mm = on_sweep = None

    def __reduce__(self):
        return float, (float(self),)


PX_FORECAST = 8.0                               # px холста прогнозов без `px_на_мм` (как у калибровки)


def with_remainder(response, brush, pressure, thinner, px_on_mm, length, plan=False):
    """О-1: прогноз (`нагрузка_для` — число, `план_покрытия` — словарь) и остаток кисти: у кисти без живой краски — ответ
    как есть (бит в бит), иначе — с путём остатка `хватит` этой нагрузкой и нажимом. длина — мм хода (ряда)."""
    from .tool import Loading
    if length is not None and not float(length) > 0.0:
        raise ValueError(f'длина {length} — мм хода (ряда), больше 0')
    if not isinstance(getattr(brush, '_краска', None), Loading):
        return response
    n = response['нагрузка'] if plan else float(response)
    canvas = lasts(brush, pressure=pressure, load=n, thinner=thinner or None, px_on_mm=px_on_mm or PX_FORECAST)
    mm = canvas['мм']
    if plan:
        response = dict(response, lasts_mm=mm, lasts=canvas)
        if length is not None:
            response['рядов_до_добора'] = int(mm // float(length))
        return response
    response = Load(n)
    response.lasts, response.lasts_mm = canvas, mm
    if length is not None:
        response.on_sweep = bool(mm >= float(length))
    return response
