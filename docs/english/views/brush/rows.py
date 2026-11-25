# READING VIEW of движок/кисть/ряды.py with names in English. Not runnable; edit the original.
"""В-3: серия рядов числом — `план_покрытия`: сколько проходов параллельных рядов и какая нагрузка дадут покрытие.

Серия — параллельные прямые ряды одной кистью с шагом `шаг_рядов` (доля ширины кисти `кисть.ширина`, px); проход —
все ряды поля один раз; следующий проход — те же ряды заново (по сырому подряд или по высохшему предыдущему). Покрытие —
1 − просвет грунта (`х.просвет`) в середине поля, без крайних рядов.

Закон накопления (замер живых серий, смена Движка 186). Просвет сквозь краску в точке — произведение просветов ходов,
что легли в неё (так `х.просвет` и считает свет сквозь слои и плёнки), а ходы с разными сидами между собой
независимы. Значит, оптическая глубина серии D = −ln просвета в точке — сумма глубин рядов: D(y) = Σ_k D₁(y − k·шаг),
где D₁(u) — поперечный профиль глубины одиночного хода. Покрытие поля — 1 − среднее exp(−D) поперёк. Три поправки,
снятые узлом таблицы: `снятие` — ряд, ложась на сырой соседний ряд своего прохода, снимает долю его глубины под собой
(прозрачная нагрузки 0,2 — до 0,5; отрицательное — затёк в просветы соседа и прибавил); `поверх` — во сколько раз
глубже ложится ход по высохшей краске прежних проходов (кроющая малой нагрузкой — до 1,2: ловит рельеф);
`поверх_сырого` — то же по сырой (прозрачная густая — 0,6–0,8: снимает прежний проход).

Профиль и снятие — таблица `ряды.json` рядом с модулем: пресет × ширина × нагрузка (14 узлов) × нажим × вид ×
разбавитель (узлы 0, 1 — в файле, 0,3, 0,5, 0,7 — по файлу на узел, `калибровка`) → профиль глубины D₁(u) (u — расстояние от оси хода в ширинах кисти, `U`) и `снятие`; между узлами —
по log ширины, нажиму и log нагрузки. Узел — среднее трёх сидов (кисть уже 2,5 мм — восьми) хода 45 мм
тёмно-синей краской (70, 80, 110) по грунту (166, 146, 129) при 8 px/мм; снятие — пара ходов по сырому со сдвигом в
полширины против одного, поверх — второй ход в то же место по высохшему и по сырому. Не кроющая краска другого цвета: глубина профиля — во столько раз больше или меньше, во
сколько глубина одиночного хода этим цветом против тёмно-синего по таблице одиночного хода (`калибровка`).
Пересчёт — `python3 -m кисть.ряды --пересчитать [--процессов 16]` (~1,5 ч на 10 процессах); хеш — хеш физики таблицы
одиночного хода и код замера узла этого модуля.

Только счёт: холст, кисть и журнал не трогаются, ничего не кладётся.
"""
import inspect
import json
import math
import os
import sys
import textwrap
import warnings

import numpy as np

from . import calibration as calib

FAYL = os.path.join(calib._PABrushET, 'ряды.json')

# ----------------------------------------------------------------- стенд узла
U = tuple(round(-1.2 + 0.025 * i, 3) for i in range(97))        # поперёк, в ширинах кисти от оси хода
WithAndDY = (1, 2, 3)
SEEDS_NARROW = tuple(range(1, 9))
WIDTHS_MM = {'плоская': (2.0, 4.0, 9.0, 18.0), 'мягкая': (1.0, 2.0, 5.0, 15.0), 'плоская_узкая': (0.75, 1.5, 2.5),
             'круглая': (1.0, 2.0, 3.5, 7.0), 'сухая': (2.0, 6.0, 12.0), 'лайнер': (0.25, 0.5), 'мастихин': (2.0, 5.5, 8.0)}
NAGRAtZBrushAnd = (0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 0.9)   # кроющая встаёт 0,02–0,05 — там чаще
NAZhAndMY = (0.3, 0.6, 0.8)
ShAG = (0.4, 1.2)               # шаг рядов, доли ширины кисти
PRResponseCanvasResponseDResponseIn = 8                   # больше проходов план не предлагает
WithDInAndG = 0.5                    # пара ходов узла для снятия по сырому: второй ниже на полширины кисти
PResponseLYa = ('профиль', 'снятие', 'поверх', 'поверх_сырого')
_FLOOR_D = 1e-4                  # глубина полосы, ниже которой ход не лёг — снятия нет


def _canvas(sh):
    from .canvas import Canvas
    h = max(200, int(2 * 1.3 * sh + WithDInAndG * sh) + 40)
    return Canvas(400, h, ground=calib.GRAtNT, px_on_mm=calib.PX_ON_MM, cores=1, hints=False), h


def _sweeps(preset, width_mm, load, pressure, kind, thinner, seed, times, shift=0.0, over=False):
    """`раз` ходов по сырому (сиды сид, сид+1000, …), каждый следующий ниже на `сдвиг` ширин кисти, на своём чистом
    холсте: (просвет по рядам — профиль по оси y, ось y первого хода, ширина следа первого хода, просвет поверх).
    поверх — затем ещё ход (сид + 2000) в то же место: 'сухо' — по высохшему первому, 'сыро' — по сырому; его просвет
    по рядам — четвёртым (иначе None)."""
    from .brush import Brush, path_with_pressure
    from .paint import Paint
    from .gap import map
    sh = float(width_mm) * calib.PX_ON_MM
    canvas, h = _canvas(sh)
    brush = Brush.from_file(os.path.join(calib._PREWithETY, preset + '.json'), width=sh)
    yc = h // 2
    w = None
    for r in range(times):
        with_ = seed + 1000 * r
        yr = yc + r * shift * sh
        points = path_with_pressure([(calib.CanvasResponseD[0], yr), (calib.CanvasResponseD[1], yr)], sh, pressure=float(pressure), entry=sh / 2, exit=sh / 2, seed=with_)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            response = canvas.stroke(points, Paint(calib.TsInET, kind), brush, layer='узел', seed=with_, load=float(load),
                       thinner=float(thinner) if thinner else None)
        if w is None:
            w = (response.width if response else 0.0) or sh
    frame = (calib.ResponseBrushNResponse[0], 0, calib.ResponseBrushNResponse[1], h)
    t = np.asarray(map(canvas, frame)[0], np.float64).mean(axis=1) if 'узел' in canvas.layers else np.ones(h)
    if not over:
        return t, yc, w, None
    if over == 'сухо':
        canvas.dry(1.0)
    points = path_with_pressure([(calib.CanvasResponseD[0], yc), (calib.CanvasResponseD[1], yc)], sh, pressure=float(pressure), entry=sh / 2, exit=sh / 2, seed=seed + 2000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        canvas.stroke(points, Paint(calib.TsInET, kind), brush, layer='узел', seed=seed + 2000, load=float(load),
               thinner=float(thinner) if thinner else None)
    return t, yc, w, (np.asarray(map(canvas, frame)[0], np.float64).mean(axis=1) if 'узел' in canvas.layers else np.ones(h))


def measurement_node(preset, width_mm, load, pressure, kind, thinner):
    """Профиль глубины хода поперёк D₁(u) = −ln просвета по `U` и `снятие` — словарь; 'глубина' — глубина полосы
    одного хода (для сверки). Снятие — из пары ходов по сырому со сдвигом `СДВИГ` ширины (как соседние ряды серии):
    глубина пары D₂(y) = D₁(y)·(1 − снятие·f(y − сдвиг)) + D₁(y − сдвиг), снятие — наименьшими квадратами по месту, где
    второй ход лёг на первый. Отрицательное — сырой ход не снял, а добавил (краска затекла в просветы первого)."""
    sh = float(width_mm) * calib.PX_ON_MM
    seeds = SEEDS_NARROW if float(width_mm) < calib.NARROW_MM else WithAndDY
    one, two, three, four, ww = [], [], [], [], []
    for with_ in seeds:
        t1, yc, w, t3 = _sweeps(preset, width_mm, load, pressure, kind, thinner, with_, 1, over='сухо')
        t1b, _, _, t4 = _sweeps(preset, width_mm, load, pressure, kind, thinner, with_ + 300, 1, over='сыро')
        one.append(t1b), four.append(t4)
        t2 = _sweeps(preset, width_mm, load, pressure, kind, thinner, with_ + 500, 2, WithDInAndG)[0]
        one.append(t1), two.append(t2), three.append(t3), ww.append(w)
    t1, t2, t3, t4 = (np.mean(a, axis=0) for a in (one, two, three, four))
    y = np.arange(len(t1)) - yc
    band = np.abs(y) <= calib.PResponsePERYoBrush / 2 * float(np.mean(ww))
    D1 = -math.log(max(float(t1[band].mean()), 1e-9))
    D = -np.log(np.clip(np.interp(np.asarray(U) * sh, y, t1, left=1.0, right=1.0), 1e-9, 1.0))
    if D1 < _FLOOR_D:
        return {'профиль': [round(float(v), 5) for v in D], 'снятие': 0.0, 'поверх': 1.0, 'поверх_сырого': 1.0,
                'глубина': round(D1, 5)}
    Dy = -np.log(np.clip(t1, 1e-9, 1.0))
    Dsdv = np.interp(y - WithDInAndG * sh, y, Dy, left=0.0, right=0.0)
    D2 = -np.log(np.clip(t2, 1e-9, 1.0))
    weight = np.clip(Dsdv / D1, 0.0, 1.0) * Dy                  # где второй ход лёг на первый
    denominator = float((weight * weight).sum())
    removal = float((weight * (Dy + Dsdv - D2)).sum() / denominator) if denominator > 1e-12 else 0.0
    D3 = -math.log(max(float(t3[band].mean()), 1e-9))
    over = float(np.clip((D3 - D1) / D1, 0.3, 3.0))         # глубина хода по высохшему первому к глубине по грунту
    return {'профиль': [round(float(v), 5) for v in D], 'снятие': round(float(np.clip(removal, -1.0, 0.9)), 4),
            'поверх': round(over, 4), 'поверх_сырого': round(float(np.clip((-math.log(max(float(t4[band].mean()), 1e-9)) - D1) / D1,
                                                                          0.3, 3.0)), 4),
            'глубина': round(D1, 5)}


def _node(z):
    return measurement_node(*z)


def _tasks():
    for preset, shsh in WIDTHS_MM.items():
        for kind in calib.InAndDY:
            for r in calib.RAZBAInAndTELAnd:
                for sh in shsh:
                    for nzh in NAZhAndMY:
                        for n in NAGRAtZBrushAnd:
                            yield (preset, sh, n, nzh, kind, r)


def hash():
    """Хеш физики одиночного хода (`калибровка.хеш_физики`) и кода замера узла с осями — 16 знаков."""
    import hashlib
    h = hashlib.sha256(calib.hash_physics().encode())
    for f in (_canvas, _sweeps, measurement_node, _node):
        h.update(calib._code_without_words(textwrap.dedent(inspect.getsource(f)), without_import=True).encode())
    h.update(json.dumps([U, WithAndDY, SEEDS_NARROW, WIDTHS_MM, NAGRAtZBrushAnd, NAZhAndMY, _FLOOR_D, WithDInAndG], ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def recompute(processes=16, file=FAYL, print=print):
    import multiprocessing as mp
    import time
    processes = max(1, min(16, int(processes)))
    canvas = hash()                       # до счёта: исходник модуля, правленный во время счёта, хеша не собьёт
    tasks = list(_tasks())
    t = time.time()
    with mp.get_context('fork').Pool(processes) as pool:
        numbers = pool.map(_node, tasks, chunksize=4)
    seconds = time.time() - t
    by = dict(zip(tasks, numbers))
    presets = {}
    for preset, shsh in WIDTHS_MM.items():
        d = {'ширины_мм': list(shsh)}
        for field in PResponseLYa:          # [вид][разбавитель][ширина][нажим][нагрузка]
            d[field] = [[[[[by[(preset, sh, n, nzh, kind, r)][field] for n in NAGRAtZBrushAnd] for nzh in NAZhAndMY] for sh in shsh]
                        for r in calib.RAZBAInAndTELAnd] for kind in calib.InAndDY]
        presets[preset] = d
    t = {'хеш': canvas, 'узлов': len(tasks), 'секунд': round(seconds, 1), 'процессов': processes, 'U': list(U),
         'нагрузки': list(NAGRAtZBrushAnd), 'нажимы': list(NAZhAndMY), 'пресеты': presets}
    calib.write_part(t, file, PResponseLYa)
    print(f'ряды: {len(tasks)} узлов за {seconds:.0f} с на {processes} процессах → {file}')
    _TABLAndTsA.clear()
    return seconds


# ----------------------------------------------------------------- чтение
_TABLAndTsA = {}


def table(file=FAYL):
    if file not in _TABLAndTsA:
        if not os.path.exists(file):
            raise FileNotFoundError(f'план_покрытия: нет таблицы {file} — python3 -m кисть.ряды --пересчитать')
        t = calib.read_parts(file, lambda t: PResponseLYa)
        if t.get('хеш') != hash():
            warnings.warn(f'план_покрытия: таблица рядов снята с другой физики (хеш {t.get("хеш")}, ядро {hash()}) — '
                          'python3 -m кисть.ряды --пересчитать', stacklevel=3)
        _TABLAndTsA[file] = t
    return _TABLAndTsA[file]


def _node(t, preset, sh, pressure, load, iv, r):
    """Профиль и снятие в точке — интерполяцией по log ширины, нажиму, log нагрузки (вне осей — по краю) и разбавителю."""
    parts = [(w, _node_r(t, preset, sh, pressure, load, iv, ir)) for ir, w in calib.weights_thinner(t, r)]
    return [sum(w * h[k] for w, h in parts) for k in range(len(PResponseLYa))]


def _node_r(t, preset, sh, pressure, load, iv, ir):
    """То же в узле разбавителя номер iр."""
    pr = t['пресеты'][preset]
    shsh = pr['ширины_мм']
    i, a = calib._index(shsh, min(max(sh, shsh[0]), shsh[-1]), True)
    j, b = calib._index(NAZhAndMY, min(max(pressure, NAZhAndMY[0]), NAZhAndMY[-1]), False)
    k, c = calib._index(NAGRAtZBrushAnd, min(max(load, NAGRAtZBrushAnd[0]), NAGRAtZBrushAnd[-1]), True)
    i1, j1, k1 = min(i + 1, len(shsh) - 1), min(j + 1, len(NAZhAndMY) - 1), min(k + 1, len(NAGRAtZBrushAnd) - 1)
    out = []
    for field in PResponseLYa:
        m = pr[field][iv, ir]
        v = 0.0
        for ii, wa in ((i, 1 - a), (i1, a)):
            for jj, wb in ((j, 1 - b), (j1, b)):
                for kk, wc in ((k, 1 - c), (k1, c)):
                    v = v + wa * wb * wc * m[ii, jj, kk]
        out.append(v)
    return out


def _series(profile, removal, over, spacing, passes, drying, over_wet=None):
    """Покрытие середины поля после каждого прохода: 1D счёт глубины поперёк рядов (сетка — 1/40 ширины кисти).
    профиль — глубина одиночного хода по `U`; снятие идёт по доле глубины нового хода к его середине (0–1); поверх —
    во сколько раз глубже ложится ход на краску прежних проходов (в полную силу там, где их глубина — не меньше
    глубины одного хода, меньше — долей)."""
    f = np.asarray(profile, np.float64)
    middle_sweep = float(f[np.abs(np.asarray(U)) <= 0.3].mean())
    touch = np.clip(f / middle_sweep, 0.0, 1.0) if middle_sweep > 1e-9 else np.zeros_like(f)
    n = int(round(1.0 / 0.025))
    d = spacing * n                                           # шаг рядов в отсчётах сетки
    half = (len(f) - 1) // 2
    rows = int(math.ceil((2 * half + 4 * n) / d)) + 4
    L = int(rows * d + 2 * half + 2)
    axes = [half + 1 + k * d for k in range(rows)]
    x = np.arange(L, dtype=np.float64)
    uu = np.arange(-half, half + 1)
    sl = [(np.interp(x - response, uu, f, left=0.0, right=0.0), np.interp(x - response, uu, touch, left=0.0, right=0.0)) for response in axes]
    dry, wet = np.zeros(L), np.zeros(L)
    middle = (x >= axes[rows // 2 - 1]) & (x < axes[rows // 2 - 1] + d)    # один период в середине поля
    result = []
    for p in range(passes):
        g = over if drying or over_wet is None else over_wet     # проход по высохшему или по сырому прежнему
        for s, brush in sl:
            wet = wet * (1.0 - removal * brush) + s * (1.0 + (g - 1.0) * np.minimum(dry / max(middle_sweep, 1e-9), 1.0)) \
                if p else wet * (1.0 - removal * brush) + s
        result.append(float(1.0 - np.exp(-(dry + wet)[middle]).mean()))
        # следующий проход снимает только с рядов своего прохода: и по высохшему, и по сырому (живые серии по сырому
        # 2–3 прохода — в 0,05 от этого счёта; снятие и с прежних проходов занижало их до 0,15)
        dry, wet = dry + wet, np.zeros(L)
    return result


def plan_coverage(brush, coverage, spacing_rows=0.8, drying=False, pressure=0.6, kind=None, thinner=None, color=None,
                  px_on_mm=None, passes=None):
    """В-3: сколько проходов серии параллельных рядов и какая нагрузка дадут `покрытие` (1 − просвет грунта, как у
    `х.просвет`, в середине поля без крайних рядов) — по таблице профилей хода и закону накопления (модуль).

    шаг_рядов — расстояние между осями рядов в долях ширины кисти (`кисть.ширина`), 0,4–1,2; сушка — False: все
    проходы подряд по сырому, True: каждый следующий — по высохшему предыдущему (`х.подсушить(1.0)` между проходами);
    проходов — число проходов (None — наименьшее, при котором покрытие достижимо, до 8). Кисть, нажим, вид,
    разбавитель, цвет, px_на_мм — как у `нагрузка_для` (не кроющей краске нужен `цвет=`).

    Ответ — словарь: 'проходов', 'нагрузка' (на каждый ход), 'покрытие' (список: после каждого прохода),
    'разброс' (σ покрытия поля по сиду, оценка), 'шаг_px' (шаг рядов на холсте), 'одним_ходом' (покрытие одного хода
    этой нагрузкой — `по_таблице`). Недостижимое, вне таблицы и шаг вне 0,4–1,2 — `ValueError` с ближайшим
    достижимым. Ничего не кладёт."""
    target = float(coverage)
    if not 0.0 < target < 1.0:
        raise ValueError(f'план_покрытия: покрытие {target:g} — от 0 до 1, не включая')
    spacing = float(spacing_rows)
    if not ShAG[0] - 1e-9 <= spacing <= ShAG[1] + 1e-9:
        raise ValueError(f'план_покрытия: шаг_рядов {spacing:g} вне {ShAG[0]:g}…{ShAG[1]:g} ширины кисти — ближайшее '
                         f'{min(max(spacing, ShAG[0]), ShAG[1]):g}; реже 1,2 ряды не смыкаются (полосы грунта), чаще 0,4 — '
                         'закон не снят')
    if passes is not None and not (1 <= int(passes) <= PRResponseCanvasResponseDResponseIn):
        raise ValueError(f'план_покрытия: проходов {passes} — от 1 до {PRResponseCanvasResponseDResponseIn}')
    t = table()
    tk = calib.table()
    preset, sh = calib._preset_brushes(brush, px_on_mm)
    if preset not in t['пресеты']:
        raise ValueError(f'план_покрытия: для «{preset}» рядов не снято — пробником')
    iv, r = calib._kind(calib._kind_paints(kind, color)), calib._thinner(thinner)
    s = calib._color(color)
    if (iv or r) and s is None:
        raise ValueError(calib.NEEDED_COLOR.format(who='план_покрытия', where=calib._what_behind_paint(iv, r)))
    depth = bool(iv or r)
    kr, σk = calib._coverage(tk, preset, sh, pressure, iv, r, s)
    kr = np.maximum.accumulate(np.asarray(kr, np.float64))
    if depth:      # профили сняты тёмно-синей краской узла: глубина краски цвета — во столько раз, во сколько у одиночного хода
        kr_node = np.maximum.accumulate(np.asarray(calib._coverage(tk, preset, sh, pressure, iv, r, calib._color(calib.TsInET))[0], np.float64))

    def one(n):
        return calib._by_load(kr, n, depth=depth), calib._by_load(σk, n)

    def _D(c):
        return -math.log(max(1.0 - c, 1e-9))

    def series(n, P, c=None):
        profile, sn, pv, pvs = _node(t, preset, sh, pressure, n, iv, r)
        k = 1.0
        if depth or c is not None:
            c0 = calib._by_load(kr_node, n, depth=True) if depth else one(n)[0]
            k = _D(one(n)[0] if c is None else c) / _D(c0) if _D(c0) > 1e-9 else 1.0
        return _series(profile * k, sn, pv, spacing, P, drying, pvs)

    grid = np.exp(np.linspace(math.log(calib.NAGRAtZBrushAnd[0]), math.log(calib.NAGRAtZBrushAnd[-1]), 61))
    where = f'«{preset}» {sh:g} мм, нажим {pressure:g}, {calib.InAndDY[iv]}, разбавитель {r:g}, шаг {spacing:g}, ' + \
          ('по высохшему' if drying else 'по сырому')
    if (iv or r) and color is not None:
        where += ', цвет ({:g}, {:g}, {:g})'.format(*tuple(color)[:3])
    PP = [int(passes)] if passes is not None else list(range(1, PRResponseCanvasResponseDResponseIn + 1))
    for P in PP:
        curve = np.maximum.accumulate(np.array([series(n, P)[-1] for n in grid]))
        if curve[-1] + 1e-9 < target:
            if P == PP[-1]:
                raise ValueError(f'план_покрытия: покрытие {target:g} недостижимо — {where}: {P} '
                                 f'{"проход" if P == 1 else "проходов"} при нагрузке {calib.NAGRAtZBrushAnd[-1]:g} дают '
                                 f'{curve[-1]:.3f}; больше — чаще ряды, больше проходов, плотнее или темнее краска')
            continue
        if curve[0] > target + 1e-9:
            raise ValueError(f'план_покрытия: покрытие {target:g} недостижимо — {where}: {P} '
                             f'{"проход" if P == 1 else "проходов"} при нагрузке {calib.NAGRAtZBrushAnd[0]:g} дают уже '
                             f'{curve[0]:.3f}; меньше — реже ряды (шаг до 1,2) или прозрачнее краска')
        k = int(np.argmax(curve >= target - 1e-12))
        lo, hi = (grid[k - 1], grid[k]) if k else (grid[0], grid[0])
        for _ in range(30 if k else 0):                       # делением пополам по log нагрузки
            m = math.sqrt(lo * hi)
            lo, hi = (m, hi) if series(m, P)[-1] < target else (lo, m)
        n = round(float(hi), 5)
        row = series(n, P)
        c, σ = one(n)
        # σ поля: сдвиг одиночного хода на его σ, поделённый на корень числа независимых ходов в точке
        up = series(n, P, c=min(c + σ, 0.999))[-1]
        spread = abs(up - row[-1]) / math.sqrt(max(1.0, P / spacing))
        return {'проходов': P, 'нагрузка': n, 'покрытие': [round(v, 4) for v in row], 'разброс': round(spread, 4),
                'шаг_px': round(spacing * sh * float(px_on_mm or calib.PX_ON_MM), 1),
                'одним_ходом': round(c, 4)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python3 -m кисть.ряды', description=__doc__.split('\n')[0])
    ap.add_argument('--пересчитать', action='store_true')
    ap.add_argument('--процессов', type=int, default=16)
    a = ap.parse_args(argv)
    if a.recompute:
        recompute(a.processes)
    else:
        t = table()
        print(f'ряды: хеш {t["хеш"]}, ядро {hash()}, узлов {t["узлов"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
