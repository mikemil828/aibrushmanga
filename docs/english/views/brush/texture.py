# READING VIEW of движок/кисть/фактура.py with names in English. Not runnable; edit the original.
"""В-5: рельеф числом — `борозды_для`: σ и размах борозд одного хода (`х.борозды`, со светом и без) до кладки, и
нагрузка (или `ровно`, разбавитель) под заказ σ.

Таблица `фактура.json` рядом с модулем (смена Движка 187). Узел — прямой ход 45 мм (`путь_с_нажимом`, вход и сход
в полширины) на чистом холсте 600×400 при 8 px/мм, грунт (166, 146, 129); мерка — `х.борозды` тела следа: средняя
половина длины хода, средняя половина ширины следа (`о.ширина`). Оси: пресет × ширина (мм) × нагрузка × нажим ×
вид × (разбавитель, `ровно`) × светлота краски L (серые 30 … 245) → средние по сидам σ и размах без света, со светом
0,4 и 0,8, σ по сиду и доля тона. Сидов на узел — по ширине (`СИДЫ_ПО_ШИРИНЕ`): σ одиночного хода по сиду у кисти
уже 2 мм гуляет на 30–60 %, у широкой — на 5–10 % (окно мерки шире, борозд в нём больше).

Разбавитель r кладёт ход с ровно max(ровно, r) (`краска.разбавить`); у полукроющей и прозрачной `ровно` без
разбавителя мазок не берёт. Поэтому узлы — пары (r, ровно от r до 1), `СОЧЕТАНИЯ`.

`разброс_тона`: узел снят с 1 (как у пресетов); с 0 — те же первые сиды при нажиме 0,8, доля «тон» = σ²(0) / σ²(1)
парой одних сидов (рябь тона прядей — своя, поверх борозд), и σ²(рт) = σ²(1) · (тон + (1 − тон) · рт²).

Цвет краски. Укладка хода от цвета не зависит (5е-2): покрытие и толщина слоя те же бит в бит, цвет слоя — краска
× тон прядей. Поэтому ход узла кладётся один раз серой краской 110, а для серых 30–225 цвет и пропуск слоя
пересчитываются в рамке окна (σ того же хода живой краской — тот же до 2 %); у 245 тон прядей упирается в белое, и ход
кладётся живой краской. Счёт идёт по светлоте OKLab краски (`светлота`): хрома σ почти не меняет — у кроющей
до 5 %, у прозрачной насыщенная на 10–15 % спокойнее серой той же светлоты (замер 40 цветов).

Между узлами — по log нагрузки, log ширины, нажиму, разбавителю, `ровно` и L прямо; сила света s —
σ² = σ₀² + (σ₀,₄² − σ₀²)·(s/0,4)^p, p — по узлу 0,8. Вне осей — `ValueError` с ближайшим. Пересчёт —
`python3 -m кисть.фактура --пересчитать [--процессов 16]` (~25 мин: ПК на 16 процессах считает ~в 4 раза быстрее
одного — упор в память); хеш — хеш физики одиночного хода (`калибровка.хеш_физики`), код света и мерки, код замера
узла этого модуля.

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

FAYL = os.path.join(calib._PABrushET, 'фактура.json')

# ----------------------------------------------------------------- стенд узла
CanvasResponseLWithT = (600, 400)
CanvasResponseD = (120, 480)                 # x хода, px (45 мм при 8 px/мм); y — середина холста
# сидов на узел по ширине кисти, мм: σ одиночного хода по сиду у узкой — 30–60 %, у широкой — 5–10 % (окно шире)
SEEDS_BY_WIDTH = ((9.0, 3), (4.0, 5), (2.5, 7), (1.5, 9), (0.0, 10))
SEEDS_TONE = (1, 2, 3)            # те же сиды с `разброс_тона` 0 — парой к 1: доля тона в σ²
# мягкая и круглая от 1,5 до 3,5 мм переходят от острого кончика к плашмя: σ по ширине идёт волной — узлы чаще;
# у плоской с `ровно` σ от 4 до 6 мм падает вдвое
WIDTHS_MM = {'плоская': (2.0, 4.0, 5.0, 6.5, 9.0, 13.0, 18.0), 'мягкая': (1.0, 1.5, 2.0, 3.0, 3.5, 5.0, 15.0),
             'круглая': (1.0, 1.5, 2.0, 2.5, 3.5, 7.0),
             'плоская_узкая': (0.75, 1.5, 2.5)}
NAGRAtZBrushAnd = (0.03, 0.08, 0.18, 0.35, 0.6, 0.9)
NAZhAndMY = (0.3, 0.8)
TONE_PRESSURE = 0.8                  # нажим, при котором снимается доля тона
# Разбавитель r кладёт ход с ровно = max(ровно, r) (`краска.разбавить`; у не кроющей ровно без разбавителя не идёт).
# Узлы — пары (r, ровно мазка): по каждому r — свои узлы ровно от r до 1
RAZBAInAndTELAnd = (0.0, 0.3, 0.5, 0.7, 1.0)
EVEN_AT = {0.0: (0.0, 0.5, 1.0), 0.3: (0.3, 1.0), 0.5: (0.5, 1.0), 0.7: (0.7, 1.0), 1.0: (1.0,)}
WithResponseHETANAndYa = tuple((r, ro) for r in RAZBAInAndTELAnd for ro in EVEN_AT[r])
WithInETLResponseTY = (30, 70, 110, 150, 190, 225, 245)        # серые краски узла; до 225 — перекраской хода серой 110
PEREBrushRAWithBrushA = 225
BAZA = 110
WithAndLY = (0.4, 0.8)
PResponseLYa = ('σ0', 'σ4', 'σ8', 'р0', 'р4', 'р8', 'сσ0', 'сσ4', 'тон')   # σ и размах без света, со светом 0,4 и 0,8;
# σ по сиду без света и со светом 0,4; тон — σ² хода кисти с `разброс_тона` 0 к σ² с 1 (со светом 0,4, пары сидов)


def _even_stroke(kind, thinner, even):
    """Ровно, с которым ляжет ход: max(ровно, разбавитель); у не кроющей без разбавителя — 0."""
    r = float(thinner or 0.0)
    if kind != 'кроющая' and r <= 0.0:
        return 0.0
    return max(float(even or 0.0), r)


def _sweep(preset, width_mm, load, pressure, kind, thinner, even, rt, seed, gray):
    """Один ход узла серой краской: (холст, окно тела следа)."""
    from .canvas import Canvas
    from .brush import Brush, path_with_pressure
    from .paint import Paint
    sh = float(width_mm) * calib.PX_ON_MM
    canvas = Canvas(CanvasResponseLWithT[0], CanvasResponseLWithT[1], ground=calib.GRAtNT, px_on_mm=calib.PX_ON_MM, cores=1, hints=False)
    brush = Brush.from_file(os.path.join(calib._PREWithETY, preset + '.json'), width=sh, spread_tone=float(rt))
    yc = CanvasResponseLWithT[1] // 2
    points = path_with_pressure([(CanvasResponseD[0], yc), (CanvasResponseD[1], yc)], sh, pressure=float(pressure), entry=sh / 2, exit=sh / 2, seed=seed)
    d = {}
    if even:
        d['ровно'] = float(even)
    if thinner:
        d['разбавитель'] = float(thinner)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        response = canvas.stroke(points, Paint((gray,) * 3, kind), brush, layer='узел', seed=seed, load=float(load), **d)
    w = (response.width if response and response.width else 0) or sh
    dl = CanvasResponseD[1] - CanvasResponseD[0]
    window = (CanvasResponseD[0] + dl // 4, max(0, int(round(yc - w / 4))), CanvasResponseD[1] - dl // 4, min(CanvasResponseLWithT[1], int(round(yc + w / 4)) + 1))
    return canvas, window


def _gauge(canvas, window):
    """σ и размах борозд окна: без света, со светом `СИЛЫ`."""
    b = [canvas.furrows(window, light=False)] + [canvas.furrows(window, light=True, strength=s) for s in WithAndLY]
    return [r['σ'] for r in b] + [r['размах'] for r in b]


def _recolor(sl, ts, pr, k, frame):
    """Цвет и пропуск слоя хода в рамке (y0, y1, x0, x1) — как у того же хода краской, светлее в k раз (линейный свет):
    цвет слоя — краска × тон прядей, пропуск — цвет в степени показателя плёнки (`калибровка._плёнка`). ц, пр —
    массивы хода как лёг (вне рамки краска не меряется)."""
    from .paint import _PResponseL
    y0, y1, x0, x1 = frame
    ts0 = ts[y0:y1, x0:x1]
    ts2 = ts0 * k
    if pr is not None:
        lts = np.log(np.clip(ts0, _PResponseL, 1.0)).sum(axis=-1)
        e = np.where(lts < -1e-9, np.log(np.maximum(pr[y0:y1, x0:x1], 1e-300)).sum(axis=-1) / np.minimum(lts, -1e-9), 0.0)
        sl.transmittance[y0:y1, x0:x1] = np.exp(e[..., None] * np.log(np.clip(ts2, _PResponseL, 1.0)))
    sl.color[y0:y1, x0:x1] = ts2


def _measurements(preset, width_mm, load, pressure, kind, r, ro, rt, seed):
    """Один сид: [L][6] — мерка хода всеми серыми `СВЕТЛОТЫ`."""
    from .paint import in_linear
    canvas, window = _sweep(preset, width_mm, load, pressure, kind, r, ro, rt, seed, BAZA)
    sl = canvas.layers['узел']
    ts = np.array(sl.color[:], np.float32)
    pr = None if sl.transmittance is None else np.array(sl.transmittance[:], np.float32)
    sl.color = ts.copy()
    if pr is not None:
        sl.transmittance = pr.copy()
    field = 40                                       # свет и мерка берут соседей окна не дальше нескольких px
    frame = (max(0, window[1] - field), min(CanvasResponseLWithT[1], window[3] + field), max(0, window[0] - field), min(CanvasResponseLWithT[0], window[2] + field))
    linear = lambda v: float(in_linear(np.array([float(v)], np.float64))[0])
    row = []
    for L in WithInETLResponseTY:
        if L <= PEREBrushRAWithBrushA:
            _recolor(sl, ts, pr, linear(L) / linear(BAZA), frame)
            row.append(_gauge(canvas, window))
        else:
            row.append(_gauge(*_sweep(preset, width_mm, load, pressure, kind, r, ro, rt, seed, L)))
    return row


def measurement_node(preset, width_mm, load, pressure, kind, thinner, even):
    """Узел (кисть с `разброс_тона` 1): [L][поле `ПОЛЯ`] — средние по сидам; у нажима `ТОН_НАЖИМ` — и доля тона."""
    seeds = seeds_for(width_mm)
    by = np.array([_measurements(preset, width_mm, load, pressure, kind, thinner, even, 1.0, with_) for with_ in seeds])
    tone = np.ones(len(WithInETLResponseTY))                    # (сиды, L, 6)
    if float(pressure) == TONE_PRESSURE:
        ts = [with_ for with_ in SEEDS_TONE if with_ in seeds]
        without = np.array([_measurements(preset, width_mm, load, pressure, kind, thinner, even, 0.0, with_) for with_ in ts])
        with1 = (by[:len(ts), :, 1] ** 2).mean(axis=0)
        tone = np.where(with1 > 1e-6, (without[:, :, 1] ** 2).mean(axis=0) / np.maximum(with1, 1e-6), 1.0)
    sr, sσ = by.mean(axis=0), by[:, :, :2].std(axis=0)
    return [[round(float(v), 3) for v in list(sr[i]) + list(sσ[i]) + [tone[i]]] for i in range(len(WithInETLResponseTY))]


def seeds_for(width_mm):
    n = next(n for sh, n in SEEDS_BY_WIDTH if float(width_mm) >= sh)
    return tuple(range(1, n + 1))


def _node(z):
    return measurement_node(*z)


def _tasks():
    for preset, shsh in WIDTHS_MM.items():
        for kind in calib.InAndDY:
            for r, ro in WithResponseHETANAndYa:
                if kind != 'кроющая' and r == 0.0 and ro > 0.0:
                    continue                           # не кроющей ровно без разбавителя не идёт
                for sh in shsh:
                    for nzh in NAZhAndMY:
                        for n in NAGRAtZBrushAnd:
                            yield (preset, sh, n, nzh, kind, r, ro)


def hash():
    """Хеш физики одиночного хода, света и мерки борозд, кода замера узла с осями — 16 знаков."""
    import hashlib
    h = hashlib.sha256(calib.hash_physics().encode())
    for name in ('свет.py', 'глаз.py'):
        with open(os.path.join(calib._PABrushET, name), encoding='utf-8') as f:
            h.update(calib._code_without_words(f.read()).encode())
    from . import canvas
    for f in (canvas.Canvas.image, canvas.Canvas._color, canvas.Canvas.relief, _sweep, _gauge, _recolor, _measurements,
              measurement_node, seeds_for):
        h.update(calib._code_without_words(textwrap.dedent(inspect.getsource(f)), without_import=True).encode())
    h.update(json.dumps([CanvasResponseLWithT, CanvasResponseD, SEEDS_BY_WIDTH, SEEDS_TONE, WIDTHS_MM, NAGRAtZBrushAnd, NAZhAndMY, TONE_PRESSURE, WithResponseHETANAndYa,
                         WithInETLResponseTY, PEREBrushRAWithBrushA, BAZA, WithAndLY], ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def recompute(processes=16, file=FAYL, print=print):
    """Снять таблицу заново (не больше 16 процессов) и записать `файл`. Отвечает секундами счёта."""
    import multiprocessing as mp
    import time
    processes = max(1, min(16, int(processes)))
    canvas = hash()                       # до счёта: исходник модуля, правленный во время счёта, хеша не собьёт
    tasks = list(_tasks())
    tasks.sort(key=lambda z: -z[1] * len(seeds_for(z[1])))         # тяжёлые — первыми
    t = time.time()
    with mp.get_context('fork').Pool(processes) as pool:
        numbers = pool.map(_node, tasks, chunksize=2)
    seconds = time.time() - t
    by = dict(zip(tasks, numbers))
    presets = {}
    for preset, shsh in WIDTHS_MM.items():
        # [вид][сочетание (разбавитель, ровно)][ширина][нажим][нагрузка][L][поле]; не кроющей ровно без разбавителя не
        # идёт — там узел (0, 0)
        presets[preset] = {'ширины_мм': list(shsh), 'узлы': [[[[[by[(preset, sh, n, nzh, kind, r, ro if (kind == 'кроющая' or r) else 0.0)]
                                                             for n in NAGRAtZBrushAnd] for nzh in NAZhAndMY] for sh in shsh]
                                                          for r, ro in WithResponseHETANAndYa] for kind in calib.InAndDY]}
    t = {'хеш': canvas, 'узлов': len(tasks), 'секунд': round(seconds, 1), 'процессов': processes,
         'оси': {'нагрузка': NAGRAtZBrushAnd, 'нажим': NAZhAndMY, 'сочетания': WithResponseHETANAndYa, 'L': WithInETLResponseTY, 'сила': WithAndLY,
                 'поля': PResponseLYa},
         'пресеты': presets}
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(t, f, ensure_ascii=False, separators=(',', ':'))
    print(f'фактура: {len(tasks)} узлов за {seconds:.0f} с на {processes} процессах → {file}')
    _TABLAndTsA.clear()
    return seconds


# ----------------------------------------------------------------- чтение
_TABLAndTsA = {}


def table(file=FAYL):
    if file not in _TABLAndTsA:
        if not os.path.exists(file):
            raise FileNotFoundError(f'борозды_для: нет таблицы {file} — python3 -m кисть.фактура --пересчитать')
        with open(file, encoding='utf-8') as f:
            t = json.load(f)
        for pr in t['пресеты'].values():
            pr['узлы'] = np.asarray(pr['узлы'], np.float64)
        if t.get('хеш') != hash():
            warnings.warn(f'борозды_для: таблица фактуры снята с другой физики (хеш {t.get("хеш")}, ядро {hash()}) — '
                          'python3 -m кисть.фактура --пересчитать', stacklevel=3)
        _TABLAndTsA[file] = t
    return _TABLAndTsA[file]


def _weights(axes, v, log=False):
    """[(индекс, вес)] линейной интерполяции по оси (v уже в пределах)."""
    i, a = calib._index(axes, v, log)
    return [(i, 1.0 - a)] + ([(i + 1, a)] if a > 0 and i + 1 < len(axes) else [])


def _in_within(name, v, axes, where=''):
    if not (axes[0] - 1e-9 <= v <= axes[-1] + 1e-9):
        nearest = min(max(v, axes[0]), axes[-1])
        raise ValueError(f'борозды_для: {name} {v:g} вне таблицы{where} — в ней {axes[0]:g}…{axes[-1]:g}; ближайшее — {nearest:g}')


def _point(t, preset, sh, load, pressure, iv, razb, even, rt, L):
    """Поля σ и размаха (`ПОЛЯ` без «тон») в точке. Разбавитель r кладёт ход с ровно max(ровно, r): у узла r — узлы
    ровно от r; по `разброс_тона` — σ² = σ²(1) · (тон + (1 − тон) · рт²), тон — у узла нажима `ТОН_НАЖИМ`."""
    pr = t['пресеты'][preset]
    at = pr['узлы']
    it = NAZhAndMY.index(TONE_PRESSURE)
    result = np.zeros(len(PResponseLYa) - 1)
    for ir, wr in _weights(RAZBAInAndTELAnd, razb):
        rn = RAZBAInAndTELAnd[ir]
        axes = EVEN_AT[rn] if (iv == 0 or rn > 0) else (0.0,)
        ro = min(max(float(even or 0.0), rn), axes[-1]) if (iv == 0 or rn > 0) else 0.0
        for io, wo in _weights(axes, ro):
            k = WithResponseHETANAndYa.index((rn, axes[io]))
            for ish, wsh in _weights(pr['ширины_мм'], sh, True):
                for ig, wg in _weights(NAGRAtZBrushAnd, load, True):
                    for iL, wL in _weights(WithInETLResponseTY, L):
                        tone = at[iv, k, ish, it, ig, iL, -1]
                        m = tone + (1.0 - tone) * rt ** 2
                        for in_, wn in _weights(NAZhAndMY, pressure):
                            w = wr * wo * wsh * wg * wL * wn
                            if w:
                                result += w * at[iv, k, ish, in_, ig, iL, :-1] * math.sqrt(max(m, 0.0))
    return result


def _with_light(p, strength):
    """(σ, размах) при силе света: σ² = σ₀² + (σ₀,₄² − σ₀²)·(s/0,4)^p, p — по узлу 0,8; размах — тем же законом."""
    if strength <= 0:
        return p[0], p[3]
    def law(in0, in4, in8):
        d4, d8 = in4 ** 2 - in0 ** 2, in8 ** 2 - in0 ** 2
        if d4 <= 1e-9:
            return math.sqrt(max(in0 ** 2 + max(d8, 0.0) * (strength / WithAndLY[1]) ** 2, 0.0))
        p = float(np.clip(math.log(max(d8, 1e-9) / d4, 2.0), 0.5, 3.0))
        return math.sqrt(max(in0 ** 2 + d4 * (strength / WithAndLY[0]) ** p, 0.0))
    return law(p[0], p[1], p[2]), law(p[3], p[4], p[5])


def _axis_paints(brush, kind, thinner, even, color, px_on_mm):
    from .eye import lightness
    preset, sh = calib._preset_brushes(brush, px_on_mm)
    if preset not in WIDTHS_MM:
        raise ValueError(f'борозды_для: кисть «{preset}» — таблица фактуры для {", ".join(WIDTHS_MM)}; борозды этой '
                         'кисти — пробником и `х.борозды`')
    _in_within('ширина', sh, WIDTHS_MM[preset], f' «{preset}» (мм)')
    iv = calib._kind(calib._kind_paints(kind, color))
    if color is None:
        raise ValueError('борозды_для: нужен цвет=… — σ борозд идёт за светлотой краски: на этом грунте тёмная (L 30) '
                         'рябит почти вдвое сильнее средней (L 90); дай краску хода: (r, g, b), `Краска` или `цвет(L, C, h)`')
    calib._color(color)                                    # проверка вида цвета
    L = float(lightness(np.asarray(tuple(color)[:3], np.float64)))
    rt = float(brush.spread_tone)
    _in_within('разброс_тона кисти', rt, (0.0, 1.0))
    return preset, sh, iv, L, rt


BrushRResponseET = 0.2      # заказ σ: нагрузка не ниже этой — плёнка, а не протирка; ниже — только если иначе заказ не взять


def furrows_for(brush, σ=None, load=None, pressure=0.6, kind=None, thinner=None, even=None, strength=None,
                color=None, light=True, px_on_mm=None):
    """В-5: борозды одного хода числом — то, что `х.борозды` тела следа покажет после кладки, до неё.

    С нагрузкой (без σ) — прогноз: {'σ', 'размах' — со светом силы `сила` (умолчание — рабочая 0,4), 'σ_без_света',
    'размах_без_света', 'разброс' и 'разброс_без_света' — σ по сиду (один ход гуляет вокруг среднего на столько)}.
    С заказом σ= (со светом; `свет=False` — без него) — то, что дало бы это σ: {'нагрузка', 'ровно', 'разбавитель'} и
    прогноз в этой точке. Заданные `ровно` и `разбавитель` не трогаются — подбирается нагрузка (а с данной нагрузкой —
    `ровно` у кроющей или разбавитель); не заданные: сперва нагрузка без `ровно` и разбавителя (из нескольких —
    наибольшая: кроет плотнее); нет её или она ниже 0,2 (`КРОЕТ`: протирка, сквозь неё грунт) — `ровно` (кроющая), затем
    разбавитель: наименьшие, при которых заказ берётся нагрузкой от 0,2, иначе — с наибольшей нагрузкой.

    Кисть — пресет `плоская`, `мягкая`, `круглая`, `плоская_узкая` (ширина в мм или px при `px_на_мм`); `разброс_тона`
    берётся из неё (0–1). цвет — краска хода (sRGB, `Краска` — с неё и вид, `цвет(L, C, h)`), обязателен. Стенд таблицы —
    чистый грунт (166, 146, 129), 8 px/мм, ход 45 мм; по своему грунту, по лежащей краске, рядами — ближе к пробнику.
    Вне осей и недостижимое — `ValueError` с ближайшим. Только счёт: ничего не кладёт."""
    t = table()
    preset, sh, iv, L, rt = _axis_paints(brush, kind, thinner, even, color, px_on_mm)
    _in_within('нажим', float(pressure), NAZhAndMY)
    strength = 0.4 if strength is None else float(strength)
    _in_within('сила', strength, (0.0, WithAndLY[1]))
    for name, v, axes in (('разбавитель', thinner, RAZBAInAndTELAnd), ('ровно', even, (0.0, 1.0))):
        if v is not None:
            _in_within(name, float(v), axes)
    opaque = iv == 0

    def forecast(n, r, ro):
        p = _point(t, preset, sh, n, float(pressure), iv, r, ro, rt, L)
        with_, rz = _with_light(p, strength)
        sb = math.sqrt(max(p[6] ** 2 + (p[7] ** 2 - p[6] ** 2) * (strength / WithAndLY[0]) ** 2, 0.0)) if strength < WithAndLY[0] else p[7]
        return {'σ': round(float(with_), 2), 'размах': round(float(rz), 1), 'σ_без_света': round(float(p[0]), 2),
                'размах_без_света': round(float(p[3]), 1), 'разброс': round(float(sb), 2),
                'разброс_без_света': round(float(p[6]), 2)}

    if σ is None:
        if load is None:
            raise ValueError('борозды_для: дай нагрузка= (прогноз) или σ= (заказ)')
        n = _load(load)
        return forecast(n, float(thinner or 0.0), float(even or 0.0))

    target, field = float(σ), ('σ' if light else 'σ_без_света')
    measure = lambda n, r, ro: forecast(n, r, ro)[field]
    grid_n = np.exp(np.linspace(math.log(NAGRAtZBrushAnd[0]), math.log(NAGRAtZBrushAnd[-1]), 61))
    grid_p = np.linspace(0.0, 1.0, 41)

    def roots(f, xs):
        """x, где f(x) = цель (линейно между точками сетки), по возрастанию."""
        v = np.array([f(x) - target for x in xs])
        brush = []
        for i in range(len(xs) - 1):
            if v[i] == 0 or v[i] * v[i + 1] < 0:
                brush.append(float(xs[i] + (xs[i + 1] - xs[i]) * v[i] / (v[i] - v[i + 1])) if v[i] != v[i + 1] else float(xs[i]))
        if v[-1] == 0:
            brush.append(float(xs[-1]))
        return brush, v + target

    where = f'«{preset}» {sh:g} мм, нажим {float(pressure):g}, {calib.InAndDY[iv]}, L краски {L:.0f}, разброс_тона {rt:g}'
    if load is not None:                     # нагрузка дана — подобрать ровно или разбавитель
        n = _load(load)
        if opaque and even is None:
            name, f = 'ровно', (lambda x: measure(n, float(thinner or 0.0), x))
        elif thinner is None:
            name, f = 'разбавитель', (lambda x: measure(n, x, float(even or 0.0)))
        else:
            raise ValueError('борозды_для: с σ= и нагрузка= оставь свободным ровно= (кроющая) или разбавитель=')
        brush, v = roots(f, grid_p)
        if not brush:
            raise ValueError(f'борозды_для: σ {target:g} L недостижимо нагрузкой {n:g} ({where}): {name} 0…1 даёт '
                             f'{v.min():.1f}…{v.max():.1f} L; другая нагрузка — без нагрузка=')
        x = brush[0]
        r = x if name == 'разбавитель' else float(thinner or 0.0)
        ro = x if name == 'ровно' else float(even or 0.0)
        return dict(load=n, even=round(ro, 3), thinner=round(r, 3), **forecast(n, r, ro))

    r0, ro0 = float(thinner or 0.0), float(even or 0.0)
    brush, v = roots(lambda x: measure(x, r0, ro0), grid_n)
    best = (brush[-1], r0, ro0) if brush else None                 # (нагрузка, разбавитель, ровно): из корней — наибольшая нагрузка
    min, max = float(v.min()), float(v.max())
    free = [p for p in (('ровно',) if opaque and even is None else ()) + (('разбавитель',) if thinner is None else ())]
    for name in free:
        if best is not None and best[0] >= BrushRResponseET:
            break
        for x in grid_p[1:]:
            r, ro = (x, ro0) if name == 'разбавитель' else (r0, x)
            brush, vv = roots(lambda n: measure(n, r, ro), grid_n)
            min, max = min(min, float(vv.min())), max(max, float(vv.max()))
            if not brush or (best is not None and brush[-1] <= best[0]):
                continue
            # между узлами сетки — точнее: тот же поиск по самому полю при найденной нагрузке
            n = brush[-1]
            kk, _ = roots((lambda y: measure(n, y, ro0)) if name == 'разбавитель' else (lambda y: measure(n, r0, y)),
                          np.linspace(max(0.0, x - 0.025), x, 6))
            x = kk[0] if kk else x
            best = (n, x, ro0) if name == 'разбавитель' else (n, r0, x)
            if n >= BrushRResponseET:
                break
    if best is not None:
        n, r, ro = round(best[0], 4), round(best[1], 3), round(best[2], 3)
        return dict(load=n, even=ro, thinner=r, **forecast(n, r, ro))
    nearest = min if target < min else max
    than = ' и '.join(free) or 'нагрузка'
    raise ValueError(f'борозды_для: σ {target:g} L{"" if light else " без света"} недостижимо ({where}; подбиралось: '
                     f'нагрузка{", " + than if free else ""}): в таблице {min:.1f}…{max:.1f} L; ближайшее — '
                     f'{nearest:.1f} L' + ('; ровнее — шире кисть, мягче волос или светлее/ближе к грунту краска'
                                        if target < min else '; рябее — темнее краска, уже щетина, легче рука'))


def _load(load):
    from .tool import NAGRAtZBrushAnd as AndMENA
    n = AndMENA[load] if isinstance(load, str) else float(load)
    _in_within('нагрузка', n, NAGRAtZBrushAnd)
    return n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python3 -m кисть.фактура', description=__doc__.split('\n')[0])
    ap.add_argument('--пересчитать', action='store_true')
    ap.add_argument('--процессов', type=int, default=16)
    a = ap.parse_args(argv)
    if a.recompute:
        recompute(a.processes)
    else:
        t = table()
        print(f'фактура: хеш {t["хеш"]}, ядро {hash()}, узлов {t["узлов"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
