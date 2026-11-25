# READING VIEW of движок/кисть/смешение.py with names in English. Not runnable; edit the original.
"""В-4 (К-17 баланса): смешение по сырому числом — `смешение_для`: какой станет след одного хода по лежащей сырой
краске (цвет тела следа, доля тона лежащего в нём, `о.сырое`, `о.цвет`) до кладки, и нагрузка (и, если нужно, нажим)
под заказ доли.

Почему не таблица (замер смены Движка 188). Цвет следа по сырому идёт не за одним числом: кисть набирает лежащую
краску (смесь K/S), кладёт её борозды и пряди с тоном прядей, вмешивает на месте и перекрывает лежащее покрытием —
каждый пиксель следа своя смесь, а `х.средний` берёт среднее L по пикселям. Одна доля смеси K/S на узел, снятая одной
парой красок, на другой паре того же хода ошибается до 30 L (светлое по тёмному и тёмное по светлому — доли 0,04 и
0,36 одного хода); семья пикселя «смесь K/S, лежащее, грунт», подогнанная по 8 парам красок попиксельно, — до 4,5 L
на прозрачной. Цвет вне таблицы краски не держит. Поэтому счёт — сам ход на своём стенде: холст стенда в памяти,
не холст модели; ход 0,07 с, 24 сида и подмалёвок — 2–3 с; заказ доли — 8–40 с.

Стенд (в мм — тот же при любом `px_на_мм`; при 8 px/мм — 600×400): чистый грунт (166, 146, 129), гладкий; подмалёвок —
сырая заливка 50×25 мм одной кроющей краской `лежит`, `плоская` 5 мм, `ровно=1`, нажим 0,6, ряды вдоль короткой стороны
шагом 3,25 мм (сиды `СИДЫ_ЗАЛИВКИ`); его толщина — нагрузкой заливки (`подмалёвок=`) или сырой толщиной (`толщина=`,
единицы `Слой.толщина`; нагрузка заливки под неё — по кривой толщины заливки). Поверх — ход `путь_с_нажимом` прямо
(45 мм или `длина=` 10–200 мм: по сырому кисть пачкается по ходу — тело длинного хода темнее и ближе к лежащему) поперёк
рядов кистью модели (её копией: память кисти не трогается), сиды `СИДЫ` 101–124 (не 1–12 — те у живой
сверки).

Мерка — тело следа: средняя половина длины хода, средняя половина ширины следа (`о.ширина`), без света. Цвет — как у
`х.средний` (L — среднее L серой шкалы по пикселям, C и h — из средних a, b OKLab). Доля тона лежащего — где цвет тела
следа лёг на отрезке «краска хода → лежащее» в OKLab (проекция; 0 — чистая краска хода, 1 — тон лежащего; у прозрачной
сюда входит и просвет лежащего сквозь плёнку). Лежащее — цвет заливки стенда в том же окне до хода.

Только счёт: холст, кисть и журнал модели не трогаются.
"""
import math
import warnings

import numpy as np

GRAtNT = (166, 146, 129)
PCanvas = 8.0                                   # px/мм стенда по умолчанию
DLAndNA = 45.0                               # ход стенда по умолчанию, мм (`длина=`: 10–200)
DLAndNY = (10.0, 200.0)
FIELDS_MM = (15.0, 12.5)                     # от края стенда до хода и до заливки, мм
HEIGHT_MM = 50.0                           # стенд: (длина + 30) × 50 мм; заливка — от 12,5 мм с краёв (при 45 мм —
                                           # 600×400 и 400×200 px при 8 px/мм)
ROWS_MM = (5.0, 3.25, 1.25)                # ширина `плоская` заливки, шаг рядов, вход и сход
WithAndDY = tuple(range(101, 125))              # сиды прогноза: в крутом месте один ход гуляет на 5–15 L
SEEDS_SEARCH = WithAndDY[:8]                     # сиды перебора нагрузки под заказ доли (итог — всеми `СИДЫ`)
SEEDS_FILLS = 1000                        # сид ряда заливки — 1000 + номер ряда
LOADS_FILLS = (0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0)   # кривая толщины заливки (под `толщина=`)
NAGRAtZBrushAnd = (0.03, 1.0)                     # пределы нагрузки хода
GRID_LOADS = tuple(float(v) for v in np.round(np.exp(np.linspace(math.log(0.03), 0.0, 11)), 4))
NAZhAndMY = (0.35, 0.5, 0.65, 0.8)            # перебор нажима под заказ доли, если нажим не дан (ближние к 0,6 — первыми)
NAZhAndM = 0.6

LAYER_LIES = 'подмалёвок'                 # слой заливки стенда
LAYER_SWEEP = 'ход'                         # слой хода стенда, когда ход — не в слой лежащего (над ним)
LAYER_BOTTOM = 'низ'                          # В-4-3: нижняя заливка стенда, когда сырое и в слое хода, и под ним
RAWithBrushLADBrushAnd = ('свой', 'ниже', 'поверх', 'два')   # ход в слой лежащего; в слой выше с `мешать_ниже`; в слой выше без
                                           # смешения; в слой лежащего с `мешать_ниже` над второй сырой заливкой
SEEDS_FILLS_TOP = 2000                   # 'два': сид ряда верхней заливки — 2000 + номер ряда
WithYRResponse = 1e-4                                # сырая толщина, ниже которой сырого нет
_ZALAndInBrushAnd = {}                              # (rgb, нагрузка, px) -> Холст стенда с заливкой (память процесса)
_TResponseLShchAndNY = {}                              # px -> {как: кривая (нагрузка заливки, толщина)}
HOW_FILLS = ('ровно', 'рядами', 'дважды')   # заливки стенда по толщине: `ровно=1`; без `ровно`; два прохода
NERResponseInNResponse = 0.2                              # В-4-3: σ/среднее сырой толщины слоя места выше — слой бороздами (ряды без
                                           # `ровно`: 0,35–0,45; ряды `ровно=1`: 0,07–0,09) — заливка стенда 'рядами'


def _px(v, px):
    return float(v) * px


def _geometry(dl):
    """Стенд под ход длиной дл мм: (ширина стенда, (x0, y0, x1, y1) заливки, (x от, x до) хода), мм."""
    sh = float(dl) + 2 * FIELDS_MM[0]
    z = FIELDS_MM[1]
    return sh, (z, z, sh - z, HEIGHT_MM - z), (FIELDS_MM[0], FIELDS_MM[0] + float(dl))


def _canvas(px, dl=DLAndNA):
    from .canvas import Canvas
    return Canvas(int(round(_geometry(dl)[0] * px)), int(round(HEIGHT_MM * px)), ground=GRAtNT, px_on_mm=px, cores=1,
                 hints=False)


def _rows(canvas, rgb, load, px, seed, dl, how, layer):
    """Ряды заливки стенда на холст `х` (кладёт на `х`!): `плоская` 5 мм, нажим 0,6, сиды `сид` + номер ряда."""
    from .brush import Brush
    from .path import path_with_pressure
    from .paint import Paint
    from . import calibration as calib
    import os
    sh, spacing, vkh = (_px(v, px) for v in ROWS_MM)
    brush = Brush.from_file(os.path.join(calib._PREWithETY, 'плоская.json'), width=sh)
    x0, y0, x1, y1 = (_px(v, px) for v in _geometry(dl)[1])
    i = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for pass_ in range(2 if how == 'дважды' else 1):
            x = x0 + sh / 4 + pass_ * spacing / 2
            while x <= x1 - sh / 4 + 1e-6:
                t = path_with_pressure([(x, y0), (x, y1)], sh, pressure=0.6, entry=vkh, exit=vkh, seed=seed + i)
                canvas.stroke(t, Paint(rgb, 'кроющая'), brush, layer=layer, seed=seed + i,
                        load=float(load), **({'ровно': 1.0} if how == 'ровно' else {}))
                x, i = x + spacing, i + 1


def _option(lies, load, how):
    return tuple(round(float(v), 3) for v in tuple(lies)[:3]), round(float(load), 4), str(how)


def _in_memory(option, build):
    import copy
    if option not in _ZALAndInBrushAnd:
        canvas = build()
        if len(_ZALAndInBrushAnd) >= 24:
            _ZALAndInBrushAnd.pop(next(iter(_ZALAndInBrushAnd)))
        _ZALAndInBrushAnd[option] = canvas
    return copy.deepcopy(_ZALAndInBrushAnd[option])


def fill(lies, load, px=PCanvas, seed=SEEDS_FILLS, dl=DLAndNA, how='ровно'):
    """Холст стенда с сырым подмалёвком в слое `СЛОЙ_ЛЕЖИТ`: краска `лежит` (sRGB), нагрузка заливки `нагрузка`, сид
    ряда — `сид` + номер ряда, стенд под ход длиной `дл` мм; `как` — 'ровно': ряды с `ровно=1` (сырая толщина до 0,75),
    'рядами': без `ровно`, как обычный подмалёвок (до 1,05), 'дважды': два прохода рядами без `ровно`, второй со
    сдвигом на полшага (густой подмалёвок). Из памяти процесса — копией (холст памяти не меняется)."""
    rgb, n, brush = _option(lies, load, how)
    option = (rgb, n, float(px), int(seed), round(float(dl), 2), brush)

    def build():
        canvas = _canvas(px, dl)
        _rows(canvas, rgb, float(load), px, int(seed), dl, brush, LAYER_LIES)      # нагрузка — как дана (ключ — округлённая)
        return canvas
    return _in_memory(option, build)


def fill_two(bottom, top, px=PCanvas, dl=DLAndNA):
    """В-4-3: стенд с сырым в двух слоях — `низ` и `верх` = (rgb, нагрузка заливки, как): нижняя заливка в слое
    `СЛОЙ_НИЗ` (сиды `СИДЫ_ЗАЛИВКИ`), по ней сырой — верхняя в `СЛОЙ_ЛЕЖИТ` (сиды `СИДЫ_ЗАЛИВКИ_ВЕРХ`, без смешения
    с нижней: ряды без `мешать_ниже`). Ход 'два' — в `СЛОЙ_ЛЕЖИТ` с `мешать_ниже=True`. Копией из памяти процесса."""
    n, in_ = _option(*bottom), _option(*top)
    option = ('два', n, in_, float(px), round(float(dl), 2))

    def build():
        canvas = _canvas(px, dl)
        _rows(canvas, n[0], float(bottom[1]), px, SEEDS_FILLS, dl, n[2], LAYER_BOTTOM)
        _rows(canvas, in_[0], float(top[1]), px, SEEDS_FILLS_TOP, dl, in_[2], LAYER_LIES)
        return canvas
    return _in_memory(option, build)


def wet_thickness(canvas, region):
    """Средняя сырая толщина краски в прямоугольнике (x0, y0, x1, y1): сумма по слоям толщина × влажность × (1 − 0,85 ×
    схват) — то, что кисть может взять (`укладка`, сырая краска под мазком); единицы `Слой.толщина`."""
    from .numbers import _WithCanvasInAT
    where = canvas._region(region)
    total = None
    for with_ in canvas.layers.values():
        h = np.asarray(with_.thickness[where], np.float64) * np.asarray(with_.moisture[where], np.float64)
        h = h * (1.0 - _WithCanvasInAT * np.asarray(with_.degree_of_set[where], np.float64))
        total = h if total is None else total + h
    return 0.0 if total is None or total.size == 0 else float(total.mean())


def _window_sweep(px, w, dl=DLAndNA):
    yc = HEIGHT_MM * px / 2
    a, b = (_px(v, px) for v in _geometry(dl)[2])
    dl = b - a
    return (int(round(a + dl / 4)), int(round(yc - w / 4)), int(round(b - dl / 4)), int(round(yc + w / 4)) + 1)


def thickness_fills(load, px=PCanvas, how='ровно'):
    """Сырая толщина подмалёвка стенда (нагрузка заливки) в окне мерки хода."""
    canvas = fill((128, 128, 128), load, px, how=how)
    w = ROWS_MM[0] * px
    return wet_thickness(canvas, _window_sweep(px, w))


def _curves(px):
    if px not in _TResponseLShchAndNY:
        _TResponseLShchAndNY[px] = {how: [(n, thickness_fills(n, px, how)) for n in LOADS_FILLS] for how in HOW_FILLS}
    return _TResponseLShchAndNY[px]


def _bottom_test_bench(px):
    """Самая тонкая сырая заливка стенда (ряды `ровно=1`, нагрузка 0,1)."""
    return _curves(px)[HOW_FILLS[0]][0][1]


def _load_fills(thickness, px, unevenly=False):
    """(нагрузка заливки, как) под сырую толщину: до верха рядов `ровно=1` (0,75 при нагрузке 1) — ими (числа В-4 те
    же), толще — рядами без `ровно` (до 1,05: обычный подмалёвок плоской 5 мм нагрузкой 1), ещё толще — двумя проходами
    рядами (`КАК_ЗАЛИВКИ`)."""
    kr = _curves(px)
    bottom, top = kr[HOW_FILLS[0]][0][1], kr[HOW_FILLS[-1]][-1][1]
    if not (bottom - 1e-9 <= thickness <= top + 1e-9):
        nearest = float(np.clip(thickness, bottom, top))
        times = ', '.join(f'{brush} до {kr[brush][-1][1]:.3g}' for brush in HOW_FILLS)
        raise ValueError(f'смешение_для: сырая толщина лежащего {thickness:.3g} вне стенда — заливка даёт '
                         f'{bottom:.3g}…{top:.3g} ({times}; нагрузки {LOADS_FILLS[0]:g}…{LOADS_FILLS[-1]:g}); '
                         f'ближайшее — {nearest:.3g}')
    for how in HOW_FILLS[1:] if unevenly else HOW_FILLS:      # В-4-3: слой бороздами — без `ровно`
        hh = [h for _, h in kr[how]]
        if unevenly and how == HOW_FILLS[1] and thickness < hh[0]:
            return float(np.interp(thickness, [h for _, h in kr[HOW_FILLS[0]]], [n for n, _ in kr[HOW_FILLS[0]]])), HOW_FILLS[0]
        if thickness <= hh[-1] + 1e-9 or how == HOW_FILLS[-1]:
            return float(np.interp(thickness, hh, [n for n, _ in kr[how]])), how


def _lab(canvas, window):
    """Средние по пикселям окна: (L серой шкалы, L, a, b OKLab)."""
    from .eye import _kind, _frame, oklab, _L
    lab = oklab(_kind(canvas, _frame(canvas, window), False)).reshape(-1, 3)
    return float(_L(lab[:, 0]).mean()), lab.mean(axis=0)


def share_tone(lab_trace, lab_paint, lab_lies):
    """Где цвет следа на отрезке «краска хода → лежащее» в OKLab: 0 — краска, 1 — лежащее (проекция, не обрезана)."""
    d = np.asarray(lab_lies, np.float64) - np.asarray(lab_paint, np.float64)
    n = float(d @ d)
    if n < 1e-10:
        return float('nan')
    return float((np.asarray(lab_trace, np.float64) - lab_paint) @ d / n)


def measurement_sweep(canvas, brush, paint, load, pressure, seed, thinner=None, even=None, px=PCanvas, dl=DLAndNA, layout='свой'):
    """Один ход стенда по холсту стенда `х` (кладёт на `х`!): {'L', 'lab', 'окно', 'ответ'} — мерка тела следа и
    ответ мазка. Раскладка: 'свой' — ход в слой заливки, 'ниже' — в слой над ней с `мешать_ниже=True`, 'поверх' — в
    слой над ней без смешения, 'два' — в слой верхней заливки `заливка_два` с `мешать_ниже=True`. Им же меряется живое
    в тесте пункта."""
    from .path import path_with_pressure
    sh = float(brush.width)
    yc = HEIGHT_MM * px / 2
    a, b = (_px(v, px) for v in _geometry(dl)[2])
    t = path_with_pressure([(a, yc), (b, yc)], sh, pressure=float(pressure), entry=sh / 2, exit=sh / 2, seed=seed)
    d = {}
    if thinner:
        d['разбавитель'] = float(thinner)
    if even:
        d['ровно'] = float(even)
    if layout in ('ниже', 'два'):
        d['мешать_ниже'] = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        response = canvas.stroke(t, paint, brush.copy(), layer=LAYER_LIES if layout in ('свой', 'два') else LAYER_SWEEP, seed=seed,
                    load=load, **d)
    w = (response.width if response is not None and response.width else 0) or sh
    window = _window_sweep(px, w, dl)
    L, lab = _lab(canvas, window)
    return {'L': L, 'lab': lab, 'окно': window, 'ответ': response}


def _mean_color(Ls, labs):
    from .eye import color, hex
    L = float(np.mean(Ls))
    a, b = float(np.mean([v[1] for v in labs])), float(np.mean([v[2] for v in labs]))
    C = math.hypot(a, b)
    h = math.degrees(math.atan2(b, a)) % 360.0 if C >= 1e-6 else 0.0
    rgb = color(L, C, h)
    return {'L': round(L, 2), 'C': round(C, 4), 'h': round(h, 1), 'rgb': rgb, 'hex': hex(rgb)}


def _response_mean(responses):
    from .eye import color, hex
    tsv = [response.color for response in responses if response is not None and response.color is not None]
    if tsv:
        L = float(np.mean([c['L'] for c in tsv]))
        a = float(np.mean([c['C'] * math.cos(math.radians(c['h'])) for c in tsv]))
        b = float(np.mean([c['C'] * math.sin(math.radians(c['h'])) for c in tsv]))
        C = math.hypot(a, b)
        h = math.degrees(math.atan2(b, a)) % 360.0 if C >= 1e-6 else 0.0
        rgb = color(L, C, h)
        color_response = {'L': round(L, 2), 'C': round(C, 4), 'h': round(h, 1), 'rgb': rgb, 'hex': hex(rgb)}
    else:
        color_response = None
    wet = {brush: round(float(np.mean([response.wet[brush] for response in responses if response is not None])), 3)
           for brush in ('по сырому', 'подобрано', 'к положенному')}
    return {'цвет': color_response, 'сырое': wet}


def _paint(color, kind):
    from .paint import Paint
    from . import calibration as calib
    if color is None:
        raise ValueError('смешение_для: нужен цвет=… — краска хода: (r, g, b), `Краска` или `цвет(L, C, h)`')
    calib._color(color)
    return Paint(tuple(int(round(float(v))) for v in tuple(color)[:3]), calib._kind_paints(kind, color))


def _forecast(brush, paint, lies_rgb, fills, load, pressure, thinner, even, px, dl, seeds=WithAndDY, layout='свой'):
    from .eye import oklab
    if layout == 'два':
        base = fill_two(fills[0], fills[1], px, dl)
    else:
        load_fills, how_z = fills
        base = fill(lies_rgb, load_fills, px, dl=dl, how=how_z)
    w0 = ROWS_MM[0] * px
    Ll, lab_l = _lab(base, _window_sweep(px, max(float(brush.width), 1.0), dl))
    lab_brush = oklab(np.asarray(tuple(paint)[:3], np.float64))
    with_ = base.snapshot()
    Ls, labs, shares, responses = [], [], [], []
    for seed in seeds:
        z = measurement_sweep(base, brush, paint, load, pressure, seed, thinner, even, px, dl, layout)
        Ls.append(z['L'])
        labs.append(z['lab'])
        shares.append(share_tone(z['lab'], lab_brush, lab_l))
        responses.append(z['ответ'])
        base.roll_back(with_)
    result = _mean_color(Ls, labs)
    result.update(share=round(float(np.mean(shares)), 3), spread_L=round(float(np.std(Ls)), 2),
                spread_shares=round(float(np.std(shares)), 3), seed=len(seeds))
    result['ответ'] = _response_mean(responses)
    if layout == 'два':
        sl = _wet_layers(fill_two(fills[0], fills[1], px, dl), _window_sweep(px, w0, dl))
        result['лежит'] = {'rgb': tuple(int(v) for v in lies_rgb), 'L': round(Ll, 2), 'длина_мм': float(dl),
                         'толщина': round(sl[LAYER_BOTTOM][0] + sl[LAYER_LIES][0], 4),
                         'два': [{'слой': and_, 'rgb': tuple(int(v) for v in z[0]), 'нагрузка_заливки': round(float(z[1]), 3),
                                  'заливка': z[2], 'толщина': round(sl[and_][0], 4)}
                                 for and_, z in ((LAYER_BOTTOM, fills[0]), (LAYER_LIES, fills[1]))]}
        return result
    result['лежит'] = {'rgb': tuple(int(v) for v in lies_rgb), 'L': round(Ll, 2), 'нагрузка_заливки': round(load_fills, 3),
                     'толщина': round(wet_thickness(fill(lies_rgb, load_fills, px, dl=dl, how=how_z),
                                                    _window_sweep(px, w0, dl)), 4),
                     'длина_мм': float(dl)}
    if how_z != 'ровно':
        result['лежит']['заливка'] = how_z
    return result


def _load(load):
    from .tool import NAGRAtZBrushAnd as AndMENA
    if isinstance(load, str):
        if load not in AndMENA or load == 'сухая':
            raise ValueError(f'смешение_для: нагрузка {load!r} — числом или {", ".join(brush for brush in AndMENA if brush != "сухая")}')
        return float(AndMENA[load])
    n = float(load)
    if not (NAGRAtZBrushAnd[0] - 1e-9 <= n <= NAGRAtZBrushAnd[1] + 1e-9):
        nearest = min(max(n, NAGRAtZBrushAnd[0]), NAGRAtZBrushAnd[1])
        raise ValueError(f'смешение_для: нагрузка {n:g} вне стенда — {NAGRAtZBrushAnd[0]:g}…{NAGRAtZBrushAnd[1]:g}; ближайшее — {nearest:g}')
    return n


def _unevenness(canvas, region, layer):
    """В-4-3: σ/среднее сырой толщины слоя в области — борозды рядов (0 — ровный)."""
    where = canvas._region(region)
    with_ = canvas.layers[layer]
    h = np.asarray(with_.thickness[where], np.float64) * np.asarray(with_.moisture[where], np.float64)
    return float(h.std() / h.mean()) if h.size and h.mean() > 1e-9 else 0.0


def _wet_layers(canvas, region):
    """Сырая толщина места по слоям: имя -> (сырая, вся толщина) — как `сырая_толщина`, слой за слоем."""
    from .numbers import _WithCanvasInAT
    where = canvas._region(region)
    result = {}
    for and_, with_ in canvas.layers.items():
        t = np.asarray(with_.thickness[where], np.float64)
        h = t * np.asarray(with_.moisture[where], np.float64) * (1.0 - _WithCanvasInAT * np.asarray(with_.degree_of_set[where], np.float64))
        result[and_] = (float(h.mean()) if h.size else 0.0, float(t.mean()) if t.size else 0.0)
    return result


def _colors_layers(canvas, place, layer):
    """В-4-3: (sRGB краски слоя `слой` в месте — среднее его цвета по покрытию, sRGB того, что под ним — грунт и слои
    ниже, как их видит глаз без света), 0–255."""
    from .paint import from_linear
    where = canvas._region(place)
    names = list(canvas.layers)
    brush = np.empty(np.asarray(canvas.layers[names[0]].coverage[where]).shape + (3,), np.float64)
    brush[:] = canvas.ground
    for and_ in names[:names.index(layer)]:
        with_ = canvas.layers[and_]
        a = np.asarray(with_.coverage[where], np.float64)[..., None]
        if with_.transmittance is not None:
            brush *= np.asarray(with_.transmittance[where], np.float64)
        brush += (np.asarray(with_.color[where], np.float64) - brush) * a
    with_ = canvas.layers[layer]
    a = np.asarray(with_.coverage[where], np.float64)[..., None]
    own = (np.asarray(with_.color[where], np.float64) * a).reshape(-1, 3).sum(0) / max(float(a.sum()), 1e-9)
    in8 = lambda lin: tuple(int(round(v)) for v in np.clip(from_linear(lin) * 255.0, 0, 255))
    return in8(own), in8(brush.reshape(-1, 3).mean(0))


def _layout_canvas(canvas, place, layer, mix_below):
    """Слои места против слоя хода — как их увидит `х.мазок(…, слой=слой, мешать_ниже=…)`: ход мешается с сырой краской
    своего слоя, с `мешать_ниже` — и слоёв под ним. Ответ: (раскладка стенда, сырая толщина, с которой мешается
    (или — для 'поверх' — сырая под ходом), {'ход', 'мешается', 'не_мешается', 'выше'}, слово или None)."""
    by = _wet_layers(canvas, place)
    names = list(by)
    if layer in names:
        below, above = names[:names.index(layer)], names[names.index(layer) + 1:]
    else:
        below, above = names, []                     # слоя нет — мазок заведёт его наверху
    own = by.get(layer, (0.0, 0.0))[0]
    under = sum(by[and_][0] for and_ in below)
    layers = {'ход': layer, 'мешать_ниже': bool(mix_below),
            'мешается': [and_ for and_ in ([layer] if layer in by else []) + (below if mix_below else []) if by[and_][0] > WithYRResponse],
            'не_мешается': [and_ for and_ in ([] if mix_below else below) if by[and_][0] > WithYRResponse],
            'выше': [and_ for and_ in above if by[and_][1] > WithYRResponse]}
    words = []
    if layers['выше']:
        words.append(f'над слоем {layer!r} в месте лежит краска слоёв {", ".join(map(repr, layers["выше"]))} — на холсте '
                     'след будет под ней (прогноз — цвет самого следа, без неё)')
    mixes = own + (under if mix_below else 0.0)
    if mix_below and own > WithYRResponse and under > WithYRResponse:     # В-4-3: стенд в два слоя — лежащее своего над лежащим нижних
        if len(layers['мешается']) > 2:
            words.append(f'сырое в {len(layers["мешается"])} слоях — стенд кладёт два: свой и всё под ним одной заливкой '
                         '(цвет — как видно под слоем хода)')
        return 'два', (own, under), layers, ('; '.join(words) or None)
    if mixes > WithYRResponse:
        layout = 'свой' if own >= (under if mix_below else 0.0) else 'ниже'
        return layout, mixes, layers, ('; '.join(words) or None)
    if under > WithYRResponse:
        words.insert(0, f'ход ляжет поверх: в слое {layer!r} сырого нет, а сырое под ним (слои '
                        f'{", ".join(map(repr, layers["не_мешается"]))}) ход не берёт — нужно мешать_ниже=True '
                        'или слой лежащего; доля — просвет и непокрытое, не смешение')
        return 'поверх', under, layers, '; '.join(words)
    raise ValueError(f'смешение_для: в месте {tuple(place)} сырой краски нет (толщина × влажность ≈ 0) — по сухому '
                     'смешения нет: след ляжет поверх (`нагрузка_для`, `о.цвет`)')


def _lies(lies, canvas, thickness, underpainting, px, layer=None, mix_below=False, lies_in=None):
    """(rgb лежащего, (нагрузка заливки стенда, как), раскладка, слои, слово)."""
    if (thickness is not None) and (underpainting is not None):
        raise ValueError('смешение_для: толщина лежащего — одно из двух: толщина= (сырая, как `Слой.толщина`) или '
                         'подмалёвок= (нагрузка заливки)')
    try:
        t = tuple(float(v) for v in tuple(lies))
    except (TypeError, ValueError):
        t = ()
    layer_canvas = 'основа' if layer is None else str(layer)
    if len(t) == 4:                                  # место холста модели
        if canvas is None:
            raise ValueError('смешение_для: лежит=(x0, y0, x1, y1) — место холста: дай и холст=х')
        if lies_in is not None:
            raise ValueError('смешение_для: лежит_в= — только без холста; с холст= слои места берутся с холста')
        rgb = tuple(canvas.mean(t, light=False)['rgb'])
        layout, wet, layers, word = _layout_canvas(canvas, t, layer_canvas, mix_below)
        if layout == 'два':
            if thickness is not None or underpainting is not None:
                raise ValueError('смешение_для: сырое места в двух слоях (слой хода и под ним, мешать_ниже) — толщины '
                                 'берутся с холста слой за слоем; толщина= и подмалёвок= не давай')
            bottom = _bottom_test_bench(px)
            if min(wet) >= bottom:
                own_rgb, under_rgb = _colors_layers(canvas, t, layer_canvas)
                names = list(canvas.layers)
                under_n = [and_ for and_ in names[:names.index(layer_canvas)] if and_ in layers['мешается']]
                uneven_under = any(_unevenness(canvas, t, and_) > NERResponseInNResponse for and_ in under_n)
                uneven_own = _unevenness(canvas, t, layer_canvas) > NERResponseInNResponse
                if uneven_under or uneven_own:
                    word = '; '.join([with_ for with_ in (word,) if with_] + [
                        f'прогноз мимо до 10 L: сырое в двух слоях бороздами (σ/среднее толщины > {NERResponseInNResponse:g}) — '
                        'стенд кладёт оба своими рядами поперёк хода, борозды места не повторяет (на пробах — 1–10 L, '
                        'больше у толстого слоя хода бороздами вдоль хода); по ровному сырому — в 2 L; точнее — х.проба'])
                layers['раскладка'] = layout
                return rgb, ((under_rgb,) + _load_fills(wet[1], px, uneven_under),
                             (own_rgb,) + _load_fills(wet[0], px, uneven_own)), layout, layers, word
            few = layer_canvas if wet[0] < bottom else 'под ним'
            layout = 'свой' if wet[0] >= wet[1] else 'ниже'
            word = '; '.join([with_ for with_ in (word,) if with_] + [
                f'сырое в двух слоях, но {"в слое " + repr(few) if few == layer_canvas else few} его мало '
                f'({min(wet):.3g} < {bottom:.3g} — тоньше заливки стенда): стенд кладёт всё ({sum(wet):.3g}) одной '
                'заливкой цвета места'])
            thickness = sum(wet)
        elif thickness is None and underpainting is None:
            thickness = wet
    elif len(t) == 3:
        from . import calibration as calib
        calib._color(t)
        rgb = tuple(int(round(v)) for v in t)
        lv = layer_canvas if lies_in is None else str(lies_in)
        if lv == layer_canvas:
            layout, word = 'свой', None
        elif mix_below:
            layout, word = 'ниже', None
        else:
            layout = 'поверх'
            word = (f'ход ляжет поверх: лежащее в слое {lv!r}, ход — в {layer_canvas!r} без мешать_ниже — с сырым под собой '
                     'не мешается; доля — просвет и непокрытое, не смешение')
        layers = {'ход': layer_canvas, 'мешать_ниже': bool(mix_below), 'мешается': [] if layout == 'поверх' else [lv],
                'не_мешается': [lv] if layout == 'поверх' else [], 'выше': []}
    else:
        raise ValueError(f'смешение_для: лежит={lies!r} — цвет сырой краски (r, g, b) или место холста (x0, y0, x1, y1) '
                         'с холст=')
    layers['раскладка'] = layout
    if underpainting is not None:
        n = float(underpainting)
        if not (LOADS_FILLS[0] - 1e-9 <= n <= LOADS_FILLS[-1] + 1e-9):
            nearest = min(max(n, LOADS_FILLS[0]), LOADS_FILLS[-1])
            raise ValueError(f'смешение_для: подмалёвок {n:g} вне стенда — нагрузка заливки {LOADS_FILLS[0]:g}…'
                             f'{LOADS_FILLS[-1]:g}; ближайшее — {nearest:g}')
        return rgb, (n, 'ровно'), layout, layers, word
    if thickness is None:
        raise ValueError('смешение_для: сколько лежит — толщина= (сырая толщина, как `Слой.толщина`; с места холста — '
                         'сама) или подмалёвок= (нагрузка заливки рядами `плоская` 5 мм, `ровно=1`)')
    return rgb, _load_fills(float(thickness), px), layout, layers, word


def _refine(forecast, target, n, a, b, nzh, steps=4, exactly=0.02):
    """Корень перебора (сиды `СИДЫ_ПОИСКА`) — всеми сидами `СИДЫ`: ложное положение внутри отрезка сетки [a, b] по log
    нагрузки; в крутом месте 8 сидов и 24 расходятся на 0,05–0,1 доли. Ответ — прогноз в лучшей точке."""
    def in_(x):
        p = forecast(round(float(x), 4))
        p.update(load=round(float(x), 4), pressure=nzh)
        return p
    best = in_(n)
    if abs(best['доля'] - target) <= exactly or a == b:
        return best
    la, lb = math.log(a), math.log(b)
    fa, fb = in_(a)['доля'] - target, in_(b)['доля'] - target
    for _ in range(steps):
        if fa * fb > 0:
            break
        x = la + (lb - la) * fa / (fa - fb) if fa != fb else (la + lb) / 2
        p = in_(math.exp(x))
        f = p['доля'] - target
        if abs(f) < abs(best['доля'] - target):
            best = p
        if abs(f) <= exactly:
            break
        if f * fa < 0:
            lb, fb = x, f
        else:
            la, fa = x, f
    return best


def mixing_for(brush, lies, color=None, share=None, load=None, pressure=None, kind=None, thinner=None, even=None,
                 thickness=None, underpainting=None, length=None, canvas=None, px_on_mm=None, layer=None, mix_below=False,
                 lies_in=None):
    """В-4: один ход по лежащей сырой краске числом — до кладки.

    лежит — сырая краска под ходом: цвет (r, g, b) с толщиной (`толщина=` — сырая, единицы `Слой.толщина`, или
    `подмалёвок=` — нагрузка заливки стенда 0,1–1) или место холста (x0, y0, x1, y1) с `холст=х`: цвет — `х.средний`
    места без света, толщина — сырая краска места (сумма слоёв толщина × влажность × (1 − 0,85 × схват)), если не дана.
    цвет — краска хода (sRGB, `Краска` — с неё и вид, `цвет(L, C, h)`), обязателен. Кисть — любая, как есть (копией). длина — ход, мм.

    С нагрузкой — прогноз: {'доля' — доля тона лежащего в теле следа (0 — краска хода, 1 — лежащее; OKLab),
    'L', 'C', 'h', 'rgb', 'hex' — цвет тела следа (как `х.средний` средней половины хода и ширины), 'разброс_L',
    'разброс_доли' — по сиду (один ход гуляет на столько), 'ответ' — среднее `о.цвет` и `о.сырое` хода, 'лежит' —
    цвет, L и сырая толщина подмалёвка стенда, 'нагрузка', 'нажим'}. С заказом `доля=` — нагрузка под неё (нажим дан —
    при нём; нет — сперва 0,6, затем ближние 0,5, 0,65, 0,35, 0,8) и прогноз в этой точке; из нескольких нагрузок —
    наибольшая (кроет плотнее). Недостижимое и вне стенда — `ValueError` с ближайшим.

    Слой (В-4-2) — как у `х.мазок`: `слой=` — слой хода (умолчание 'основа', как у мазка), `мешать_ниже=` — мешать и с
    сырым слоёв под ним. Ход берёт сырое только своего слоя (с `мешать_ниже` — и нижних). С `холст=` слои места
    читаются с холста: толщина — сырое тех слоёв, с которыми ход смешается; стенд кладёт заливку в свой слой, ход —
    в тот же слой ('свой'), в слой над ней с `мешать_ниже` ('ниже') или без смешения ('поверх': сырое только под ходом,
    а он его не берёт — прогноз цвета есть, но 'слово' и предупреждение: «ход ляжет поверх»). Без холста лежащее —
    в слое хода, иначе `лежит_в=` — имя его слоя (под слоем хода). В ответе 'слои': {'ход', 'мешать_ниже', 'мешается',
    'не_мешается', 'выше', 'раскладка'}; 'слово' — когда прогноз не о смешении или над ходом лежит чужой слой.

    Счёт — ход на своём стенде (`кисть/смешение.py`): гладкий грунт (166, 146, 129), подмалёвок — заливка рядами одной
    кроющей краской (сырая толщина 0,07–0,75 — ряды `ровно=1`, до 1,05 — без `ровно`, толще — два прохода), ход прямо поперёк рядов длиной
    `длина=` (мм, умолчание 45), 24 сида. Свой путь, зерно, прозрачный или многослойный подмалёвок,
    лежащее пятнами — ближе к `х.проба`. Только счёт: холст модели, кисть и журнал не меняются."""
    px = float(canvas.px_on_mm) if (canvas is not None and canvas.px_on_mm) else float(px_on_mm or PCanvas)
    paint = _paint(color, kind)
    dl = DLAndNA if length is None else float(length)
    if not (DLAndNY[0] <= dl <= DLAndNY[1]):
        nearest = min(max(dl, DLAndNY[0]), DLAndNY[1])
        raise ValueError(f'смешение_для: длина хода {dl:g} мм — стенд {DLAndNY[0]:g}…{DLAndNY[1]:g} мм; ближайшее — {nearest:g}')
    lies_rgb, nz, rk, layers, word = _lies(lies, canvas, thickness, underpainting, px, layer, mix_below, lies_in)

    def ready(p):
        p['слои'] = dict(layers)
        if word:
            p['слово'] = word
            warnings.warn('смешение_для: ' + word, stacklevel=3)
        return p
    if thinner is not None and not (0.0 <= float(thinner) <= 1.0):
        raise ValueError(f'смешение_для: разбавитель {float(thinner):g} — от 0 до 1')
    if pressure is not None and not (0.05 <= float(pressure) <= 1.0):
        nearest = min(max(float(pressure), 0.05), 1.0)
        raise ValueError(f'смешение_для: нажим {float(pressure):g} — от 0,05 до 1; ближайшее — {nearest:g}')
    if share is None:
        if load is None:
            raise ValueError('смешение_для: дай нагрузка= (прогноз) или доля= (заказ)')
        n = _load(load)
        nzh = NAZhAndM if pressure is None else float(pressure)
        p = _forecast(brush, paint, lies_rgb, nz, n, nzh, thinner, even, px, dl, layout=rk)
        p.update(load=n, pressure=nzh)
        return ready(p)
    if load is not None:
        raise ValueError('смешение_для: с доля= нагрузку не давай — она и подбирается (нажим — можно)')
    target = float(share)
    if not (0.0 < target < 1.0):
        raise ValueError(f'смешение_для: доля {target:g} — от 0 до 1 (доля тона лежащего в следе)')
    pressures = (float(pressure),) if pressure is not None else (NAZhAndM,) + tuple(sorted(NAZhAndMY, key=lambda v: abs(v - NAZhAndM)))
    grid = GRID_LOADS
    min, max, where = 2.0, -1.0, None
    for nzh in pressures:
        v = np.array([_forecast(brush, paint, lies_rgb, nz, n, nzh, thinner, even, px, dl, SEEDS_SEARCH, rk)['доля']
                      for n in grid]) - target
        if np.all(np.isfinite(v)):
            if min > v.min() + target:
                min = float(v.min() + target)
            if max < v.max() + target:
                max = float(v.max() + target)
        roots = []                                   # (нагрузка по сетке, отрезок сетки)
        for i in range(len(grid) - 1):
            if v[i] == 0 or v[i] * v[i + 1] < 0:
                a, b = math.log(grid[i]), math.log(grid[i + 1])
                t = v[i] / (v[i] - v[i + 1]) if v[i] != v[i + 1] else 0.0
                roots.append((math.exp(a + (b - a) * t), (grid[i], grid[i + 1])))
        if v[-1] == 0:
            roots.append((grid[-1], (grid[-1], grid[-1])))
        if roots:
            n, (a, b) = roots[-1]
            return ready(_refine(lambda x: _forecast(brush, paint, lies_rgb, nz, x, nzh, thinner, even, px, dl,
                                                       layout=rk), target, float(n), a, b, nzh))
    if max < 0:
        raise ValueError('смешение_для: у краски хода и лежащего один цвет — доли тона нет')
    nearest = min if target < min else max
    than = f'нажим {pressures[0]:g}' if len(pressures) == 1 else f'нажимы {", ".join(f"{v:g}" for v in pressures)}'
    raise ValueError(f'смешение_для: доля {target:g} недостижима ({than}, нагрузка {grid[0]:g}…{grid[-1]:g}): ход берёт '
                     f'{min:.2f}…{max:.2f} тона лежащего; ближайшее — {nearest:.2f}' +
                     ('; меньше лежащего — гуще краска, легче рука, подсушить лежащее (`х.подсушить`)' if target < min
                      else '; больше лежащего — тоньше краска и сильнее нажим, мягкая кисть или растушёвка вытертой'))
