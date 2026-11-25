# READING VIEW of движок/кисть/край.py with names in English. Not runnable; edit the original.
"""К-5 (К-17 баланса): мягкость края числом — `край_для`: какой край (`о.край`: кайма в мм и px, рваность, слово)
даст один ход до кладки, и что взять под заказ каймы — нагрузку, нажим и, если нужно, поле `кайма` кисти.

Почему стенд, а не таблица (замер Движка 191, 960 точек × 3 сида). Кайма идёт не за одним числом: у `мягкая` 32 по
сухому она растёт от тяжёлой руки к лёгкой (0,12 → 0,35 мм), по сырому — наоборот и не монотонно: нагрузка 0,05–0,1
кладёт 0,06 мм, 0,2–0,3 — 0,42 мм; сырое толщиной 0,1 гасит кайму совсем, 0,3–0,4 даёт самую широкую, 0,7 — снова
уже. Поле `кайма` у щетины расширяет край (плоская 40: 0,25 → 0,62 мм), круглая 22 по сырому больше 0,23 мм не даёт.
Кисть модели — любая, с любыми полями; таблица на каждую не снимется. Поэтому счёт — сам ход на своём стенде в памяти:
ход 0,06–0,15 с; прогноз — 24 сида (48, где один ход гуляет на 0,08 мм и больше), 2–7 с; заказ — перебор сетки по
6 сидов и сверка лучших всеми, по сухому 20–110 с, по сырому 70–150 с.

Стенд (в мм — тот же при любом `px_на_мм`; при 8 px/мм — 600×200 px): чистый грунт (166, 146, 129), гладкий; ход
`путь_с_нажимом` прямо, 50 мм (`длина=` 10–200), от 12,5 мм с края, посередине высоты, краской (40, 40, 60), в слой
'основа', кистью модели (её копией), сиды `СИДЫ` 101–124. По сырому — под ходом в том же слое заливка: `плоская`
5 мм рядами вдоль хода, шаг 3,25 мм, `ровно=1`, нажим 0,6, светлой (220, 200, 160), сиды 50 + номер ряда; нагрузка
заливки — 0,6 (`по_сырому=True`) или по кривой сырой толщины (`по_сырому=` числом, или место холста).

Мерка — `о.край` хода (`ответ.py`: кайма — px от 10 до 50 % уровня тела поперёк хода, медиана двух сторон). Слово —
большинство по сидам. Только счёт: холст, кисть и журнал модели не трогаются.
"""
import math
import warnings

import numpy as np

GRAtNT = (166, 146, 129)
PCanvas = 8.0
DLAndNA = 50.0                                # ход стенда, мм
DLAndNY = (10.0, 200.0)
FIELD_MM = 12.5                              # от края стенда до хода, мм
HEIGHT_MM = 25.0
BrushRAWithBrushA = (40, 40, 60)
ZALAndInBrushA = (220, 200, 160)
ROWS_MM = (5.0, 3.25, 5.0, 21.25)           # ширина `плоская` заливки, шаг рядов, от края стенда по x, по высоте —
                                            # до последнего ряда (ряды от 5 мм до 21,25 мм)
LOAD_FILLS = 0.6                      # `по_сырому=True`
LOADS_FILLS = (0.15, 0.3, 0.45, 0.6, 0.8, 1.0)   # кривая сырой толщины (под `по_сырому=` числом)
WithAndDY = tuple(range(101, 125))               # прогноз
SEEDS_SEARCH = WithAndDY[:6]                      # перебор под заказ
SEEDS_WIDER = tuple(range(101, 149))          # прогноз, где один ход гуляет на ШУМ и больше: 48 сидов
ShAtM = 0.08                                  # мм σ по сиду — с него в ответе 'шум' (по сырому — 0,1–0,2 мм)
GRID_LOADS = (0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.65, 0.85)
GRID_PRESSURE = (0.35, 0.5, 0.65, 0.85)
FIELDS_FRINGE = (0.15, 0.3, 0.5, 0.7)          # поле `кайма` кисти, если её как есть не хватает (по возрастанию)
DResponsePAtWithBrush = (0.04, 0.12)                       # своя цель перебора: в max(0,04 мм, 12 %) от заказа (приёмка — 0,08 / 25 %)
WEIGHT_SPREAD = 1.0                          # точка перебора судится |μ − заказ| + σ: ровная лучше точной (по сырому
                                            # один ход гуляет на 0,1–0,2 мм)
WithInERResponseBrush = 4                                  # лучших точек перебора, что сверяются всеми `СИДЫ`
WITHOUT_PRICE = (0.85, 10.0)                     # Долг-т: закрыто не ниже и тон следа от цвета краски не дальше (L) — край «без цены»
WithYRResponse = 1e-3                                 # сырая толщина, ниже которой ход идёт по сухому
_ZALAndInBrushAnd = {}                               # (нагрузка, px, длина) -> Холст стенда (память процесса)
_BrushRAndInYE = {}                                # (px, длина) -> [(нагрузка заливки, толщина)]


def _px(v, px):
    return float(v) * px


def _canvas(px, dl):
    from .canvas import Canvas
    return Canvas(int(round((float(dl) + 2 * FIELD_MM) * px)), int(round(HEIGHT_MM * px)), ground=GRAtNT, px_on_mm=px,
                 cores=1, hints=False)


def _window(px, dl):
    """Окно хода стенда (x0, y0, x1, y1), px: средняя половина длины, ±1 мм поперёк — там меряется сырая толщина."""
    a, b = _px(FIELD_MM, px), _px(FIELD_MM + dl, px)
    yc = _px(HEIGHT_MM / 2, px)
    return (int(a + (b - a) / 4), int(yc - px), int(b - (b - a) / 4), int(yc + px) + 1)


def fill(load, px=PCanvas, dl=DLAndNA):
    """Холст стенда: None — сухой грунт; число — сырая заливка этой нагрузкой (ряды вдоль хода). Копией из памяти."""
    import copy
    if load is None:
        return _canvas(px, dl)
    option = (round(float(load), 4), float(px), round(float(dl), 2))
    if option not in _ZALAndInBrushAnd:
        from .brush import Brush
        from .path import path_with_pressure
        from . import calibration as calib
        import os
        canvas = _canvas(px, dl)
        sh, spacing, from_, to = (_px(v, px) for v in ROWS_MM)
        brush = Brush.from_file(os.path.join(calib._PREWithETY, 'плоская.json'), width=sh)
        x1 = canvas.width - from_
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y, i = from_, 0
            while y <= to + 1e-6:
                t = path_with_pressure([(from_, y), (x1, y)], sh, pressure=0.6, seed=50 + i)
                canvas.stroke(t, ZALAndInBrushA, brush, seed=50 + i, load=float(load), even=1, layer='основа')
                y, i = y + spacing, i + 1
        if len(_ZALAndInBrushAnd) >= 16:
            _ZALAndInBrushAnd.pop(next(iter(_ZALAndInBrushAnd)))
        _ZALAndInBrushAnd[option] = canvas
    return copy.deepcopy(_ZALAndInBrushAnd[option])


def thickness_test_bench(load, px=PCanvas, dl=DLAndNA):
    """Сырая толщина заливки стенда под ходом (единицы `Слой.толщина`, как `смешение.сырая_толщина`)."""
    from .mixing import wet_thickness
    return wet_thickness(fill(load, px, dl), _window(px, dl))


def _curve(px, dl):
    option = (float(px), round(float(dl), 2))
    if option not in _BrushRAndInYE:
        _BrushRAndInYE[option] = [(n, thickness_test_bench(n, px, dl)) for n in LOADS_FILLS]
    return _BrushRAndInYE[option]


def _load_fills(thickness, px, dl):
    brush = _curve(px, dl)
    t = [v for _, v in brush]
    if not (t[0] - 1e-9 <= thickness <= t[-1] + 1e-9):
        nearest = min(max(thickness, t[0]), t[-1])
        raise ValueError(f'край_для: сырое под ходом {thickness:.3f} — стенд кладёт заливку {t[0]:.3f}…{t[-1]:.3f}; '
                         f'ближайшее — {nearest:.3f}' + ('; тоньше — ход почти по сухому: по_сырому=None'
                                                      if thickness < t[0] else ''))
    return float(np.interp(thickness, t, [n for n, _ in brush]))


def _wet(by_wet, canvas, place, layer, px, dl):
    """(нагрузка заливки стенда или None, сырая толщина под ходом, откуда)."""
    if place is not None:
        if canvas is None:
            raise ValueError('край_для: место= читается с холст=х')
        from .mixing import _wet_layers
        sl = _wet_layers(canvas, tuple(int(round(v)) for v in place))
        and_ = layer or 'основа'
        t = sl.get(and_, (0.0, 0.0))[0]
        if by_wet is not None and by_wet is not True:
            raise ValueError('край_для: место= или по_сырому= числом — одно из двух')
        if t < WithYRResponse:
            return None, round(t, 4), f'место, слой {and_!r}: сухо'
        return _load_fills(t, px, dl), round(t, 4), f'место, слой {and_!r}'
    if by_wet is None or by_wet is False:
        return None, 0.0, 'сухо'
    if by_wet is True:
        return LOAD_FILLS, round(thickness_test_bench(LOAD_FILLS, px, dl), 4), 'стенд: заливка 0,6'
    t = float(by_wet)
    if t < WithYRResponse:
        return None, 0.0, 'сухо'
    return _load_fills(t, px, dl), round(t, 4), 'толщина'


def measurement_sweep(canvas, brush, load, pressure, seed, px=PCanvas, dl=DLAndNA, thinner=None, paint=BrushRAWithBrushA):
    """Один ход стенда по холсту `х` (кладёт на `х`!) — ответ мазка. Им же меряется живое в тесте пункта."""
    from .path import path_with_pressure
    sh = float(brush.width)
    yc = _px(HEIGHT_MM / 2, px)
    t = path_with_pressure([(_px(FIELD_MM, px), yc), (_px(FIELD_MM + dl, px), yc)], sh, pressure=float(pressure), seed=seed)
    d = {'разбавитель': float(thinner)} if thinner else {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return canvas.stroke(t, paint, brush.copy(), seed=seed, load=load, layer='основа', **d)


def _forecast(brush, nz, load, pressure, px, dl, thinner, paint, seeds=WithAndDY):
    base = fill(nz, px, dl)
    with_ = base.snapshot()
    brush, r, words, closed, tone = [], [], [], [], []
    for seed in seeds:
        response = measurement_sweep(base, brush, load, pressure, seed, px, dl, thinner, paint)
        base.roll_back(with_)
        if response is None or response.edge is None:
            brush.append(0.0), r.append(0.0), words.append('нет следа')
            continue
        closed.append(float(response.covered)), tone.append(float(response.color['L']))
        brush.append(float(response.edge['кайма'])), r.append(float(response.edge['рваность'])), words.append(response.edge['слово'])
    brush = np.asarray(brush) / px
    if seeds == WithAndDY and float(brush.std()) >= ShAtM:                # гуляет — точнее среднее: 48 сидов
        return _forecast(brush, nz, load, pressure, px, dl, thinner, paint, SEEDS_WIDER)
    word = max(set(words), key=lambda w: (words.count(w), w))
    noise = ({'шум': f'один ход гуляет на ±{float(brush.std()):.2f} мм: в кайму {float(brush.mean()):.2f} попадает среднее ходов, '
                   f'а один ход — в ±{2 * float(brush.std()):.2f}; ровнее — по сухому или другой нагрузкой'}
           if float(brush.std()) >= ShAtM else {})
    price = _price(closed, tone, float(brush.mean()), paint, nz is not None) if tone else {}
    return {**noise, **price, 'кайма_мм': round(float(brush.mean()), 3), 'кайма_px': round(float(brush.mean() * px), 2),
            'разброс_мм': round(float(brush.std()), 3), 'рваность_px': round(float(np.mean(r)), 2), 'слово': word,
            'слова': {w: words.count(w) for w in sorted(set(words))}, 'сидов': len(seeds)}


def _price(covered, tone, edge_band, paint, by_wet):
    """Долг-т: чем заплачено за край — закрыто и тон тела следа (L серой шкалы, как `о.цвет`) против цвета краски, и
    словом. Бледнее — тон ушёл от краски к лежащему под ходом (грунту стенда, по сырому — заливке), темнее он стал или
    светлее — по обеим L."""
    from .eye import lightness
    z, L = float(np.mean(covered)), float(np.mean(tone))
    Lk = float(lightness(np.asarray(paint, np.float64)))
    Ln = float(lightness(np.asarray(ZALAndInBrushA if by_wet else GRAtNT, np.float64)))
    d = (L - Lk) * (1.0 if Ln >= Lk else -1.0)                # > 0 — ушёл к лежащему
    tone_ = (f'ход бледнее заказанного цвета на {abs(d):.0f} L: след L {L:.0f}, краска L {Lk:.0f}' if d >= 0.5 else
            f'тон следа L {L:.0f}, краски L {Lk:.0f}')
    expensive = z < WITHOUT_PRICE[0] or d > WITHOUT_PRICE[1]
    word = f'кайма {edge_band:.2f} мм — ' + (f'ценой: закрыто {z:.2f}, {tone_}' if expensive else f'без цены: закрыто {z:.2f}, {tone_}')
    return {'закрыто': round(z, 3), 'тон': round(L, 1), 'бледнее': round(d, 1), 'цена': word.replace('.', ',')}


def _load(load):
    from .tool import NAGRAtZBrushAnd as AndMENA
    if isinstance(load, str):
        if load not in AndMENA or load == 'сухая':
            raise ValueError(f'край_для: нагрузка {load!r} — числом или {", ".join(brush for brush in AndMENA if brush != "сухая")}')
        return float(AndMENA[load])
    n = float(load)
    if not (0.03 <= n <= 1.0):
        raise ValueError(f'край_для: нагрузка {n:g} — от 0,03 до 1; ближайшее — {min(max(n, 0.03), 1.0):g}')
    return n


def _tolerance(target):
    return max(DResponsePAtWithBrush[0], DResponsePAtWithBrush[1] * target)


def edge_for(brush, edge_band_mm=None, load=None, pressure=None, by_wet=None, canvas=None, place=None, layer=None,
             field=True, length=None, px_on_mm=None, thinner=None, color=None):
    """К-5: край одного хода числом — до кладки; ничего не кладёт.

    С `нагрузка=` (и `нажим=`, умолчание 0,6) — прогноз `о.край`: {'кайма_мм', 'кайма_px', 'разброс_мм' (σ по сиду),
    'рваность_px', 'слово' ('мягкий' / 'жёсткий' / 'рваный' — большинство сидов), 'слова', 'сидов' (24; где один ход
    гуляет на 0,08 мм и больше — 48), 'шум' (только тогда: словом, на сколько гуляет один ход), 'нагрузка', 'нажим',
    'кайма_кисти' (поле кисти), 'сырое': {'толщина', 'откуда'}}; и цена края (Долг-т): 'закрыто' (`о.закрыто`), 'тон'
    (L тела следа), 'бледнее' (на сколько L тело следа ушло от краски к лежащему под ходом), 'цена' — словом: «кайма
    0,50 мм — ценой: закрыто 0,29, ход бледнее заказанного цвета на 80 L: …» или «— без цены: …». Цвет — `цвет=` (без
    него краска стенда (40, 40, 60)), под ходом — грунт стенда или его светлая заливка: на своём листе сдвиг тона другой,
    `закрыто` — то же.

    С заказом `кайма_мм=` — то же в точке, где 24 сида дают кайму ближе всего к заказу, и что взять: 'нагрузка',
    'нажим', 'поле' — {} или {'кайма': v} (поле кисти, если её как есть не хватило; `поле=False` — не трогать), 'кисть' —
    копия кисти с этим полем (её и класть). Нажим дан — только при нём. Шире или уже достижимого — `ValueError` с
    ближайшим и тем, чем делать иначе (шире 0,7 мм — растушёвкой: `х.переход`, рецепт «Пятно со спадом»).

    Сырое под ходом: `по_сырому=None` — сухой грунт; `True` — заливка стенда нагрузкой 0,6 (сырая толщина ~0,39);
    число — сырая толщина слоя хода (единицы `Слой.толщина`, как `смешение_для(толщина=)`, стенд 0,10…0,72); или
    `холст=х, место=(x0, y0, x1, y1)` — толщина сырого слоя хода `слой=` (умолчание 'основа') в месте; холст только
    читается. Ход берёт сырое только своего слоя: сырое в других слоях кайму не меняет.

    Стенд — `кисть/край.py`: прямой ход 50 мм (`длина=`), гладкий грунт, сырое — ровные ряды одной краской вдоль хода.
    Свой путь, зерно, борозды места — ближе к `х.проба`."""
    px = float(canvas.px_on_mm) if (canvas is not None and canvas.px_on_mm) else float(px_on_mm or PCanvas)
    dl = DLAndNA if length is None else float(length)
    if not (DLAndNY[0] <= dl <= DLAndNY[1]):
        raise ValueError(f'край_для: длина хода {dl:g} мм — стенд {DLAndNY[0]:g}…{DLAndNY[1]:g} мм; ближайшее — '
                         f'{min(max(dl, DLAndNY[0]), DLAndNY[1]):g}')
    if pressure is not None and not (0.05 <= float(pressure) <= 1.0):
        raise ValueError(f'край_для: нажим {float(pressure):g} — от 0,05 до 1; ближайшее — {min(max(float(pressure), 0.05), 1.0):g}')
    if thinner is not None and not (0.0 <= float(thinner) <= 1.0):
        raise ValueError(f'край_для: разбавитель {float(thinner):g} — от 0 до 1')
    paint = BrushRAWithBrushA if color is None else tuple(int(round(float(v))) for v in tuple(color)[:3])
    nz, thickness, from_where = _wet(by_wet, canvas, place, layer, px, dl)
    wet = {'толщина': thickness, 'откуда': from_where}
    own = float(getattr(brush, 'кайма', 0.0) or 0.0)

    def result(p, brush, n, nzh, field_):
        p.update(load=round(float(n), 4), pressure=float(nzh), edge_band_brushes=round(float(getattr(brush, 'кайма', 0.0) or 0.0), 3),
                 wet=dict(wet), field=dict(field_))
        return p

    if edge_band_mm is None:
        if load is None:
            raise ValueError('край_для: дай нагрузка= (прогноз) или кайма_мм= (заказ)')
        n = _load(load)
        nzh = 0.6 if pressure is None else float(pressure)
        return result(_forecast(brush, nz, n, nzh, px, dl, thinner, paint), brush, n, nzh, {})
    if load is not None:
        raise ValueError('край_для: с кайма_мм= нагрузку не давай — она и подбирается (нажим — можно)')
    target = float(edge_band_mm)
    if not (0.0 < target <= 5.0):
        raise ValueError(f'край_для: кайма {target:g} мм — больше 0 (мм, от 10 до 50 % тела поперёк хода)')
    pressures = (float(pressure),) if pressure is not None else GRID_PRESSURE
    brushes = [(brush, {})]
    if field and getattr(brush, 'форма', '') != 'мастихин':
        brushes += [(brush.with_(edge_band=v), {'кайма': v}) for v in FIELDS_FRINGE if v > own + 1e-6]
    points, checked, checked = [], [], set()       # точки: (оценка, μ, σ, i кисти, нагрузка, нажим)
    for i, (brush, _) in enumerate(brushes):
        for n in GRID_LOADS:
            for nzh in pressures:
                p = _forecast(brush, nz, n, nzh, px, dl, thinner, paint, SEEDS_SEARCH)
                if p['слово'] != 'нет следа':
                    μ, σ = p['кайма_мм'], p['разброс_мм']
                    points.append((abs(μ - target) + WEIGHT_SPREAD * σ, μ, σ, i, n, nzh))
        if not points:
            continue
        if not any(t[3] == i and abs(t[1] - target) <= _tolerance(target) for t in points) and i + 1 < len(brushes):
            continue                                 # эта кисть не достаёт — шире полем
        for ots, μ, σ, j, n, nzh in sorted(points)[:WithInERResponseBrush]:     # лучшие — всеми сидами; кисть, что достала, — без шире
            if (j, n, nzh) in checked:
                continue
            checked.add((j, n, nzh))
            p = _forecast(brushes[j][0], nz, n, nzh, px, dl, thinner, paint)
            checked.append((abs(p['кайма_мм'] - target) + WEIGHT_SPREAD * p['разброс_мм'], p, j, n, nzh))
            if abs(p['кайма_мм'] - target) <= _tolerance(target) and p['разброс_мм'] <= max(0.05, 0.2 * target):
                break
        _, p, j, n, nzh = min(checked, key=lambda t: t[0])
        if abs(p['кайма_мм'] - target) <= max(0.08, 0.25 * target):
            brush, field_ = brushes[j]
            p = result(p, brush, n, nzh, field_)
            p['кисть'] = brush.copy()
            return p
    if not points:
        raise ValueError('край_для: ход не оставляет следа ни при какой нагрузке и нажиме сетки — кисть или краска?')
    _, μ, σ, i, n, nzh = min(points, key=lambda t: abs(t[1] - target))     # ближайшее без веса разброса — тоже 24 сидами
    if (i, n, nzh) not in checked:
        checked.append((0.0, _forecast(brushes[i][0], nz, n, nzh, px, dl, thinner, paint), i, n, nzh))
    _, p, i, n, nzh = min(checked, key=lambda t: abs(t[1]['кайма_мм'] - target))   # ближайшее — проверенное 24 сидами
    if abs(p['кайма_мм'] - target) <= max(0.08, 0.25 * target):   # проверенное в допуске заказа — заказ, не ошибка
        brush, field_ = brushes[i]
        p = result(p, brush, n, nzh, field_)
        p['кисть'] = brush.copy()
        return p
    field_ = brushes[i][1]
    than = (f'нажим {pressures[0]:g}' if len(pressures) == 1 else 'нажим 0,35…0,85') + ', нагрузка 0,05…0,85' + (
        f', поле кайма до {max(own, FIELDS_FRINGE[-1]):g}' if field and len(brushes) > 1 else ', поле кисти как есть')
    if target > p['кайма_мм']:
        advice = ('; шире одним ходом нельзя — растушёвкой поверх: `х.переход` (рецепт «Пятно со спадом»)'
                 + ('' if field or own >= FIELDS_FRINGE[-1] else ', или поле=True — шире полем кисти'))
    else:
        advice = '; уже — тяжелее рука (нагрузка больше), щетиной без каймы, по сухому или подсушенному'
    pk = field_
    nearest = (f'{p["кайма_мм"]:.2f} ± {p["разброс_мм"]:.2f} мм (нагрузка {n:g}, нажим {nzh:g}' + (f', поле кайма {pk["кайма"]:g}' if pk else '')
            + (f'; закрыто {p["закрыто"]:.2f}' if 'закрыто' in p else '') + ')')
    wet = '' if nz is None else f', по сырому {thickness:.2f}'
    raise ValueError(f'край_для: кайма {target:g} мм недостижима ({than}{wet}): ближайшее — {nearest}' + advice)
