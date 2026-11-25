# READING VIEW of движок/кисть/повтор.py with names in English. Not runnable; edit the original.
"""Повтор журнала мазков:

    python -m кисть.повтор журнал.jsonl --out лист.png

Строка журнала — JSON:
    {"layer": "земля", "size": 7.8, "color": [138, 158, 128], "ctrl": [[x, y], ...], "opacity": 1.0}
По ctrl строится путь с нажимом «вход — тело — сход». Необязательные поля строки:
"colors" — несколько красок на кисти: [[r, g, b, доля], ...] или [[[r, g, b], доля], ...] (вместо "color");
"layout" — раскладка красок: "полосы", "ряд", "смесь", "кончик"; "load" — нагрузка: "протирка", "обычная",
"пастозная" или число; "wipe": true — вытереть кисть перед мазком (иначе она несёт подобранное прошлым
мазком той же кистью); "end" — конец мазка: "подъём", "отрыв", "остановка", "смахивание"; "mix_below": true —
мешать и с сырой краской слоёв ниже; "level" — 0–1: кроющая краска текучая — за ворсом смыкается в ровный слой без борозд, а с лежащим встык или внахлёст сырым ходом своего слоя срастается без щели (п. 15, К4: шов двух текучих ходов решается целиком — встык и внахлёст смыкается по всей длине, зазор, оставленный рукой, остаётся зазором; с ходом без "level" не смыкается; кромка у грунта как легла; К7: набор "colors" полосами с "level" — плёнка ровная, полосы красок раздельны, граница 1–3 px); "ridges" — 0–1: кисть с краской лёгкой рукой кладёт на гребни лежащей краски, ложбины пропускает (п. 13); "reserve" — сухой срыв (п. 17): на сколько ширин кисти хватает краски, больше нуля — ход начинается плотно и рвётся островами; "touch" — касание плоской кисти: 1 — плашмя, 0 — на ребре; "seed", "preset" (имя из папки пресетов или путь к .json),
"paint" — вид краски: "кроющая" (так без поля), "полукроющая", "прозрачная", "лессировочная" (мутность 0) или мутность 0–1;
"pressure" (нажим тела 0–1), "dry" (подсушить на столько перед мазком);
"pressure_along" — нажим по ходу: [[доля пути 0–1, множитель ≥ 0], ...], доли строго по возрастанию; нажим точек
пути (после "end"/"pressure") множится на профиль по доле пройденного пути, между узлами по прямой
(`кисть.нажим_вдоль`, тот же счёт, что у анализатора); null — как без поля; множитель 0 — кисть над холстом:
следа и набора нет, после него кисть входит заново (`кисть.участки_касания`, 11ж);
с полем нажим ведёт рука (К8.1): плоская кисть плашмя касается шириной по самому нажиму — слабо поставленная садится
серединой обоймы и раскрывается до всей ширины к нажиму 0,55 (`Холст.мазок(…, нажим_руки=True)`); без поля — как было;
"tool" — снятие сырой краски по ходу (К8.4), строка краски не кладёт: "тряпка" (комок ткани в "size" px: тонкий тон и
полосы нитей и складок вдоль хода) или "скребок" (кромка "size" px поперёк хода: до грунта, валик за концами кромки,
комок на подъёме), или "черенок" (К-14а: круглый твёрдый конец поперечником "size" px процарапывает борозду — сырое
раздвинуто в валики у боков, объём тот же; с нажима 0,8 — до сухого или грунта, слабее — борозда в краске; сида не
берёт); путь и нажим — как у мазка, "layer" — только этот слой; сухое не трогает (`Холст.снять`);
"нож" (М-3, `Холст.прижать`) — кромка "size" px плашмя срезает высохшее и сырое выше плоскости лезвия; "depth" — мм,
глубже стольких под самой высокой точкой под ножом лезвие за ход не садится (без поля — глубину ведёт нажим);
"medium" — 0–1 (Г-1): краска с разбавителем — тоньше, прозрачнее, течёт (борозды смыкаются и у не кроющей), сухой срыв
"reserve" тянется дольше; с "load": "сухая" и 0 — ошибка журнала; без поля — густая, как было;
"fibers" — 0–1 (К6), только с "load": "сухая" (с другой нагрузкой — ошибка журнала): ворс полупустой кисти расчёсан — тело хода плотное, край рвётся волокнами
вдоль хода, просветы длинные по ходу; без поля — как было (клочки);
"load_along" — отдача краски по ходу (К5), только у мягкой кисти: [[доля пути 0–1, множитель 0–1], ...], доли строго по
возрастанию, между узлами по прямой; рука отпускает кисть — след той же ширины и с той же каймой, плёнка тоньше во столько
раз (тон сходит по ходу без смены нажима и ширины; отпущенная кисть краску не тратит и сырое не снимает); с полем, хоть с единицей, плёнка ложится на путь волоска
(на дуге внутренний бок следа не толще наружного), наплыв остановки и ком идут за отдачей — ход по кругу и спиралью тоже; null — как без поля; у щетины
и вместе с отрывом кисти ("pressure_along" с множителем 0) — ошибка с номером строки;
"entry" — длина входа нажима, px журнала; "angle" — у тычка из одной точки направление кисти, градусы от оси x (11ж).
Кисть в руке по ходу (К1) — поля в форме "pressure_along": [[доля пути 0–1, градусы], ...], доли строго по возрастанию,
между узлами по прямой, null — как без поля; единицы — градусы везде:
"rotation_along" — поворот кисти в пальцах от направления хода: 0 — кисть поперёк хода (широкой стороной), 90 — ребром
вдоль хода; угол держится от хода и на дуге; "tilt_along" — наклон ручки, 0–85 (0 — отвесно): пряди тянутся за кистью
дальше (закон ядра, `укладка`, вдоль_к); "azimuth_along" — куда наклонена ручка, от направления хода (0 — пряди тянутся
назад по ходу, как без поля). Перевод в столбцы ядра (x, y, нажим, dt, наклон °, азимут рад, поворот ° от оси x) —
`углы_вдоль`; у тычка из одной точки хода нет — там "angle", поля по ходу не действуют. Хоть одно поле есть — у кисти
поворот держит рука: доворота плоской кисти за ходом с отставанием нет (у "rotation_along" без поля — 0 по всему ходу).
Кисть одна на каждую пару «пресет, размер»: её грязь переходит из мазка в мазок, пока нет "wipe". Строка {"canvas": [ширина, высота], "ground": [r, g, b]} задаёт холст
(если стоит до первого мазка), строка {"dry": 0.5} — подсушить всё, {"set": 0.5} — дать схватиться, {"wet": 1.0} —
размочить сухое (М-3, `Холст.размочить`);
у всех можно "layer" и "area": [x0, y0, x1, y1] (координаты журнала: множится на масштаб). Строка {"tape": имя, "shape": [в, ш],
"mask": …} — лента скотча (Н-3, `Холст.скотч`; маска — zlib(packbits) в base64, у масштаба — ближайшим пикселем; "layer",
"height" мм), {"untape": имя или null} — снять её (`Холст.снять_скотч`). "set" в строке мазка — схватить всё перед мазком.
Сырые точки пути (К2) — "points": [[x, y, нажим, dt, наклон, азимут, поворот], …] вместо "ctrl": путь как есть — ломаная
по самим точкам (угол остаётся углом), нажим и углы между точками по прямой, входа, схода и нажима по умолчанию нет.
Столбцов — от трёх (x, y, нажим 0–1), у всех точек поровну: dt — мс на шаг от прошлой точки в px журнала (нет или
null — 1 мс на px), наклон ручки — градусы 0–85, азимут — куда наклонена ручка, градусы от направления хода (0 — пряди тянутся назад
по ходу, как без столбца), поворот — градусы на холсте от оси x, как "angle" (направление кисти, широкая сторона
поперёк него; между точками — кратчайшей дугой; без столбца плоская кисть доворачивается за ходом сама). Одна точка — тычок; точка, повторённая на месте, — стояние. "ctrl", "end", "entry", "pressure", "angle"
при "points" — ошибка с номером строки; "pressure_along" множит нажим точек, поле К1 заменяет свой столбец (остальные —
как дала рука). В Python то же — `Холст.мазок(точки_журнала(points, масштаб, кисть.отставание · ширина), …)`; свои точки ядра ломаной — `Холст.мазок(точки, …, ломаная=True)`.
Дорисовка (К3): `--save лист.npz` пишет всё, от чего зависит следующий мазок, — холст, кисти (точная ширина и грязь каждой), масштаб, сид, пресет и
номер последней положенной строки; `--from лист.npz [--start N]` берёт это состояние и кладёт строки журнала с номера N
(номер строки файла, как в сообщениях об ошибке; без `--start` — со строки после сохранённой). Журнал в два захода —
те же пиксели и тот же `.npz`, что журнал целиком. В Python — `загрузить_состояние`, `повторить(…, состояние=, начало=)`,
`сохранить_состояние`.
Контрольные точки (1г): без `--from` повтор пишет состояние в папку `<имя --out>.точки/` (`--checkpoints`) перед
пометкой нового этапа, раз в `--every-sec` секунд (300), каждые `--every` мазков и в конце; печатает ход и сколько
осталось. `--resume` — с ближайшей точки, по которую журнал не менялся (после обрыва и после правки строки), итог бит в
бит как с начала. В папке — последняя точка каждого этапа и до 12 промежуточных лесенкой (у конца журнала гуще),
не больше `--checkpoints-max` ГБ (2); лишнее уходит само (сначала промежуточные, дальние от конца), строкой в stderr; `--save` — жёсткая ссылка на точку конца. В Python — `собрать`."""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from dataclasses import asdict

import numpy as np

from .brush import Brush, path_with_pressure, gesture, pressure_along, segments_touch, polyline, jab, _window
from .path import generator_sweep
from .canvas import Canvas, EmptyStroke
from .soft import release_along
from .paint import Paint
from .tool import hint

from . import scale as mm_

_PResponseDWithBrushAZBrushAnd = False      # 5б: повтор молчит (стенд подсказок включает их здесь)
_NAndZhE = {}              # М-7: мазков этого прогона, что в масштабе < 1 легли ниже настройки: {'мягкая': n, 'узкая': n}
from .journal import FAndZAndBrushA, difference_physics

# Е-7: поля строк журнала. Строка с незнакомым полем — ошибка с номером строки и ближайшим именем; в журнале без
# заголовка "physics" (записан до встроенного журнала) — предупреждение, поле пропускается, как раньше.
FIELDS_CANVAS = {'canvas', 'ground', 'px_per_mm', 'grain', 'physics', 'seed', 'seeds'}
FIELDS_INTENTS = {'stage', 'object', 'intent'}
FIELDS_STROKE = {'lines', 'layer', 'size', 'color', 'colors', 'ctrl', 'points', 'core', 'opacity', 'layout', 'load', 'wipe', 'end',
              'mix_below', 'level', 'ridges', 'reserve', 'touch', 'seed', 'preset', 'paint', 'pressure', 'dry', 'set',
              'pressure_along', 'tool', 'fibers', 'load_along', 'entry', 'angle', 'rotation_along', 'tilt_along',
              'azimuth_along', 'brush_tone', 'stripe', 'stripe_len', 'brush', 'hand', 'path_seed', 'path_rng', 'medium',
              'swatch', 'depth'} | FIELDS_INTENTS
FIELDS_DRYING = {'dry', 'set', 'wet', 'layer', 'area'} | FIELDS_INTENTS
FIELDS_LAYERS = {'layer_drop', 'layer_order'} | FIELDS_INTENTS
FIELDS_TAPES = {'tape', 'shape', 'mask', 'layer', 'height', 'untape'} | FIELDS_INTENTS   # Н-3: скотч — лента и снятие
FIELDS_BRUSHES = {'brush', 'def', 'wipe', 'dirt', 'pick', 'colors', 'paint', 'layout', 'pick_linear', 'pick_cover', 'swatch', 'left'} | FIELDS_INTENTS
FIELDS_PALETTE = {'palette', 'color', 'paint'} | FIELDS_INTENTS     # М-1: замес палитры
FIELDS_LOADING = {'load_brush', 'wipe', 'rinse', 'color', 'colors', 'paint', 'swatch', 'amount', 'layout'} | FIELDS_INTENTS   # Д-1: живая краска кисти
NOT_WITH_CORES = ('ctrl', 'points', 'end', 'entry', 'pressure', 'angle', 'pressure_along', 'path_seed', 'path_rng') + ('rotation_along', 'tilt_along', 'azimuth_along')
WithInResponseBResponseDNYE = ('mark', 'note', 'saved')      # строки, которые повтор не исполняет: поля в них — любые

_PREWithETY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'пресеты')


def _brush(name, size, cache, tone_brushes=None, scale=1.0):
    option = (name, round(size, 2)) if tone_brushes is None else (name, round(size, 2), float(tone_brushes))
    if option not in cache:
        if name is None:
            name = 'плоская' if size >= 20 else 'плоская_узкая'
        path = name if name.endswith('.json') else os.path.join(_PREWithETY, name + '.json')
        replacements = {} if tone_brushes is None else {'тон_кисти': float(tone_brushes)}       # 11а: поле журнала "brush_tone"
        brush = Brush.from_file(path, width=size, **replacements)
        if scale != 1.0 and brush.spacing_mm is None:      # Д-8: шаг ворса пресета — px журнала, как "size"
            brush.spacing = float(brush.spacing) * scale
        cache[option] = brush
    return cache[option]


FIELDS_ANGLE = ('rotation_along', 'tilt_along', 'azimuth_along')


def _profile_angle(profile, where, limit=None):
    """Профиль угла по ходу — [[доля пути, градусы], …]: доли 0–1 строго по возрастанию, градусы — конечные,
    с пределом (мин, макс) — в нём. Ответ — массив (n, 2) или None. Кривой — ValueError с `где` в начале."""
    if profile is None:
        return None
    try:
        p = np.asarray(profile, dtype=np.float64)
    except (TypeError, ValueError):
        p = None
    if p is None or p.ndim != 2 or p.shape[1] != 2 or len(p) < 1 or not np.isfinite(p).all():
        raise ValueError(f'{where}: должен быть списком пар [доля пути, градусы]')
    if (p[:, 0] < 0).any() or (p[:, 0] > 1).any():
        raise ValueError(f'{where}: доля пути вне 0–1')
    if (np.diff(p[:, 0]) <= 0).any():
        raise ValueError(f'{where}: доли пути не по возрастанию')
    if limit is not None and ((p[:, 1] < limit[0]).any() or (p[:, 1] > limit[1]).any()):
        raise ValueError(f'{where}: градусы вне {limit[0]:g}–{limit[1]:g}')
    return p


def _sweep_points(xy):
    """Направление хода у каждой точки ломаной, радианы без скачков: среднее соседних отрезков; у стоящих на месте
    точек — направление ближайшего отрезка с длиной."""
    d = np.diff(xy, axis=0)
    has = np.hypot(d[:, 0], d[:, 1]) > 1e-6
    if not has.any():
        return np.zeros(len(xy))
    ug = np.arctan2(d[:, 1], d[:, 0])
    number = np.arange(len(d))
    brush_nearest = np.where(has, number, -1)
    np.maximum.accumulate(brush_nearest, out=brush_nearest)                  # пустой отрезок берёт направление прошлого
    first = int(np.argmax(has))
    brush_nearest[brush_nearest < 0] = first
    ug = np.unwrap(ug[brush_nearest])
    return np.concatenate([[ug[0]], 0.5 * (ug[:-1] + ug[1:]), [ug[-1]]])


def angles_along(points, rotation=None, slant=None, azimuth=None):
    """К1, поля журнала "rotation_along", "tilt_along", "azimuth_along": профили [[доля пути, градусы], …] -> столбцы
    ядра (наклон °, азимут рад, поворот ° от оси x) у точек мазка (x, y, нажим, dt). Доля пути — по длине ломаной
    точек, как у `нажим_вдоль`; между узлами по прямой. Поворот и азимут — от направления хода в этой точке: кисть
    держится к ходу и на дуге. Нет ни одного профиля — точки как были (тот же объект)."""
    rotation = _profile_angle(rotation, 'rotation_along')
    tilt = _profile_angle(slant, 'tilt_along', (0.0, 85.0))
    az = _profile_angle(azimuth, 'azimuth_along')
    t = np.asarray(points, np.float64)
    if (rotation is None and tilt is None and az is None) or t.ndim != 2 or t.shape[1] < 4 or len(t) < 2:
        return points
    t = t[:, :4]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t[:, :2], axis=0).T))])
    d = s / max(s[-1], 1e-9)
    sweep = _sweep_points(t[:, :2])

    def by(p):
        return np.zeros(len(t)) if p is None else np.interp(d, p[:, 0], p[:, 1])
    # ядро (`путь._путь`): угол кисти = поворот − 90°, поперёк кисти — её широкая ось; поворот 0 журнала — угол = ход
    return np.column_stack([t, by(tilt), sweep + np.radians(by(az)), np.degrees(sweep) + by(rotation) + 90.0])


NOT_WITH_BY_DOTS = ('ctrl', 'end', 'entry', 'pressure', 'angle', 'path_seed', 'path_rng')


def _sweep_polyline(xy, lag=0.0):
    """Куда тянутся пряди у точек набитой ломаной, радианы без скачков — как без столбца азимута у ядра (`путь._путь`):
    касательная хода, сглаженная окном в 1,5 px, а у плоской кисти — доворот за ходом с отставанием (px пути; тот же
    закон, что у поворота самой кисти). Излом руки пряди проходят на этой длине, а не скачком в одной точке: скачок
    рвал след строкой дыр поперёк кисти."""
    d = np.diff(xy, axis=0)
    length = np.hypot(d[:, 0], d[:, 1])
    s = np.concatenate([[0.0], np.cumsum(length)])
    if s[-1] <= 1e-6:
        return np.zeros(len(xy))
    ss = np.linspace(0.0, s[-1], max(int(np.ceil(s[-1] / 0.25)) + 1, 3))
    ds = ss[1] - ss[0]
    has = np.concatenate([[True], length > 1e-9])              # стоящие на месте точки длины не дают
    x, y = np.interp(ss, s[has], xy[has, 0]), np.interp(ss, s[has], xy[has, 1])
    dx, dy = _window(np.gradient(x, ds), int(1.5 / ds)), _window(np.gradient(y, ds), int(1.5 / ds))
    sweep = np.unwrap(np.arctan2(dy, dx))
    if lag > 0:
        share = 1.0 - np.exp(-ds / max(float(lag), 1e-3))
        sweep = sweep.tolist()
        at = sweep[min(len(sweep) - 1, int(2.0 / ds))]
        behind_sweep = []
        for f in sweep:
            at += (f - at) * share
            behind_sweep.append(at)
        sweep = np.array(behind_sweep)
    return np.interp(s, ss, sweep)


def points_journal(points, scale=1.0, lag=0.0):
    """К2, поле журнала "points": [[x, y, нажим, dt, наклон, азимут, поворот], …] -> точки ядра (x, y, нажим, dt_мс,
    наклон °, азимут рад, поворот ° ядра) — столько столбцов, сколько дано (без dt — четыре), путь — ломаная по самим
    точкам (`кисть.ломаная`: отрезки набиты точками через полпикселя, нажим и углы между точками руки по прямой).
    Координаты и dt — журнала (× масштаб: скорость руки в ширинах кисти от масштаба не зависит); dt нет или null —
    1 мс на px пути (скорость тела у `путь_с_нажимом`). Поворот — градусы на холсте от оси x, как "angle": направление
    кисти, её широкая сторона встаёт поперёк него. Азимут — градусы от направления хода, как "azimuth_along": 0 — пряди
    тянутся назад по ходу, как без столбца (у одной точки хода нет — от поворота); отставание — px пути, на которых
    пряди плоской кисти доворачиваются за изломом хода: `кисть.отставание · ширина`, у круглой — 0 (так зовёт повтор).
    Ни входа, ни схода, ни сплайна. Кривое поле — ValueError."""
    try:
        p = np.array(points, dtype=np.float64)
    except (TypeError, ValueError):
        p = None
    if p is not None and p.shape == (1, 2):  # З-10: тычок одной точкой [x, y] — как `[(x, y)]` в Python: нажим строит ядро
        return p * scale
    if p is None or p.ndim != 2 or not 3 <= p.shape[1] <= 7 or len(p) < 1:
        raise ValueError('points: должен быть списком точек [x, y, нажим, dt, наклон, азимут, поворот], '
                         'столбцов поровну, от трёх до семи (тычок — одна точка, можно [[x, y]])')
    if p.shape[1] == 3:
        p = np.column_stack([p, np.full(len(p), np.nan)])
    no_dt = np.isnan(p[:, 3])
    p[no_dt, 3] = 0.0
    if not np.isfinite(p).all():
        raise ValueError('points: нечисло (null можно только в dt)')
    if (p[:, 2] < 0).any() or (p[:, 2] > 1).any():
        raise ValueError('points: нажим вне 0–1')
    if (p[:, 3] < 0).any():
        raise ValueError('points: dt меньше 0')
    if p.shape[1] >= 5 and ((p[:, 4] < 0).any() or (p[:, 4] > 85).any()):
        raise ValueError('points: наклон вне 0–85')
    p[:, :2] *= scale
    p[:, 3] *= scale
    steps = np.concatenate([[0.0], np.hypot(*np.diff(p[:, :2], axis=0).T)])
    p[no_dt, 3] = steps[no_dt]
    if p.shape[1] >= 6:
        p[:, 5] = np.radians(p[:, 5])
    p = polyline(p)
    if p.shape[1] >= 6:                      # азимут — от хода в этой точке ломаной
        if (steps > 1e-6).any():
            p[:, 5] += _sweep_polyline(p[:, :2], lag)
        elif p.shape[1] >= 7:
            p[:, 5] += np.radians(p[:, 6])
    if p.shape[1] >= 7:
        p[:, 6] += 90.0
    return p


def _mm_replay(px_on_mm, scale):
    """Д-8: px на мм холста повтора. Журнал без "px_per_mm" писан на родном холсте физики (8 px/мм): в другом масштабе
    холст получает 8 · масштаб, чтобы числа физики шли за ним (`мм.py`); в масштабе 1 — как в журнале."""
    if px_on_mm is None:
        from .mm import RResponseDNResponseY
        return None if scale == 1.0 else RResponseDNResponseY * float(scale)
    return float(px_on_mm) * scale


def _job_brushes(z):
    """Д-1: строка живой краски кисти — {"load_brush": имя, цвет, "amount", "layout"}, {"wipe": имя}, {"rinse": имя} —
    её ключ; иначе None (у мазка "wipe": true — поле, а не строка)."""
    if 'ctrl' in z or 'points' in z or 'core' in z or 'brush' in z:
        return None
    return next((p for p in ('load_brush', 'wipe', 'rinse') if isinstance(z.get(p), str)), None)


def _loading_journal(z, job, number, brushes, mixes=None):
    """Д-1: строка живой краски — тем же вызовом кисти, каким её писали: `набрать(цвет, количество)`, `вытереть()`, `промыть()`."""
    brush = brushes.get(z[job])
    if brush is None:
        raise ValueError(f'журнал, строка {number}: кисть {z[job]} не определена (нет строки {{"brush": "{z[job]}", "def": …}} выше)')
    if job == 'wipe':
        brush.wipe()
    elif job == 'rinse':
        brush.rinse()
    else:
        from .palette import color_from_lines
        try:
            brush.charge(color_from_lines(z, mixes or {}, _color(z)), z.get('amount'), z.get('layout', 'полосы'))
        except ValueError as e:
            raise ValueError(f'журнал, строка {number}: {e}') from None


def _brush_journal(z, number, brushes, scale, mixes=None):
    """Строки кисти встроенного журнала (`кисть/журнал.py`): "def" — кисть всеми полями (новая или правленая: грязь и
    набранное у правленой остаются), "wipe", "dirt", "pick" / "pick_linear"."""
    name = z['brush']
    if 'def' in z:
        fields = dict(z['def'])
        from .tool import Brush
        extra = set(fields) - set(Brush.__dataclass_fields__)
        if extra:
            raise ValueError(f'журнал, строка {number}: кисть {name}: ' + '; '.join(hint(l, Brush.__dataclass_fields__) for l in sorted(extra)))
        if scale != 1.0 and fields.get('ширина_мм') is None:
            fields['ширина'] = float(fields['ширина']) * scale
        if scale != 1.0 and fields.get('шаг_мм') is None and 'шаг' in fields:      # Д-8: шаг ворса — px журнала, как ширина
            fields['шаг'] = float(fields['шаг']) * scale
        brush = brushes.get(name)
        if brush is None:
            brushes[name] = Brush.from_dict(fields)
        else:
            for field, v in fields.items():
                setattr(brush, field, v)
        return
    brush = brushes.get(name)
    if brush is None:
        raise ValueError(f'журнал, строка {number}: кисть {name} не определена (нет строки {{"brush": "{name}", "def": …}} выше)')
    if z.get('wipe'):
        brush._remainder = None                   # прежняя строка {"brush", "wipe": true}: подобранного на кончиках нет (живую краску снимает строка {"wipe": имя})
    if 'dirt' in z:
        brush._remainder = tuple(np.asarray(a, dtype=np.float64) for a in z['dirt'])
    if 'pick_linear' in z:
        from .tool import _Paints
        paints = [(np.asarray(c, dtype=np.float64), float(d)) for c, d in z['pick_linear']]
        if z.get('pick_cover') is not None:
            paints = _Paints(paints)
            paints.turbidity = float(z['pick_cover'])
        brush._paint, brush._charge = (paints, z.get('layout', 'полосы')), None
        if z.get('left') is not None:        # Д-1: живая краска как есть — сколько её в ворсе и на кончиках
            from .tool import Loading
            brush._paint = Loading(brush._paint, *z['left'])
    elif 'pick' in z:
        if z['pick'] is None:
            brush._paint = brush._charge = None
        else:
            from .palette import color_from_lines
            brush.charge(color_from_lines(z, mixes or {}, _color(z, 'pick')), None, z.get('layout', 'полосы'))


def _color(z, field='color'):
    """Цвет строки: "color" (или "pick") / "colors" и "paint" — вид одной краски, всех красок набора или списком по
    краске. "color": null — None (мазок набранной краской)."""
    charge = 'colors' in z or field == 'pick' and isinstance(z[field], list) and len(z[field]) > 0 and isinstance(z[field][0], list)
    color = z['colors'] if 'colors' in z else z.get(field, (0, 0, 0))
    if color is None or 'paint' not in z:
        return color
    if not charge:
        return Paint(color, z['paint'])
    views = z['paint'] if isinstance(z['paint'], list) else [z['paint']] * len(color)
    if len(views) != len(color):
        raise ValueError('"paint" списком — по виду на каждую краску "colors"')
    return [(Paint(e[:3] if len(e) != 2 else e[0], in_), e[3] if len(e) == 4 else e[1] if len(e) == 2 else 1.0)
            for e, in_ in zip(color, views)]


def _check_fields(z, number, strict, stated):
    """Е-7: незнакомое поле — ошибка с ближайшим именем (у старого журнала — предупреждение раз на поле)."""
    if any(brush in z for brush in WithInResponseBResponseDNYE):
        return
    if 'canvas' in z:
        known = FIELDS_CANVAS
    elif 'palette' in z:
        known = FIELDS_PALETTE
    elif 'brush' in z and isinstance(z['brush'], str) and not ('core' in z or 'ctrl' in z or 'points' in z):
        known = FIELDS_BRUSHES
    elif _job_brushes(z) is not None:
        known = FIELDS_LOADING
    elif 'ctrl' in z or 'points' in z or 'core' in z:
        known = FIELDS_STROKE
    elif 'dry' in z or 'set' in z or 'wet' in z:
        known = FIELDS_DRYING
    elif 'layer_drop' in z or 'layer_order' in z:
        known = FIELDS_LAYERS
    elif 'tape' in z or 'untape' in z:
        known = FIELDS_TAPES
    else:
        known = set()
    extra = [brush for brush in z if brush not in known]
    if not extra:
        return
    if not known:
        raise ValueError(f'журнал, строка {number}: не понять, что за строка (поля {sorted(z)}): мазку нужны "ctrl", "points" '
                         'или "core"; сушке — "dry", "set" или "wet"; пометке — "mark"; заметке — "note"')
    all = FIELDS_CANVAS | FIELDS_STROKE | FIELDS_BRUSHES | FIELDS_LAYERS | FIELDS_TAPES | FIELDS_PALETTE | FIELDS_LOADING | set(WithInResponseBResponseDNYE)
    text = '; '.join(hint(l, known if hint(l, known) != f'нет поля {l}' else all) for l in extra)
    if strict:
        raise ValueError(f'журнал, строка {number}: {text}')
    for l in extra:
        if l not in stated:
            stated.add(l)
            warnings.warn(f'журнал, строка {number}: {hint(l, known)} — журнал без "physics", поле пропущено', stacklevel=4)


def _seed_lines(z, number, seed, tree, canvas, current):
    """Сид мазка строки: "seed"; без него — у журнала с "seeds": "tree" (А-9) из сида листа, этапа, объекта и точек
    строки ("ctrl" / "points" / "core" как записаны), как `Холст.сид_мазка`; у прежнего журнала — от номера строки."""
    if 'seed' in z:
        return int(z['seed'])
    if not tree:
        return seed * 1_000_003 + number
    wet = z.get('core', z.get('points', z.get('ctrl')))
    return canvas.seed_stroke(np.asarray(wet, dtype=np.float64), z.get('stage', current['stage']), z.get('object', current['object']),
                       take=True)


def select(lines, stage=None, object=None):
    """А-6: строки журнала, у которых "stage" и "object" — эти (None — любое; сравнение строкой, этап 4 и "4" — одно).
    У строки без своих полей они — от последней пометки {"mark": …} выше. Ответ — список (номер строки, запись) мазков,
    снятий, пометок и заметок."""
    response, current = [], {'stage': None, 'object': None}
    for number, row in enumerate(lines, 1):
        row = row.strip()
        if not row or row.startswith('#'):
            continue
        z = json.loads(row)
        if 'mark' in z:
            current = {p: z.get(p, current[p]) for p in current}
        if not any(brush in z for brush in ('ctrl', 'points', 'core', 'mark', 'note')):
            continue
        e, response = z.get('stage', current['stage']), z.get('object', current['object'])
        if (stage is None or str(e) == str(stage)) and (object is None or str(response) == str(object)):
            response.append((number, z))
    return response


def save_state(path, state):
    """К3: состояние повтора (словарь от `повторить(…, состояние=)`) в .npz — холст и всё, от чего зависит следующий
    мазок: каждая кисть кэша (и чистая) со своей шириной и грязью, масштаб, сид, пресет, номер последней положенной
    строки. Ширина кисти — та, с которой её завела первая строка журнала (точно, а не ключ кэша: ключ округлён до
    сотых, и две строки с размерами 19,191 и 19,199 пишут одной кистью — шириной первой). Кисти — по порядку ключей:
    журнал целиком и журнал в два захода дают один файл."""
    brushes, arrays = [], {}
    def order(kk):
        return ('' if kk[0][0] is None else '!' + str(kk[0][0]),) + tuple(kk[0][1:])
    for option, brush in sorted(state['кэш'].items(), key=order):
        j = len(brushes)
        if brush._remainder is not None:
            arrays[f'k{j}'], arrays[f'd{j}'] = (np.asarray(a, dtype=np.float64) for a in brush._remainder)
        brushes.append({'пресет': option[0], 'размер': option[1], 'тон': option[2] if len(option) > 2 else None,
                      'ширина': float(brush.width), 'грязная': brush._remainder is not None,
                      'набрано': getattr(brush, '_набрано', None), 'баланс': getattr(brush, '_баланс', None)})
    named = []                                 # кисти встроенного журнала: всеми полями, с грязью и набранной краской
    for name, brush in sorted(state.get('кисти', {}).items()):
        j = len(named)
        if brush._remainder is not None:
            arrays[f'nk{j}'], arrays[f'nd{j}'] = (np.asarray(a, dtype=np.float64) for a in brush._remainder)
        paint = None
        if brush._paint is not None:
            kr, layout = brush._paint
            paint = {'краски': [[np.asarray(c, np.float64).tolist(), float(d)] for c, d in kr],
                      'мутность': getattr(kr, 'мутность', None), 'раскладка': layout}
            if hasattr(brush._paint, 'запас'):       # Д-1: живая краска — сколько её в ворсе и на кончиках
                paint['осталось'] = [brush._paint.reserve.tolist(), brush._paint.drop.tolist()]
        named.append({'имя': name, 'поля': asdict(brush), 'грязная': brush._remainder is not None, 'краска': paint,
                        'набрано': getattr(brush, '_набрано', None), 'баланс': getattr(brush, '_баланс', None)})
    meta = {'масштаб': float(state['масштаб']), 'сид': int(state['сид']), 'пресет': state['пресет'],
            'строка': int(state['строка']), 'кисти': brushes}
    if named:
        meta['именные'] = named
    state['холст'].save(path, beyond=(meta, arrays))


def load_file_state(path):
    """К3: состояние повтора из файла `--save` — словарь для `повторить(…, состояние=)`: 'холст', 'кэш' (кисти той же
    ширины, что были, с их грязью), 'масштаб', 'сид', 'пресет', 'строка' (последняя положенная строка журнала)."""
    canvas, meta, arrays = Canvas.load_file(path, beyond=True, hints=_PResponseDWithBrushAZBrushAnd)
    if meta is None or 'кисти' not in meta:
        raise ValueError(f'{path}: в файле только холст, состояния кистей и места в журнале нет '
                         '(записан до К3 или `Холст.сохранить`) — дорисовать с него нельзя')
    cache = {}
    for j, d in enumerate(meta['кисти']):
        if 'ширина' not in d:
            raise ValueError(f'{path}: кисти записаны без точной ширины (файл прежней записи `--save`) — '
                             'дорисовка с него разойдётся с журналом целиком; пересобрать первый заход')
        brush = _brush(d['пресет'], d['ширина'], cache, d['тон'], float(meta['масштаб']))
        if (len(cache), round(d['ширина'], 2)) != (j + 1, d['размер']):
            raise ValueError(f'{path}: кисть {j} записана криво — ширина {d["ширина"]} не под ключ {d["размер"]}')
        if d['грязная']:
            brush._remainder = (arrays[f'k{j}'], arrays[f'd{j}'])
        if d['набрано'] is not None or d['баланс'] is not None:
            brush._charged, brush._balance = d['набрано'], d['баланс']
    from .tool import Brush, _Paints
    named = {}
    for j, d in enumerate(meta.get('именные', [])):
        brush = named[d['имя']] = Brush.from_dict(d['поля'])
        if d['грязная']:
            brush._remainder = (arrays[f'nk{j}'], arrays[f'nd{j}'])
        if d['краска'] is not None:
            kr = [(np.asarray(c, dtype=np.float64), float(share)) for c, share in d['краска']['краски']]
            if d['краска']['мутность'] is not None:
                kr = _Paints(kr)
                kr.turbidity = d['краска']['мутность']
            brush._paint = (kr, d['краска']['раскладка'])
            if d['краска'].get('осталось') is not None:
                from .tool import Loading
                brush._paint = Loading(brush._paint, *d['краска']['осталось'])
        if d['набрано'] is not None or d['баланс'] is not None:
            brush._charged, brush._balance = d['набрано'], d['баланс']
    return {'холст': canvas, 'кэш': cache, 'кисти': named, 'масштаб': meta['масштаб'], 'сид': meta['сид'],
            'пресет': meta['пресет'], 'строка': meta['строка']}


def repeat(lines, width=4800, height=3000, ground=(150, 141, 126), scale=1.0, preset=None,
              seed=0, print=None, state=None, start=1, selection=None, point=None, cores=None):
    """Положить мазки журнала на новый холст. строки — итератор строк JSONL. Ответ: (холст, число мазков).
    состояние (К3) — словарь: после повтора в нём всё для `сохранить_состояние`; если в нём уже есть 'холст' и 'кэш'
    (`загрузить_состояние`) — мазки идут поверх этого холста этими кистями, а не с чистого листа (масштаб, сид и
    пресет — те, что в вызове: при дорисовке их дают прежними). начало — с какой строки журнала класть (счёт с 1):
    строки до неё пропускаются не глядя, номера строк и сиды мазков — как в журнале целиком. отбор (А-6) — (этап, объект):
    класть только мазки и снятия с этими "stage" и "object" (None — любое; без своих полей — от пометки выше), остальные
    строки (холст, кисти, сушка) — все. Мазок без следа — предупреждение с номером строки. точка (Е-5) — вызов
точка(положена, з, тек, n) перед каждой строкой, которую повтор исполняет: положена — последняя положенная строка
(состояние в `состояние` — после неё), з — строка, которая сейчас ляжет, тек — этап и объект до неё, n — мазков;
так `собрать` пишет контрольные точки. ядра (Е-1) — сколько процессов считают мазки: мазки, чьи прямоугольники не
пересекаются, идут разом, итог — бит в бит как на одном (`кисть/ядра.py`); None — `KIST_CORES` или ядра машины, не
больше 16; 1 — одно ядро."""
    from .cores import number_cores
    box = []                    # очередь мазков холста этого повтора: в конце и при ошибке её процессы останавливаются
    try:
        return _repeat(lines, width, height, ground, scale, preset, seed, print, state, start, selection, point,
                          number_cores(cores), box)
    except ValueError as e:      # ошибка мазка, считавшегося на ядрах, звучит позже его строки: номер строки — с ним
        if getattr(e, 'метка', None) is not None and not str(e).startswith('журнал, строка'):
            raise ValueError(f'журнал, строка {e.label}: {e}') from None
        raise
    finally:
        for used in box:
            used._stop()
            used.canvas.__dict__.pop('_готово', None)
            used.canvas.__dict__.pop('_метка', None)
        canvas = (state or {}).get('холст')
        if canvas is not None:                # К-3: рамки и пропуск — только на время повтора
            for field in ('_рамки', '_мимо', '_метка'):
                canvas.__dict__.pop(field, None)


def _repeat(lines, width, height, ground, scale, preset, seed, print, state, start, selection, point, cores, box):
    def ready(number, measured, words):       # предупреждения мазка — с номером строки, в порядке журнала
        for text, kind in words:
            empty = isinstance(kind, type) and issubclass(kind, EmptyStroke)
            warnings.warn(f'журнал, строка {number}: {text}' if empty else text, EmptyStroke if empty else kind, stacklevel=4)

    def boundary(layers=False):             # Е-1: до сушки, снятия, слоёв, точки и конца — все мазки досчитаны (холст делает это и сам)
        if canvas is not None:
            canvas._boundary()

    if state is None:
        state = {}
    canvas = state.get('холст')
    cache = state.setdefault('кэш', {})
    brushes = state.setdefault('кисти', {})
    state['_досчитать'] = boundary
    laid = int(state.get('строка', 0))
    frames, past, before = state.get('рамки'), state.get('мимо') or (), state.get('перед')   # К-3 (`задним.py`)
    strict, stated = False, set()
    px_on_mm = grain = None
    seed_sheet, tree = 0, False      # А-9: "seed" и "seeds": "tree" заголовка
    current = {'stage': None, 'object': None}
    n = 0
    for number, row in enumerate(lines, 1):
        if number < start:
            if '"physics"' in row or '"mark"' in row:
                z = json.loads(row)
                strict = strict or 'physics' in z and 'canvas' in z
                if 'canvas' in z:
                    tree = z.get('seeds') == 'tree'
                if 'mark' in z:
                    current = {p: z.get(p, current[p]) for p in current}
            continue
        row = row.strip()
        if not row or row.startswith('#'):
            continue
        try:
            z = json.loads(row)
        except json.JSONDecodeError as e:
            raise ValueError(f'журнал, строка {number}: не JSON ({e})') from None
        if point is not None and canvas is not None:
            state.update(canvas=canvas, scale=scale, seed=seed, preset=preset, row=laid)
            point(laid, z, current, n)                           # точку пишет после `состояние['_досчитать']()`
        laid = number
        if 'canvas' in z and 'physics' in z:
            strict = True
        _check_fields(z, number, strict, stated)
        if 'mark' in z:
            current = {p: z.get(p, current[p]) for p in current}
        if any(brush in z for brush in WithInResponseBResponseDNYE):
            continue
        if 'canvas' in z and canvas is None:
            width, height = z['canvas']
            ground = z.get('ground', ground)
            px_on_mm, grain = z.get('px_per_mm'), z.get('grain')
            seed_sheet, tree = int(z.get('seed', 0)), z.get('seeds') == 'tree'
            if z.get('seeds') not in (None, 'tree'):
                raise ValueError(f'журнал, строка {number}: "seeds" — только "tree" (сид строки из листа, этапа, объекта и точек)')
            if z.get('physics') not in (None, FAndZAndBrushA):
                warnings.warn(difference_physics(z['physics']), stacklevel=2)
            continue
        if canvas is None:
            canvas = Canvas(int(width * scale), int(height * scale), ground,
                      px_on_mm=_mm_replay(px_on_mm, scale), grain=grain, seed=seed_sheet, cores=cores, hints=_PResponseDWithBrushAZBrushAnd)
        if not tree:                       # прежний журнал: сиды от номера строки, счёта повторов нет (и `.npz` прежний)
            canvas._repeats = None
        if selection is not None and any(brush in z for brush in ('ctrl', 'points', 'core')):
            e, response = z.get('stage', current['stage']), z.get('object', current['object'])
            if (selection[0] is not None and str(e) != str(selection[0])) or (selection[1] is not None and str(response) != str(selection[1])):
                continue
        if 'tape' in z or 'untape' in z:                          # Н-3: скотч — лента ложится, лента снята
            boundary(layers=True)
            try:
                if 'tape' in z:
                    from .tape import mask_from_lines, in_size
                    if 'mask' not in z or 'shape' not in z:
                        raise ValueError('у ленты нужны "mask" и "shape"')
                    mask = in_size(mask_from_lines(z['mask'], z['shape']), canvas.height, canvas.width)
                    canvas.tape(mask, layer=z.get('layer'), name=z['tape'], height_mm=z.get('height'))
                else:
                    canvas.remove_tape(z['untape'])
            except (KeyError, ValueError) as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            continue
        if 'layer_drop' in z or 'layer_order' in z:                # 1в: слой убран, порядок слоёв — `Холст.убрать_слой`, `порядок_слоёв`
            if 'layer_drop' in z and 'layer_order' in z:
                raise ValueError(f'журнал, строка {number}: "layer_drop" и "layer_order" — разными строками')
            boundary(layers=True)
            try:
                if 'layer_drop' in z:
                    canvas.remove_layer(z['layer_drop'])
                else:
                    canvas.order_layers(z['layer_order'])
            except (KeyError, ValueError) as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            continue
        if 'palette' in z:                                      # М-1: замес палитры — завести или поправить
            try:
                canvas.palette.add(z['palette'], _color(z))
            except ValueError as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            continue
        if _job_brushes(z) is not None:                           # Д-1: живая краска кисти
            _loading_journal(z, _job_brushes(z), number, brushes, canvas.__dict__.get('_замесы'))
            continue
        if isinstance(z.get('brush'), str) and not any(p in z for p in ('ctrl', 'points', 'core')):
            try:                                                 # А-1: кисть встроенного журнала
                _brush_journal(z, number, brushes, scale, canvas.__dict__.get('_замесы'))
            except KeyError as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            continue
        wet = 'points' in z                                   # К2: сырые точки пути вместо "ctrl"
        core = 'core' in z
        if 'ctrl' not in z and not wet and not core:
            region = z.get('area')
            if region is not None and scale != 1.0:          # 11ж: прямоугольник — в координатах журнала
                region = [float(v) * scale for v in region]
            boundary()
            if 'dry' in z:
                canvas.dry(float(z['dry']), layer=z.get('layer'), region=region)
            if 'set' in z:
                canvas.set(float(z['set']), layer=z.get('layer'), region=region)
            if 'wet' in z:                                      # М-3: размочить
                canvas.re_wet(float(z['wet']), layer=z.get('layer'), region=region)
            continue
        if 'dry' in z or 'set' in z:
            boundary()
        if 'dry' in z:
            canvas.dry(float(z['dry']))
        if 'set' in z:
            canvas.set(float(z['set']))
        size = float(z.get('size', 12.0)) * scale
        if core:                                                # А-1: точки ядра как легли в Python, без пересчёта пути
            extra = [field for field in NOT_WITH_CORES if field in z]
            if extra:
                raise ValueError(f'журнал, строка {number}: при "core" нет полей ' + ', '.join(f'"{p}"' for p in extra)
                                 + ' — "core" — точки ядра, путь и нажим в них')
            points = np.array(z['core'], dtype=np.float64)
            if points.ndim != 2 or not 2 <= points.shape[1] <= 7 or len(points) < 1:
                raise ValueError(f'журнал, строка {number}: "core" — список точек [x, y, нажим, dt, наклон, азимут, поворот], от двух столбцов')
            if scale != 1.0:
                points[:, :2] *= scale
                if points.shape[1] >= 4:
                    points[:, 3] *= scale
            seed_stroke = _seed_lines(z, number, seed, tree, canvas, current)
            pieces = None
        else:
            size = float(z.get('size', 12.0)) * scale
            if wet:
                extra = [field for field in NOT_WITH_BY_DOTS if field in z]
                if extra:
                    raise ValueError(f'журнал, строка {number}: при "points" нет полей ' + ', '.join(f'"{p}"' for p in extra)
                                     + ' — путь, нажим и поворот заданы точками')
                try:
                    if 'tool' in z:                  # К8.4: тряпка и скребок за ходом не доворачиваются
                        ctrl = points_journal(z['points'], scale, 0.0)
                    else:
                        brush = _brush(z.get('preset', preset), size, cache, z.get('brush_tone'), scale)
                        ctrl = points_journal(z['points'], scale, 0.0 if brush.shape == 'круглая' else brush.lag * size)
                except ValueError as e:
                    raise ValueError(f'журнал, строка {number}: {e}') from None
            else:
                ctrl = np.asarray(z['ctrl'], dtype=np.float64) * scale
                if ctrl.ndim != 2 or ctrl.shape[1] < 2:
                    raise ValueError(f'журнал, строка {number}: ctrl должен быть списком точек [x, y]')
            seed_stroke = _seed_lines(z, number, seed, tree, canvas, current)
            if 'path_rng' in z:              # А-3: генератор пути из Python — в том же состоянии
                rng = generator_sweep(z['path_rng'])
            else:
                rng = np.random.default_rng([int(z.get('path_seed', seed_stroke)), 7])
            pressure = float(z.get('pressure', 0.85))
            entry = None if z.get('entry') is None else float(z['entry']) * scale      # 11ж: длина входа, px журнала
            on_place = len(ctrl) == 1 or not (np.hypot(*np.diff(ctrl[:, :2], axis=0).T) > 1e-6).any()
            if wet:
                points = ctrl
            elif on_place and (z.get('angle') is not None or 'end' in z):
                # 11ж: тычок из одной точки — касание на месте; "angle" — направление кисти, градусы от оси x (y вниз);
                # "end" у тычка не действует (жест на нулевом пути давал нажим 0 и следа не было)
                points = jab(ctrl[0, 0], ctrl[0, 1], pressure, z.get('angle'))     # З-10: тот же, что `тычок` в Python
            elif 'end' in z:
                points = gesture(ctrl, size, z['end'], rng, pressure=pressure, entry=entry)
            else:
                points = path_with_pressure(ctrl, size, rng, pressure=pressure, entry=entry)
            if not on_place and any(z.get(field) is not None for field in FIELDS_ANGLE):     # К1: кисть в руке по ходу
                try:
                    hand = points
                    points = angles_along(points, z.get('rotation_along'), z.get('tilt_along'), z.get('azimuth_along'))
                    if wet:                        # К2: поле по ходу заменяет свой столбец точек, остальные — как дала рука
                        for column, field in ((4, 'tilt_along'), (5, 'azimuth_along'), (6, 'rotation_along')):
                            if z.get(field) is None and hand.shape[1] > column:
                                points[:, column] = hand[:, column]
                except ValueError as e:
                    raise ValueError(f'журнал, строка {number}: {e}') from None
            pieces = None
            if z.get('pressure_along') is not None:                              # 11д: нажим по ходу
                try:
                    points = pressure_along(points, z['pressure_along'])
                    pieces = segments_touch(points, z['pressure_along'])          # 11ж: множитель 0 — кисть над холстом
                except ValueError as e:
                    raise ValueError(f'журнал, строка {number}: {e}') from None
        if 'tool' in z:                      # К8.4: снятие сырой краски — тряпка или скребок, краски не кладёт
            if 'brush' in z:
                raise ValueError(f'журнал, строка {number}: у снятия ("tool") кисти нет — "brush" убрать')
            boundary()
            try:
                if z['tool'] == 'нож':               # М-3: `прижать` — срез ходом ножа; глубина — мм, с масштабом не меняется
                    canvas.press(points, width=size, layer=z.get('layer'), depth=z.get('depth'))
                else:
                    canvas.remove(points, z['tool'], size, seed_stroke, z.get('layer'))
            except ValueError as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            n += 1
            continue
        if isinstance(z.get('brush'), str):  # А-1: кисть встроенного журнала — по имени
            if z['brush'] not in brushes:
                raise ValueError(f'журнал, строка {number}: кисть {z["brush"]} не определена (нет строки {{"brush": "{z["brush"]}", "def": …}} выше)')
            if 'preset' in z or 'brush_tone' in z:
                raise ValueError(f'журнал, строка {number}: при "brush" нет "preset" и "brush_tone" — кисть задана строкой "def"')
            brush = brushes[z['brush']]
        else:
            brush = _brush(z.get('preset', preset), size, cache, z.get('brush_tone'), scale)
        if z.get('wipe'):
            brush.wipe()
        try:
            color = _color(z)              # вид краски: одна краска — она, набор — каждая или списком по краске
            if 'swatch' in z:            # М-1: замес — цвет из строки палитры выше (правка замеса — его мазки новым цветом)
                from .palette import color_from_lines
                color = color_from_lines(z, canvas.__dict__.get('_замесы', {}), color)
        except (ValueError, KeyError) as e:
            raise ValueError(f'журнал, строка {number}: {e}') from None
        bands = None                        # п. 13: ширина полосы набора и длина куска краски в ней, px журнала
        if z.get('stripe') is not None or z.get('stripe_len') is not None:
            bands = tuple(None if z.get(field) is None else float(z[field]) * scale for field in ('stripe', 'stripe_len'))
            if any(v is not None and v <= 0 for v in bands):
                raise ValueError(f'журнал, строка {number}: "stripe" и "stripe_len" — больше нуля')
        ridges = z.get('ridges')            # п. 13, часть 2: кисть с краской лёгкой рукой идёт по гребням лежащей краски, 0–1
        if ridges is not None and not 0.0 <= float(ridges) <= 1.0:
            raise ValueError(f'журнал, строка {number}: "ridges" — от 0 до 1')
        even = z.get('level')              # п. 15: текучая кроющая краска — борозды своих волосков за ворсом смыкаются, 0–1; К4: с лежащим рядом сырым текучим ходом своего слоя ход смыкается швом целиком, щели на стыке нет
        if even is not None and not 0.0 <= float(even) <= 1.0:
            raise ValueError(f'журнал, строка {number}: "level" — от 0 до 1')
        reserve = z.get('reserve')            # п. 17: сухой срыв — на сколько ширин кисти хватает краски
        if reserve is not None and not float(reserve) > 0.0:
            raise ValueError(f'журнал, строка {number}: "reserve" — больше нуля')
        fibers = z.get('fibers')           # К6: полупустая кисть ("load": "сухая") с расчёсанным ворсом — тело плотное, край волокнами, 0–1
        if fibers is not None and not 0.0 <= float(fibers) <= 1.0:
            raise ValueError(f'журнал, строка {number}: "fibers" — от 0 до 1')
        if fibers and ((brush.load if z.get('load') is None else z['load']) != 'сухая' or brush.shape == 'мастихин'):
            raise ValueError(f'журнал, строка {number}: "fibers" — только с "load": "сухая" (полупустая кисть, не мастихин)')
        thinner = z.get('medium')       # Г-1: краска с разбавителем — тоньше, прозрачнее, течёт, 0–1
        if thinner is not None:
            from .paint import check_thinner
            try:
                thinner = check_thinner(thinner, brush.load if z.get('load') is None else z['load'])
            except ValueError as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
        release = z.get('load_along')        # К5: мягкая кисть по ходу отдаёт плёнку тоньше
        if release is not None:
            try:
                release_along(release, brush)
            except ValueError as e:
                raise ValueError(f'журнал, строка {number}: {e}') from None
            if pieces is not None:
                raise ValueError(f'журнал, строка {number}: "load_along" вместе с отрывом кисти ("pressure_along" с множителем 0) — разбить на строки')
        how = dict(stage=z.get('stage', current['stage']), object=z.get('object', current['object']),
                   layer=str(z.get('layer', 'основа')), opacity=float(z.get('opacity', 1.0)),
                   layout=z.get('layout', 'полосы'), load=z.get('load'), mix_below=bool(z.get('mix_below', False)),
                        touch=z.get('touch'), bands=bands, by_ridges=float(ridges) if ridges else None,
                        even=float(even) if even else None, reserve=None if reserve is None else float(reserve), load_along=release, fibers=float(fibers) if fibers else None,
                   pressure_hand=z.get('pressure_along') is not None or bool(z.get('hand')), thinner=thinner)
        if not box:                                             # Е-1: холст повтора считает на стольких ядрах, сколько дано повтору
            ya = canvas.__dict__.get('_ядра')
            if ya is not None and ya.cores != cores:
                ya.close()
                canvas._cores = None
            if canvas.__dict__.get('_ядра') is None:
                canvas._cores = cores
            if canvas._queue() is not None:
                box.append(canvas._queue())
                canvas._ready = ready
        if scale < 1.0:                                        # М-7: черновик — мягкая и узкая держат px
            what = mm_.holds_px(brush, how['нагрузка'], canvas.px_on_mm)
            what = 'мягкая' if 'мягкая' in what else 'узкая' if 'узкая' in what else None     # мягкая узкая — мягкой
            if what:
                _NAndZhE[what] = _NAndZhE.get(what, 0) + 1
        if frames is not None:                                    # К-3: рамки мазков строки — повтору; мимо — учесть, не класть
            canvas._frames, canvas._label = frames, number
            canvas._past = number in past
            if before is not None and not canvas._past:
                before(number, canvas, brush, how['слой'])
        for k, piece in enumerate([points] if pieces is None else pieces):
            measurement, canvas.measurement = getattr(canvas, 'замер', True), False    # 3г: ответ мазка повтору не нужен — без замера следа
            try:
                if box:                                         # Е-1: мазок — в очередь на ядра
                    canvas._label = number
                    try:
                        canvas.stroke(piece, color, brush, seed=seed_stroke + k, **how)
                    except ValueError as e:
                        raise ValueError(f'журнал, строка {getattr(e, "метка", number)}: {e}') from None if hasattr(e, 'метка') else e
                    continue
                with warnings.catch_warnings(record=True) as empty:
                    warnings.simplefilter('always', EmptyStroke)
                    canvas.stroke(piece, color, brush, seed=seed_stroke + k, **how)
            finally:
                canvas.measurement = measurement
            for w in empty:
                warnings.warn(f'журнал, строка {number}: {w.message}' if issubclass(w.category, EmptyStroke) else w.message,
                              w.category if not issubclass(w.category, EmptyStroke) else EmptyStroke, stacklevel=3)
        n += 1
        if print and n % 1000 == 0:
            print(f'{n} мазков')
    if canvas is None:
        canvas = Canvas(int(width * scale), int(height * scale), ground,
                  px_on_mm=_mm_replay(px_on_mm, scale), grain=grain, seed=seed_sheet, cores=cores, hints=_PResponseDWithBrushAZBrushAnd)
        if not tree:
            canvas._repeats = None
    if box:
        box[0].close()
    canvas.__dict__.pop('_мимо', None)
    state.pop('_досчитать', None)
    canvas._stage, canvas._object = current['stage'], current['object']        # как у холста, писавшего журнал: от последней пометки
    state.update(canvas=canvas, scale=scale, seed=seed, preset=preset, row=laid)
    return canvas, n


def _hashes_lines(lines):
    """Е-5, А-8: хеш начала журнала по каждую строку: ответ[k] — sha256 строк 1…k (ответ[0] — пустого начала). Строка
    берётся как записана, без конца строки: правка любой буквы в строке k меняет ответ[k], ответ[k+1], …, а не раньше."""
    h, response = hashlib.sha256(), [hashlib.sha256().hexdigest()]
    for row in lines:
        h.update(row.rstrip('\r\n').encode('utf-8'))
        h.update(b'\n')
        response.append(h.copy().hexdigest())
    return response


_KIND_POINTS = re.compile(r'^строка_(\d{7})\.json$')
_KIND_PARTS = re.compile(r'^строка_\d{7}\.(?:npz|json)?\.?часть(\d*)')   # недописанная запись: `строка_N.частьPID…`
_KIND_NPZ = re.compile(r'^строка_(\d{7})\.npz$')
LIMIT_POINTS_GB = 2.0         # 1г: умолчание предела папки точек — Л4 (6 этапов по ~200 МБ) держится весь
LEWithENBrushA = 12                  # 1г: промежуточных точек (не последних в этапе) — не больше, и без предела


def points_folders(folder):
    """Е-5: контрольные точки в папке — список (строка, путь .npz, словарь .json), по строке. Точка — пара
    `строка_NNNNNNN.npz` (состояние `сохранить_состояние` после строки N) и `.json` рядом (хеш начала журнала,
    параметры повтора); `.json` пишется последним, без него `.npz` не точка (запись оборвалась)."""
    response = []
    if not os.path.isdir(folder):
        return response
    for name in sorted(os.listdir(folder)):
        m = _KIND_POINTS.match(name)
        if not m:
            continue
        path = os.path.join(folder, name)
        try:
            with open(path, encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        npz = path[:-5] + '.npz'
        if os.path.exists(npz):
            response.append((int(m.group(1)), npz, d))
    return response


def _write_point(folder, row, state, hash, params, why):
    os.makedirs(folder, exist_ok=True)
    name = os.path.join(folder, f'строка_{row:07d}')
    part = f'{name}.часть{os.getpid()}'     # у каждого процесса своя: два повтора в одну папку не мешают друг другу
    save_state(part + '.npz', state)
    os.replace(part + '.npz', name + '.npz')
    with open(part + '.json', 'w', encoding='utf-8') as f:
        json.dump({'строка': row, 'хеш': hash, 'параметры': params, 'почему': why}, f, ensure_ascii=False)
    os.replace(part + '.json', name + '.json')


def _size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _remove(path):
    for path in path:            # `.json` первым: без него `.npz` уже не точка, даже если удаление оборвётся
        try:
            os.remove(path)
        except FileNotFoundError:        # убрал другой повтор той же папки
            pass


def _live(pid):
    if not pid or int(pid) == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _gb(bytes):
    return f'{bytes / 1e9:.2f} ГБ' if bytes >= 1e9 else f'{bytes / 1e6:.1f} МБ'


def _min(with_):
    with_ = int(round(with_))
    return f'{with_ // 60}:{with_ % 60:02d}'


def _segments(lines):
    """1г: номер участка журнала после каждой строки — сколько раз до неё (включительно) сменился этап пометкой, как
    этап считает `повторить` (`тек`): ответ[k] — участок после строки k, ответ[0] = 0."""
    response, stage, subarea = [0], None, 0
    for row in lines:
        if '"mark"' in row:
            try:
                z = json.loads(row)
            except ValueError:
                z = {}
            if isinstance(z, dict) and 'mark' in z and str(z.get('stage', stage)) != str(stage):
                stage, subarea = z.get('stage', stage), subarea + 1
        response.append(subarea)
    return response


def collect(lines, folder, width=4800, height=3000, ground=(150, 141, 126), scale=1.0, preset=None, seed=0,
            selection=None, resume=False, every=None, every_with=300.0, print=None, times_in_with=30.0,
            limit=LIMIT_POINTS_GB, cores=None):
    """Е-5, А-8: повтор с контрольными точками в папке. Точка — состояние повтора после строки (как `--save`) и хеш
    журнала по эту строку; пишется перед каждым новым этапом (строка с другим "stage", обычно пометка 1а), каждые
    `каждые` мазков (None — нет), каждые `каждые_с` секунд счёта (None — нет) и после последней строки.
    продолжить=True — начать с ближайшей точки, начало журнала по которую не менялось: после обрыва — с последней,
    после правки строки — с последней точки до неё (правка в этапе 5 не пересчитывает этапы 1–4); итог тот же бит в бит,
    что повтор с начала. Параметры повтора (масштаб, сид, пресет, холст, отбор, версия физики) входят в точку: с другими
    точка не берётся.
    Что держится в папке (1г): точки этапов — последняя точка каждого этапа (это точка «перед этапом» следующего, у
    последнего этапа — самая свежая) — и промежуточные лесенкой: ближе к концу журнала гуще, дальше реже (зазор между
    соседними точками растёт с тем, как далеко он от конца), так что правка недавнего мазка внутри длинного этапа
    начинается с точки рядом с ней. Промежуточных — не больше `ЛЕСЕНКА` (12). Уходят точки, с которыми журнал
    разошёлся, и обрывки записи. предел — ГБ на папку (None — без предела, держит только `ЛЕСЕНКА`): перед записью
    (место под самую большую) и после неё, пока папка больше предела, уходит промежуточная точка, у которой зазор
    между её соседями меньше всего к их дали от конца; промежуточных нет — точка этапа, без которой соседние стоят
    ближе всего (последняя — только если она одна больше предела). О каждом удалении — строка в `печать` (нет — в
    stderr).
    печать — куда писать прогресс (строка, мазки, этап, прошло и осталось), раз в `раз_в_с` секунд, и о точках.
    Ответ — (холст, мазков положено в этот заход, состояние, с какой строки начат заход)."""
    lines = list(lines)
    hashes = _hashes_lines(lines)
    subarea = _segments(lines)
    tell = print or (lambda s: print(s, file=sys.stderr, flush=True))
    limit_b = None if limit is None else float(limit) * 1e9
    params = {'physics': FAndZAndBrushA, 'ширина': width, 'высота': height, 'грунт': [float(v) for v in ground],
                 'масштаб': float(scale), 'пресет': preset, 'сид': int(seed),
                 'отбор': None if selection is None else [None if v is None else str(v) for v in selection]}
    strokes = [0]                             # мазков в строках по k-ю: для оставшегося времени и разрежения точек
    for row in lines:
        strokes.append(strokes[-1] + ('"ctrl"' in row or '"points"' in row or '"core"' in row))
    hold = {}                               # строка -> (путь .npz, почему, байт пары)
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):  # обрывки записи (убит на записи) и `.npz` без `.json` — не точки
            path = os.path.join(folder, name)
            m = _KIND_PARTS.match(name)
            if m and not _live(m.group(1)):
                tell(f'убрал обрывок записи точки {path} ({_gb(_size(path))})')
                _remove([path])
            m = _KIND_NPZ.match(name)
            if m and not os.path.exists(path[:-4] + '.json') and time.time() - os.path.getmtime(path) > 60:
                tell(f'убрал {path}: без .json это не точка ({_gb(_size(path))})')
                _remove([path])
    for row, path, d in points_folders(folder):
        pair = [path[:-4] + '.json', path]
        if d.get('параметры') == params and row < len(hashes) and d.get('хеш') == hashes[row]:
            hold[row] = (path, d.get('почему', ''), sum(_size(p) for p in pair))
        else:
            why = 'журнал до неё изменился' if d.get('параметры') == params else 'другие параметры повтора'
            tell(f'убрал точку {path} (строка {row}): {why}')
            _remove(pair)

    def remove_point(row, why):
        path, was, bytes = hold.pop(row)
        tell(f'убрал точку {path} (строка {row}, {was}, {_gb(bytes)}): {why}')
        _remove([path[:-4] + '.json', path])

    def displace(why):
        """Убрать одну точку: сначала промежуточную, где лесенка гуще всего, точки этапов — когда промежуточных нет."""
        row = sorted(hold)
        last_in = {}
        for with_ in row:
            last_in[subarea[with_]] = with_
        anchors = set(last_in.values())    # последняя в этапе: «перед этапом» следующего, у последнего — самая свежая
        end = strokes[row[-1]]
        price = []
        for i, with_ in enumerate(row):
            if with_ in anchors:
                continue
            to, it, after = strokes[row[i - 1]] if i else 0, strokes[with_], strokes[row[i + 1]]
            # без неё правка мазка между ней и следующей начнётся раньше на «она − до» мазков; правят чаще недавнее —
            # вес правки в d мазках от конца ~ 1 / (d + 1); цена — сколько мазков заново это стоит в среднем
            price.append(((it - to) * np.log((end - it + 1) / (end - after + 1)), with_, after - to, end - after))
        if price:
            _, with_, gap, gave = min(price)
            remove_point(with_, f'{why}; промежуточная: без неё соседние точки в {gap} мазках друг от друга, '
                            f'следующая — в {gave} мазках от последней')
            return
        if len(row) == 1:
            remove_point(row[0], why + ' (поднять --checkpoints-max)')
            return
        edges = [0] + row
        # без точки i соседние оставшиеся стоят на мазков[следующей] − мазков[прежней]: убрать ту, где это меньше
        price = [(strokes[edges[i + 2]] - strokes[edges[i]], row[i]) for i in range(len(row) - 1)]
        remove_point(min(price)[1], why + '; промежуточных не осталось — точка этапа')

    def tidy(beyond=0):
        """Лесенка промежуточных точек и предел папки; сверх — байт, которые вот-вот запишутся."""
        while len(hold) - len({subarea[with_] for with_ in hold}) > LEWithENBrushA:
            displace(f'промежуточных точек больше {LEWithENBrushA}')
        if limit_b is None:
            return
        while hold and sum(v[2] for v in hold.values()) + beyond > limit_b:
            if len(hold) == 1:
                displace(f'под новую точку места в пределе {_gb(limit_b)} нет' if beyond else
                          f'одна больше предела {_gb(limit_b)}')
                break
            displace(f'папка больше предела {_gb(limit_b)}')

    tidy()
    state, start = {}, 1
    if resume and hold:
        row = max(hold)
        path = hold[row][0]
        t = time.perf_counter()
        state = load_file_state(path)
        start = row + 1
        if print:
            print(f'продолжаю с точки {path} (строка {row}), загрузка {time.perf_counter() - t:.1f} с')
    elif print and resume:
        print(f'точек под этот журнал в {folder} нет — с начала')
    written = set()
    state['рамки'] = {k: v for k, v in _read_frames(folder, hashes).items() if k < start}   # К-3: рамки мазков
    total, already = len(lines), strokes[start - 1]
    t0 = last_t = progress_t = time.perf_counter()
    last_n = 0

    def lay_point(laid, why):
        if laid in hold or laid in written:
            return
        t = time.perf_counter()
        tidy(max((v[2] for v in hold.values()), default=0))
        state.get('_досчитать', lambda: None)()              # Е-1: мазки, что ещё считаются на ядрах, — в холст
        _write_point(folder, laid, state, hashes[laid], params, why)
        path = os.path.join(folder, f'строка_{laid:07d}.npz')
        written.add(laid)
        hold[laid] = (path, why, _size(path) + _size(path[:-4] + '.json'))
        if print:
            print(f'точка: строка {laid} ({why}), {_gb(hold[laid][2])}, запись {time.perf_counter() - t:.1f} с')
        tidy()

    def point(laid, z, current, n):
        nonlocal last_t, last_n, progress_t
        here = time.perf_counter()
        why = None
        if 'mark' in z and str(z.get('stage', current['stage'])) != str(current['stage']):
            why = f'перед этапом {z["stage"]}'
        elif every and n - last_n >= every:
            why = f'{n - last_n} мазков'
        elif every_with is not None and here - last_t >= every_with:
            why = f'{here - last_t:.0f} с счёта'
        if why and n > last_n and laid >= start:
            lay_point(laid, why)
            last_t, last_n = time.perf_counter(), n
        if print and here - progress_t >= times_in_with:
            progress_t = here
            done = strokes[laid] - already
            left = strokes[-1] - strokes[laid]
            passed = here - t0
            score = f', осталось ~{_min(passed / done * left)}' if done > 0 else ''
            print(f'строка {laid} из {total} ({100 * laid / max(total, 1):.0f} %), мазков {strokes[laid]} из '
                   f'{strokes[-1]}, этап {current["stage"]}; прошло {_min(passed)}{score}')

    canvas, n = repeat(lines, width, height, ground, scale, preset, seed, state=state, start=start,
                     selection=selection, point=point, cores=cores)
    end = state['строка']
    if end >= start:
        lay_point(end, 'конец журнала')
    _write_frames(folder, hashes, state.pop('рамки'))
    return canvas, n, state, start


RAMBrushAnd = 'рамки.json'


def _read_frames(folder, hashes):
    """К-3: рамки мазков из `<папка>/рамки.json` — {строка: [рамка куска или None, …]}, только строк, начало журнала
    по которые то же (рамка строки зависит лишь от строк до неё и её самой)."""
    try:
        with open(os.path.join(folder, RAMBrushAnd), encoding='utf-8') as f:
            d = json.load(f)
    except (OSError, ValueError):
        return {}
    return {int(k): v[1] for k, v in d.get('строки', {}).items() if int(k) < len(hashes) and hashes[int(k)][:16] == v[0]}


def _write_frames(folder, hashes, frames):
    """К-3: рамки мазков повтора — в `<папка>/рамки.json` с хешем начала журнала по каждую строку."""
    os.makedirs(folder, exist_ok=True)
    d = {'строки': {str(k): [hashes[k][:16], v] for k, v in sorted(frames.items()) if k is not None and k < len(hashes)}}
    part = os.path.join(folder, f'{RAMBrushAnd}.часть{os.getpid()}')
    with open(part, 'w', encoding='utf-8') as f:
        json.dump(d, f, separators=(',', ':'))
    os.replace(part, os.path.join(folder, RAMBrushAnd))


def _save_how_point(path, state, folder):
    """`--save` (1г): если точка той же строки лежит в папке точек — это тот же файл байт в байт, и `--save` становится
    жёсткой ссылкой на неё, а не вторым файлом того же объёма. Ссылка встаёт на место целиком; файл точки и файл `--save`
    потом живут порознь: удаление точки `--save` не трогает, запись поверх `--save` (и `Холст.сохранить`) ссылку рвёт, а
    точку не переписывает. Другой диск или ссылки нельзя — пишется сам файл."""
    if folder is not None:
        point = os.path.join(folder, f'строка_{int(state["строка"]):07d}.npz')
        if os.path.exists(point) and os.path.exists(point[:-4] + '.json'):
            part = f'{path}.часть{os.getpid()}'
            try:
                os.link(point, part)
                os.replace(part, path)
                return
            except OSError:
                if os.path.exists(part):
                    os.remove(part)
    save_state(path, state)


def _row_draft(scale, px_on_mm):
    """М-7: строка итога повтора в масштабе < 1 — сколько мазков мягкой и узкой кисти (`мм.держит_px`) этого прогона
    легли ниже настройки физики; при масштабе 1 (и больше) — None."""
    if scale >= 1.0:
        return None
    m, at = _NAndZhE.get('мягкая', 0), _NAndZhE.get('узкая', 0)
    p = mm_.RResponseDNResponseY * scale if px_on_mm is None else float(px_on_mm)
    from .hints import _mn
    return (f'черновик ×{scale:g} ({p:g} px/мм, физика настроена на {mm_.RResponseDNResponseY:g}): ниже настройки легли '
            f'{_mn(m, "мазок", "мазка", "мазков")} мягкой и {at} узкой кисти (уже {mm_.NARROW_PX:g} px здесь) — воздух и тонкое '
            f'судить по 1:1')


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m кисть.повтор', description='Повтор журнала мазков движком «Кисть».')
    p.add_argument('журнал', help='файл .jsonl, «-» — стандартный ввод')
    p.add_argument('--out', default=None, help='готовая картинка PNG 16 бит (нужна, кроме --list)')
    p.add_argument('--stage', default=None, help='только мазки этапа ("stage"; без своего поля — от пометки "mark" выше)')
    p.add_argument('--object', dest='объект', default=None, help='только мазки объекта ("object")')
    p.add_argument('--list', dest='список', action='store_true',
                   help='не класть, а напечатать отобранные строки (мазки, снятия, пометки, заметки) с номерами')
    p.add_argument('--width', type=int, default=4800)
    p.add_argument('--height', type=int, default=3000)
    p.add_argument('--ground', type=int, nargs=3, default=(150, 141, 126), metavar=('R', 'G', 'B'))
    p.add_argument('--scale', type=float, default=None, help='множитель координат и размеров журнала (без ключа — 1; при --from — как в сохранённом)')
    p.add_argument('--preset', default=None, help='пресет для всех мазков (по умолчанию плоская щетина по размеру)')
    p.add_argument('--seed', type=int, default=None, help='сид журнала (без ключа — 0; при --from — как в сохранённом)')
    p.add_argument('--no-light', action='store_true', help='без света по рельефу')
    p.add_argument('--light', type=float, default=None, help='сила света (по умолчанию сдержанная)')
    p.add_argument('--shadow', type=float, nargs='?', const=True, default=0.0,
                   help='падающая тень по рельефу (п. 20): без числа — рабочая сила, число 0–1 — сила; без ключа тени нет')
    p.add_argument('--save', default=None, help='сохранить состояние .npz: холст, кисти, масштаб, сид, номер последней строки')
    p.add_argument('--from', dest='из', default=None, metavar='ХОЛСТ.npz',
                   help='дорисовка (К3): взять состояние из файла --save и класть журнал поверх него; размер и грунт холста — из файла')
    p.add_argument('--start', type=int, default=None, metavar='N',
                   help='с какой строки журнала класть (номер строки файла, с 1); только с --from; без ключа — со строки после сохранённой')
    p.add_argument('--layers', default=None, help='папка: слои RGBA, карта толщины и слой света отдельными PNG')
    p.add_argument('--checkpoints', default=None, metavar='ПАПКА',
                   help='куда писать контрольные точки (Е-5); без ключа — рядом с --out: <имя>.точки/')
    p.add_argument('--no-checkpoints', dest='без_точек', action='store_true', help='не писать контрольных точек')
    p.add_argument('--every', type=int, default=None, metavar='N', help='точка каждые N мазков (без ключа — по времени)')
    p.add_argument('--every-sec', dest='каждые_с', type=float, default=300.0, metavar='С',
                   help='точка каждые С секунд счёта (300); 0 — не по времени')
    p.add_argument('--checkpoints-max', dest='предел', type=float, default=LIMIT_POINTS_GB, metavar='ГБ',
                   help=f'предел папки точек, ГБ ({LIMIT_POINTS_GB:g}); лишнее уходит само, о каждом удалении — строка; 0 — без предела')
    p.add_argument('--cores', dest='ядра', type=int, default=None, metavar='N',
                   help='Е-1: сколько процессов считают мазки (итог бит в бит как на одном); без ключа — KIST_CORES или ядра машины, не больше 16; 1 — одно ядро')
    p.add_argument('--resume', action='store_true',
                   help='продолжить с ближайшей точки, по которую журнал не менялся: после обрыва и после правки строки')
    a = p.parse_args(argv)
    _NAndZhE.clear()
    if a.spisok:
        f = sys.stdin if a.journal == '-' else open(a.journal, encoding='utf-8')
        try:
            selected = select(f, a.stage, a.object)
        finally:
            if f is not sys.stdin:
                f.close()
        for number, z in selected:
            if 'core' in z or 'points' in z:
                z = {brush: v for brush, v in z.items() if brush not in ('core', 'points')} | {'точек': len(z.get('core', z.get('points')))}
            print(number, json.dumps(z, ensure_ascii=False))
        print(f'{sum(1 for _, z in selected if any(brush in z for brush in ("ctrl", "points", "core")))} мазков', file=sys.stderr)
        return 0
    if a.out is None:
        p.error('нужен --out (или --list — напечатать строки)')
    if a.start is not None and a.from_ is None:
        p.error('--start без --from: класть не с начала можно только поверх сохранённого состояния')
    if a.start is not None and a.start < 1:
        p.error('--start — номер строки журнала, от 1')
    if a.resume and a.from_ is not None:
        p.error('--resume и --from — разные пути: --resume берёт точку из папки точек, --from — файл --save')
    if a.from_ is not None and a.checkpoints is not None:
        p.error('--checkpoints и --from: точки пишет повтор с начала журнала; с --from — только --save')
    if a.resume and a.without_points:
        p.error('--resume без точек: продолжать не с чего (убрать --no-checkpoints)')
    if a.every is not None and a.every < 1:
        p.error('--every — число мазков, от 1')
    if a.limit < 0:
        p.error('--checkpoints-max — ГБ, от 0 (0 — без предела)')
    state, start = {}, 1
    if a.from_ is not None:
        t = time.perf_counter()
        try:
            state = load_file_state(a.from_)
        except (OSError, ValueError, KeyError) as e:
            p.error(f'--from: {e}')
        for option, name, given in (('--scale', 'масштаб', a.scale), ('--seed', 'сид', a.seed), ('--preset', 'пресет', a.preset)):
            if given is not None and given != state[name]:
                p.error(f'{option} {given} — а сохранённое положено с {state[name]}: дорисовка идёт с тем же, ключ можно не давать')
        a.scale, a.seed, a.preset = state['масштаб'], state['сид'], state['пресет']
        start = state['строка'] + 1 if a.start is None else a.start
        print(f'состояние {a.from_}: строка {state["строка"]}, загрузка {time.perf_counter() - t:.1f} с; кладу со строки {start}',
              file=sys.stderr)
    a.scale = 1.0 if a.scale is None else a.scale
    a.seed = 0 if a.seed is None else a.seed
    t = time.perf_counter()
    f = sys.stdin if a.journal == '-' else open(a.journal, encoding='utf-8')
    selection = None if a.stage is None and a.object is None else (a.stage, a.object)
    print = lambda s: print(s, file=sys.stderr, flush=True)
    try:
        if a.from_ is None and not a.without_points:     # Е-5, А-8: повтор с контрольными точками
            folder = a.checkpoints or os.path.splitext(a.out)[0] + '.точки'
            canvas, n, state, _ = collect(f, folder, a.width, a.height, tuple(a.ground), a.scale, a.preset, a.seed,
                                         selection=selection, resume=a.resume, every=a.every, cores=a.cores,
                                         every_with=a.every_with or None, print=print, limit=a.limit or None)
        else:
            folder = None
            canvas, n = repeat(f, a.width, a.height, tuple(a.ground), a.scale, a.preset, a.seed,
                             print=print, state=state, start=start, selection=selection, cores=a.cores)
    finally:
        if f is not sys.stdin:
            f.close()
    dt = time.perf_counter() - t
    canvas.png(a.out, light=not a.no_light, strength=a.light, shadow=a.shadow)
    if a.save:
        _save_how_point(a.save, state, folder)
    if a.layers:
        os.makedirs(a.layers, exist_ok=True)
        for name in canvas.layers:
            canvas.png_layer(os.path.join(a.layers, f'слой_{name}.png'), name)
        canvas.png_thicknesses(os.path.join(a.layers, 'толщина.png'))
        canvas.png_light(os.path.join(a.layers, 'свет.png'), strength=a.light, shadow=a.shadow)
    print(f'{n} мазков за {dt:.1f} с ({n / max(dt, 1e-9):.0f} в секунду) -> {a.out}', file=sys.stderr)
    row = _row_draft(a.scale, canvas.px_on_mm)
    if row:
        print(row, file=sys.stderr)
    if canvas.empty:     # v4, 1д: мазки без следа — счётом в итоге (слово `ПустойМазок` — на первый каждого вида)
        print('без следа: ' + ', '.join(f'{kind} — {k}' for kind, k in canvas.empty.items()), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
