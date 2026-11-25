# READING VIEW of движок/кисть/калибровка.py with names in English. Not runnable; edit the original.
"""5е, 5е-2 (В-1, В-2, В-7): калибровка покрытия — таблица одиночного хода по сухому грунту и обратная функция.

Таблица `калибровка.json` рядом с модулем: пресет × ширина (мм) × нагрузка × нажим × вид краски × разбавитель (0, 0,3, 0,5, 0,7, 1) →
покрытие (1 − просвет грунта, как у `х.просвет`), толщина плёнки (единицы h, как `Слой.толщина`), доля полосы с краской
(1 − просветы: место, где покрытие не ниже половины тела). Узел — прямой ход 45 мм на чистом холсте 400×200 при 8 px/мм,
грунт (166, 146, 129), путь `путь_с_нажимом` со входом и сходом в полширины; меряется полоса посередине: средняя
половина хода по длине (22,5 мм), внутренние 80 % ширины следа (`о.ширина`) поперёк. Число узла — среднее восьми сидов (кисть уже 2,5 мм — 32: её ход по сиду гуляет на 20–30 %); `разброс` — σ
одиночного хода по сиду.

Цвет краски (5е-2). Кроющая густая кроет телом — её покрытие от цвета не зависит. Прозрачная, полукроющая и любая
краска с разбавителем 1 пропускают свет сквозь плёнку (`пропуск` слоя = цвет плёнки в степени её толщины), и светлая
пропускает больше тёмной: при той же нагрузке она кроет слабее в 2–5 раз. Укладка хода (покрытие и толщина слоя, тон
прядей) от цвета не зависит — бит в бит; поэтому узел снимается одной краской (70, 80, 110), а в таблице, кроме её
покрытия, лежит `сквозь`: просвет того же хода, пересчитанный по пикселям формулой `х.просвет` для серых красок с
линейным светом exp(s), s — `СЕТКА`. Просвет краски цвета (r, g, b) — сумма по каналам с весом яркости значений этой
кривой при s = ln канала. У жидкой краски (разбавитель 1) тон прядей сходится пигментной смесью, и светлая краска от
этого темнеет, а тёмная нет: узел жидкой повторяется серыми красками `УРОВНИ` (сиды `СИДЫ_СДВИГА`), в таблице — `сдвиг`
ln света светлого канала. Не кроющей краске `нагрузка_для` и `по_таблице` без `цвет=` отвечают ошибкой.

Между узлами — интерполяция: покрытие и доля по log нагрузки (покрытие не кроющей — по log оптической глубины
плёнки, `_глубина`), толщина по самой нагрузке; по log ширины, по нажиму. Обратная — `нагрузка_для`: по кривой
покрытия от нагрузки при ширине, нажиме, виде и разбавителе кисти; вне таблицы и недостижимое — `ValueError` словами
с ближайшим достижимым.

Пересчёт — `python3 -m кисть.калибровка --пересчитать [--процессов 16]`; в таблице — `хеш` физики, с которой она снята
(код укладки, чисел, кисти, путей и замера узла без комментариев и строк описания, и пресеты). `--сверить` — хеш и
живые ходы на узлах против таблицы. Тест `тесты/test_калибровка_5е.py` падает, если таблица разошлась с ядром.

Таблица — по сухому одиночному ходу на чистом грунте. По сырому, внахлёст, сериями — не она (В-3): там пробник и
`х.перебор`. Холст, кисти и журнал листа таблица не трогает — только читает свой файл.

Разбавитель (Долг-т). Покрытие падает с разбавителем гладко (кривые по 11 точкам, 7 пресетов), кроме начала: любой
разбавитель больше 0 включает текучесть (`ровно`), и у лайнера покрытие кроющей сразу падает на 0,1. Между узлами —
прямой по самому покрытию (толщине, доле, σ). Узлы 0 и 1 — в `калибровка.json`, остальные — по файлу на узел
(`калибровка_р0.3.json` …): файл больше 5 МБ в git не идёт.
"""
import ast
import hashlib
import inspect
import json
import math
import os
import sys
import textwrap
import warnings

import numpy as np

_PABrushET = os.path.dirname(os.path.abspath(__file__))      # кисть/
FAYL = os.path.join(_PABrushET, 'калибровка.json')
_PREWithETY = os.path.join(os.path.dirname(_PABrushET), 'пресеты')

# ----------------------------------------------------------------- стенд узла
PX_ON_MM = 8.0
GRAtNT = (166, 146, 129)
CanvasResponseLWithT = (400, 200)
CanvasResponseD = (20, 380)                  # x хода, px (45 мм); y — середина холста
ResponseBrushNResponse = (110, 290)                # x полосы замера: средняя половина хода
PResponsePERYoBrush = 0.8                    # доля ширины следа поперёк, внутренняя
TsInET = (70, 80, 110)             # краска узла: тёмная против светлого грунта
WithAndDY = (1, 2, 3, 4, 5, 6, 7, 8)
SEEDS_NARROW = tuple(range(1, 33))   # кисть уже 2,5 мм (20 px): ход по сиду гуляет на 20–30 % — среднее по 32 ходам
NARROW_MM = 2.5
# 5е-2: ln линейного света серой краски, для которой в узле хранится просвет (`сквозь`): от белой (0) до самой тёмной
# (ln `краска._ПОЛ`); чаще у светлых — там просвет не кроющей меняется быстрее всего
WithETBrushA = (0.0, -0.03, -0.08, -0.2, -0.4, -0.8, -1.4, -2.2, -3.2, -4.82831)

# 5е-2, жидкая краска: ln света серых красок, которыми ход узла повторяется ради сдвига тона светлой краски (темнее
# exp(−0,8) сдвига нет), и сиды, на которых он меряется
AtRResponseInNAnd = (0.0, -0.08, -0.2, -0.4)
WITHOUT_SHIFT = -0.8
SEEDS_SHIFT = (1, 2)

# ----------------------------------------------------------------- сетка узлов
InAndDY = ('кроющая', 'полукроющая', 'прозрачная')
RAZBAInAndTELAnd = (0.0, 1.0, 0.3, 0.5, 0.7)     # порядок оси в таблице: первые два — в главном файле, дальше — по файлу на узел
IN_MAIN = 2
NAZhAndMY = (0.3, 0.45, 0.6, 0.8)
NAGRAtZBrushAnd = (0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.28, 0.4, 0.55, 0.7, 0.9)
WIDTHS_MM = {                     # по пресету, мм (при 8 px/мм ×8 — px). Узкая кисть с остриём (`остриё` 1: мягкая,
    # круглая, плоская_узкая) от 12 до 16 px переходит от острого кончика к плашмя (`числа._ОСТРАЯ`): покрытие там
    # встаёт ступенью вдвое — узлы через 1 px; в остальном покрытие от ширины почти не зависит — узлы реже
    'плоская': (2.0, 3.0, 4.0, 6.0, 9.0, 12.5, 18.0),
    'мягкая': (1.0, 1.5, 1.625, 1.75, 1.875, 2.0, 3.0, 5.0, 9.0, 15.0),
    'плоская_узкая': (0.75, 1.0, 1.25, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5),
    'круглая': (1.0, 1.25, 1.5, 1.625, 1.75, 1.875, 2.0, 2.5, 3.5, 5.0, 7.0),
    'сухая': (2.0, 3.0, 4.5, 6.0, 9.0, 12.0),
    'лайнер': (0.25, 0.375, 0.5),
    'мастихин': (2.0, 3.0, 5.5, 8.0),
}
PResponseLYa = ('покрытие', 'толщина', 'доля', 'разброс', 'разброс_толщины')
ZAMER = ('покрытие', 'толщина', 'доля')
FIELDS_COLORS = ('сквозь', 'разброс_сквозь', 'сдвиг')   # 5е-2: по узлу — списки: просвет и его σ по сиду на точках `СЕТКА`; сдвиг тона жидкой на `УРОВНИ`

# файлы, от которых зависит одиночный ход по сухому (хеш — по их коду без комментариев и строк описания)
_FAndZAndBrushA = ('укладка.py', 'числа.py', 'инструмент.py', 'кисть.py', 'пряди.py', 'мягкая.py', 'борозда.py', 'краска.py',
           'мм.py', 'путь.py', 'срастание.py', 'зерно.py', 'просвет.py', 'нож.py')


# ----------------------------------------------------------------- замер узла
def _film(pk, tsv, pr, paint):
    """5е-2: что в полосе хода от цвета краски не зависит — (голое место 1 − покрытие, показатель плёнки e, ln тона).
    пк, цв (линейный), пр — покрытие, цвет и пропуск слоя, краска — линейный цвет, которым ход положен. Плёнка
    пропускает `цвет ** e` (`краска.пропуск_плёнки`), e — по толщине, один на все каналы; цвет плёнки в пикселе —
    краска × тон пряди. Кроющая густая (пр None) света сквозь плёнку не пускает: e = 0."""
    from .paint import _PResponseL
    bare = 1.0 - pk
    if pr is None:
        return bare, np.zeros(pk.shape), np.zeros(tsv.shape)
    lts = np.log(np.clip(tsv, _PResponseL, 1.0)).sum(axis=-1)
    e = np.where(lts < -1e-9, np.log(np.maximum(pr, 1e-300)).sum(axis=-1) / np.minimum(lts, -1e-9), 0.0)
    return bare, e, np.log(np.maximum(tsv, 1e-12) / np.maximum(np.asarray(paint, np.float64), 1e-12))


def _see_through(film, s):
    """5е-2: просвет полосы хода (`_плёнка`) для краски с ln линейного света s (число — серая, тройка — по каналам) —
    той же формулой, что `х.просвет`: голое место × яркость пропуска плёнки."""
    from .paint import _PResponseL
    from .gap import YaRBrushResponseWithT
    bare, e, l_tone = film
    return float((bare * (np.exp(e[..., None] * np.clip(np.asarray(s, np.float64) + l_tone, math.log(_PResponseL), 0.0))
                          @ YaRBrushResponseWithT.astype(np.float64))).mean())


def _sweep_node(preset, width_mm, load, pressure, kind, thinner, seed, color):
    """Один ход стенда краской `цвет` (sRGB) на своём чистом холсте: None (не лёг) или словарь — 'покрытие' (1 − просвет
    `х.просвет`), 'толщина', 'доля' и 'плёнка' (`_плёнка` полосы замера)."""
    from .canvas import Canvas
    from .brush import Brush, path_with_pressure
    from .paint import Paint, in_linear
    sh = float(width_mm) * PX_ON_MM
    canvas = Canvas(CanvasResponseLWithT[0], CanvasResponseLWithT[1], ground=GRAtNT, px_on_mm=PX_ON_MM, cores=1, hints=False)
    brush = Brush.from_file(os.path.join(_PREWithETY, preset + '.json'), width=sh)
    yc = CanvasResponseLWithT[1] // 2
    points = path_with_pressure([(CanvasResponseD[0], yc), (CanvasResponseD[1], yc)], sh, pressure=float(pressure), entry=sh / 2, exit=sh / 2, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        response = canvas.stroke(points, Paint(color, kind), brush, layer='узел', seed=seed, load=float(load),
                   thinner=float(thinner) if thinner else None)
    if not response:
        return None
    w = response.width or sh
    y0 = max(0, int(round(yc - PResponsePERYoBrush / 2 * w)))
    y1 = min(CanvasResponseLWithT[1], max(y0 + 1, int(round(yc + PResponsePERYoBrush / 2 * w)) + 1))
    p = canvas.gap((ResponseBrushNResponse[0], y0, ResponseBrushNResponse[1], y1), holes=0)
    sl = canvas.layers['узел']
    pk = np.asarray(sl.coverage[y0:y1, ResponseBrushNResponse[0]:ResponseBrushNResponse[1]], np.float64)
    tl = np.asarray(sl.thickness[y0:y1, ResponseBrushNResponse[0]:ResponseBrushNResponse[1]], np.float64)
    tsv = np.asarray(sl.color[y0:y1, ResponseBrushNResponse[0]:ResponseBrushNResponse[1]], np.float64)
    pr = None if sl.transmittance is None else np.asarray(sl.transmittance[y0:y1, ResponseBrushNResponse[0]:ResponseBrushNResponse[1]], np.float64)
    body = float(np.percentile(pk, 90))
    return {'покрытие': float(1.0 - p.summary['просвет']), 'толщина': float(tl.mean()),
            'доля': float((pk >= 0.5 * body).mean()) if body > 0 else 0.0,
            'плёнка': _film(pk, tsv, pr, in_linear(color))}


def _gray(s):
    """sRGB 0–255 серой краски с ln линейного света s."""
    from .paint import from_linear
    return tuple(float(v) for v in from_linear(np.full(3, math.exp(s))) * 255.0)


def measurement_node(preset, width_mm, load, pressure, kind, thinner, seed=1, color=TsInET, shift=False):
    """Один ход узла: покрытие, толщина, доля полосы с краской, `сквозь` (просвет по `СЕТКА`) — словарь. Холст свой,
    чистый. цвет — краска хода (таблица снята с `ЦВЕТ`; другой — живой ход для сверки). сдвиг — у жидкой краски
    (разбавитель 1) ещё и `сдвиг`: тон прядей жидкой краски сходится к общему в пигментном состоянии
    (`укладка._текучая`), и светлая краска от этого темнеет (белая — на ~10 %), а тёмная нет — этого по ходу тёмной
    краской не узнать. Поэтому ход повторяется серыми красками `УРОВНИ`, и для каждой ищется сдвиг ln света, с которым
    счёт по плёнке хода даёт живой просвет."""
    r = _sweep_node(preset, width_mm, load, pressure, kind, thinner, seed, color)
    if r is None:
        d = {'покрытие': 0.0, 'толщина': 0.0, 'доля': 0.0, 'сквозь': [1.0] * len(WithETBrushA)}
    else:
        film = r.pop('плёнка')
        d = dict(r, see_through=[_see_through(film, s) for s in WithETBrushA])
    if shift:
        d['сдвиг'] = [0.0] * len(AtRResponseInNAnd)
        for k, s in enumerate(AtRResponseInNAnd if r is not None and thinner else ()):
            zh = _sweep_node(preset, width_mm, load, pressure, kind, thinner, seed, _gray(s))
            if zh is None:
                continue
            target, lo, hi = 1.0 - zh['покрытие'], -1.0, 0.3
            for _ in range(24):                      # просвет растёт со светом краски — делением пополам
                d = 0.5 * (lo + hi)
                lo, hi = (lo, d) if _see_through(film, s + d) > target else (d, hi)
            d['сдвиг'][k] = 0.5 * (lo + hi)
    return d


def seeds_node(width_mm):
    return SEEDS_NARROW if float(width_mm) < NARROW_MM else WithAndDY


def _node(task):
    preset, sh, n, nzh, kind, r = task
    rr = [measurement_node(preset, sh, n, nzh, kind, r, seed, shift=seed in SEEDS_SHIFT) for seed in seeds_node(sh)]
    d = {p: float(np.mean([r[p] for r in rr])) for p in ZAMER}
    d['разброс'] = float(np.std([r['покрытие'] for r in rr]))            # одиночный ход по сиду: σ покрытия
    d['разброс_толщины'] = float(np.std([r['толщина'] for r in rr]))
    through = np.array([r['сквозь'] for r in rr])
    d['сквозь'], d['разброс_сквозь'] = [float(v) for v in through.mean(axis=0)], [float(v) for v in through.std(axis=0)]
    d['сдвиг'] = [float(v) for v in np.mean([r['сдвиг'] for r in rr if 'сдвиг' in r], axis=0)]
    return d


# ----------------------------------------------------------------- хеш физики
def _code_without_words(text, without_import=False):
    """Код без комментариев и строк описания: правка слов хеш не меняет, правка числа или строки кода — меняет.
    без_импорта — и без строк import (у замера узла: что он зовёт, хешируется своими файлами)."""
    tree = ast.parse(text)
    if without_import:
        for node in ast.walk(tree):
            body = getattr(node, 'body', None)
            if isinstance(body, list):
                node.body = [at for at in body if not isinstance(at, (ast.Import, ast.ImportFrom))] or [ast.Pass()]
    for node in ast.walk(tree):
        body = getattr(node, 'body', None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], 'value', None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return ast.dump(tree, include_attributes=False)


def hash_physics(folder=None):
    """sha256 (первые 16 знаков) кода физики хода, замера узла, сетки и пресетов; папка — пакет `кисть/` (своя копия —
    для сторожа), пресеты — рядом с ней."""
    folder = folder or _PABrushET
    h = hashlib.sha256()
    for name in _FAndZAndBrushA:
        with open(os.path.join(folder, name), encoding='utf-8') as f:
            h.update(name.encode() + b'\0' + _code_without_words(f.read()).encode() + b'\0')
    for f in (measurement_node, _sweep_node, _film, _see_through, _gray, _node):
        h.update(_code_without_words(textwrap.dedent(inspect.getsource(f)), without_import=True).encode())
    h.update(json.dumps([PX_ON_MM, GRAtNT, CanvasResponseLWithT, CanvasResponseD, ResponseBrushNResponse, PResponsePERYoBrush, TsInET, WithAndDY, SEEDS_NARROW, NARROW_MM, WithETBrushA, AtRResponseInNAnd, WITHOUT_SHIFT, SEEDS_SHIFT]).encode())
    presets = os.path.join(os.path.dirname(folder), 'пресеты')
    for name in sorted(WIDTHS_MM):
        with open(os.path.join(presets, name + '.json'), encoding='utf-8') as f:
            h.update(json.dumps(json.load(f), sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- пересчёт
def _tasks():
    for preset, shsh in WIDTHS_MM.items():
        for kind in InAndDY:
            for r in RAZBAInAndTELAnd:
                for sh in shsh:
                    for nzh in NAZhAndMY:
                        for n in NAGRAtZBrushAnd:
                            yield (preset, sh, n, nzh, kind, r)


def part(file, r):
    """Имя файла узла разбавителя `р` при главном файле таблицы."""
    root, tail = os.path.splitext(file)
    return f'{root}_р{r:g}{tail}'


def write_part(t, file, fields):
    """Таблица т (поля пресетов — [вид][разбавитель]…) — в главный файл (узлы `В_ГЛАВНОМ`) и по файлу на прочие узлы."""
    for k in [None] + list(range(IN_MAIN, len(RAZBAInAndTELAnd))):
        sr = slice(0, IN_MAIN) if k is None else slice(k, k + 1)
        presets = {p: {'ширины_мм': d['ширины_мм'], **{field: [kind[sr] for kind in d[field]] for field in fields}}
                   for p, d in t['пресеты'].items()}
        d = {**t, 'пресеты': presets} if k is None else {'хеш': t['хеш'], 'разбавитель': RAZBAInAndTELAnd[k], 'пресеты': presets}
        with open(file if k is None else part(file, RAZBAInAndTELAnd[k]), 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, separators=(',', ':'))


def read_parts(file, fields):
    """Таблица главного файла с узлами разбавителя из файлов-частей; нет части или она снята с другой физики — узла нет
    (`т['разбавители']` — узлы, что есть). Поля пресетов — массивы."""
    with open(file, encoding='utf-8') as f:
        t = json.load(f)
    pieces = [t]
    t['разбавители'] = list(RAZBAInAndTELAnd[:IN_MAIN])
    for r in RAZBAInAndTELAnd[IN_MAIN:]:
        if not os.path.exists(part(file, r)):
            break
        with open(part(file, r), encoding='utf-8') as f:
            d = json.load(f)
        if d['хеш'] != t['хеш']:
            break
        t['разбавители'].append(r)
        pieces.append(d)
    for name, pr in t['пресеты'].items():
        for p in fields(t):
            pr[p] = np.concatenate([np.asarray(d['пресеты'][name][p], np.float64) for d in pieces], axis=1)
    return t


def recompute(processes=16, file=FAYL, print=print):
    """Снять всю таблицу заново (в процессах, не больше 16) и записать `файл`. Отвечает секундами счёта."""
    import multiprocessing as mp
    import time
    processes = max(1, min(16, int(processes)))
    canvas = hash_physics()                       # до счёта: исходник модуля, правленный во время счёта, хеша не собьёт
    tasks = list(_tasks())
    t = time.time()
    with mp.get_context('fork').Pool(processes) as pool:
        numbers = pool.map(_node, tasks, chunksize=8)
    seconds = time.time() - t
    by = {(z[0], z[1], z[2], z[3], z[4], z[5]): h for z, h in zip(tasks, numbers)}
    presets = {}
    for preset, shsh in WIDTHS_MM.items():
        d = {'ширины_мм': list(shsh)}
        for p in PResponseLYa:     # [вид][разбавитель][ширина][нажим][нагрузка]
            d[p] = [[[[[round(by[(preset, sh, n, nzh, kind, r)][p], 5) for n in NAGRAtZBrushAnd] for nzh in NAZhAndMY]
                      for sh in shsh] for r in RAZBAInAndTELAnd] for kind in InAndDY]
        for p in FIELDS_COLORS:     # то же — списками
            d[p] = [[[[[[round(v, 4) for v in by[(preset, sh, n, nzh, kind, r)][p]] for n in NAGRAtZBrushAnd] for nzh in NAZhAndMY]
                      for sh in shsh] for r in RAZBAInAndTELAnd] for kind in InAndDY]
        presets[preset] = d
    t = {'хеш': canvas, 'узлов': len(tasks),
         'ходов': sum(len(seeds_node(z[1])) + (len(AtRResponseInNAnd) * len(SEEDS_SHIFT) if z[5] else 0) for z in tasks), 'секунд': round(seconds, 1),
         'процессов': processes,
         'стенд': {'px_на_мм': PX_ON_MM, 'грунт': GRAtNT, 'холст': CanvasResponseLWithT, 'ход_x': CanvasResponseD, 'окно_x': ResponseBrushNResponse,
                   'поперёк': PResponsePERYoBrush, 'цвет': TsInET, 'сиды': WithAndDY, 'сиды_узкой': SEEDS_NARROW, 'узкая_мм': NARROW_MM},
         'оси': {'вид': InAndDY, 'разбавитель': RAZBAInAndTELAnd, 'нажим': NAZhAndMY, 'нагрузка': NAGRAtZBrushAnd, 'сквозь': WithETBrushA, 'сдвиг': AtRResponseInNAnd},
         'поля': PResponseLYa, 'поля_цвета': FIELDS_COLORS, 'пресеты': presets}
    write_part(t, file, PResponseLYa + FIELDS_COLORS)
    print(f'калибровка: {len(tasks)} узлов, {t["ходов"]} ходов, {processes} процессов — {seconds:.0f} с → {file}')
    return seconds


# ----------------------------------------------------------------- чтение
_TABLAndTsA = {}


def table(file=FAYL):
    """Таблица из файла (раз на процесс); снятая с другой физики — предупреждение, числа те, что в файле."""
    if file not in _TABLAndTsA:
        if not os.path.exists(file):
            raise FileNotFoundError(f'калибровка: нет таблицы {file} — python3 -m кисть.калибровка --пересчитать')
        t = read_parts(file, lambda t: list(t['поля']) + list(t.get('поля_цвета', ())))
        for pr in t['пресеты'].values():
            if 'сдвиг' in pr:
                # сдвиг тона жидкой — только потемнение светлой краски (замер: 0…−0,115). Где ход почти не кроет (просвет
                # белой и серой exp(−0,4) разнится меньше 0,002), поиск сдвига упирается в свои границы: просвет от него
                # не зависит — там сдвига нет; вверх (светлее белой) сдвига не бывает
                through = pr['сквозь']
                empty = (through[..., 0] - through[..., WithETBrushA.index(-0.4)]) < 0.002
                pr['сдвиг как снят'] = pr['сдвиг']               # для сверки с живым узлом
                pr['сдвиг'] = np.where(empty[..., None], 0.0, np.clip(pr['сдвиг'], -0.2, 0.0))
        if t.get('хеш') != hash_physics():
            warnings.warn(f'калибровка: таблица снята с другой физики (хеш {t.get("хеш")}, ядро {hash_physics()}) — '
                          'числа могут разойтись с мазком; python3 -m кисть.калибровка --пересчитать', stacklevel=3)
        _TABLAndTsA[file] = t
    return _TABLAndTsA[file]


def _names_preset():
    names = {}
    for preset in WIDTHS_MM:
        with open(os.path.join(_PREWithETY, preset + '.json'), encoding='utf-8') as f:
            names[json.load(f)['имя']] = preset
    return names


def _preset_brushes(brush, px_on_mm):
    """(пресет, ширина в мм) кисти; кисть с полями не как у пресета — предупреждение с их именами."""
    from dataclasses import asdict
    from .brush import Brush
    if isinstance(brush, str):
        if brush not in WIDTHS_MM:
            raise ValueError(f'калибровка: пресет {brush!r} — в таблице {", ".join(WIDTHS_MM)}')
        raise ValueError('калибровка: нужна кисть (`Кисть.из_файла(…, ширина=…)`), а не имя пресета — ширина берётся из неё')
    preset = _names_preset().get(brush.name)
    if preset is None:
        raise ValueError(f'калибровка: кисть «{brush.name}» не из пресетов ({", ".join(WIDTHS_MM)}) — таблицы для неё нет; '
                         'нагрузку — пробником или `х.перебор`')
    if brush.width_mm is not None:
        sh = float(brush.width_mm)
    else:
        sh = float(brush.width) / float(px_on_mm or PX_ON_MM)
    sample = asdict(Brush.from_file(os.path.join(_PREWithETY, preset + '.json')))
    own = asdict(brush)
    other = sorted(p for p in sample if p not in ('ширина', 'ширина_мм', 'нагрузка') and own[p] != sample[p])
    if other:
        warnings.warn(f'калибровка: у кисти «{preset}» не как у пресета: {", ".join(other)} — таблица снята с пресета, '
                      'ход может лечь иначе; проверь пробником', stacklevel=3)
    return preset, sh


def _index(axes, v, log):
    """(i, доля) — между узлами i и i+1; v внутри [оси[0], оси[-1]]."""
    oo = np.log(axes) if log else np.asarray(axes, np.float64)
    vv = math.log(v) if log else float(v)
    if len(oo) == 1:
        return 0, 0.0
    i = int(np.clip(np.searchsorted(oo, vv) - 1, 0, len(oo) - 2))
    return i, float(np.clip((vv - oo[i]) / (oo[i + 1] - oo[i]), 0.0, 1.0))


def _in_within(name, v, axes, where):
    if not (axes[0] - 1e-9 <= v <= axes[-1] + 1e-9):
        nearest = min(max(v, axes[0]), axes[-1])
        raise ValueError(f'калибровка: {name} {v:g} вне таблицы{where} — в ней {axes[0]:g}…{axes[-1]:g}; ближайшее — {nearest:g}')


def _thinner(r):
    """Разбавитель числом 0–1 (None — густая, 0)."""
    r = 0.0 if r is None else float(r)
    if not 0.0 <= r <= 1.0:
        raise ValueError(f'калибровка: разбавитель {r:g} вне таблицы — в ней 0 (густая) … 1 (жидкая); ближайшее — {min(max(r, 0.0), 1.0):g}')
    return r


def weights_thinner(t, r):
    """[(номер узла оси разбавителя таблицы т, вес)] — прямая между соседними узлами."""
    if len(t['разбавители']) < len(RAZBAInAndTELAnd) and r not in t['разбавители']:
        raise ValueError(f'калибровка: разбавитель {r:g} — в этой таблице только {", ".join(f"{v:g}" for v in t["разбавители"])} (нет файлов-частей или они сняты с '
                         'другой физики) — пересчитать: python3 -m кисть.калибровка --пересчитать, python3 -m кисть.ряды --пересчитать')
    nodes = sorted(t['разбавители'])
    i, a = _index(nodes, r, False)
    return [(t['разбавители'].index(nodes[k]), w) for k, w in ((i, 1.0 - a), (min(i + 1, len(nodes) - 1), a)) if w > 0.0]


def _kind(kind):
    kind = 'лессировочная' if kind == 'лессировка' else kind
    if kind == 'лессировочная':
        raise ValueError("калибровка: вид 'лессировочная' — мутность 0: не кроет, покрытие 0 при любой нагрузке; "
                         'под покрытие её в таблице нет; толщину плёнки — `нагрузка_для(кисть, толщина=…, вид=\'лессировочная\')` (как у прозрачной)')
    if kind not in InAndDY:
        raise ValueError(f'калибровка: вид {kind!r} — в таблице {", ".join(InAndDY)}; мутность числом — пробником')
    return InAndDY.index(kind)


def _curve(t, preset, sh, pressure, iv, r, field):
    """Значения поля по узлам нагрузки при ширине ш, нажиме и разбавителе р — интерполяцией по ширине (log), нажиму и
    разбавителю."""
    return sum(w * _curve_node(t, preset, sh, pressure, iv, ir, field) for ir, w in weights_thinner(t, r))


def _curve_node(t, preset, sh, pressure, iv, ir, field):
    """То же в узле разбавителя номер iр."""
    pr = t['пресеты'][preset]
    shsh = pr['ширины_мм']
    _in_within('ширина', sh, shsh, f' «{preset}» (мм)')
    _in_within('нажим', pressure, NAZhAndMY, '')
    i, a = _index(shsh, sh, True)
    j, b = _index(NAZhAndMY, pressure, False)
    m = pr[field][iv, ir]
    j1 = min(j + 1, len(NAZhAndMY) - 1)
    i1 = min(i + 1, len(shsh) - 1)
    return ((1 - a) * ((1 - b) * m[i, j] + b * m[i, j1]) + a * ((1 - b) * m[i1, j] + b * m[i1, j1]))


LAndNEYNResponse = ('толщина', 'разброс_толщины')    # плёнка идёт за нагрузкой прямо — между узлами по самой нагрузке, прочее — по log


def _depth(coverage):
    """ln оптической глубины плёнки: покрытие = 1 − exp(−глубина). У не кроющей краски глубина идёт за нагрузкой
    степенью (тонкая плёнка — покрытие ∝ нагрузка^p, толстая — насыщается), и между узлами нагрузки она точнее прямой
    по log нагрузки вдвое (5е-2, замер по узлам через один)."""
    return np.log(-np.log(1.0 - np.clip(coverage, 1e-6, 1.0 - 1e-6)))


def _by_load(curve, load, field='покрытие', depth=False):
    i, a = _index(NAGRAtZBrushAnd, load, field not in LAndNEYNResponse)
    ni, ve = curve[i], curve[min(i + 1, len(NAGRAtZBrushAnd) - 1)]
    if depth and min(ni, ve) > 1e-6:
        return float(1.0 - math.exp(-math.exp((1 - a) * _depth(ni) + a * _depth(ve))))
    return float((1 - a) * ni + a * ve)


def _turbidity_zero(kind):
    if isinstance(kind, str):
        return kind in ('лессировочная', 'лессировка')
    return isinstance(kind, (int, float, np.floating)) and not isinstance(kind, bool) and float(kind) == 0.0


def _glaze(kind, color):
    """О-1, хвост Ф-1: краска мутности 0 (`'лессировочная'`, мутность 0 числом — доводом `вид` или видом `цвет`) под
    `толщина=`: толщина плёнки от мутности не зависит — берётся строка `'прозрачная'` (плоская 24 px, нагрузка 0,0189 —
    толщина 0,00567 у обеих). Под `покрытие=` — прежняя ошибка `_вид`: мутность 0 не кроет."""
    vk = getattr(color, 'вид', None)
    if kind is not None:
        if not _turbidity_zero(kind):
            return False
        if vk is not None and not _turbidity_zero(vk):
            raise ValueError(f'калибровка: вид={kind!r}, а краска цвет= — {vk!r}; оставь одно')
        return True
    return _turbidity_zero(vk)


def _kind_paints(kind, color):
    """Вид словом: довод `вид` или вид краски `цвет` (`Краска(rgb, 'прозрачная')`); оба и разные — ошибка."""
    vk = getattr(color, 'вид', None)
    vk = 'кроющая' if vk == 1.0 else vk
    if kind is None:
        return 'кроющая' if vk is None else vk
    if vk is not None and (not isinstance(vk, str) or _kind(vk) != _kind(kind)):
        raise ValueError(f'калибровка: вид={kind!r}, а краска цвет= — {vk!r}; оставь одно')
    return kind


def _color(color):
    """Цвет краски (sRGB 0–255: кортеж, `Краска`, ответ `цвет(L, C, h)`) -> ln линейного света по каналам; None — None."""
    if color is None:
        return None
    from .paint import in_linear, _PResponseL
    try:
        ts = np.asarray(tuple(color), np.float64)
    except (TypeError, ValueError):
        ts = np.zeros(0)
    if ts.shape != (3,) or not np.all(np.isfinite(ts)) or ts.min() < 0.0 or ts.max() > 255.0:
        raise ValueError(f'калибровка: цвет={color!r} — нужен sRGB (r, g, b) 0–255, как у `Краска`, или `цвет(L, C, h)`')
    return np.log(np.clip(in_linear(ts), _PResponseL, 1.0))


def _gap_colors(kr, s, log=True, shift=None):
    """Просвет краски с ln каналов s по кривым `сквозь` кр (узлы нагрузки, точки `СЕТКА`): по каналу — между точками
    сетки (просвет — по логарифму: у ровной плёнки он exp(e·s); σ — прямо), каналы — с весом яркости, как у
    `х.просвет`. сдвиг (узлы нагрузки, `УРОВНИ`) — жидкая краска: светлый канал темнее на сдвиг своего уровня."""
    from .gap import YaRBrushResponseWithT
    x = -np.asarray(WithETBrushA)
    kr = np.asarray(kr, np.float64)
    rows = np.arange(len(kr))
    result = np.zeros(len(kr))
    for weight, sc in zip(YaRBrushResponseWithT.astype(np.float64), s):
        ss = np.full(len(kr), float(sc))
        if shift is not None:
            uu = [-u for u in AtRResponseInNAnd] + [-WITHOUT_SHIFT]
            ss = np.minimum(ss + np.array([np.interp(-sc, uu, list(row) + [0.0]) for row in shift]), 0.0)
        xx = np.clip(-ss, x[0], x[-1])
        i = np.clip(np.searchsorted(x, xx) - 1, 0, len(x) - 2)
        a = (xx - x[i]) / (x[i + 1] - x[i])
        ni, ve = kr[rows, i], kr[rows, i + 1]
        if log:
            v = np.exp((1 - a) * np.log(np.maximum(ni, 1e-6)) + a * np.log(np.maximum(ve, 1e-6)))
            v = np.where(np.maximum(ni, ve) <= 1e-6, 0.0, v)
        else:
            v = (1 - a) * ni + a * ve
        result = result + weight * v
    return result


NEEDED_COLOR = ('{кто}: нужен цвет=…: покрытие не кроющей зависит от цвета — {где} пропускает свет сквозь плёнку, светлая '
              'кроет слабее тёмной в 2–5 раз; дай краску хода: цвет=(r, g, b), `Краска` или `цвет(L, C, h)`')


def _coverage(t, preset, sh, pressure, iv, r, s):
    """Кривая покрытия по узлам нагрузки и её σ при разбавителе р — прямой между узлами разбавителя."""
    parts = [(w, _coverage_node(t, preset, sh, pressure, iv, ir, s)) for ir, w in weights_thinner(t, r)]
    return sum(w * h[0] for w, h in parts), sum(w * h[1] for w, h in parts)


def _coverage_node(t, preset, sh, pressure, iv, ir, s):
    """То же в узле разбавителя номер iр: кроющая густая — поля таблицы, не кроющая — по `сквозь` и цвету."""
    if iv == 0 and ir == 0:
        return _curve_node(t, preset, sh, pressure, iv, ir, 'покрытие'), _curve_node(t, preset, sh, pressure, iv, ir, 'разброс')
    if 'сквозь' not in t['пресеты'][preset]:
        raise ValueError('калибровка: таблица снята до 5е-2, цвета в ней нет — python3 -m кисть.калибровка --пересчитать')
    shift = _curve_node(t, preset, sh, pressure, iv, ir, 'сдвиг') if ir else None
    return (1.0 - _gap_colors(_curve_node(t, preset, sh, pressure, iv, ir, 'сквозь'), s, shift=shift),
            _gap_colors(_curve_node(t, preset, sh, pressure, iv, ir, 'разброс_сквозь'), s, log=False, shift=shift))


def _what_behind_paint(iv, r):
    return f'{InAndDY[iv]}{f" с разбавителем {r:g}" if r else ""} краска'


def by_table(brush, load, pressure=0.6, kind=None, thinner=None, px_on_mm=None, color=None):
    """Что даст одиночный ход по сухому грунту: {'покрытие', 'толщина', 'доля', 'разброс', 'разброс_толщины'} по
    таблице. цвет — краска хода (sRGB 0–255, `Краска` — с неё и вид, `цвет(L, C, h)`): не кроющей (прозрачная,
    полукроющая, любая с разбавителем) он обязателен — её покрытие зависит от цвета; кроющей густой не нужен.
    Вне таблицы и не кроющая без цвета — ValueError."""
    t = table()
    preset, sh = _preset_brushes(brush, px_on_mm)
    from .tool import NAGRAtZBrushAnd as AndMENA
    n = AndMENA[load] if isinstance(load, str) else float(load)
    _in_within('нагрузка', n, NAGRAtZBrushAnd, '')
    iv, r = _kind(_kind_paints(kind, color)), _thinner(thinner)
    s = _color(color)
    if (iv or r) and s is None:
        raise ValueError(NEEDED_COLOR.format(who='по_таблице', where=_what_behind_paint(iv, r)))
    response = {p: round(_by_load(_curve(t, preset, sh, pressure, iv, r, p), n, p), 4) for p in PResponseLYa}
    if iv or r:
        kr, σ = _coverage(t, preset, sh, pressure, iv, r, s)
        response['покрытие'], response['разброс'] = round(_by_load(kr, n, depth=True), 4), round(_by_load(σ, n), 4)
    return response


def load_for(brush, coverage=None, pressure=0.6, kind=None, thinner=None, thickness=None, px_on_mm=None, color=None):
    """В-2: нагрузка числом, при которой одиночный ход этой кистью по сухому грунту кроет на `покрытие` (1 − просвет
    грунта, как у `х.просвет`) или кладёт плёнку `толщина` (h, как `Слой.толщина`) — по таблице `калибровка.json`.

    Кисть — из пресета (`Кисть.из_файла`), ширина в мм или px (px — при `px_на_мм`, по умолчанию 8). Вид — `'кроющая'`
    (умолчание), `'полукроющая'`, `'прозрачная'`; `'лессировочная'` (мутность 0, и числом) — только под `толщина=`, строкой
    `'прозрачная'` (толщина плёнки от мутности не зависит), под `покрытие=` — `ValueError` (не кроет); разбавитель — 0–1 (узлы 0, 0,3, 0,5, 0,7, 1). цвет — краска хода (sRGB
    0–255, `Краска` — с неё берётся и вид, `цвет(L, C, h)`): под `покрытие` не кроющей (прозрачная, полукроющая, любая
    с разбавителем) обязателен — светлая кроет слабее тёмной при той же нагрузке; кроющей густой и под `толщина`
    не нужен. Недостижимое (покрытие выше, чем у нагрузки 0,9, ниже, чем у 0,002), вне таблицы и не кроющая без
    цвета — `ValueError` словами."""
    if (coverage is None) == (thickness is None):
        raise ValueError('нагрузка_для: одно из двух — покрытие= или толщина=')
    field, target = ('покрытие', float(coverage)) if coverage is not None else ('толщина', float(thickness))
    t = table()
    preset, sh = _preset_brushes(brush, px_on_mm)
    glaze = field == 'толщина' and _glaze(kind, color)
    iv, r = (InAndDY.index('прозрачная') if glaze else _kind(_kind_paints(kind, color))), _thinner(thinner)
    s = _color(color)
    if field == 'покрытие':
        if (iv or r) and s is None:
            raise ValueError(NEEDED_COLOR.format(who='нагрузка_для', where=_what_behind_paint(iv, r)))
        kr = _coverage(t, preset, sh, pressure, iv, r, s)[0]
    else:
        kr = _curve(t, preset, sh, pressure, iv, r, field)
    kr = np.maximum.accumulate(kr)      # не убывает: первая нагрузка, что даёт цель
    where = f'«{preset}» {sh:g} мм, нажим {pressure:g}, {"лессировочная (толщина — как у прозрачной)" if glaze else InAndDY[iv]}, разбавитель {r:g}'
    if field == 'покрытие' and (iv or r):
        where += ', цвет ({:g}, {:g}, {:g})'.format(*tuple(color)[:3])
    if target > kr[-1] + 1e-9:
        raise ValueError(f'нагрузка_для: {field} {target:g} недостижимо — {where}: наибольшее {kr[-1]:.3f} при нагрузке '
                         f'{NAGRAtZBrushAnd[int(np.argmax(kr >= kr[-1] - 1e-9))]:g}; больше — шире кисть, плотнее (кроющая, без '
                         'разбавителя) или темнее краска, или второй ход по высохшему')
    if target < kr[0] - 1e-9:
        raise ValueError(f'нагрузка_для: {field} {target:g} недостижимо — {where}: наименьшее {kr[0]:.4f} при нагрузке '
                         f'{NAGRAtZBrushAnd[0]:g}; тоньше — вытертая кисть (нагрузка 0) по сырому или прозрачнее краска')
    k = int(np.argmax(kr >= target - 1e-12))
    if k == 0:
        return float(NAGRAtZBrushAnd[0])
    a = (target - kr[k - 1]) / (kr[k] - kr[k - 1]) if kr[k] > kr[k - 1] else 1.0
    if field == 'покрытие' and (iv or r) and kr[k] > kr[k - 1] > 1e-6:      # не кроющая: между узлами — по глубине плёнки
        a = float(np.clip((_depth(target) - _depth(kr[k - 1])) / (_depth(kr[k]) - _depth(kr[k - 1])), 0.0, 1.0))
    if field in LAndNEYNResponse:
        return round(float(NAGRAtZBrushAnd[k - 1] + a * (NAGRAtZBrushAnd[k] - NAGRAtZBrushAnd[k - 1])), 5)
    lo, lv = math.log(NAGRAtZBrushAnd[k - 1]), math.log(NAGRAtZBrushAnd[k])
    return round(float(math.exp(lo + a * (lv - lo))), 5)


# ----------------------------------------------------------------- сверка
def verify(node=8, nodes=None, file=FAYL, print=print):
    """Хеш и живые ходы на узлах против таблицы: `узлы` — список (пресет, ширина_мм, нагрузка, нажим, вид, разбавитель)
    или `узлов` через равный шаг по сетке. Отвечает списком расхождений словами (пустой — сходится)."""
    t = table(file)
    problems = []
    if t['хеш'] != hash_physics():
        problems.append(f'хеш таблицы {t["хеш"]} ≠ хеш ядра {hash_physics()}')
    if nodes is None:
        tasks = list(_tasks())
        spacing = max(1, len(tasks) // node)
        nodes = tasks[spacing // 2::spacing][:node]
    for z in nodes:
        preset, sh, n, nzh, kind, r = z
        pr = t['пресеты'][preset]
        and_ = (InAndDY.index(kind), t['разбавители'].index(r), pr['ширины_мм'].index(sh), NAZhAndMY.index(nzh), NAGRAtZBrushAnd.index(n))
        the_live = _node(z)
        for p in PResponseLYa + FIELDS_COLORS:
            if p not in pr:
                problems.append(f'в таблице нет поля «{p}» — снята до 5е-2')
                break
            for k, (v, zh) in enumerate(zip(np.atleast_1d(pr['сдвиг как снят' if p == 'сдвиг' else p][and_]), np.atleast_1d(the_live[p]))):
                if abs(zh - v) > 1e-4 + 1e-4 * abs(v):
                    problems.append(f'узел {z}: {p}{f" [{k}]" if p in FIELDS_COLORS else ""} в таблице {v:.5f}, '
                                f'живой ход {zh:.5f}')
    print('калибровка: сходится с ядром' if not problems else 'калибровка: разошлась с ядром —\n  ' + '\n  '.join(problems))
    return problems


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog='python3 -m кисть.калибровка', description=__doc__.split('\n')[0])
    ap.add_argument('--пересчитать', action='store_true', help='снять таблицу заново')
    ap.add_argument('--процессов', type=int, default=16, help='процессов счёта, не больше 16')
    ap.add_argument('--сверить', action='store_true', help='хеш и живые узлы против таблицы')
    a = ap.parse_args(argv)
    if a.recompute:
        recompute(a.processes)
        _TABLAndTsA.clear()
    if a.verify or not a.recompute:
        return 1 if verify() else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
