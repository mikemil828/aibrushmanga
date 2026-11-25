# READING VIEW of движок/кисть/однообразие.py with names in English. Not runnable; edit the original.
"""К-12, Б-9: зеркало однообразия — что у серии мазков одинаково и насколько их места похожи на сетку или случай.

Только читает журнал (`Холст(…, журнал=)` или файл .jsonl): холст, сиды и журнал не меняются, случайных чисел нет.
Зона — мазки журнала с этим "object", "stage" (от пометки выше, как у `кисть.повтор --object`), "layer", центром в
области или перечнем: номерами строк журнала или самими строками (словари). Мазки слоя, убранного "layer_drop", не
считаются; снятия ("tool") — тоже.

Что меряется у каждого мазка (по строке журнала): длина пути (ломаная по точкам строки "ctrl" / "points" / "core"),
наклон хорды начало → конец (° 0–180: 0 — вдоль x, 90 — вдоль y вниз), изгиб (длина / хорда), ширина ("size" или
кисти), профиль нажима ("pressure", "end", "entry" строки или нажим точек "core" пятью отсчётами), кисть (поля кисти
без имени и ширины: кисти, заведённые заново с теми же полями, — одна), цвет, нагрузка ("load"), центр (середина
пути). Узость признака — 1 − ln(n_эфф) / ln(min(6, мазков)), где n_эфф = exp(энтропия) — сколько вариантов у признака
на деле: длина и ширина — ступенями вдвое, наклон — по 30°, нагрузка — по 0,1, остальное — как есть. «Один приём с
дрожанием» — среднее узости длины, наклона, ширины, нажима и кисти, 0–1. Это зеркало, не запрет: что делать с цифрой,
решает модель.

Раскраска точками (М-12): у каждого хода — длина пути / ширина и площадь следа (длина × ширина + круг ширины: след
касания без пути — круг); в сводке `'ход'` — p10/p50/p90 длины/ширины, доля ходов короче `КОРОТКИЙ` своей ширины,
площадь p50 (мм² по px/мм журнала, px²). От доли `ДОЛЯ_ТОЧЕК` в словах — «N % ходов короче 1,5 своей ширины — похоже на
раскраску точками, а не мазки», у мест — в `'узко'`. В оценку «один приём» не идёт.
"""
import json
import math
import os

import numpy as np

InARAndANTResponseIn = 6               # n_эфф, при котором признак разнообразен (узость 0)
PResponseRResponseG = 0.5                 # от этой оценки отчёт говорит «похоже на один приём» / «похоже на сетку»
PRAndZNABrushAnd = ('длина', 'наклон', 'ширина', 'нажим', 'кисть')      # из них оценка
_BEZ = ('имя', 'ширина', 'ширина_мм')                            # поля кисти, что не делают её другой
BrushResponseRResponseTBrushAndY = 1.5              # М-12: ход короче стольких своих ширин — «точка»
SHARE_POINTS = 0.3            # М-12: от такой доли точек — в словах и у места (журналы «Бара»: точками 52 %, письмо 3 %)


# ---------------------------------------------------------------- журнал -> мазки

def _lines(journal):
    if isinstance(journal, (str, os.PathLike)):
        with open(os.path.expanduser(os.fspath(journal)), encoding='utf-8') as f:
            for number, s in enumerate(f, 1):
                yield number, s
    else:
        for number, s in enumerate(journal, 1):
            yield number, (s.decode('utf-8') if isinstance(s, bytes) else s)


class _Reader:
    """Разбор журнала по строкам с памятью (кисти, набранное, пометка, вытертые) — `мазки_журнала` разом, `Дочитка`
    — по мере роста файла."""

    def __init__(self, brushes=None, meta=None):
        self.brushes = {} if brushes is None else brushes
        self.meta = {} if meta is None else meta
        self.charged, self.strokes, self.px_mm, self.clean = {}, [], None, set()
        self.mixes, self.mix_brushes = {}, {}      # М-1: палитра по строкам "palette" и замес, набранный кистью
        self.current = {'stage': None, 'object': None}

    def lines(self, pairs):
        brushes, charged, current, clean = self.brushes, self.charged, self.current, self.clean
        for number, s in pairs:
            s = s.strip()
            if not s or s.startswith('#'):
                continue
            z = json.loads(s)
            if 'canvas' in z:
                self.px_mm = z.get('px_per_mm')
                self.meta['холст'], self.meta['px_на_мм'] = tuple(z['canvas'][:2]), self.px_mm
            if isinstance(z.get('brush'), str) and ('wipe' in z or 'def' in z) and 'ctrl' not in z \
                    and 'points' not in z and 'core' not in z:
                clean.add(z['brush'])
            if 'mark' in z:
                current = self.current = {p: z.get(p, current[p]) for p in current}
            if 'def' in z and isinstance(z.get('brush'), str):
                d, px_mm = z['def'], self.px_mm
                sh = (d['ширина_мм'] * px_mm) if d.get('ширина_мм') and px_mm else d.get('ширина')
                brushes[z['brush']] = (json.dumps({brush: v for brush, v in d.items() if brush not in _BEZ}, sort_keys=True), sh)
            if 'palette' in z:
                self.mixes[z['palette']] = z.get('color')
                continue
            if 'pick' in z and isinstance(z.get('brush'), str):
                charged[z['brush']] = z['pick']
                self.mix_brushes[z['brush']] = z.get('swatch')
            if isinstance(z.get('load_brush'), str):          # Д-1: набор живой краски — как "pick"
                charged[z['load_brush']] = z.get('color', z.get('colors'))
                self.mix_brushes[z['load_brush']] = z.get('swatch')
            if 'layer_drop' in z:
                self.strokes = [m for m in self.strokes if m['слой'] != z['layer_drop']]
            wet = z.get('ctrl', z.get('points', z.get('core')))
            if wet is None or 'tool' in z or 'mark' in z:
                continue
            name = z.get('brush')
            option, sh0 = brushes.get(name, (str(name), None))
            ts = z.get('color', z.get('colors'))
            mix = z.get('swatch')
            if ts is None:
                ts = charged.get(name)
                mix = self.mix_brushes.get(name)
            if isinstance(mix, str) and self.mixes.get(mix) is not None:
                ts = self.mixes[mix]                   # правленый замес: цвет — из строки палитры, как у повтора
            if isinstance(mix, list) and any(and_ is None for and_ in mix):
                mix = None                             # набор, где не всё с палитры, — не с палитры
            if ts is not None and len(ts) and isinstance(ts[0], (list, tuple)):
                ts = ts[0]
            wiped = name in clean or bool(z.get('wipe'))
            clean.discard(name)
            self.strokes.append({'номер': number, 'запись': z, 'этап': z.get('stage', current['stage']),
                               'объект': z.get('object', current['object']), 'слой': z.get('layer', 'основа'), 'кисть': option,
                               'имя кисти': str(name), 'вытерта': wiped, 'ширина': float(z.get('size') or sh0 or 1.0),
                               'цвет': None if ts is None else tuple(float(v) for v in list(ts)[:3]),
                               'замес': (mix if isinstance(mix, str) else None if mix is None
                                         else ' + '.join(sorted(set(mix))))})
        return self.strokes


def strokes_journal(journal, brushes=None, meta=None):
    """Мазки журнала (путь, строки или байты) — список словарей: 'номер' (строка журнала), 'запись' (сама строка),
    'этап', 'объект', 'слой', 'кисть' (ключ полей кисти), 'имя кисти', 'ширина', 'цвет' (sRGB первой краски или None),
    'вытерта' (кисть вытерта или заведена между прошлым своим ходом и этим). кисти — {имя: (ключ, ширина px)} из начала
    журнала (для куска, чьи кисти заведены раньше); дополняется. мета — словарь: в него 'холст' (ширина, высота) и
    'px_на_мм' из строки "canvas"."""
    if isinstance(journal, (bytes, bytearray)):
        journal = bytes(journal).decode('utf-8').splitlines()
    return _Reader(brushes, meta).lines(_lines(journal))


class ReadRest:
    """Журнал листа, что растёт по ходу письма (подсказка С14): каждый `мазки()` дочитывает только новые полные строки.
    Файл переписан (перезапуск с сохранения обрезал его, лист с начала) — последняя прочитанная строка не на месте, —
    читается с начала. Признаки мазков (`_признаки_мазка`) считаются раз."""

    def __init__(self, path):
        self.path = os.fspath(path)
        self._first()

    def _first(self):
        self.reader, self.place, self.number, self.tail = _Reader(), 0, 0, b''

    def strokes(self):
        with open(self.path, 'rb') as f:
            if self.place:
                f.seek(self.place - len(self.tail))
                if f.read(len(self.tail)) != self.tail:
                    self._first()
                    f.seek(0)
            new = f.read()
        end = new.rfind(b'\n') + 1
        if end:
            lines = new[:end].split(b'\n')[:-1]
            n0 = self.number
            self.reader.lines((n0 + i, s.decode('utf-8')) for i, s in enumerate(lines, 1))
            self.number += len(lines)
            self.place += end
            self.tail = lines[-1][-64:] + b'\n'
        return self.reader.strokes

    @property
    def meta(self):
        return self.reader.meta


def select(strokes, object=None, stage=None, layer=None, region=None, numbers=None):
    """Мазки зоны: "object", "stage", "layer" (сравнение строкой; список — любой из), центр в области (x0, y0, x1,
    y1), номера строк журнала."""
    def da(v, needed):
        if needed is None:
            return True
        needed = needed if isinstance(needed, (list, tuple, set)) else [needed]
        return str(v) in {str(n) for n in needed}
    numbers = None if numbers is None else set(int(n) for n in numbers)
    response = []
    for m in strokes:
        if not (da(m['объект'], object) and da(m['этап'], stage) and da(m['слой'], layer)):
            continue
        if numbers is not None and m['номер'] not in numbers:
            continue
        if region is not None:
            c = _features_stroke(m)['центр']
            x0, y0, x1, y1 = region
            if not (x0 <= c[0] < x1 and y0 <= c[1] < y1):
                continue
        response.append(m)
    return response


def _features_stroke(m):
    p = m.get('_п')
    if p is not None:
        return p
    z = m['запись']
    wet = z.get('ctrl', z.get('points', z.get('core')))
    a = np.asarray([list(q) for q in wet], np.float64)
    a = a.reshape(len(a), -1) if a.size else np.zeros((0, 2))
    t = a[:, :2]
    segment = np.hypot(*np.diff(t, axis=0).T) if len(t) > 1 else np.zeros(0)
    dl = float(segment.sum())
    if dl > 1e-9:
        d = np.concatenate([[0.0], np.cumsum(segment)])
        center = np.array([np.interp(dl / 2, d, t[:, 0]), np.interp(dl / 2, d, t[:, 1])])
    else:
        center = t.mean(0) if len(t) else np.zeros(2)
    chord = t[-1] - t[0] if len(t) > 1 else np.zeros(2)
    canvas_dl = float(np.hypot(*chord))
    slant = float(np.degrees(np.arctan2(chord[1], chord[0])) % 180.0) if canvas_dl > 1e-6 else math.nan
    if 'core' in z and 'ctrl' not in z and a.shape[1] >= 3 and len(a) > 1:
        d = np.concatenate([[0.0], np.cumsum(segment)])
        np = np.interp(np.linspace(0, d[-1], 5), d, a[:, 2])
        pressure = 'точки ' + ' '.join(f'{v:.1f}' for v in np)
    else:
        r = z.get('pressure')
        pressure = (f'{float(r):.2f}' if isinstance(r, (int, float)) else ('вдоль' if r is not None else '—'))
        pressure += f' «{z["end"]}»' if z.get('end') else ''
        pressure += f' вход {z["entry"]}' if z.get('entry') is not None else ''
    r = m['ширина'] / 2.0
    frame = ((float(t[:, 0].min()) - r, float(t[:, 1].min()) - r, float(t[:, 0].max()) + r, float(t[:, 1].max()) + r)
             if len(t) else (0.0, 0.0, 0.0, 0.0))
    lines = z.get('lines') or None                   # К-14б: ход кистью из пучков — линии пучков [[длина px, ширина px, нажим], …]
    p = {'линии': None if lines is None else np.asarray(lines, np.float64).reshape(len(lines), -1),
         'длина': dl, 'наклон': slant, 'изгиб': dl / canvas_dl if canvas_dl > 1e-6 else math.nan, 'ширина': m['ширина'], 'рамка': frame,
         'нажим': pressure, 'нагрузка': float(z['load']) if isinstance(z.get('load'), (int, float)) else math.nan,
         'центр': center}
    m['_п'] = p
    return p


def _by_lines(p, field):
    """К-14б: значения признака серии — у хода кистью из пучков (`Кисть.просветы`, строка с "lines") по одному на линию
    пучка (длина касания, ширина пучка), у остальных — одно на ход. поле — 'длина' или 'ширина'."""
    k = {'длина': 0, 'ширина': 1}[field]
    v = []
    for q in p:
        if q['линии'] is None:
            v.append(q[field])
        else:
            v.extend(q['линии'][:, k].tolist())
    return v


def _pressure_by_lines(p):
    """К-14б: профили нажима серии; у линии пучка — профиль хода и её нажим сверх порога касания пучка ступенями 0,1."""
    v = []
    for q in p:
        if q['линии'] is None:
            v.append(q['нажим'])
        else:
            v.extend(f'{q["нажим"]} · пучок {n:.1f}' for n in q['линии'][:, 2])
    return v


# ---------------------------------------------------------------- статистика

def _eff(labels):
    """exp(энтропия) меток: сколько вариантов на деле."""
    _, n = np.unique(np.asarray(labels), return_counts=True)
    p = n / n.sum()
    return float(math.exp(-(p * np.log(p)).sum()))


def _narrowness(eff, n):
    ref = min(InARAndANTResponseIn, n)
    if ref <= 1:
        return 1.0
    return float(np.clip(1.0 - math.log(max(eff, 1.0)) / math.log(ref), 0.0, 1.0))


def _continuous(v, bins):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    p10, p50, p90 = np.percentile(v, [10, 50, 90])
    sr, σ = float(v.mean()), float(v.std())
    return {'ср': round(sr, 3), 'σ': round(σ, 3), 'σ/ср': round(σ / sr, 3) if sr > 1e-9 else None,
            'p10': round(float(p10), 3), 'p50': round(float(p50), 3), 'p90': round(float(p90), 3),
            'эфф': round(_eff(bins(v)), 2), 'n': int(v.size)}


def _angular(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    z = np.exp(2j * np.radians(v)).mean()
    sr = float(np.degrees(np.angle(z)) / 2 % 180.0)
    R = float(abs(z))
    σ = float(np.degrees(math.sqrt(max(-2 * math.log(max(R, 1e-12)), 0.0))) / 2)
    brush = np.floor(((v + 15.0) % 180.0) / 30.0).astype(int)              # 6 корзин по 30°, 0° — в середине первой
    _, n = np.unique(brush, return_counts=True)
    main = sorted(((int(c), int(m)) for c, m in zip(*np.unique(brush, return_counts=True))), key=lambda t: -t[1])
    return {'ср': round(sr, 1), 'σ': round(σ, 1), 'R': round(R, 3), 'эфф': round(_eff(brush), 2), 'n': int(v.size),
            'главные': [(c * 30, round(m / v.size, 3)) for c, m in main[:3]]}


def _categories(v):
    # М-11: счёт словарём, не массивом строк numpy (ширина — по длиннейшей строке ×4 байта: ключи кистей на 210 тыс.
    # ходов — 1,3 ГБ); порядок значений — тот же, что у np.unique (по кодам), счёт и argsort — те же
    count = {}
    for x in v:
        x = str(x)
        count[x] = count.get(x, 0) + 1
    value = sorted(count)
    n = np.asarray([count[x] for x in value])
    i = np.argsort(-n)
    p = n / n.sum()
    return {'разных': int(len(value)), 'частое': str(value[i[0]]), 'доля частого': round(float(n[i[0]] / n.sum()), 3),
            'эфф': round(float(math.exp(-(p * np.log(p)).sum())), 2), 'n': int(n.sum()),
            'главные': [(str(value[j]), round(float(n[j] / n.sum()), 3)) for j in i[:3]]}


def _mixes(strokes):
    """М-1: замесы зоны — сколько разных имён замесов у ходов с краской, доля ходов не с палитры (цвет без имени
    замеса), сколько таких и разных цветов у них; 'первые' — (ходов, не с палитры, разных) первых `подсказки.ФОРМУЛА_ХОДОВ`
    ходов с краской, если по ним звучит подсказка «цвет формулой» (С16), иначе None."""
    from .hints import FORMULA_SWEEPS, FORMULA_SHARE
    with_paint = [m for m in strokes if m['цвет'] is not None]
    if not with_paint:
        return None
    without = [m for m in with_paint if m.get('замес') is None]
    names = set()
    for m in with_paint:
        if m.get('замес') is not None:
            names.update(m['замес'].split(' + '))
    first = None
    p = with_paint[:FORMULA_SWEEPS]
    if len(p) >= FORMULA_SWEEPS:
        pb = [m for m in p if m.get('замес') is None]
        r = len({json.dumps(m['запись'].get('color', m['запись'].get('colors', m['цвет']))) for m in pb})
        if len(pb) >= FORMULA_SHARE * len(p) and r >= FORMULA_SHARE * len(pb):
            first = (len(p), len(pb), r)
    return {'замесов': len(names), 'не с палитры': round(len(without) / len(with_paint), 3), 'без имени': len(without),
            'разных без имени': len({m['цвет'] for m in without}), 'n': len(with_paint), 'первые': first}


def _colors(colors):
    has = [c for c in colors if c is not None]
    if not has:
        return None
    from .eye import oklab
    a = np.asarray(has, np.float64)
    even = len({tuple(np.round(c, 2)) for c in a})
    lab = oklab(a)
    on_eye = len({tuple(q) for q in np.floor(lab / (0.02, 0.02, 0.02)).astype(int)})    # ступень ~2 L, ΔE OKLab 0,02
    return {'разных': even, 'на глаз': on_eye, 'n': len(has)}


# ---------------------------------------------------------------- Б-9 регулярность

WithResponseWithEDEY = 8             # М-11: сколько ближайших брать у дерева разом; равных дальше — шаром


def _neighbors(c):
    """Расстояние до ближайшего соседа и вектор к нему — точно как полный перебор (np.hypot, из равных — меньший
    номер), но деревом (scipy cKDTree): n·log n и память на n, а не n² (М-11: журнал 590 тыс. строк перебором кусками
    просил 11 ГБ). Дерево даёт кандидатов, расстояние пересчитывается тем же np.hypot; если у кандидатов последний не
    дальше ближайшего (равных больше, чем взято), — все в шаре этого радиуса."""
    from scipy.spatial import cKDTree
    n = len(c)
    if n < 2:
        return np.full(n, np.inf), np.zeros_like(c)
    k = min(n, WithResponseWithEDEY + 1)
    jj = cKDTree(c).query(c, k=k)[1]
    own = np.arange(n)[:, None]
    h = np.hypot(c[:, None, 0] - c[jj, 0], c[:, None, 1] - c[jj, 1])
    h[jj == own] = np.inf
    d = h.min(1)
    j = np.where(h == d[:, None], jj, n).min(1)
    if k < n:
        few = np.flatnonzero(h[np.arange(n), (np.where(np.isfinite(h), h, -1.0)).argmax(1)] <= d * (1 + 1e-9) + 1e-12)
        zero = few[d[few] == 0]
        if zero.size:                                     # центры в одной точке: сосед — меньший номер той же точки
            _, gr = np.unique(c, axis=0, return_inverse=True)
            gr = gr.ravel()
            p = np.lexsort((np.arange(n), gr))           # по группе, внутри — по номеру
            start = np.r_[0, np.flatnonzero(np.diff(gr[p])) + 1]
            first = np.empty(n, int)
            second = np.full(n, n)
            first[gr[p[start]]] = p[start]
            more = start[(start + 1 < n) & (np.r_[start[1:], n] > start + 1)]
            second[gr[p[more]]] = p[more + 1]
            j[zero] = np.where(first[gr[zero]] == zero, second[gr[zero]], first[gr[zero]])
            few = few[d[few] > 0]
        if few.size:                                     # равных ближайшему больше, чем взято: шаром
            balls = cKDTree(c).query_ball_point(c[few], d[few] * (1 + 1e-9) + 1e-9)
            for i, sh in zip(few, balls):
                sh = np.asarray([q for q in sh if q != i], int)
                hh = np.hypot(c[i, 0] - c[sh, 0], c[i, 1] - c[sh, 1])
                d[i] = hh.min()
                j[i] = sh[hh == d[i]].min()
    return d, c[j] - c


def _area(c):
    """Площадь выпуклой оболочки точек (монотонная цепь)."""
    p = sorted(set(map(tuple, np.round(c, 6))))
    if len(p) < 3:
        return 0.0
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    bottom, top = [], []
    for q in p:
        while len(bottom) >= 2 and cross(bottom[-2], bottom[-1], q) <= 0:
            bottom.pop()
        bottom.append(q)
    for q in reversed(p):
        while len(top) >= 2 and cross(top[-2], top[-1], q) <= 0:
            top.pop()
        top.append(q)
    response = np.asarray(bottom[:-1] + top[:-1])
    return float(abs(np.dot(response[:, 0], np.roll(response[:, 1], 1)) - np.dot(response[:, 1], np.roll(response[:, 0], 1))) / 2)


def _evenness(v):
    """Насколько значения равномерны на своём размахе, 0–1: D — Колмогоров против равномерного; сверх шума выборки
    (0,87/√n — столько даёт сам равномерный случай) каждые 0,25 D снимают единицу."""
    v = np.sort(np.asarray(v, np.float64)[np.isfinite(v)])
    n = len(v)
    if n < 5 or v[-1] - v[0] < 1e-9:
        return 0.0
    u = (v - v[0]) / (v[-1] - v[0])
    i = np.arange(1, n + 1) / n
    D = float(max(np.abs(i - u).max(), np.abs(u - (i - 1 / n)).max()))
    return float(np.clip(1.0 - max(D - 0.87 / math.sqrt(n), 0.0) / 0.25, 0.0, 1.0))


def regularity_strokes(strokes):
    """Б-9: по центрам, наклонам, длинам и шагу (порядок журнала) — похоже ли на сетку с дрожанием или на равномерный
    случай. Ответ — словарь: 'сетка', 'случай' 0–1, 'итог' — большее из них, 'слово', и из чего они:
      'сосед σ/ср' — разброс расстояний до ближайшего соседа (сетка ~0,1–0,3, случай ~0,5, рука по форме — выше);
      'сетка' — большее из 'шаг повторён' и среднего двух: соседи на равных расстояниях (σ/ср от 0,35 к 0,1) и
          порядок (от 0,3 к 0,8) — рука, что кроет место ровно, даёт σ/ср ~0,4 и порядок ~0,3; 'случай' — среднее
          (R около 1, прыжки шага, ровность мест x, y, ровность наклона и длины);
      'R' — Кларк–Эванс: ср. расстояние до соседа / 0,5·√(площадь оболочки / n) (сетка 1,4–2, случай ~1, кучно < 1);
      'порядок' — |⟨e^{imφ}⟩| векторов к соседу, наибольший по m = 2, 4, 6 (ряд, квадрат, соты — 1);
      'шаг/сосед' — медиана шага от мазка к следующему по журналу / ср. расстояние до соседа (рука и ряд ~1–2, случай —
          прыжки через всю зону); 'шаг повторён' — доля шагов, что повторяют прошлый вектор с точностью 25 %;
      'ровность x', 'ровность y', 'ровность наклона', 'ровность длины' — 1, когда значения ровно размазаны по размаху
          (так даёт равномерный случай), 0 — когда сбиты к местам."""
    n = len(strokes)
    if n < 5:
        return {'мазков': n, 'сетка': 0.0, 'случай': 0.0, 'итог': 0.0, 'слово': f'мазков {n} — мало для мерки (от 5)'}
    p = [_features_stroke(m) for m in strokes]
    c = np.asarray([q['центр'] for q in p])
    d, brush_neighbor = _neighbors(c)
    sr_d = float(d.mean())
    cv = float(d.std() / sr_d) if sr_d > 1e-9 else 0.0
    S = _area(c)
    R = sr_d / (0.5 * math.sqrt(S / n)) if S > 1e-9 else 0.0
    φ = np.arctan2(brush_neighbor[:, 1], brush_neighbor[:, 0])
    order = max(float(abs(np.exp(1j * m * φ).mean())) for m in (2, 4, 6))
    spacing = np.diff(c, axis=0)
    dsh = np.hypot(*spacing.T)
    spacing_neighbor = float(np.median(dsh) / sr_d) if sr_d > 1e-9 else 0.0
    replay = np.hypot(*(spacing[1:] - spacing[:-1]).T) <= 0.25 * np.maximum(dsh[1:], 1e-9)
    spacing_repeated = float(replay.mean()) if len(replay) else 0.0
    even = {'x': _evenness(c[:, 0]), 'y': _evenness(c[:, 1]),
            'наклона': _evenness([q['наклон'] for q in p]), 'длины': _evenness([q['длина'] for q in p])}
    if sr_d < 1e-9:                           # все в одной точке: сетка вырождена
        grid = 1.0
    else:
        grid = max(spacing_repeated, float(np.mean([np.clip((0.35 - cv) / 0.25, 0, 1), np.clip((order - 0.3) / 0.5, 0, 1)])))
    chance = float(np.mean([1.0 - np.clip(abs(R - 1.0) / 0.35, 0, 1), np.clip((spacing_neighbor - 2.0) / 4.0, 0, 1),
                            (even['x'] + even['y']) / 2, (even['наклона'] + even['длины']) / 2]))
    result = max(grid, chance)
    if result < PResponseRResponseG:
        word = 'места не сетка и не равномерный случай'
    elif grid >= chance:
        why = (f'шаг к следующему повторяет прошлый у {_h(100 * spacing_repeated, 0)} % — ряды' if spacing_repeated >= 0.5
                  else f'соседи на ровных расстояниях (σ/ср {_h(cv)}), порядок {_h(order)}')
        word = f'места похожи на сетку с дрожанием ({_h(grid)}): {why}'
    else:
        word = (f'места похожи на равномерный случай ({_h(chance)}): R {_h(R)}, шаг к следующему — '
                 f'{_h(spacing_neighbor, 1)} расстояния до соседа (прыжки)')
    return {'мазков': n, 'сетка': round(grid, 3), 'случай': round(chance, 3), 'итог': round(result, 3), 'слово': word,
            'сосед ср': round(sr_d, 2), 'сосед σ/ср': round(cv, 3), 'R': round(R, 3), 'порядок': round(order, 3),
            'шаг/сосед': round(spacing_neighbor, 3), 'шаг повторён': round(spacing_repeated, 3),
            **{f'ровность {brush}': round(v, 3) for brush, v in even.items()}}


# ---------------------------------------------------------------- К-12 отчёт

def _h(v, n=2):
    return f'{v:.{n}f}'.replace('.', ',')


class Monotony:
    """Отчёт зеркала: `оценка` 0–1 («один приём с дрожанием»), `слова` — строка, что узко; `сводка` — числа по
    признакам ('длина', 'наклон', 'изгиб', 'ширина', 'нажим', 'кисть', 'конец', 'цвета', 'нагрузка' и 'узость' каждого),
    `регулярность` — Б-9 (`регулярность_мазков`), `мазков`, `номера` — строки журнала зоны. `о['длина']` — то же, что
    `о.сводка['длина']`; print(о) — слова."""
    __slots__ = ('оценка', 'слова', 'сводка', 'регулярность', 'мазков', 'номера', 'зона', 'места', 'объекты', 'клетки',
                 'клетка', 'карта', 'холст_размер')

    def __getitem__(self, option):
        return self.summary[option]

    def __repr__(self):
        return self.words


def report(strokes, zone='', reference=None, px_mm=None):
    """Однообразие мазков (список от `мазки_журнала` / `отобрать`). образец — `Однообразие` другой зоны (меркой
    образца, К-4): к словам — во сколько раз у образца разброс (n_эфф) шире по каждому узкому признаку. px_мм — px/мм
    журнала: площадь хода в мм² (М-12)."""
    response = Monotony()
    response.zone, response.strokes, response.numbers = zone, len(strokes), [m['номер'] for m in strokes]
    response.places, response.objects, response.cells, response.cell, response.map = None, None, None, None, None
    n = len(strokes)
    if n == 0:
        response.score, response.summary, response.regularity = 0.0, {}, regularity_strokes([])
        response.words = f'{zone}: мазков нет' if zone else 'мазков нет'
        return response
    p = [_features_stroke(m) for m in strokes]
    log2 = lambda v: np.floor(np.log2(np.maximum(v, 1e-3))).astype(int)
    with_ = {'длина': _continuous(_by_lines(p, 'длина'), log2),
         'наклон': _angular([q['наклон'] for q in p]),
         'изгиб': _continuous([q['изгиб'] for q in p], lambda v: np.floor(np.log(v) / math.log(1.1)).astype(int)),
         'ширина': _continuous(_by_lines(p, 'ширина'), log2),
         'нажим': _categories(_pressure_by_lines(p)),
         'кисть': _categories([m['кисть'] for m in strokes]),
         'цвета': _colors([m['цвет'] for m in strokes]),
         'замесы': _mixes(strokes),
         'нагрузка': _continuous([q['нагрузка'] for q in p], lambda v: np.floor(v / 0.1).astype(int))}
    narrowness = {brush: (_narrowness(v['эфф'], v['n']) if v is not None else None) for brush, v in with_.items() if brush not in ('цвета', 'замесы')}
    with_['узость'] = {brush: (round(v, 3) if v is not None else None) for brush, v in narrowness.items()}
    has = [narrowness[brush] for brush in PRAndZNABrushAnd if narrowness[brush] is not None]
    response.score = round(float(np.mean(has)) if has else 0.0, 3)
    with_['ход'] = _sweep(p, px_mm)
    with_lines = [len(q['линии']) for q in p if q['линии'] is not None]
    with_['линии'] = ({'ходов': len(with_lines), 'линий': int(sum(with_lines)),
                   'на ход': tuple(round(float(v), 1) for v in np.percentile(with_lines, [10, 50, 90]))} if with_lines else None)
    response.regularity = regularity_strokes(strokes)
    response.summary = with_
    response.words = _words(response, reference)
    return response


def _points(p):
    """М-12: (длина / ширина, площадь следа px²) каждого хода (у кисти из пучков — каждой линии пучка) — массивы."""
    dl = np.asarray(_by_lines(p, 'длина'), np.float64)       # К-14б: у кисти из пучков ход — линии пучков
    sh = np.maximum(np.asarray(_by_lines(p, 'ширина'), np.float64), 1e-6)
    return dl / sh, dl * sh + math.pi * sh * sh / 4.0


def _sweep(p, px_mm=None):
    """М-12: сводка `'ход'` — 'длина/ширина' (p10, p50, p90), 'короче 1,5 ширины' (доля ходов короче `КОРОТКИЙ` своей
    ширины), 'площадь px²' и 'площадь мм²' (p50; без px/мм журнала — None)."""
    r, pl = _points(p)
    p10, p50, p90 = (round(float(v), 2) for v in np.percentile(r, [10, 50, 90]))
    a = float(np.median(pl))
    return {'длина/ширина': (p10, p50, p90), 'короче 1,5 ширины': round(float((r < BrushResponseRResponseTBrushAndY).mean()), 3),
            'площадь px²': round(a, 1), 'площадь мм²': round(a / px_mm ** 2, 3) if px_mm else None}


def word_points(share):
    """М-12: «N % ходов короче 1,5 своей ширины» — зеркалу, месту и подсказке одним словом."""
    return f'{100 * share:.0f} % ходов короче {_h(BrushResponseRResponseTBrushAndY, 1)} своей ширины'


def _words(response, reference):
    with_, at, n = response.summary, response.summary['узость'], response.strokes
    narrow, wide = [], []

    def times(brush):
        if reference is None or not with_.get(brush):
            return ''
        if isinstance(reference, Monotony):
            e, by = (reference.summary[brush] or {}).get('эфф') if reference.summary.get(brush) else None, ''
        else:                                              # мерка манеры образца (К-4): по картинке
            m = getattr(reference, 'образец', None) or reference
            e = {'наклон': ('направления',), 'длина': ('штрих', 'длина'), 'ширина': ('ширина мазка',)}.get(brush)
            if e is None or not hasattr(m, 'сводка'):
                return ''
            d = m.summary
            for h in e:
                d = d.get(h, {})
            if d.get('на полу', 0) > 0.5:                 # штрих уже первого сдвига линейки — не различается
                return ''
            e, by = d.get('эфф'), ' (образец — по картинке)'
        if not e:
            return ''
        k = e / max(with_[brush]['эфф'], 1e-9)
        return f'; у образца вариантов в {_h(k, 1)} раза больше{by}' if k >= 1.2 else ''
    for brush in ('нажим', 'кисть', 'длина', 'наклон', 'ширина', 'нагрузка'):
        v = with_.get(brush)
        if v is None or at.get(brush) is None:
            continue
        if brush in ('нажим', 'кисть'):
            if brush == 'кисть':
                t = 'одна кисть' if v['разных'] == 1 else f'кистей {v["разных"]}, у частой {_h(100 * v["доля частого"], 0)} %'
            else:
                t = ', '.join(f'{m} — {_h(100 * d, 0)} %' for m, d in v['главные'][:2])
                t = f'профиль нажима {t}'
        elif brush == 'наклон':
            t = 'наклон ' + ', '.join(f'{g}°±15 — {_h(100 * d, 0)} %' for g, d in v['главные'][:2] if d >= 0.1)
        else:
            ed = {'длина': ' px', 'ширина': ' px', 'нагрузка': ''}[brush]
            z = 2 if brush == 'нагрузка' else 1
            t = f'{brush} {_h(v["p10"], z)}–{_h(v["p90"], z)}{ed} (σ/ср {_h(v["σ/ср"] or 0)})'
        t += f', вариантов {_h(v["эфф"], 1)}' + times(brush)
        (narrow if at[brush] >= 0.6 else wide).append((brush, t))
    if response.score >= PResponseRResponseG:
        head = f'мазков {n} — похоже на один приём с дрожанием ({_h(response.score)})'
    else:
        head = f'мазков {n} — разные приёмы ({_h(response.score)})'
    parts = [(response.zone + ': ' if response.zone else '') + head]
    ln = with_.get('линии')
    if ln:
        parts.append(f'ходов кистью из пучков {ln["ходов"]}, линий в них {ln["линий"]} (на ход p50 {_h(ln["на ход"][1], 0)}) — '
                     'длина, ширина и нажим таких ходов считаны по линиям пучков')
    canvas = with_.get('ход')
    if canvas and canvas['короче 1,5 ширины'] >= SHARE_POINTS:
        pl = (f'{_h(canvas["площадь мм²"])} мм²' if canvas['площадь мм²'] is not None else f'{canvas["площадь px²"]:.0f} px²')
        parts.append(word_points(canvas['короче 1,5 ширины']) + ' — похоже на раскраску точками, а не мазки (длина/ширина '
                     f'p10–p50–p90 {_h(canvas["длина/ширина"][0])}–{_h(canvas["длина/ширина"][1])}–{_h(canvas["длина/ширина"][2])}, '
                     f'площадь хода p50 {pl})')
    if narrow:
        parts.append('узко: ' + '; '.join(t for _, t in narrow))
    if wide:
        parts.append('разнообразно: ' + ', '.join(brush for brush, _ in wide))
    ts = with_.get('цвета')
    if ts:
        t = f'цветов {ts["разных"]} разных, на глаз {ts["на глаз"]}'
        z = with_.get('замесы')
        if z:
            t += f'; замесов {z["замесов"]}, не с палитры {_h(100 * z["не с палитры"], 0)} %'
        if z and z['без имени'] >= 20 and z['разных без имени'] >= 0.8 * z['без имени']:
            t += ' — цвет формулой на каждый мазок, не замесами (под рукой: х.палитра_с, х.палитра)'
        parts.append(t)
        if z and z.get('первые') is not None:
            from .hints import fact_formulas, TOOLS_FORMULAS
            f = z['первые']
            parts.append('подсказка: ' + fact_formulas(response.zone or 'зона', *f) + ' — под рукой: ' + ', '.join(TOOLS_FORMULAS))
    parts.append(response.regularity['слово'])
    return '. '.join(parts)


# ---------------------------------------------------------------- зеркало с адресом (v4, РЕШЕНИЕ DEN 18.11 23:33 +04)

CELL_MM = 50.0        # клетка мест листа, мм (без px/мм в журнале — КЛЕТКА_PX)
CELL_PX = 400
PLACE_SWEEPS = 30        # ходов зоны в клетке — от стольких место меряется
THRESHOLD_PLACES = 0.6       # оценка места, от которой оно в «худших» и в подсказке С14
MEWithT = 8                # худших мест — в словах и номерами на карте
InYTERTA = 0.9           # доля ходов с вытертой перед ними кистью, от которой это — в словах (в оценку не идёт)
# узость признака места, 0–1: (край − разброс) / ход, обрезано в [0, 1] (пилоты 18.11: заливка зоны — фон портрета
# 0,80–0,92, мостовая экспр 0,69–0,80; ручные черты портрета, этап 9 — 0,10)
NARROWNESS_PLACES = {'кисть': (0.5, -0.4), 'нагрузка': (0.25, 0.2), 'ширина': (0.4, 0.3), 'длина': (0.6, 0.4),
                'наклон': (30.0, 20.0)}


def _cv(v):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(v.std() / v.mean()) if v.size and v.mean() > 1e-9 else 0.0


def _at(sign, spread):
    edge, sweep = NARROWNESS_PLACES[sign]
    return float(np.clip((spread - edge) / -sweep if sweep < 0 else (edge - spread) / sweep, 0.0, 1.0))


def gauge_places(strokes):
    """Зеркало места: узость кисти (доля частой), нагрузки, ширины, длины (σ/ср), наклона (σ, °) ходов — 0–1 по
    `УЗОСТЬ_МЕСТА`; 'оценка' — их среднее; 'узко' — словами признаки с узостью от 0,6 и вытирание перед ходом (от
    `ВЫТЕРТА` ходов; в оценку не идёт) и раскраска точками (от `ДОЛЯ_ТОЧЕК` ходов короче `КОРОТКИЙ` своей ширины, М-12; в
    оценку не идёт); 'узко по' — те же [(признак, узость, слова)], узкие первыми; 'числа' — из чего.
    Нагрузка — числом (σ/ср) или словом (доля частого); у ходов без нагрузки в строке — не меряется."""
    n = len(strokes)
    p = [_features_stroke(m) for m in strokes]
    brushes = {}
    for m in strokes:
        brushes.setdefault(m['кисть'], []).append(m.get('имя кисти') or '?')
    names = max(brushes.values(), key=len)
    name = max(sorted(set(names)), key=names.count)
    share = len(names) / n
    ng = [m['запись'].get('load') for m in strokes]
    has = [v for v in ng if v is not None]
    sh = np.asarray(_by_lines(p, 'ширина'))                  # К-14б: у кисти из пучков — по линиям пучков
    dl = np.asarray(_by_lines(p, 'длина'))
    nk = _angular([q['наклон'] for q in p])
    at, h, by = {}, {}, []
    at['кисть'] = _at('кисть', share)
    h['кисть'] = (name, round(share, 3))
    by.append(('кисть', at['кисть'], f'одной кистью «{name}» — {100 * share:.0f} %'))
    if has and len(has) >= 0.9 * n:
        if all(isinstance(v, (int, float)) for v in has):
            v = np.asarray(has, np.float64)
            at['нагрузка'] = _at('нагрузка', _cv(v))
            p10, p90 = np.percentile(v, [10, 90])
            h['нагрузка'] = (round(float(p10), 3), round(float(p90), 3), round(_cv(v), 3))
            by.append(('нагрузка', at['нагрузка'], f'нагрузка {_h(p10)}' + ('' if p90 - p10 < 0.005 else f'–{_h(p90)}')))
        else:
            with_ = [str(v) for v in has]
            part = max(sorted(set(with_)), key=with_.count)
            d = with_.count(part) / len(with_)
            at['нагрузка'] = _at('кисть', d)
            h['нагрузка'] = (part, round(d, 3))
            by.append(('нагрузка', at['нагрузка'], f'нагрузка «{part}» — {100 * d:.0f} %'))
    for brush, v in (('ширина', sh), ('длина', dl)):
        at[brush] = _at(brush, _cv(v))
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        h[brush] = (round(float(p10), 1), round(float(p50), 1), round(float(p90), 1), round(_cv(v), 3))
        by.append((brush, at[brush], f'{brush} {p50:.0f} px' if p90 - p10 < 0.5 else f'{brush} {p10:.0f}–{p90:.0f} px'))
    if nk is not None:
        at['наклон'] = _at('наклон', nk['σ'])
        h['наклон'] = (nk['ср'], nk['σ'])
        by.append(('наклон', at['наклон'], f'наклон {nk["ср"]:.0f}°±{max(nk["σ"], 0.0):.0f}°'))
    by = [t for t in by if t[1] >= 0.6]
    r, _ = _points(p)
    tch = float((r < BrushResponseRResponseTBrushAndY).mean())
    h['короче 1,5 ширины'] = round(tch, 3)
    if tch >= SHARE_POINTS:                               # М-12: в оценку не идёт, как вытирание
        by.append(('точки', tch, word_points(tch)))
    vt = float(np.mean([bool(m.get('вытерта')) for m in strokes]))
    h['вытерта'] = round(vt, 3)
    if vt >= InYTERTA:
        by.append(('вытерта', vt, 'вытерта перед каждым ходом' if vt >= 0.995 else f'вытерта перед {100 * vt:.0f} % ходов'))
    return {'мазков': n, 'оценка': round(float(np.mean(list(at.values()))), 3), 'узость': {brush: round(v, 3) for brush, v in at.items()},
            'узко': [t[2] for t in by], 'узко по': sorted(by, key=lambda t: -t[1]), 'числа': h}


def _cell_px(meta, cell):
    px = (meta or {}).get('px_на_мм')
    if cell is None:
        return float(CELL_MM * px) if px else float(CELL_PX)
    return float(cell * px) if px else float(cell * CELL_PX / CELL_MM)


def _name_zones(m):
    return ('объект', str(m['объект'])) if m['объект'] is not None else ('слой', str(m['слой']))


def places_strokes(strokes, cell_px=CELL_PX, zones=None):
    """Места: мазки каждого объекта (без пометки объекта — слоя) по клеткам `клетка_px` (по центру хода); место — зона ×
    клетка от `МЕСТО_ХОДОВ` ходов. Ответ — (места, объекты, клетки): места — список словарей ('объект' или 'слой', 'клетка',
    'рамка' — ходы места в клетке, x0, y0, x1, y1 px, 'мазков', 'оценка', 'узость', 'узко', 'числа', 'номера'), худшие
    первыми; объекты — {(вид, имя): {'мазков', 'мест', 'худших' (от `ПОРОГ_МЕСТА`), 'оценка' — худшего места}};
    клетки — клетки листа, каждая своим худшим местом (тем же словарём), худшие первыми. зоны — только эти (вид, имя)."""
    gr = {}
    for m in strokes:
        z = _name_zones(m)
        if zones is not None and z not in zones:
            continue
        c = _features_stroke(m)['центр']
        brush = (int(math.floor(c[0] / cell_px)), int(math.floor(c[1] / cell_px)))
        gr.setdefault((z, brush), []).append(m)
    with_palette = any(m.get('замес') is not None for m in strokes)     # М-1: лист с палитрой — у места доля не с палитры

    def place(mm, brush):
        x0, y0 = brush[0] * cell_px, brush[1] * cell_px
        r = np.asarray([_features_stroke(m)['рамка'] for m in mm])
        frame = (max(x0, float(r[:, 0].min())), max(y0, float(r[:, 1].min())),
                 min(x0 + cell_px, float(r[:, 2].max())), min(y0 + cell_px, float(r[:, 3].max())))
        d = gauge_places(mm)
        d.update({'клетка': (round(x0), round(y0), round(x0 + cell_px), round(y0 + cell_px)),
                  'рамка': tuple(int(round(max(v, 0.0))) for v in frame), 'номера': [m['номер'] for m in mm]})
        tsv = [m for m in mm if m['цвет'] is not None]
        if with_palette and tsv:
            d['не с палитры'] = round(sum(m.get('замес') is None for m in tsv) / len(tsv), 3)
        return d
    places, objects = [], {}
    for (z, brush), mm in gr.items():
        response = objects.setdefault(z, {'мазков': 0, 'мест': 0, 'худших': 0, 'оценка': None})
        response['мазков'] += len(mm)
        if len(mm) < PLACE_SWEEPS:
            continue
        d = place(mm, brush)
        d[z[0]] = z[1]
        places.append(d)
        response['мест'] += 1
        response['худших'] += d['оценка'] >= THRESHOLD_PLACES
        response['оценка'] = d['оценка'] if response['оценка'] is None else max(response['оценка'], d['оценка'])
    option = lambda d: (-d['оценка'], -d['мазков'], d['клетка'][1], d['клетка'][0])
    places.sort(key=option)
    cells = {}
    for d in places:                                   # клетка листа — её худшим местом
        cells.setdefault(d['клетка'], d)
    return places, objects, list(cells.values())


def word_places(d):
    """Место одной строкой: «объект «фон», рамка (x0, y0, x1, y1): N ходов, оценка — узко: …»."""
    z = f'объект «{d["объект"]}»' if 'объект' in d else f'слой «{d["слой"]}»' if 'слой' in d else 'клетка листа'
    return (f'{z}, рамка {tuple(d["рамка"])}: {d["мазков"]} ходов, {_h(d["оценка"])}'
            + (' — ' + ', '.join(d['узко']) if d['узко'] else '')
            + (f'; не с палитры {_h(100 * d["не с палитры"], 0)} %' if d.get('не с палитры') else ''))


def _words_points_places(response, ed):
    """М-12: строка «точками» — места, где от `ДОЛЯ_ТОЧЕК` ходов короче `КОРОТКИЙ` своей ширины, больше всего точек
    первыми (до трёх рамками); таких мест нет — пусто."""
    t = sorted((d for d in response.places if d['числа'].get('короче 1,5 ширины', 0) >= SHARE_POINTS),
               key=lambda d: (-d['числа']['короче 1,5 ширины'], -d['мазков'], d['клетка'][1], d['клетка'][0]))
    if not t:
        return ''
    return (f'\nточками (от {100 * SHARE_POINTS:.0f} % ходов короче {_h(BrushResponseRResponseTBrushAndY, 1)} своей ширины) — мест {len(t)} из '
            f'{len(response.places)} (клетка {ed}); больше всего: '
            + '; '.join(f'{"объект «" + d["объект"] if "объект" in d else "слой «" + d["слой"]}», рамка {tuple(d["рамка"])} — '
                        f'{100 * d["числа"]["короче 1,5 ширины"]:.0f} % из {d["мазков"]}' for d in t[:3]))


def _words_places(response, cell_mm):
    ed = f'{cell_mm:.0f} мм' if cell_mm else f'{response.cell:.0f} px'
    return _words_places_score(response, ed) + _words_points_places(response, ed)


def _words_places_score(response, ed):
    worst = [d for d in response.places if d['оценка'] >= THRESHOLD_PLACES]
    if not response.places:
        return f'мест нет: ни в одной клетке {ed} у зоны (объект, без пометки — слой) нет {PLACE_SWEEPS} ходов'
    if not worst:
        return (f'мест {len(response.places)} (объект × клетка {ed}, от {PLACE_SWEEPS} ходов); от {_h(THRESHOLD_PLACES)} — ни одного, '
                f'наибольшая оценка {_h(response.places[0]["оценка"])}: ' + word_places(response.places[0]))
    h = [f'места (объект × клетка {ed}, от {PLACE_SWEEPS} ходов): {len(response.places)}, из них от {_h(THRESHOLD_PLACES)} — {len(worst)}; '
         f'худшие:']
    h += [f'  {i}. ' + word_places(d) for i, d in enumerate(worst[:MEWithT], 1)]
    zones = sorted(((z, v) for z, v in response.objects.items() if v['худших']), key=lambda t: (-t[1]['худших'], -t[1]['мазков']))
    h.append('по зонам — мест от порога из мест: ' + ', '.join(f'{z[1]} {v["худших"]}/{v["мест"]}' for z, v in zones[:12])
             + (f' и ещё {len(zones) - 12}' if len(zones) > 12 else ''))
    return '\n'.join(h)


def _words_mix(strokes, zones=12):
    """М-1: строка «замесы по зонам» — у каждой зоны (объект, без пометки — слой) замесов и доля не с палитры."""
    gr = {}
    for m in strokes:
        gr.setdefault(_name_zones(m), []).append(m)
    if not any(m.get('замес') is not None for m in strokes):
        return '\nзамесы по зонам: палитры нет — все ходы с краской не с палитры' if any(m['цвет'] is not None for m in strokes) else ''
    h = []
    for z, mm in sorted(gr.items(), key=lambda t: -len(t[1])):
        with_ = _mixes(mm)
        if with_ is not None:
            h.append(f'{z[1]} {with_["замесов"]} (не с палитры {_h(100 * with_["не с палитры"], 0)} %)')
    if not h:
        return ''
    return '\nзамесы по зонам: ' + ', '.join(h[:zones]) + (f' и ещё {len(h) - zones}' if len(h) > zones else '')


def map_places(response, path, background=None, size=1600):
    """PNG: лист (фон — sRGB (В, Ш, 3) холста; без него — серый), клетки листа подкрашены оценкой своего худшего места
    (от 0,4 — жёлтым к красному), места от `ПОРОГ_МЕСТА` — рамками, худшие `МЕСТ` — толще и номерами, как в словах.
    Ответ — массив uint8."""
    from PIL import Image, ImageDraw
    from .eye import _font
    Sh, In = response.canvas_size
    k = min(1.0, float(size) / max(Sh, In))
    sh, in_ = max(1, int(round(Sh * k))), max(1, int(round(In * k)))
    if background is None:
        base = Image.new('RGB', (sh, in_), (128, 128, 128))
    else:
        base = Image.fromarray(np.clip(np.asarray(background) + 0.5, 0, 255).astype(np.uint8)).resize((sh, in_), Image.LANCZOS)
    base = Image.blend(base, Image.new('RGB', (sh, in_), (255, 255, 255)), 0.35)
    layer = Image.new('RGBA', (sh, in_), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for c in response.cells:
        t = (c['оценка'] - 0.4) / (1.0 - 0.4)
        if t <= 0:
            continue
        x0, y0, x1, y1 = (v * k for v in c['клетка'])
        d.rectangle((x0, y0, x1 - 1, y1 - 1), fill=(255, int(200 * (1 - t)), 0, int(15 + 65 * t)))
    base = Image.alpha_composite(base.convert('RGBA'), layer).convert('RGB')
    d = ImageDraw.Draw(base)
    font = _font(max(12, int(round(in_ / 40))))
    worst = [m for m in response.places if m['оценка'] >= THRESHOLD_PLACES]
    for i, m in enumerate(worst):
        x0, y0, x1, y1 = (v * k for v in m['рамка'])
        d.rectangle((x0, y0, x1, y1), outline=(200, 0, 0) if i < MEWithT else (150, 40, 0), width=4 if i < MEWithT else 1)
    for i, m in enumerate(worst[:MEWithT], 1):
        x0, y0 = m['рамка'][0] * k, m['рамка'][1] * k
        d.rectangle((x0 + 2, y0 + 2, x0 + 4 + font.getlength(str(i)), y0 + 4 + font.size), fill=(200, 0, 0))
        d.text((x0 + 3, y0 + 2), str(i), fill=(255, 255, 255), font=font)
    if path is not None:
        base.save(os.path.expanduser(str(path)))
    return np.asarray(base)


def monotony(journal, object=None, stage=None, region=None, strokes=None, layer=None, reference=None, path=None,
                cell=None, background=None):
    """К-12: зеркало однообразия зоны журнала (путь .jsonl, строки или байты). Зона — объект, этап, слой, область
    (центр мазка в (x0, y0, x1, y1)), мазки (номера строк журнала или сами строки-словари); не задано — весь журнал.
    образец — `Однообразие` зоны образца (или словарь тех же отборов по этому журналу) или мерка манеры картины
    (`манера.мерка`, `х.манера(…, образец=…)`, К-4: наклон, длина штриха и ширина мазка — по картинке): во сколько раз у него
    вариантов больше. Ответ — `Однообразие`.
    Без отбора объекта, этапа, слоя и мазков (область — можно) — и с адресом: `места` — каждый объект (без пометки —
    слой) по клеткам листа `клетка` мм (без неё — `КЛЕТКА_ММ`), худшие первыми, с рамкой, оценкой и словами, что узко
    (`мерка_места`); `объекты` — по зонам: мест и худших; `клетки` — клетки листа худшим местом; `путь` — PNG-карта
    (`карта_мест`, фон — sRGB холста); худшие места — и в `слова`."""
    meta = {}
    if strokes is not None and len(strokes) and isinstance(next(iter(strokes)), dict):
        all = strokes_journal([json.dumps(z, ensure_ascii=False) for z in strokes], meta=meta)
        numbers = None
    else:
        all = strokes_journal(journal, meta=meta)
        numbers = strokes
    zone = ', '.join(f'{brush} {v}' for brush, v in (('объект', object), ('этап', stage), ('слой', layer), ('область', region))
                     if v is not None)
    if isinstance(reference, dict):
        reference = report(select(all, **reference), ', '.join(f'{brush} {v}' for brush, v in reference.items()))
    zone_strokes = select(all, object, stage, layer, region, numbers)
    response = report(zone_strokes, zone, reference, meta.get('px_на_мм'))
    if object is None and stage is None and layer is None and strokes is None:
        response.cell = _cell_px(meta, cell)
        response.places, response.objects, response.cells = places_strokes(zone_strokes, response.cell)
        response.words += '.\n' + _words_places(response, cell or (CELL_MM if meta.get('px_на_мм') else None))
        response.words += _words_mix(zone_strokes)
        if path is not None:
            r = meta.get('холст')
            if r is None:
                xs = [_features_stroke(m)['рамка'] for m in zone_strokes] or [(0, 0, 1, 1)]
                r = (int(max(r[2] for r in xs)) + 1, int(max(r[3] for r in xs)) + 1)
            response.canvas_size = r
            response.map = map_places(response, path, background)
    elif path is not None or cell is not None:
        raise ValueError('однообразие: путь= и клетка= — у зеркала без отбора (места по всему листу или области)')
    return response


def regularity(journal, object=None, stage=None, region=None, strokes=None, layer=None):
    """Б-9: мерка регулярности зоны журнала (отбор — как у `однообразие`) — словарь `регулярность_мазков`."""
    return monotony(journal, object, stage, region, strokes, layer).regularity
