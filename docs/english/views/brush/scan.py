# READING VIEW of движок/кисть/перебор.py with names in English. Not runnable; edit the original.
"""v4, 3д (Б-8): перебор — пачка мазков с одним (или несколькими разом) параметром на нескольких значениях; лист
вариантов рядом и замеры каждого; холст не тронут.

`х.перебор(мазки, нагрузка=[0.006, 0.012, 0.025])` — поверх `х.варианты` (2г): каждое значение — свой вариант в
своём процессе, на холсте, каким он был при вызове; мазки варианта — те же мазки, у которых параметр заменён этим
значением. Мазок — словарь доводов `х.мазок` (`{'точки': …, 'цвет': …, 'кисть': …, 'слой': …, …}`) или кортеж
`(точки, цвет, кисть)` / `(точки, цвет, кисть, {доводы})`. Параметр — довод `х.мазок` (нагрузка, касание, запас,
цвет, кисть, слой…) или поле кисти (ширина, сухость, кайма…: тогда мазок идёт копией своей кисти с этим полем).
Несколько параметров — списки одной длины, вариант k берёт k-е значение каждого.

Сиды мазков у всех вариантов одни (из сида листа, этапа, объекта и точек) — варианты различаются только параметром.
Выбранный ложится в холст и журнал `х.взять(п[k])`, как если бы его мазки писались прямо.

Замеры варианта (на его участке, тон без света рельефа, как у мерок «Глаз»): 'средний' — `х.средний` области;
'след' — где L изменилось от 0,5: 'доля' области, 'dL' (среднее изменения), 'dL p5', 'dL p95', 'L' (тон там);
'просвет' — `х.просвет` области сквозь нижний слой мазков: 'просвет', 'доля', 'дыр'; 'мазки' — ответы мазков
(L, тело, закрыто, просветы; пустой — None); с `цель` — 'разность' (`х.разность`: среднее, σ, dab); с `мера` — 'своё'
(что вернула `мера(участок, область)`); 'однообразие' — оценка зеркала мазков варианта.
"""
import copy
import warnings
from dataclasses import fields

import numpy as np

from .eye import oklab, hex, _L, _in_8, _sheet as sheet_panel, _scale, _h
from .tool import Brush
from .paint import from_linear

_DResponseInResponseDY = ('точки', 'цвет', 'кисть', 'слой', 'сид', 'непрозрачность', 'раскладка', 'нагрузка', 'мешать_ниже', 'касание',
           'полосы', 'по_гребням', 'ровно', 'запас', 'ломаная', 'нагрузка_вдоль', 'волокна', 'нажим_руки', 'этап', 'объект',
           'намерение', 'разбавитель')
_FIELDS_BRUSHES = tuple(f.name for f in fields(Brush))
AndZMENENResponse = 0.5          # след варианта — где L изменилось хотя бы на столько


class Scan:
    """Ответ `х.перебор`: `.варианты` — `Вариант` 2г (`п[k]` — то же; `х.взять(п[k])` кладёт его в холст и журнал),
    `.значения` — словарь параметров каждого, `.замеры` — словарь замеров каждого (модуль), `.область`, `.лист` —
    uint8 «до | вариант 0 | вариант 1 …» по три в ряд с подписью значения и главных замеров (`путь` — PNG);
    `print(п)` — таблица."""
    __slots__ = ('варианты', 'значения', 'замеры', 'область', 'лист')

    def __getitem__(self, k):
        return self.variants[k]

    def __len__(self):
        return len(self.variants)

    def __iter__(self):
        return iter(self.variants)

    def __str__(self):
        lines = [f'перебор {self.region}: {len(self)} вар.']
        for k, (z, m) in enumerate(zip(self.values, self.measurements)):
            with_, pr = m['след'], m['просвет']
            t = (f'{k}: {_values(z)} · L {m["средний"]["L"]:.1f} · след {with_["доля"]:.3f} dL {with_["dL"]:+.1f} '
                 f'(p5 {with_["dL p5"]:+.1f}, p95 {with_["dL p95"]:+.1f}) · просвет {pr["просвет"]:.3f}, дыр {pr["дыр"]}')
            if 'разность' in m:
                t += f' · к цели dL {m["разность"]["среднее"]:+.1f} σ {m["разность"]["σ"]:.1f}'
            lines.append(t)
        return '\n'.join(lines)

    __repr__ = __str__


def _values(z):
    def one(v):
        if isinstance(v, Brush):
            return f'{v.name} {v.width:g}'
        if hasattr(v, 'вид'):                           # Краска
            return f'{v.kind} {hex(tuple(int(round(c)) for c in v))}'
        if isinstance(v, float):
            return f'{v:g}'
        if isinstance(v, tuple) and len(v) == 3 and all(isinstance(c, (int, float)) for c in v):
            return hex(tuple(int(round(c)) for c in v))
        return str(v)
    return ', '.join(f'{brush}={one(v)}' for brush, v in z.items())


def _stroke(m, j):
    """Мазок пачки -> словарь доводов `х.мазок`."""
    if isinstance(m, dict):
        d = dict(m)
    elif isinstance(m, (tuple, list)) and len(m) in (3, 4) and (len(m) == 3 or isinstance(m[3], dict)):
        d = {'точки': m[0], 'цвет': m[1], 'кисть': m[2], **(m[3] if len(m) == 4 else {})}
    else:
        raise TypeError(f'перебор: мазок {j} — словарь доводов х.мазок или (точки, цвет, кисть[, {{доводы}}])')
    foreign = [brush for brush in d if brush not in _DResponseInResponseDY]
    if foreign:
        raise TypeError(f'перебор: мазок {j}: доводов {foreign} у х.мазок нет (есть: {", ".join(_DResponseInResponseDY)})')
    if 'точки' not in d or d.get('кисть') is None:
        raise TypeError(f'перебор: мазок {j}: нужны точки и кисть')
    return d


def _with_value(d, param):
    d = dict(d)
    for name, v in param.items():
        if name in _DResponseInResponseDY:
            d[name] = v
        else:                                            # поле кисти: мазок идёт копией своей кисти с этим полем
            brush = copy.copy(d['кисть'])
            setattr(brush, name, v)
            d['кисть'] = brush
    return d


def _L_region(at, region):
    return _L(oklab(np.clip(from_linear(at.image(light=False, region=region)) * 255.0, 0.0, 255.0))[..., 0])


def scan(canvas, strokes, region=None, target=None, path=None, scale=1, cores=None, measure=None, **param):
    """Б-8: подробно — модуль."""
    from .gap import gap
    if not param:
        raise TypeError('перебор: параметр — довод х.мазок или поле кисти списком значений: перебор(мазки, нагрузка=[0.01, 0.02])')
    for name, zn in param.items():
        if name not in _DResponseInResponseDY and name not in _FIELDS_BRUSHES:
            from .tool import hint
            raise TypeError(f'перебор: {name!r} — не довод х.мазок и не поле кисти ({hint(name, _DResponseInResponseDY + _FIELDS_BRUSHES)})')
        if name in ('точки',):
            raise TypeError('перебор: точки не перебираются — другие места кладёт `х.варианты` с куском')
        if isinstance(zn, (str, bytes)) or not hasattr(zn, '__len__') or len(zn) < 1:
            raise TypeError(f'перебор: {name} — список значений, например {name}=[a, b, c]')
    lengths = {len(v) for v in param.values()}
    if len(lengths) > 1:
        raise ValueError(f'перебор: списки параметров разной длины {[(brush, len(v)) for brush, v in param.items()]} — '
                         'вариант k берёт k-е значение каждого')
    n = lengths.pop()
    batch = [_stroke(m, j) for j, m in enumerate(strokes)]
    if not batch:
        raise ValueError('перебор: мазков нет')
    values = [{name: zn[k] for name, zn in param.items()} for k in range(n)]
    charge = [[_with_value(d, z) for d in batch] for z in values]
    if region is None:                                  # рамка всех мазков всех значений: место, что они читают и пишут
        frames = []
        for charge in charge:
            for d in charge:
                how = {brush: v for brush, v in d.items() if brush not in ('точки', 'кисть', 'сид', 'касание', 'цвет', 'этап', 'объект', 'намерение')}
                r = canvas.rectangle_stroke(d['точки'], d['кисть'], seed=d.get('сид'), touch=d.get('касание'),
                                          color=d.get('цвет'), **how)
                if r is not None:
                    frames.append(r)
        if not frames:
            raise ValueError('перебор: ни один мазок холста не тронет (все вне холста)')
        region = (min(r[0] for r in frames), min(r[1] for r in frames), max(r[2] for r in frames), max(r[3] for r in frames))
    layers = list(canvas.layers)
    layer_strokes = None
    for d in batch:                                      # нижний слой мазков: сквозь него — просвет
        with_ = d.get('слой', 'основа')
        if layer_strokes is None or (with_ in layers and (layer_strokes not in layers or layers.index(with_) < layers.index(layer_strokes))):
            layer_strokes = with_

    def piece(at, k):
        region = (max(region[0], at.x0), max(region[1], at.y0), min(region[2], at.x0 + at.width), min(region[3], at.y0 + at.height))
        to = _L_region(at, region)
        responses = []
        for d in charge[k]:
            response = at.stroke(**d)
            if not response or not hasattr(response, 'тело'):          # пустой мазок или холст без замера (`замер=False`)
                responses.append(None)
            else:
                responses.append({'L': None if response.color is None else response.color['L'], 'тело': response.body, 'закрыто': response.covered,
                               'просветы': response.gaps})
        at.finish_computing()
        after = _L_region(at, region)
        d = after - to
        trace = np.abs(d) >= AndZMENENResponse
        if trace.any():
            ds = d[trace]
            with_ = {'доля': round(float(trace.mean()), 4), 'dL': round(float(ds.mean()), 2),
                 'dL p5': round(float(np.percentile(ds, 5)), 2), 'dL p95': round(float(np.percentile(ds, 95)), 2),
                 'L': round(float(after[trace].mean()), 2)}
        else:
            with_ = {'доля': 0.0, 'dL': 0.0, 'dL p5': 0.0, 'dL p95': 0.0, 'L': None}
        pr = gap(at, region, layer=layer_strokes if layer_strokes in at.layers else None, area=4, holes=None)
        inside = [d for d in pr.holes if not d['у края']]   # дыра у края области — чаще место мимо мазков, не дыра
        measurement = {'средний': at.mean(region), 'след': with_,
                 'просвет': {'просвет': round(float(pr.map[trace].mean()), 4) if trace.any() else 1.0,
                             'доля': round(float((pr.map[trace] >= pr['порог']).mean()), 4) if trace.any() else 1.0,
                             'дыр': len(inside), 'видно': pr['видно'],
                             'дыры': [(d['центр'], d['площадь']) for d in inside[:5]]},
                 'мазки': responses}
        if target is not None:
            r = at.difference(target, region).summary
            measurement['разность'] = {'среднее': r['среднее'], 'σ': r['σ'], 'dab': r['dab']}
        if measure is not None:
            measurement['своё'] = measure(at, region)
        return measurement

    with warnings.catch_warnings(record=True) as words:
        warnings.simplefilter('always')
        vv = canvas.variants(piece, n, region, cores)
    for w in words:                                      # «одинаковые картинки» варианты говорят о куске — здесь своё слово
        if not str(w.message).startswith('варианты: одинаковые картинки'):
            warnings.warn(w.message, stacklevel=2)
    identical = {}
    for in_ in vv:
        identical.setdefault(in_.image.tobytes(), []).append(in_.number)
    pairs = [[_values(values[i]) for i in g] for g in identical.values() if len(g) > 1]
    if pairs:
        warnings.warn(f'перебор: значения {pairs} дали одну картинку — {", ".join(param)} здесь ничего не меняет',
                      stacklevel=2)
    p = Scan()
    p.variants, p.values, p.region = vv, values, vv[0].region
    p.measurements = []
    for in_ in vv:
        z = dict(in_.response)
        z['однообразие'] = round(float(in_.monotony.score), 3)
        p.measurements.append(z)
    from .eye import look
    panels = [look(canvas, p.region, scale=scale)]
    signatures = ['до']
    for in_, z, m in zip(vv, values, p.measurements):
        panels.append(_scale(_in_8(np.clip(from_linear(in_.image) * 255.0, 0.0, 255.0)), scale))
        with_ = m['след']
        signatures.append(f'{in_.number}: {_values(z)} · dL {_h(with_["dL"])} · просвет {_h(m["просвет"]["просвет"], 3)}')
    p.sheet = sheet_panel(panels, signatures, path)
    return p
