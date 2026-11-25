# READING VIEW of движок/кисть/сводка.py with names in English. Not runnable; edit the original.
"""v4, Св-1: сводка этапа — факты этапа числами, не длиннее 12 строк: на `х.сохранить` и на смене этапа пометкой, файлом
рядом с сохранением (`<путь>.сводка.txt`) и вызовом `х.сводка()`. Только читает: холст, кисти, сиды и журнал не меняются.

Счёт этапа — на мазок, по полям его ответа (`учесть`), и лежит в счёте подсказок (`подсказки.Подсказки.счёт['э']`): откат
снимка и `загрузить` возвращают его вместе с ним. Развёртка следа (`ширина`, `просветы`, `разрывы`, `край`) при
включённых подсказках считается вместе с мазком — по тем же полям и тем же счётом, что ответу по запросу: числа сводки —
те же, что у ответов по одному; сам ответ по-прежнему ленив. Глушится тем же, чем подсказки: `подсказки=False`, `KIST_HINTS=0`, `х.подсказки.выключить()`.
"""
import math
import re

from . import under_top as N

PRResponseWithInETY = 0.10         # след с `о.просветы` от стольких — в счёт
ΔL, ΔAB = 8.0, 0.02     # тон следа ушёл от названного: L серой шкалы или a, b OKLab (длина) — как С13
WithTRResponseBrush = 12
_UNDER_HAND = 'под рукой: print(о), х.пустые, х.просвет(область), х.взгляд(область, без=[…]), х.однообразие(этап=…), х.подсказки'


class Summary(str):
    """Текст сводки; `.числа` — её числа словарём (пустая сводка — пустой словарь)."""
    numbers = {}

    def __new__(cls, text='', numbers=None):
        with_ = str.__new__(cls, text)
        with_.numbers = dict(numbers or {})
        return with_


def new():
    return {'мазков': 0, 'прочитано': 0, 'следов': 0, 'без замера': 0, 'без следа': {}, 'порван': 0, 'край рваный': 0,
            'просветы': 0, 'просветы худший': None, 'с цветом': 0, 'ушло': 0, 'ушло дальше': None, 'невидимые': 0, 'закрыли': {},
            'сказано': 0}


def account(e, measured, tone, kind):
    """Мазок досчитан: изм — `ответ.Ответ`, прямоугольник без замера или None; тон — названный (L, a, b) или None;
    вид — вид причины пустого. Поля берутся из `__dict__`: в «ответ прочитан» чтение сводкой не идёт."""
    e['мазков'] += 1
    if measured is None:
        if kind is not None:
            e['без следа'][kind] = e['без следа'].get(kind, 0) + 1
        return
    d = getattr(measured, '__dict__', {})
    if 'закрыто' not in d:
        e['без замера'] += 1
        return
    frame = [int(v) for v in measured]
    r = d.pop('_св', d)                # развёртка следа, сочтённая с мазком (`Холст._положить`)
    color = d.get('цвет')
    if d.get('видно', 1.0) < N.ZABrushRYT:
        e['невидимые'] += 1
        if d.get('закрыл') is not None:
            e['закрыли'][d['закрыл']] = e['закрыли'].get(d['закрыл'], 0) + 1
    if color is None:
        return
    e['следов'] += 1
    e['порван'] += bool(r.get('разрывы'))
    e['край рваный'] += (r.get('край') or {}).get('слово') == 'рваный'
    p = float(r.get('просветы', 0.0))
    if p >= PRResponseWithInETY:
        e['просветы'] += 1
        if e['просветы худший'] is None or p > e['просветы худший'][0]:
            e['просветы худший'] = [p, frame]
    if tone is None or d.get('видно', 1.0) < N.ZABrushRYT:      # невидимый: тон его места — чужой краски сверху
        return
    e['с цветом'] += 1
    C, h = float(color['C']), math.radians(float(color['h']))
    dL, dab = float(color['L']) - tone[0], math.hypot(C * math.cos(h) - tone[1], C * math.sin(h) - tone[2])
    if abs(dL) >= ΔL or dab >= ΔAB:
        e['ушло'] += 1
        measure = max(abs(dL) / ΔL, dab / ΔAB)
        if e['ушло дальше'] is None or measure > e['ушло дальше'][0]:
            e['ушло дальше'] = [measure, dL, dab, frame]


def _place(p, stage):
    """Худшее место однообразия среди ходов этапа по журналу листа (`однообразие.места_мазков`) или None."""
    from . import monotony as Response
    r = p.strokes_sheet()
    if r is None:
        return None
    strokes, meta = r
    own = [m for m in strokes if ('' if m['этап'] is None else str(m['этап'])) == stage]
    places = Response.places_strokes(own, Response._cell_px(meta, None))[0]
    if not places:
        return None
    d = places[0]
    return {'рамка': tuple(d['рамка']), 'мазков': d['мазков'], 'оценка': float(d['оценка']),
            'узко': [t[2] for t in d['узко по'] if t[0] != 'вытерта'][:3]}


def collect(p, stage):
    """Сводка этапа (ключ счёта) по счёту подсказок `п`; этап без мазков — пустая."""
    from .hints import InAndDY, _mn, _h
    e = p.count.get('э', {}).get(stage)
    if e is None or not e['мазков']:
        return Summary()
    empty = sum(e['без следа'].values())
    place, waiting = _place(p, stage), p.unanswered()
    h = {brush: e[brush] for brush in ('мазков', 'прочитано', 'следов', 'без замера', 'порван', 'край рваный', 'просветы', 'с цветом', 'ушло',
                           'невидимые')}
    h.update({'этап': stage or None, 'без следа': empty, 'место': place, 'неотвеченные': [canvas['вид'] for canvas in waiting]})
    t = [f'сводка этапа «{stage}»' if stage else 'сводка (этапа нет)']
    t[0] += f': {_mn(e["мазков"], "мазок", "мазка", "мазков")}, ответов прочитано {e["прочитано"]} из {e["мазков"] - empty}'
    if e['без замера']:
        t[0] += f', без замера {e["без замера"]}'
    if e['следов']:
        t.append(f'рваные: порван ход у {e["порван"]}, рваный край у {e["край рваный"]} из {e["следов"]}')
        canvas = e['просветы худший']
        t.append(f'просветы от {_h(PRResponseWithInETY, 2)}: {e["просветы"]} из {e["следов"]}'
                 + (f', больше всех {_h(canvas[0], 2)} — рамка {tuple(canvas[1])}' if canvas else ''))
    if e['с цветом']:
        canvas = e['ушло дальше']
        t.append(f'тон следа от названного (ΔL от {_h(ΔL, 0)} или Δa, b от {_h(ΔAB, 2)}): {e["ушло"]} из {e["с цветом"]}'
                 + (f', дальше всех ΔL {"+" if canvas[1] > 0 else "−"}{_h(abs(canvas[1]), 0)}, Δa, b {_h(canvas[2], 3)} — рамка {tuple(canvas[3])}' if canvas else ''))
    layers = sorted(e['закрыли'].items(), key=lambda q: (-q[1], q[0]))
    t.append(f'невидимые (видно меньше {_h(N.ZABrushRYT, 2)}, под верхним слоем): {e["невидимые"]}'
             + (' — закрыл ' + ', '.join(f'«{and_}» {n}' for and_, n in layers[:3]) if layers else ''))
    t.append(f'без следа: {empty}' + (' — ' + ', '.join(
        f'{in_[:40]} {n}' for in_, n in sorted(e['без следа'].items(), key=lambda q: (-q[1], q[0]))[:2]) if empty else ''))
    if place is not None:
        t.append(f'самое однообразное место: рамка {place["рамка"]}, {_mn(place["мазков"], "ход", "хода", "ходов")}, оценка '
                 f'{_h(place["оценка"], 2)}' + (' — ' + ', '.join(place['узко']) if place['узко'] else ''))
    if waiting:
        names = {}
        for canvas in waiting:
            names[InAndDY[canvas['вид']]] = names.get(InAndDY[canvas['вид']], 0) + 1
        and_ = [k + (f' ×{n}' if n > 1 else '') for k, n in names.items()]
        t.append(f'подсказки, чьих инструментов после них не звали: {len(waiting)} — ' + ', '.join(and_[:6]) + ('…' if len(and_) > 6 else ''))
    t.append(_UNDER_HAND)
    return Summary('\n  '.join(t[:WithTRResponseBrush]), h)


_AndNWithTRAtMENT = re.compile(r'х\.(мазок\([^)]*\)|\w+)|\b(?:[ок]\.)(\w+)|\b(print|нагрузка_для|смешение_для|мерка)\(|(кисть\.повтор)')


def tools(text):
    """Имена инструментов строки подсказки («под рукой: …») — как их считает `Подсказки.взят`."""
    _, _, tail = text.partition(' — под рукой: ')
    names = set()
    for m in _AndNWithTRAtMENT.finditer(tail):
        canvas, field, function, _ = m.groups()
        if canvas is not None:
            names.add('разбавитель' if canvas.startswith('мазок') and 'разбавитель' in canvas else 'новый слой' if canvas.startswith('мазок') else canvas)
        elif field is not None:
            names.add('кисть' if m.group(0).startswith('к.') else 'ответ')
        elif function is not None:
            names.add('ответ' if function == 'print' else function)
    if 'варианты' in names:
        names |= {'взять'}
    return names
