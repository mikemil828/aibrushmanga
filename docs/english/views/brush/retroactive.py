# READING VIEW of движок/кисть/задним.py with names in English. Not runnable; edit the original.
"""К-3, А-7: правка журнала задним числом — по фильтру, одной командой — и пересборка только затронутого.

    python -m кисть.задним журнал.jsonl --out лист.png [фильтр] [правка]

Фильтр (всё вместе — «и»): `--lines 4-200`, `--layer`, `--stage`, `--object`, `--swatch` (замес), `--area x0 y0 x1 y1`
(точка пути мазка в прямоугольнике, px журнала). Правка: `--delete`; `--color r g b`, `--set-swatch имя`, `--dL`, `--dC`,
`--dh` (градусы), `--warmer` (b OKLab +), `--move dx dy`, `--load`, `--pressure`, `--pressure-mul`, `--size`. Прежний
журнал — рядом, `<журнал>.до_правки` (`.2`, …); удалённая строка становится заметкой {"note": …, "deleted": строка}
(номера строк и сиды прочих те же), в конце журнала — заметка о правке.

Пересборка. Мазок читает и пишет холст только в своей рамке (`ядра.py`); рамки мазков пишет повтор
(`собрать`, `<out>.точки/рамки.json`). Затронуто — правленые строки и всё позднее, что от них зависит по цепи:
рамка пересекает рамку затронутого (у правленой строки — прежнюю и новую), кисть несёт подобранное затронутым мазком
(без вытирания между ними), сушка или размачивание правленой строкой — её область. Чтобы затронутое легло как в
полном повторе, кладутся и мазки, от которых оно зависит (раньше него, рамка пересекает или кисть та же) —
«пересчитано»; прочие мазки только учитываются (сид, счётчик, слой) и не считаются. Основа — контрольная точка до первой
правленой строки (нет — чистый холст). Итог — холст конца прежнего журнала, в рамках затронутого — пересобранный. Мягкой
кисти, которая спрашивает, есть ли в слое сырое, ответ сверяется: не виден вне непосчитанного — честный повтор.
Не взялось (нет точки конца или рамок, скотч, нож и снятие, правка пометки, холста или слоёв, рамка разошлась) —
честный повтор с ближайшей точки (`собрать`), сказано в ответе. Итог бит в бит тот же, что полный повтор правленого
журнала."""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

from . import replay as p_
from .replay import (collect, points_folders, load_file_state, save_state, _hashes_lines, _read_frames,
                     _write_frames, _job_brushes, _write_point, _remove, FAndZAndBrushA)

MAZResponseBrush = ('ctrl', 'points', 'core')
ALMOST_EVERYTHING = 0.9          # доля мазков после основы, с которой пересобирать затронутое дороже честного повтора


class Edit(dict):
    """Ответ правки: 'правлено' (строк), 'пересчитано' (мазков), 'мазков' (после основы), 'доля' (пересчитанных от
    всех мазков журнала), 'площадь' (доля листа под затронутым), 'время' с, 'как' ('затронутое' или 'честный повтор:
    почему'), 'основа' (строка точки или 0), 'рамка' (x0, y0, x1, y1 затронутого), 'путь' — PNG «до | после» 1:1."""

    def __str__(self):
        d = self
        return (f'правка: строк правлено {d["правлено"]}, пересчитано {d["пересчитано"]} из {d["всего"]} мазков '
                f'({100 * d["доля"]:.1f} %), площадь {100 * d["площадь"]:.1f} % листа, {d["время"]:.1f} с; {d["как"]}; '
                f'основа — строка {d["основа"]}').replace('.', ',') + (f'; до | после: {d["путь"]}' if d.get('путь') else '')


# ------------------------------------------------------------------ правка строк по фильтру
def _shift_tone(rgb, dL=0.0, dC=0.0, dh=0.0, warmer=0.0):
    from .eye import oklab, from_oklab
    L, a, b = oklab(np.asarray(rgb[:3], np.float64))
    C, h = math.hypot(a, b), math.atan2(b, a) + math.radians(dh)
    C = max(C + dC, 0.0)
    lab = np.array([min(max(L + dL, 0.0), 1.0), C * math.cos(h), C * math.sin(h) + warmer])
    return [round(float(v), 2) for v in from_oklab(lab)]


def _in_region(z, region, scale=1.0):
    t = np.asarray(z.get('core', z.get('points', z.get('ctrl'))), np.float64)
    if t.ndim != 2 or t.shape[1] < 2:
        return False
    x0, y0, x1, y1 = region
    return bool(((t[:, 0] >= x0) & (t[:, 0] <= x1) & (t[:, 1] >= y0) & (t[:, 1] <= y1)).any())


def _lines_range(v):
    if v is None:
        return None
    if isinstance(v, str):
        a, _, b = v.partition('-')
        return int(a), int(b or a)
    if isinstance(v, int):
        return v, v
    return int(v[0]), int(v[1])


def edit_lines(lines, lines_with=None, layer=None, stage=None, object=None, mix=None, region=None, delete=False,
                   color=None, new_mix=None, dL=0.0, dC=0.0, dh=0.0, warmer=0.0, shift=None, load=None,
                   pressure=None, pressure_in=None, width=None):
    """Правка строк журнала (список строк) у мазков по фильтру. Ответ — (новые строки, номера правленых, описание).
    Мазок — строка с "ctrl", "points" или "core"; фильтр — диапазон строк ("4-200" или (4, 200)), слой, этап и объект
    (без своих полей — от пометки выше), замес ("swatch"), область (точка пути внутри). Тон (dL, dC — OKLab, dh — °,
    теплее — b OKLab) у мазка с замесом — от цвета замеса на его месте журнала, замес снимается."""
    range = _lines_range(lines_with)
    if not any([delete, color is not None, new_mix, dL, dC, dh, warmer, shift, load is not None, pressure is not None,
                pressure_in is not None, width is not None]):
        raise ValueError('правка: не сказано, что править (удалить, цвет, замес, dL/dC/dh/теплее, сдвиг, нагрузка, нажим, ширина)')
    new, edited, current, mixes = list(lines), [], {'stage': None, 'object': None}, {}
    for number, row in enumerate(lines, 1):
        with_ = row.strip()
        if not with_ or with_.startswith('#'):
            continue
        z = json.loads(with_)
        if 'mark' in z:
            current = {brush: z.get(brush, current[brush]) for brush in current}
        if 'palette' in z:
            mixes[z['palette']] = p_._color(z)
        if not any(brush in z for brush in MAZResponseBrush):
            continue
        if range and not range[0] <= number <= range[1]:
            continue
        if layer is not None and str(z.get('layer', 'основа')) != str(layer):
            continue
        if stage is not None and str(z.get('stage', current['stage'])) != str(stage):
            continue
        if object is not None and str(z.get('object', current['object'])) != str(object):
            continue
        if mix is not None and z.get('swatch') != mix:
            continue
        if region is not None and not _in_region(z, region):
            continue
        was = dict(z)
        if delete:
            z = {'note': 'правка задним числом: мазок удалён', 'deleted': was}
        else:
            if color is not None:
                z.pop('colors', None)
                z.pop('swatch', None)
                z['color'] = [float(v) for v in color]
            if new_mix is not None:
                z['swatch'] = str(new_mix)
            if dL or dC or dh or warmer:
                if isinstance(z.get('swatch'), str):
                    if z['swatch'] not in mixes:
                        raise ValueError(f'правка, строка {number}: замес {z["swatch"]} выше не заведён')
                    z['color'] = [float(v) for v in np.asarray(mixes[z['swatch']], float)[:3]]
                    del z['swatch']
                if 'swatch' in z:
                    raise ValueError(f'правка, строка {number}: тон у набора с замесами — править замесы')
                if 'colors' in z:
                    z['colors'] = [[_shift_tone(c[0], dL, dC, dh, warmer), c[1]] if isinstance(c[0], (list, tuple))
                                   else _shift_tone(c[:3], dL, dC, dh, warmer) + list(c[3:]) for c in z['colors']]
                elif 'color' in z:
                    z['color'] = _shift_tone(z['color'], dL, dC, dh, warmer)
                else:
                    raise ValueError(f'правка, строка {number}: у мазка нет цвета (краска кисти) — тон не сдвинуть')
            if shift is not None:
                for brush in MAZResponseBrush:
                    if brush in z:
                        z[brush] = [[round(p[0] + shift[0], 4), round(p[1] + shift[1], 4)] + list(p[2:]) for p in z[brush]]
            if load is not None:
                z['load'] = load
            if pressure is not None:
                if 'ctrl' not in z:
                    raise ValueError(f'правка, строка {number}: нажим числом — у мазка с "ctrl"; у точек — нажим_в (множитель)')
                z['pressure'] = float(pressure)
            if pressure_in is not None:
                if 'ctrl' in z:
                    z['pressure'] = round(min(float(z.get('pressure', 0.85)) * pressure_in, 1.0), 6)
                for brush in ('points', 'core'):
                    if brush in z:
                        z[brush] = [list(p[:2]) + [round(min(max(p[2] * pressure_in, 0.0), 1.0), 6)] + list(p[3:]) for p in z[brush]]
            if width is not None:
                if isinstance(z.get('brush'), str):
                    raise ValueError(f'правка, строка {number}: ширина именной кисти — в её строке "def"')
                z['size'] = float(width)
        if z != was:
            if load is not None or pressure is not None or pressure_in is not None or width is not None:
                z.pop('lines', None)                         # К-14б: линии пучков хода сняты до правки — с новым нажимом они другие; зеркало возьмёт ход целиком
            new[number - 1] = json.dumps(z, ensure_ascii=False, separators=(',', ':')) + ('\n' if row.endswith('\n') else '')
            edited.append(number)
    what = {brush: v for brush, v in dict(lines=lines_with, layer=layer, stage=stage, object=object, mix=mix, region=region,
                                 delete=delete or None, color=color, new_mix=new_mix, dL=dL or None, dC=dC or None,
                                 dh=dh or None, warmer=warmer or None, shift=shift, load=load, pressure=pressure,
                                 pressure_in=pressure_in, width=width).items() if v is not None}
    return new, edited, what


def _reserve(path):
    for i in range(1, 1000):
        z = f'{path}.до_правки' + ('' if i == 1 else f'.{i}')
        if not os.path.exists(z):
            return z
    raise RuntimeError(f'{path}: прежних журналов больше 999')


# ------------------------------------------------------------------ зависимость мазков
def _crosses(r, frames):
    return any(q[0] < r[2] and q[2] > r[0] and q[1] < r[3] and q[3] > r[1] for q in frames)


def _parsing(lines, start, preset, scale):
    """Строки после основы: номер -> (вид, ключ кисти, вытирает, запись)."""
    response, live = {}, set()
    for number, row in enumerate(lines, 1):
        with_ = row.strip()
        if not with_ or with_.startswith('#'):
            continue
        z = json.loads(with_)
        job = _job_brushes(z)
        if job is not None:
            live.add(z[job])
        if number < start:
            continue
        if any(brush in z for brush in p_.WithInResponseBResponseDNYE):
            response[number] = ('пометка' if 'mark' in z else 'заметка', None, False, z)
        elif 'canvas' in z or 'physics' in z:
            response[number] = ('холст', None, False, z)
        elif any(brush in z for brush in ('tape', 'untape', 'layer_drop', 'layer_order')) or ('tool' in z and any(brush in z for brush in MAZResponseBrush)):
            response[number] = ('граница', None, False, z)
        elif 'palette' in z:
            response[number] = ('замес', None, False, z)
        elif job is not None:
            response[number] = ('загрузка', ('имя', z[job]), job != 'load_brush', z)
        elif isinstance(z.get('brush'), str) and not any(brush in z for brush in MAZResponseBrush):
            # строка кисти: подобранное с холста сбрасывают только "wipe" и "dirt" (вид и набор — нет)
            response[number] = ('кисть', ('имя', z['brush']), bool(z.get('wipe')) or 'dirt' in z, z)
        elif not any(brush in z for brush in MAZResponseBrush):
            response[number] = ('сушка', None, False, z)
        else:
            if isinstance(z.get('brush'), str):
                option = ('имя', z['brush'])
            else:
                r = round(float(z.get('size', 12.0)) * scale, 2)
                option = ('кэш', z.get('preset', preset), r, None if z.get('brush_tone') is None else float(z['brush_tone']))
            response[number] = ('мазок', option, bool(z.get('wipe')), z)
    for number, (kind, option, wipes, z) in list(response.items()):      # живая краска: вытирание грязи запаса не трогает
        if kind in ('мазок', 'кисть') and option[0] == 'имя' and option[1] in live:
            response[number] = (kind, option, False, z)
    return response


def _affected(parsing, edited, frames_was, forecast, full, scale=1.0):
    """Вперёд по журналу: затронутые мазки (номера), область затронутого (рамки), кисти, грязные к концу."""
    D, dirty, mixes, A = [], set(), set(), set()
    for number in sorted(parsing):
        kind, option, wipes, z = parsing[number]
        edited = number in edited
        if kind == 'замес':
            if edited:
                mixes.add(z['palette'])
            else:
                mixes.discard(z['palette'])
        elif kind == 'сушка' and edited:
            response = z.get('area')
            D.append(full if response is None else tuple(int(math.floor(float(v) * scale)) + (1 if i > 1 else 0) for i, v in enumerate(response)))
        elif kind in ('кисть', 'загрузка'):
            if edited or (kind == 'загрузка' and option in dirty):
                dirty.add(option)
            elif kind == 'кисть' and wipes:
                dirty.discard(option)
        elif kind == 'мазок':
            rr = [r for r in frames_was.get(number, []) + forecast.get(number, []) if r is not None]
            if 'dry' in z or 'set' in z:
                if edited:
                    D.append(full)
            by_brushes = not wipes and option in dirty
            by_mix = isinstance(z.get('swatch'), str) and z['swatch'] in mixes or (
                isinstance(z.get('swatch'), list) and any(and_ in mixes for and_ in z['swatch'] if and_))
            if edited or by_brushes or by_mix or _crosses_any(rr, D):
                A.add(number)
                D.extend(rr)
                dirty.add(option)
            else:
                dirty.discard(option)
    return A, D, dirty


def _crosses_any(rr, D):
    return any(_crosses(r, D) for r in rr)


def _dependent(parsing, A, frames):
    """Назад по журналу: что ещё класть, чтобы затронутые легли как в полном повторе (рамки и кисти раньше них)."""
    R, N, needed = set(), [], set()
    for number in sorted(parsing, reverse=True):
        kind, option, wipes, z = parsing[number]
        if kind == 'кисть':
            if wipes:
                needed.discard(option)
            continue
        if kind == 'удалён':                  # удалённый мазок: под его прежней рамкой — то, что лежало до него
            if number in A:
                N.extend(r for r in frames.get(number, []) if r is not None)
            continue
        if kind != 'мазок':
            continue
        rr = [r for r in frames.get(number, []) if r is not None]
        if number in A or option in needed or _crosses_any(rr, N):
            R.add(number)
            N.extend(rr)
            if wipes:
                needed.discard(option)
            else:
                needed.add(option)
    return R


# ------------------------------------------------------------------ пересборка
def _params(width, height, ground, scale, preset, seed):
    return {'physics': FAndZAndBrushA, 'ширина': width, 'высота': height, 'грунт': [float(v) for v in ground],
            'масштаб': float(scale), 'пресет': preset, 'сид': int(seed), 'отбор': None}


def _end_lines(lines):
    return max((i for i, with_ in enumerate(lines, 1) if with_.strip() and not with_.strip().startswith('#')), default=0)


class _Honest(Exception):
    pass


def rebuild(old, new, folder, edited, width=4800, height=3000, ground=(150, 141, 126), scale=1.0,
                preset=None, seed=0, cores=None, print=None, end_state=None, frames=None, framing_to=False):
    """Пересобрать правленый журнал (`новые` — строки) после `старые`: затронутое — поверх холста конца прежнего,
    точка конца и рамки — в `папка`. Живой холст (`папка` None): конец прежнего — `конец_сост` (состояние повтора:
    'холст', 'кисти', …), рамки его мазков — `рамки`, основа — чистый холст, точек нет. кадр_до — в числах '_до':
    (рамка, картинка) холста до правки по рамке затронутого с полем 16 px. Ответ — (состояние конца, словарь чисел,
    рамка затронутого или None); рамки мазков правленого журнала — в состоянии, 'рамки_мазков'."""
    t0 = time.perf_counter()
    tell = print or (lambda s: None)
    khs, khn = _hashes_lines(old), _hashes_lines(new)
    params = _params(width, height, ground, scale, preset, seed)
    edited = sorted(set(edited))
    f = edited[0] if edited else len(new) + 1
    strokes_total = sum(1 for with_ in new if any(f'"{brush}"' in with_ for brush in MAZResponseBrush) and '"note"' not in with_)
    numbers = {'правлено': len(edited), 'всего': strokes_total}

    def honest(why):
        if folder is None:                # живой холст: повтор правленого журнала целиком, рамки — по пути
            state_ = {'рамки': {}}
            p_.repeat(new, width, height, ground, scale, preset, seed, state=state_, cores=cores)
            state_['рамки_мазков'] = state_.pop('рамки')
            start = 1
        else:
            canvas, n, state_, start = collect(new, folder, width, height, ground, scale, preset, seed, resume=True, cores=cores)
        numbers.update(recomputed=sum(1 for i, with_ in enumerate(new, 1) if i >= start and any(f'"{brush}"' in with_ for brush in MAZResponseBrush)
                                     and '"note"' not in with_),
                     how=f'честный повтор: {why}', base=start - 1, area=1.0)
        return state_, None

    points = [] if folder is None else [t for t in points_folders(folder) if t[2].get('параметры') == params]
    end_with = _end_lines(old)
    end = [t for t in points if t[0] == end_with and t[2].get('хеш') == khs[end_with]]
    if end_state is not None:
        end = [(end_with, None, None)]
    base = [t for t in points if t[0] < f and t[0] < len(khn) and t[2].get('хеш') == khn[t[0]]]
    base = max(base, key=lambda t: t[0]) if base else None
    s0 = base[0] if base else 0
    try:
        if not end:
            raise _Honest('нет точки конца прежнего журнала (собрать его повтором с точками)')
        parsing = _parsing(new, s0 + 1, preset, scale)
        was = _parsing(old, s0 + 1, preset, scale)
        for number, d in list(parsing.items()) + list(was.items()):
            if d[0] == 'граница' or (d[0] in ('пометка', 'холст') and number in edited):
                raise _Honest(f'строка {number}: {"скотч, слои, нож или снятие" if d[0] == "граница" else "правка пометки или холста"}'
                              ' — зависимость не по рамкам')
        frames_was = dict(frames) if folder is None else _read_frames(folder, khs)
        no = [n for n, d in was.items() if d[0] == 'мазок' and n not in frames_was]
        if no:
            raise _Honest(f'рамок прежнего повтора нет у {len(no)} мазков (строка {no[0]}) — '
                          + ('холст взят из .npz или мазки легли мимо этого холста' if folder is None else 'повтор с точками их пишет'))
        header = next((json.loads(with_) for with_ in new if '"canvas"' in with_ and not with_.strip().startswith('#')), {})
        sh, in_ = header.get('canvas', (width, height))
        full = (0, 0, int(sh * scale), int(in_ * scale))
        for n in edited:                                   # удалённый мазок — его прежняя рамка и кисть
            if n in was and was[n][0] == 'мазок' and parsing.get(n, ('',))[0] != 'мазок':
                parsing[n] = ('удалён',) + was[n][1:]
        forecast = {}
        for n in edited:
            if n in parsing and parsing[n][0] == 'мазок':
                rr = [r for r in frames_was.get(n, []) if r is not None]
                zn, zs = parsing[n][3], was.get(n, (None,) * 4)[3] or {}
                sd = _shift_lines(zs, zn, scale)
                if sd is not None and rr:
                    rr = rr + [(r[0] + sd[0], r[1] + sd[1], r[2] + sd[0], r[3] + sd[1]) for r in rr]
                forecast[n] = rr
        for attempt in range(4):
            parsing_A = {n: (('мазок',) + d[1:] if d[0] == 'удалён' else d) for n, d in parsing.items()}
            A, D, dirty = _affected(parsing_A, set(edited), frames_was, forecast, full, scale)
            frames_R = {n: [r for r in frames_was.get(n, []) + forecast.get(n, []) if r is not None] for n in parsing}
            R = _dependent(parsing, A, frames_R)
            past = {n for n, d in parsing.items() if d[0] == 'мазок' and n not in R}
            if attempt == 0 and len(R) >= ALMOST_EVERYTHING * max(len(R) + len(past), 1):
                raise _Honest(f'затронуто почти всё ({len(R)} мазков из {len(R) + len(past)} после основы) — честный повтор '
                              'дешевле прохода со склейкой')
            response = _pass(new, base, s0, past, params, cores, frames_was)
            if response is None:
                raise _Honest('мягкой кисти не видно сырого слоя вне непосчитанного')
            state, frames_n = response
            new = False
            for n in R:
                fact = [r for r in frames_n.get(n, []) if r is not None]
                if n in A:
                    if not all(any(q[0] <= r[0] and q[1] <= r[1] and r[2] <= q[2] and r[3] <= q[3] for q in frames_R[n]) for r in fact):
                        forecast[n] = forecast.get(n, []) + fact
                        new = True
                elif fact != [r for r in frames_was.get(n, []) if r is not None]:
                    raise _Honest(f'строка {n}: рамка незатронутого мазка разошлась с прежней')
            if not new:
                break
            tell(f'правка: рамки затронутого шире прежних — попытка {attempt + 2}')
        else:
            raise _Honest('рамки затронутого не сошлись за 4 попытки')
        result = end_state if end_state is not None else load_file_state(end[0][1])
        if framing_to:
            rr = [r for r in D if r[2] > r[0] and r[3] > r[1]]
            if rr:
                numbers['_до'] = _framing(result['холст'], (min(r[0] for r in rr), min(r[1] for r in rr),
                                                     max(r[2] for r in rr), max(r[3] for r in rr)))
        frame = _add_up(result, state, D, dirty)
    except _Honest as e:
        tell(f'правка: {e} — честный повтор')
        if framing_to and end and end[0][1] is not None:     # «до» — весь лист с точки конца прежнего, пока она есть
            canvas0 = load_file_state(end[0][1])['холст']
            numbers['_до'] = _framing(canvas0, (0, 0, canvas0.width, canvas0.height), 0)
        state, _ = honest(str(e))
        numbers['время'] = time.perf_counter() - t0
        numbers['доля'] = numbers['пересчитано'] / max(strokes_total, 1)
        return state, numbers, None
    frames = dict(frames_was)
    for n in list(frames):
        if n > s0 and (n not in parsing or parsing[n][0] != 'мазок'):
            del frames[n]
    for n in edited:
        frames.pop(n, None)
    frames.update({n: v for n, v in frames_n.items() if n is not None})
    for n in A - set(frames_n):
        frames.pop(n, None)
    result['строка'] = state['строка']
    result['рамки_мазков'] = frames
    if folder is not None:
        for line, path, d in points_folders(folder):         # точки после правки — не этого журнала
            if line >= f:
                _remove([path[:-4] + '.json', path])
        _write_point(folder, result['строка'], {brush: v for brush, v in result.items() if brush != 'рамки_мазков'}, khn[result['строка']],
                        params, 'конец журнала (правка)')
        _write_frames(folder, khn, frames)
    area = _area(D, full)
    numbers.update(recomputed=len(R), how='затронутое', base=s0, area=area, time=time.perf_counter() - t0,
                 share=len(R) / max(strokes_total, 1))
    return result, numbers, frame


def _shift_lines(zs, zn, scale):
    for brush in MAZResponseBrush:
        if brush in zs and brush in zn and len(zs[brush]) == len(zn[brush]) and zs[brush]:
            dx, dy = (zn[brush][0][0] - zs[brush][0][0]) * scale, (zn[brush][0][1] - zs[brush][0][1]) * scale
            return (int(math.floor(dx)), int(math.floor(dy))) if (dx or dy) else None
    return None


def _pass(lines, base, s0, past, params, cores, frames_was):
    """Повтор правленого журнала с основы: мазки `мимо` — учтены, не положены. Ответ (состояние, рамки) или None —
    мягкой кисти ответ «есть ли в слое сырое» не виден вне рамок непосчитанного."""
    state = load_file_state(base[1]) if base else {}
    frames, skipped, doubt = {}, [], []
    parsing = {}

    def before(number, canvas, brush, layer):
        if getattr(brush, 'кайма', 0.0) > 0.0 and brush.shape != 'мастихин' and not doubt:
            canvas._boundary()
            with_ = canvas._layers.get(layer)
            w = None if with_ is None else with_._moisture
            if w is None:
                doubt.append(number)
                return
            wet = np.array(w[0:canvas.height, 0:canvas.width]) > 0
            for k in skipped:
                for r in frames_was.get(k, []):
                    if r is not None:
                        wet[r[1]:r[3], r[0]:r[2]] = False
            if not wet.any():
                doubt.append(number)

    skipped.extend(sorted(past))       # консервативно: рамки всех непосчитанных мазков после основы
    state.update(frames=frames, past=past, before=before)
    p_.repeat(lines, params['ширина'], params['высота'], params['грунт'], params['масштаб'],
                 params['пресет'], params['сид'], state=state, start=s0 + 1, cores=cores)
    for brush in ('рамки', 'мимо', 'перед'):
        state.pop(brush, None)
    if doubt:
        return None
    return state, frames


def _add_up(result, state, D, dirty):
    """Холст конца прежнего журнала, в рамках затронутого — пересобранный; кисти, грязные от затронутого, — пересобранные;
    палитра, счётчики и сиды — пересборки (она шла по всем строкам)."""
    from .canvas import Layer
    khi, khp = result['холст'], state['холст']
    khp._boundary()
    if list(khi._layers) != list(khp._layers):
        raise _Honest(f'слои разошлись: {list(khi._layers)} и {list(khp._layers)}')
    rr = [r for r in D if r[2] > r[0] and r[3] > r[1]]
    mask = None
    for name, sp in khp._layers.items():
        si = khi._layers[name]
        for field in Layer._InWithE:
            a, b = getattr(si, field), getattr(sp, field)
            if b is None and a is None:
                continue
            if b is None or a is None:
                # массив по надобности (`пропуск`, `текучая`) заводит мазок, которому он нужен. Нет у пересборки, есть у
                # прежнего: его завели мазки вне пересчитанного (тогда вне затронутого он не пуст — берётся прежний, в
                # затронутом пуст) или только прежние затронутые (вне затронутого пуст — у полного повтора его нет).
                # Есть у пересборки, нет у прежнего: завели только затронутые — заводится пустым, в затронутом — новый.
                empty = Layer._PAtWithTResponse.get(field)
                if empty is None:
                    raise _Honest(f'слой «{name}»: массив «{field}» есть только в одном холсте')
                shape = (a if b is None else b).shape[:2]
                if mask is None or mask.shape != shape:
                    mask = np.zeros(shape, bool)
                    for r in rr:
                        mask[max(r[1], 0):r[3], max(r[0], 0):r[2]] = True
                if b is None:
                    outside = np.asarray(a)[~mask]
                    if not (outside != empty).any():
                        setattr(si, field, None)              # завели только прежние затронутые: у полного повтора его нет
                        continue
                    for r in rr:
                        a[r[1]:r[3], r[0]:r[2]] = empty
                    continue
                a = khi._transmittance(si) if field == 'пропуск' else khi._flowing(si)
            for r in rr:
                a[r[1]:r[3], r[0]:r[2]] = b[r[1]:r[3], r[0]:r[2]]
    for field in ('_повторы', '_счётчик', '_этап', '_объект', '_замесы'):
        if field in khp.__dict__:
            setattr(khi, field, copy.deepcopy(khp.__dict__[field]))
    for option, brush in state['кэш'].items():
        my = ('кэш', option[0], option[1], option[2] if len(option) > 2 else None)
        if my in dirty or option not in result['кэш']:
            result['кэш'][option] = brush
    for name, brush in state.get('кисти', {}).items():
        if ('имя', name) in dirty or name not in result.get('кисти', {}):
            result.setdefault('кисти', {})[name] = brush
    if not rr:
        return None
    return (min(r[0] for r in rr), min(r[1] for r in rr), max(r[2] for r in rr), max(r[3] for r in rr))


def _area(D, full):
    if not D:
        return 0.0
    sh, in_ = full[2], full[3]
    m = np.zeros((in_, sh), bool)
    for r in D:
        m[max(int(r[1]), 0):int(r[3]), max(int(r[0]), 0):int(r[2])] = True
    return float(m.mean())


def _framing(canvas, frame, field=16):
    """Холст в рамке (с полем, по холсту) со светом, sRGB uint8: (рамка с полем, картинка)."""
    from .eye import _kind
    x0, y0, x1, y1 = frame
    r = (max(x0 - field, 0), max(y0 - field, 0), min(x1 + field, canvas.width), min(y1 + field, canvas.height))
    return r, np.round(_kind(canvas, r, True)).astype(np.uint8)


def to_after(to, became, path):
    """PNG «до | после» 1:1: до — (рамка, картинка) от `_кадр`, стало — холст после правки."""
    from .variants import _png
    r, a = to
    b = _framing(became, r, 0)[1]
    seam = np.full((a.shape[0], 4, 3), 255, np.uint8)
    _png(path, np.concatenate([a, seam, b], 1))
    return path


def edit(journal, out=None, folder=None, width=4800, height=3000, ground=(150, 141, 126), scale=1.0, preset=None,
            seed=0, cores=None, print=None, path_png=None, **edit):
    """Правка журнала-файла по фильтру и пересборка затронутого (шапка модуля). папка — точки повтора (без неё —
    `<out>.точки/`); out — PNG листа после правки. `**правка` — доводы `править_строки`. Ответ — `Правка`."""
    if folder is None:
        if out is None:
            raise ValueError('правка: нужна папка точек или out (точки — в <out>.точки/)')
        folder = os.path.splitext(out)[0] + '.точки'
    with open(journal, encoding='utf-8') as f:
        old = f.readlines()
    new, edited, what = edit_lines(old, **edit)
    if not edited:
        raise ValueError(f'правка: под фильтр {what} не попал ни один мазок — журнал не тронут')
    if new and not new[-1].endswith('\n'):
        new[-1] += '\n'
    new.append(json.dumps({'note': 'правка задним числом', 'edit': what, 'lines': len(edited)}, ensure_ascii=False) + '\n')
    reserve = _reserve(journal)
    os.replace(journal, reserve)
    with open(journal, 'w', encoding='utf-8') as f:
        f.writelines(new)
    state, numbers, frame = rebuild(old, new, folder, edited, width, height, ground, scale, preset, seed, cores, print,
                                     framing_to=bool(path_png))
    state.pop('рамки_мазков', None)
    to = numbers.pop('_до', None)
    response = Edit(numbers, reserve=reserve, journal=journal, edited=edited)
    response['рамка'] = frame
    if out:
        state['холст'].png(out)
    if to is not None:
        response['путь'] = to_after(to, state['холст'], path_png)
    response['состояние'] = state
    return response


def _transfer(canvas, khp):
    """Честный повтор живого холста: поля слоёв, слои и счёт повтора `хп` — в холст `х` (на месте: его процессы и
    общая память те же)."""
    from .canvas import Layer, _in_shared
    canvas._boundary(writing=True)
    khp._boundary()
    shared = canvas.__dict__.get('_ядер', 1) != 1
    layers = {}
    for name, sp in khp._layers.items():
        si = canvas._layers.get(name)
        if si is None:
            si = sp
            if shared:
                for field in Layer._InWithE:
                    setattr(si, field, _in_shared(getattr(si, field)))
        else:
            for field in Layer._InWithE:
                a, b = getattr(si, field), getattr(sp, field)
                if b is None:
                    setattr(si, field, None)
                elif a is None or a.shape != b.shape:
                    setattr(si, field, _in_shared(b) if shared else b)
                else:
                    a[...] = np.asarray(b)
        layers[name] = si
    canvas._layers = layers
    for field in ('_повторы', '_счётчик', '_этап', '_объект', '_замесы', '_ленты'):
        if field in khp.__dict__:
            setattr(canvas, field, copy.deepcopy(khp.__dict__[field]))
        else:
            canvas.__dict__.pop(field, None)


def edit_canvas(canvas, cores=None, path_png=None, print=None, **edit):
    """`Холст.править`: правка журнала живого холста (`журнал=`) по фильтру и пересборка затронутого на нём самом.
    Рамки мазков холст помнит сам по номеру строки журнала; основа — чистый холст (точек нет), мазки вне затронутого
    только учитываются. Кисти пользователя — те же объекты: что на них после правки, — как в полном повторе."""
    zh = getattr(canvas, 'журнал', None)
    if zh is None:
        raise ValueError('править: у холста нет журнала — правка идёт по нему (Холст(…, журнал=путь)); '
                         'журнал-файл — python -m кисть.задним')
    if canvas.__dict__.get('_только_в') is not None:
        raise ValueError('править: в куске `варианты` нельзя — править до варианты или после взять')
    canvas._boundary(writing=True)
    zh._f.flush()
    with open(zh.path, encoding='utf-8') as f:
        old = f.readlines()
    new, edited, what = edit_lines(old, **edit)
    if not edited:
        raise ValueError(f'править: под фильтр {what} не попал ни один мазок — журнал не тронут')
    if new and not new[-1].endswith('\n'):
        new[-1] += '\n'
    new.append(json.dumps({'note': 'правка задним числом', 'edit': what, 'lines': len(edited)}, ensure_ascii=False) + '\n')
    brushes = {z[1]: z[0] for z in zh.brushes.values()}
    end = {'холст': canvas, 'кэш': {}, 'кисти': dict(brushes), 'масштаб': 1.0, 'сид': 0, 'пресет': None,
             'строка': _end_lines(old)}
    frames = {n: v for n, v in canvas.__dict__.get('_рамки', {}).items() if isinstance(n, int)}
    to = None
    state, numbers, frame = rebuild(old, new, None, edited, canvas.full_width, canvas.full_height, canvas.ground_srgb, 1.0,
                                     None, 0, cores, print, end_state=end, frames=frames, framing_to=bool(path_png))
    if state is not end:                             # честный повтор: его холст — в этот
        if path_png:
            to = _framing(canvas, (0, 0, canvas.width, canvas.height), 0)
        _transfer(canvas, state['холст'])
    for name, brush in state.get('кисти', {}).items():      # кисти пользователя — те же объекты, с тем, что на них теперь
        mine = brushes.get(name)
        if mine is not None and brush is not mine:
            brush._remainder, brush._charged                       # чтение досчитывает мазки кисти на ядрах
            for field in {p for p in (*brush.__dict__, *mine.__dict__) if p.startswith('_')}:      # грязь, краска, набор — как лежат
                if field in brush.__dict__:
                    mine.__dict__[field] = brush.__dict__[field]
                else:
                    mine.__dict__.pop(field, None)
    canvas._frames = {n: v for n, v in state.get('рамки_мазков', {}).items() if n is not None}
    canvas.__dict__.pop('_метка', None)
    canvas._snapshots = []                                    # снимки до правки — не этого журнала
    canvas.__dict__['_правок'] = canvas.__dict__.get('_правок', 0) + 1
    reserve = _reserve(zh.path)
    zh._f.close()
    os.replace(zh.path, reserve)
    with open(zh.path, 'w', encoding='utf-8') as f:
        f.writelines(new)
    zh._f = open(zh.path, 'r+b')
    zh._f.seek(0, os.SEEK_END)
    zh.lines = len(new)
    if numbers.get('_до') is not None:
        to = numbers.pop('_до')
    numbers.pop('_до', None)
    response = Edit(numbers, reserve=reserve, journal=zh.path, edited=edited)
    response['рамка'] = frame
    if path_png and to is not None:
        response['путь'] = to_after(to, canvas, path_png)
    return response


def main(argv=None):
    p = argparse.ArgumentParser(prog='python -m кисть.задним', description='Правка журнала задним числом и пересборка затронутого.')
    p.add_argument('журнал')
    p.add_argument('--out', required=True, help='PNG листа; точки повтора — <out>.точки/')
    p.add_argument('--checkpoints', default=None)
    p.add_argument('--lines'); p.add_argument('--layer'); p.add_argument('--stage'); p.add_argument('--object')
    p.add_argument('--swatch'); p.add_argument('--area', nargs=4, type=float)
    p.add_argument('--delete', action='store_true'); p.add_argument('--color', nargs=3, type=float)
    p.add_argument('--set-swatch'); p.add_argument('--dL', type=float, default=0.0); p.add_argument('--dC', type=float, default=0.0)
    p.add_argument('--dh', type=float, default=0.0); p.add_argument('--warmer', type=float, default=0.0)
    p.add_argument('--move', nargs=2, type=float); p.add_argument('--load'); p.add_argument('--pressure', type=float)
    p.add_argument('--pressure-mul', type=float); p.add_argument('--size', type=float)
    p.add_argument('--scale', type=float, default=1.0); p.add_argument('--preset'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cores', type=int); p.add_argument('--before-after', default=None, help='PNG «до | после» по рамке затронутого')
    a = p.parse_args(argv)
    load = a.load
    if load is not None:
        try:
            load = float(load)
        except ValueError:
            pass
    response = edit(a.journal, out=a.out, folder=a.checkpoints, scale=a.scale, preset=a.preset, seed=a.seed, cores=a.cores,
                    path_png=a.before_after, print=lambda s: print(s, file=sys.stderr),
                    lines_with=a.lines, layer=a.layer, stage=a.stage, object=a.object, mix=a.swatch, region=a.area,
                    delete=a.delete, color=a.color, new_mix=a.set_swatch, dL=a.dL, dC=a.dC, dh=a.dh, warmer=a.warmer,
                    shift=a.move, load=load, pressure=a.pressure, pressure_in=a.pressure_mul, width=a.size)
    print(response)


if __name__ == '__main__':
    main()
