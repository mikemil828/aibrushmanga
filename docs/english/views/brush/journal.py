# READING VIEW of движок/кисть/журнал.py with names in English. Not runnable; edit the original.
"""Журнал пишется сам (А-1): `Холст(…, журнал='лист.jsonl')` — каждый вызов холста и кисти, от которого зависят
пиксели, идёт строкой JSONL в том виде, в каком её читает `кисть.повтор`. Своего кода записи модели не нужно.

Строки, которые пишет холст:
    {"canvas": [ш, в], "ground": [r, g, b], "px_per_mm": …, "grain": [шаг мм, глубина, сид], "physics": ФИЗИКА}
    {"brush": "к1", "def": {все поля Кисть}}             — кисть, как она есть (без файлов пресетов); снова — если поля менялись
    {"brush": "к1", "wipe": true}                          — кисть вытерли (`кисть.вытереть()`)
    {"brush": "к1", "dirt": [[…], […]]}                    — что на кончиках, если кисть испачкали вне этого холста
    {"brush": "к1", "pick": [r, g, b], "layout": "полосы"} — `кисть.набрать(…)` ("pick": null — набранного нет)
    {"palette": "охра", "color": [r, g, b], "paint": …}   — `х.палитра`: замес завели или поправили (М-1); мазок и набор
                                                             замесом — с полем "swatch" (имя или список имён)
    {"brush": "к1", "layer": …, "color": …, "core": [[x, y, нажим, dt, …], …], "seed": …, …}   — `мазок`: точки ядра как есть
    {"tool": "тряпка", "size": …, "core": …, "seed": …}    — `снять` ("тряпка", "скребок", "черенок")
    {"dry": 0.5, "layer": …, "area": […]} · {"set": 0.5, …} · {"wet": 1.0, …} — `подсушить`, `схватить`, `размочить`
    {"tool": "нож", "size": …, "core": …, "depth": …}   — `прижать` (срез высохшего ходом ножа)
    {"mark": "текст", "stage": …, "object": …, "intent": …} — `пометка`: этап или объект; повтор не исполняет
    {"note": …}                                            — `заметка`: что угодно JSON (отвергнутое, замеры); повтор пропускает
    {"saved": "этап4.npz", "counter": …, "brushes": …}     — `сохранить`: отсюда `Холст.загрузить(…, журнал=)` пишет дальше
Состояние кисти пишется перед мазком, который его видит: строк wipe, dirt, pick столько, сколько нужно повтору.
У каждой строки из пометки берутся "stage" и "object" (пока не сменит следующая пометка), у мазка — ещё "intent".

Откат (`Холст.откатить`) обрезает файл до места снимка (А-5): проба с откатом журнал не меняет. `Холст.загрузить(…,
журнал=)` обрезает журнал до строки "saved" и пишет дальше; срезанный хвост (М-12) — в `<журнал>.до_отката` (`.2`, … —
старое не затирается) и строкой в stderr."""
import json
import os
import sys
from dataclasses import asdict

import numpy as np

from .tool import wet_field, Loading

FAndZAndBrushA = 'Долг-ф'    # версия физики мазка: правка физики, что меняет пиксели принятых журналов, меняет и её (И-4)
# И-4: версии физики по порядку и что каждая поменяла в пикселях; новая версия — строкой в конец. До Ф-3 версию
# не поднимали: журналы всех правок от `3dc98a6` до `b46970c` несут в заголовке `3dc98a6` (имена — их коммиты).
HISTORY_PHYSICS = (
    ('3dc98a6', 'первая версия в заголовке журнала'),
    ('a96081b', 'У-1: вытертая кисть (нагрузка 0) по сырому растушёвывает без цвета набора'),
    ('0c942e4', 'Г-3: вытертая мягкая кисть ровняет сырое к бороздкам своего волоса'),
    ('f733f9a', 'Г-2: свет картинки — белила под светом не серят'),
    ('781bc69', 'касание: тычок — отпечаток пучка, ход короче ширины кисти — во всю ширину'),
    ('1902811', 'касание-2: тычок узкой кисти с острым кончиком — пятном'),
    ('bca5b96', 'Д-1-2: набранная кисть (`набрать`) — полная кисть один объём, сходит иначе'),
    ('6cf60bc', 'Р-1: запас ворса набранной кисти в мм — на холсте не 8 px/мм кисть сходит на другом пути'),
    ('b46970c', 'Ф-2: текучий ход (`ровно`, разбавитель) по сухой краске затекает в её низины'),
    ('Ф-3', 'Ф-3: ход не ровняет сухую краску своего слоя под собой (рельеф сухого стоит); корку и полусухое — слабее'),
    ('нож-1', 'мастихин: нож считается своим ядром (гладкая плоскость, вал по ходу, валики у боков, ком на отрыве, по сухому — только новая краска); сырая краска, согнанная кистью на соседнее место, несёт влажность места, откуда пришла'),
    ('тычок-2', 'Н-2: тычок круглой — круг пучка с бахромой кончиков (не стопка прядей); короткий ход до 1,5 ширины — кисть встаёт во всю ширину и под «остановкой», без тонкого места штрихом и ухода прядей с первого касания'),
    ('З-1', 'З-1: вытертая мягкая кисть (нагрузка 0) у края краски своего слоя и за ним — за краем не кладёт ничего (была плёнка без цвета по всему следу — серое на светлом нижнем слое); входя в краску с пустого места, не несёт в неё серого (капля брала цвет первого пикселя кромки); краска, согнанная ею на почти пустое место, несёт свой цвет. В сыром слое под всем следом — как было'),
    ('Д-7', 'Д-7: у пучка плоской кисти есть толщина (поле кисти `пучок_мм`; без поля — 0,225 ширины, 0,5–3 мм): на ребре (`касание` < 1) кисть повёрнута ребром к ходу — след сплошной полосой толщины пучка, ширится с нажимом и наклоном, ворс вдоль хода (были нити со щелью и до всей ширины к нажиму 0,85: клин ребром по нажиму); то же — `"rotation_along"` 90° (была линия 2 px); тычок плоской от `касания` не зависит. Плашмя, и на дугах, — как было бит в бит; прежнее ребро — `пучок_мм` 0'),
    ('К-14а', 'К-14а: новый инструмент снятия — черенок (`"tool": "черенок"`): твёрдый круглый конец раздвигает сырое в валики, объём на холсте тот же. Пиксели прежних журналов те же бит в бит: прежний движок строки с черенком не знает'),
    ('К-14б', 'К-14б: кисть из пучков — поля кисти `просветы`, `в_пучке`, `веер` (пресеты `веерная`, `растрёпанная`): ворс стоит пучками с просветами, ход кладёт линии пучков, тычок — их чёрточки; строка хода такой кистью несёт линии полем "lines". Пиксели прежних журналов те же бит в бит: без этих полей кисть — как была; прежний движок полей не знает'),
    ('К-14б-2', 'К-14б-2: тычок кистью из пучков с толщиной пучка (`растрёпанная`) — пятна клоков: пучок ложится всей толщиной, как плоская той же ширины, и расходом своих кончиков сверх неё (были крапины по 2–8 px: толщина пучка в длину отпечатка не шла). Ходы такой кистью, веерная (`пучок_мм` 0) и кисти без `просветы` — как были бит в бит'),
    ('Долг-ф', 'Долг-ф: живая краска (ход без цвета после `набрать`) расходится в ворсе по соседним гребням, как у скупо набранной (`запас`), и кончик кормится из ворса в меру остатка — кисть пустеет вся, а не серединой (были «рельсы»: две нити по краям и пустая середина), исход — сухой кистью по всей ширине; тычок круглой — без запаса гребня вдоль хода (бедный гребень клал щель через точку: узкая круглая 5–6 px — кольцом или «буквой» с дыркой); почти пустая мягкая кисть с краской у края краски слоя — цвет пустого места слоя и плёнки в тысячные мазка в каплю не идёт (как у вытертой, З-1; были серые крапины). Ходы с цветом плоской, круглой и щетины, тычки других кистей — как были'),
)


def difference_physics(version):
    """Что сказать, когда журнал записан физикой `версия`, а у движка `ФИЗИКА`: что поменялось с тех пор."""
    names = [in_ for in_, _ in HISTORY_PHYSICS]
    if version not in names:
        return (f'журнал записан физикой {version}, у движка {FAndZAndBrushA}: этой версии движок не знает (журнал с другого '
                'движка или новее этого) — пиксели могут разойтись с листом')
    after = HISTORY_PHYSICS[names.index(version) + 1:]
    return (f'журнал записан физикой {version}, у движка {FAndZAndBrushA}: пиксели разойдутся с листом; с тех пор — '
            + '; '.join(f'{in_}: {what}' for in_, what in after))


def _in_json(v):
    """Числа numpy и кортежи -> то, что json пишет точно (float — repr, назад тот же бит)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_in_json(x) for x in v]
    if isinstance(v, dict):
        return {str(brush): _in_json(x) for brush, x in v.items()}
    return v


def _kind(rgb):
    kind = getattr(rgb, 'вид', None)
    return None if kind is None else _in_json(kind)


def color_lines(color):
    """Цвет вызова (`мазок`, `набрать`) -> поля строки журнала: "color" или "colors" и "paint" (вид краски одной или
    списком — по краске набора). None — "color": null (мазок набранной краской)."""
    if color is None:
        return {'color': None}
    from .palette import names_colors
    names = names_colors(color)         # М-1: мазок замесом палитры — рядом с цветом имя ("swatch")
    d = _color_lines(color)
    if names is not None:
        d['swatch'] = names
    return d


def _color_lines(color):
    ts = list(color)
    from .tool import _number
    if ts and _number(ts[0]):
        d = {'color': [_in_json(v) for v in ts]}
        kind = _kind(color)
        if kind is not None and kind != 'кроющая':
            d['paint'] = kind
        return d
    paints, views = [], []
    for e in ts:
        e = list(e)
        if len(e) == 2 and not _number(e[0]):
            rgb, share = e
        elif len(e) == 4:
            rgb, share = e[:3], e[3]
        else:
            rgb, share = e, 1.0
        paints.append([_in_json(v) for v in list(rgb)[:3]] + [_in_json(share)])
        views.append(_kind(rgb) or 'кроющая')
    d = {'colors': paints}
    if any(in_ != 'кроющая' for in_ in views):
        d['paint'] = views[0] if all(in_ == views[0] for in_ in views) else views
    return d


class Journal:
    """Файл журнала одного холста. Пишет строки, помнит кисти (имя, поля, что на кончиках по журналу), место для отката."""

    def __init__(self, path, canvas, resume=None):
        self.path = os.fspath(path)
        self.pause = 0
        self.brushes = {}          # id(кисть) -> [кисть, имя, поля, остаток по журналу, краска по журналу]
        self.number = 0           # последняя выданная кисть «кN»
        self.lines = 0           # К-3: строк в файле — номер последней (мазок помнит по нему свою рамку, `Холст.править`)
        self.stage = self.object = None
        if resume is None:
            self._f = open(self.path, 'wb')
            header = {'canvas': [canvas.full_width, canvas.full_height], 'ground': list(canvas.ground_srgb)}
            if canvas.px_on_mm is not None:
                header['px_per_mm'] = canvas.px_on_mm
            if canvas.grain is not None:
                header['grain'] = list(canvas.grain)
            header['physics'] = FAndZAndBrushA
            if canvas.seed:
                header['seed'] = canvas.seed
            header['seeds'] = 'tree'      # А-9: строка без "seed" берёт сид из листа, этапа, объекта и своих точек
            self.row(header)
        else:
            self._resume(*resume)

    def _resume(self, npz, counter):
        """Журнал холста, взятого из `npz`: обрезать после строки {"saved": имя файла, "counter": счётчик} (последней
        такой) и писать дальше."""
        if not os.path.exists(self.path):
            raise ValueError(f'журнал {self.path}: файла нет — продолжать нечего')
        name = os.path.basename(os.fspath(npz))
        place, first = None, None
        with open(self.path, 'rb') as f:
            bytes = lines = 0
            for wet in f:
                bytes += len(wet)
                lines += 1
                if first is None:
                    first = json.loads(wet)
                if b'"saved"' in wet:
                    z = json.loads(wet)
                    if z.get('saved') == name and z.get('counter') == counter:
                        place = (bytes, z, lines)
        if (first or {}).get('physics') != FAndZAndBrushA:
            raise ValueError(f'журнал {self.path}: физика {(first or {}).get("physics")}, у движка {FAndZAndBrushA} — '
                             'продолжать этот лист этим движком нельзя')
        if place is None:
            raise ValueError(f'журнал {self.path}: нет строки {{"saved": "{name}"}} со счётчиком {counter} — '
                             'холст сохранён не с этим журналом или журнал обрезан раньше')
        bytes, z, self.lines = place
        self._save_tail(bytes, name)
        self._f = open(self.path, 'r+b')
        self._f.truncate(bytes)
        self._f.seek(bytes)
        self.number = int(z.get('brushes', 0))
        self.stage, self.object = z.get('stage'), z.get('object')

    def _save_tail(self, bytes, name):
        """М-12: строки журнала после точки сохранения `имя` (с байта `байт`) — до обрезки в `<журнал>.до_отката` (есть —
        `.до_отката.2`, `.3`, …: прежняя копия не затирается) байт в байт и строка в stderr. Хвоста нет — ничего."""
        if os.path.getsize(self.path) <= bytes:
            return None
        with open(self.path, 'rb') as f:
            f.seek(bytes)
            tail = f.read()
        copy, k = self.path + '.до_отката', 2
        while os.path.exists(copy):
            copy, k = f'{self.path}.до_отката.{k}', k + 1
        with open(copy, 'xb') as f:
            f.write(tail)
            f.flush()
            os.fsync(f.fileno())
        lines = sum(1 for s in tail.splitlines() if s.strip())
        sl = 'строка' if lines % 10 == 1 and lines % 100 != 11 else \
            'строки' if 2 <= lines % 10 <= 4 and not 12 <= lines % 100 <= 14 else 'строк'
        print(f'журнал обрезан до {name}: {lines} {sl} после неё — в {copy}', file=sys.stderr, flush=True)
        return copy

    # ------------------------------------------------------------------ запись
    def row(self, d):
        self._f.write(json.dumps(d, ensure_ascii=False, separators=(', ', ': ')).encode('utf-8') + b'\n')
        self._f.flush()
        self.lines += 1

    def _intent(self, d, stage=None, object=None, intent=None):
        e = self.stage if stage is None else stage
        response = self.object if object is None else object
        if e is not None:
            d['stage'] = _in_json(e)
        if response is not None:
            d['object'] = _in_json(response)
        if intent is not None:
            d['intent'] = _in_json(intent)
        return d

    def brush(self, brush):
        """Имя кисти в журнале; перед мазком — строки, без которых повтор не придёт к той же кисти: определение (первый
        раз и после правки полей), вытирание, грязь с другого холста, набранная краска."""
        z = self.brushes.get(id(brush))
        fields = asdict(brush)
        if z is None:
            self.number += 1
            z = [brush, f'к{self.number}', None, None, None]
            self.brushes[id(brush)] = z
        if z[2] != fields:
            self.row({'brush': z[1], 'def': _in_json(fields)})
            z[2] = fields
        kr = brush._paint
        if kr is not z[4] and (isinstance(kr, Loading) or isinstance(z[4], Loading)):      # Д-1: живая краска — своими строками
            if kr is None and brush._remainder is None:
                self.row({'rinse': z[1]})
                z[3] = z[4] = None
            elif isinstance(kr, Loading):
                charge = getattr(brush, '_набор', None)
                with_charge = bool(kr.jobs) and kr.jobs[0][0] in ('load', 'rinse') and charge is not None and charge[2] is kr   # дела начаты набором (или промывкой перед ним): от прежнего на кисти не зависят
                if with_charge or (kr.jobs and kr.from_ is z[4]):
                    for job in kr.jobs:
                        if job[0] == 'load':
                            fields_colors = color_lines(charge[0])
                            d = {'load_brush': z[1]}
                            d.update(fields_colors)
                            d.update(amount=_in_json(job[1]), layout=charge[1])
                            self.row(d)
                        else:
                            self.row({job[0]: z[1]})
                            if wet_field(brush, '_остаток') is None:
                                z[3] = None
                else:                             # загрузка пришла не делами после хода этого журнала (ход на другом холсте, откат чужого снимка): как есть
                    paints, layout = kr
                    self.row({'brush': z[1], 'pick_linear': [[_in_json(c), float(d)] for c, d in paints],
                                 'pick_cover': getattr(paints, 'мутность', None), 'layout': layout,
                                 'left': [_in_json(kr.reserve), _in_json(kr.drop)]})
            if isinstance(kr, Loading):
                z[4] = kr
        if wet_field(brush, '_остаток') is not z[3]:        # 2б: мазок этого холста, что ещё считается на ядрах, — тот же объект-метка
            rest = brush._remainder                             # (мазок другого холста — досчитывается)
            if rest is None:
                self.row({'brush': z[1], 'wipe': True})
            else:
                self.row({'brush': z[1], 'dirt': [_in_json(np.asarray(a, np.float64)) for a in rest]})
            z[3] = rest
        if brush._paint is not z[4]:
            charge = getattr(brush, '_набор', None)
            if brush._paint is None:
                self.row({'brush': z[1], 'pick': None})
            elif charge is not None and charge[2] is brush._paint:
                fields_colors = color_lines(charge[0])
                peak = {'brush': z[1], 'pick': fields_colors.pop('color', None) if 'color' in fields_colors else fields_colors.pop('colors')}
                peak.update(fields_colors)
                peak['layout'] = charge[1]
                self.row(peak)
            else:                                 # краска на кисти пришла не через `набрать` (откат чужого снимка): как есть
                paints, layout = brush._paint
                self.row({'brush': z[1], 'pick_linear': [[_in_json(c), float(d)] for c, d in paints],
                             'pick_cover': getattr(paints, 'мутность', None), 'layout': layout})
            z[4] = brush._paint
        return z[1]

    def after_stroke(self, brush):
        z = self.brushes[id(brush)]
        z[3], z[4] = wet_field(brush, '_остаток'), brush._paint

    def finished(self, brush, label, remainder):
        """2б: мазок кисти досчитан на ядрах — вместо его метки журнал помнит то, что осталось на кисти."""
        z = self.brushes.get(id(brush))
        if z is not None and z[3] is label:
            z[3] = remainder

    # ------------------------------------------------------------------ откат
    def place(self):
        """Где журнал сейчас: для снимка холста."""
        return (self._f.tell(), self.number, self.stage, self.object,
                {brush: list(z) for brush, z in self.brushes.items()}, self.lines)

    def roll_back(self, place):
        bytes, self.number, self.stage, self.object, brushes, self.lines = place
        self._f.truncate(bytes)
        self._f.seek(bytes)
        self._f.flush()
        self.brushes = {brush: list(z) for brush, z in brushes.items()}

    def saved(self, path, counter):
        """Строка {"saved": имя файла, "counter": счётчик мазков, "brushes": сколько кистей названо}: с неё
        `Холст.загрузить(…, журнал=)` продолжит журнал. Повтор её пропускает."""
        self.row(self._intent({'saved': os.path.basename(os.fspath(path)), 'counter': int(counter), 'brushes': self.number}))

    def close(self):
        f = getattr(self, '_ф', None)
        if f is not None and not f.closed:
            f.close()

    __del__ = close
