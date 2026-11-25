# READING VIEW of движок/кисть/инструмент.py with names in English. Not runnable; edit the original.
"""Кисть как инструмент: пресет, нагрузка, набранная краска; разбор цвета из вызова."""
import json
import os
from dataclasses import dataclass, fields, asdict

import numpy as np

from .fast import clamp



from .numbers import _FLOOR_FILM, _WithAtCanvasAYa
from . import numbers as H

NAGRAtZBrushAnd = {'протирка': 0.10, 'обычная': 0.28, 'пастозная': 0.90, 'сухая': _WithAtCanvasAYa[0]}   # 'сухая' (п. 10, инструмент 2) — полупустая кисть: `укладка`, `числа._СУХАЯ`


def scale_thicknesses(width):
    """Узкая кисть несёт меньше: толщина плёнки идёт как ширина в степени 0,8 (1 при 40 px). У кисти уже
    `_ПОЛ_ПЛЁНКИ` px (п. 10) плёнка дальше не тоньшает: тонкая кисть кладёт краску кончиком, слой у неё как у кисти
    этой ширины, иначе след в 3–6 px по сырому тонет в лежащей краске."""
    return (max(float(width), _FLOOR_FILM) / (40.0 * H._MM)) ** 0.8      # Д-8: 40 px и пол — px холста 8 px/мм (`мм.в_мм`)


def _after_strokes(brush, field):
    """Поле кисти, которое пишет мазок: если мазки этой кисти ещё считаются на ядрах — сперва досчитать их."""
    d = brush.__dict__
    waits = d.get('_ждёт_ядра')
    if waits is not None:
        waits.brush_brushes(brush)
    return d.get(field + '·')


def wet_field(brush, field):
    """То же поле как лежит, без досчёта мазков на ядрах (в нём может стоять метка мазка очереди) — для `ядра.py` и журнала."""
    return brush.__dict__.get(field + '·')


@dataclass
class Brush:
    name: str = 'кисть'
    shape: str = 'плоская'            # 'плоская' | 'круглая' | 'мастихин'
    width: float = 32.0              # px, ширина следа при полном нажиме
    load: object = 'обычная'      # 'протирка' | 'обычная' | 'пастозная' или число: толщина плёнки при ширине 40 px
    thickness: float = 1.0              # множитель к толщине от нагрузки (оставлен от версии 1)
    spacing: float = 2.5                  # px между ведущими волосками (шаг борозд), в среднем
    unevenness_spacing: float = 0.46     # разброс шага (лог-нормальный); 0 — гребёнка
    rows: int = 3                     # рядов ворса: больше — мельче борозды (задние ряды заглаживают)
    spread_lengths: float = 0.34        # на какую долю нажима короткая прядь касается позже длинной
    spread_stiffness: float = 0.45   # разброс нажима кончиков: от него глубина борозд разная
    reserve: float = 14.0               # на сколько ширин кисти хватает краски
    viscosity: float = 0.6             # 0..1: глубина борозд, сгусток на отрыве, падение отдачи на скорости
    pickup: float = 0.5               # 0..1: какую долю сырой краски под собой берёт волосок
    hiding_power: float = 1.0         # >1 — кроет более тонким слоем
    lag: float = 0.55          # поворот плоской кисти отстаёт на столько ширин пути
    splaying: float = 0.5       # доля ширины следа при нулевом нажиме
    scraps: float = 0.6               # 0..1: насколько пряди живут врозь (просветы, разная длина хвостов)
    spread_tone: float = 1.0         # недомешанная краска: 1 — соседние пряди расходятся на 3–6 уровней светлоты
    tone_brushes: float = 0.0            # 11а/11в: 0 (умолчание) — среднее тона по кисти равно набранной краске; 1 — вся порция ещё и светлее или темнее набранной (σ 7 %, до 17 %), просить явно
    dryness: float = 0.0              # сдвиг порога сухой кисти: 0 — нагруженная, 0.3 — полусухая
    bead: float = 0.6                # валик с одной стороны пастозного мазка и сгусток на отрыве
    carry_out: float = 1.0                # на сколько ширин кисть несёт подобранную краску
    brodyagi: float = 0.0              # (не используется в версии 2; оставлен, чтобы читались старые пресеты)
    touch: float = 1.0              # плоская кисть: 1 — плашмя (вся ширина и при лёгком нажиме), 0 — на ребре/уголке
    width_mm: object = None          # п. 9: ширина в мм — тогда `ширина` берётся из неё по px_на_мм холста; None — в px
    spacing_mm: object = None             # п. 9: шаг ведущих волосков в мм; None — `шаг` в px (не масштабируется)
    point: float = 1.0               # п. 12: узкая кисть (до 12 px, к 16 px сходит на нет) собрана в острый кончик: след идёт за нажимом, концы клином (травинка, линия); 0 — щетина в обойме: и узкая лежит плашмя всей шириной, серия ходов внахлёст сходится в массу (пресет `плоская`)
    unevenness_reserve: float = 1.0    # К8.3: насколько краска на волоске лежит неровно вдоль хода (`борозда.неровный_запас`): щетина — 1; мягкий длинный волос лайнера набирает её ровно — след не рвётся и не худеет на сотне px
    edge_band: float = 0.0                # п. 10, инструмент 4: мягкая кисть — доля ширины с каждого бока, где волос к краю пучка короче, реже и беднее краской (`мягкая.py`); 0 — щетина, как было
    tuft_mm: object = None           # Д-7: толщина пучка плоской, мм: на ребре (`касание` 0) след — полоса этой толщины; None — по ширине кисти (`укладка.толщина_пучка`: 0,225 ширины, 0,5–3 мм); 0 — толщины нет (как до Д-7)
    gaps: float = 0.0             # К-14б: доля ширины кисти без волоса (0–0,9): ворс стоит пучками, между ними пусто — ход кладёт не полосу, а линии пучков, у каждого своя длина, нагрузка и жёсткость (веерная, растрёпанная); 0 — сплошной ворс, как было
    in_tuft: float = 1.0              # К-14б: сколько гребней ворса в пучке в среднем (работает с `просветы`): 1 — пучок в один гребень, линия шириной в `шаг`; 4 — клок старой кисти
    fan: float = 0.0                 # К-14б: ворс разложен веером — кончики стоят дугой: середина кисти выдаётся за края на эту долю ширины (веерная — 0,15–0,25); начало и конец следа и отпечаток тычка идут дугой; 0 — кончики в линию, как было

    def __setattr__(self, name, v):
        """Правка поля кисти (`к.разброс_тона = 0.3`) — только известного: опечатка — ошибка с ближайшим именем (Е-7), а не
        новый атрибут, которого не видят ни мазок, ни журнал. Своё (`_остаток`, `_краска`…) — с подчёркиванием."""
        if not name.startswith('_') and name not in _FIELDS_BRUSHES:
            raise AttributeError(f'кисть: {hint(name, _FIELDS_BRUSHES)} (поля — `python -m кисть.справка кисть`)')
        object.__setattr__(self, name, v)

    # 2б: что мазок оставляет в кисти (подобранное, набранное, баланс). Мазок на ядрах (`ядра.py`) досчитывается позже
    # вызова: чтение такого поля сперва досчитывает мазки этой кисти — снаружи кисть всегда та же, что на одном ядре
    _remainder = property(lambda self: _after_strokes(self, '_остаток'), lambda self, v: self.__dict__.__setitem__('_остаток·', v))
    _charged = property(lambda self: _after_strokes(self, '_набрано'), lambda self, v: self.__dict__.__setitem__('_набрано·', v))
    _balance = property(lambda self: _after_strokes(self, '_баланс'), lambda self, v: self.__dict__.__setitem__('_баланс·', v))

    def __post_init__(self):
        self._remainder = None          # что осталось на кончиках после прошлого мазка: (K/S по местам, есть ли)
        self._paint = None           # набранная краска: (список (линейный RGB, доля), раскладка)
        self._charge = None            # как её набрали: (цвет вызова, раскладка, тот же объект _краска) — для журнала (А-1)

    # ---------------------------------------------------------------- файлы
    @classmethod
    def from_file(cls, path, **replacements):
        if not os.path.exists(path):
            beside = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
            if os.path.exists(beside):
                path = beside
        with open(path, encoding='utf-8') as f:
            d = json.load(f)
        d.update(replacements)
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        extra = set(d) - names
        if extra:
            raise ValueError('неизвестные параметры кисти: ' + '; '.join(hint(l, names) for l in sorted(extra)))
        return cls(**d)

    def in_file(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    def with_(self, **replacements):
        """Копия кисти с другими параметрами. Копия чистая: подобранная краска не переходит."""
        d = asdict(self)
        d.update(replacements)
        return type(self).from_dict(d)

    # ------------------------------------------------------- краска на кисти
    def wipe(self):
        """Вытереть кисть о тряпку: следующий мазок начнётся чистой краской (подобранного на кончиках нет). У кисти с
        живой краской (Д-1, `набрать(цвет, количество)`) тряпка снимает и краску: с кончиков — всю, из ворса — большую
        часть (`числа._ЗАГРУЗКА`): следующий ход — бедный, протиркой."""
        self._remainder = None
        kr = self._paint
        if isinstance(kr, Loading):
            self._change(kr.after('wipe', kr.reserve * H._ZAGRAtZBrushA[1], kr.drop * 0.0))
        return self

    def rinse(self):
        """Д-1: промыть кисть — чистая и пустая: ни подобранного, ни набранной краски; мазок без цвета — после `набрать`."""
        self._remainder = self._paint = self._charge = None
        self._rinse = True              # для журнала: следующий набор живой краски — после промывки (строка "rinse")
        return self

    def charge(self, color, amount=None, layout='полосы'):
        """Взять краску (sRGB 0..255 или список [(rgb, доля), ...]) поверх того, что осталось на кончиках.
        Мазок без цвета (`цвет=None`) пишет набранной.

        количество (Д-1) — сколько краски взято, в долях полной кисти: 1 — кисть полна, 0,5 — половина. Полная кисть —
        один и тот же объём краски, каким бы ходом её ни клали (Д-1-2; свежая кисть хода с цветом в вызове набрана со
        случаем хода — вокруг того же объёма). С количеством краска на кисти живая: ход без цвета тратит её, следующий начинается с
        остатка — кисть пустеет от хода к ходу, след беднеет и рвётся; сколько осталось — `к.осталось` и поле ответа
        мазка `осталось`. None — как было: кисть полна на каждом ходе. Имя замеса палитры — `х.набрать(к, 'имя', …)`."""
        from .paint import in_linear
        if isinstance(amount, str):       # прежний вызов `набрать(цвет, 'смесь')`: второй довод был раскладкой
            amount, layout = None, amount
        pair = (parse_color(color, in_linear), layout)
        if amount is not None:
            q = float(amount)
            if not 0.0 < q <= 1.0:
                raise ValueError(f'набрать: количество {amount} — доля полной кисти, больше 0 и не больше 1 (None — кисть полна на каждом ходе)')
            pair = Loading(pair, np.full(H._BrushResponseRZAndN, q), np.full(H._BrushResponseRZAndN, min(q / H._ZAGRAtZBrushA[0], 1.0)),
                            ((('rinse',),) if self.__dict__.get('_промыта') and self._remainder is None else ()) + (('load', q),))
        self.__dict__.pop('_промыта', None)
        self._paint = pair
        self._charge = (color, layout, self._paint)
        return self

    @property
    def left(self):
        """Д-1: сколько набранной краски осталось в ворсе, в долях полной кисти (0–1); None — краска не живая
        (`набрать` без количества: кисть полна на каждом ходе) или не набрана."""
        kr = self._paint
        return round(float(kr.reserve.mean()), 4) if isinstance(kr, Loading) else None

    def _change(self, new):
        """Д-1: то, что на кисти, стало другим (ход потратил, тряпка сняла) — набор остаётся тем же набором."""
        if self._charge is not None and self._charge[2] is self._paint:
            self._charge = (self._charge[0], self._charge[1], new)
        self._paint = new

    def _remember(self, places, ks_drops):
        """Что осталось на кончиках — в память кисти (следующий мазок начнётся с этого). места — место гребня поперёк
        кисти в долях ширины (−0,5…0,5), кс_капли — K/S капель гребней. Перенесено из `укладка.py` как было."""
        from .numbers import _BrushResponseRZAndN
        place = clamp(((places + 0.5) * _BrushResponseRZAndN).astype(np.intp), 0, _BrushResponseRZAndN - 1)
        ks_drops = np.asarray(ks_drops, np.float64)       # Е-3: `np.bincount` — те же суммы в том же порядке, что `np.add.at`
        C = ks_drops.shape[1]
        ks_rest = np.bincount((place[:, None] * C + np.arange(C)).ravel(), ks_drops.ravel(), _BrushResponseRZAndN * C).reshape(_BrushResponseRZAndN, C)
        has_rest = np.bincount(place, None, _BrushResponseRZAndN).astype(np.float64)
        empty = has_rest == 0
        if empty.any():
            nearest = np.abs(np.arange(_BrushResponseRZAndN)[:, None] - np.nonzero(~empty)[0][None, :]).argmin(axis=1)
            ks_rest[empty] = ks_rest[~empty][nearest[empty]]
            has_rest[empty] = has_rest[~empty][nearest[empty]]
        self._remainder = (ks_rest / has_rest[:, None], np.ones(_BrushResponseRZAndN))

    def copy(self):
        """Копия с тем, что на кисти (подобранное, набранное): мазок копией не пачкает эту кисть (проба, процессы)."""
        import copy
        n = self.with_()
        n._remainder = copy.deepcopy(self._remainder)
        n._paint = copy.deepcopy(self._paint)
        if self._charge is not None and self._charge[2] is self._paint:
            n._charge = (self._charge[0], self._charge[1], n._paint)
        return n

    def in_pixels(self, px_on_mm):
        """Кисть в пикселях холста с масштабом px_на_мм (п. 9). Без размеров в мм — она сама; иначе — копия
        с `ширина` и `шаг` в px и с тем же, что на кончиках (что подобрала копия, `Холст.мазок` вернёт сюда)."""
        if self.width_mm is None and self.spacing_mm is None:
            return self
        if px_on_mm is None:
            raise ValueError('кисть в мм: у холста не задан px_на_мм')
        z = {'ширина_мм': None, 'шаг_мм': None}
        if self.width_mm is not None:
            z['ширина'] = float(self.width_mm) * float(px_on_mm)
        if self.spacing_mm is not None:
            z['шаг'] = float(self.spacing_mm) * float(px_on_mm)
        n = self.with_(**z)
        n._remainder, n._paint = self._remainder, self._paint      # Д-1: и живая краска — та же (`Загрузка`)
        return n

    def dirty(self):
        return self._remainder is not None

    def thickness_film(self, load=None):
        """Толщина плёнки полного мазка этой кистью, единицы h."""
        n = self.load if load is None else load
        if isinstance(n, str):
            if n not in NAGRAtZBrushAnd:
                raise ValueError(f'нагрузка: {sorted(NAGRAtZBrushAnd)} или число, а не {n!r}')
            n = NAGRAtZBrushAnd[n]
        # п. 10: пол плёнки — для тонкой краски (не толще «обычной»); пастозная у узкой кисти идёт по ширине, как раньше
        by_width = (max(float(self.width), H._MM) / (40.0 * H._MM)) ** 0.8
        return float(self.thickness) * max(float(n) * by_width, min(float(n), NAGRAtZBrushAnd['обычная']) * scale_thicknesses(self.width))

    def charged(self, load=None):
        """П. 13в: насколько кисть набрана — сама нагрузка числом (с множителем `толщина`). У кисти от `_ПОЛ_ПЛЁНКИ` px и
        у тонкой краски (до «обычной») это то же, что плёнка в долях `масштаб_толщины`; у густой краски на кисти уже —
        больше: плёнку узкая кисть кладёт тоньше, а набрана так же полно, как широкая."""
        n = self.load if load is None else load
        return float(self.thickness) * float(NAGRAtZBrushAnd[n] if isinstance(n, str) else n)


_FIELDS_BRUSHES = frozenset(f.name for f in fields(Brush))


def hint(name, known):
    """Е-7: «нет поля lvel, может быть level?» — ближайшее известное имя, если оно есть."""
    import difflib
    b = difflib.get_close_matches(str(name), sorted(known), n=1, cutoff=0.6)
    return f'нет поля {name}' + (f', может быть {b[0]}?' if b else '')


def _number(v):
    return isinstance(v, (int, float, np.integer, np.floating))


def parse_color(color, in_linear):
    """Одна краска (r, g, b) или список [(rgb, доля), ...] -> список (линейный RGB (3,), доля), доли в сумме 1."""
    ts = list(color)
    if len(ts) == 0:
        raise ValueError('цвет: пусто')
    if _number(ts[0]):
        if len(ts) != 3:
            raise ValueError('цвет: нужны три числа (r, g, b) или список [(rgb, доля), ...]')
        paints = _with_kind([(np.asarray(in_linear(ts), dtype=np.float64), 1.0)], [getattr(color, 'мутность', 1.0)])
        if getattr(color, 'вид', None) == 'медиум':
            paints.medium = True          # Ф-2: плёнка без пигмента — недомеса прядей у неё нет (`Холст._положить`)
        return paints
    paints, turbidity = [], []
    for e in ts:
        e = list(e)
        if len(e) == 2 and not _number(e[0]):
            rgb, share = e
        elif len(e) == 4:
            rgb, share = e[:3], e[3]
        elif len(e) == 3:
            rgb, share = e, 1.0
        else:
            raise ValueError('цвет: элемент списка — (rgb, доля), [r, g, b, доля] или [r, g, b]')
        if float(share) > 0:
            paints.append((np.asarray(in_linear(list(rgb)), dtype=np.float64), float(share)))
            turbidity.append(getattr(rgb, 'мутность', 1.0))
    if not paints:
        raise ValueError('цвет: все доли нулевые')
    sum = sum(d for _, d in paints)
    return _with_kind([(c, d / sum) for c, d in paints], turbidity)


class Loading(tuple):
    """Д-1: живая краска на кисти — та же пара (краски, раскладка), что `Кисть._краска`, и сколько её: запас — краска в
    ворсе по местам поперёк кисти (`_КОРЗИН` мест), капля — на кончиках; обе — в долях объёма полной кисти
    (`укладка._полная_кисть`: от хода и его случая не зависит, Д-1-2). Ход без цвета начинает с них и оставляет новые (`укладка._с_загрузки`, `_в_загрузку`). дела — что с кистью
    делали после последнего хода (для журнала): ('load', количество), ('wipe',); от — с какой загрузки они начаты."""

    def __new__(cls, pair, reserve=None, drop=None, jobs=(), from_=None):
        response = tuple.__new__(cls, pair)
        if reserve is not None:
            response.reserve, response.drop, response.jobs, response.from_ = np.asarray(reserve, np.float64), np.asarray(drop, np.float64), tuple(jobs), from_
        return response

    def after(self, job, reserve, drop):
        """Та же краска с другим количеством. дело None — ход (дела с чистого листа), иначе — имя дела кисти."""
        if job is None:
            return Loading(tuple(self), reserve, drop)
        return Loading(tuple(self), reserve, drop, self.jobs + ((job,),), self.from_ if self.jobs else self)

    def __deepcopy__(self, memo):
        import copy
        return Loading(copy.deepcopy(tuple(self), memo), self.reserve.copy(), self.drop.copy(), self.jobs, None)


class _Paints(list):
    """Краски набора с видом (п. 10, инструмент 3): `мутность` — средняя по долям (набор на кисти мешается в одну плёнку)."""


def _with_kind(paints, turbidity):
    if all(m >= 1.0 for m in turbidity):
        return paints                       # кроющие — как было, простой список
    brush = _Paints(paints)
    brush.turbidity = float(sum(m * d for m, (_, d) in zip(turbidity, paints)))
    return brush
