# READING VIEW of движок/кисть/мм.py with names in English. Not runnable; edit the original.
"""Д-8: физика в мм. Числа мазка (`числа.py`) и света (`свет.py`) с размером в px настроены на холст `РОДНОЙ` px/мм
(8). На холсте с другим `px_на_мм` мазок и картинка считаются с этими числами, пересчитанными в его px: длина — × k,
частота на px — / k, проходы сглаживания [1 2 1]/4 (их радиус идёт как корень из числа) — × k², где
k = px_на_мм / `РОДНОЙ`. Толщина плёнки берёт ширину кисти в px родного холста (`числа._ММ`): кисть 10 мм кладёт ту же
плёнку при 4, 8 и 16 px/мм. Холст без `px_на_мм` и холст 8 px/мм — k = 1: ничего не пересчитывается, бит в бит прежнее.

`в_мм(px_на_мм)` — на время мазка или картинки: числа пересчитаны во всех модулях `кисть`, что их взяли, после —
прежние. Что пересчитано — `ЗАКОНЫ`; чего здесь нет (числа прямо в коде, дискретность волоска тоньше пикселя) —
держит px (README, «Физика в мм»)."""
import sys
from contextlib import contextmanager

from . import numbers as H

RResponseDNResponseY = 8.0

# имя числа -> ((индексы или None — само число), степень k). Индексы — места в кортеже с размером длины в px.
ZABrushResponseNY = {
    # толщина плёнки — `числа._ММ` (ширина кисти в px родного холста); здесь — борозда под волоском, запас кончика,
    # проскок волоска
    '_ЗАЗОР_МИН': (None, 1), '_ПОЛБОРОЗДЫ': (None, 1),
    '_БОР_ПОДЪЁМ': ((0, 1), 1), '_БОР_КАПЛЯ': ((0, 1, 2), 1), '_БОР_ЗАПАС': ((0, 1), 1),
    '_БОР_ПРОСКОК': ((2, 3), 1), '_БОР_ГРЕБЕНЬ': ((1,), 1), '_БОР_ТОН_ВЕС': ((2,), 1),
    # свет рельефа: px высоты в единице h
    'ВЫСОТА_РЕЛЬЕФА': (None, 1),
    # Г-2: перелив света выше белого к соседям — σ в px
    'ПЕРЕЛИВ': (None, 1),
}
# Замерено и не взято (стенд Д-8, `проверка/` у Приёмки): кромка и бок густой кисти (`_БОК`, `_УГОЛ`, `_КРАЙ_*` —
# меньше 0,2 L), срастание (`_СРАСТ_*` — ход 16 px/мм дальше от 8, а не ближе), проходы жидкой краски (`_ВИД`,
# `_РОВНО_ШОВ` — разницы нет), пороги узкой кисти (`_ПОЛ_ПЛЁНКИ`, `_ОСТРАЯ`, `_ТЫЧОК`, `_ДРОЖЬ`, `_КРУГЛАЯ`, `_ЩЕЛЬ_УЗК`,
# `_ТРИ_УЗК`: лес ½ лучше на 0,1–0,3 L, а узкая 8 px в ×3 по сухому темнее на 14 L — её физика держит px и в коде).
# Д-8а: px прямо в составе кисти (`пряди._состав`: наименьший шаг ворса, место первого ведущего, запас у бока, добавка
# борозды, сход к боку, отставание гребня, наименьшая прядь) берут множитель `числа._ММ` там же.
HAWithTResponseTY = {'_БОР_ПРОСКОК': (0,)}                          # раз на 100 px — / k
PRResponseCanvasResponseDY = {'_БОР_ТОН_ВЕС': (0,)}   # проходы [1 2 1]/4 — × k², целые, от 1


def _k(px_on_mm):
    return 1.0 if px_on_mm is None else float(px_on_mm) / RResponseDNResponseY


def _recomputing(v, places, f):
    if places is None:
        return f(v)
    v = list(v)
    for i in places:
        v[i] = f(v[i])
    return tuple(v)


def _new(k):
    """{имя: новое значение} для холста с множителем k."""
    from . import light
    was = {and_: getattr(H, and_) if hasattr(H, and_) else getattr(light, and_) for and_ in set(ZABrushResponseNY) | set(HAWithTResponseTY) | set(PRResponseCanvasResponseDY)}
    n = dict(was)
    for and_, (places, st) in ZABrushResponseNY.items():
        n[and_] = _recomputing(n[and_], places, lambda v: v * k ** st)
    for and_, places in HAWithTResponseTY.items():
        n[and_] = _recomputing(n[and_], places, lambda v: v / k)
    for and_, places in PRResponseCanvasResponseDY.items():
        n[and_] = _recomputing(n[and_], places, lambda v: max(1, int(round(v * k * k))))
    return was, n


_WithLResponseInARAnd = [-1, []]          # Е-3: (сколько модулей в sys.modules, словари модулей `кисть`) — не обходить все модули на каждый мазок


def _dicts():
    if _WithLResponseInARAnd[0] != len(sys.modules):
        _WithLResponseInARAnd[1] = [m.__dict__ for name_m, m in list(sys.modules.items())
                       if m is not None and (name_m == 'кисть' or name_m.startswith('кисть.'))]
        _WithLResponseInARAnd[0] = len(sys.modules)
    return _WithLResponseInARAnd[1]


def _substitute(was, n):
    """Числа в каждом модуле `кисть`, что держит их (своим именем, тем же значением)."""
    for d in _dicts():
        for and_, v in n.items():
            if and_ in d and d[and_] is was[and_]:
                d[and_] = v


_NResponseInYE = {}


def _new_remembering(k):
    """`_новые(k)` — один раз на k (Е-3), пока прежние числа те же объекты."""
    from . import light
    was = {and_: getattr(H, and_) if hasattr(H, and_) else getattr(light, and_) for and_ in set(ZABrushResponseNY) | set(HAWithTResponseTY) | set(PRResponseCanvasResponseDY)}
    z = _NResponseInYE.get(k)
    if z is None or any(z[0][and_] is not v for and_, v in was.items()):
        z = _NResponseInYE[k] = _new(k)
    return z


@contextmanager
def in_mm(px_on_mm):
    """На время блока числа физики — для холста `px_на_мм` (k = px_на_мм / 8); при k = 1 — ничего не трогает."""
    k = _k(px_on_mm)
    if k == 1.0 or getattr(H, '_ММ', 1.0) != 1.0:      # родной масштаб или уже пересчитано (вложенный вызов)
        yield
        return
    was, n = _new_remembering(k)
    _substitute(was, n)
    H._MM = k
    try:
        yield
    finally:
        H._MM = 1.0
        _substitute(n, was)
