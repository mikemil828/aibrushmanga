# READING VIEW of движок/кисть/краска.py with names in English. Not runnable; edit the original.
"""Смешение пигментов: цвет -> кривая отражения (12 полос) -> Кубелка–Мунк -> цвет.

Своя модель, без чужих таблиц:
  * три гладких базисных спектра (синий, зелёный, красный участки 400–700 нм),
    в сумме дающих единицу: белый = ровная кривая, любой RGB = их взвешенная сумма;
  * каждая краска описывается парой (K, S) на полосу: поглощение и рассеяние.
    Рассеяние растёт с отражением (светлый пигмент рассеивает сильнее — как белила),
    поэтому чёрный не «съедает» белый с первой капли;
  * K и S смешиваются линейно по долям красок, отражение смеси — по Кубелке–Мунку;
  * обратно в RGB — наименьшие квадраты на те же базисные спектры
    (ход «цвет -> спектр -> цвет» без смешения возвращает цвет точно).
"""
import numpy as np

from .fast import clamp

from .numbers import _InAndD, _RAZBAInAndTEL

PResponseLResponseWith = 12
_LYaMBDA = np.linspace(400.0, 700.0, PResponseLResponseWith)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


_B_SIN = _sigmoid((492.0 - _LYaMBDA) / 13.0)
_B_EDGE = _sigmoid((_LYaMBDA - 588.0) / 13.0)
_B_GREEN = 1.0 - _B_SIN - _B_EDGE
BAZAndWith = np.stack([_B_EDGE, _B_GREEN, _B_SIN], axis=1)          # (12, 3)
ResponseBRATNResponse = np.linalg.pinv(BAZAndWith)                              # (3, 12)

_PResponseL = 0.008      # самое тёмное отражение (идеально чёрных красок нет)
_PResponseTResponseLResponseBrush = 0.985  # самое светлое
_RAWithWithEYaNAndE0 = 0.15  # рассеяние самой тёмной краски (у белой 1 + это)


def in_linear(srgb):
    """sRGB 0..255 (или массив) -> линейный свет 0..1."""
    c = np.asarray(srgb, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def from_linear(lin):
    """линейный свет 0..1 -> sRGB 0..1 (гамма)."""
    c = clamp(np.asarray(lin, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1.0 / 2.4) - 0.055)


def reflectance(rgb):
    """линейный RGB (..., 3) -> кривая отражения (..., 12)."""
    rgb = clamp(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    return _PResponseL + (_PResponseTResponseLResponseBrush - _PResponseL) * (rgb @ BAZAndWith.T)


def in_ks(rgb):
    """линейный RGB (..., 3) -> пигментное состояние (..., 24): K на 12 полос и S на 12 полос."""
    r = reflectance(rgb)
    s = r + _RAWithWithEYaNAndE0
    k = (1.0 - r) ** 2 / (2.0 * r) * s
    return np.concatenate([k, s], axis=-1)


def from_ks(ks):
    """пигментное состояние (..., 24) -> линейный RGB (..., 3)."""
    ks = np.asarray(ks, dtype=np.float64)
    q = np.maximum(ks[..., :PResponseLResponseWith], 0.0) / np.maximum(ks[..., PResponseLResponseWith:], 1e-9)
    r = 1.0 + q - np.sqrt(q * q + 2.0 * q)
    rgb = ((r - _PResponseL) / (_PResponseTResponseLResponseBrush - _PResponseL)) @ ResponseBRATNResponse.T
    return clamp(rgb, 0.0, 1.0)


def mix(a, b, share_b=0.5):
    """Смесь двух красок (линейный RGB). доля_b — доля второй краски, число или массив (...)."""
    t = np.asarray(share_b, dtype=np.float64)[..., None]
    return from_ks(in_ks(a) * (1.0 - t) + in_ks(b) * t)


def mix_srgb(a, b, share_b=0.5):
    """То же для цветов sRGB 0..255; ответ — sRGB 0..255 (дробные)."""
    return from_linear(mix(in_linear(a), in_linear(b), share_b)) * 255.0


# ---------------------------------------------------------------- вид краски (п. 10, инструмент 3)
InAndDY = {'кроющая': 1.0, 'полукроющая': _InAndD[0], 'прозрачная': _InAndD[1], 'лессировочная': 0.0, 'медиум': 0.0}   # мутность: доля плёнки, что кроет своим цветом; Ф-1: лессировочная — чистая лессировка, не кроет вовсе


class Paint(tuple):
    """Краска с видом: `Краска((r, g, b), 'прозрачная')` — идёт везде, где цвет (в `мазок`, `набрать`, в список красок).

    вид — 'кроющая' (как просто (r, g, b)), 'полукроющая', 'прозрачная', 'лессировочная' (мутность 0: тёмное не
    высветляет), 'медиум' (Ф-2: масло или лак без пигмента — плёнка без цвета, `МЕДИУМ`; цвет лежащего не меняет, кладёт
    только толщину) или число-мутность 0..1. Сколько нижнего видно
    сквозь слой, решает толщина плёнки в этом месте: мутная доля кроет своим цветом (покрытие растёт с толщиной),
    прозрачная темнит лежащее на просвет — свет идёт сквозь слой дважды, пропуск = цвет ** (толщина / `_ВИД[2]`)."""
    def __new__(cls, rgb=(255, 255, 255), kind='кроющая'):
        with_ = super().__new__(cls, (255.0, 255.0, 255.0) if kind == 'медиум' else (float(v) for v in tuple(rgb)[:3]))
        if isinstance(kind, str):
            if kind not in InAndDY:
                raise ValueError(f'вид краски: {sorted(InAndDY)} или число 0..1, а не {kind!r}')
            with_.turbidity = InAndDY[kind]
        else:
            with_.turbidity = float(clamp(kind, 0.0, 1.0))
        with_.kind = kind
        return with_

    def __reduce__(self):
        return (Paint, (tuple(self), self.kind))


MEDAndAtM = Paint(kind='медиум')      # Ф-2: бесцветная плёнка — `х.мазок(путь, МЕДИУМ, кисть, разбавитель=1)`


def transmittance_film(color, thickness, turbidity):
    """Пропуск света плёнкой не кроющей краски (туда и обратно): цвет (..., 3) линейный, толщина (...) в единицах h."""
    return np.power(clamp(color, _PResponseL, 1.0), ((1.0 - turbidity) / _InAndD[2]) * np.maximum(thickness, 0.0)[..., None])


def thin(thinner, load, turbidity, even, reserve):
    """Г-1: краска с разбавителем (0–1) — доводы мазка, которыми она ложится: (толщина-нагрузка числом, мутность, ровно, запас).
    нагрузка — 'протирка', 'обычная', 'пастозная' или число (как у мазка; None сюда не идёт — нагрузка кисти). Жидкая
    плёнка тоньше и прозрачнее, течёт, как с `ровно` (и у не кроющей), сухой срыв тянется дольше; 0 — всё как дано."""
    from .tool import NAGRAtZBrushAnd
    m = float(thinner or 0.0)
    if m <= 0.0:
        return load, turbidity, even, reserve
    n = NAGRAtZBrushAnd[load] if isinstance(load, str) else float(load)
    return (n * (1.0 - _RAZBAInAndTEL[0] * m), float(turbidity) * (1.0 - _RAZBAInAndTEL[1] * m), max(float(even or 0.0), m),
            None if reserve is None else float(reserve) * (1.0 + _RAZBAInAndTEL[2] * m))


def check_thinner(thinner, load):
    """Г-1: разбавитель мазка — 0–1, не с полупустой ('сухая') и не с вытертой (0) кистью; иначе ValueError."""
    if thinner is None:
        return None
    m = float(thinner)
    if not 0.0 <= m <= 1.0:
        raise ValueError(f'мазок: разбавитель ("medium") — от 0 до 1, а не {thinner!r}')
    if m and (load == 'сухая' or (not isinstance(load, str) and float(load) <= 0.0)):
        raise ValueError('мазок: разбавитель ("medium") — краске на кисти; с нагрузкой \'сухая\' (полупустая) и 0 (вытертая) не идёт')
    return m
