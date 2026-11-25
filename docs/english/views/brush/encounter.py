# READING VIEW of движок/кисть/встреча.py with names in English. Not runnable; edit the original.
"""v4, Г-8: с чем встретился ход — сырая чужая краска под кистью и мягкая по сухому в новом слое.

Только читает: холст, кисть, сиды и журнал не меняются; файлы физики не тронуты. `Лежало` — то, что лежало на холсте
до хода: вид `перед_записью` у `укладка.положить_мазок` (в прямоугольнике мазка, до записи в слой). `разобрать` — после
хода, по полям следа ответа мазка: `о.чужое` и `о.серый` (`ответ.Ответ`), словами — в `print(о)` и подсказках С19, С20.

Чужое (С19). Сырое, с которым мешается ход (своего слоя, с `мешать_ниже` — и слоёв под ним; схватившееся — в меру
подвижности, как у физики мазка), под кистью: средний тон, взвешенный толщиной сырого. Говорится, когда всё вместе:
подобрано к положенному от `ПОДОБРАНО`, лежащее далеко от названного цвета (`ДАЛЕКО_L` или `ДАЛЕКО_AB`), тон следа
ушёл от названного к лежащему (`СДВИГ_L` или `СДВИГ_AB`).

Серый (С20). Мягкая кисть кроющей краской без разбавителя там, где в её слое краски ещё нет, а сырого под кистью нет ни
в одном слое, по лежащему далёкого тона: тон следа прошёл к названному меньше `ПУТЬ` пути от лежащего (OKLab, проекция).
Прозрачная и полукроющая краска и разбавитель — заказ просвета: не судим.

Пороги — из стенда (`тесты/test_встреча_Г8.py`, числа — справочник, «Stroke response»).
"""
import math

import numpy as np

from . import eye
from .paint import from_linear

PResponseDResponseBRANResponse = 0.1         # снято сырого к положенному (С13 — та же планка)
FAR_L = 20.0         # лежащее от названного: L серой шкалы
FAR_AB = 0.05        # … или a, b OKLab (длина)
SHIFT_L = 8.0           # тон следа от названного к лежащему: L
SHIFT_AB = 0.02         # … или a, b OKLab
PAtT = 0.5              # С20: след прошёл к названному меньше такой доли пути от лежащего
WithYRResponse = 1e-4             # сырая толщина, от которой краска сырая (`ответ.замерить`)
PAtWithTResponse = 0.02            # С20: покрытие своего слоя под кистью, ниже которого места в нём ещё нет
NOT_LESS = 0.1         # доля места под кистью с сырым, меньше которой встреча — край, а не ход по сырому
PResponse = 16384              # тон лежащего — не больше чем по стольким пикселям (как `ответ.ТОН_ПО`)
NEYTRAL = 0.02         # насыщенность, ниже которой у цвета нет тона (h)


def _lab(linear):
    """Линейный RGB (…, 3) -> (L серой шкалы, a, b OKLab) средним по пикселям, как `х.средний`."""
    lab = eye.oklab(np.clip(from_linear(np.asarray(linear, np.float64).reshape(-1, 3)) * 255.0, 0.0, 255.0))
    return np.stack([eye._L(lab[:, 0]), lab[:, 1], lab[:, 2]], 1)


def the_named_one(paints):
    """Названный цвет хода [(линейный RGB, доля)] -> (L, a, b)."""
    weight = np.array([float(d) for _, d in paints])
    return (_lab([c for c, _ in paints]) * (weight / max(float(weight.sum()), 1e-12))[:, None]).sum(0)


def _tone(L, a, b):
    C = math.hypot(a, b)
    rgb = eye.color(L, C, math.degrees(math.atan2(b, a)) % 360.0)
    return {'L': round(float(L), 2), 'C': round(C, 4), 'h': round(math.degrees(math.atan2(b, a)) % 360.0 if C >= 1e-6 else 0.0, 1),
            'rgb': rgb, 'hex': eye.hex(rgb)}


def _Δh(from_, in_):
    """Поворот тона из → в, °; None — один из цветов без тона."""
    if math.hypot(from_[1], from_[2]) < NEYTRAL or math.hypot(in_[1], in_[2]) < NEYTRAL:
        return None
    d = math.degrees(math.atan2(in_[2], in_[1]) - math.atan2(from_[2], from_[1]))
    return round((d + 180.0) % 360.0 - 180.0, 1)


class LayThere:
    """Что лежало в прямоугольнике мазка до записи. дальше — прежний вид `перед_записью` (снимки, скотч): зовётся следом."""

    def __init__(self, canvas, target, lower, soft, further=None):
        self.canvas, self.target, self.lower, self.soft, self.further = canvas, target, lower, soft, further
        self.p = self.wet = self.color = self.own = self.to = None
        self.dry = False

    def __call__(self, p):
        self._take(p)
        if self.further is not None:
            self.further(p)

    def _take(self, p):
        from .laying import _WithCanvasInAT
        canvas = self.canvas
        x0, y0, x1, y1 = (int(v) for v in p)
        where = (slice(y0 - canvas.y0, y1 - canvas.y0), slice(x0 - canvas.x0, x1 - canvas.x0))
        sum = tsv = None
        for with_ in (self.target,) + tuple(self.lower):
            in_ = np.asarray(with_.moisture[where])
            if not in_.any():
                continue
            with_n = np.asarray(with_.thickness[where], np.float64) * in_ * (1.0 - _WithCanvasInAT * np.asarray(with_.degree_of_set[where]))
            if with_n.max() <= WithYRResponse:
                continue
            c = np.asarray(with_.color[where], np.float64) * with_n[..., None]
            sum, tsv = (with_n, c) if sum is None else (sum + with_n, tsv + c)
        self.p = (x0, y0, x1, y1)
        if sum is not None:
            self.wet, self.color = sum, tsv / np.maximum(sum, 1e-12)[..., None]
        if self.soft and self.wet is None:
            self.own = np.array(self.target.coverage[where])
            self.dry = not any((np.asarray(with_.thickness[where]) * np.asarray(with_.moisture[where]) > WithYRResponse).any()
                                for with_ in canvas._layers.values() if with_ is not self.target)
            if self.dry and float(self.own.min()) < PAtWithTResponse:
                self.to = canvas._color(where[0].start, where[0].stop, where[1].start, where[1].stop)


def _shares(mask, *fields):
    """Значения полей в пикселях маски — не больше `ПО` штук (каждый k-й)."""
    n = np.flatnonzero(mask)
    if len(n) > PResponse:
        n = n[::-(-len(n) // PResponse)]
    return [a.reshape((-1,) + a.shape[mask.ndim:])[n] for a in fields]


def parse(l, response, fields, paints, opaque):
    """Ответ мазка `о`, поля его следа, лежавшее `л`: `о.чужое`, `о.серый` (или None). кроющая — краска кроет и не разбавлена."""
    if l.p is None or response.color is None or not paints:
        return
    under = fields['касание'] > 0.5
    h = math.radians(float(response.color['h']))
    trace = np.array([float(response.color['L']), float(response.color['C']) * math.cos(h), float(response.color['C']) * math.sin(h)])
    name = the_named_one(paints)
    place = int(under.sum())
    if l.wet is not None:
        where = under & (l.wet > WithYRResponse)
        n = int(where.sum())
        taken = float(response.wet['к положенному'])
        if n < NOT_LESS * max(place, 1) or taken < PResponseDResponseBRANResponse:
            return
        weight, c = _shares(where, l.wet, l.color)
        lying = (_lab(c) * weight[:, None]).sum(0) / float(weight.sum())
        d, with_ = lying - name, trace - name
        far = abs(d[0]) >= FAR_L or math.hypot(d[1], d[2]) >= FAR_AB
        # сдвиг — только в сторону лежащего: тон прядей и просвет другого слоя сюда не идут
        brush_L = with_[0] * math.copysign(1.0, d[0]) >= SHIFT_L
        brush_ab = (with_[1] * d[1] + with_[2] * d[2]) / max(math.hypot(d[1], d[2]), 1e-9) >= SHIFT_AB      # проекция на путь к лежащему
        if far and (brush_L or brush_ab):
            response.__dict__['чужое'] = {'подобрано': taken, 'лежало': _tone(*lying), 'названо': _tone(*name), 'ΔL': round(float(with_[0]), 1),
                                   'Δh': _Δh(name, trace), 'Δab': round(math.hypot(with_[1], with_[2]), 3)}
        return
    if l.to is not None and opaque and place:
        where = under & (l.own < PAtWithTResponse)
        if int(where.sum()) < 0.5 * place:
            return
        c, = _shares(where, l.to)
        lying = _lab(c).mean(0)
        d = name - lying
        if not (abs(d[0]) >= FAR_L or math.hypot(d[1], d[2]) >= FAR_AB):
            return
        m = np.array([1.0 / 255.0, 1.0, 1.0])            # L серой шкалы — в единицы a, b
        path = float(((trace - lying) * m * (d * m)).sum() / ((d * m) ** 2).sum())
        if path < PAtT:
            response.__dict__['серый'] = {'закрыто': response.__dict__['закрыто'], 'путь': round(path, 2), 'лежало': _tone(*lying), 'названо': _tone(*name)}


def word_foreign(h, short=False):
    """Факт С19: в `print(о)` — полностью, в подсказке — коротко (строка подсказки — до ~220 знаков)."""
    Δh = '' if h['Δh'] is None else f" / Δh {h['Δh']:+.0f}°"
    if short:
        if h['Δh'] is None or abs(h['Δh']) < 5.0:          # тон не повернулся — сдвиг насыщенности
            Δh = ''
        return (f"ход по сырому чужому цвету: подобрано {h['подобрано']:.2f}, тон сдвинут на ΔL {h['ΔL']:+.0f}"
                f"{Δh or ' / Δa, b ' + format(h['Δab'], '.3f')} (лежало L {h['лежало']['L']:.0f}, названо L {h['названо']['L']:.0f})")
    return (f"ход по сырому чужому цвету: подобрано {h['подобрано']:.2f} к положенному, тон сдвинут на ΔL {h['ΔL']:+.0f}{Δh}, "
            f"Δa, b {h['Δab']:.3f} (лежало L {h['лежало']['L']:.0f} {h['лежало']['hex']}, названо L {h['названо']['L']:.0f} {h['названо']['hex']})")


def word_gray(with_, L):
    """Факт С20 — и в `print(о)`, и в подсказке. L — тон следа."""
    return (f"мягкая по сухому в новом слое — серый: закрыто {with_['закрыто']:.2f}, тон следа L {L:.0f} — {100 * with_['путь']:.0f} % пути "
            f"от лежащего (L {with_['лежало']['L']:.0f}) к названному (L {with_['названо']['L']:.0f})")
