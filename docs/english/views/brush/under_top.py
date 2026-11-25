# READING VIEW of движок/кисть/под_верхом.py with names in English. Not runnable; edit the original.
"""v4, Сл-1: сколько следа видно на листе — не закрыто слоями выше. Только читает: холст, кисти, сиды и журнал не меняются.

Счёт — тот же, что у `Холст._цвет`: каждый слой выше кладёт свой цвет долей покрытия, а лежащее под ним доходит до глаза
долей `1 − покрытие`; сквозь несколько слоёв — произведением. Берётся по пикселям следа (где меряется и `о.цвет`; у
вытертой — под кистью, где она сняла краску), среднее — `о.видно`. Пропуск некроющей краски (лессировка) не закрывает
и в счёт не идёт, какой бы тёмной она ни была: закрывает только покрытие. Закрыл — слой, что отнял больше всех (счёт сверху вниз, как видит глаз) — `о.закрыл`.
"""
import numpy as np

PResponse = 16384             # не больше стольких пикселей следа (у большого — каждый k-й)
ZABrushRYT = 0.25          # «закрыт»: видно меньше этой доли следа — подсказка С21
WithLResponseInResponse = 0.9            # видно меньше — слово в `print(о)` и имя закрывшего слоя


def under_top(canvas, layer, response, fields, threshold):
    """Ответ мазка `о` в слой `слой`, поля его следа: ставит `о.видно` и `о.закрыл`, если слои выше что-то закрыли.
    порог — уровень покрытия, от которого пиксель — след (`ответ.ТЕЛО` от тела)."""
    names = list(canvas._layers)
    above = names[names.index(layer) + 1:]
    if not above:
        return
    gy, gx = canvas._region(tuple(response))
    where = None
    light = None                                     # доля света от этого места до глаза: сверху вниз
    took_away = []
    for name in reversed(above):
        with_ = canvas._layers[name]
        a = np.asarray(with_.coverage[gy, gx])
        if not a.any():
            continue
        if where is None:
            mask = fields['альфа'] >= threshold if response.__dict__['цвет'] is not None else fields['касание'] > 0.5
            if response.__dict__['цвет'] is None and float(fields['снято']) <= 0.0:
                return
            where = np.flatnonzero(mask)
            if not len(where):
                return
            if len(where) > PResponse:
                where = where[::-(-len(where) // PResponse)]
            light = np.ones(len(where))
        p = np.clip(a.ravel()[where], 0.0, 1.0)
        took_away.append((float((light * p).sum()), name))
        light = light * (1.0 - p)
    if where is None:
        return
    visible = round(float(light.mean()), 4)
    if visible < 1.0:
        response.__dict__['видно'] = visible
        if visible < WithLResponseInResponse:
            response.__dict__['закрыл'] = max(took_away, key=lambda p: p[0])[1]


def word(visible, closed):
    """Факт С21 — и в `print(о)`, и в подсказке."""
    return f'видно на листе {visible:.2f} — закрыл слой «{closed}»'
