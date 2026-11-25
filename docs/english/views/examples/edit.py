# READING VIEW of движок/примеры/правка.py with names in English. Not runnable; edit the original.
"""Правка высохшей краски: «прижать» и «размочить» на чистом холсте (М-3). Лист проб инструмента, не кусок картины.

    python3 примеры/правка.py [папка]      # из папки «движок»; пишет правка.png и печатает мерки

Холст 8 px/мм. Две пробы:
1. `срезать_вал` — пастозный ход высушен; мастихин плашмя поперёк него, нажим 1, два хода (`х.прижать`): гребни под
   ножом срезаны до тела мазка — ровная площадка с кромкой от ножа, цвет тот же, срезанное ушло на нож;
2. `смягчить_границу` — два цвета встык высушены: растушёвка их не берёт; `х.размочить(1.0, область=…)` — и та же
   растушёвка вытертой мягкой кистью вяжет границу, как сырую.
Импорт ничего не кладёт: `from примеры.правка import срезать_вал, смягчить_границу` (из «движок»)."""
import os
import sys
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (150, 141, 126)


def _brush(name, **fields):
    return Brush.from_file(os.path.join(PREWithETY, name + '.json'), **fields)


def _canvas():
    canvas = Canvas(400, 200, ground=GRAtNT, px_on_mm=8.0)
    canvas.measurement = False
    return canvas


def cut_off_bank(sweeps=2, depth=None):
    """Ответ: (холст, гребни до, гребни после, [ответы `Прижато`]); гребни — p99 − медиана рельефа под ножом, h."""
    canvas = _canvas()
    canvas.stroke(gesture([[30, 100], [370, 103]], 120, 'остановка', seed=50, pressure=0.8), (200, 60, 40), _brush('плоская', width=60), load=1.0)
    canvas.dry(1.0)
    ridges = lambda: float(np.percentile(canvas.relief(70, 130, 180, 220), 99) - np.median(canvas.relief(70, 130, 180, 220)))   # noqa: E731
    to = ridges()
    knife = _brush('мастихин')
    responses = [canvas.press(gesture([[200, 30], [205, 170]], 44, 'подъём', seed=60 + k, pressure=1.0), knife, depth=depth) for k in range(sweeps)]
    return canvas, to, ridges(), responses


def smyagchit_boundary(re_wet=True):
    """Ответ: (холст, наибольший шаг тона на px поперёк границы в долях перепада): сухая ~0,8, размоченная ~0,2."""
    canvas = _canvas()
    brush = _brush('плоская', width=60)
    canvas.stroke(gesture([[30, 70], [370, 70]], 60, 'остановка', seed=5, pressure=0.8), (40, 70, 160), brush, load='обычная')
    brush.wipe()
    canvas.stroke(gesture([[30, 128], [370, 128]], 60, 'остановка', seed=6, pressure=0.8), (220, 190, 60), brush, load='обычная')
    canvas.dry(1.0)
    if re_wet:
        canvas.re_wet(1.0, region=(100, 60, 300, 140))
    m = _brush('мягкая', width=40)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')                      # по сухому растушёвка пуста — `ПустойМазок`
        for i, x in enumerate(range(120, 290, 14)):
            canvas.stroke(gesture([[x, 75 if i % 2 else 125], [x + 6, 125 if i % 2 else 75]], 40, 'подъём', seed=20 + i, pressure=0.6), None, m, load=0)
    L = np.asarray(canvas.image(light=False), float)[80:120, 150:250].mean(axis=(1, 2))
    return canvas, float(np.abs(np.diff(L)).max() / abs(L[-1] - L[0]))


if __name__ == '__main__':
    folder = sys.argv[1] if len(sys.argv) > 1 else '.'
    canvas, to, after, responses = cut_off_bank()
    print(f'прижать: гребни под ножом {to:.2f} → {after:.2f} h')
    for response in responses:
        print('  ', response)
    canvas.png(os.path.join(folder, 'правка.png'))
    for word, wet in (('сухая', False), ('размоченная', True)):
        print(f'граница {word}: шаг тона {smyagchit_boundary(wet)[1]:.2f} перепада на px')
