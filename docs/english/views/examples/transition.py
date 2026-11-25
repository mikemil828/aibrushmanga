# READING VIEW of движок/примеры/переход.py with names in English. Not runnable; edit the original.
"""Гладкий переход тона по сырому (У-1): два сырых пятна одного слоя и растушёвка стыка вытертой мягкой кистью.
Лист проб кисти, не кусок картины. Замер — `х.переход(…)`: ширина спада 10–90 % и наибольшая ступень L на 8 px.

    python3 примеры/переход.py [папка] [--до]   # из папки «движок»; --до — прежний рецепт (короткие ходы в одну сторону)

Холст 1200×400. Пятна — `плоская` 90, нагрузка 0,6, ряды через 55 px: слева L 75, справа L 90 (C 0,03, h 250), стык
у x = 600 внахлёст на 30 px; правое кладётся справа налево — к стыку, кончая ход на сыром левом. Растушёвка —
`мягкая` 90, `нагрузка=0`, без цвета (кисть вытерта: своей краски нет, пучок заливается лежащей и носит её), нажим 0,35,
`'подъём'`, ряды через 24 px, шесть проходов со сдвигом 5 px; ход 200–500 px, середина хода у стыка ± 80 px, каждый
следующий — в другую сторону, `к.вытереть()` перед каждым. Мерка 18.11, сид примера: переход ~240 px, ступень ~1,6 L;
прежний рецепт (ход 120 px в одну сторону, `'остановка'`, 4 прохода через 12 px) — ~10 px, ступень ~12 L.
Импорт ничего не кладёт: `from примеры.переход import пятна, растушевать` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture, color

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (150, 141, 126)
TYoMNAYa, WithInETLAYa = color(75, 0.03, 250), color(90, 0.03, 250)
Sh, In, WithTYBrush = 1200, 400, 600


def patch(canvas, layer='небо'):
    """Два сырых пятна плоской 90 в одном слое, стык у x = СТЫК внахлёст."""
    brush = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=90)
    for i, y in enumerate(range(-10, In + 60, 55)):
        for j, (a, b, ts) in enumerate(((-80, WithTYBrush + 30, TYoMNAYa), (Sh + 80, WithTYBrush - 30, WithInETLAYa))):
            canvas.stroke(gesture([[a, y], [(a + b) / 2, y + 2], [b, y]], 90, 'остановка', seed=100 + 10 * i + j, pressure=0.65),
                    ts, brush, layer=layer, seed=100 + 10 * i + j, load=0.6)
            brush.wipe()


def blend(canvas, layer='небо', width=90, pressure=0.35, lengths=(200, 500), spread=80, passes=6, spacing=24,
                end='подъём', there_here=True, seed=7000):
    """Растушёвка стыка у x = СТЫК вытертой мягкой кистью: ходы поперёк стыка, разной длины и места, туда и обратно.
    Ответ — сколько ходов."""
    brush = Brush.from_file(os.path.join(PREWithETY, 'мягкая.json'), width=width)
    r = np.random.default_rng(seed)                       # места ходов — от сида примера; мазки — своими сидами
    n = 0
    for p in range(passes):
        for y in np.arange(-20 + (p * 5) % spacing, In + 20, spacing):
            brush.wipe()
            d, c = r.uniform(*lengths), WithTYBrush + r.uniform(-spread, spread)
            a, b = c - d / 2, c + d / 2
            if there_here and n % 2:
                a, b = b, a
            yy = y + r.uniform(-3, 3)
            canvas.stroke(gesture([[a, yy], [b, yy + r.uniform(-4, 4)]], width, end, seed=seed + n, pressure=pressure),
                    None, brush, layer=layer, seed=seed + n, load=0)          # без цвета: кисть вытерта
            n += 1
    return n


PREZhNAndY = dict(width=48, pressure=0.45, lengths=(120, 120), spread=0, passes=4, spacing=12, end='остановка', there_here=False)


def sheet(previous=False, cores=None, journal=None):
    """Холст с пятнами и растушёвкой. Ответ — (холст, мерка стыка до растушёвки, мерка после)."""
    canvas = Canvas(Sh, In, ground=GRAtNT, px_on_mm=8.0, cores=cores, journal=journal)
    patch(canvas)
    region = (WithTYBrush - 400, 100, WithTYBrush + 400, 300)          # 200 px по высоте — среднее поперёк
    to = canvas.transition(region)
    blend(canvas, **(PREZhNAndY if previous else {}))
    return canvas, to, canvas.transition(region)


if __name__ == '__main__':
    where_to = next((a for a in sys.argv[1:] if not a.startswith('--')), '.')
    os.makedirs(where_to, exist_ok=True)
    canvas, to, after = sheet('--до' in sys.argv, journal=os.path.join(where_to, 'переход.jsonl'))
    for name, m in (('пятна', to), ('растушёвка', after)):
        print(f"{name}: L {m['L_от']} → {m['L_до']}, переход 10–90 % {m['ширина']} px ({m['от_10']}–{m['до_90']}), "
              f"ступень {m['ступень']} L у x {m['где']}")
    canvas.look((WithTYBrush - 300, 0, WithTYBrush + 300, In), path=os.path.join(where_to, 'переход.png'))
