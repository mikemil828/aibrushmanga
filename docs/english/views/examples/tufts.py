# READING VIEW of движок/примеры/пучки.py with names in English. Not runnable; edit the original.
"""Веерная и растрёпанная кисть (К-14б): ход кладёт не полосу, а пучок линий. Лист проб кистей, не кусок картины.

    python3 примеры/пучки.py [папка]   # из папки «движок»: лист 1:1 и кропы ×4, числа в окно

Холст 100×50 мм (800×400 px при 8 px/мм). Левая половина — сухой грунт, правая — сырой слой охры. Верхний ряд —
`веерная`, нижний — `растрёпанная`. В каждой четверти: пять тычков (нажим 0,4–0,9) и десять одинаково заданных
протяжек (та же длина, наклон, нажим и кисть — разные только место и сид).

- у веерной ворс стоит редкими пучками в один ряд, кончики — дугой (`просветы`, `в_пучке`, `веер`): протяжка — линии
  разной длины и толщины, лёгкая рука — только середина веера; тычок — дуга чёрточек;
- у растрёпанной ворс торчит клоками разной длины и жёсткости: протяжка — полосы с просветами, тычок — пятна клоков с голым между ними;
- по сырому линии берут лежащую краску: цвет каждой — свой;
- соседние ходы с тем же вызовом и разными сидами — разные раскладки пучков, а не сдвиг одной картинки.

`серия(имя)` — 30 одинаково заданных ходов кистью пресета на чистом холсте и зеркало однообразия этой серии
(`х.однообразие`): у кисти из пучков длина, ширина и нажим считаются по линиям пучков (в журнале — "lines").
Импорт ничего не кладёт: `from примеры.пучки import лист, серия` (из «движок»)."""
import os
import sys
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

from brush import Canvas, Brush, gesture, jab

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (214, 204, 184)
ResponseCanvasRA, ZELYoNAYa = (196, 150, 70), (40, 62, 30)
BrushAndWithTAnd = ('веерная', 'растрёпанная')
NAZhAndMY = (0.4, 0.55, 0.7, 0.8, 0.9)


def _brush(name, **fields):
    return Brush.from_file(os.path.join(PREWithETY, name + '.json'), **fields)


def sheet(journal=None):
    """Лист проб. Ответ: (холст, места) — в `места` по кисти и половине ('сухое', 'сырое') ответы тычков и протяжек."""
    canvas = Canvas(800, 400, ground=GRAtNT, px_on_mm=8.0, journal=journal)
    pl = _brush('плоская', width_mm=7.5)
    for i in range(8):                                          # сырой слой справа: ряды внахлёст
        y = 28.0 + 50.0 * i
        canvas.stroke(gesture([[395, y + 4], [810, y - 3]], 60, 'остановка', seed=500 + i, pressure=0.7), ResponseCanvasRA, pl.with_(), layer='низ', seed=500 + i, load=0.45)
    places = {}
    for r, name in enumerate(BrushAndWithTAnd):
        brush = _brush(name)
        for h, half in enumerate(('сухое', 'сырое')):
            x0, y0 = 400 * h, 200 * r
            jabs, drag = [], []
            for j in range(5):
                x = x0 + 40 + 80 * j
                seed = 1000 * r + 100 * h + j
                jabs.append(canvas.stroke(jab(x, y0 + 28, NAZhAndMY[j], angle=-90), ZELYoNAYa, brush.with_(), layer='верх', seed=seed))
                for q in range(2):
                    y = y0 + 122 + 72 * q
                    seed = 1000 * r + 100 * h + 10 + 5 * q + j
                    drag.append(canvas.stroke(gesture([[x - 4, y], [x + 4, y - 56]], brush.width, 'подъём', seed=seed, pressure=0.7), ZELYoNAYa, brush.with_(), layer='верх', seed=seed))
            places[name, half] = {'тычки': jabs, 'протяжки': drag}
    return canvas, places


def series(name, n=30, pressure=0.7, folder=None):
    """n одинаково заданных протяжек кистью пресета по чистому холсту (разные только место и сид) и зеркало этой серии.
    Ответ: (холст, `Однообразие`)."""
    folder = folder or tempfile.mkdtemp(prefix='пучки_')
    os.makedirs(folder, exist_ok=True)
    canvas = Canvas(800, 400, ground=GRAtNT, px_on_mm=8.0, journal=os.path.join(folder, f'серия_{name}.jsonl'))
    brush = _brush(name)
    for i in range(n):
        x, y = 44 + 79 * (i % 10), 110 + 125 * (i // 10)
        canvas.stroke(gesture([[x - 5, y], [x + 5, y - 70]], brush.width, 'подъём', seed=i, pressure=pressure), ZELYoNAYa, brush.with_(), layer='верх', seed=i)
    return canvas, canvas.monotony()


if __name__ == '__main__':
    where_to = sys.argv[1] if len(sys.argv) > 1 else None
    canvas, places = sheet()
    for (name, half), m in places.items():
        print(f'{name}, {half}: тычков со следом {sum(response is not None for response in m["тычки"])} из 5, протяжек {sum(response is not None for response in m["протяжки"])} из 10')
    series = {}
    for name in ('плоская_узкая',) + BrushAndWithTAnd:
        series[name] = series(name, folder=where_to)
        response = series[name][1]
        print(f'серия 30 одинаковых ходов, {name}: зеркало {response.score:.3f}; узость {response.summary["узость"]}; линии {response.summary["линии"]}')
    if where_to:
        os.makedirs(where_to, exist_ok=True)
        canvas.look((0, 0, 800, 400), 1, path=os.path.join(where_to, 'пучки_1к1.png'))
        for name, (x0, y0) in (('веерная_сухое', (0, 0)), ('веерная_сырое', (400, 0)), ('растрёпанная_сухое', (0, 200)), ('растрёпанная_сырое', (400, 200))):
            canvas.look((x0, y0, x0 + 200, y0 + 200), 4, path=os.path.join(where_to, f'пучки_{name}_x4.png'))
            canvas.look((x0 + 200, y0, x0 + 400, y0 + 200), 4, path=os.path.join(where_to, f'пучки_{name}_x4б.png'))
        for name, (khs, _) in series.items():
            khs.look((0, 0, 800, 400), 1, path=os.path.join(where_to, f'серия_{name}_1к1.png'))
            khs.look((0, 20, 240, 200), 4, path=os.path.join(where_to, f'серия_{name}_x4.png'))
