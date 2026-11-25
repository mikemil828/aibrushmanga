# READING VIEW of движок/примеры/задним.py with names in English. Not runnable; edit the original.
"""Правка задним числом (К-3, А-7) на чистом холсте: низ 30 ходов, верх 60 поверх, низ одного места теплее на 0,02 —
`х.править` пересобирает только то, что лежит над этим местом и рядом. Лист проб инструмента, не кусок картины.

    python3 примеры/задним.py [папка]      # из папки «движок»; пишет задним_до_после.png и печатает ответ правки

Импорт ничего не кладёт: `from примеры.задним import низ_и_верх, теплее_место` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

from brush import Canvas, Brush, gesture

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
MEWithTResponse = (0, 0, 100, 80)           # px: клетка левого верха — её низ станет теплее


def bottom_and_top(journal, cores=None):
    """Холст 600×400, 8 px/мм, с журналом: низ — 30 ходов плоской 40 px, по ходу в клетке 100×80 (6 × 5), сушка, верх —
    60 ходов плоской 24 px по сырому верхнему слою, по два в клетке. Ответ — (холст, кисть низа, кисть верха)."""
    canvas = Canvas(600, 400, ground=(166, 146, 129), px_on_mm=8.0, journal=journal, cores=cores)
    bottom = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=40)
    top = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=24)
    cells = [(100 * i, 80 * j) for j in range(5) for i in range(6)]
    for k, (x, y) in enumerate(cells):
        bottom.wipe()
        canvas.stroke(gesture([[x + 22, y + 40], [x + 78, y + 42]], 40, 'остановка', seed=k, pressure=0.7), (120 + 2 * k, 140, 170),
                bottom, layer='низ')
    canvas.dry(0.3)
    for k, (x, y) in enumerate(cells * 2):
        top.wipe()
        dy = 30 if k < 30 else 50
        canvas.stroke(gesture([[x + 35, y + dy], [x + 65, y + dy + 2]], 24, 'остановка', seed=100 + k, pressure=0.7),
                (200, 180 - k, 150), top, layer='верх')
    return canvas, bottom, top


def warmer_place(canvas, path=None):
    """Низ места `МЕСТО` теплее на 0,02 (b OKLab): одна команда, холст после неё — уже пересобранный."""
    return canvas.edit(layer='низ', region=MEWithTResponse, warmer=0.02, path=path)


if __name__ == '__main__':
    where_to = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(where_to, exist_ok=True)
    canvas, *_ = bottom_and_top(os.path.join(where_to, 'задним.jsonl'))
    response = warmer_place(canvas, path=os.path.join(where_to, 'задним_до_после.png'))
    print(response)
    canvas.close()
