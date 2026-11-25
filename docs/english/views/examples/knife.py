# READING VIEW of движок/примеры/нож.py with names in English. Not runnable; edit the original.
"""Мастихин: пять проб ножа на чистом холсте. Лист проб инструмента, не кусок картины.

    python3 примеры/нож.py [папка]        # из папки «движок»; пишет нож.png и печатает мерки

Холст 8 px/мм, нож 5,5 мм (44 px). Пробы слева направо:
1. `плоскость` — нож с краской по голому грунту: гладкая лента, валики по бокам, ком там, где нож оторван;
2. `размазать` — нож без краски (`нагрузка=0`, цвет `None`) поперёк двух сырых пастозных ходов, нажим 0,8: сырое срезано
   до тонкой плёнки и протянуто по ходу, краски в слое столько же;
3. `по_сухому` — те же ходы высушены, нож с краской поперёк: нажим 0,3 садит краску на вершины, 0,95 оставляет в низинах;
4. `срезать` — нож без краски с нажимом 1 по толстому сырому: слой снят до следа, краска ушла на нож (`о.на_ноже`);
5. `ребро` — `касание=0`: с краской — линия-валик, без краски по сырому — борозда с валиками.
Импорт ничего не кладёт: `from примеры.нож import плоскость, размазать, по_сухому, срезать, ребро` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (150, 141, 126)
BrushRAWithNAYa, WithAndNYaYa, WithInETLAYa, ResponseCanvasRA = (200, 60, 40), (40, 70, 160), (240, 235, 220), (210, 190, 150)


def _knife(**fields):
    return Brush.from_file(os.path.join(PREWithETY, 'мастихин.json'), **fields)


def _canvas(sh=400, in_=200, px=8.0):
    canvas = Canvas(sh, in_, ground=GRAtNT, px_on_mm=px)
    canvas.measurement = False
    return canvas


def _two_sweep(canvas, dry=False):
    """Два пастозных хода щетиной — что нож размазывает или по чему идёт."""
    brush = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=40)
    for j, (y, ts) in enumerate(((60, BrushRAWithNAYa), (140, WithAndNYaYa))):
        canvas.stroke(gesture([[30, y], [370, y + 3]], 40, 'остановка', seed=30 + j, pressure=0.7), ts, brush, load=0.9)
        brush.wipe()
    if dry:
        canvas.dry(1.0)


def plane(load=0.6, pressure=0.7, px=8.0):
    """Нож с краской по голому грунту. Ответ — холст."""
    k = px / 8.0
    canvas = _canvas(int(400 * k), int(120 * k), px)
    canvas.stroke(gesture([[40 * k, 60 * k], [360 * k, 60 * k]], 44 * k, 'подъём', seed=10, pressure=pressure), ResponseCanvasRA, _knife(width=44 * k), load=load)
    return canvas


def razmazat(pressure=0.8):
    """Нож без краски поперёк двух сырых пастозных ходов. Ответ — холст и толщина слоя до хода ножа."""
    canvas = _canvas()
    _two_sweep(canvas)
    to = np.array(canvas.layers['основа'].thickness[0:200, 0:400], np.float64)
    canvas.stroke(gesture([[200, 20], [205, 185]], 44, 'подъём', seed=40, pressure=pressure), None, _knife(), load=0)
    return canvas, to


def by_dry(pressure=0.5, load=0.3):
    """Нож с краской поперёк двух высохших пастозных ходов. Ответ — холст и толщина слоя до хода ножа."""
    canvas = _canvas()
    _two_sweep(canvas, dry=True)
    to = np.array(canvas.layers['основа'].thickness[0:200, 0:400], np.float64)
    canvas.stroke(gesture([[200, 20], [205, 185]], 44, 'остановка', seed=40, pressure=pressure), WithInETLAYa, _knife(), load=load)
    return canvas, to


def cut_off(pressure=1.0):
    """Нож без краски с полным нажимом по толстому сырому слою. Ответ — холст, толщина до и ответ мазка ножа."""
    canvas = _canvas()
    canvas.stroke(gesture([[30, 100], [370, 103]], 120, 'остановка', seed=50, pressure=0.8), BrushRAWithNAYa,
            Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=60), load=1.0)
    to = np.array(canvas.layers['основа'].thickness[0:200, 0:400], np.float64)
    canvas.measurement = True
    response = canvas.stroke(gesture([[120, 30], [280, 170]], 44, 'остановка', seed=60, pressure=pressure), None, _knife(), load=0)
    return canvas, to, response


def edge():
    """Ребро ножа (`касание=0`): с краской по грунту и сырому, без краски по сырому. Ответ — холст и толщина до ножа."""
    canvas = _canvas()
    canvas.stroke(gesture([[30, 100], [370, 103]], 120, 'остановка', seed=50, pressure=0.8), BrushRAWithNAYa,
            Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=60), load=1.0)
    to = np.array(canvas.layers['основа'].thickness[0:200, 0:400], np.float64)
    canvas.stroke(gesture([[120, 20], [120, 180]], 44, 'остановка', seed=41, pressure=0.6), WithInETLAYa, _knife(), load=0.6, touch=0)
    canvas.stroke(gesture([[280, 40], [280, 160]], 44, 'остановка', seed=42, pressure=0.8), None, _knife(), load=0, touch=0)
    return canvas, to


def _framing(canvas):
    brush = canvas.image(surface='полуматовое')
    return (np.clip(np.where(brush <= 0.0031308, brush * 12.92, 1.055 * np.power(np.maximum(brush, 1e-9), 1 / 2.4) - 0.055), 0, 1) * 255 + 0.5).astype(np.uint8)


if __name__ == '__main__':
    from PIL import Image
    where_to = sys.argv[1] if len(sys.argv) > 1 else '.'
    frames = []
    canvas = plane()
    h = np.asarray(canvas.layers['основа'].thickness)
    print(f'плоскость: плёнка {np.median(h[40:80, 100:300]):.2f}, валик у бока {h[30:50, 150].max():.2f}, ком на отрыве {h[60, 300:390].max():.2f}')
    frames.append(np.pad(_framing(canvas), ((40, 40), (0, 0), (0, 0)), constant_values=255))
    canvas, to = razmazat()
    h = np.asarray(canvas.layers['основа'].thickness)
    print(f'размазать: объём {to.sum():.0f} → {h.sum():.0f}; под ножом на ходе {to[60, 190:215].mean():.2f} → {h[60, 190:215].mean():.2f}; в щели между ходами {h[100, 190:215].mean():.2f}')
    frames.append(_framing(canvas))
    for pressure in (0.3, 0.95):
        canvas, to = by_dry(pressure)
        h = np.asarray(canvas.layers['основа'].thickness)
        print(f'по сухому, нажим {pressure}: покрыто новым на ходе {float(((h - to)[45:75, 185:220] > 0.05).mean()):.2f}; наименьший прирост {float((h - to).min()):.3f}')
        frames.append(_framing(canvas))
    canvas, to, response = cut_off()
    print(f'срезать: объём {to.sum():.0f} → {np.asarray(canvas.layers["основа"].thickness).sum():.0f}; {response}')
    frames.append(_framing(canvas))
    canvas, to = edge()
    frames.append(_framing(canvas))
    Image.fromarray(np.concatenate([np.pad(k, ((0, 0), (0, 6), (0, 0)), constant_values=255) for k in frames], 1)).save(os.path.join(where_to, 'нож.png'))
