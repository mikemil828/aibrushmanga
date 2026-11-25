# READING VIEW of движок/примеры/пользование.py with names in English. Not runnable; edit the original.
"""Пример пользования движком «Кисть» из Python: python3 примеры/пользование.py [папка для листа].
То же журналом: python -m кисть.повтор примеры/пользование.jsonl --out лист.png (из папки «движок»). Описание — README_ПОЛЬЗОВАНИЕ.md."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок», где лежит пакет «кисть»

import numpy as np
from brush import Canvas, Brush, Paint, gesture, path_with_pressure, pressure_along

canvas = Canvas(400, 240, ground=(150, 141, 126), px_on_mm=8.0)
rng = np.random.default_rng(1)

# 1. подмалёвок: широкая плоская щетина, три хода внахлёст, слой «низ»
flat = Brush.from_file('пресеты/плоская.json', width=48)
for k in range(6):
    y = 24 + 40 * k
    points = gesture([[10, y], [200, y + 4], [390, y]], 48, 'остановка', rng, pressure=0.7)
    canvas.stroke(points, (70, 92, 84), flat, layer='низ', seed=10 + k, load='обычная')
    flat.wipe()

# 2. удар в две краски по сырому, пастозно
strike = Brush.from_file('пресеты/плоская.json', width=32)
points = gesture([[60, 70], [150, 60], [230, 80]], 32, 'отрыв', rng, pressure=0.6)
straight = canvas.stroke(points, [((232, 226, 205), 0.3), ((188, 132, 52), 0.7)], strike, layer='низ', seed=21,
               layout='полосы', load='пастозная')

# 3. высушить и положить лессировку в свой слой
canvas.dry(1.0)
soft = Brush.from_file('пресеты/мягкая.json', width_mm=6)
points = path_with_pressure([[40, 170], [200, 180], [360, 165]], 48, rng, pressure=0.5)
canvas.stroke(points, Paint((70, 34, 22), 'лессировочная'), soft, layer='лессировка', seed=31, load='протирка')

# 4. тонкая линия острой кистью, нажим падает к концу
narrow = Brush.from_file('пресеты/плоская_узкая.json', width=6)
points = pressure_along(gesture([[250, 40], [300, 90], [370, 110]], 6, 'подъём', rng, pressure=0.8), [[0, 1], [1, 0.4]])
canvas.stroke(points, (30, 28, 26), narrow, layer='верх', seed=41)

where_to = sys.argv[1] if len(sys.argv) > 1 else '.'
canvas.png(os.path.join(where_to, 'лист.png'), light=True)          # PNG 16 бит, sRGB
canvas.save(os.path.join(where_to, 'лист.npz'))               # состояние холста: слои, сырость, толщина
image = canvas.image(light=False)     # numpy (В, Ш, 3), линейный свет 0..1
print('записанный прямоугольник удара:', straight, '· слои снизу вверх:', list(canvas.layers))
