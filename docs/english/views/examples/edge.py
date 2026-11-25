# READING VIEW of движок/примеры/ребро.py with names in English. Not runnable; edit the original.
"""Ребро плоской кисти (Д-7): у пучка есть толщина — на ребре след сплошной полосой в 1–2 мм. Лист проб кисти, не
кусок картины.

    python3 примеры/ребро.py [папка]   # из папки «движок»: лист 1:1 и кропы ×4, числа в окно

Холст 75×50 мм (600×400 px при 8 px/мм; при других px/мм — те же мм). `плоская` 5 мм:

1. три хода на ребре (`касание=0`) с нажимом 0,3 / 0,6 / 0,9 — полоса 1,1–1,6 мм без щелей, шире с нажимом;
2. та же кисть плашмя, тот же ход — 5 мм: одна кисть кладёт и плоскость, и линию;
3. ход с поворотом кисти в пальцах (`углы_вдоль(точки, поворот=…)`, 0 → 90°): плашмя — и сходит в ребро;
4. вход нажимом: свои точки, нажим 0,15 → 0,6 на первых 40 % хода, без `нажим_вдоль` — след начинается узко и
   раскрывается за нажимом;
5. тычки плоской 0,5–1,5 мм (4–12 px) — отпечаток в размер кисти, плашмя и на ребре один и тот же;
6. прежнее ребро — кисть без толщины (`пучок_мм=0`): след идёт за нажимом от нити до всей ширины, с `нажим_вдоль`
   и `нажим_руки=True` — клин.

`мерка(х)` — ширины полос в мм и щели, отпечатки тычков в долях кисти, вход в долях ширины тела.
Импорт ничего не кладёт: `from примеры.ребро import лист, мерка` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture, pressure_along, jab
from brush.replay import angles_along

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (200, 190, 170)
BrushRAWithBrushA = (60, 50, 70)
NAZhAndMY = (0.3, 0.6, 0.9)
JABS_MM = (0.5, 0.75, 1.0, 1.5)      # 4, 6, 8, 12 px при 8 px/мм


def _flat(width_mm, **replacements):
    return Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width_mm=width_mm, **replacements)


def sheet(seed=0, px_on_mm=8.0):
    """Лист проб. Ответ: (холст, места) — места проб в px холста, для `мерка`."""
    k = px_on_mm                                               # px в мм
    canvas = Canvas(int(75 * k), int(50 * k), ground=GRAtNT, px_on_mm=px_on_mm)
    W = 5.0 * k
    places = {'k': k, 'ребро': [], 'тычки': []}
    # 1. ребро: касание 0, три нажима
    for i, n in enumerate(NAZhAndMY):
        y = (4 + 4 * i) * k
        canvas.stroke(gesture([[5 * k, y], [45 * k, y]], W, 'подъём', seed=seed + i, pressure=n), BrushRAWithBrushA, _flat(5.0, touch=0), load=0.5)
        places['ребро'].append((y, 14 * k, 32 * k))             # ось, тело хода от x до x
    # 2. та же кисть плашмя
    y = 19 * k
    canvas.stroke(gesture([[5 * k, y], [45 * k, y]], W, 'подъём', seed=seed + 5, pressure=0.6), BrushRAWithBrushA, _flat(5.0), load=0.5)
    places['плашмя'] = (y, 14 * k, 32 * k)
    # 3. поворот кисти в пальцах: плашмя -> ребром
    y = 27 * k
    canvas.stroke(angles_along(gesture([[5 * k, y], [45 * k, y]], W, 'подъём', seed=seed + 6, pressure=0.6), rotation=[[0, 0], [0.25, 0], [0.7, 90]]),
            BrushRAWithBrushA, _flat(5.0), load=0.5)
    places['поворот'] = (y, 5 * k, 45 * k)
    # 4. вход нажимом: свои точки, без нажим_вдоль
    y = 35 * k
    s = np.linspace(0.0, 1.0, max(int(40 * k / 3), 40))
    p = np.where(s < 0.4, 0.15 + 0.45 * s / 0.4, 0.6) * np.clip((1.0 - s) / 0.15, 0.0, 1.0)
    canvas.stroke(np.column_stack([(5 + 40 * s) * k, np.full(len(s), y), p]), BrushRAWithBrushA, _flat(5.0), load=0.5, seed=seed + 7)
    places['вход'] = (y, 5 * k, 45 * k)
    # 5. тычки 0,5–1,5 мм: плашмя (верхний ряд) и на ребре (нижний)
    for r, touch in enumerate((1, 0)):
        for j, mm in enumerate(JABS_MM):
            x, y = (52 + 6 * j) * k, (5 + 5 * r) * k
            canvas.stroke(jab(x, y, pressure=0.85, angle=0), BrushRAWithBrushA, _flat(mm, touch=touch), load=0.5, seed=seed + 20 + j)
            places['тычки'].append((x, y, mm, touch))
    # 6. прежнее ребро: кисть без толщины пучка — клин по нажиму руки
    y = 44 * k
    canvas.stroke(pressure_along(gesture([[5 * k, y], [45 * k, y]], W, 'подъём', seed=seed + 8, pressure=0.9), [[0, 1], [0.45, 0.9], [1, 0.3]]),
            BrushRAWithBrushA, _flat(5.0, touch=0, tuft_mm=0), load=0.5, pressure_hand=True)
    places['клин'] = (y, 5 * k, 45 * k)
    return canvas, places


def _coverage(canvas):
    return np.asarray(next(iter(canvas.layers.values())).coverage) > 0.5


def _band(A, y, x0, x1, half_width):
    """Ширина следа по столбцам (px) и наибольшая щель внутри следа (px) на участке хода."""
    body = A[int(y - half_width):int(y + half_width) + 1, int(x0):int(x1)]
    width, slit = body.sum(axis=0), 0
    for c in range(body.shape[1]):
        has = np.nonzero(body[:, c])[0]
        if len(has):
            slit = max(slit, int(has[-1] - has[0] + 1 - len(has)))
    return width, slit


def gauge(canvas, places):
    """Числа листа: 'ребро_мм' — медиана ширины полосы на ребре по трём нажимам, 'щель' — наибольшая щель в них, px,
    'пусто' — доля столбцов тела без следа; 'плашмя_мм'; 'поворот_мм' — ширина в начале и в конце хода с поворотом;
    'вход' — ширина через 1 мм от начала следа в долях ширины тела; 'тычки' — [(мм кисти, касание, поперёк, вдоль
    в долях кисти)]; 'клин_мм' — ширина клина прежнего ребра в четверти хода и у конца."""
    k, A = places['k'], _coverage(canvas)
    response = {'ребро_мм': [], 'щель': 0, 'пусто': 0.0}
    for y, x0, x1 in places['ребро']:
        width, slit = _band(A, y, x0, x1, 1.9 * k)
        response['ребро_мм'].append(float(np.median(width)) / k)
        response['щель'] = max(response['щель'], slit)
        response['пусто'] = max(response['пусто'], float((width == 0).mean()))
    y, x0, x1 = places['плашмя']
    response['плашмя_мм'] = float(np.median(_band(A, y, x0, x1, 3.9 * k)[0])) / k
    y, x0, x1 = places['поворот']
    width = _band(A, y, x0, x1, 3.9 * k)[0]
    response['поворот_мм'] = (float(np.median(width[int(4 * k):int(8 * k)])) / k, float(np.median(width[int(30 * k):int(34 * k)])) / k)
    y, x0, x1 = places['вход']
    width = _band(A, y, x0, x1, 3.9 * k)[0]
    start = int(np.nonzero(width > 0)[0][0])
    response['вход'] = float(width[start + int(k)]) / max(float(np.median(width[int(22 * k):int(30 * k)])), 1.0)
    response['тычки'] = []
    for x, y, mm, touch in places['тычки']:
        patch = A[int(y - 2.4 * k):int(y + 2.4 * k), int(x - 2.4 * k):int(x + 2.4 * k)]
        ys, xs = np.nonzero(patch)
        W = mm * k
        response['тычки'].append((mm, touch, (ys.max() - ys.min() + 1) / W if len(ys) else 0.0, (xs.max() - xs.min() + 1) / W if len(xs) else 0.0))
    y, x0, x1 = places['клин']
    width = _band(A, y, x0, x1, 3.9 * k)[0]
    response['клин_мм'] = (float(np.median(width[int(8 * k):int(12 * k)])) / k, float(np.median(width[int(30 * k):int(33 * k)])) / k)
    return response


if __name__ == '__main__':
    where_to = sys.argv[1] if len(sys.argv) > 1 else None
    canvas, places = sheet()
    m = gauge(canvas, places)
    print('ребро, нажим 0,3 / 0,6 / 0,9: ' + ' / '.join(f'{v:.2f}' for v in m['ребро_мм']) + f" мм; щель {m['щель']} px; плашмя {m['плашмя_мм']:.2f} мм")
    print(f"поворот 0 → 90°: {m['поворот_мм'][0]:.2f} → {m['поворот_мм'][1]:.2f} мм; вход нажимом: {m['вход']:.2f} ширины тела")
    print('тычки (мм кисти, касание: поперёк × вдоль, в ширинах кисти): ' + '; '.join(f'{mm} мм, {touch}: {a:.2f} × {b:.2f}' for mm, touch, a, b in m['тычки']))
    print(f"прежнее ребро (пучок_мм=0), клин: {m['клин_мм'][0]:.2f} → {m['клин_мм'][1]:.2f} мм")
    if where_to:
        os.makedirs(where_to, exist_ok=True)
        k = int(places['k'])
        canvas.look((0, 0, 75 * k, 50 * k), 1, path=os.path.join(where_to, 'ребро_1к1.png'))
        canvas.look((3 * k, 1 * k, 22 * k, 15 * k), 4, path=os.path.join(where_to, 'ребро_x4.png'))
        canvas.look((49 * k, 1 * k, 74 * k, 14 * k), 4, path=os.path.join(where_to, 'тычки_x4.png'))
        canvas.look((3 * k, 30 * k, 22 * k, 48 * k), 4, path=os.path.join(where_to, 'вход_клин_x4.png'))
