# READING VIEW of движок/примеры/черенок.py with names in English. Not runnable; edit the original.
"""Процарапывание черенком по сырому (К-14а): светлая линия одним движением. Лист проб инструмента, не кусок картины.

    python3 примеры/черенок.py [папка]   # из папки «движок»: лист 1:1 и кропы ×4, числа в окно

Холст 75×75 мм (600×600 px при 8 px/мм). Светлый слой высушен, поверх — тёмный сырой (нагрузка 0,6, ряды
внахлёст на полкисти). Шесть пар линий
`х.снять(точки, 'черенок', ширина=…)`: конец 1 и 3 мм, нажим 0,3 / 0,6 / 0,9, прямая и дуга.

- нажим 0,9 — конец идёт по сухому: по оси линии открыт нижний слой, по бокам — валики из раздвинутой краски;
- нажим 0,6 — под концом плёнка: линия темнее; 0,3 — борозда в краске, нижнего не видно;
- конец 3 мм при том же нажиме оставляет плёнку втрое толще, чем 1 мм; линия у него шире, а чистая полоса по оси —
  узкая: конец круглый;
- объём краски на листе тот же, сухой слой не тронут.

`мерка(х, места)` — L по оси каждой прямой и L нижнего слоя там же, объём слоёв до и после, валики.
Импорт ничего не кладёт: `from примеры.черенок import лист, мерка` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (200, 190, 170)
WithInETLAYa, TYoMNAYa = (214, 170, 92), (38, 30, 24)
LAndNAndAnd = ((1, 0.3), (1, 0.6), (1, 0.9), (3, 0.3), (3, 0.6), (3, 0.9))      # конец, мм; нажим


def _L(image):
    """Светлота L* линейной картинки."""
    brush = np.asarray(image, np.float64)[..., :3]
    Y = 0.2126 * brush[..., 0] + 0.7152 * brush[..., 1] + 0.0722 * brush[..., 2]
    return 116.0 * np.cbrt(np.maximum(Y, 1e-9)) - 16.0


def _volume(canvas):
    return sum(float(np.asarray(with_.thickness, np.float64).sum()) for with_ in canvas.layers.values())


def sheet(px_on_mm=8.0, load_bottom=0.25):
    """Лист проб. Ответ: (холст, места) — в `места` линии, ответы черенка и снятое до процарапывания (для `мерка`)."""
    k = px_on_mm
    canvas = Canvas(int(75 * k), int(75 * k), ground=GRAtNT, px_on_mm=px_on_mm)
    pl = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width_mm=7.5)
    rows = [(2.5 + 6.9 * i) * k for i in range(12)]
    for i, y in enumerate(rows):
        canvas.stroke(gesture([[-5 * k, y], [80 * k, y]], 7.5 * k, 'остановка', seed=i, pressure=0.7), WithInETLAYa, pl.with_(), layer='низ', load=load_bottom)
    canvas.dry(1.0)
    places = {'k': k, 'L_низа': _L(canvas.image(light=False)),
             'низ': {p: np.array(getattr(canvas.layers['низ'], p)) for p in ('толщина', 'цвет', 'покрытие', 'влажность', 'схват')}}
    for i in range(24):                                        # тёмный — рядами внахлёст на полкисти: кроет сплошь
        y = (1.0 + 3.45 * i) * k
        canvas.stroke(gesture([[-5 * k, y + 0.9 * k], [80 * k, y - 0.6 * k]], 7.5 * k, 'остановка', seed=100 + i, pressure=0.7), TYoMNAYa, pl.with_(), layer='верх', load=0.6)
    places['объём_до'] = _volume(canvas)
    places['верх_до'] = np.array(canvas.layers['верх'].thickness)
    places['L_до'] = _L(canvas.image(light=False))
    places['прямые'], places['дуги'] = [], []
    u = np.linspace(0.0, np.pi, 40)
    for i, (mm, n) in enumerate(LAndNAndAnd):
        y = (8 + 10.5 * i) * k
        points = [[7 * k, y, n], [37 * k, y + 1.5 * k, n]]
        places['прямые'].append((mm, n, points, canvas.remove(points, 'черенок', width=mm * k)))
        arc = np.column_stack([(55 + 11 * np.cos(u) * (1 if i % 2 else -1)) * k, (y / k - 2 + 4 * np.sin(u)) * k, np.full(len(u), n)])
        places['дуги'].append((mm, n, arc, canvas.remove(arc, 'черенок', width=mm * k)))
    return canvas, places


def gauge(canvas, places):
    """Числа листа: 'объём_%' — на сколько процентов изменился объём краски листа; 'низ_цел' — сухой слой бит в бит;
    'прямые' — по каждой: (мм, нажим, L по оси — медиана, L нижнего слоя там же, L тёмного до процарапывания,
    валик — во сколько раз слой у боков борозды толще, чем лежал, ответ черенка)."""
    k = places['k']
    L = _L(canvas.image(light=False))
    top = np.asarray(canvas.layers['верх'].thickness, np.float64)
    response = {'объём_%': 100.0 * (_volume(canvas) - places['объём_до']) / places['объём_до'],
             'низ_цел': all(np.array_equal(was, np.asarray(getattr(canvas.layers['низ'], p))) for p, was in places['низ'].items()),
             'прямые': []}
    for mm, n, ((xa, ya, _), (xb, yb, _)), response in places['прямые']:
        xs = np.arange(int(xa + 5 * k), int(xb - 5 * k))
        ys = np.round(ya + (yb - ya) * (xs - xa) / (xb - xa)).astype(int)
        side = int(round(response.width / 2.0)) + 1 if response else int(mm * k / 2)
        band = [ys + s * (side + j) for s in (-1, 1) for j in range(max(2, int(0.4 * k)))]
        bead = float(np.mean([top[r, xs].mean() for r in band]) / max(np.mean([places['верх_до'][r, xs].mean() for r in band]), 1e-9))
        response['прямые'].append((mm, n, float(np.median(L[ys, xs])), float(np.median(places['L_низа'][ys, xs])),
                                float(np.median(places['L_до'][ys, xs])), bead, response))
    return response


if __name__ == '__main__':
    where_to = sys.argv[1] if len(sys.argv) > 1 else None
    canvas, places = sheet()
    m = gauge(canvas, places)
    print(f"объём краски листа: {m['объём_%']:+.4f} %; сухой нижний слой бит в бит: {'да' if m['низ_цел'] else 'НЕТ'}")
    for mm, n, Lo, Ln, Ld, bead, response in m['прямые']:
        print(f'конец {mm} мм, нажим {n}: L по оси {Lo:.1f} (нижний слой {Ln:.1f}, тёмный до {Ld:.1f}); валики — слой у боков ×{bead:.2f}')
        print('   ', response)
    if where_to:
        os.makedirs(where_to, exist_ok=True)
        k = int(places['k'])
        canvas.look((0, 0, 75 * k, 75 * k), 1, path=os.path.join(where_to, 'черенок_1к1.png'))
        canvas.look((12 * k, 14 * k, 30 * k, 32 * k), 4, path=os.path.join(where_to, 'черенок_1мм_x4.png'))
        canvas.look((12 * k, 45 * k, 30 * k, 63 * k), 4, path=os.path.join(where_to, 'черенок_3мм_x4.png'))
        canvas.look((46 * k, 13 * k, 64 * k, 31 * k), 4, path=os.path.join(where_to, 'черенок_дуги_x4.png'))
        canvas.look((46 * k, 45 * k, 64 * k, 63 * k), 4, path=os.path.join(where_to, 'черенок_дуги3_x4.png'))
