# READING VIEW of движок/примеры/за_край.py with names in English. Not runnable; edit the original.
"""Растушёвка за край краски слоя (З-1): вытертая мягкая кисть идёт из сырого пятна своего слоя на пустое место слоя
и обратно. Лист проб кисти, не кусок картины.

    python3 примеры/за_край.py [папка]   # из папки «движок»: кропы ×4 до и после, числа в окно

Холст 460×220. Слой «низ» — светлые ходы `плоская` 100, высушен. Слой «пятно» — одна светлая краска `мягкая` 48,
стоячими ходами: плотно у x 40–110 (нагрузка 0,4), дальше плёнка сходит на нет (0,2 → 0,08 → 0,02, до x ≈ 215) — край
без шва, как у второго сеанса по сухому. Растушёвка — `мягкая` 48, `нагрузка=0`, цвет `None`, нажим 0,45, `'подъём'`:
ходы через 14 px, чётные — из пятна за край (x 70 → 430), нечётные — с пустого места в пятно (430 → 70).

Что должно быть (и что мерит `мерка`): вне краски слоя ход не делает ничего — нижний слой тот же, в слое «пятно» там
пусто; в самой краске слоя нет цвета темнее её самой — у слоя одна краска, тёмному взяться неоткуда. До З-1 было
иначе: за краем краски ложилась плёнка без цвета (серое на светлом), а ход, входящий в пятно с пустого места, вёл
в светлое серые хвосты.
Импорт ничего не кладёт: `from примеры.за_край import лист, мерка` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (166, 146, 129)
NAndZ, BrushRAWithBrushA = (243, 240, 235), (255, 255, 251)
Sh, In = 460, 220
InNE = (320, 20, 440, 200)        # область, где краски слоя «пятно» нет: дальше двух ширин кисти от края пятна
IN_PAINT = (40, 30, 200, 190)    # область пятна с его сходящим краем


def patch(canvas, seed=0):
    """Высохший светлый «низ» и сырое пятно слоя «пятно», сходящее на нет к x ≈ 215."""
    pl = Brush.from_file(os.path.join(PREWithETY, 'плоская.json'), width=100)
    for i, y in enumerate(range(20, In + 40, 60)):
        canvas.stroke(gesture([[-40, y], [Sh + 40, y]], 100, 'остановка', seed=seed + i, pressure=0.7), NAndZ, pl.with_(), layer='низ', load=0.6)
    canvas.dry(1.0)
    mg = Brush.from_file(os.path.join(PREWithETY, 'мягкая.json'), width=48)
    n = 0
    for x, load in ((40, 0.4), (75, 0.4), (110, 0.4), (140, 0.2), (165, 0.08), (190, 0.02)):
        for y0 in (15, 110):
            canvas.stroke(gesture([[x, y0], [x, y0 + 95]], 48, 'остановка', seed=seed + 100 + n, pressure=0.5), BrushRAWithBrushA, mg.with_(), layer='пятно', load=load)
            n += 1
    return mg


def blend(canvas, mg, seed=0):
    """Вытертая мягкая 48 поперёк края пятна: из краски на пустое место слоя и обратно."""
    for j, y in enumerate(range(30, In - 20, 14)):
        a, b = (70, 430) if j % 2 == 0 else (430, 70)
        canvas.stroke(gesture([[a, y], [b, y]], 48, 'подъём', seed=seed + 300 + j, pressure=0.45), None, mg.with_(), layer='пятно', load=0)


def _state(canvas):
    """Картинка (серое sRGB, без света) и поля слоя «пятно» — для мерки."""
    x0, y0, x1, y1 = IN_PAINT
    sl = canvas.layers['пятно']
    return dict(kind=np.asarray(canvas.look(light=False), dtype=np.float64).mean(axis=2),
                color=np.array(sl.color)[y0:y1, x0:x1].mean(axis=2), coverage=np.array(sl.coverage),
                thickness=np.array(sl.thickness))


def gauge(to, after):
    """Что сделала растушёвка: вне краски слоя — среднее |ΔL| картинки (серое sRGB 0–255), наибольшее потемнение
    пикселя, пиксели слоя с плёнкой; в краске слоя — самый тёмный цвет плёнки, что кроет (покрытие ≥ 0,3; линейный
    свет 0–1), до и после, и пиксели с тёмной плёнкой (цвет ниже 0,6 при покрытии ≥ 0,01)."""
    x0, y0, x1, y1 = InNE
    d = (after['вид'] - to['вид'])[y0:y1, x0:x1]
    a0, b0, a1, b1 = IN_PAINT
    def dark(with_):
        A = with_['покрытие'][b0:b1, a0:a1]
        return float(with_['цвет'][A >= 0.3].min()), int(((with_['цвет'] < 0.6) & (A >= 0.01)).sum())
    return dict(outside_dL=float(np.abs(d).mean()), outside_darker=float(-d.min()),
                outside_film=int((after['толщина'][y0:y1, x0:x1] > 0).sum()),
                color_to=dark(to)[0], color_after=dark(after)[0], dark_to=dark(to)[1], dark_after=dark(after)[1])


def sheet(seed=0, cores=1, journal=None):
    """Холст с пятном и растушёвкой за край. Ответ — (холст, мерка)."""
    canvas = Canvas(Sh, In, ground=GRAtNT, px_on_mm=8.0, cores=cores, journal=journal)
    mg = patch(canvas, seed)
    to = _state(canvas)
    blend(canvas, mg, seed)
    return canvas, gauge(to, _state(canvas)), to


if __name__ == '__main__':
    where_to = next((a for a in sys.argv[1:] if not a.startswith('--')), '.')
    os.makedirs(where_to, exist_ok=True)
    for seed in (0, 1000, 2000):
        canvas, m, _ = sheet(seed)
        print(f"сид {seed}: вне краски слоя |ΔL| {m['вне_dL']:.2f}, темнее всего на {m['вне_темнее']:.1f}, пикселей с плёнкой {m['вне_плёнки']}; "
              f"в краске: самый тёмный цвет {m['цвет_до']:.3f} → {m['цвет_после']:.3f}, тёмной плёнки {m['тёмных_до']} → {m['тёмных_после']} px")
        if seed == 0:
            canvas.look((120, 30, 270, 180), 4, light=False, path=os.path.join(where_to, 'за_край_x4.png'))
            canvas.look((0, 0, Sh, In), 2, light=False, path=os.path.join(where_to, 'за_край.png'))
