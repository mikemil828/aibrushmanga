# READING VIEW of движок/примеры/пятно_по_кругу.py with names in English. Not runnable; edit the original.
"""Пятно со спадом по кругу (5а): белый диск по сырому небу и растушёвка вытертой мягкой кистью крест-накрест —
ходы по радиусу и дуги по кругу. Лист проб кисти, не кусок картины. Замер — `замер_круга(х, центр)`: профиль L(r)
(медиана по секторам 5°), спад 10–90 %, наибольшая ступень на 8 px, σ L по углу на окружности середины спада.

    python3 примеры/пятно_по_кругу.py [папка] [--приём радиус|круг|крест] [--масштаб 0.5]   # из папки «движок»

Холст 1200×1200, 8 px/мм. Небо — `плоская` 90, `разброс_тона=0.3`, нагрузка 0,6, ряды через 45 px (через 55 —
щели грунта, σ по углу самого неба ~4 L). Диск — `мягкая` 48, белила (250, 250, 248) нагрузкой 0,5, хорды через
24 px, r 150, по сырому небу. Растушёвка — `мягкая` 90, `нагрузка=0`, без цвета, нажим 0,35, `'подъём'`,
`к.вытереть()` перед каждым ходом:
  (1) 240 ходов по радиусу от r ~40 до ~560 (разброс концов ±30 и ±40 px), углы через 1,5°, каждый следующий — обратно;
  (2) дуги в четверть круга по окружностям r 80…520 через 24 px, внахлёст на четверть дуги, через одну — обратно;
  (3) то же (1) и (2) вторым проходом своими сидами (углы и окружности сдвинуты);
  (4) дуги в полкруга по окружностям r 100…500 и 115…515 через 30 px.
Без (4) спад тот же, а σ по углу ~2,3 L: короткие дуги не разносят тон по кругу. Только по радиусу — лучи
(σ по углу ~6 L, мелкими по углу — 5,9), только по кругу — спада нет (~30 px), углы и дуги вразброс — пятно
лепестками (σ ~12 L). Мерка 18.11 (Движок 169), сид примера: спад 218 px, σ по углу 1,6 L, ~1400 ходов, ~4 мин на
одном ядре.
Импорт ничего не кладёт: `from примеры.пятно_по_кругу import небо, диск, растушевать, замер_круга` (из «движок»)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

import numpy as np

from brush import Canvas, Brush, gesture, color
from brush.eye import lightness
from brush.paint import from_linear

PREWithETY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'пресеты')
GRAtNT = (150, 141, 126)
NEBResponse, BELAndLA = color(72, 0.03, 250), (250, 250, 248)


def _brush(name, w, **fields):
    return Brush.from_file(os.path.join(PREWithETY, name + '.json'), width=w, **fields)


def sky(canvas, m=1.0, layer='небо'):
    """Сырое небо плоской 90 рядами через 45 px (м — масштаб: px и ширины множатся)."""
    brush, w = _brush('плоская', 90 * m, spread_tone=0.3), 90 * m
    for i, y in enumerate(np.arange(-10 * m, canvas.height + 60 * m, 45 * m)):
        canvas.stroke(gesture([[-80 * m, y], [canvas.width / 2, y + 2 * m], [canvas.width + 80 * m, y]], w, 'остановка', seed=100 + i,
                     pressure=0.65), NEBResponse, brush, layer=layer, seed=100 + i, load=0.6)
        brush.wipe()


def disk(canvas, center, r=150, m=1.0, layer='небо'):
    """Белый диск мягкой 48 хордами через 24 px по сырому."""
    cx, cy = center
    w = 48 * m
    brush = _brush('мягкая', w)
    for n, y in enumerate(np.arange(cy - r + 0.3 * w, cy + r, 24 * m)):
        hx = np.sqrt(max(r * r - (y - cy) ** 2, 0.0))
        a, b = (cx - hx, cx + hx) if n % 2 == 0 else (cx + hx, cx - hx)
        canvas.stroke(gesture([[a, y], [(a + b) / 2, y + m], [b, y]], w, 'остановка', seed=300 + n, pressure=0.6),
                BELAndLA, brush, layer=layer, seed=300 + n, load=0.5)
        brush.wipe()


def _radius(canvas, brush, w, center, r0, r1, n, seed, layer):
    rng = np.random.default_rng(seed)
    for i in range(n):
        φ = 2 * np.pi * i / n + rng.uniform(-0.01, 0.01)
        a, b = max(0.0, r0 + rng.uniform(-30, 30) * w / 90), r1 + rng.uniform(-40, 40) * w / 90
        if i % 2:
            a, b = b, a
        p = [[center[0] + t * np.cos(φ), center[1] + t * np.sin(φ)] for t in np.linspace(a, b, 4)]
        brush.wipe()
        canvas.stroke(gesture(p, w, 'подъём', seed=seed + i, pressure=0.35), None, brush, layer=layer, seed=seed + i, load=0)
    return n


def _circle(canvas, brush, w, center, r0, r1, spacing, arc, seed, layer):
    rng = np.random.default_rng(seed)
    i = 0
    for r in np.arange(r0, r1, spacing):
        φ0 = rng.uniform(0, 2 * np.pi)
        for φa in np.arange(φ0, φ0 + 2 * np.pi - 1e-6, arc * 0.75):
            φ = np.linspace(φa, φa + arc, 12)
            if i % 2:
                φ = φ[::-1]
            rr = r + rng.uniform(-5, 5) * w / 90
            p = [[center[0] + rr * np.cos(t), center[1] + rr * np.sin(t)] for t in φ]
            brush.wipe()
            canvas.stroke(gesture(p, w, 'подъём', seed=seed + i, pressure=0.35), None, brush, layer=layer, seed=seed + i, load=0)
            i += 1
    return i


def blend(canvas, center, m=1.0, technique='крест', layer='небо'):
    """Растушёвка пятна по кругу вытертой мягкой 90: 'крест' — рецепт (ходы по радиусу, дуги в четверть и в полкруга);
    'радиус', 'круг' — одна часть, для сравнения. Ответ — сколько ходов."""
    w = 90 * m
    brush = _brush('мягкая', w)
    R = lambda *a: _radius(canvas, brush, w, center, 40 * m, 560 * m, 240, *a, layer)
    K = lambda r0, r1, spacing, arc, seed: _circle(canvas, brush, w, center, r0 * m, r1 * m, spacing * m, arc, seed, layer)
    if technique == 'радиус':
        return R(5000)
    if technique == 'круг':
        return K(80, 520, 24, np.pi / 2, 6000)
    return (R(5000) + K(80, 520, 24, np.pi / 2, 6000) + R(5500) + K(90, 530, 24, np.pi / 2, 6500)
            + K(100, 500, 30, np.pi, 6700) + K(115, 515, 30, np.pi, 6900))


def measurement_circle(canvas, center, r_max=None, light=False):
    """Профиль L(r) от центра — медиана по секторам 5° средних L в кольцах по 1 px; ядро — медиана r 5–40, фон —
    последние 10 % профиля. Спад 10–90 % — от последнего места до середины, где тон ещё выше 90 % пути от фона к ядру,
    до первого после середины ниже 10 %. Ступень — наибольшая разность профиля на 8 px в спаде (±16 px). σ по углу —
    σ средних L 72 секторов на кольце середины ±4 px; 'σ мелкая' — её часть с периодом короче 60° (лучи, ворс),
    'σ крупная' — длиннее (форма пятна). Ответ — словарь с 'профиль' (numpy)."""
    cx, cy = center
    r_max = int(r_max or min(cx, cy, canvas.width - cx, canvas.height - cy))
    x0, y0 = int(cx - r_max), int(cy - r_max)
    frame = (x0, y0, x0 + 2 * r_max, y0 + 2 * r_max)
    L = lightness(np.clip(from_linear(canvas.image(light=light, region=frame)) * 255.0, 0.0, 255.0)).astype(np.float64)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]]
    dx, dy = xx + x0 + 0.5 - cx, yy + y0 + 0.5 - cy
    R = np.hypot(dx, dy)
    A = np.degrees(np.arctan2(dy, dx)) % 360
    ri, ai = R.astype(int), (A // 5).astype(int)
    m = ri < r_max
    sum, count = np.zeros((r_max, 72)), np.zeros((r_max, 72))
    np.add.at(sum, (ri[m], ai[m]), L[m])
    np.add.at(count, (ri[m], ai[m]), 1)
    seconds = np.where(count > 0, sum / np.maximum(count, 1), np.nan)
    profile = np.nanmedian(seconds, axis=1)
    a, b = float(np.nanmedian(profile[5:40])), float(np.nanmedian(profile[-max(8, r_max // 10):]))
    t = (profile - b) / (a - b)
    with_ = int(np.nanargmin(np.abs(t[20:] - 0.5))) + 20
    above, below = np.nonzero(t[:with_ + 1] > 0.9)[0], np.nonzero(t[with_:] < 0.1)[0]
    i90 = int(above[-1]) + 1 if len(above) else 0
    i10 = with_ + int(below[0]) if len(below) else r_max
    d = np.abs(profile[8:] - profile[:-8])
    lo, hi = max(0, i90 - 16), min(len(d), i10 + 16)
    st = int(np.nanargmax(d[lo:hi])) + lo
    ring = (R > with_ - 4) & (R < with_ + 4)
    a2 = (A // 2.5).astype(int)
    seconds2 = np.array([L[ring & (a2 == k)].mean() for k in range(144)])
    F = np.abs(np.fft.rfft(seconds2 - seconds2.mean())) / len(seconds2) * np.sqrt(2)
    seconds5 = np.array([L[ring & (ai == k)].mean() for k in range(72)])
    return {'ядро': round(a, 1), 'фон': round(b, 1), 'r_90': i90, 'r_10': i10, 'спад': i10 - i90, 'середина': with_,
            'ступень': round(float(d[st]), 2), 'где': st + 4, 'σ по углу': round(float(seconds5.std()), 2),
            'σ крупная': round(float(np.sqrt((F[1:7] ** 2).sum())), 2),
            'σ мелкая': round(float(np.sqrt((F[7:] ** 2).sum())), 2), 'профиль': profile}


def sheet(technique='крест', m=1.0, cores=None, journal=None):
    """Холст: небо, диск r 150, растушёвка. Ответ — (холст, замер диска до растушёвки, замер после)."""
    canvas = Canvas(int(1200 * m), int(1200 * m), ground=GRAtNT, px_on_mm=8.0 * m, cores=cores, journal=journal)
    ts = (600 * m, 600 * m)
    sky(canvas, m)
    disk(canvas, ts, 150 * m, m)
    to = measurement_circle(canvas, ts)
    blend(canvas, ts, m, technique)
    return canvas, to, measurement_circle(canvas, ts)


if __name__ == '__main__':
    arg = sys.argv[1:]
    def _option(name, umolch):
        return arg[arg.index(name) + 1] if name in arg else umolch
    technique, m = _option('--приём', 'крест'), float(_option('--масштаб', 1.0))
    where_to = next((a for i, a in enumerate(arg) if not a.startswith('--') and (i == 0 or not arg[i - 1].startswith('--'))), '.')
    os.makedirs(where_to, exist_ok=True)
    canvas, to, after = sheet(technique, m, journal=os.path.join(where_to, 'пятно_по_кругу.jsonl'))
    for name, z in (('диск', to), (technique, after)):
        print(f"{name}: L {z['ядро']} → {z['фон']}, спад 10–90 % {z['спад']} px (r {z['r_90']}–{z['r_10']}), "
              f"ступень {z['ступень']} L у r {z['где']}, σ по углу {z['σ по углу']} L (мелкая {z['σ мелкая']}, "
              f"крупная {z['σ крупная']})")
    canvas.look(path=os.path.join(where_to, 'пятно_по_кругу.png'))
