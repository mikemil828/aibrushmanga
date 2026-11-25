# READING VIEW of движок/примеры/цикл.py with names in English. Not runnable; edit the original.
"""Рабочий цикл на чистом холсте: пробник → кладка → кроп → замер → правка. Лист проб кисти, не кусок картины.
Замер — ответом самого мазка (`о = х.мазок(…)`: `о.цвет`, `о.закрыто`, `о.просветы`, `о.ширина`, `о.сырое`, `о.край`)
и вызовами холста (`х.борозды`); своего счёта по пикселям в примере нет.

    python3 примеры/цикл.py [папка] [--мал] [--попытка N]   # из папки «движок»; --мал — лист вдвое меньше (для теста)

Лист 640×400 px на сером грунте, четыре пробы, каждая — своим слоем:
1. заливка — плоская 100 px, ряды внахлёст, нагрузка от пробника; замер — просветы хода (грунт в бороздах) и борозды σ L;
2. шкала нажима — круглая 32 px, четыре хода нажимом 0,2…0,8; замер — ширина следа и L хода;
3. сырое в сырое — светлые ходы поперёк тёмной полосы: по сырому своего слоя и по высохшему в новом слое; замер — L хода
   и сколько сырого он подобрал;
4. край — мягкая 48 px нажимом 0,35 и 0,85; замер — кайма (от 10 до 50 % уровня тела следа поперёк хода).
Один прогон — одна попытка (без номера — последняя в ПОПЫТКИ). После прогона открыть кроп `цикл_N.png` глазами,
решить, что не так, и дописать в ПОПЫТКИ строку: что увидел → что меняю. Число подтверждает глаз, а не решает.
В живой сессии (`python3 -i примеры/цикл.py …`) — `х.откатить(снимок)` и `писать_лист(…)` с новым.
Импорт ничего не кладёт: `from примеры.цикл import пробник, писать_лист, кроп, замер` (из «движок»).
Журнал пишет холст сам: `цикл.jsonl` рядом; `python -m кисть.повтор цикл.jsonl --out повтор.png` — те же пиксели.
Нужны numpy и Pillow (кроп)."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # папка «движок»

from brush import Canvas, Brush, gesture

GRAtNT = (150, 141, 126)
TYoMNAYa, WithInETLAYa = (52, 60, 78), (226, 220, 200)


# ---------- 1. пробник: какая нагрузка кроет грунт ----------
def probe(ground, loads=(0.24, 0.45, 0.6), width=100):
    """Ход плоской на пробу (`х.проба`: положить, взять ответ мазка, откатить) — какая нагрузка кроет грунт.
    Ответ — наименьшая, при которой просветов (места под кистью, где краска не легла) меньше 0,5 %."""
    trial, brush = Canvas(500, 200, ground=ground, px_on_mm=8.0), Brush.from_file('пресеты/плоская.json', width=width)
    gaps = {}
    for load in loads:
        response, _ = trial.trial(gesture([[40, 100], [250, 102], [460, 100]], width, 'остановка', seed=1, pressure=0.65),
                           TYoMNAYa, brush, seed=1, load=load, light=False)    # о — ответ мазка: что вышло
        gaps[load] = 100 * response.gaps
        print(f'пробник: нагрузка {load}: просветы {gaps[load]:.2f} %, закрыто {response.covered:.2f}, L {response.color["L"]:.0f}')
    return min((n for n, g in gaps.items() if g < 0.5), default=max(loads))


# ---------- 2. кладка: четыре пробы ----------
def write_sheet(canvas, m, load, spread, even=None):
    """Лист проб на холсте `х`; `м` — масштаб листа (1 или 0,5). Ответ — ответы мазков по пробам: {проба: [о, …]}."""
    Sh, response = canvas.width, {'заливка': [], 'шкала': [], 'сырое': [], 'край': []}
    canvas.mark('заливка', stage=1, object='заливка')
    brush = Brush.from_file('пресеты/плоская.json', width=100 * m, spread_tone=spread)
    for i, y in enumerate((50, 110)):                       # ряды внахлёст на 40 %, ход за края холста
        response['заливка'].append(canvas.stroke(gesture([[-80 * m, y * m], [Sh / 2, (y + 3) * m], [Sh + 80 * m, y * m]], 100 * m, 'остановка', seed=10 + i, pressure=0.65),
                                    TYoMNAYa, brush, layer='заливка', seed=10 + i, load=load, even=even))

    canvas.mark('шкала нажима', stage=2, object='шкала')
    brush = Brush.from_file('пресеты/круглая.json', width=32 * m)
    for i, pressure in enumerate(NAZhAndMY):
        x = (20 + 156 * i) * m
        response['шкала'].append(canvas.stroke(gesture([[x, 210 * m], [x + 130 * m, 210 * m]], 32 * m, 'остановка', seed=20 + i, pressure=pressure),
                                  WithInETLAYa, brush, layer='шкала', seed=20 + i, load=0.6))
        brush.wipe()

    canvas.mark('сырое в сырое', stage=3, object='сырое')
    brush = Brush.from_file('пресеты/плоская.json', width=64 * m)
    for i, x0 in enumerate((20, 340)):                      # две тёмные полосы в одном слое; правая — досуха
        canvas.stroke(gesture([[x0 * m, 290 * m], [(x0 + 280) * m, 290 * m]], 64 * m, 'остановка', seed=30 + i, pressure=0.65),
                TYoMNAYa, brush, layer='сырое', seed=30 + i, load=0.8)
        brush.wipe()
    canvas.dry(1.0, region=(int(330 * m), int(240 * m), Sh, int(340 * m)))
    brush = Brush.from_file('пресеты/плоская.json', width=24 * m)
    for i, (x, layer) in enumerate(((160, 'сырое'), (480, 'по_сухому'))):   # светлый ход поперёк: по сырому — тем же слоем
        response['сырое'].append(canvas.stroke(gesture([[x * m, 250 * m], [x * m, 330 * m]], 24 * m, 'остановка', seed=40 + i, pressure=0.75),
                                  WithInETLAYa, brush, layer=layer, seed=40 + i, load=0.3))
        brush.wipe()

    canvas.mark('край мягкой', stage=4, object='край')
    brush = Brush.from_file('пресеты/мягкая.json', width=48 * m)
    for i, (x0, pressure) in enumerate(((20, 0.35), (340, 0.85))):
        response['край'].append(canvas.stroke(gesture([[x0 * m, 365 * m], [(x0 + 280) * m, 365 * m]], 48 * m, 'остановка', seed=50 + i, pressure=pressure),
                                 WithInETLAYa, brush, layer='край', seed=50 + i, load=0.05))
        brush.wipe()
    return response


NAZhAndMY = (0.2, 0.4, 0.6, 0.8)


# ---------- 3–4. глаз: кроп и замер ----------
def crop(canvas, where_to, name):
    """Кроп листа со светом, ×2 у малого листа, PNG 8 бит в `куда` — `х.взгляд`."""
    path = os.path.join(where_to, name)
    canvas.look(scale=2 if canvas.width < 400 else 1, path=path)
    return path


def measurement(canvas, response, m):
    """Числа проб — из ответов мазков `о` (`писать_лист`) и мерки холста: заливка (просветы хода %, борозды σ L), шкала
    (ширина следа и L по нажиму), сырое (L хода, доля подобранного сырого), край (кайма, px)."""
    z = {'заливка грунт %': round(100 * max(a.gaps for a in response['заливка']), 2),
         'заливка фактура σ L': round(canvas.furrows((int(20 * m), int(20 * m), canvas.width - int(20 * m), int(140 * m)), width=12 * m)['σ'], 1)}
    for pressure, a in zip(NAZhAndMY, response['шкала']):
        z[f'шкала {pressure}: ширина px, L'] = (round(a.width / m), round(a.color['L']))
    for name, a in zip(('по сырому', 'по сухому'), response['сырое']):
        z[f'сырое: L хода {name}'] = a.color['L']
        z[f'сырое: подобрано {name}'] = a.wet['подобрано']
    for pressure, a in zip((0.35, 0.85), response['край']):
        z[f'край {pressure}: кайма px'] = round(a.edge['кайма'] / m, 1)
    return z


# Попытки листа: каждая следующая дописана после взгляда на кроп прошлой — что увидел и что меняю.
PResponsePYTBrushAnd = [
    {'разброс': 1.0, 'ровно': None, 'увидел': '', 'меняю': 'кисть как есть'},
    {'разброс': 1.0, 'ровно': 0.4, 'увидел': 'цикл_1.png: в заливке борозды, по ним нитями светит грунт',
     'меняю': 'заливка ровнее: ровно 0.4'},
]   # цикл_2.png: борозды ушли, грунта нет; шкала, сырое и край — как в первой: правка тронула только заливку

# ---------- попытка: кладка, кроп, замер; правка — следующей строкой ПОПЫТКИ после взгляда на кроп ----------
if __name__ == '__main__':
    small = '--мал' in sys.argv
    where_to = next((a for a, to in zip(sys.argv[1:], sys.argv) if not a.startswith('--') and to != '--попытка'), '.')
    os.makedirs(where_to, exist_ok=True)
    m = 0.5 if small else 1.0
    attempt = int(sys.argv[sys.argv.index('--попытка') + 1]) if '--попытка' in sys.argv else len(PResponsePYTBrushAnd)
    NAGRAtZBrushA = probe(GRAtNT)

    canvas = Canvas(int(640 * m), int(400 * m), ground=GRAtNT, px_on_mm=8.0, journal=os.path.join(where_to, 'цикл.jsonl'))   # журнал пишется сам
    p = PResponsePYTBrushAnd[attempt - 1]
    snapshot = canvas.snapshot()                                  # до куска: откат вернёт холст и журнал
    response = write_sheet(canvas, m, NAGRAtZBrushA, p['разброс'], p['ровно'])     # ответы мазков: что вышло у каждого хода
    print(response['сырое'][0])                                 # ответ мазка словами: цвет, закрыто, ширина, сырое, край
    print(f'попытка {attempt} — кроп:', crop(canvas, where_to, f'цикл_{attempt}.png'), '· замер:', measurement(canvas, response, m))
    canvas.note({'попытка': attempt, 'увидел': p['увидел'], 'меняю': p['меняю'], 'замер': measurement(canvas, response, m)}, stage=4, object='край')
    canvas.save(os.path.join(where_to, 'цикл.npz'))
    print('журнал:', canvas.journal.path)
