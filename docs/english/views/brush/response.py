# READING VIEW of движок/кисть/ответ.py with names in English. Not runnable; edit the original.
"""v4, 3г (К-1): ответ мазка — прямоугольник и замер следа. Только читает: холст, кисть, сиды и журнал не меняются.

`Холст.мазок` отвечает `Ответ` — тем же кортежем (x0, y0, x1, y1), что и раньше, с полями замера. Сразу, в конце
самого мазка (на ядрах — в процессе мазка, числа те же, что на одном ядре), считается дешёвое: `цвет` — тон холста по
закрашенным пикселям следа, `тело`, `закрыто`, `сырое` — из полей мазка (покрытие новой плёнки, касание кисти, снятое
сырое). Дорогое — `ширина`, `просветы`, `разрывы`, `край` — по запросу: считается при первом обращении к любому из
них из полей следа, которые ответ несёт с собой (покрытие и касание в прямоугольнике, 1 байт на пиксель, и путь);
от того, что легло на холст после мазка, не зависит.

Развёртка следа: поля покрытия и касания берутся вдоль хода рядами поперёк него (ряд на 1 px хода, у кисти шире
32 px — на 1/32 ширины; отсчёт поперёк через 0,5 px, у кисти от 24 px — через 1 px); в ней меряются ширина, просветы,
разрывы и край. Уровень тела — 90-й процентиль покрытия под кистью; «краска легла» — покрытие от половины его.
"""
import math

import numpy as np

from . import eye
from .paint import from_linear

UNDER_BRUSH = 0.5        # касание, от которого пиксель — под кистью
TELResponse = 0.5              # доля уровня тела, от которой краска «легла» (край следа — по ней же)
EDGE_BAND_FROM = 0.1          # кайма края: от этой доли уровня тела до ТЕЛО, px
RYaDResponseIn = 1500            # потолок рядов развёртки: длинный ход берётся реже
TONE_BY = 16384          # тон следа: не больше стольких закрашенных пикселей (у большого следа — каждый k-й)
_BY_REQUEST = ('ширина', 'просветы', 'разрывы', 'край')
_HAndTAYuT = _BY_REQUEST + ('цвет', 'тело', 'закрыто', 'сырое', 'чужое', 'серый', 'видно', 'закрыл', 'слова')


class Response(tuple):
    """Ответ `Холст.мазок`: записанный прямоугольник (x0, y0, x1, y1) — как кортеж, и что вышло:

    цвет      — тон следа на холсте, как у `х.средний`: {'L', 'C', 'h', 'rgb', 'hex'}; среднее тона холста (без света
                рельефа, сквозь слои выше) по закрашенным пикселям следа — где покрытие новой плёнки не ниже половины
                уровня тела (если их больше 16 тыс. — по каждому k-му). None — следа не видно;
    тело      — 0–1: покрытие тела следа (90-й процентиль покрытия новой плёнки под кистью): кроющий ход ~1,
                протирка 0,2–0,4;
    закрыто   — 0–1: насколько новая краска закрыла лежавшее под кистью (среднее покрытие плёнки под кистью):
                кроющий ход ~1, протирка 0,2–0,4, сухая кисть — доля островов;
    просветы  — 0–1: доля места внутри следа (между его кромками, от первой краски хода до последней), где краска не
                легла — покрытие ниже половины уровня тела: борозды до грунта, острова сухой кисти, порванный ход;
    ширина    — px: ширина следа поперёк хода (медиана по ходу, от края до края);
    сырое     — {'по сырому': доля места под кистью, где лежала сырая краска, 'подобрано': какую долю её кисть сняла,
                'к положенному': снятое к положенному этим мазком (по объёму краски)}; у мастихина ещё 'на нож' —
                какая доля сырого под ножом срезана на нож и ушла с холста (нажим от 0,85);
    разрывы   — список мест, где ход порвался: {'от': (x, y), 'до': (x, y), 'длина': px, 'место': (доля хода от, до),
                'причина': 'нажим' — кисть оторвалась от холста, 'краска' — кисть шла, краска не легла}; вход и подъём
                кисти в начале и конце хода — не разрыв, а краска, что кончилась раньше хода, — разрыв до его конца;
    край      — {'кайма': px от 10 до 50 % уровня тела поперёк хода (медиана двух сторон), 'левый', 'правый' — то же
                по сторонам (слева и справа по ходу), 'рваность': px, насколько кромка гуляет вдоль хода, 'слово':
                'жёсткий' | 'мягкий' | 'рваный'}; None — следа не видно.
    осталось  — Д-1: сколько живой краски осталось в ворсе кисти после хода, доля полной кисти 0–1 (`к.набрать(цвет,
                количество)`, ход без цвета); None — краска не живая: кисть полна на каждом ходе.

    чужое     — Г-8: ход по сырому чужому цвету — {'подобрано': снятое к положенному, 'лежало', 'названо': тоны (L, C, h,
                rgb, hex), 'ΔL', 'Δh', 'Δab': тон следа от названного ('Δh' None — у одного из цветов тона нет)};
                None — нет (пороги — `встреча.py`);
    серый     — Г-8: мягкая кроющей краской по сухому в новом слое — {'закрыто', 'путь': какую долю пути от лежащего к
                названному прошёл тон следа, 'лежало', 'названо'}; None — нет.
    видно     — Сл-1: доля следа, видная на листе, — не закрытая слоями выше (`под_верхом.py`); 1 — над ходом пусто,
                ход в верхнем слое или следа нет;
    закрыл    — Сл-1: имя слоя, что закрыл больше всех, когда `видно` меньше 0,9; None — нет.

    `цвет`, `тело`, `закрыто`, `сырое`, `чужое`, `серый`, `видно` сочтены вместе с мазком; `ширина`, `просветы`, `разрывы`, `край` — при первом
    обращении к любому из них (или `print(о)`): развёртка следа из его полей, что ответ несёт с собой.
    `print(о)` — строкой словами; `tuple(о)`, `о[0]`, `x0, y0, x1, y1 = о`, `о == (…)` — как у кортежа."""

    left = None
    foreign = None
    gray = None
    visible = 1.0
    closed = None

    def __new__(cls, rectangle, **fields):
        response = tuple.__new__(cls, rectangle)
        response.__dict__.update(fields)
        return response

    def __getnewargs__(self):
        return (tuple(self),)

    def __getattribute__(self, name):
        if name in _HAndTAYuT:                    # 5б: ответ прочитан — в счёт подсказок (С5)
            d = tuple.__getattribute__(self, '__dict__')
            at = d.get('_учёт')
            if at is not None:
                d['_учёт'] = None
                at.read()
        return tuple.__getattribute__(self, name)

    def __getattr__(self, name):
        if name in _BY_REQUEST and '_след' in self.__dict__:
            self.__dict__.update(_unfold(self.__dict__.pop('_след')))
            return self.__dict__[name]
        raise AttributeError(name)

    @property
    def words(self):
        h = [f'{tuple(self)}']
        if self.color is None:
            h.append('следа не видно')
        else:
            h.append(f"цвет L {self.color['L']:.0f} C {self.color['C']:.3f} h {self.color['h']:.0f} {self.color['hex']}")
        h.append(f'закрыто {self.covered:.2f}, просветы {self.gaps:.2f}')
        if self.width:
            h.append(f'ширина {self.width:.0f} px')
        if self.wet['по сырому'] > 0:
            h.append(f"по сырому {self.wet['по сырому']:.2f}, подобрано {self.wet['подобрано']:.2f}" + (f", на нож {self.wet['на нож']:.2f}" if self.wet.get('на нож') else ''))
        if self.foreign is not None:
            from .encounter import word_foreign
            h.append(word_foreign(self.foreign))
        if self.gray is not None:
            from .encounter import word_gray
            h.append(word_gray(self.gray, self.color['L']))
        if self.closed is not None:
            from .under_top import word
            h.append(word(self.visible, self.closed))
        if self.break_:
            h.append('разрывы: ' + '; '.join(f"{r['от']}–{r['до']} {r['длина']:.0f} px ({r['причина']})" for r in self.break_))
        if self.left is not None:
            h.append(f'на кисти осталось {self.left:.2f}' + (' — краска кончилась, набрать' if self.left < 0.02 else ''))
        if self.edge is not None:
            h.append(f"край {self.edge['слово']} (кайма {self.edge['кайма']:.1f} px, рваность {self.edge['рваность']:.1f} px)")
        return ' · '.join(h)

    def __str__(self):
        return self.words


def _tone(canvas, p, where):
    """Средний тон холста по пикселям `где` прямоугольника `п` — тем же счётом, что `глаз.средний` (L серой шкалы,
    a, b OKLab; тон без света рельефа)."""
    if not where.any():
        return None
    tone = canvas.image(light=False, region=p)[where]
    if len(tone) > TONE_BY:
        tone = tone[::-(-len(tone) // TONE_BY)]
    lab = eye.oklab(np.clip(from_linear(tone) * 255.0, 0.0, 255.0))
    L, ma, mb = float(eye._L(lab[:, 0]).mean()), float(lab[:, 1].mean()), float(lab[:, 2].mean())
    C = math.hypot(ma, mb)
    h = math.degrees(math.atan2(mb, ma)) % 360.0 if C >= 1e-6 else 0.0
    rgb = eye.color(L, C, h)
    return {'L': round(L, 2), 'C': round(C, 4), 'h': round(h, 1), 'rgb': rgb, 'hex': eye.hex(rgb)}


def _take(fields, X, Y):
    """Поля (одной формы) в точках (X, Y) их пикселей (центр пикселя — целое + 0,5), билинейно; вне поля — 0."""
    In, Sh = fields[0].shape
    u, v = X - 0.5, Y - 0.5
    j0, i0 = np.floor(u), np.floor(v)
    fu, fv = u - j0, v - i0
    j0, i0 = np.clip(j0.astype(np.int64) + 1, 0, Sh), np.clip(i0.astype(np.int64) + 1, 0, In)   # в поле с каймой нулей в 1 px
    inside = (u > -1.0) & (u < Sh) & (v > -1.0) & (v < In)                                     # дальше каймы — ноль, а не край поля
    w00, w01, w10, w11 = (1 - fv) * (1 - fu) * inside, (1 - fv) * fu * inside, fv * (1 - fu) * inside, fv * fu * inside
    result = []
    for a in fields:
        b = np.zeros((In + 2, Sh + 2))
        b[1:-1, 1:-1] = a
        result.append(b[i0, j0] * w00 + b[i0, j0 + 1] * w01 + b[i0 + 1, j0] * w10 + b[i0 + 1, j0 + 1] * w11)
    return result


def _unrolling(alpha, touch, x, y, W, angle):
    """Поля вдоль хода рядами поперёк: (покрытие, касание — (рядов, отсчётов), x и y рядов, длина хода до ряда px)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    steps = np.hypot(np.diff(x), np.diff(y))
    length = np.concatenate([[0.0], np.cumsum(steps)])
    n = int(min(max(round(length[-1] / max(1.0, W / 32.0)), 1), RYaDResponseIn)) + 1
    s = np.linspace(0.0, length[-1], n)
    xs, ys = (np.interp(s, length, x), np.interp(s, length, y)) if length[-1] > 0 else (x[:1].copy(), y[:1].copy())
    s = s[:len(xs)]
    d = max(2.0, 0.1 * W)                                    # касательная — по хорде ±d px: дрожь пути её не качает
    tx = np.interp(s + d, length, x) - np.interp(s - d, length, x) if length[-1] > 0 else np.ones(1)
    ty = np.interp(s + d, length, y) - np.interp(s - d, length, y) if length[-1] > 0 else np.zeros(1)
    norm = np.hypot(tx, ty)
    tx, ty = np.where(norm > 1e-9, tx / np.maximum(norm, 1e-9), 1.0), np.where(norm > 1e-9, ty / np.maximum(norm, 1e-9), 0.0)
    R, dt = 0.65 * W + 6.0, (0.5 if W < 24 else 1.0)         # поперёк: шире следа с запасом; отсчёт 0,5 px, у широкой кисти 1 px
    t = np.arange(-R, R + 0.5 * dt, dt)                      # t > 0 — справа по ходу (y вниз)
    X = xs[:, None] - ty[:, None] * t[None, :] - angle[0]
    Y = ys[:, None] + tx[:, None] * t[None, :] - angle[1]
    A, K = _take((alpha, touch), X, Y)
    return A, K, xs, ys, s, t


def _rows(m):
    """Участки подряд идущих True: [(от, до включительно), …]."""
    m = np.asarray(m, bool)
    if not m.any():
        return []
    brush = np.flatnonzero(np.diff(np.concatenate([[False], m, [False]]).astype(np.int8)))
    return [(int(a), int(b) - 1) for a, b in zip(brush[::2], brush[1::2])]


def _run_medians(a, window):
    """Бегущая медиана ряда (окно нечётное), края — зеркалом."""
    window = int(window) | 1
    if window <= 1 or len(a) < 3:
        return a.copy()
    r = min(window // 2, len(a) - 1)
    b = np.pad(a, r, mode='reflect')
    return np.median(np.lib.stride_tricks.sliding_window_view(b, 2 * r + 1), axis=1)


def _quantile(a, q):
    """q-й квантиль ряда (ближайший по порядку) — без сортировки всего ряда."""
    k = int(round(q * (a.size - 1)))
    return float(np.partition(a, k)[k])


def measure(canvas, rectangle, fields):
    """Ответ мазка по полям следа от `укладка.положить_мазок(замер=…)` и тону холста `х` в его прямоугольнике: дешёвое —
    сразу, развёртка следа (`_развернуть`) — по запросу, из полей, сложенных в ответ."""
    alpha, touch, W = fields['альфа'], fields['касание'], float(fields['W'])
    under = touch > UNDER_BRUSH
    under_brush = alpha[under]
    place = under_brush.size
    body = _quantile(under_brush, 0.9) if place else 0.0
    threshold = TELResponse * body
    visible = place > 0 and body >= 0.01
    wet = fields['сырое']
    if wet is not None and place:
        lay_there = wet[under]
        by_wet, lay_there = float((lay_there > 1e-4).mean()), float(lay_there.sum())
    else:
        lay_there, by_wet = 0.0, 0.0
    removed, laid = float(fields['снято']), float(fields['положено'])
    response = dict(color=_tone(canvas, rectangle, alpha >= threshold) if visible else None, body=round(body, 4),
             covered=round(float(under_brush.mean()), 4) if place else 0.0,
             wet={'по сырому': round(by_wet, 3), 'подобрано': round(min(removed / lay_there, 1.0), 3) if lay_there > 1e-9 else 0.0,
                    'к положенному': round(removed / laid, 3) if laid > 1e-9 else 0.0})
    if 'на_ноже' in fields:                      # «мастихин»: срезанное на нож с холста ушло
        response['сырое']['на нож'] = round(min(float(fields['на_ноже']) / lay_there, 1.0), 3) if lay_there > 1e-9 else 0.0
    if not visible:
        return Response(rectangle, width=0.0, gaps=1.0, break_=[], edge=None, **response)
    x, y = np.asarray(fields['x']), np.asarray(fields['y'])
    k = max(1, int(len(x) / max(float(np.hypot(np.diff(x), np.diff(y)).sum()), 1.0)))      # путь с собой — точка на ~1 px хода
    brush8 = lambda a: np.rint(np.clip(a, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Response(rectangle, _trace=dict(alpha=brush8(alpha), touch=brush8(touch), body=body, W=W, angle=tuple(fields['угол']),
                                           x=np.append(x[:-1:k], x[-1]).astype(np.float32), y=np.append(y[:-1:k], y[-1]).astype(np.float32)), **response)


def _unfold(trace):
    """Дорогая часть ответа — развёртка следа: {'ширина', 'просветы', 'разрывы', 'край'} из полей, что ответ несёт с собой."""
    W, body = trace['W'], trace['тело']
    threshold = TELResponse * body
    response = dict(width=0.0, gaps=1.0, break_=[], edge=None)
    A, K, xs, ys, s, t = _unrolling(trace['альфа'] / 255.0, trace['касание'] / 255.0, trace['x'], trace['y'], W, trace['угол'])
    laid, brush = A >= threshold, K > UNDER_BRUSH
    dt, spacing = float(t[1] - t[0]), (float(s[1] - s[0]) if len(s) > 1 else 1.0)     # отсчёт поперёк и ряд вдоль, px
    f = brush.sum(1) * dt                                       # ширина касания ряда, px
    g = (laid & brush).sum(1) * dt                             # из неё — с краской
    has = laid.any(1)
    left = np.where(has, t[np.argmax(laid, 1)], np.nan)      # кромки следа ряда: первая и последняя точка с краской
    right = np.where(has, t[laid.shape[1] - 1 - np.argmax(laid[:, ::-1], 1)], np.nan)
    range = np.where(has, right - left + dt, 0.0)
    if has.any():                                            # просветы — внутри следа: между кромками ряда, от первого ряда с краской до последнего
        a, b = int(np.argmax(has)), len(has) - 1 - int(np.argmax(has[::-1]))
        inside = brush[a:b + 1] & np.where(has[a:b + 1, None], (t[None, :] >= left[a:b + 1, None]) & (t[None, :] <= right[a:b + 1, None]), True)
        response['просветы'] = round(float((inside & ~laid[a:b + 1]).sum() / max(int(inside.sum()), 1)), 4)
    f_body = float(np.median(f[f > 0])) if (f > 0).any() else 0.0
    torn_off = f < 0.25 * f_body
    share = np.where(f > 0, g / np.maximum(f, dt), 0.0)
    share = _run_medians(share, max(3, min(int(round(0.25 * W / spacing)), 41)))
    normal = float(np.percentile(share[~torn_off], 75)) if (~torn_off).any() else 0.0
    empty = ~torn_off & (share < min(0.5, 0.5 * normal))
    whole = ~torn_off & ~empty
    if whole.any():
        response['ширина'] = round(float(np.median(range[whole])), 1)
    # --- разрывы: кисть оторвалась между касаниями ('нажим') или шла без краски ('краска')
    shorter = max(3.0, 0.2 * W)
    first, last = (int(np.flatnonzero(whole)[0]), int(np.flatnonzero(whole)[-1])) if whole.any() else (len(s), -1)
    L = float(s[-1]) if len(s) > 1 and s[-1] > 0 else 1.0
    for reason, mask in (('нажим', torn_off), ('краска', empty)):
        for a, b in _rows(mask):
            inside = a > first and b < last
            tail = reason == 'краска' and a > first and s[b] - s[a] + spacing >= max(shorter, 0.5 * W)   # краска кончилась до конца хода
            if s[b] - s[a] + spacing < shorter or not (inside or tail):
                continue
            response['разрывы'].append({'от': (int(round(xs[a])), int(round(ys[a]))), 'до': (int(round(xs[b])), int(round(ys[b]))),
                                 'длина': round(float(s[b] - s[a] + spacing), 1), 'место': (round(float(s[a] / L), 3), round(float(s[b] / L), 3)),
                                 'причина': reason})
    response['разрывы'].sort(key=lambda r: r['место'])
    # --- край: кайма (10–50 % уровня тела) и рваность кромки вдоль хода, по сторонам
    if whole.sum() >= 3:
        bottom = A >= EDGE_BAND_FROM * body
        left10 = t[np.argmax(bottom, 1)]
        right10 = t[bottom.shape[1] - 1 - np.argmax(bottom[:, ::-1], 1)]
        fringe, ragged = [], []
        window = max(5, min(int(round(W / spacing)), 41))
        for rim, outer, sign in ((left, left10, 1.0), (right, right10, -1.0)):
            e = rim[whole]
            fringe.append(float(np.median(sign * (e - outer[whole]))))
            rest = e - _run_medians(e, window)
            ragged.append(float(np.percentile(rest, 90) - np.percentile(rest, 10)) / 2.563)
        edge_band, raggedness = max(0.0, 0.5 * (fringe[0] + fringe[1])), 0.5 * (ragged[0] + ragged[1])
        word = 'рваный' if raggedness >= max(1.5, 0.05 * W) else 'мягкий' if edge_band >= max(2.0, 0.05 * W) else 'жёсткий'
        response['край'] = {'кайма': round(edge_band, 1), 'левый': round(max(fringe[0], 0.0), 1), 'правый': round(max(fringe[1], 0.0), 1),
                     'рваность': round(raggedness, 2), 'слово': word}
    return response
