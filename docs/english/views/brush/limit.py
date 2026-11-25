# READING VIEW of движок/кисть/предел.py with names in English. Not runnable; edit the original.
"""З-2 (В-6): предел светлоты — сколько выйдет L у поля рядов краски по подложке и где его потолок, до кладки.
Счёт — живой стенд: поле кладётся на своём холсте этой самой кистью, краской и по такой же подложке, 16 раз.

Откуда предел (замер смены Движка 228). Белила (255, 255, 251) мягкой 6 мм нагрузкой 0,5 по высохшему сливочному
дают L 249–250, а не 254,6, и тремя проходами не светлее. Покрытие при этом 0,997–1,0: подложка почти не видна.
Светлоту съедает тон прядей: краска на кисти недомешана (`Кисть.разброс_тона`), тон пряди — множитель к цвету
краски, в среднем около 1, а светлее белого пряди быть не может. Тёмные пряди темнят, светлые обрезаны
белым, и среднее выходит ниже краски. С `разброс_тона=0` те же ряды дают 254,5, с 0,2 — 252,9. Новый проход тоже
кладёт пряди своего тона, поэтому проходы потолок не поднимают. Тёмная краска упирается в обратное: сквозь просветы
видна светлая подложка.

Стенд (З-2б, смена Движка 230; в мм — тот же при любом `px_на_мм`). Грунт (200, 190, 170); подложка — слой плоской
12,5 мм нагрузкой 0,6 цветом подложки, её ходы — свои у каждого сида: рельеф подложки водит ход (мастихин 4 мм
белилами по серой — σ поля 2,7 L по одной подложке и 9 L по разным). По сухой она высушена, поле идёт в слой над
ней и сохнет после каждого прохода; по сырой поле идёт в её же слой. Поле — ряды (`_геометрия`: не меньше 7, у узкой
кисти больше) прямых ходов жестом 'подъём' через четверть ширины кисти, ход — 40 мм или 6,67 ширины; замер — средняя
светлота окна посередине поля, как у `х.средний`, без света и со светом рельефа.

Сиды. Прогноз — среднее 16 полей сидов `СИДЫ` (свои у стенда: подложка и ходы у каждого свои). Предел и заказ `L=` —
перебор нагрузок `НАГРУЗКИ` и проходов 1–3 тремя сидами `СИДЫ_ПОИСКА`, найденная точка считается заново сидами
`СИДЫ`, и в ответ идёт этот второй счёт: лучшая из шумных сама завышена. Проходы — из одного прогона поля.

Подложка держится в памяти процесса (`_ПОДЛОЖКИ`, не больше `ПАМЯТЬ_МБ`): одна на сид, на все нагрузки и вызовы.
Сухая тёмная подложка (линейный цвет не выше `ПЕРЕКРАСКА`) не кладётся заново, а перекрашивается из серой L 180:
геометрия слоя от цвета не зависит, цвет слоя — линейный цвет краски на тон прядей (сверено кладкой: до тысячных L).
Светлую перекрасить нельзя — светлые пряди обрезаны белым до смешения рядов (перекраска врёт до 2,4 L), она кладётся
своим цветом.

Счёт идёт в процессах (`ПРОЦЕССОВ`, не больше ядер и `KIST_CORES`); из процесса-потомка пула — в нём самом. Только
счёт: холст, кисть и журнал модели не трогаются.
"""
import math
import os
import pickle
import warnings
from collections import OrderedDict
from dataclasses import asdict

import numpy as np

from . import calibration as calib

# ----------------------------------------------------------------- стенд
PX_ON_MM = 8.0                 # px/мм стенда, если не дан свой; длины ниже — в px при 8 px/мм
GRAtNT = (200, 190, 170)
UNDER_WIDTH = 100               # px: плоская подложки (12,5 мм)
UNDER_PAINT = 180.0             # L серой подложки-основы для перекраски
PEREBrushRAWithBrushA = 0.4               # линейный цвет подложки не выше — перекраской основы (sRGB до 168), выше — кладкой
ShAG = 0.25                     # шаг рядов поля, доля ширины кисти
RYaDResponseIn = 7                      # рядов поля у кисти 6 мм и шире
DLAndNA = 320                    # px: ход не короче (40 мм); у кисти шире 6 мм — 6,67 ширины (жест 'подъём' сходит на 2–4 ширинах)
FIELD_X = (0.25, 0.75)          # окно замера вдоль хода — доли длины хода
PRResponseCanvasResponseDResponseIn = 3                   # проходов в переборе предела и заказа
PASSES_MAX = 8              # проходов в вызове не больше
NAGRAtZBrushAnd = (0.1, 0.15, 0.2, 0.28, 0.4, 0.6, 0.9)      # нагрузки перебора предела и заказа
WithAndDY = tuple(range(41001, 41017))                     # 16 полей прогноза и ответа
SEEDS_SEARCH = tuple(range(42001, 42004))              # перебор предела и заказа
PRResponseTsEWithWithResponseIn = 10
CANVAS_MPX = 6.0                 # холст поля не больше, Мп
MEMORY_MB = 400                # подложки в памяти процесса не больше


def _gray(L):
    return (float(L),) * 3


def _geometry(W, px=PX_ON_MM):
    """(рядов, полуокно поперёк px, длина хода px) поля кистью W px при px/мм."""
    m = float(px) / PX_ON_MM
    h = max(0.5 * W, min(16.0 * m, 3.0 * W))
    rows = max(RYaDResponseIn, int(math.ceil(2.0 * (h + ShAG * W) / (ShAG * W))) + 1)
    return rows, h, max(float(DLAndNA) * m, 6.67 * W)


def _sizes(W, px=PX_ON_MM):
    """(Ш, В, окно, рядов, y0, длина) холста поля кистью W px."""
    m = float(px) / PX_ON_MM
    rows, h, length = _geometry(W, px)
    y0 = 30.0 * m + W / 2
    yc = y0 + (rows - 1) * ShAG * W / 2
    Sh, In = int(length + 80 * m), int(2 * y0 + (rows - 1) * ShAG * W)
    window = (int(40 * m + FIELD_X[0] * length), int(round(yc - h)), int(40 * m + FIELD_X[1] * length), int(round(yc + h)))
    return Sh, In, window, rows, y0, length


def _canvas(W, under, wet, seed, px=PX_ON_MM):
    """Холст поля кистью W px по подложке `под` (L или sRGB), её ходы — свои у сида. Ответ: (холст, окно, рядов, y0, длина)."""
    from .canvas import Canvas
    from .brush import Brush
    from .path import gesture
    m = float(px) / PX_ON_MM
    Sh, In, window, rows, y0, length = _sizes(W, px)
    canvas = Canvas(Sh, In, ground=GRAtNT, px_on_mm=float(px), cores=1, hints=False, measurement=False)
    shp = UNDER_WIDTH * m
    pl = Brush.from_file(os.path.join(calib._PREWithETY, 'плоская.json'), width=shp)
    tsv = tuple(float(v) for v in under) if isinstance(under, (tuple, list, np.ndarray)) else _gray(under)
    spacing = 50 * m
    for i in range(int(math.ceil((In + spacing) / spacing))):
        y = i * spacing
        canvas.stroke(gesture([[-40 * m, y], [Sh + 40 * m, y]], shp, 'остановка', seed=seed * 100 + i, pressure=0.7), tsv, pl.with_(),
               layer='под', load=0.6)
    if not wet:
        canvas.dry(1.0)
    return canvas, window, rows, y0, length


def _rows(canvas, brush, W, rows, y0, length, paint, load, pressure, seed, pass_, layer, px=PX_ON_MM):
    """Один проход поля: `рядов` ходов кистью к краской (L, sRGB или `Краска`)."""
    from .path import gesture
    from .paint import Paint
    m = float(px) / PX_ON_MM
    if isinstance(paint, Paint):
        tsv = paint
    else:
        tsv = tuple(float(v) for v in paint) if isinstance(paint, (tuple, list, np.ndarray)) else _gray(paint)
    for j in range(rows):
        with_ = seed * 1000 + 100 * pass_ + j
        y = y0 + j * ShAG * W
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            canvas.stroke(gesture([[40 * m, y], [40 * m + length, y]], W, 'подъём', seed=with_, pressure=float(pressure)), tsv, brush.with_(),
                   layer=layer, load=float(load), seed=with_)


def field(preset, width_mm, load, pressure, paint, underlayer, seed, passes=1, wet=False):
    """Одно живое поле стенда кистью пресета при 8 px/мм, подложка кладкой своим цветом: (L, L со светом) окна после
    `проходов` проходов (по сухой каждый сохнет, по сырой — подряд в слой подложки). Для сверки со стороны."""
    from .brush import Brush
    W = float(width_mm) * PX_ON_MM
    brush = Brush.from_file(os.path.join(calib._PREWithETY, preset + '.json'), width=W)
    canvas, window, rows, y0, length = _canvas(W, underlayer, wet, seed)
    for p in range(int(passes)):
        _rows(canvas, brush, W, rows, y0, length, paint, load, pressure, seed, p, 'под' if wet else 'верх')
        if not wet:
            canvas.dry(1.0, layer='верх')
    return canvas.mean(window)['L'], canvas.mean(window, light=True)['L']


# ----------------------------------------------------------------- подложки в памяти
_PResponseDLResponseZhBrushAnd = OrderedDict()      # (Ш, В, px, сырая, цвет или None — основа, сид) -> холст с подложкой, pickle
_ZAMERY = OrderedDict()        # (ключ точки, нагрузка, сиды, проходов) -> замер
_ZAMERResponseIn = 256


def _recolor(tsv):
    """Можно ли сухую подложку sRGB `цв` перекрасить из основы: ни одна прядь не дойдёт до белого."""
    from .paint import in_linear
    return float(np.max(in_linear(np.asarray(tsv, np.float64)))) <= PEREBrushRAWithBrushA


def _option_underlayer(W, px, tsv, wet, seed):
    Sh, In = _sizes(W, px)[:2]
    base = (not wet) and _recolor(tsv)
    return (Sh, In, float(px), bool(wet), None if base else tuple(float(v) for v in tsv), int(seed))


def _underlayer_piece(z):
    W, px, tsv, wet, seed = z
    option = _option_underlayer(W, px, tsv, wet, seed)
    canvas = _canvas(W, UNDER_PAINT if option[4] is None else option[4], wet, seed, px)[0]
    return option, pickle.dumps(canvas, protocol=pickle.HIGHEST_PROTOCOL)


def _remember(option, b):
    _PResponseDLResponseZhBrushAnd[option] = b
    _PResponseDLResponseZhBrushAnd.move_to_end(option)


def _tight(needed):
    """Выкинуть давние подложки сверх `ПАМЯТЬ_МБ` (нужные этому счёту остаются)."""
    total = sum(len(b) for b in _PResponseDLResponseZhBrushAnd.values())
    for option in list(_PResponseDLResponseZhBrushAnd):
        if total <= MEMORY_MB * 1e6:
            break
        if option not in needed:
            total -= len(_PResponseDLResponseZhBrushAnd.pop(option))


def _take_underlayer(W, px, tsv, wet, seed):
    """Свой холст с подложкой цвета `цв` (sRGB) сида — из памяти; основа перекрашена."""
    from .paint import in_linear
    option = _option_underlayer(W, px, tsv, wet, seed)
    canvas = pickle.loads(_PResponseDLResponseZhBrushAnd[option])
    if option[4] is None:
        sp = canvas.layers['под']
        k = np.asarray(in_linear(np.asarray(tsv, np.float64)), np.float64) / float(in_linear(UNDER_PAINT))
        sp.color[...] = (np.asarray(sp.color, np.float64) * k).astype(np.float32)
    return canvas


def _field_piece(z):
    """Поле сида: {'L', 'Lс' — [проход], 'тон', 'покрытие', 'толщина' — [проход] (по сырой — nan)}."""
    from .eye import lightness
    from .paint import from_linear
    brush, W, px, paint, under, wet, load, pressure, passes, seed = z
    canvas = _take_underlayer(W, px, under, wet, seed)
    window, rows, y0, length = _sizes(W, px)[2:]
    x0, a, x1, b = window
    response = {p: np.full(passes, np.nan) for p in ('L', 'Lс', 'тон', 'покрытие', 'толщина')}
    for p in range(passes):
        _rows(canvas, brush, W, rows, y0, length, paint, load, pressure, seed, p, 'под' if wet else 'верх', px)
        if not wet:
            canvas.dry(1.0, layer='верх')
            sv = canvas.layers['верх']
            coverage = np.asarray(sv.coverage[a:b, x0:x1], np.float64)
            Ls = lightness(np.clip(from_linear(np.asarray(sv.color[a:b, x0:x1], np.float64)) * 255.0, 0.0, 255.0))
            response['покрытие'][p] = float(coverage.mean())
            response['толщина'][p] = float(np.asarray(sv.thickness[a:b, x0:x1], np.float64).mean())
            response['тон'][p] = float((Ls * coverage).sum() / coverage.sum()) if coverage.sum() > 0 else np.nan
        response['L'][p] = canvas.mean(window)['L']
        response['Lс'][p] = canvas.mean(window, light=True)['L']
    return response


def _processes(n):
    from . import cores
    if n < 2 or not cores.allowed():
        return 1
    return max(1, min(PRResponseTsEWithWithResponseIn, cores.number_cores(None), n))


def _at_once(f, zz):
    """ф по списку заданий — в процессах (fork: подложки потомкам видны из памяти), порядок ответов — как у заданий."""
    zz = list(zz)
    n = _processes(len(zz))
    if n > 1:
        import multiprocessing as mp
        try:
            contour = mp.get_context('fork')
        except ValueError:
            contour = None
        if contour is not None:
            with contour.Pool(n) as pool:
                return pool.map(f, zz, chunksize=1)
    return [f(z) for z in zz]


class _Point:
    """Кисть, краска, подложка, нажим, сырая — всё, кроме нагрузки и проходов: счёт полей с памятью."""

    def __init__(self, brush, W, px, paint, under, wet, pressure):
        self.brush, self.W, self.px, self.paint, self.under, self.wet, self.pressure = brush, W, px, paint, under, wet, pressure
        self.option = pickle.dumps((sorted(asdict(brush).items()), W, px, tuple(paint), getattr(paint, 'вид', None),
                                  under, wet, pressure))

    def measurements(self, pairs, passes):
        """{(нагрузка, сиды): замер} для пар (нагрузка, сиды), не короче `проходов` проходов."""
        r = self.measurements_at_once([(n, ss, passes) for n, ss in dict.fromkeys(pairs)])
        return {(n, ss): z for (n, ss, _), z in r.items()}

    def _ready(self, n, ss, passes):
        for p in range(passes, PASSES_MAX + 1):          # счёт с большим числом проходов годится: проходы — подряд
            z = _ZAMERY.get((self.option, n, ss, p))
            if z is not None:
                return z
        return None

    def measurements_at_once(self, triples):
        """{(нагрузка, сиды, проходов): {'L', 'Lс', 'тон', 'покрытие', 'толщина' — [сид][проход]}}; чего нет в памяти —
        считается одним заходом в процессы."""
        triples = list(dict.fromkeys(triples))
        no = [t for t in triples if self._ready(*t) is None]
        if no:
            seeds = sorted({with_ for _, ss, _ in no for with_ in ss})
            options = {with_: _option_underlayer(self.W, self.px, self.under, self.wet, with_) for with_ in seeds}
            lay = [(self.W, self.px, self.under, self.wet, with_) for with_ in seeds if options[with_] not in _PResponseDLResponseZhBrushAnd]
            for option, b in _at_once(_underlayer_piece, lay):
                _remember(option, b)
            for with_ in seeds:
                _PResponseDLResponseZhBrushAnd.move_to_end(options[with_])
            _tight(set(options.values()))
            zz = [(self.brush, self.W, self.px, self.paint, self.under, self.wet, n, self.pressure, p, with_)
                  for n, ss, p in no for with_ in ss]
            rr = iter(_at_once(_field_piece, zz))
            for n, ss, p in no:
                kk = [next(rr) for _ in ss]
                _ZAMERY[(self.option, n, ss, p)] = {q: np.array([brush[q] for brush in kk]) for q in kk[0]}
            while len(_ZAMERY) > _ZAMERResponseIn:
                _ZAMERY.popitem(last=False)
        return {t: self._ready(*t) for t in triples}


# ----------------------------------------------------------------- вызов
def _L(color, who):
    """(L, sRGB) цвета (sRGB 0–255, `Краска`, `цвет(L, C, h)`) или числа L (серый)."""
    from .eye import lightness
    if isinstance(color, (int, float, np.floating)) and not isinstance(color, bool):
        if not 0.0 <= float(color) <= 255.0:
            raise ValueError(f'предел_светлоты: {who} L {float(color):g} — светлота 0–255')
        ts = np.asarray(_gray(color), np.float64)
    else:
        try:
            ts = np.asarray(tuple(color), np.float64)
        except (TypeError, ValueError):
            ts = np.zeros(0)
        if ts.shape != (3,) or not np.all(np.isfinite(ts)) or ts.min() < 0.0 or ts.max() > 255.0:
            raise ValueError(f'предел_светлоты: {who}={color!r} — нужен sRGB (r, g, b) 0–255 или светлота L числом')
    return float(lightness(ts)), tuple(float(v) for v in ts)


def _sign(Lk, Lp):
    """+1 — краска светлее подложки (предел сверху), −1 — темнее (предел снизу)."""
    return 1.0 if Lk >= Lp else -1.0


def limit_lightness(brush, color, underlayer, load=None, pressure=0.6, passes=1, wet=False, L=None, light=False,
                    px_on_mm=None, limit=False):
    """З-2 (В-6): что даст поле плотных рядов краской `цвет` этой кистью по подложке — до кладки, живым стендом.

    кисть — любая, как есть (копией): ширина в мм (`ширина_мм`) или px при `px_на_мм` (без него 8). цвет — sRGB
    (r, g, b), `Краска` (с видом) или светлота L числом; подложка — sRGB или L. нагрузка — нагрузка ходов (число
    больше 0 и до 1 или имя; без неё — нагрузка кисти); нажим — рука (`жест`, 0,05–1); проходов — 1–8: по сухой каждый
    по высохшему прежнему, по сырой — подряд; сырая — подложка сырая, ход в её слой. свет — L со светом рельефа
    (`х.средний(…, свет=True)`), иначе без.

    Три вызова. Прогноз (без `L` и `предел`) — 16 полей стенда в этой точке. `предел=True` — прогноз и предел: перебор
    нагрузок `НАГРУЗКИ` и проходов 1–3 тремя сидами, найденная точка заново — 16 сидами ответа. `L=` — заказ: нагрузка
    (из `НАГРУЗКИ`) и число проходов, что его дают дешевле всего (меньше проходов, затем меньше нагрузка), ответ — счёт
    этой точки 16 сидами; недостижимое — `ValueError` с ближайшим достижимым.

    Ответ — словарь: 'L' (среднее 16 полей стенда), 'σ_поля' (разброс одного поля по сиду), 'ошибка_прогноза'
    (σ_поля / 4 — на столько не известно само среднее), 'σ' (обе вместе: на сколько одно поле уходит от 'L'), 'сидов',
    'нагрузка', 'проходов', 'краска_L', 'подложка_L', 'по_проходам' (L после 1…`проходов` проходов), 'причина' (на
    сколько уводит от краски: по сухой — 'тон прядей' и 'подложка', по сырой — 'смешение с сырой подложкой'; со
    `свет=True` — ещё 'свет рельефа'), 'тон' (L слоя хода без подложки), 'покрытие', 'толщина' (слоя хода; по сырой —
    None), 'слова'; с `предел=True` — ещё 'предел' (светлее у светлой краски, темнее у тёмной не вышло), 'предел_нагрузка',
    'предел_проходов', 'предел_ошибка'. Одни доводы — один ответ.

    Стенд не считает (`ValueError`): нагрузку 0 и выше 1, нажим вне 0,05–1, проходов вне 1–8, холст поля больше
    `ХОЛСТ_МП` Мп (широкая кисть при большом `px_на_мм`)."""
    from .tool import NAGRAtZBrushAnd as AndMENA
    from .brush import Brush
    from .paint import Paint
    if not isinstance(brush, Brush):
        raise ValueError(f'предел_светлоты: нужна кисть (`Кисть.из_файла(…, ширина=…)`), а не {brush!r}')
    px = float(px_on_mm or PX_ON_MM)
    if not px > 0.0:
        raise ValueError(f'предел_светлоты: px_на_мм {px:g} — больше 0')
    W = float(brush.width_mm) * px if brush.width_mm is not None else float(brush.width)
    Sh, In = _sizes(W, px)[:2]
    if Sh * In > CANVAS_MPX * 1e6:
        raise ValueError(f'предел_светлоты: стенд не считает поле кистью {W / px:g} мм при {px:g} px/мм — холст поля '
                         f'{Sh * In / 1e6:.1f} Мп, потолок {CANVAS_MPX:g}; меньше `px_на_мм` — то же поле в мм')
    pressure = float(pressure)
    if not 0.05 <= pressure <= 1.0:
        raise ValueError(f'предел_светлоты: нажим {pressure:g} — от 0,05 до 1; ближайшее — {min(max(pressure, 0.05), 1.0):g}')
    (Lk, ts_brush), (Lp, ts_p) = _L(color, 'цвет'), _L(underlayer, 'подложка')
    paint = color if isinstance(color, Paint) else Paint(ts_brush)
    wet = bool(wet)
    passes = int(passes)
    if not 1 <= passes <= PASSES_MAX:
        raise ValueError(f'предел_светлоты: проходов {passes} — стенд считает 1–{PASSES_MAX}')
    if L is not None and load is not None:
        raise ValueError('предел_светлоты: с заказом L= нагрузку не давай — она и подбирается')

    def number(n):
        v = float(AndMENA[n]) if isinstance(n, str) and n in AndMENA else n
        if isinstance(v, (str, bool)) or not 0.0 < float(v) <= 1.0:
            raise ValueError(f'предел_светлоты: нагрузка {n!r} — число больше 0 и до 1 или имя ({", ".join(AndMENA)}); '
                             'вытертой кистью (0) поле не кладётся')
        return float(v)

    n0 = None if L is not None else number(brush.load if load is None else load)
    option = 'Lс' if light else 'L'
    zn = _sign(Lk, Lp)
    t = _Point(brush.with_(), W, px, paint, ts_p, wet, pressure)
    variant = [(n, p) for n in NAGRAtZBrushAnd for p in range(1, PRResponseCanvasResponseDResponseIn + 1)]

    def exactly(n, p):
        """Точка сидами ответа: замер [сид][проход] не короче п проходов."""
        return t.measurements([(n, WithAndDY)], p)[(n, WithAndDY)]

    def mean(n, p):
        return float(exactly(n, p)[option][:, p - 1].mean())

    def scan():
        """Нагрузки и проходы сидами перебора (разом — поля своей точки сидами ответа): ({точка: L}, лучшая)."""
        own = [(n0, WithAndDY, passes)] if n0 is not None else []
        measure = t.measurements_at_once([(n, SEEDS_SEARCH, PRResponseCanvasResponseDResponseIn) for n in NAGRAtZBrushAnd] + own)
        value = {(n, p): float(measure[(n, SEEDS_SEARCH, PRResponseCanvasResponseDResponseIn)][option][:, p - 1].mean()) for n, p in variant}
        return value, max(variant, key=lambda in_: (zn * value[in_], -in_[1], -in_[0]))

    def response_for(n, p):
        z = exactly(n, p)
        k = p - 1
        v = z[option][:, k]
        Lb, Ls = float(z['L'][:, k].mean()), float(z['Lс'][:, k].mean())
        σp = float(v.std(ddof=1))
        osh = σp / math.sqrt(len(v))
        if wet:
            Lt = coverage = thick = None
            reason = {'смешение с сырой подложкой': round(Lb - Lk, 2)}
        else:
            Lt, coverage, thick = (float(np.nanmean(z[q][:, k])) for q in ('тон', 'покрытие', 'толщина'))
            reason = {'тон прядей': round(Lt - Lk, 2), 'подложка': round(Lb - Lt, 2)}
        if light:
            reason['свет рельефа'] = round(Ls - Lb, 2)
        return {'L': round(float(v.mean()), 2), 'σ': round(math.hypot(σp, osh), 2), 'σ_поля': round(σp, 2),
                'ошибка_прогноза': round(osh, 2), 'сидов': len(v), 'нагрузка': n, 'проходов': p,
                'краска_L': round(Lk, 2), 'подложка_L': round(Lp, 2), 'тон': None if Lt is None else round(Lt, 2),
                'покрытие': None if coverage is None else round(coverage, 4), 'толщина': None if thick is None else round(thick, 4),
                'по_проходам': [round(float(u), 2) for u in z[option][:, :p].mean(axis=0)], 'причина': reason}

    rt = float(brush.spread_tone)

    def words(d):
        main = max(d['причина'], key=lambda brush: abs(d['причина'][brush]))
        return (f'выйдет L {d["L"]:.1f} ±{d["σ"]:.1f} (нагрузка {d["нагрузка"]:g}, проходов {d["проходов"]}): одно поле '
                f'гуляет на {d["σ_поля"]:.1f}, само среднее {d["сидов"]} полей стенда известно до '
                f'{d["ошибка_прогноза"]:.1f}; ' +
                (f'предел {d["предел"]:.1f} ±{d["предел_ошибка"]:.1f} нагрузкой {d["предел_нагрузка"]:g}, проходов '
                 f'{d["предел_проходов"]}; ' if 'предел' in d else '') +
                f'краска L {Lk:.1f}; больше всего уводит {main} ({d["причина"][main]:+.1f} L)' +
                (f' — разброс_тона {rt:g}: недомешанная краска, пряди светлее белого не бывают'
                 if main == 'тон прядей' and zn > 0 else '') +
                ('' if 'предел' in d else '; где потолок — `предел=True`'))

    def with_limit(d, best):
        """Предел — найденная точка заново сидами ответа; своя точка вызова, если вышла дальше неё."""
        pr = mean(*best)
        if zn * (d['L'] - pr) > 0.0:
            best, pr = (d['нагрузка'], d['проходов']), d['L']
        osh = float(exactly(*best)[option][:, best[1] - 1].std(ddof=1)) / math.sqrt(len(WithAndDY))
        d.update({'предел': round(pr, 2), 'предел_нагрузка': best[0], 'предел_проходов': best[1],
                  'предел_ошибка': round(osh, 2)})
        return d

    if L is None:
        if limit:
            _, best = scan()
            d = with_limit(response_for(n0, passes), best)
        else:
            d = response_for(n0, passes)
        d['слова'] = words(d)
        return d
    target = float(L)
    value, best = scan()
    # дешевле: меньше проходов, затем меньше нагрузка; точка перебора считается заново сидами ответа и должна заказ взять
    usable = sorted((in_ for in_ in variant if zn * (value[in_] - target) >= 0.0), key=lambda in_: (in_[1], in_[0]))
    choice = None
    for in_ in usable[:2]:
        if zn * (mean(*in_) - target) >= 0.0:
            choice = in_
            break
    if choice is None:
        pr_L = mean(*best)
        if zn * (pr_L - target) >= 0.0:
            choice = best
    if choice is None:
        pr_osh = float(exactly(*best)[option][:, best[1] - 1].std(ddof=1)) / math.sqrt(len(WithAndDY))
        advice = ''
        if zn > 0 and rt > 0.0 and pr_L < Lk - 0.5:
            # та же точка краской без недомеса — сидами перебора: сколько съедает тон прядей
            even = _Point(brush.with_(spread_tone=0.0), W, px, paint, ts_p, wet, pressure)
            L0 = float(even.measurements([(best[0], SEEDS_SEARCH)], best[1])[(best[0], SEEDS_SEARCH)][option][:, best[1] - 1].mean())
            if L0 - pr_L > 0.5:
                if target < L0:
                    advice = (f'; ближе к краске — краской ровнее: около `кисть.с(разброс_тона='
                             f'{rt * (L0 - target) / (L0 - pr_L):.2f})` (с разброс_тона {rt:g} тон прядей съедает '
                             f'{L0 - pr_L:.1f} L, с 0 выйдет около {L0:.1f})')
                else:
                    advice = (f'; тон прядей съедает {L0 - pr_L:.1f} L, но и краской без недомеса (`разброс_тона=0`) '
                             f'выйдет около {L0:.1f}')
        where = (f'кисть «{brush.name}» {W / px:g} мм, нажим {pressure:g}, краска L {Lk:.1f}, подложка L {Lp:.1f} '
               f'{"сырая" if wet else "сухая"}{", со светом" if light else ""}')
        raise ValueError(f'предел_светлоты: L {target:g} недостижимо — {where}: {"наибольшее" if zn > 0 else "наименьшее"} '
                         f'{pr_L:.1f} ±{pr_osh:.1f} нагрузкой {best[0]:g}, проходов {best[1]}{advice}')
    d = response_for(*choice)
    if limit:
        with_limit(d, best)
    d['слова'] = words(d)
    return d
