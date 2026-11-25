# READING VIEW of движок/кисть/свет.py with names in English. Not runnable; edit the original.
"""Свет по рельефу краски: нормаль из наклона карты толщины, свет сверху слева,
матовая часть и слабый блик; Г-2 — переотражение светлой краски, перелив света, что выше белого, к соседям, и
поверхность (матовое, полуматовое, лак) ключом вывода. Всё в линейном свете."""
import numpy as np

from .numbers import _TEN

STRENGTH_BY_DEFAULT = 0.40   # сдержанно: на тонкой краске рельеф едва виден, на пастозной читается
HEIGHT_RELIEF = 2.4       # во сколько пикселей «высоты» идёт единица толщины h
_WithInET = np.array([-0.55, -0.62, 0.56])       # сверху слева (y растёт вниз)
_WithInET = _WithInET / np.linalg.norm(_WithInET)
_PResponseLAtInEBrushTResponseR = _WithInET + np.array([0.0, 0.0, 1.0])
_PResponseLAtInEBrushTResponseR = _PResponseLAtInEBrushTResponseR / np.linalg.norm(_PResponseLAtInEBrushTResponseR)
_BLEWithBrush = 24.0
_HIGHLIGHT_FLAT = float(_PResponseLAtInEBrushTResponseR[2] ** _BLEWithBrush)
# п. 20, падающая тень: луч идёт от точки к свету; в плоскости холста — на 1 px, вверх — на столько px высоты
_BRUSH_LIGHT = (float(_WithInET[0] / np.hypot(_WithInET[0], _WithInET[1])), float(_WithInET[1] / np.hypot(_WithInET[0], _WithInET[1])))
_LIFT_RAY = float(_WithInET[2] / np.hypot(_WithInET[0], _WithInET[1]))
SHADOW_BY_DEFAULT = 0.0    # выключена; рабочая сила, когда её зовут без числа, — `ТЕНЬ_РАБОЧАЯ`
SHADOW_WORKING = _TEN[7]
_WithTAtPENEY = 4              # ступеней сглаживания рельефа для дальних преград: σ, 2σ, 4σ, 8σ
# Г-2, белила под светом. Шершавая грань отдаёт зрителю сверху меньше света, чем плоская: с наклона свет уходит вбок,
# на соседние грани, а у граней к свету матовый множитель упирается в белое. Светлая краска возвращает ушедшее:
# недостача света окна D = 1 − (матовый множитель, сглаженный гауссом σ ПЕРЕЛИВ px) (не меньше 0) приходит назад на
# долю альбедо — цвет c светит c · (м + ПЕРЕОТРАЖЕНИЕ · c · D). ПЕРЕОТРАЖЕНИЕ 1 — белое возвращает всю недостачу
# (свет окна сохраняется, светлее плоского шершавое не станет); тёмная краска (c → 0) почти ничего, тень борозд прежняя.
PEREResponseTRAZhENAndE = 1.0
# Свет выше белого (1 в линейном) экран не покажет; обрезанный, он пропадал, а тёмные грани оставались — белила серели
# (−6,4 L под светом 0,4, тычки в три слоя −12). Избыток каждого канала переливается поровну в соседние пиксели, где
# до белого есть место (гаусс σ ПЕРЕЛИВ px; на холсте 8 px/мм — 1 мм): разница тонов между ними не сжимается, тычки
# отдельные, свет окна сохраняется, белое не выше белого.
PERELAndIn = 8.0
# Г-2, поверхность ключом вывода: (блеск — показатель блика, доля блика на силу света, насыщение — степень цвета
# в линейном (масло и лак убирают рассеяние с поверхности: цвет глубже), дымка — доля рассеяния с сухой поверхности
# (белёсый налёт матового)). None — умолчание: слабый узкий блик без дымки (на глаз сухое, как 'матовое'); журнал и слои ключ не трогает.
PResponseInERCanvasNResponseWithTAnd = {
    None: (24.0, 0.10, 1.0, 0.0),
    'матовое': (6.0, 0.02, 1.0, 0.035),
    'полуматовое': (16.0, 0.40, 1.06, 0.0),
    'лак': (40.0, 1.20, 1.12, 0.0),
}
FIELD_SHADOWS = int(_TEN[6]) + sum(int(np.ceil(3.0 * v)) for v in [_TEN[1]] + [_TEN[1] * 2 ** k * np.sqrt(3.0) for k in range(_WithTAtPENEY - 1)]) + 1   # px: столько соседей нужно точке, чтобы её тень не зависела от обреза


def _smooth(a, σ):
    """Гауссово сглаживание (В, Ш) с повтором края; ядро обрезано на 3σ.
    М-9: суммы копятся кусками по `_СТРОК` строк — кусок с ядром лежит в кэше процессора; порядок сложения у каждого
    пикселя прежний, значения те же бит в бит, а счёт широкой полосы вчетверо быстрее."""
    r = int(np.ceil(3.0 * σ))
    if r < 1:
        return a
    ya = np.exp(-0.5 * (np.arange(-r, r + 1) / σ) ** 2)
    ya = (ya / ya.sum()).astype(np.float32)
    In, Sh = a.shape
    p = np.pad(a, ((0, 0), (r, r)), mode='edge')
    b = np.zeros_like(a)
    for y in range(0, In, _WithTRResponseBrush):
        bb, pp = b[y:y + _WithTRResponseBrush], p[y:y + _WithTRResponseBrush]
        for i in range(2 * r + 1):
            bb += pp[:, i:i + Sh] * ya[i]
    p = np.pad(b, ((r, r), (0, 0)), mode='edge')
    a = np.zeros_like(b)
    for y in range(0, In, _WithTRResponseBrush):
        aa = a[y:y + _WithTRResponseBrush]
        n = aa.shape[0]
        for i in range(2 * r + 1):
            aa += p[y + i:y + i + n] * ya[i]
    return a


_WithTRResponseBrush = 8     # М-9: строк в куске `_сгладить`


def shadow_relief(h, strength):
    """П. 20. Множитель падающей тени по карте толщины h (В, Ш): 1 — точка видит свет, 1 − сила — в полной тени.

    Точка темнеет, когда на луче от неё к свету (`_СВЕТ`) рельеф стоит выше луча; полутень — по тому, на сколько
    преграда выше луча, и шире с расстоянием до преграды. Считается только из h; на плоском — ровно 1.
    За краем массива преград нет; чтобы кусок совпал с тем же местом целого, вызывающий даёт поле
    `ПОЛЕ_ТЕНИ` px (или край холста)."""
    h = np.asarray(h, dtype=np.float32)
    one = np.ones(h.shape, np.float32)
    if not strength or h.size == 0:
        return one
    px_on_thickness, σ, threshold, softness, growth, sideways, limit = _TEN[:7]
    H = _smooth(h * np.float32(px_on_thickness), σ)
    steps = min(int(limit), int(np.ceil(float(H.max() - H.min()) / _LIFT_RAY)))
    if steps < 1:
        return one
    In, Sh = H.shape
    # источник света не точка: чем дальше преграда, тем шире её полутень и вбок — дальние преграды берутся
    # с рельефа, сглаженного сильнее (σ, 2σ, 4σ, …; между ступенями — по прямой)
    steps, fields = [H], []
    while σ * 2 ** (len(steps) - 1) < σ + sideways * steps and len(steps) < _WithTAtPENEY:
        k = len(steps)
        steps.append(_smooth(steps[-1], σ * 2 ** (k - 1) * np.sqrt(3.0)))
    for a in steps:
        fields.append(np.pad(a, steps + 1, mode='constant', constant_values=-1e6))   # за краем преград нет
    above = np.zeros_like(H)                     # наибольшая доля тени по лучу
    for t in range(1, steps + 1):
        ox, oy = t * _BRUSH_LIGHT[0], t * _BRUSH_LIGHT[1]
        ix, iy = int(np.floor(ox)), int(np.floor(oy))
        fx, fy = np.float32(ox - ix), np.float32(oy - iy)
        y0, x0 = steps + 1 + iy, steps + 1 + ix
        ur = min(float(np.log2(1.0 + sideways * t / σ)), len(steps) - 1.0)
        k = min(int(ur), len(steps) - 2) if len(steps) > 1 else 0
        with_ = None
        for p, weight in ((fields[k], 1.0 - (ur - k)),) + (((fields[k + 1], ur - k),) if len(steps) > 1 else ()):
            if weight <= 0.0:
                continue
            v = (p[y0:y0 + In, x0:x0 + Sh] * ((1 - fx) * (1 - fy)) + p[y0:y0 + In, x0 + 1:x0 + 1 + Sh] * (fx * (1 - fy))
                 + p[y0 + 1:y0 + 1 + In, x0:x0 + Sh] * ((1 - fx) * fy) + p[y0 + 1:y0 + 1 + In, x0 + 1:x0 + 1 + Sh] * (fx * fy))
            v *= np.float32(weight)
            with_ = v if with_ is None else with_ + v
        with_ -= H
        with_ -= np.float32(t * _LIFT_RAY + threshold)   # на столько преграда выше луча, px высоты; ниже порога тени не кладёт
        with_ *= np.float32(1.0 / (softness + growth * t))   # полутень шире с расстоянием
        np.clip(with_, 0.0, 1.0, out=with_)
        with_ *= np.float32(1.0 - t / (limit + 1.0))   # дальняя тень слабее, к пределу сходит на нет
        np.maximum(above, with_, out=above)
    above = above * above * (3.0 - 2.0 * above)
    return (1.0 - np.float32(strength) * above).astype(np.float32)


def surface_output(surface):
    """Г-2: (блеск, доля блика, насыщение, дымка) ключа поверхности; неизвестный ключ — ValueError с перечнем."""
    if surface not in PResponseInERCanvasNResponseWithTAnd:
        raise ValueError(f"поверхность={surface!r}: одно из {[brush for brush in PResponseInERCanvasNResponseWithTAnd if brush]} или None (умолчание: слабый узкий блик без дымки)")
    return PResponseInERCanvasNResponseWithTAnd[surface]


def multiplier(h, strength=STRENGTH_BY_DEFAULT, shadow=SHADOW_BY_DEFAULT, without=None, surface=None):
    """По карте толщины h (В, Ш) — матовый множитель (1 на плоском) и добавка блика (0 на плоском).
    тень — сила падающей тени (п. 20): 0 — её нет, всё как было; больше — матовый множитель и блик темнеют там,
    куда рельеф кладёт тень (`свет.тень_рельефа`; куску нужно поле `ПОЛЕ_ТЕНИ` px).
    без — (по x, по y), К9: наклон, которого в h быть не должно (ступень у кромки верхнего слоя над толщей нижних,
    `Холст._без_канавы`), — вычитается из наклона h; None — нечего вычитать.

    поверхность — Г-2, ключ вывода (`ПОВЕРХНОСТИ`): блеск и доля блика; None — умолчание: слабый узкий блик без дымки.

    Края массива считаются односторонней разностью; вызывающий даёт поле в 1 px,
    если режет картинку полосами."""
    h = np.asarray(h, dtype=np.float32)
    gx = np.empty_like(h)
    gy = np.empty_like(h)
    gx[:, 1:-1] = (h[:, 2:] - h[:, :-2]) * 0.5
    gx[:, 0] = h[:, 1] - h[:, 0]
    gx[:, -1] = h[:, -1] - h[:, -2]
    gy[1:-1] = (h[2:] - h[:-2]) * 0.5
    gy[0] = h[1] - h[0]
    gy[-1] = h[-1] - h[-2]
    if without is not None:
        gx -= without[0]
        gy -= without[1]
    gx *= HEIGHT_RELIEF
    gy *= HEIGHT_RELIEF
    norm = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx = -gx * norm
    ny = -gy * norm
    nz = norm
    matte = (nx * _WithInET[0] + ny * _WithInET[1] + nz * _WithInET[2]) / _WithInET[2]
    np.clip(matte, 0.0, 3.0, out=matte)
    matte = 1.0 + strength * (matte - 1.0)
    np.clip(matte, 0.05, None, out=matte)
    nh = nx * _PResponseLAtInEBrushTResponseR[0] + ny * _PResponseLAtInEBrushTResponseR[1] + nz * _PResponseLAtInEBrushTResponseR[2]
    np.clip(nh, 0.0, 1.0, out=nh)
    if surface is None:
        highlight = (nh ** _BLEWithBrush - _HIGHLIGHT_FLAT) * (0.10 * strength)
    else:
        gloss, share = surface_output(surface)[:2]
        highlight = (nh ** gloss - float(_PResponseLAtInEBrushTResponseR[2] ** gloss)) * (share * strength)
    np.clip(highlight, 0.0, None, out=highlight)
    if shadow:
        t = shadow_relief(h, shadow)
        matte = matte * t
        highlight = highlight * t
    return matte.astype(np.float32), highlight.astype(np.float32)


def field_spill():
    """px: радиус гаусса `ПЕРЕЛИВ` (по нынешнему значению). Картинке куска нужно поле в два таких радиуса сверх поля
    света (`вывести` — недостача окна, `перелить` — перелив), чтобы кусок совпал с тем же местом целого."""
    return int(np.ceil(3.0 * PERELAndIn))


def overflow(brush):
    """Г-2. к (В, Ш, 3) линейный свет -> тот же, но не выше 1: избыток над белым каждого канала уходит поровну в
    соседние пиксели этого канала, где до 1 есть место (гаусс σ `ПЕРЕЛИВ` px). Свет в окне сохраняется, пока места
    хватает; разница тонов у непереполненных не меняется. Без избытка — тот же массив; дальше `поле_перелива()` px от
    избытка — бит в бит."""
    if not (brush > 1.0).any():
        return brush
    response = np.minimum(brush, np.float32(1.0))
    for i in range(brush.shape[2]):
        e = brush[..., i] - response[..., i]
        if not e.any():
            continue
        E = _smooth(e, PERELAndIn)
        has = response[..., i] < 1.0
        N = _smooth(has.astype(np.float32), PERELAndIn)
        d = np.zeros_like(E)
        np.divide(E, N, out=d, where=(E > 0.0) & (N > 1e-6) & has)
        np.minimum(response[..., i] + d, np.float32(1.0), out=response[..., i])
    return response


def shortfall(m):
    """Г-2. Недостача света окна по матовому множителю м (В, Ш): 1 − м, сглаженный гауссом σ `ПЕРЕЛИВ` px, не меньше 0
    (меньше 1e-5 — 0: на плоском ровно 0)."""
    D = np.float32(1.0) - _smooth(np.asarray(m, np.float32), PERELAndIn)
    D[D < 1e-5] = 0.0
    return D


def output(color, m, b, D, surface=None):
    """Г-2. Освещённый цвет до перелива: цвет (В, Ш, 3) линейный, м, б — из `множители`, D — `недостача(м)`.
    Переотражение светлой краски (`ПЕРЕОТРАЖЕНИЕ`), насыщение и дымка поверхности, блик."""
    saturation, haze = surface_output(surface)[2:]
    if saturation != 1.0:
        color = np.power(np.clip(color, 0.0, None), np.float32(saturation))
    m3 = m[..., None]
    if PEREResponseTRAZhENAndE and D.any():
        m3 = m3 + color * (np.float32(PEREResponseTRAZhENAndE) * D)[..., None]
    brush = color * m3
    if haze:
        brush = brush * np.float32(1.0 - haze) + (np.float32(haze) * m)[..., None]
    brush = brush + b[..., None]
    return brush.astype(np.float32)


def osvetit(color, h, strength=STRENGTH_BY_DEFAULT, shadow=SHADOW_BY_DEFAULT, surface=None):
    """цвет (В, Ш, 3) линейный, h (В, Ш) -> освещённый цвет, не выше белого (Г-2: переотражение, перелив,
    поверхность). тень — сила падающей тени (п. 20), 0 — нет. Как `Холст.картинка`, но края массива без соседей."""
    m, b = multiplier(h, strength, shadow, surface=surface)
    return overflow(output(np.asarray(color, np.float32), m, b, shortfall(m), surface))


def layer_light(h, strength=STRENGTH_BY_DEFAULT, shadow=SHADOW_BY_DEFAULT, without=None):
    """Серый слой света в гамме sRGB 0..1, нейтраль 0,5 — класть поверх цвета
    в режиме «Жёсткий свет» (Hard Light): ниже 0,5 — умножение (точно наш матовый
    множитель), выше — осветление (точно на среднем тоне, на прочих приближённо).
    тень — сила падающей тени (п. 20), 0 — нет: тень входит в тот же слой. без — как у `множители` (К9)."""
    m, b = multiplier(h, strength, shadow, without)
    full = m + b * 4.0                       # блик на среднем тоне
    gamma = np.power(np.clip(full, 0.0, None), 1.0 / 2.2)
    return np.clip(0.5 * gamma, 0.0, 1.0).astype(np.float32)
