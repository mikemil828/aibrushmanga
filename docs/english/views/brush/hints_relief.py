# READING VIEW of движок/кисть/подсказки_рельеф.py with names in English. Not runnable; edit the original.
"""v4, М-2 (`ТРЕБОВАНИЯ_палитра.md`, РЕШЕНИЕ DEN 18.11 21:09 +04 «да»): подсказки про рельеф — счёт по слоям холста.

Читает толщину и покрытие слоёв и тон холста без света — только на границе, где холст досчитан (`х.сохранить`,
`х.закрыть`), и только слои, в которые с прошлого счёта легли мазки; на мазке нового счёта по пикселям нет. Факты
(спецификация — `проверка/КАРТА_АБ_подсказки.md`, «Б», строки С10–С12):

    С10 «толстый слой»      — слой там, где под ним лежат нижние, по медиане толще их суммы в `ТОЛЩЕ_В` раз и больше;
    С11 «валы по границам»  — гребни рельефа слоя (выше соседей на `ВАЛ_ГРЕБЕНЬ` его средней толщины) в полосе
                              ±`ВАЛ_ПОЛОСА_ММ` у границ тона занимают долю места в `ВАЛ_ВО_СКОЛЬКО` раз больше, чем в теле;
    С12 «пастозно»          — у объекта (без пометок — у слоя) толщина его слоёв от `ПАСТОЗНО_ММ` на `ПАСТОЗНО_ДОЛЯ`
                              площади и больше.

Пороги — замером (сдача `проверка/СДАЧА_v4_М-2+М-4.md`)."""

import numpy as np

THICKER_IN = 4.0            # С10: пастозная по протирке ×4,5–6,2; обычная по протирке ×2,4–2,5; пастозная жидкая ×2,5
THICKER_BOTTOM_MM = 0.01      # С10: нижние тоньше — отношение не считается (почти голый грунт)
THICKER_PLACES = 0.25       # С10: нижние лежат хотя бы под четвертью места слоя
BANK_RIDGE = 0.3        # С11: гребень — выше обоих соседей (на 0,75 и 1,5 мм поперёк) на эту долю средней толщины слоя
BANK_BAND_MM = 2.0      # С11: полоса у границ тона, ± мм
BANK_DROP = 24.0       # С11: граница тона — перепад L на 4 мм (прищур 1 мм клетки, как `края` у `х.виды`)
BANK_VO_HOW_MUCH = 2.0     # С11
BANK_MIN = 0.05           # С11: гребней в полосе — не меньше 5 % её места
BANK_VISIBLE = 0.25         # С11: слой, укрытый слоями выше больше чем на 3/4 своего места, — не считается
IMPASTO_MM = 0.3        # С12: толще любой одной «обычной» (у кисти 80 px p90 0,24 мм)
IMPASTO_SHARE = 0.25     # С12
PX_ON_MM = 4.0           # холст без px_на_мм — считать так


def _cells(a, k):
    from .eye import _cells as brush
    return brush(np.asarray(a, np.float32), k) if k > 1 else np.asarray(a, np.float32)


def _ridges(h, d):
    """Насколько точка выше обоих соседей на расстоянии d (по x или по y — где больше); за краем — сама точка."""
    result = None
    for axis in (0, 1):
        p = np.pad(h, [(d, d) if i == axis else (0, 0) for i in (0, 1)], mode='edge')
        forward = np.take(p, range(2 * d, p.shape[axis]), axis)
        back = np.take(p, range(0, p.shape[axis] - 2 * d), axis)
        a = np.minimum(h - forward, h - back)
        result = a if result is None else np.maximum(result, a)
    return result


def _extend(m, r):
    """Маска, расширенная кругом радиуса r клеток."""
    p = np.pad(m, r)
    result = np.zeros_like(p)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                result |= np.roll(np.roll(p, dy, 0), dx, 1)
    return result[r:-r, r:-r]


def _bounds_tone(canvas, cell, ppm, covers):
    """Клетки (`клетка` px) у границ тона: перепад прищуренного L ≥ `ВАЛ_ПЕРЕПАД`, расширенный на полосу. Граница краски
    с голым грунтом (кромка пятна) — не граница цвета двух красок: в счёт идут только клетки, вокруг которых на 4 клетки
    краска есть везде (`кроет` — наибольшее покрытие по слоям)."""
    from .eye import oklab, _L, _srgb, _blur, _edges
    figure = np.asarray(canvas.image(light=False), np.float32)
    linear = np.stack([_cells(figure[..., i], cell) for i in range(3)], -1)
    L = _blur(_L(oklab(_srgb(linear))[..., 0]).astype(np.float32), 2)
    drop = _edges(L, BANK_DROP)[0]
    bare = covers[:L.shape[0], :L.shape[1]] <= 0.5
    edge = (drop >= BANK_DROP) & ~_extend(bare, 4)
    r = max(1, int(round(BANK_BAND_MM * ppm / cell)))
    return _extend(edge, r) if edge.any() else edge


def count(canvas, layers, objects):
    """Факты рельефа для слоёв `слои` (имена) и объектов `объекты` ({ключ: (слои объекта, клетки — множество номеров
    y·100000 + x клеток `КЛЕТКА`)}). Ответ — [(вид, ключ, факт, инструменты)]."""
    from .eye import MM_ON_H
    from .hints import BrushLETBrushA, _h
    names = list(canvas.layers)
    if not names:
        return []
    ppm = float(canvas.px_on_mm or PX_ON_MM)
    spacing = max(1, int(round(ppm / 4.0)))              # ~0,25 мм — для гребней
    cell = 4 * spacing                                  # ~1 мм
    result = []
    thickness, covers = {}, {}

    def t(and_):
        if and_ not in thickness:
            with_ = canvas.layers[and_]
            thickness[and_] = _cells(with_.thickness[:, :], cell) * MM_ON_H
            covers[and_] = _cells(with_.coverage[:, :], cell)
        return thickness[and_], covers[and_]

    band = None
    for and_ in names:
        if and_ not in layers:
            continue
        h, p = t(and_)
        own = p > 0.5
        if not own.any():
            continue
        # С10 — толстый слой
        i = names.index(and_)
        if i:
            bottom = sum(t(j)[0] for j in names[:i])
            bottom_covers = np.maximum.reduce([t(j)[1] for j in names[:i]])
            shared = own & (bottom_covers > 0.5)
            if shared.sum() >= THICKER_PLACES * own.sum():
                top_m, bottom_m = float(np.median(h[shared])), float(np.median(bottom[shared]))
                if bottom_m >= THICKER_BOTTOM_MM and top_m >= THICKER_IN * bottom_m:
                    result.append(('С10', 'С10:с:' + and_,
                                 f'слой «{and_}» — толщина {_h(top_m, 2)} мм, под ним {_h(bottom_m, 2)} мм, ×{_h(top_m / bottom_m)} '
                                 f'(на {100.0 * shared.sum() / own.sum():.0f} % его места)',
                                 ["х.виды(область).массивы['толщина слоёв']", 'нагрузка_для(кисть, толщина=)', 'х.мазок(…, разбавитель=)']))
        # С11 — валы по границам тона
        H = _cells(canvas.layers[and_].thickness[:, :], spacing) * MM_ON_H
        middle = float(h[own].mean())
        if middle > 1e-6:
            gr = np.maximum(_ridges(H, max(1, int(round(0.75 * ppm / spacing)))), _ridges(H, max(1, int(round(1.5 * ppm / spacing)))))
            bank = _cells(gr, 4)[:h.shape[0], :h.shape[1]] > BANK_RIDGE * middle     # гребень в среднем по клетке ~1 мм
            if band is None:
                band = _bounds_tone(canvas, cell, ppm, np.maximum.reduce([t(j)[1] for j in names]))[:h.shape[0], :h.shape[1]]
            visible = own
            if i + 1 < len(names):                       # гребни под чужой краской сверху — не обводы этого слоя
                visible = own & (np.maximum.reduce([t(j)[1] for j in names[i + 1:]]) <= 0.5)
            in_bands, in_body = visible & band, visible & ~band
            if visible.sum() >= BANK_VISIBLE * own.sum() and in_bands.sum() >= max(16, 0.01 * visible.sum()) and in_body.any():
                N, M = float(bank[in_bands].mean()), float(bank[in_body].mean())
                if N >= BANK_MIN and N >= BANK_VO_HOW_MUCH * M:
                    result.append(('С11', 'С11:с:' + and_,
                                 f'слой «{and_}» — гребни вала на {100 * N:.0f} % места в полосе ±{BANK_BAND_MM:.0f} мм у границ тона, '
                                 f'в теле — на {100 * M:.0f} %',
                                 ['х.взгляд(область, свет=)', "х.виды(область).массивы['края']", 'х.проба']))
    # С12 — пастозно (объект; без пометок — слой)
    large = {}
    for ko, (sl, cells) in objects.items():
            sl = [s for s in sl if s in canvas.layers]
            if not sl or not cells:
                continue
            for s in sl:
                if s not in large:
                    large[s] = _large(canvas, s, BrushLETBrushA)
            h = sum(large[s][0] for s in sl)
            p = np.maximum.reduce([large[s][1] for s in sl])
            c = np.asarray(sorted(cells), np.int64)
            cy, cx = c // 100000 - canvas.y0 // BrushLETBrushA, c % 100000 - canvas.x0 // BrushLETBrushA
            m = (cy >= 0) & (cx >= 0) & (cy < h.shape[0]) & (cx < h.shape[1])
            hh, pp = h[cy[m], cx[m]], p[cy[m], cx[m]]
            hh = hh[pp > 0.5]
            if len(hh) < 4:
                continue
            share = float((hh >= IMPASTO_MM).mean())
            if share >= IMPASTO_SHARE:
                name = f'объект «{ko[2:]}»' if ko.startswith('о:') else f'слой «{ko[2:]}»'
                result.append(('С12', 'С12:' + ko, f'{name} — толще {_h(IMPASTO_MM)} мм на {100 * share:.0f} % его площади '
                             f'(медиана {_h(float(np.median(hh)), 2)} мм, до {_h(float(hh.max()), 2)} мм)',
                             ["х.виды(область).массивы['толщина']", 'нагрузка_для(кисть, толщина=)', 'х.мазок(…, разбавитель=)']))
    return result


def _large(canvas, layer, cell):
    """Толщина (мм) и покрытие слоя средними по клеткам `клетка` px (неполная клетка у края — по своим пикселям)."""
    from .eye import MM_ON_H
    with_ = canvas.layers[layer]
    In, Sh = canvas.height, canvas.width
    ny, nx = -(-In // cell), -(-Sh // cell)
    result = []
    for a in (with_.thickness[:, :], with_.coverage[:, :]):
        b = np.zeros((ny * cell, nx * cell), np.float32)
        b[:In, :Sh] = a
        n = np.zeros_like(b)
        n[:In, :Sh] = 1.0
        s = b.reshape(ny, cell, nx, cell).sum((1, 3))
        result.append(s / np.maximum(n.reshape(ny, cell, nx, cell).sum((1, 3)), 1.0))
    return result[0] * MM_ON_H, result[1]
