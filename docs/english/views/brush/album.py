# READING VIEW of движок/кисть/альбом.py with names in English. Not runnable; edit the original.
"""Мн-3: альбом образца — самые непохожие места картинки вблизи, ×2, под каждым — числа мерки этого места и его адрес.

Место — клетка снимка `КАДР` px (меньше — если клеток иначе не хватает на `МЕСТ`). Чем клетки отличаются, берётся из готовой
мерки (`_признаки`): края, мазки, тон в пятне, краска. Места — по одному на вид (`_непохожие`), из клеток, что не на границе
двух масс (`_цельно`).

Числа под кропом — мерка тех же px, та же, что у `мерка(картинка, px, область=адрес)` (`_место`), а не кусок общей: большой
снимок мерка разбирает мелкими уровнями по плиткам, и вне плиток мелких мазков в ней нет. Места считаются разом процессами.
С холстом рядом — то же место холста: та же доля рамки, кадр — та же доля длинной стороны картины. Случайных чисел нет;
холст и мерки только читаются. Рецептов и оценок «похоже» нет: картинка, числа, адрес."""
import math
import multiprocessing
import os

import numpy as np

from . import eye
from . import parsing as R

BrushADR = 145                # px снимка: сторона кропа; ×2 — 290 px панели, по четыре в ряд — лист 1200 px
BAND_PAIRS = 0.55        # доля высоты кадра: с холстом рядом — средняя полоса места, две пары в ряд, три ряда
FRAMING_MIN = 48             # px: мельче кадр не режется — пусть мест будет меньше
AtInELAndHENAndE = 2
MEWithT = 8                  # мест образца; с холстом рядом — `ПАР`
PAR = 6
TsELYCanvas = 0.7               # доля клеток, из которых берутся места (`_цельно`)
PRResponseTsEWithWithResponseIn = 8
ShRAndFT, WithTRResponseBrush = 12, 6      # px и строк подписи под кропом


class Album:
    """Ответ `мерка(…).альбом()` и `х.манера(…).альбом()`: `лист` — PNG-массив (В, Ш, 3); `места` — словари 'адрес' (x0, y0, x1, y1)
    в px картинки, 'мерка' — числа места (`_место`: 'мазки разбором', 'края пятен', 'тон в пятне', 'края', 'направления'), с
    холстом — 'холст': {'адрес' в px холста, 'мерка'}; `слова` — то же строкой (print)."""
    __slots__ = ('лист', 'места', 'слова')

    def __repr__(self):
        return self.words


# ---------------------------------------------------------------- места

def _grid(In, Sh, framing, places):
    """Клетки снимка: [(x0, y0, x1, y1)] — кадром `кадр` px через полкадра, от края до края; впритык их меньше
    `мест` вдвое — кадр мельче (не мельче `КАДР_МИН`): места через полкадра впритык не ложатся."""
    brush = 1.0
    while True:
        sh, in_ = min(int(framing * brush), Sh), min(int(framing * brush), In)
        if (Sh // sh) * (In // in_) >= 2 * places or framing * brush * 0.95 < FRAMING_MIN:
            break
        brush *= 0.95
    x, y = ([int(round(v)) for v in np.linspace(0, N - n, 2 * (N // n) - 1)] for N, n in ((Sh, sh), (In, in_)))
    return [(a, b, a + sh, b + in_) for b in y for a in x]


def _features(m, cells):
    """Клетка — строка: (группы по столбцам, числа). Края: доля места и доля жёстких; мазки: сколько крупных, длина к ширине,
    направление (удвоенный угол с весом порядка), порядок; тон в пятне p50 и p90; краска — средняя L, a, b (места одной
    краски и одной фактуры — одно место, ровное небо и ровная стена — разные)."""
    from .manner import RABResponseHAndY
    a = m.arrays
    s = RABResponseHAndY / m.px_on_mm                                      # px снимка -> px счёта
    r = a['разбор']
    response = r['области']
    whole = {at['k'] for at in r['уровни'] if not at['плитки']}        # уровни, разобранные на всём снимке: клетки сравнимы
    take = r['мазки'] & np.isin(response['уровень'], sorted(whole))
    ox, oy = response['x'][take], response['y'][take]
    ksh = np.log(np.maximum(response['длина'][take] / np.maximum(response['ширина'][take], 1e-6), 1.0))
    rows = []
    snapshot = a['снимок']
    stiff, soft, tone, θ, with_, spacing = (a[and_] for and_ in ('жёсткие', 'мягкие', 'тон в пятне', 'направление', 'порядок', 'шаг'))
    kk = tone.shape[0] / stiff.shape[0]

    def cut(b, x0, y0, x1, y1, m_):
        i0, j0 = min(int(y0 / m_), b.shape[0] - 1), min(int(x0 / m_), b.shape[1] - 1)
        return b[i0:max(int(math.ceil(y1 / m_)), i0 + 1), j0:max(int(math.ceil(x1 / m_)), j0 + 1)]

    for x0, y0, x1, y1 in cells:
        r4 = (x0 * s, y0 * s, x1 * s, y1 * s)
        zh, mg = cut(stiff, *r4, kk), cut(soft, *r4, kk)
        t = cut(tone, *r4, 1.0)[::4, ::4]                          # карта размыта на пятно 4 мм: хватает отсчёта на мм
        g, p = np.radians(cut(θ, *r4, spacing)), cut(with_, *r4, spacing)
        here = (ox >= x0) & (ox < x1) & (oy >= y0) & (oy < y1)
        rows.append((zh.mean() + mg.mean(), zh.sum() / max(zh.sum() + mg.sum(), 1),
                     here.sum(), np.median(ksh[here]) if here.any() else 0.0,
                     (p * np.cos(2 * g)).mean(), (p * np.sin(2 * g)).mean(), p.mean(),
                     np.median(t), np.percentile(t, 90),
                     *eye.oklab(snapshot[y0:y1, x0:x1].mean((0, 1))[None, None])[0, 0]))
    return np.array(rows, np.float64), np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3])


def _whole(m, cells):
    """Какие клетки показывают одно место, а не границу двух: краска четвертей клетки разошлась не больше, чем у половины
    клеток картины."""
    sn = m.arrays['снимок']
    r = []
    for x0, y0, x1, y1 in cells:
        xs, ys = (x0 + x1) // 2, (y0 + y1) // 2
        q = np.array([[sn[a:b, c:d].mean((0, 1)) for c, d in ((x0, xs), (xs, x1))] for a, b in ((y0, ys), (ys, y1))])
        r.append(np.sqrt(((eye.oklab(q) - eye.oklab(q).mean((0, 1))) ** 2).sum(-1)).max())
    return np.array(r) <= np.quantile(r, TsELYCanvas)


def _dissimilar(numbers, group, cells, places, whole):
    """Номера клеток — по одной на вид места: столбцы рангами 0–1, группы признаков — с равным весом. Начало — самые непохожие
    (первая дальше всех от середины, следующая — дальше всех от взятых, не кроя их); дальше клетки делятся между ними, и
    место — середина своего вида (клетка, что ближе всех к остальным его клеткам), пока не сойдётся: самая крайняя
    клетка картины — обычно граница двух предметов, а не вид места. Места друг друга не кроют."""
    r = []
    for v in numbers.T:                                             # ранг 0–1, у равных — средний
        _, and_, n = np.unique(np.round(v, 6), return_inverse=True, return_counts=True)
        r.append(((np.cumsum(n) - 0.5 * (n - 1)) / len(v))[and_])
    r = np.stack(r, 1)
    r /= np.sqrt(np.bincount(group)[group])
    x0, y0, x1, y1 = np.array(cells).T
    apart = lambda j: (x0 >= x1[j]) | (x1 <= x0[j]) | (y0 >= y1[j]) | (y1 <= y0[j])
    D = np.sqrt(((r[:, None] - r[None]) ** 2).sum(-1))
    allowed = np.ones(len(r), bool)
    to, taken = np.sqrt(((r - np.median(r, 0)) ** 2).sum(1)), []
    while len(taken) < places and allowed.any():
        integers = allowed & whole                                    # цельных без нахлёста не осталось — любая
        j = int(np.where(integers if integers.any() else allowed, to, -1.0).argmax())
        to = D[j] if not taken else np.minimum(to, D[j])
        taken.append(j)
        allowed &= apart(j)
    for _ in range(50):
        kind = D[:, taken].argmin(1)
        was = list(taken)
        for n in range(len(taken)):
            allowed = np.logical_and.reduce([apart(j) for k, j in enumerate(taken) if k != n]) if len(taken) > 1 else np.ones(len(r), bool)
            own = kind == n
            allowed &= whole
            taken[n] = int(np.where(own & allowed, D[:, own].sum(1), np.inf).argmin()) if (own & allowed).any() else taken[n]
        if taken == was:
            break
    how_much = np.bincount(D[:, taken].argmin(1), minlength=len(taken))
    return [taken[n] for n in np.argsort(-how_much, kind='stable')]


# ---------------------------------------------------------------- числа места

def _place(snapshot, px_on_mm, limit, side, window):
    """Числа куска снимка — те же, что `мерка` этого куска: мазки разбором и их форма, края пятен, тон в пятне, края, направления
    (кусок мельче окна — None)."""
    from . import manner as M
    rgb = M._brush_working(snapshot.astype(np.float64), px_on_mm)
    L = eye._L(eye.oklab(rgb)[..., 0])
    strokes, edges_patches, _ = M._analysis(snapshot, px_on_mm, limit, side)
    return {'мазки разбором': strokes, 'края пятен': edges_patches,
            'тон в пятне': M._tone_in_patch(L)[1], 'края': M._edges_places(L, px_on_mm)[0],
            'направления': M._directions(L, window * M.RABResponseHAndY)[0] if min(rgb.shape[:2]) >= window * M.RABResponseHAndY else None}


def _task(a):
    return _place(*a)


def _at_once(tasks, cores=None):
    """`_место` каждой задачи: процессами (не больше `ПРОЦЕССОВ`, ядер и задач), в процессе-работнике — подряд."""
    n = min(cores or PRResponseTsEWithWithResponseIn, PRResponseTsEWithWithResponseIn, len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1,
            len(tasks))
    if n < 2 or multiprocessing.current_process().daemon or 'fork' not in multiprocessing.get_all_start_methods():
        return [_task(a) for a in tasks]
    with multiprocessing.get_context('fork').Pool(n) as pool:
        return pool.map(_task, tasks, chunksize=1)


def _piece(m, cell):
    x0, y0, x1, y1 = cell
    return (m.arrays['снимок'][y0:y1, x0:x1], m.px_on_mm, m.from_where['предел'], m.from_where['сторона'], m.from_where['окно'])


# ---------------------------------------------------------------- подписи и лист

def _signature(number, address, h, px_on_mm):
    """Строки подписи под кропом."""
    from .manner import _h
    r, brush, t, n = h['мазки разбором'], h['края'], h['тон в пятне'], h['направления']
    tr = lambda v: '–'.join(_h(v[and_]) for and_ in ('p10', 'p50', 'p90'))
    x0, y0, x1, y1 = address
    process = lambda v: f'{_h(100 * v, 0)} %'
    with_ = [f'{number}. x {x0}–{x1}, y {y0}–{y1} · {_h((x1 - x0) / px_on_mm, 0)}×{_h((y1 - y0) / px_on_mm, 0)} мм',
         f'мазков {r["мазков"]}, на дм² {_h(r["на дм²"], 0)}; протяжек {process(r["доли"]["протяжек"])}' if r['мазков'] else 'мазков не видно',
         f'без мазков {_h(100 * r["без мазков"], 0)} %; края {_h(100 * brush["доля"], 0)} %'
         + (f', жёстких {_h(100 * brush["жёстких"], 0)} %' if brush['жёстких'] is not None else '')]
    if r['слово'] is None:
        with_.append(f'ширина {tr(r["ширина"])}, длина {tr(r["длина"])} мм')
    else:
        with_.append('размер не числом: ' + (f'целых мазков меньше {R.FEW_STROKES}' if r['слово'].startswith('целых') else
                                        'мазков мало' if r['слово'].startswith('размер держится') else 'на снимке мельче другой'))
    f = r['форма']
    if f is not None:
        with_.append(f'вход {_h(f["вход"]["длина"]["p50"])}, сход {_h(f["сход"]["длина"]["p50"])} шир., сужен '
                 f'{_h(100 * f["вход"]["сужен"], 0)} и {_h(100 * f["сход"]["сужен"], 0)} %')
    else:
        with_.append('форма не числом: ' + (f'концов видно меньше {R.FEW_STROKES}' if r['слово формы'].startswith('мазков') else
                                       'на снимке мельче другая'))
    with_.append(f'тон в пятне {_h(t["p10"])}–{_h(t["p50"])}–{_h(t["p90"])} L')
    if n is not None and n['главные']:
        with_[-1] += f'; ход {n["главные"][0][0]}° — {_h(100 * n["главные"][0][1], 0)} %'
    return with_


def _line(number, d):
    """Место — строкой отчёта: полные слова мерки."""
    from .manner import _h, _words_analysis

    def one(address, h):
        t, brush, n = h['тон в пятне'], h['края'], h['направления']
        direction = f'; направление {n["главные"][0][0]}°±15 — {_h(100 * n["главные"][0][1], 0)} %' if n and n['главные'] else ''
        return (f'{tuple(address)} px: {_words_analysis(h["мазки разбором"], h["края пятен"])}; тон в пятне {_h(t["p50"])} L; '
                f'края {_h(100 * brush["доля"], 0)} % места{direction}')
    with_ = f'{number}. {one(d["адрес"], d["мерка"])}'
    if 'холст' in d:
        with_ += f'\n   холст {one(d["холст"]["адрес"], d["холст"]["мерка"])}'
    return with_


def _transfer(lines, font, width):
    """Строки, не шире `ширина` px: длинная — переносом по словам."""
    result = []
    for with_ in lines:
        t = ''
        for sl in with_.split(' '):
            p = f'{t} {sl}' if t else sl
            if t and font.getlength(p) > width:
                result.append(t)
                p = sl
            t = p
        result.append(t)
    return result


def _sheet(cells, header, in_row, panel, path):
    """Ячейки [(кроп, подпись)] по `в_ряд` в ряд под шапкой; кроп — в `панель` (ш, в) px, подпись — под ним."""
    from PIL import Image, ImageDraw
    font, field, line = eye._font(ShRAndFT), 8, ShRAndFT + 2
    sh, in_ = panel
    signatures = [_transfer(p, font, sh) for _, p in cells]
    bottom = line * max(WithTRResponseBrush, *(len(p) for p in signatures)) + 4
    W = in_row * (sh + field) + field
    top = _transfer([header], font, W - 2 * field)
    y0 = line * len(top) + 6
    rows = math.ceil(len(cells) / in_row)
    sheet = Image.new('RGB', (W, y0 + rows * (in_ + bottom)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for i, with_ in enumerate(top):
        d.text((field, 2 + i * line), with_, fill=(0, 0, 0), font=font)
    for i, ((crop, _), p) in enumerate(zip(cells, signatures)):
        x, y = field + i % in_row * (sh + field), y0 + i // in_row * (in_ + bottom)
        im = Image.fromarray(crop)
        k = min(sh / im.width, in_ / im.height)
        # целое увеличение — повтором px, без придуманных полутонов между ними
        im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))),
                       Image.NEAREST if abs(k - round(k)) < 1e-9 else Image.LANCZOS)
        sheet.paste(im, (x, y))
        for j, with_ in enumerate(p):
            d.text((x, y + in_ + 1 + j * line), with_, fill=(0, 0, 0), font=font)
    if path is not None:
        sheet.save(os.path.expanduser(str(path)))
    return np.asarray(sheet)


# ---------------------------------------------------------------- альбом

def album(m, canvas=None, path=None, places=None, cores=None):
    """Альбом мерки `м` (`Мерка`): до `мест` самых непохожих мест ×2 с числами и адресом, одним PNG (`путь`); `холст` — `Мерка`
    холста: его те же места рядом (та же доля меренной рамки, кадр — та же доля длинной стороны картины)."""
    from .manner import _h
    places = int(places or (PAR if canvas is not None else MEWithT))
    if places < 1:
        raise ValueError(f'альбом: мест={places} — нужно хотя бы одно')
    snapshot = m.arrays['снимок']
    In, Sh = snapshot.shape[:2]
    cells = _grid(In, Sh, BrushADR, MEWithT)
    taken = [cells[j] for j in _dissimilar(*_features(m, cells), cells, max(places, MEWithT), _whole(m, cells))][:places]
    if canvas is not None:                                         # те же места, что без холста, — их средняя полоса
        taken = [(x0, (y0 + y1 - p) // 2, x1, (y0 + y1 - p) // 2 + p) for x0, y0, x1, y1 in taken
                 for p in [max(int(round((y1 - y0) * BAND_PAIRS)), 1)]]
    tasks = [_piece(m, kl) for kl in taken]
    own = []
    if canvas is not None:
        vkh, shkh = canvas.arrays['снимок'].shape[:2]
        k = canvas.from_where['сторона'] / m.from_where['сторона']         # кадр — та же доля длинной стороны картины
        for x0, y0, x1, y1 in taken:
            sh, in_ = (int(round(min(max(a * k, 16), n))) for a, n in ((x1 - x0, shkh), (y1 - y0, vkh)))
            x = int(round(min(max((x0 + x1) / 2 / Sh * shkh - sh / 2, 0), shkh - sh)))
            y = int(round(min(max((y0 + y1) / 2 / In * vkh - in_ / 2, 0), vkh - in_)))
            own.append((x, y, x + sh, y + in_))
        tasks += [_piece(canvas, kl) for kl in own]
    numbers = _at_once(tasks, cores)

    def where(gauge_, kl):
        ox, oy = gauge_.from_where['угол']
        return (kl[0] + ox, kl[1] + oy, kl[2] + ox, kl[3] + oy)

    a = Album()
    a.places, cells = [], []
    for i, kl in enumerate(taken):
        d = {'адрес': where(m, kl), 'мерка': numbers[i]}
        cells.append((snapshot[kl[1]:kl[3], kl[0]:kl[2]], _signature(i + 1, d['адрес'], d['мерка'], m.px_on_mm)))
        if canvas is not None:
            kl = own[i]
            d['холст'] = {'адрес': where(canvas, kl), 'мерка': numbers[len(taken) + i]}
            cells.append((canvas.arrays['снимок'][kl[1]:kl[3], kl[0]:kl[2]],
                           _signature(f'{i + 1} холст', d['холст']['адрес'], d['холст']['мерка'], canvas.px_on_mm)))
        a.places.append(d)
    w, h = taken[0][2] - taken[0][0], taken[0][3] - taken[0][1]
    header = (f'{m.what} {_h(m.size[0], 0)}×{_h(m.size[1], 0)} мм, {Sh}×{In} px ({_h(m.px_on_mm, 2)} px/мм): мест {len(taken)}, по одному на вид '
             f'(края, мазки, тон в пятне, краска), ×{AtInELAndHENAndE}; числа — мерка самого места, адрес — px картинки')
    r = m['мазки разбором']
    tr = lambda v: '–'.join(_h(v[and_]) for and_ in ('p10', 'p50', 'p90'))
    header += ('. Вся картинка: размер мазка не числом' if r['слово'] is not None else
              f'. Вся картинка: мазок шириной {tr(r["ширина"])}, длиной {tr(r["длина"])} мм (p10–p50–p90)')
    if canvas is not None:
        header += (f'. Правее — то же место холста: та же доля рамки, кадр — та же доля длинной стороны (×{_h(w * AtInELAndHENAndE / (own[0][2] - own[0][0]), 2)}), '
                  f'адрес — px холста')
    a.sheet = _sheet(cells, header, 4, (w * AtInELAndHENAndE, h * AtInELAndHENAndE), path)
    a.words = '\n'.join([header] + [_line(i + 1, d) for i, d in enumerate(a.places)])
    return a
