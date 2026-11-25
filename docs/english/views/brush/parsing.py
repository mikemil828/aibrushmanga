# READING VIEW of движок/кисть/разбор.py with names in English. Not runnable; edit the original.
"""Мн-1, Мн-1б: разбор картинки живописи на мазки-кандидаты — области одной краски с осью, длиной, шириной и видом.

Наследник `архив/анализатор/разбор.py` (шаг 1), переписанный под образец любого масштаба: чисел под один образец нет.
Миллиметров разбор не знает: одна картинка, заданная как 655 и как 250 мм, даёт те же области. Копирования мазков и
подбора здесь нет — только меры.

Один проход (`_проход`) — разбор картинки на области одной краски; все его размеры — в долях `КИСТЬ` рабочих px:

1. Бассейны: водораздел силы перепада цвета от местных минимумов — мелкие куски, границы на перепадах краски.
2. Слияние: два соседа сливаются, пока разница красок по обе стороны стыка, средняя по всей его длине,
   меньше порога места: большее из `ТОН_МИН` (сколько глаз видит и чего не даёт сжатие), `ТОН_ДОЛЯ` местного размаха и
   `ЗЕРНО` σ зерна снимка. Узкое (меньше `УЗКОЕ` кисти) и мелкое, если оно близко к соседу, доливается к нему.
3. Вид: ход — по длинной оси области; сечения поперёк неё; `полоса` — сечения сплошные и ширина держится.

Своего масштаба у прохода нет: мазок у него всегда 4–6 рабочих px, на любой картинке и в любом уменьшении (Мн-1б,
`проверка/ЗАПИСКА_Мн-1б_середина.md`). Поэтому мазки берёт `разобрать` — по лестнице уменьшений: с каждого уровня
только полосы заметно шире этого пола. Порядок слоёв по картинке не восстановить; мазки внахлёст одной краски —
одна область; пряди растрёпанной кисти, разделённые грунтом, собирает `_пряди`."""
import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from . import fast

BrushAndWithT = 6.0               # px рабочей картинки на ширину кисти
LAB = np.array([100.0, 400.0, 400.0])   # OKLab -> единицы разбора: L 0–100 и оттенок с тем же весом, что в архиве
WithGLADAndT = 0.8            # px: гасит зерно и блоки сжатия
TONE_MIN = 5.0             # единицы разбора: ступень краски, видная глазу; шум и блоки сжатия — 1–3
TONE_SHARE = 0.6            # доля местного размаха краски (σ в окне `ОКНО_РАЗМАХА` кистей): меньше — борозды, не край
WINDOW_RANGE = 3.0
ZERNResponse, STEP_GRAIN = 2.0, 8.0   # порог — не меньше стольких σ зерна снимка (замер — ровное с зерном: мазков нет) · ступень L
AtZBrushResponseE = 0.35              # кисти: уже — борозда или кромка, доливается к соседу
MELBrushResponseE = 0.6              # площади кисти²: мельче — клочок
TResponseNBrushResponseE = 0.65             # кисти: уже и с краской между двумя соседями — переход (размытая кромка), не мазок
PResponseLE = 4.0                # кистей: шире — поле, а не мазок
PRResponseTYaZhBrushA = 2.5            # длина к ширине
PLResponseTNResponse = 0.62             # доля заполнения сечений: меньше — не одна полоса
WithGAndB = 2.0                # изогнутая полоса: длина по площади не больше стольких хорд — иначе сложена вдвое или сеть
RAZBRResponseWith = 3.2             # p90/p10 ширины сечений: больше — не одна полоса
WithTAtPENAnd = tuple(round(0.05 * i, 2) for i in range(1, 21))   # доли порога: слияние идёт от ближних красок к дальним
LIMIT_PX = 3.0           # px снимка: признак уже не различается (пиксель, резкость съёмки, сжатие)
PResponseLResponseWithA = (8.0, 13.0)      # рабочих px: с уровня лестницы берутся полосы такой ширины (1,3–2,2 `КИСТЬ`) — выше пола разбора
                          # (4–6 px); соседние уровни (шаг √2) перекрываются
WINDOW_LADDERS = 5.0       # кистей: окно местного размаха на уровне лестницы — шире полосы, чтобы клочки внутри мазка слились
DYRY = 0.5                # чужого внутри полосы больше такой доли её площади — сеть, а не полоса
InNAtTRAnd = 0.5              # доля площади полосы на уже принятом мазке: больше — она его часть
DELAndT = 0.5               # части одного уровня закрывают такую долю принятого — оно было группой мазков, а не мазком
TO_ZHE = 0.5               # пересечение к объединению: область следующего уровня — то же целое, что принято уровнем грубее
PRResponseWithInET = 1.0 / 3.0       # доля ширины целого: часть у́же и вдоль него — борозда или просвет грунта, а не мазок поверх
InDResponseL = 20.0              # градусов: направление части — вдоль целого
CanvasResponseD = 4.0 / 3.0           # ширин обычного мазка картинки: пряди вместе не шире — один ход; шире — соседние мазки (штриховка)
BYuDZhET = 350000           # рабочих px на уровень: больше (и больше самого снимка) — уровень считается по плиткам
SIDE_MIN = 60.0        # рабочих px: короткая сторона самого грубого уровня
FEW_STROKES = 12          # целых мазков меньше — размеры не называются
FABrushTAtRA = 0.25            # L 0–100: медианное отклонение светлоты ровнее — не мазок


def in_units(rgb):
    return fast.oklab(rgb) * LAB


def _gauss(a, σ):
    return ndimage.gaussian_filter(a, (σ, σ, 0) if a.ndim == 3 else σ, mode='reflect') if σ > 0 else a


def _strength(lab, σ):
    E = 0.0
    for k in range(3):
        E = E + ndimage.gaussian_filter(lab[k], σ, order=(1, 0)) ** 2 + ndimage.gaussian_filter(lab[k], σ, order=(0, 1)) ** 2
    return np.sqrt(E)


def _grain(lab0, few=-1.0):
    """Зерно снимка в каждом пикселе, единицы разбора: σ остатка лапласиана (Иммеркер) в окнах 9 px, по ступеням светлоты
    (у OKLab зерно тёмного крупнее) — 20-й процентиль окон ступени: самые ровные места, фактура краски в него не входит.
    Везде не больше `мало` — 0.0: порогу места оно безразлично."""
    core = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64) / 6.0
    e = 0.0
    for k in range(3):
        e = e + ndimage.convolve(lab0[k], core, mode='reflect') ** 2
    σ = np.sqrt(np.maximum(ndimage.uniform_filter(e, 9, mode='reflect'), 0.0))
    st = np.clip(np.floor(lab0[0] / STEP_GRAIN), 0, 11).astype(np.int64)
    # 20-й процентиль ступени не больше `мало`, если таких окон хватает на оба числа, между которыми он стоит
    lasts = lambda n, k: k >= (0.2 * (n - 1)).astype(np.int64) + 2
    n_st = np.bincount(st.ravel(), minlength=12)
    brush_st = np.bincount(st.ravel()[(σ <= few).ravel()], minlength=12)
    if (lasts(n_st, brush_st) | (n_st < 400)).all() and lasts(np.array([σ.size]), np.array([brush_st.sum()]))[0]:
        return 0.0
    shared = float(np.percentile(σ, 20))
    flat, by = σ.ravel(), np.argsort(st.ravel().astype(np.uint8), kind='stable')
    shared, i0 = np.full(12, shared), 0
    for j, n in enumerate(n_st):
        if n >= 400:
            shared[j] = np.percentile(flat[by[i0:i0 + n]], 20)
        i0 += n
    map = shared[st]
    return _gauss(map, 3.0)


def _range(lab, σ, q=3):
    """Местный размах краски (σ в окне `σ` px). Окно в разы шире пикселя, поэтому размытия считаются на клетках q×q px и
    возвращаются к пикселям по прямой: то же с точностью до сотых долей, в q² раз дешевле."""
    from PIL import Image
    In, Sh = lab.shape[1:]
    p = np.pad(lab, ((0, 0), (0, -In % q), (0, -Sh % q)), mode='edge')
    h, w = p.shape[1] // q, p.shape[2] // q
    cells = lambda a: a.reshape(h, q, w, q).mean((1, 3))
    back = lambda a: np.asarray(Image.fromarray(a.astype(np.float32)).resize((w * q, h * q), Image.BILINEAR), np.float64)
    d2 = 0.0
    for j in range(len(p)):
        d2 = d2 + (p[j] - back(_gauss(cells(p[j]), σ / q))) ** 2
    return np.sqrt(np.maximum(back(_gauss(cells(d2), σ / q)), 0.0))[:In, :Sh]


def _in_a_row(m, N):
    """Номера 0…N-1 с пропусками — подряд, порядок тот же."""
    has = np.zeros(N, np.int64)
    has[m] = 1
    return (np.cumsum(has) - 1)[m]


def _basins(E):
    """Водораздел силы перепада: каждый пиксель стекает к нижнему из 8 соседей (при равенстве — к первому) до местного
    минимума; кусок — все пиксели одного минимума, граница — на гребне перепада."""
    In, Sh = E.shape
    P = np.pad(E, 1, constant_values=np.inf)
    better, neighbor = E.copy(), np.zeros((In, Sh), np.uint8)
    WithResponseWithEDAnd = ((-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for j, (dy, dx) in enumerate(WithResponseWithEDAnd):
        v = P[1 + dy:1 + dy + In, 1 + dx:1 + dx + Sh]
        np.copyto(neighbor, j + 1, where=v < better)
        np.minimum(better, v, out=better)
    neighbor = neighbor.ravel()
    where_to = np.arange(In * Sh) + np.array([0] + [dy * Sh + dx for dy, dx in WithResponseWithEDAnd])[neighbor]
    while True:
        further = where_to[where_to]
        if (further == where_to).all():
            break
        where_to = further
    return _in_a_row(where_to, In * Sh).reshape(In, Sh)


def _pairs(labels):
    """Соседние по стороне пиксели разных кусков: (номер a, номер b) и плоские адреса обоих."""
    In, Sh = labels.shape
    p, flat = [], labels.ravel()
    for r, spacing in ((labels[:, :-1] != labels[:, 1:], 1), (labels[:-1] != labels[1:], Sh)):
        i = np.flatnonzero(r)
        if spacing == 1:                                              # строка короче на px: номер px — с учётом строк
            i = i + i // (Sh - 1)
        p.append((flat[i], flat[i + spacing], i, i + spacing))
    return tuple(np.concatenate(x) for x in zip(*p))


def _sum(option, n, v):
    return np.stack([np.bincount(option, v[:, k], n) for k in range(v.shape[1])], 1)


def _merge(labels, lab, threshold_map, weight):
    """Слияние кругами, от ближних красок к дальним (`СТУПЕНИ` порога): каждый кусок сливается с соседом наименьшей
    разницы красок у стыка, пока она меньше порога. Стык копит суммы красок обеих сторон по своим пикселям (краска
    стороны — средняя её первичного куска), поэтому разница — по всей длине стыка. Ответ — метки 0…N-1."""
    N = int(labels.max()) + 1
    flat = labels.ravel()
    w = np.maximum(np.bincount(flat, weight.ravel(), N), 1e-9)
    paint = np.stack([np.bincount(flat, (lab[k] * weight).ravel(), N) for k in range(3)], 1) / w[:, None]
    a, b, ia, ib = _pairs(labels)
    threshold_p = 0.5 * (threshold_map.ravel()[ia] + threshold_map.ravel()[ib])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    d = paint[lo] - paint[hi]
    option, sample = fast.unique(lo * np.int64(N) + hi)
    n = np.bincount(sample).astype(np.float64)
    with_ = np.concatenate([_sum(sample, len(option), d), np.bincount(sample, threshold_p)[:, None], n[:, None]], 1)
    lo, hi = option // N, option % N
    new, first = np.arange(N), np.zeros(N, np.int64)             # глава куска; его лучший стык
    best, touched = np.zeros(N), np.zeros(N, bool)
    step = 0
    # кусок зовётся наименьшим номером своих первичных: порядок номеров и ключей — как у плотных. Пересчитываются только
    # стыки слитых кусков; сумма стыков — в порядке прежних ключей
    relative = np.sqrt((with_[:, :3] ** 2).sum(1)) / np.maximum(with_[:, 3], 1e-12)          # разница в долях порога
    while len(lo) and step < len(WithTAtPENAnd):
        e = np.flatnonzero(relative < WithTAtPENAnd[step])
        if not len(e):
            step += 1
            continue
        # лучший сосед куска: наименьшая разница; при равенстве — первый по номеру стыка. Цепочки лучших — одним куском
        e = e[np.argsort(option[e])]
        st = np.concatenate([e, e])
        who = np.concatenate([hi[e], lo[e]])                      # стыки, где кусок второй, — ключом раньше
        response = relative[st]
        best[who] = np.inf
        np.minimum.at(best, who, response)
        da = np.flatnonzero(response == best[who])[::-1]
        first[who[da]] = st[da]                                  # в обратном порядке: остаётся первый
        st = first[who]
        a, b = lo[st], hi[st]
        while True:                                               # глава цепочки — её наименьший номер
            ra, rb = new[a], new[b]
            l2 = ra < rb
            l3 = rb < ra
            if not (l2.any() or l3.any()):
                break
            np.minimum.at(new, rb[l2], ra[l2])
            np.minimum.at(new, ra[l3], rb[l3])
            canvas = np.concatenate([a, b])
            while True:
                g = new[new[canvas]]
                if (g == new[canvas]).all():
                    break
                new[canvas] = g
        nodes = np.concatenate([a, b])
        touched[nodes] = True
        i = np.flatnonzero(touched[lo] | touched[hi])
        touched[nodes] = False
        i = i[np.argsort(option[i])]
        l2, h2 = new[lo[i]], new[hi[i]]
        live = l2 != h2
        l2, h2, sp = l2[live], h2[live], with_[i[live]]
        per = l2 > h2
        sp[per, :3] *= -1.0
        l2, h2 = np.minimum(l2, h2), np.maximum(l2, h2)
        kl, sample = fast.unique(l2 * np.int64(N) + h2)
        sp = _sum(sample, len(kl), sp)
        j, extra = i[:len(kl)], i[len(kl):]                      # места новых стыков — места прежних
        option[j], lo[j], hi[j], with_[j] = kl, kl // N, kl % N, sp
        relative[j] = np.sqrt((sp[:, :3] ** 2).sum(1)) / np.maximum(sp[:, 3], 1e-12)
        relative[extra] = np.inf
        lo[extra] = hi[extra] = 0                               # пустое место: кусок 0 с собой
        zh = lo != hi
        if 4 * np.count_nonzero(zh) < 3 * len(lo):
            option, lo, hi, with_, relative = option[zh], lo[zh], hi[zh], with_[zh], relative[zh]
    while True:
        further = new[new]
        if (further == new).all():
            break
        new = further
    return _in_a_row(new[flat], N).reshape(labels.shape)


def _rim(labels):
    edge = np.zeros(labels.shape, bool)
    r = labels[:, 1:] != labels[:, :-1]
    edge[:, 1:] |= r
    edge[:, :-1] |= r
    r = labels[1:] != labels[:-1]
    edge[1:] |= r
    edge[:-1] |= r
    return edge


def _thickness(labels, N, to=None):
    """Ширина области по расстоянию до её края: у полосы ширины w среднее расстояние ≈ (w + 2) / 4, px. `до` — нужна точно
    только у областей у́же стольких px: у остальных — не меньше `до`."""
    edge, m = _rim(labels), labels.ravel()
    n = np.maximum(np.bincount(m, minlength=N), 1)
    dt = (fast.distances(edge) if to is None else fast.close(edge)) + 1.0
    thick = 4.0 * np.bincount(m, dt.ravel(), N) / n - 2.0
    if to is not None and (thick[m[dt.ravel() == 5.0]] < to).any():   # узкая по усечённому расстоянию, а есть и дальние px
        return _thickness(labels, N)
    return thick, n


def _between(a, b, paint, threshold, N):
    """Какие куски a — переход: краска лежит на отрезке между красками двух его соседей b (в сторону — меньше порога
    места), а соседи разные. Так выходит размытая кромка двух масс снимка и тонкий край мазка."""
    da = np.zeros(N, bool)
    if not len(a):
        return da
    p = fast.by_key(a * np.int64(N) + b)
    a, b = a[p], b[p]
    start = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    how_much = np.diff(np.r_[start, len(a)])
    j = np.repeat(np.arange(len(a)), np.repeat(how_much, how_much))   # пары соседей одного куска: j и q
    pr = how_much * how_much
    q = np.arange(len(j)) - np.repeat(np.r_[0, np.cumsum(pr)[:-1]], pr)
    q = np.repeat(start, pr) + q % np.repeat(how_much, pr)
    i, kj = a[j], paint[b[j]]
    d = (paint[i] - kj).T                                       # от соседа j к куску
    g = (paint[b[q]] - kj).T                                    # от соседа j к соседу q
    sk = lambda u, v: u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    gg = sk(g, g)
    t = sk(d, g) / np.maximum(gg, 1e-9)
    r = d - t * g
    past = np.sqrt(sk(r, r))
    da[i[(t > 0.05) & (t < 0.95) & (past < threshold[i]) & (gg > threshold[i] ** 2)]] = True
    return da


def _top_up(labels, lab, threshold_map, brush):
    """Узкое и мелкое — к соседу, ближнему по краске среди тех, с кем у него стык не короче 0,3 самого длинного; узкое,
    контрастное к соседу на 2 порога и больше, — свой мазок (белый удар, тонкая черта)."""
    joints = None                                                  # длины стыков и суммы красок меняются только у слитых
    channel = [c.ravel() for c in lab] + [threshold_map.ravel()]
    for _ in range(6):
        N = int(labels.max()) + 1
        thick, n = _thickness(labels, N, max(TResponseNBrushResponseE * brush, AtZBrushResponseE * brush, 4.0))
        fine = n < max(MELBrushResponseE * brush * brush, 12.0)
        thin = thick < max(TResponseNBrushResponseE * brush, 4.0)
        narrow = (thick < max(AtZBrushResponseE * brush, 2.0)) | fine | thin
        if not narrow.any():
            break
        flat = labels.ravel()
        if joints is None:
            sums = [np.bincount(flat, v, N) for v in channel]
            a, b, _, _ = _pairs(labels)
            joints = fast.unique_count(np.minimum(a, b) * np.int64(N) + np.maximum(a, b))
            a, b, dl = joints[0] // N, joints[0] % N, joints[1]
        paint = np.stack(sums[:3], 1) / n[:, None]
        threshold = sums[3] / n
        a, b, dl = np.concatenate([a, b]), np.concatenate([b, a]), np.concatenate([dl, dl])
        p = fast.by_key(a * np.int64(N) + b)
        all = a, b, dl = a[p], b[p], dl[p]
        long = np.zeros(N)
        np.maximum.at(long, a, dl)
        usable = narrow[a] & (dl >= 0.3 * long[a])
        a, b = a[usable], b[usable]
        decides = (thin & ~fine)[a]
        between = _between(a[decides], b[decides], paint, threshold, N)
        dE = np.sqrt(((paint[a] - paint[b]) ** 2).sum(1))
        p = np.lexsort((b, dE, a))
        a, b, dE = a[p], b[p], dE[p]
        first = np.r_[True, a[1:] != a[:-1]] if len(a) else np.zeros(0, bool)
        a, b, dE = a[first], b[first], dE[first]
        pour = fine[a] | ((thick[a] < max(AtZBrushResponseE * brush, 2.0)) & (dE < 2.0 * threshold[a])) | between[a]
        a, b = a[pour], b[pour]
        # в круг — только к соседу, который сам не льётся (или льётся позже): цепочек и петель нет
        busy = np.zeros(N, bool)
        new = np.arange(N)
        for i in np.argsort(n[a], kind='stable'):
            x, y = a[i], b[i]
            if not busy[x] and new[y] == y:
                new[x] = y
                busy[y] = True
        if (new == np.arange(N)).all():
            break
        where_to = _in_a_row(new, N)                                    # нов[нов] == нов: номера живых — подряд
        merged = np.zeros(int(where_to.max()) + 1, bool)
        merged[where_to[new != np.arange(N)]] = True
        labels = where_to[labels]
        flat = labels.ravel()
        i = np.flatnonzero(merged[flat])
        previous, sums = sums, []
        for v, was in zip(channel, previous):
            with_ = np.bincount(flat[i], v[i], len(merged))
            with_[where_to] = np.where(merged[where_to], with_[where_to], was)
            sums.append(with_)
        a, b, dl = (where_to[all[0]], where_to[all[1]], all[2])
        da = a < b
        option, sample = fast.unique(a[da] * np.int64(len(merged)) + b[da])
        a, b, dl = option // len(merged), option % len(merged), np.bincount(sample, dl[da], len(option)).astype(np.int64)
    return labels


def _even(labels, L, whom):
    """У каких областей из `кого` медианное отклонение светлоты меньше `ФАКТУРА`."""
    number = np.cumsum(whom) - 1
    i = np.flatnonzero(whom[labels])
    response = np.zeros(len(whom), bool)
    if len(i):
        m, v = number[labels[i]], L[i]
        n = int(number[-1]) + 1
        response[whom] = fast.medians(np.abs(v - fast.medians(v, m, n)[m]), m, n) < FABrushTAtRA
    return response


def _region(labels, L_wet, brush, bands=None):
    """Свойства областей массивом: длина, ширина, ось, вид — по моментам и сечениям поперёк длинной оси. `полосы` —
    ширины (от, до) px: у областей заведомо у́же — только 'у края' (лестница их не берёт), 'ровное' ищется среди полос такой
    ширины. Ответ — (области, (номера областей, px) посчитанного)."""
    In, Sh = labels.shape
    total = N = int(labels.max()) + 1
    m = labels.ravel()
    here, take = slice(None), None
    if bands is None:
        thick, _ = _thickness(labels, N)
    else:                                                         # ширина — не больше 1,35 толщины или она сама
        edge = _rim(labels)
        dt = fast.close(edge).ravel() + 1.0                     # дальше 4 px — «не меньше»
        thick = 4.0 * np.bincount(m, dt, N) / np.maximum(np.bincount(m, minlength=N), 1) - 2.0
        far = np.zeros(N, bool)
        far[m[dt == 5.0]] = True
        take = np.flatnonzero((1.35 * thick >= 0.999 * bands[0]) | far)
        number = np.full(N, -1)
        number[take] = np.arange(len(take))
        m = number[m]
        here = np.flatnonzero(m >= 0)
        m, thick, N = m[here], thick[take], len(take)
    n = np.bincount(m, minlength=N).astype(np.float64)
    ys, xs = np.divmod(np.arange(In * Sh)[here], Sh)
    xs, ys = xs.astype(np.float64), ys.astype(np.float64)
    cx, cy = np.bincount(m, xs, N) / n, np.bincount(m, ys, N) / n
    dx, dy = xs - cx[m], ys - cy[m]
    xx, xy, yy = (np.bincount(m, v, N) / n for v in (dx * dx, dx * dy, dy * dy))
    angle = 0.5 * np.arctan2(2 * xy, xx - yy)
    c, s = np.cos(angle), np.sin(angle)
    along = dx * c[m] + dy * s[m]
    across = -dx * s[m] + dy * c[m]
    # сечения шагом 2 px вдоль оси
    number = np.floor(along / 2.0).astype(np.int64)
    number0 = np.full(N, np.iinfo(np.int64).max)
    np.minimum.at(number0, m, number)
    number1 = np.full(N, np.iinfo(np.int64).min)
    np.maximum.at(number1, m, number)
    number = number1 - number0 + 1
    start = np.r_[0, np.cumsum(number)]
    k = start[m] + (number - number0[m])
    K = int(start[-1])
    in_section = np.bincount(k, minlength=K)
    lo, hi = np.full(K, np.inf), np.full(K, -np.inf)
    np.minimum.at(lo, k, across)
    np.maximum.at(hi, k, across)
    whose = np.repeat(np.arange(N), number)
    has = in_section >= 2
    width_with = np.where(has, hi - lo + 1.0, 0.0)
    dense = np.where(has, in_section / 2.0 / np.maximum(width_with, 1.0), 0.0)
    width, spread, filled = np.ones(N), np.ones(N), np.ones(N)
    threshold = np.lexsort((width_with, whose))
    threshold = threshold[has[threshold]]
    chn = np.bincount(whose[threshold], minlength=N)
    nn = np.r_[0, np.cumsum(chn)]
    for j, name in ((0.5, 0), (0.9, 1), (0.1, 2)):
        i = nn[:-1] + np.minimum((chn * j).astype(np.int64), np.maximum(chn - 1, 0))
        v = np.where(chn > 0, width_with[threshold][np.minimum(i, max(len(threshold) - 1, 0))], 1.0)
        if name == 0:
            width = v
        elif name == 1:
            p90 = v
        else:
            p10 = v
    spread = p90 / np.maximum(p10, 1.0)
    filled = np.bincount(whose[has], dense[has], N) / np.maximum(np.bincount(whose[has], minlength=N), 1)
    length = number * 2.0 - 1.0
    if bands is not None:                                        # толщина точно — у кого она решает
        exactly = far[take] & ~((width > bands[1]) & (thick > bands[1] + 0.001))
        if exactly.any():
            whose = np.zeros(total, bool)
            whose[take[exactly]] = True
            dt = fast.distances(edge | ~whose[labels]) + 1.0
            thick = np.where(exactly, 4.0 * np.bincount(m, dt.ravel()[here], N) / np.maximum(n, 1) - 2.0, thick)
    # изогнутая полоса: сечения прямой оси берут прогиб за ширину — ширина по расстоянию до края, длина по площади
    # дуга пересекает каждое сечение один раз: её длина по площади — не больше `СГИБ` хорд (полуокружность — 1,6; шпилька
    # и сеть — от двух: сечение идёт через них дважды и больше)
    by_areas = n / np.maximum(thick, 1.0)
    arc = (filled < 0.8) & (by_areas > 1.15 * length) & (by_areas <= WithGAndB * length)
    width = np.where(arc, thick, np.minimum(width, np.maximum(1.35 * thick, 1.0)))
    length = np.where(arc, n / np.maximum(thick, 1.0), length)
    relative = length / np.maximum(width, 1.0)
    # одна полоса: сечения сплошные и ширина держится; у дуги — площадь ≈ длина × толщина по построению
    band = np.where(arc, thick > 0.5 * (n / np.maximum(length, 1)), (filled >= PLResponseTNResponse) & (spread <= RAZBRResponseWith))
    whom = np.ones(N, bool) if bands is None else band & (width >= bands[0]) & (width < bands[1])
    even = _even(m, L_wet.ravel()[here], whom)
    kind = np.full(N, 'удар', dtype=object)
    kind[relative >= PRResponseTYaZhBrushA] = 'протяжка'
    kind[~band] = 'пятно'
    kind[~band & (relative >= PRResponseTYaZhBrushA)] = 'поле'
    kind[(width > PResponseLE * brush) | (n > 3.0 * PResponseLE * PResponseLE * brush * brush)] = 'поле'
    kind[even] = 'ровное'
    kind[((kind == 'протяжка') | (kind == 'удар')) & (np.minimum(width, thick) < LIMIT_PX)] = 'мельче предела'
    at_edges = np.zeros(total, bool)
    for cut in (labels[0], labels[-1], labels[:, 0], labels[:, -1]):
        at_edges[cut] = True
    response = {'вид': kind, 'площадь': n, 'x': cx, 'y': cy, 'угол': np.degrees(angle) % 180.0, 'длина': length, 'ширина': width,
         'толщина': thick, 'полоса': band, 'заполнение': filled, 'разброс': spread}
    t = m, here
    if take is not None:
        for and_, v in response.items():
            response[and_] = np.full(total, 'мельче' if and_ == 'вид' else 0, v.dtype)
            response[and_][take] = v
        t = take[m], here
    response['у края'] = at_edges
    return response, t


def _pass(rgb, brush, window=WINDOW_RANGE, bands=None):
    """Один разбор рабочей картинки с кистью `кисть` px: (метки, области). `окно` — окно местного размаха, кистей."""
    lab0 = np.ascontiguousarray(np.moveaxis(in_units(rgb), -1, 0))   # каналы — подряд
    lab = np.stack([_gauss(c, max(WithGLADAndT, 0.11 * brush)) for c in lab0])
    range = _range(lab, window * brush / 2.0)
    threshold = np.maximum(np.maximum(TONE_MIN, TONE_SHARE * range), ZERNResponse * _grain(lab0, TONE_MIN / ZERNResponse * (1 - 1e-9)))
    E = _strength(lab, max(0.7, 0.08 * brush))
    labels = _basins(E)
    weight = 1.0 / (1.0 + (E / np.maximum(0.25 * threshold, 1e-6)) ** 2)
    labels = _merge(labels, lab, threshold, weight)
    labels = _top_up(labels, lab, threshold, brush)
    labels = _merge(labels, lab, threshold, weight)
    # пиксель у стыка — к той стороне, чья краска ближе, после слияния делает водораздел: стык уже на гребне перепада
    response, (m, here) = _region(labels, lab0[0], brush, bands)
    n = np.maximum(response['площадь'], 1.0)
    response['краска'] = np.stack([np.bincount(m, lab[j].ravel()[here], len(n)) for j in range(3)], 1) / n[:, None]
    response['порог'] = np.bincount(m, threshold.ravel()[here], len(n)) / n
    return labels, response, threshold


def stroke(kind):
    return (kind == 'протяжка') | (kind == 'удар')


# ---------------------------------------------------------------- лестница уменьшений

def _brush_level(a, k):
    """Картинка уровня k: в k раз мельче снимка (k < 1 — крупнее)."""
    from PIL import Image
    if abs(k - 1.0) < 1e-9:
        return a
    return np.asarray(Image.fromarray(a).resize((max(8, int(round(a.shape[1] / k))), max(8, int(round(a.shape[0] / k)))),
                                                Image.LANCZOS))


def _levels(In, Sh, limit_px):
    """Лестница уменьшений от грубого к мелкому, шаг √2. Самый мелкий — тот, где нижний край полосы ширин равен пределу
    снимка; самый грубый — где короткая сторона ещё `СТОРОНА_МИН` рабочих px."""
    k0 = limit_px / PResponseLResponseWithA[0]
    n = int(math.floor(2.0 * math.log2(max(min(In, Sh) / SIDE_MIN / k0, 1.0))))
    return [k0 * 2.0 ** (i / 2.0) for i in range(n, -1, -1)]


def _with_holes(labels, who):
    """У каких областей из `кто` внутри чужого больше `ДЫРЫ` своей площади: это сеть (грунт между ходами, фон вокруг
    мазков), а не полоса, какой бы ровной ни вышла её ширина."""
    da = np.zeros(len(who), bool)
    slices = ndimage.find_objects(labels + 1)
    for i in np.flatnonzero(who):
        if slices[i] is None:
            continue
        m = labels[slices[i]] == i
        own = int(m.sum())
        if m.size - own <= DYRY * own:                          # чужого во всей рамке меньше
            continue
        with_paint = m.any(1)                                      # и внутри строк, от края до края
        between = m.shape[1] - m[:, ::-1].argmax(1) - m.argmax(1)
        if int(between[with_paint].sum()) - own <= DYRY * own:
            continue
        da[i] = int(ndimage.binary_fill_holes(m).sum()) - own > DYRY * own
    return da


def _windows(In, Sh, k):
    """Окна снимка, по которым считается уровень k: весь снимок, пока рабочая картинка уровня не больше `БЮДЖЕТ` px (и
    самого снимка); иначе — 9 плиток сеткой 3×3, подобных снимку, с теми же центрами на всех уровнях (плитка мелкого
    уровня лежит в плитке грубого). Ответ — [(y0, y1, x0, x1)]."""
    share = max(BYuDZhET, float(In * Sh)) * k * k / float(In * Sh)
    if share >= 0.98:
        return [(0, In, 0, Sh)]
    f = math.sqrt(share) / 3.0
    h, w = max(int(round(In * f)), 8), max(int(round(Sh * f)), 8)
    return [(y, y + h, x, x + w) for y in (int(round(In * t - h / 2.0)) for t in (1 / 6, 1 / 2, 5 / 6))
            for x in (int(round(Sh * t - w / 2.0)) for t in (1 / 6, 1 / 2, 5 / 6))]


def normal_width(response):
    """Ширина обычного мазка картинки и какие полосы — мазки, а какие поля: (ширина, маска мазков). Поле — полоса шире
    `ПОЛЕ` ширин мазка этой же картинки (масса снимка, слившаяся на грубом уровне: стена, небо). Ширина мазка — медиана по
    площади ширины самих мазков: ищется сверху, от медианы всех полос, пока не сойдётся (полосы шире `ПОЛЕ` медиан
    уходят в поля, медиана оставшихся — снова). Обрезанное краем ширины не знает."""
    sh, w = response['ширина'], response['вес'] * response['площадь'] * ~response['у края']
    own, W = w > 0, np.inf
    for _ in range(64):
        if not own.any():
            break
        i = np.argsort(sh[own], kind='stable')
        c = np.cumsum(w[own][i])
        W = float(sh[own][i][min(int(np.searchsorted(c, 0.5 * c[-1])), len(i) - 1)])
        further = own & (sh <= PResponseLE * W)
        if further.sum() == own.sum():
            break
        own = further
    return W, own | ((w <= 0) & (sh <= PResponseLE * W))


def _strands(response):
    """Ход растрёпанной кисти — несколько полос одной краски, разделённых грунтом. Пряди одного хода: краска одна (ближе
    порога места), направление одно (`ВДОЛЬ`), короткая лежит вдоль длинной и за её концы выходит не больше общей
    ширины, просвет между ними у́же `ПРОСВЕТ` общей ширины, а вместе они не шире `ХОД` обычных мазков этой картинки
    (половина площади мазков — в мазках не шире) — соседние мазки полной ширины в один не собираются. Ответ: у каждой
    области номер хода (её номер — сама)."""
    n = len(response['x'])
    whose = np.arange(n)
    if n < 2 or not (response['вес'] > 0).any():
        return whose
    limit = CanvasResponseD * normal_width(response)[0]
    xy = np.stack([response['x'], response['y']], 1)
    at = np.radians(response['угол'])
    axis, across = np.stack([np.cos(at), np.sin(at)], 1), np.stack([-np.sin(at), np.cos(at)], 1)
    bottom, top = -0.5 * response['ширина'], 0.5 * response['ширина']              # поперечник хода от его главной пряди
    for j in np.argsort(-response['площадь'], kind='stable'):              # от крупных к мелким: мелкая — к крупной
        if whose[j] != j or response['длина'][j] < PRResponseTYaZhBrushA * response['ширина'][j]:
            continue
        while True:                                                 # прядь за прядью, от ближней
            d = xy - xy[j]
            along, sideways = d @ axis[j], d @ across[j]
            p0, p1 = sideways - 0.5 * response['ширина'], sideways + 0.5 * response['ширина']
            shared = np.maximum(top[j], p1) - np.minimum(bottom[j], p0)
            slit = np.maximum(p0 - top[j], bottom[j] - p1)
            deviation = np.abs((response['угол'] - response['угол'][j] + 90.0) % 180.0 - 90.0)
            d = (whose == np.arange(n)) & (response['площадь'] <= response['площадь'][j]) & (np.arange(n) != j) & (shared <= limit)
            d &= (slit <= PRResponseWithInET * shared) & (deviation <= InDResponseL) & (response['длина'] >= PRResponseTYaZhBrushA * response['ширина'])
            d &= np.abs(along) + 0.5 * response['длина'] <= 0.5 * response['длина'][j] + shared
            d &= np.sqrt(((response['краска'] - response['краска'][j]) ** 2).sum(1)) < np.minimum(response['порог'], response['порог'][j])
            brush = np.flatnonzero(d)
            if not len(brush):
                break
            brush = brush[np.argmin(slit[brush])]
            whose[brush] = j
            whose[whose == brush] = j
            bottom[j], top[j] = min(bottom[j], p0[brush]), max(top[j], p1[brush])
    response['ширина'] = np.where(whose == np.arange(n), np.maximum(response['ширина'], top - bottom), response['ширина'])
    response['площадь'] = np.bincount(whose, response['площадь'], n)
    return whose


def _collect(new, response, whose):
    """Области, чей номер в `чья` — другая область, уходят в неё: (номера меток, области)."""
    sweep = whose == np.arange(len(whose))
    return np.r_[0, np.cumsum(sweep)[whose]].astype(np.int64)[new], {and_: v[sweep] for and_, v in response.items()}


def parse(rgb, limit_px=None):
    """Мазки снимка по лестнице уменьшений. `_проход` своего масштаба не имеет: мазок у него всегда 4–6 рабочих px, какую
    картинку ни дай. Поэтому мазок берётся с того уровня лестницы, где он шире этого пола — `ПОЛОСА` рабочих px, — а не с
    одного рабочего масштаба. Уровни идут от грубого к мелкому. Полоса мелкого уровня, больше чем на `ВНУТРИ` лежащая на
    уже принятом мазке, — его часть. Часть той же краски — борозда или клочок внутри хода, мазком не считается. Часть
    другой краски (дальше от краски целого, чем порог места, с которым оно принято) — мазок поверх него. Если такие
    части одного уровня (две и больше) закрывают `ДЕЛИТ` принятого, оно было не мазком, а их группой (ряд ударов,
    слившийся на грубом уровне) — снимается. Принятое, от которого на следующем уровне не осталось того же целого (пересечение
    к объединению меньше `ТО_ЖЕ`), снимается тоже: мазок держится на двух соседних уровнях, слившаяся масса распадается.
    Тонкое вдоль хода обычной ширины (`ПРОСВЕТ`, `ВДОЛЬ`) — борозда, блик или просвет грунта, на поле — мазок. Мелкие уровни большой картинки считаются по плиткам (`_окна`): у их мазков
    `'вес'` > 1 — скольких мазков снимка стоит один найденный (площадь снимка к площади, где такой мазок мог лечь в плитку
    целиком); у мазка, обрезанного краем плитки, вес 0, краем снимка — 1.

    Ответ: 'метки' — номер мазка + 1 в px снимка (0 — не мазок; мелкие уровни — только в плитках); 'области' — массивы по
    мазкам: 'длина', 'ширина', 'x', 'y', 'площадь' (px снимка), 'угол', 'вид', 'у края', 'уровень', 'вес'; 'уровни'."""
    a = np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    In, Sh = a.shape[:2]
    labels_with = np.zeros((In, Sh), np.int64)
    near_with = np.zeros((In, Sh), np.int64)                          # то же с запасом в рабочий px уровня: край принятого
    fields = ('длина', 'ширина', 'x', 'y', 'угол', 'площадь', 'у края', 'уровень', 'вес', 'краска', 'порог', 'просвет в')
    result = {and_: [] for and_ in fields}
    live, levels = [], []
    waits = np.zeros(1, bool)                                      # принятые прошлым уровнем: распадутся ли на этом
    limit = limit_px or LIMIT_PX
    for k in _levels(In, Sh, limit):
        windows = _windows(In, Sh, k)
        tiles = len(windows) > 1
        taken = 0
        views = []
        shared, all_pr, all_region = np.zeros(len(waits)), np.zeros(len(waits)), np.zeros(len(waits))
        for y0, y1, x0, x1 in windows:
            b = _brush_level(a[y0:y1, x0:x1], k)
            if min(b.shape[:2]) < 24:
                continue
            labels, response, _ = _pass(b, BrushAndWithT, WINDOW_LADDERS, PResponseLResponseWithA)
            h, w = y1 - y0, x1 - x0
            yi = np.minimum((np.arange(h) * (b.shape[0] / h)).astype(np.int64), b.shape[0] - 1)
            xi = np.minimum((np.arange(w) * (b.shape[1] / w)).astype(np.int64), b.shape[1] - 1)
            m_with = labels[yi][:, xi]                                # метки уровня в px снимка
            N, M = len(response['вид']), len(waits)
            n_with = np.maximum(np.bincount(m_with.ravel(), minlength=N), 1).astype(np.float64)
            views.append((y0, y1, x0, x1, labels, response, yi, xi, m_with, n_with))
            if waits.any():                                        # какая область уровня ближе всех к принятому целиком
                point = labels_with[y0:y1, x0:x1]
                where = waits[point]
                option, shared = fast.unique_count(point[where] * np.int64(N) + m_with[where])
                whose, region = option // N, option % N
                threshold = np.lexsort((-shared, whose))
                first = np.r_[True, whose[threshold][1:] != whose[threshold][:-1]] if len(whose) else np.zeros(0, bool)
                whose, region, shared = whose[threshold][first], region[threshold][first], shared[threshold][first]
                net = _with_holes(labels, np.isin(np.arange(N), region))
                np.add.at(shared, whose, np.where(net[region], 0, shared))
                np.add.at(all_pr, whose, np.bincount(point[where], minlength=M)[whose])
                np.add.at(all_region, whose, n_with[region])
        # принятое, которое на следующем уровне не то же целое, — масса, слившаяся на грубом уровне, а не мазок
        fell_apart = waits & (all_pr > 0) & (shared < TO_ZHE * (all_pr + all_region - shared))
        for j in np.flatnonzero(fell_apart):
            live[j - 1] = False
        if fell_apart.any():
            labels_with[fell_apart[labels_with]] = 0
            near_with[fell_apart[near_with]] = 0
        with_this = len(live)
        for y0, y1, x0, x1, labels, response, yi, xi, m_with, n_with in views:
            h, w = y1 - y0, x1 - x0
            ms, exactly = near_with[y0:y1, x0:x1], labels_with[y0:y1, x0:x1]   # виды: правятся на месте
            N, M = len(response['вид']), len(live) + 1
            usable = response['полоса'] & (response['вид'] != 'ровное') & (response['ширина'] >= PResponseLResponseWithA[0]) & (response['ширина'] < PResponseLResponseWithA[1])
            usable &= ~_with_holes(labels, usable)
            where = usable[m_with] & (ms > 0)                           # перекрытие полос уровня с принятыми
            option, shared = fast.unique_count(m_with[where] * np.int64(M) + ms[where])
            who, whose = option // M, option % M
            on_accepted = np.bincount(who, shared, N) / n_with
            threshold = np.lexsort((-shared, who))
            first = np.r_[True, who[threshold][1:] != who[threshold][:-1]] if len(who) else np.zeros(0, bool)
            owner = np.zeros(N, np.int64)
            owner[who[threshold][first]] = whose[threshold][first]
            part = usable & (on_accepted >= InNAtTRAnd)
            pl_accepted = np.bincount(ms.ravel(), minlength=M).astype(np.float64)
            # часть другой краски, чем целое (дальше от его краски, чем порог места, с которым целое принято), — мазок
            # поверх него; той же краски — борозда или клочок внутри хода
            kr_accepted = np.concatenate([np.zeros((1, 3))] + result['краска'])
            threshold_accepted = np.concatenate([np.full(1, np.inf)] + result['порог'])
            foreign = part & (np.sqrt(((response['краска'] - kr_accepted[owner]) ** 2).sum(1)) >= threshold_accepted[owner])
            covered = np.bincount(owner[foreign], (on_accepted * n_with)[foreign], M) / np.maximum(pl_accepted, 1.0)
            group = (covered >= DELAndT) & (np.bincount(owner[foreign], minlength=M) >= 2)
            group[0] = False
            if tiles and group.any():                                            # принятое шире плитки по ней одной группой не назвать
                group &= pl_accepted >= np.bincount(near_with.ravel(), minlength=M) * 0.999
            # вдоль целого и в разы у́же его (или короче его ширины) — борозда, блик её гребня, просвет грунта
            sh_pr = np.concatenate([np.full(1, 1.0)] + result['ширина'])[owner]
            at_pr = np.concatenate([np.zeros(1)] + result['угол'])[owner]
            deviation = np.abs((response['угол'] - at_pr + 90.0) % 180.0 - 90.0)
            gap = foreign & (response['ширина'] * k <= PRResponseWithInET * sh_pr) & ((deviation <= InDResponseL) | (response['длина'] * k <= sh_pr))
            gap &= ~group[owner]                            # просвет в ходе или мазок на массе — решается в конце
            take = usable & (~part | foreign)
            for j in np.flatnonzero(group):
                live[j - 1] = False
            if group.any():
                ms[group[ms]] = 0
                exactly[group[exactly]] = 0
            number = np.zeros(N, np.int64)
            number[take] = M + np.arange(int(take.sum()))
            new = number[labels]
            wider = ndimage.grey_dilation(new, size=3)[yi][:, xi]  # край области уровня известен до его рабочего px
            new = new[yi][:, xi]
            np.copyto(exactly, new, where=new > 0)                  # мазок мелкого уровня — поверх принятого
            np.copyto(ms, new, where=new > 0)
            np.copyto(ms, wider, where=(wider > 0) & (ms == 0))
            n = int(take.sum())
            live += [True] * n
            taken += n
            d, sh, at = response['длина'][take] * k, response['ширина'][take] * k, np.radians(response['угол'][take])
            weight = np.ones(n)
            if tiles:                                            # где мазок мог лечь в плитку целиком; длиннее половины
                # плитки — как половина её: вес одного случайного куска не растёт без края (счёт таких — снизу)
                bx = np.abs(np.cos(at)) * d + np.abs(np.sin(at)) * sh
                by = np.abs(np.sin(at)) * d + np.abs(np.cos(at)) * sh
                place = np.maximum(w - bx, 0.5 * w) * np.maximum(h - by, 0.5 * h) * len(windows)
                weight = np.where(response['у края'][take] | (place <= 0), 0.0, In * Sh / np.maximum(place, 1e-9))
            for and_, v in (('длина', d), ('ширина', sh), ('x', response['x'][take] * k + x0), ('y', response['y'][take] * k + y0),
                         ('площадь', response['площадь'][take] * k * k), ('угол', response['угол'][take]),
                         ('у края', response['у края'][take]), ('уровень', np.full(n, k)), ('вес', weight),
                         ('краска', response['краска'][take]), ('порог', response['порог'][take]),
                         ('просвет в', np.where(gap, owner, 0)[take])):
                result[and_].append(v)
        waits = np.zeros(len(live) + 1, bool)
        waits[with_this + 1:] = True
        waits[1:] &= np.array(live, bool)
        levels.append({'k': k, 'плитки': tiles, 'взято': taken,
                       'доля': sum((y1 - y0) * (x1 - x0) for y0, y1, x0, x1 in windows) / float(In * Sh)})
    live = np.array(live, bool)
    for j in range(len(levels)):                                  # по горстке кусков на всю картину не пересчитать: сколько видно
        if levels[j]['плитки']:
            weight, ur = result['вес'], np.concatenate(result['уровень'])
            weight = np.concatenate(weight)
            this = (ur == levels[j]['k']) & (weight > 0)
            if (this & live).sum() < FEW_STROKES:
                weight[this] = 1.0
            result['вес'] = [weight]
    new = np.r_[0, np.where(live, np.cumsum(live), 0)]
    response = {and_: (np.concatenate(v) if v else np.zeros((0, 3) if and_ == 'краска' else 0))[live] for and_, v in result.items()}
    response['у края'] = response['у края'].astype(bool)
    # тонкое вдоль принятого: в ходе обычной ширины и поштучно — борозда или просвет, на поле — мазок поверх него
    in_lump = new[response.pop('просвет в').astype(np.int64)]
    not_field = np.r_[False, normal_width(response)[1]]
    how_much = np.bincount(in_lump, minlength=len(not_field))          # мазков хватает на свой размер — это мазки на массе
    whose = np.where((in_lump > 0) & not_field[in_lump] & (how_much < FEW_STROKES)[in_lump], in_lump - 1, np.arange(len(in_lump)))
    for _ in range(len(levels)):
        whose = whose[whose]
    new, response = _collect(new, response, whose)
    new, response = _collect(new, response, _strands(response))
    response['вид'] = np.where(response['длина'] >= PRResponseTYaZhBrushA * response['ширина'], 'протяжка', 'удар').astype(object)
    for at in levels:
        at['осталось'] = int((response['уровень'] == at['k']).sum())
    return {'метки': new[labels_with], 'области': response, 'уровни': levels, 'картинка': a}


# ---------------------------------------------------------------- форма одного мазка (Мн-2)

def shape(r, j, cut=None):
    """Форма мазка `j` разбора `р` в ширинах его кисти (`рука.замер`) с местом: 'номер', 'x', 'y', 'концы' — (y, x) кончиков
    входа и схода, 'слово'; px снимка. Краску мазка от соседей отделяет снимок, а не ступенька рабочего px. None — мазка не видно."""
    from . import hand
    S, response = r['метки'], r['области']
    if cut is None:
        where = np.argwhere(S == j + 1)
        if not len(where):
            return None
        cut = tuple(slice(int(a), int(b) + 1) for a, b in zip(where.min(0), where.max(0)))
    w = float(response['ширина'][j])
    k, field = max(1, int(w // hand.ResponseTWithHYoTResponseIn)), int(math.ceil(w)) + 3
    y0, x0 = (max(0, with_.start - field) for with_ in cut)
    y1, x1 = (min(n, with_.stop + field) for with_, n in zip(cut, S.shape))
    h, sh = (y1 - y0) // k * k, (x1 - x0) // k * k
    cells = lambda a: a[:h, :sh].reshape((h // k, k, sh // k, k) + a.shape[2:]).mean((1, 3)) if k > 1 else a
    mask = cells((S[y0:y1, x0:x1] == j + 1).astype(np.float64)) > 0.5
    if mask.sum() < 12:
        return None
    z = hand.measurement(hand.trace(in_units(cells(r['картинка'][y0:y1, x0:x1].astype(np.float64))), mask, w / k), w / k)
    if z is None:
        return None
    z['W'] *= k
    z['концы'] = [(y0 + a * k + 0.5 * (k - 1), x0 + b * k + 0.5 * (k - 1)) for a, b in z['концы']]
    z.update({'номер': int(j), 'x': float(response['x'][j]), 'y': float(response['y'][j]), 'вес': float(response['вес'][j]), 'слово': hand.words(z)})
    return z


FResponseRM = 200                # мазков меряется не больше: картина крупнее — каждый n-й по порядку разбора


def shape(r, integers):
    """Форма целых мазков (номера `целые`), чья ширина различает концы; не больше `ФОРМ`."""
    from . import hand
    e = integers[r['области']['ширина'][integers] >= hand.InAndDNResponse]
    if len(e) > FResponseRM:
        e = e[np.linspace(0, len(e) - 1, FResponseRM).round().astype(np.int64)]
    slices = ndimage.find_objects(r['метки'], max(int(r['метки'].max()), 1))
    return [z for z in (shape(r, j, slices[j]) for j in e if j < len(slices) and slices[j] is not None) if z is not None]


# ---------------------------------------------------------------- края пятен

EDGE_SHOULDER = 8            # px: полудлина прямого отрезка края
MELBrushAYa, BrushRAtPNAYa = 0.7, 3.0  # px: сигмы пары, по которой меряется переход края


def edges_patches(rgb):
    """Границы пятен (наследник `архив/анализатор/край.py`): гребень силы перепада краски выше ступени `ТОН_МИН`.
    Ответ — {'границ': доля площади, 'прямых': доля пикселей границы на прямом отрезке ±`КРАЙ_ПЛЕЧО` px, 'переход':
    массив — ширина перехода 20–80 % в каждом пикселе границы, px}. Переход — по отношению силы перепада на двух
    сигмах: у ступени, размытой на s, сила на сигме σ идёт как 1/√(s² + σ²)."""
    lab = np.ascontiguousarray(np.moveaxis(in_units(rgb), -1, 0))
    σ = 1.5
    strength, angle = _tensor(lab, σ)
    c, s = np.cos(angle), np.sin(angle)
    gr = strength > TONE_MIN / (σ * math.sqrt(2 * math.pi))
    y, x = np.nonzero(gr)                                         # гребень ищется там, где перепад не мал
    vz = lambda i: ndimage.map_coordinates(strength, [y + i * s[y, x], x + i * c[y, x]], order=1, mode='nearest')
    gr[y, x] = (strength[y, x] >= vz(1)) & (strength[y, x] > vz(-1))
    if not gr.any():
        return {'границ': 0.0, 'прямых': None, 'переход': np.zeros(0)}
    y, x = np.nonzero(gr)
    thicker = ndimage.binary_dilation(gr, np.ones((3, 3))).astype(np.float32)
    cc, ss = c[y, x], s[y, x]
    d = np.r_[-EDGE_SHOULDER:0, 1:EDGE_SHOULDER + 1].astype(np.float64)
    Y, X = y[:, None] + d * cc[:, None], x[:, None] - d * ss[:, None]        # касательная — поперёк перепада
    inside = (Y >= 0) & (Y <= gr.shape[0] - 1) & (X >= 0) & (X <= gr.shape[1] - 1)
    hit = ndimage.map_coordinates(thicker, [Y, X], order=0, mode='constant') > 0.5
    angle = ndimage.map_coordinates(angle, [Y, X], order=0, mode='nearest')
    even = np.abs(np.cos(angle - angle[y, x][:, None])) > math.cos(math.radians(20.0))
    straight = (hit & even & inside).mean(1) >= 0.9
    r2 = (_tensor(lab, MELBrushAYa, (y, x))[0] / np.maximum(_tensor(lab, BrushRAtPNAYa, (y, x))[0], 1e-6)) ** 2
    top = (BrushRAtPNAYa / MELBrushAYa) ** 2
    r2 = np.clip(r2, 1.02, top)
    s2 = np.maximum((BrushRAtPNAYa ** 2 - r2 * MELBrushAYa ** 2) / (r2 - 1.0), 0.0)
    return {'границ': float(gr.mean()), 'прямых': float(straight.mean()), 'переход': 1.683 * np.sqrt(s2)}


def _tensor(lab, σ, where=None):
    """(сила, угол) тензора структуры по трём каналам `lab` (3, В, Ш); `где` — только в этих px."""
    xx = yy = xy = 0.0
    for c in lab:
        gx, gy = (ndimage.gaussian_filter(c, σ, order=response) for response in ((0, 1), (1, 0)))
        if where is not None:
            gx, gy = gx[where], gy[where]
        xx, yy, xy = xx + gx * gx, yy + gy * gy, xy + gx * gy
    return (np.sqrt(np.maximum(0.5 * (xx + yy + np.sqrt((xx - yy) ** 2 + 4 * xy * xy)), 0.0)),
            0.5 * np.arctan2(2 * xy, xx - yy))
