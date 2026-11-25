# READING VIEW of движок/кисть/снятие.py with names in English. Not runnable; edit the original.
"""К8.4: снятие сырой краски по ходу — тряпка и скребок; К-14а: черенок (в журнале — строка с полем "tool").

Тряпка — комок ткани: прижатая, она касается плёнки верхами нитей и складок, а ложбины между ними до краски не
достают. Нить и складка тащатся вместе с комком, поэтому каждая оставляет полосу вдоль хода: где за проход касание
было чаще, краски снято больше. Сырое впитывается в ткань, пока ткань не напиталась: к концу длинного хода она
снимает слабее. Пигмент, вошедший в грунт, ткань не берёт — на месте остаётся тонкий тон.

Скребок — жёсткая кромка поперёк хода: где она прошла, сырая краска снята до грунта. Снятое копится валом перед
кромкой; вал выдавливается за её концы — по краям следа ложится валик краски, на подъёме остаток вала остаётся
комком. Сухой и схватившийся слой ни то ни другое не трогает: снимается только сырая доля плёнки, несхваченная.

Черенок (К-14а) — круглый твёрдый конец ручки кисти: полушарие поперечником в `ширина`. Краски он не снимает, а
раздвигает: всё сырое, что выше его поверхности, уходит из-под него в стороны и ложится валиками у боков борозды
(объём на холсте тот же). Сечение борозды — дуга конца: до твёрдого он доходит одной линией посередине, к бокам плёнка
толще. Твёрдое — грунт с его зерном, сухая и схватившаяся краска — его не пускает: конец едет по верхам твёрдого, в
ямках под ним краска остаётся. Нажим — зазор: под слабой рукой между концом и твёрдым остаётся плёнка (борозда в
краске), под сильной конец садится на твёрдое."""
import numpy as np

from .fast import clamp

from .paint import in_ks, from_ks
from .grain import Grain
from .strands import _noise
from .numbers import _TAAt, _WithNYaTAndE, _WithBrushREBResponseBrush, _TEN

# К-14а — числа черенка (`черенок`). Лежат здесь, а не в `числа.py`: мазка кисти они не касаются, а `числа.py` входит
# в хеш физики таблиц калибровки (`калибровка._ФИЗИКА`) — новая строка там сделала бы таблицы чужими без причины
_HERENResponseBrush = (0.8, 0.1, 0.03, 0.6, 0.02, 0.01)   # нажим, с которого конец садится на твёрдое (плёнка под ним рвётся); зазор между концом и твёрдым, h, у конца 1 мм при половине этого нажима (зазор — как радиус конца и как этот нажим к нажиму руки без единицы: паста из сходящейся щели выжимается тем труднее, чем щель тоньше и конец шире); плёнка над твёрдым, h (12 мкм — зерно пигмента), тоньше которой под концом не остаётся: твёрдое по твёрдому стирает её дочиста; валик за боком конца не шире этой доли поперечника конца и ещё 4 px; нажим, ниже которого конец над холстом; сухая доля слоя, h, с которой он для конца твёрдый (под ним не берётся и нижнее)

AndNWithTRAtMENTY = ('тряпка', 'скребок', 'черенок')


def _step(points, spacing=1.0):
    """Точки (x, y, нажим, …) -> путь через `шаг` px: (x, y, нажим), нажим по прямой между точками."""
    t = np.asarray(points, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] < 3:
        raise ValueError('снятие: точки — (x, y, нажим, …)')
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(t[:, :2], axis=0).T))])
    if s[-1] <= 0:
        return t[:1, :3].copy(), np.array([[1.0, 0.0]])
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    u = np.linspace(0.0, s[-1], n)
    xy = np.column_stack([np.interp(u, s, t[:, 0]), np.interp(u, s, t[:, 1]), np.interp(u, s, clamp(t[:, 2], 0.0, 1.0))])
    d = np.gradient(xy[:, :2], axis=0)
    d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    return xy, d


def _wet(with_, where, left):
    """Сколько плёнки слоя можно снять (h): сырая доля, не схваченная, сверх `осталось` — пигмента, вошедшего в грунт
    (тряпка его не берёт, кромка скребка по грунту оставляет его меньше)."""
    h = with_.thickness[where]
    # масляная плёнка снимается целиком, пока влажность не ниже `_СНЯТИЕ[6]`; подсыхающая — по своей сырой доле
    return np.maximum(h * clamp(with_.moisture[where] / _WithNYaTAndE[6], 0.0, 1.0) * (1.0 - with_.degree_of_set[where]) - left, 0.0)


def _remove(with_, where, remove):
    """Убрать из слоя толщину `снять` (h) сырой краски: плёнка тоньше, покрытие — по оставшейся толщине."""
    h = with_.thickness[where]
    h2 = np.maximum(h - remove, 0.0)
    was = 1.0 - np.exp(-h / _TAAt)
    became = 1.0 - np.exp(-h2 / _TAAt)
    with_.coverage[where] = with_.coverage[where] * clamp(became / np.maximum(was, 1e-9), 0.0, 1.0) ** 0.7
    wet = np.maximum(h * with_.moisture[where] - remove, 0.0)
    with_.moisture[where] = np.where(h2 > 1e-9, clamp(wet / np.maximum(h2, 1e-9), 0.0, 1.0), 0.0)
    with_.thickness[where] = h2


def _lay(with_, where, dh, ks):
    """Положить в слой толщину dh (h) сырой краски цвета кс (K/S): смесь с лежащим по толщине."""
    h = with_.thickness[where]
    h2 = h + dh
    share = dh / np.maximum(h2, 1e-9)
    was = in_ks(with_.color[where].astype(np.float64))
    color = np.where((with_.coverage[where] > 1e-4)[..., None], from_ks(was * (1.0 - share[..., None]) + ks * share[..., None]),
                    from_ks(np.broadcast_to(ks, was.shape)))
    with_.color[where] = np.where((dh > 0)[..., None], color, with_.color[where])
    with_.coverage[where] = np.maximum(with_.coverage[where], 1.0 - np.exp(-h2 / _TAAt))
    with_.moisture[where] = np.where(h2 > 1e-9, (h * with_.moisture[where] + dh) / np.maximum(h2, 1e-9), with_.moisture[where])
    with_.degree_of_set[where] = np.where(h2 > 1e-9, with_.degree_of_set[where] * h / np.maximum(h2, 1e-9), with_.degree_of_set[where])
    with_.thickness[where] = h2


def _cloth(R, rng, spacing):
    """Комок ткани в своей рамке (2R+1, 2R+1): высота верхов нитей и складок 0..1."""
    brush = 2 * R + 1
    z = Grain(spacing, 1.0, int(rng.integers(1 << 30)))
    x0, y0 = (int(v) for v in rng.integers(0, 4096, 2))
    threads = z.height(x0, y0, x0 + brush, y0 + brush).astype(np.float64)
    # складки смятой ткани: гладкие гребни в 1–3 мм, своя высота у каждого места комка
    folds = _noise(rng, brush, brush, max(R / 2.0, 3.0))
    folds = 0.5 * (folds + _noise(rng, brush, brush, max(R / 2.0, 3.0)).T)
    folds = (folds - folds.min()) / max(float(np.ptp(folds)), 1e-9)
    return _WithNYaTAndE[1] * threads + (1.0 - _WithNYaTAndE[1]) * folds


def _under_lump(t, R2, dx, dy, d):
    """Высота ткани под пикселями (dx, dy — px от середины комка) у комка, который тащат по `d`: складки и нити
    стоят по ходу (ткань, тянутая по холсту, складывается вдоль тяги), строка ткани — поперёк хода. Ткань `т`
    (2·R2+1 с полем в 1 px); между узлами — по прямой."""
    along, across = dx * d[0] + dy * d[1], -dx * d[1] + dy * d[0]
    r, c = clamp(R2 + 1 + across, 0.0, 2 * R2 + 1.999), clamp(R2 + 1 + along, 0.0, 2 * R2 + 1.999)
    r0, c0 = np.floor(r).astype(np.intp), np.floor(c).astype(np.intp)
    ar, ac = r - r0, c - c0
    return (1 - ar) * ((1 - ac) * t[r0, c0] + ac * t[r0, c0 + 1]) + ar * ((1 - ac) * t[r0 + 1, c0] + ac * t[r0 + 1, c0 + 1])


def rag(canvas, points, width, seed, layers):
    """Снять сырую краску тряпкой по пути `точки` (x, y, нажим, …); `ширина` — поперечник комка, px.

    Комок мягкий: прижатый, он сминается, и пятно касания растёт с нажимом (радиус — как корень кубический нажима,
    от `_СНЯТИЕ[9]` — весь комок); давление по пятну — горбом, у края пятна верхи едва касаются. Сырое масло липнет
    к ткани и при лёгком касании: где верх нити или складки тронул плёнку, краска уходит; ложбины между верхами до
    плёнки не достают — каждый верх тащит за собой полосу вдоль хода."""
    path, dd = _step(points)
    rng = np.random.default_rng([int(seed), 811])
    R = max(2, int(round(width / 2.0)))
    R2 = int(np.ceil(1.42 * R)) + 1                      # ткань с запасом: комок повёрнут по ходу
    cloth = np.pad(_cloth(R2, rng, max(2.0, _WithNYaTAndE[2] * canvas.px_on_mm / 8.0 if canvas.px_on_mm else _WithNYaTAndE[2])), 1, mode='edge')
    k = np.arange(-R, R + 1, dtype=np.float64)
    # проход комка через пиксель длится ~2R шагов: съём за шаг — доля, чтобы весь проход снимал `_СНЯТИЕ[3]` в логарифме
    # на каждую мягкость касания, на которую верх ткани вошёл в плёнку
    behind_spacing = _WithNYaTAndE[3] / (2.0 * R)
    capacity = _WithNYaTAndE[4] * (2 * R + 1) ** 2
    absorbed = 0.0
    for (x, y, p), d in zip(path, dd):
        if p <= 0.0:
            continue
        ix, iy = int(round(x)) - canvas.x0, int(round(y)) - canvas.y0
        # К8б: комок тянут по ходу — складки и нити ткани ложатся вдоль тяги, и каждая ведёт свою полосу под любым углом
        # хода (ткань в рамке холста косой ход пересекал наискось: каждый пиксель собирал и верхи, и ложбины — туман);
        # комок стоит в своей доле пикселя, а не на сетке
        dy, dx = k[:, None] - (y - canvas.y0 - iy), k[None, :] - (x - canvas.x0 - ix)
        top = _under_lump(cloth, R2, dx, dy, d)
        r2 = (dx * dx + dy * dy) / max(R, 1) ** 2
        ya, yb, xa, xb = iy - R, iy + R + 1, ix - R, ix + R + 1
        ca, cb, da, db = max(ya, 0), min(yb, canvas.height), max(xa, 0), min(xb, canvas.width)
        if ca >= cb or da >= db:
            continue
        where = (slice(ca, cb), slice(da, db))
        frame = (slice(ca - ya, cb - ya), slice(da - xa, db - xa))
        press = np.sqrt(clamp(1.0 - r2[frame] / min(p / _WithNYaTAndE[9], 1.0) ** (2.0 / 3.0), 0.0, 1.0))
        # К8б: верх ткани, вдавленный в плёнку глубже, давит на неё сильнее и снимает больше; ложбина, что едва тронула
        # плёнку, — меньше (касание не насыщается на верхах: у прижатого комка касается вся середина, и ровный съём по
        # всему пятну давал молочную полосу без полос ткани)
        touch = clamp((top[frame] - (1.0 - (_WithNYaTAndE[7] + _WithNYaTAndE[8] * p) * press)) / _WithNYaTAndE[5], 0.0, None)
        strength = behind_spacing * max(0.0, 1.0 - absorbed / capacity)
        if strength <= 0:
            break
        share = 1.0 - np.exp(-strength * touch)
        for with_ in layers:
            remove = _wet(with_, where, _WithNYaTAndE[0]) * share
            if remove.any():
                absorbed += float(remove.sum())
                _remove(with_, where, remove)


def _window(canvas, x, y, d, m):
    """Окно холста радиусом м px у точки (x, y) хода d: срезы, координаты вдоль и поперёк хода; None — вне холста."""
    ix, iy = int(round(x)) - canvas.x0, int(round(y)) - canvas.y0
    ca, cb, da, db = max(iy - m, 0), min(iy + m + 1, canvas.height), max(ix - m, 0), min(ix + m + 1, canvas.width)
    if ca >= cb or da >= db:
        return None
    Y, X = np.mgrid[ca:cb, da:db].astype(np.float64)
    dx, dy = X + canvas.x0 - x, Y + canvas.y0 - y
    return (slice(ca, cb), slice(da, db)), dx * d[0] + dy * d[1], -dx * d[1] + dy * d[0]


def _hump(t, s):
    """Сечение горба масла полушириной s (px) по t (px от его середины): 1 - (t/s)², вне — 0."""
    return clamp(1.0 - (t / np.maximum(s, 1e-6)) ** 2, 0.0, 1.0)


def scraper(canvas, points, width, seed, layers):
    """Снять сырую краску скребком по пути `точки`: кромка шириной `ширина` px поперёк хода.

    Снятое копится валом перед кромкой там, где снято: вал — сечения по местам кромки (1 px), у каждого свой объём и
    свои краски. Вал растекается вдоль кромки от высокого к низкому и сползает к одному концу (кромку держат не ровно).
    Масло — паста с пределом текучести: у конца кромки вал течёт наружу, только когда его сечение выше предела, —
    вал, что снял мало, не выдавливается вовсе; подсохшее масло (вал суше точки студня) не течёт совсем. Выдавленное
    ложится валиком горбом той крутизны, что держит масло: полуширина — от объёма порции (больше — шире и выше),
    краски — полосами по местам вала, откуда пришли: крайнее у конца ложится дальше. На подъёме вал остаётся комом:
    каждое место — шапкой по своему объёму."""
    path, d = _step(points, 0.5)
    rng = np.random.default_rng([int(seed), 812])
    W2 = width / 2.0
    nb = max(2, int(np.ceil(width)))
    shb = width / nb                                         # ширина места вала вдоль кромки, px
    tilt = float(clamp(_WithBrushREBResponseBrush[0] * rng.standard_normal(), -0.45, 0.45))
    κ, limit, spread, flow, gel = _WithBrushREBResponseBrush[2], _WithBrushREBResponseBrush[3] * shb, _WithBrushREBResponseBrush[4], _WithBrushREBResponseBrush[5], _WithBrushREBResponseBrush[7]
    s_max = max(1.0, _WithBrushREBResponseBrush[6] * width)
    L = len(layers)
    bank = np.zeros((L, nb))                                  # объём вала по слоям и местам кромки, h·px²
    bank_ks = np.zeros((L, nb, 24))                           # K/S · объём
    bank_vl = np.zeros(nb)                                    # влажность · объём (все слои)
    m = int(np.ceil(W2 + 2.0 * s_max + 2.0))
    for i, (x, y, p) in enumerate(path):
        if p <= 0.02:
            continue
        response = _window(canvas, x, y, d[i], m)
        if response is None:
            continue
        where, along, across = response
        spacing_along = clamp(1.0 - np.abs(along) / 0.75, 0.0, 1.0)
        # кромка: полоса в шаг пути, по ширине — до концов с мягким краем в 1 px
        rim = spacing_along * clamp(W2 + 0.5 - np.abs(across), 0.0, 1.0) * min(p / 0.3, 1.0)
        place = clamp(((across + W2) / shb).astype(np.int64), 0, nb - 1)
        has = rim > 0
        for k, with_ in enumerate(layers):
            remove = _wet(with_, where, _WithBrushREBResponseBrush[1]) * rim
            if not remove.any():
                continue
            ks = in_ks(with_.color[where][has].astype(np.float64)) * remove[has][:, None]
            bank[k] += np.bincount(place[has], remove[has], nb)
            bank_vl[:] += np.bincount(place[has], (remove * with_.moisture[where])[has], nb)
            for q in range(24):
                bank_ks[k, :, q] += np.bincount(place[has], ks[:, q], nb)
            _remove(with_, where, remove)
        T = bank.sum(0)
        if T.max() <= 0:
            continue
        # вал растекается вдоль кромки от высокого сечения к низкому и сползает по крену; переносится тем, что лежит в месте
        flow = spread * (T[:-1] - T[1:]) + tilt * np.where(tilt > 0, T[:-1], T[1:])     # из места j в j+1 (минус — назад)
        flow = clamp(flow, -0.45 * T[1:], 0.45 * T[:-1])
        from_left = clamp(flow, 0, None) / np.maximum(T[:-1], 1e-12)
        from_right = clamp(-flow, 0, None) / np.maximum(T[1:], 1e-12)
        for m_ in (bank, bank_ks, bank_vl[None]):
            a = m_[:, :-1] * (from_left[:, None] if m_.ndim == 3 else from_left)
            b = m_[:, 1:] * (from_right[:, None] if m_.ndim == 3 else from_right)
            m_[:, :-1] += b - a
            m_[:, 1:] += a - b
        # у конца кромки вал держится, пока сечение не выше предела текучести; сверх него масло течёт за конец — тем
        # быстрее, чем больше превышение (паста Бингама). Подсыхающее масло густеет: предел растёт как квадрат обратной
        # доли влажности вала над точкой студня `_СКРЕБОК[7]`; вал из масла, схватившегося до студня, не течёт вовсе —
        # липким комом едет перед кромкой до подъёма. Конец кормится тем, что кромка сняла у него: где краски было
        # больше, течёт больше
        for side, j0, spacing_j in ((0, 0, 1), (1, nb - 1, -1)):
            Tk = bank[:, j0].sum()
            vl = float(bank_vl[j0] / max(Tk, 1e-12))
            current = (vl - gel) / (1.0 - gel)
            if current <= 0.0 or Tk <= limit / current ** 2:
                continue
            prev = limit / current ** 2
            if Tk <= prev:
                continue
            # течёт нависшее: у каждого места от конца внутрь — доля превышения, пока превышение есть
            portions = []
            for j in range(j0, j0 + spacing_j * nb, spacing_j):
                Tj = bank[:, j].sum()
                if Tj <= prev:
                    break
                share = flow * (1.0 - prev / Tj)
                portions.append((bank[:, j] * share, bank_ks[:, j] * share))
                bank[:, j] *= 1.0 - share
                bank_ks[:, j] *= 1.0 - share
                bank_vl[j] *= 1.0 - share
            V = sum(float(p[0].sum()) for p in portions)
            # валик — горб крутизны κ: площадь сечения A = V / шаг пути -> полуширина s = sqrt(3A / (4κ)); лежит
            # от конца кромки наружу
            s = min(np.sqrt(3.0 * (V / 0.5) / (4.0 * κ)), s_max)
            t = (across if side == 1 else -across) - W2         # px за концом кромки
            shape = _hump(t - s, s) * spacing_along
            sum = float(shape.sum())
            if V <= 0 or sum <= 0:
                continue
            # полосы: доля объёма валика от внешнего края внутрь — по порядку порций
            order = np.argsort(-t, axis=None)
            accumulated = np.empty(shape.size)
            accumulated[order] = np.cumsum(shape.ravel()[order]) / sum
            accumulated = accumulated.reshape(shape.shape)
            boundary = 0.0
            for volume, ks in portions:
                share_p = float(volume.sum()) / V
                if share_p <= 0:
                    continue
                weight = shape * ((accumulated > boundary - 1e-9) & (accumulated <= boundary + share_p + 1e-9))
                boundary += share_p
                sv = float(weight.sum())
                if sv <= 0:
                    weight, sv = shape, sum
                for k, with_ in enumerate(layers):
                    if volume[k] > 0:
                        _lay(with_, where, weight / sv * volume[k], ks[k] / volume[k])
    # на подъёме вал остаётся у кромки: каждое место вала оседает шапкой той же крутизны κ (радиус — по своему
    # объёму) перед кромкой; шапки сливаются в ком, у концов кромки он круглый, как оплывшее масло
    if bank.sum() > 0:
        live = np.nonzero(path[:, 2] > 0.02)[0]
        i = int(live[-1]) if len(live) else len(path) - 1
        T = bank.sum(0)
        s = np.minimum((2.0 * T / (np.pi * κ)) ** (1.0 / 3.0), s_max)
        response = _window(canvas, path[i, 0], path[i, 1], d[i], int(np.ceil(W2 + 2.0 * s.max() + 2.0)))
        if response is not None:
            where, along, across = response
            for jm in np.nonzero(T > 0)[0]:
                weight = _hump(np.hypot(along - s[jm], across - (-W2 + (jm + 0.5) * shb)), s[jm])
                sv = float(weight.sum())
                if sv <= 0:
                    continue
                for k, with_ in enumerate(layers):
                    if bank[k, jm] > 0:
                        _lay(with_, where, weight / sv * bank[k, jm], bank_ks[k, jm] / bank[k, jm])


def _lay_in_frame(with_, where, dh, ks):
    """`_положить` только в рамке пикселей, куда краска ложится (dh > 0): итог тот же, счёт — по малому куску."""
    ys, xs = np.nonzero(dh > 0)
    if not len(ys):
        return
    ya, yb, xa, xb = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    _lay(with_, (slice(where[0].start + ya, where[0].start + yb), slice(where[1].start + xa, where[1].start + xb)), dh[ya:yb, xa:xb], ks)


class Scratched:
    """Ответ `х.снять(…, 'черенок')`: `прямоугольник` (x0, y0, x1, y1) хода с валиками; `длина` — px пути, где конец
    шёл в краске; `до_дна` — доля этой длины, где под серединой конца сырого не осталось (открыт нижний слой или
    грунт); `осталось` — сырая плёнка под серединой конца после прохода, мм (медиана по ходу; 0 — дно открыто);
    `на_твёрдом` — доля этой длины, где нажим посадил конец на твёрдое (при ней плёнка под концом — краска в ямках
    твёрдого: зерно, рельеф сухого слоя);
    `ширина` — поперечник борозды у поверхности краски, px (медиана по ходу); `глубже` — самая глубокая
    борозда, мм; `сдвинуто` — объём раздвинутой краски, h·px² (остался на холсте валиками), `мм3` — он же в мм³;
    `почему` — слово, если конец краски не тронул."""
    __slots__ = ('прямоугольник', 'длина', 'до_дна', 'осталось', 'на_твёрдом', 'ширина', 'глубже', 'сдвинуто', 'мм3', 'почему')

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields.get(name))

    def __bool__(self):
        return bool(self.shifted)

    def __repr__(self):
        if not self.shifted:
            return f'Процарапано(ничего: {self.why})'
        return (f'Процарапано(борозда {self.length:.0f} px, шириной {self.width:.1f} px, до дна {self.to_bottom:.0%} пути'
                + ('' if self.left <= 0.004 else f' (конец шёл по верхам твёрдого — зерно, рельеф сухого: в ямках под ним плёнка {self.left:.2f} мм)'
                   if self.on_firmly > 0.5 else f' (под концом осталась плёнка {self.left:.2f} мм: до твёрдого — нажим от {_HERENResponseBrush[0]:g})') + ', '
                f'глубже всего {self.deeper:.2f} мм, раздвинуто {self.shifted:.1f} h·px² ({self.mm3:.3f} мм³) — в валиках)')


def handle_end(canvas, points, width, layers, all=None):
    """Процарапать сырую краску концом черенка по пути `точки` (x, y, нажим, …): `ширина` — поперечник конца, px.

    Конец — полушарие радиуса R = ширина / 2: его поверхность над нижней точкой — R − √(R² − r²). Нижняя точка стоит
    над твёрдым (грунт с зерном, сухая и схватившаяся краска, краска слоёв, что не снимаются) на зазор по нажиму:
    краска — паста, из-под сходящегося с твёрдым конца она выжимается тем труднее, чем тоньше зазор и шире конец
    (зазор — как радиус конца на нажим); с нажима `_ЧЕРЕНОК[0]` плёнка рвётся — конец сел на твёрдое, на его верхи
    под всем полушарием. Всё сырое выше поверхности конца — вал перед ним, по местам поперёк хода. Круглый лоб вала
    не держит: место вала в t от оси уходит к боку на шаг · t / √(a² − t²) (a — радиус касания у поверхности краски),
    у оси — оплывает к соседям. Что дошло до бока, ложится за ним валиком горбом той крутизны, что держит масло
    (как у скребка): где краски было больше, борозда шире и валик выше. На подъёме остаток вала — комом перед концом.
    Объём краски на холсте не меняется: что ушло из борозды, лежит в валиках и коме (кроме ушедшего за край холста)."""
    path, d = _step(points, 0.5)
    pm = float(canvas.px_on_mm or 8.0)
    # ход — ломаная по точкам: направление (куда смотрят бока конца) ведём средним по длине в поперечник конца —
    # у вершины ломаной валик не рвётся зубцом; сам конец идёт по точкам
    window = 2 * max(2, int(round(width))) + 1
    if len(d) > 2:
        dp = np.pad(d, ((window // 2, window // 2), (0, 0)), mode='edge')
        d = np.column_stack([np.convolve(dp[:, q], np.ones(window) / window, mode='valid') for q in (0, 1)])
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    height = _TEN[0] * pm / 8.0                                # px высоты в единице h
    R = width / 2.0
    nb = max(2, int(np.ceil(width)))
    shb = width / nb
    tj = -R + (np.arange(nb) + 0.5) * shb                     # места вала поперёк хода, px от оси
    κ, spread = _WithBrushREBResponseBrush[2] * 8.0 / pm, _WithBrushREBResponseBrush[4]          # откос масла — h на px холста 8 px/мм: на своём px — по масштабу
    s_max = _HERENResponseBrush[3] * width + 0.5 * pm                  # валик не шире этого за боком конца, px
    tt = (np.arange(int(np.ceil((R + s_max) / 0.25))) + 0.5) * 0.25
    ends = np.linspace(0.0, R + s_max, 96)
    all = layers if all is None else all
    L = len(all)
    bank, bank_ks = np.zeros((L, nb)), np.zeros((L, nb, 24))
    m = int(np.ceil(R + s_max + 2.0))
    a_side = [min(R, 1.0), min(R, 1.0)]                       # радиус касания слева и справа, px
    length = bottom = firmly = shifted = deeper = 0.0
    widths, film = [], []
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    last = None

    def lump(i):
        """Остаток вала — комом перед концом: каждое место оседает шапкой крутизны κ по своему объёму."""
        T = bank.sum(0)
        if T.sum() <= 0:
            return
        s = np.minimum((2.0 * T / (np.pi * κ)) ** (1.0 / 3.0), s_max)
        response = _window(canvas, path[i, 0], path[i, 1], d[i], int(np.ceil(R + 2.0 * s.max() + 2.0)))
        if response is not None:
            where, along, across = response
            a = max(a_side)
            for j in np.nonzero(T > 0)[0]:
                weight = _hump(np.hypot(along - (np.sqrt(max(a * a - tj[j] ** 2, 0.0)) + s[j]), across - tj[j]), s[j])
                sv = float(weight.sum())
                if sv <= 0:
                    continue
                for k, with_ in enumerate(all):
                    if bank[k, j] > 0:
                        _lay_in_frame(with_, where, weight / sv * bank[k, j], bank_ks[k, j] / bank[k, j])
        bank[:] = 0.0
        bank_ks[:] = 0.0

    for i, (x, y, p) in enumerate(path):
        if p <= _HERENResponseBrush[4]:
            if last is not None:                        # конец поднят: вал остался, где шёл
                lump(last)
                last = None
            continue
        response = _window(canvas, x, y, d[i], m)
        if response is None:
            continue
        where, along, across = response
        r2 = along * along + across * across
        under = r2 < R * R
        if not under.any():
            continue
        arc = np.where(under, (R - np.sqrt(np.maximum(R * R - r2, 0.0))) / height, np.inf)
        # сверху вниз: сырое слоя конец раздвигает, пока над ним нет твёрдого — сухой доли слоя выше или чужого слоя
        h = [np.asarray(with_.thickness[where], np.float64) for with_ in all]
        open = np.ones(r2.shape, bool)
        wet = []
        for k, with_ in enumerate(all):
            w = _wet(with_, where, 0.0) * open if any(with_ is q for q in layers) else np.zeros(r2.shape)
            wet.append(w)
            open &= (h[k] - w) <= _HERENResponseBrush[5]
        taken = sum(wet)
        S = sum(h)
        grain = getattr(canvas, '_зерно', None)
        if grain is not None:
            S = S + canvas._height_grain(where[0].start, where[0].stop, where[1].start, where[1].stop)
        F = S - taken                                      # твёрдое: грунт с зерном и всё, чего конец не берёт
        if not (taken[under] > 0).any():
            continue
        gap = 0.0 if p >= _HERENResponseBrush[0] else _HERENResponseBrush[1] * (R / pm / 0.5) * (_HERENResponseBrush[0] / p - 1.0)
        z0 = float((F - arc)[under].max()) + gap
        # на твёрдом без краски под ним (грунт) остаётся тон пигмента, вошедшего в грунт, — как под скребком; сухая
        # краска не впитывает: с неё конец сдвигает всё
        tone = np.where(S - taken - (0.0 if grain is None else S - sum(h)) > 1e-4, 0.0, _WithBrushREBResponseBrush[1])
        # плёнку тоньше зерна пигмента твёрдый конец над твёрдым не оставляет: стирает дочиста (кромка борозды резкая)
        above = z0 + arc
        above = np.where(above - F < _HERENResponseBrush[2], F, above)
        the_excess = np.where(under, np.minimum(np.maximum(S - above, 0.0), np.maximum(taken - tone, 0.0)), 0.0)
        has = the_excess > 0
        if not has.any():
            continue
        first = last is None
        last = i
        length += 0.5
        firmly += 0.5 * (p >= _HERENResponseBrush[0])
        iy, ix = np.unravel_index(np.argmin(r2), r2.shape)
        film.append(float(taken[iy, ix] - the_excess[iy, ix]))
        if film[-1] <= tone[iy, ix] + 1e-6:
            bottom += 0.5
        shifted += float(the_excess.sum())
        deeper = max(deeper, float(the_excess.max()))
        for side, sign in ((0, -1.0), (1, 1.0)):
            # радиус касания — по краске, в которую лоб входит (впереди оси): позади — свой валик у бока
            on_side = has & (sign * across > 0) & ((along > 2.0) | first)
            if on_side.any():
                a = float(min(np.abs(across[on_side]).max() + 0.5, R))
                a_side[side] = a if first else 0.5 * (a_side[side] + a)
        widths.append(a_side[0] + a_side[1])
        place = clamp(((across + R) / shb).astype(np.int64), 0, nb - 1)
        remainder = the_excess.copy()
        for k, with_ in enumerate(all):
            remove = np.minimum(remainder, wet[k])
            if not (remove > 0).any():
                continue
            remainder = remainder - remove
            here = remove > 0
            ks = in_ks(with_.color[where][here].astype(np.float64)) * remove[here][:, None]
            bank[k] += np.bincount(place[here], remove[here], nb)
            for q in range(24):
                bank_ks[k, :, q] += np.bincount(place[here], ks[:, q], nb)
            if with_.transmittance is not None:                         # не кроющая плёнка тоньше — светлее (как под ножом)
                share = np.where(h[k] > 1e-6, clamp((h[k] - remove) / np.maximum(h[k], 1e-6), 0.0, 1.0), 1.0)
                with_.transmittance[where] = np.power(np.asarray(with_.transmittance[where], np.float64), share[..., None]).astype(np.float32)
            _remove(with_, where, remove)
        # вал у оси оплывает к соседним местам (от высокого сечения к низкому)
        T = bank.sum(0)
        flow = clamp(spread * (T[:-1] - T[1:]), -0.45 * T[1:], 0.45 * T[:-1])
        from_left = clamp(flow, 0, None) / np.maximum(T[:-1], 1e-12)
        from_right = clamp(-flow, 0, None) / np.maximum(T[1:], 1e-12)
        for m_ in (bank, bank_ks):
            a = m_[:, :-1] * (from_left[:, None] if m_.ndim == 3 else from_left)
            b = m_[:, 1:] * (from_right[:, None] if m_.ndim == 3 else from_right)
            m_[:, :-1] += b - a
            m_[:, 1:] += a - b
        # круглый лоб гонит вал к бокам: место в t от оси за шаг уходит на шаг · |t| / √(a² − t²); за радиусом касания
        # лоб вала не держит вовсе — всё уходит за бок
        spacing_along = clamp(1.0 - np.abs(along) / 1.5, 0.0, 1.0)   # на внешней стороне дуги шаги расходятся: полоса шире шага
        for side, sign in ((0, -1.0), (1, 1.0)):
            a = a_side[side]
            places = [j for j in (range(nb // 2 - 1, -1, -1) if side == 0 else range(nb - nb // 2, nb))]
            if nb % 2 and side == 1:
                places = [nb // 2] + places                     # место на оси (нечётное число мест) — к правому боку
            went, went_ks = np.zeros(L), np.zeros((L, 24))
            for j in places:                                   # от оси к боку: пришедшее из соседа идёт дальше тем же шагом
                bank[:, j] += went
                bank_ks[:, j] += went_ks
                t = abs(tj[j])
                share = 1.0 if t >= a else min(0.5 * t / (np.sqrt(a * a - t * t) * shb + 1e-9), 1.0)
                went, went_ks = bank[:, j] * share, bank_ks[:, j] * share
                bank[:, j] -= went
                bank_ks[:, j] -= went_ks
            V = float(went.sum())
            if V <= 0:
                continue
            # валик привален к боку конца: изнутри его держит круглый бок (просвет под ним над краской — c(t)), снаружи —
            # откос, что держит масло (κ, h на px); сечение A = V / шаг пути — докуда откос дошёл, t_к, по площади
            c = np.where(tt < R, (R - np.sqrt(np.maximum(R * R - tt * tt, 0.0)) - (R - np.sqrt(max(R * R - a * a, 0.0)))) / height, np.inf)
            c = np.where(tt >= a, c, 0.0)
            area = (clamp(np.minimum(c[None, :], κ * (ends[:, None] - tt[None, :])), 0.0, None)).sum(1) * (tt[1] - tt[0])
            t_brush = float(np.interp(V / 0.5, area, ends))
            t = sign * across
            tc = clamp(t, 0.0, R)
            gap = np.where(t < R, (R - np.sqrt(R * R - tc * tc) - (R - np.sqrt(max(R * R - a * a, 0.0)))) / height, np.inf)
            shape = clamp(np.minimum(gap, κ * (t_brush - t)), 0.0, None) * (t >= a) * spacing_along
            sum = float(shape.sum())
            if sum <= 0:                                    # валик ушёл за край холста — с ним и краска
                continue
            for k, with_ in enumerate(all):
                if went[k] > 0:
                    _lay_in_frame(with_, where, shape / sum * went[k], went_ks[k] / went[k])
        x0, y0 = min(x0, x - m), min(y0, y - m)
        x1, y1 = max(x1, x + m + 1), max(y1, y + m + 1)
    if last is not None:
        lump(last)
    if shifted <= 0:
        live = path[:, 2] > _HERENResponseBrush[4]
        return Scratched(rectangle=None, length=0.0, to_bottom=0.0, left=None, on_firmly=0.0, width=0.0, deeper=0.0, shifted=0.0, mm3=0.0,
                           why=('нажим 0 по всему ходу: конец над холстом' if not live.any() else
                                   'под концом сырой краски нет или нажим слаб: конец прошёл над ней (зазор — по нажиму, '
                                   f'на твёрдое конец садится с нажима {_HERENResponseBrush[0]:g}); сухое и схватившееся черенок не берёт'))
    return Scratched(rectangle=(int(np.floor(x0)), int(np.floor(y0)), int(np.ceil(x1)), int(np.ceil(y1))),
                       length=length, to_bottom=bottom / max(length, 1e-9), left=float(np.median(film)) * _TEN[0] / 8.0, on_firmly=firmly / max(length, 1e-9),
                       width=float(np.median(widths)),
                       deeper=deeper * _TEN[0] / 8.0, shifted=shifted, mm3=shifted * (_TEN[0] / 8.0) / (pm * pm), why=None)


def remove(canvas, points, tool, width, seed, layer=None):
    """Снять сырую краску по ходу (К8.4). инструмент — 'тряпка', 'скребок' или 'черенок' (К-14а: краску не снимает,
    а раздвигает в валики; ответ — `Процарапано`); слой — имя слоя или None (все)."""
    if tool not in AndNWithTRAtMENTY:
        raise ValueError(f'снятие: инструмент — {", ".join(AndNWithTRAtMENTY)}, не {tool!r}')
    if not width > 0:
        raise ValueError('снятие: ширина — больше нуля')
    layers = [with_ for name, with_ in canvas.layers.items() if layer is None or name == layer][::-1]     # сверху вниз
    if not layers:
        return
    canvas._to_records(None)
    if tool == 'черенок':                              # чужой слой (при `слой=`) для конца — твёрдое
        return handle_end(canvas, points, float(width), layers, list(canvas.layers.values())[::-1])
    (rag if tool == 'тряпка' else scraper)(canvas, points, float(width), seed, layers)
