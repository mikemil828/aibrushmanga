# READING VIEW of движок/кисть/срастание.py with names in English. Not runnable; edit the original.
"""Срастание соседних мазков (версия 3, пункт 6а): просвет между прядями открывается и закрывается (затеканием щели,
пряди и борозды на месте); сырая краска,
что не поместилась на кончиках, уходит валом за кромку следа. Протяжка сырой полосы по ходу — `Ноша`: подобранное лежит
на кончике снаружи и сходит первым (`_СРАСТ_ТЯНЕТ`)."""
import numpy as np

from .fast import clamp, surround, by_samples, by_ridge, between_steps, by_hairs

from .paint import from_ks, in_ks, PResponseLResponseWith
from . import numbers as H
from .numbers import _ShBrush
from .strands import _noise


def _bridges(rng, N, m, ds, slit, bridge):
    """(N, m), 0–1: вдоль хода щель живёт `щель` px (от и до), потом её на `мост` px затягивает краска — и снова.
    У каждого столбца свой счёт; края перемычки плавные (пара px)."""
    r = np.zeros((N, m))
    s = np.arange(N) * ds
    for j in range(m):
        x = -rng.uniform(0.0, slit[1])
        while x < s[-1]:
            x += rng.uniform(*slit)
            d = rng.uniform(*bridge)
            r[:, j] = np.maximum(r[:, j], clamp(np.minimum(s - x, x + d - s) / 2.0 + 0.5, 0.0, 1.0))
            x += d
    return r


class Gap:
    """Просвет между прядями открывается и закрывается — краской, а не местом пряди: пряди и борозды стоят где стояли.
    Тонкая сырая краска двух соседних прядей то смыкается через щель плёнкой (щель затекла: краска взята у крайних
    гребней этих прядей, цвет — их же), то стянута каждая к своей пряди, и края у щели кладут тоньше (щель открыта) —
    щель живёт от половины до полутора ширин кисти, потом её затягивает перемычка. То же у бока кисти: отошедшую внутрь крайнюю прядь краска возвращает на линию кисти,
    не дальше. Густая краска держит пряди обоймой: у неё этого нет."""

    both = False   # К8б: плёнка щели — не толще, чем кладёт беднейшая из двух прядей (у кисти, которую раскрывает рука)

    def __init__(self, rng, with_, tack, leaving, thickly, N, ds, W):
        slit, bridge, opening_up, threshold, self.share, self.overlapping = H._GROWING_TOGETHER_GAP
        self.has = with_.ns >= 2 and thickly < 1.0 and (self.share > 0.0 or opening_up > 0.0)
        if not self.has:
            return
        ns = with_.ns
        self.left = np.array([(with_.center - with_.half_width)[with_.strand == j].min() for j in range(ns)])
        self.right = np.array([(with_.center + with_.half_width)[with_.strand == j].max() for j in range(ns)])
        self.kr_l = np.array([np.nonzero(with_.strand == j)[0].min() for j in range(ns)])      # крайние гребни пряди
        self.kr_p = np.array([np.nonzero(with_.strand == j)[0].max() for j in range(ns)])
        axis = with_.axis_strands
        # на сколько бок кисти отошёл внутрь от её линии, px (слева, справа)
        self.moved_away = np.stack([clamp((axis[0] + (self.left[0] - axis[0]) * tack[:, 0] + leaving[:, 0]) - self.left[0], 0.0, None),
                                clamp(self.right[-1] - (axis[-1] + (self.right[-1] - axis[-1]) * tack[:, -1] + leaving[:, -1]), 0.0, None)], axis=1)
        # ширина плёнки, px: щель между прядями (как они встали) и нахлёст на оба края; у бока — от края до линии кисти
        gap = clamp((axis[None, 1:] + (self.left[1:] - axis[1:])[None, :] * tack[:, 1:] + leaving[:, 1:])
                        - (axis[None, :-1] + (self.right[:-1] - axis[:-1])[None, :] * tack[:, :-1] + leaving[:, :-1]), 0.0, None)
        self.width = np.concatenate([gap + 2.0 * self.overlapping, self.moved_away + self.overlapping], axis=1)
        thin = 1.0 - thickly
        # ns − 1 щелей и два бока кисти. Затягивает ту щель, что есть: сомкнутым прядям и вставшему на линию боку нечего
        has_slit = np.concatenate([clamp(gap / 0.4, 0.0, 1.0), clamp(self.moved_away / 0.3, 0.0, 1.0)], axis=1)
        self.seeped = _bridges(rng, N, ns + 1, ds, (slit[0] * W, slit[1] * W), (max(bridge[0] * W, 3.0), max(bridge[1] * W, 5.0))) * has_slit * thin
        # между перемычками края прядей то и дело стянуты каждый к своей пряди — щель шире
        self.open = clamp((-_noise(rng, N, ns + 1, max(0.4 * W, 6.0) / ds) - threshold) / 0.6, 0.0, 1.0) * (1.0 - self.seeped) * thin
        opening_up = max(opening_up, 1e-6)
        # края у открытой щели кладут тоньше: на `раскрытие` px от границы пряди
        pr_m = with_.strand[with_.ridge]
        at_p = clamp(1.0 - (self.right[pr_m] - with_.u) / opening_up, 0.0, 1.0)
        at_l = clamp(1.0 - (with_.u - self.left[pr_m]) / opening_up, 0.0, 1.0)
        shch_p = np.where(pr_m < ns - 1, pr_m, ns)                    # номер щели справа от отсчёта (у последней пряди — правый бок)
        shch_l = np.where(pr_m > 0, pr_m - 1, ns - 1)                 # слева (у первой — левый бок)
        self.less_thick = clamp(1.0 - self.open[:, shch_p] * at_p[None, :] - self.open[:, shch_l] * at_l[None, :], 0.0, 1.0)

    def open(self, D, with_, width, tau_ds, touched=None):
        """Отсчёты следа D (N, nm): у открытой щели тоньше; у затёкшей крайние гребни отдают краску плёнке.
        К6, "ridges": задет (N, nm) — насколько выборку профиля задел верх лежащего гребня (`зерно.Рвёт.заполнен`). Верх
        снимает краску с ворса куском, а кусок шире щели между прядями: где верх задели края обеих соседних прядей, щель
        под ним закрыта краской этих краёв, той же толщины — кусок на гребне щелью кисти поперёк не разрезан."""
        if not self.has:
            return D
        piece = None
        if touched is not None:
            left, right = np.searchsorted(with_.ridge, self.kr_p[:-1], side='right') - 1, np.searchsorted(with_.ridge, self.kr_l[1:])
            piece = np.concatenate([np.minimum(touched[:, left], touched[:, right]), np.zeros((len(touched), 2))], axis=1)
            self.seeped = np.maximum(self.seeped, piece)
        D = D * self.less_thick
        g = with_.ridge
        n = np.maximum(np.bincount(g, minlength=with_.nr), 1)
        sum = (np.add.reduceat(D, np.searchsorted(g, np.arange(with_.nr)), axis=1) if np.all(np.diff(g) >= 0) and len(np.unique(g)) == with_.nr
                 else np.stack([D[:, g == q].sum(axis=1) for q in range(with_.nr)], axis=1))   # (N, nr)
        a = np.concatenate([self.kr_p[:-1], [self.kr_l[0], self.kr_p[-1]]])               # гребни по сторонам щели; у бока — один
        b = np.concatenate([self.kr_l[1:], [self.kr_l[0], self.kr_p[-1]]])
        # плёнка на px ширины щели: тонкая, едва кроет (`доля` толщин полного укрытия) — и не толще, чем кладут соседи
        # (на отсчёт — среднее двух крайних гребней): сухие пряди щель не затянут, борозда на дне щели остаётся
        neighbors = np.minimum(sum[:, a] / n[a], sum[:, b] / n[b]) if self.both else 0.5 * (sum[:, a] / n[a] + sum[:, b] / n[b])
        # (К8б, `оба`: щель затягивают края обеих прядей; прядь, которую нажим ещё не донёс до холста, края не даёт, и
        # плёнка от соседки к ней не идёт — иначе у недожатого края вставала бледная плашка)
        self.on_px = self.seeped * np.minimum(self.share * tau_ds, neighbors / width)
        if piece is not None:
            self.on_px = np.maximum(self.on_px, piece * 0.5 * (sum[:, a] / n[a] + sum[:, b] / n[b]) / width)
        self.a, self.b = a, b
        # что ушло плёнкой, крайние гребни не доложили (поровну с двух сторон щели; у бока — один гребень)
        taken = 0.5 * self.on_px * self.width
        share = np.zeros_like(sum)
        np.add.at(share, (slice(None), a), taken / np.maximum(sum[:, a], 1e-12))
        np.add.at(share, (slice(None), b), taken / np.maximum(sum[:, b], 1e-12))
        return D * (1.0 - clamp(share, 0.0, 0.6))[:, g]

    def in_raster(self, xs, ys, weights, with_, D, on_canvas, lag, tip, colors, i0, ax, ay, spread, bx0, by0, spread0=None, along=None, askew=None):
        """Дописать отсчёты плёнки в затёкших щелях; сколько легло плёнкой, столько крайние гребни не доложили.
        Щель — между следами двух прядей, поперёк их хода. Кончики прядей отстают от кисти каждый на своё (`вдоль(отст,
        конч)`, px вдоль ворса), и у кисти, чья ось не поперёк ворса (повёрнута рукой, К1; дуга), разница отставаний ложится
        на ось кисти долей `вкось` (N,) — косинус угла между осью и ворсом. Это не ширина щели: плёнка шла бы по оси
        кисти на всю разницу отставаний — ровной плашкой поперёк прядей в хвосте схода, где пряди отстают врозь. Из
        расстояния по оси она вычитается; плёнка идёт от края пряди по оси кисти до следа соседней."""
        if not self.has:
            return xs, ys, weights
        ns = with_.ns
        # границы прядей на холсте: правый край пряди j и левый край пряди j+1; у боков — край кисти
        u = np.concatenate([self.right[:-1], self.left[1:], [self.left[0], self.right[-1]]])
        pr = np.concatenate([np.arange(ns - 1), np.arange(1, ns), [0, ns - 1]])
        gr = np.concatenate([self.kr_p[:-1], self.kr_l[1:], [self.kr_l[0], self.kr_p[-1]]])
        X, Y = on_canvas(u, pr, lag[gr], tip[gr])
        m = ns - 1
        apart = None if along is None or m < 1 else (lambda vd: (vd[:, m:2 * m] - vd[:, :m]) * askew[:, None])(along(lag[gr], tip[gr]))   # (N, m): доля разницы отставаний краёв щели на оси кисти
        Xa, Ya, Xb, Yb = X[:, :m], Y[:, :m], X[:, m:2 * m], Y[:, m:2 * m]
        # п. 18в: число отсчётов плёнки в щели считается по ходу, каким он был бы без хвоста плашмя (`расп0`): иначе хвост
        # менял бы его для всего хода, и плёнка перемычек в теле ложилась бы на 1e-4 иначе
        X0, Y0 = (X, Y) if spread0 is None else on_canvas(u, pr, lag[gr], tip[gr], was=True)
        spacing = 0.5
        extra = ([], [], [], [])
        for q in range(ns + 1):
            if q < m:
                x0, y0 = Xa[:, q], Ya[:, q]
                gap = clamp((Xb[:, q] - x0) * ax + (Yb[:, q] - y0) * ay - (0.0 if apart is None else apart[:, q]), 0.0, None)
                from_, to = -self.overlapping * np.ones(len(gap)), gap + self.overlapping
            else:
                # бок кисти: от отошедшего края до линии кисти, внутрь — на тот же нахлёст
                sign = -1.0 if q == m else 1.0
                x0, y0 = X[:, 2 * m + (q - m)], Y[:, 2 * m + (q - m)]
                outside = self.moved_away[:, q - m] * spread
                from_, to = (-outside, self.overlapping * np.ones(len(outside))) if sign < 0 else (-self.overlapping * np.ones(len(outside)), outside)
            ii = np.nonzero(self.on_px[:, q] > 0.0)[0]
            if len(ii) == 0:
                continue
            wider = (to - from_) if spread0 is None else ((clamp((X0[:, m + q] - X0[:, q]) * ax + (Y0[:, m + q] - Y0[:, q]) * ay - (0.0 if apart is None else apart[:, q]), 0.0, None) + 2.0 * self.overlapping) if q < m else self.moved_away[:, q - m] * spread0 + self.overlapping)
            k = int(np.ceil(float(wider[ii].max()) / spacing)) + 1
            t = (np.arange(k) + 0.5) / k
            position = from_[ii, None] + (to - from_)[ii, None] * t[None, :]                            # px поперёк от края пряди
            d = np.broadcast_to((self.on_px[ii, q] * (to - from_)[ii] / k)[:, None], position.shape)
            ts = 0.5 * (colors[i0[ii], self.a[q]] + colors[i0[ii], self.b[q]])
            extra[0].append((x0[ii, None] + position * ax[ii, None] - bx0).ravel())
            extra[1].append((y0[ii, None] + position * ay[ii, None] - by0).ravel())
            extra[2].append(d.ravel())
            extra[3].append(np.repeat(ts, k, axis=0))
        if not extra[0]:
            return xs, ys, weights
        d, ts = np.concatenate(extra[2]), np.concatenate(extra[3])
        own = [d, d * ts[:, 0], d * ts[:, 1], d * ts[:, 2]]
        weights = [np.concatenate([w, own[i] if i < 4 else np.zeros(len(d))]) for i, w in enumerate(weights)]
        return np.concatenate([xs, np.concatenate(extra[0])]), np.concatenate([ys, np.concatenate(extra[1])]), weights


class Streams:
    """Струи лежащей краски в теле мазка по сырому (6а-1). Гребень кладёт свою краску; волосок, что идёт кончиком
    в лежащей сырой краске, своей под собой не оставляет: в его борозде остаётся цвет лежащей — струёй вдоль ворса.
    Какой волосок идёт в лежащей, а какой по подушке своей краски — от его жёсткости (на весь мазок) и медленно гуляет
    вдоль хода; в воздухе (проскок) волосок ничего не скребёт. Подобранное кончиком (`Ноша`) сходит в ту же струю,
    а не красит гребень. Только у тонкой краски: пастозная кроет телом."""

    def __init__(self, rng, with_, furrow, N, ds, W, thickly):
        threshold, transition, stiff, slow, length, wider = H._GROWING_TOGETHER_STREAM[:6]
        self.has = with_.leading is not None and furrow is not None and thickly < 1.0 and with_.nr >= 3
        if not self.has:
            return
        nv = with_.nr - 1
        z = (clamp(with_.presses, 0.75, 1.25) - 1.0) / 0.25
        strength = clamp((stiff * z[None, :] + slow * _noise(rng, N, nv, length * W / ds) - threshold) / transition + 0.5, 0.0, 1.0)
        strength = strength * strength * (3.0 - 2.0 * strength) * clamp(furrow[0] / 0.3, 0.0, 1.0)      # в воздухе глубина борозды 0
        strength[:, 0] = strength[:, -1] = 0.0                            # кромку следа ведут бок и край, не струя
        t = clamp(with_.d_leading[None, :] / (with_.half_furrow[None, :] * furrow[1] * wider)[:, with_.leading], 0.0, 1.0)
        self.L = strength[:, with_.leading] * (1.0 - t * t * (3.0 - 2.0 * t)) * (1.0 - thickly)        # (N, nm): 1 — струя, 0 — гребень
        n = np.maximum(np.bincount(with_.ridge, minlength=with_.nr), 1)
        self.share = np.stack([np.bincount(with_.ridge, weights=r, minlength=with_.nr) for r in self.L]) / n[None, :]   # (N, nr)
        self.equal = max(thickly, H._GROWING_TOGETHER_STREAM[8])                                        # такая доля прежнего закона (ноша красит гребень) остаётся

    @staticmethod
    def rim(thickly):
        """Во сколько раз мельче «кромка на сырой краске»: у тонкой краски она шла во всю кисть, и гребни своей не давали."""
        return thickly + (1.0 - thickly) * H._GROWING_TOGETHER_STREAM[6] if H._GROWING_TOGETHER_STREAM_ON else 1.0

    @staticmethod
    def for_(rng, with_, furrow, N, ds, W, thickly, bank, ic):
        """Струи мазка или None: выключено, по сухому, мастихин, пастозная краска."""
        if not H._GROWING_TOGETHER_STREAM_ON or bank is None or bank.carry is None:
            return None
        streams = Streams(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]), with_, furrow, N, ds, W, thickly)
        if not streams.has:
            return None
        streams.steps(ic, with_.nr)
        return streams

    def holds(self, c, takes, carry, lays):
        """Сколько лежащей краски гребень снимает с холста за шаг. В струе лежащая остаётся на месте, меж струй кончик
        снимает её, как снимал: сколько берёт, решает место на кончике (`Вал.перелив`), а не струи. Струи решают, куда
        снятое идёт: ношей — не больше, чем сойдёт в струи гребня за срок ноши (`кладёт` — что гребень кладёт за шаг;
        `self.место`), остальное мешается в каплю (грязная кисть), на холсте не остаётся и за кромку не идёт."""
        self.place = np.maximum(carry.steps * self.share_brush[c] * lays - carry.volume, 0.0)
        return takes * (1.0 - self.share_brush[c])

    def in_film(self, d, ts, i0, i1, f0, f1, g, ii, jj, edge):
        """Отсчёты плёнки: в струе под волоском своей краски нет — там лежащая (на месте) или подобранная (ношей)."""
        no_own, carry = self.on_path(i0, i1, f0, f1, g)
        d_n = d * by_samples(carry, ii, jj)
        self.exchange = d_n                                          # сколько ноши легло в струю, столько лежащей волосок там снял
        d = d * (1.0 - by_samples(no_own, ii, jj))
        # цвет ноши между двумя шагами переноса — с весом того, сколько её на каждом шаге сошло: у шага без ноши цвета нет
        # (её К-С — нули, а это белое), и без веса он подмешивал белое в начало и конец каждой струи ноши (Дв103, Д1)
        ts_n, in0, in1 = from_ks(self.ks), by_hairs(self.carry[i0] * f0, g, ii, jj)[:, None], by_hairs(self.carry[i1] * f1, g, ii, jj)[:, None]
        ts_n = (by_ridge(ts_n, i0, g, ii, jj) * in0 + by_ridge(ts_n, i1, g, ii, jj) * in1) / np.maximum(in0 + in1, 1e-300)
        d = d if edge is None else d * edge.lays(i0, i1, f0, f1, g, ii, jj)
        edge_in = [] if edge is None else [d * edge.share(i0, i1, f0, f1, g, ii, jj)]
        return d + d_n, (ts * d[:, None] + ts_n * d_n[:, None]) / np.maximum(d + d_n, 1e-12)[:, None], edge, edge_in

    def removed(self, removed, i0, i1, f0, f1, g, ii, jj, edge):
        """Съём по отсчётам: в струе лежащая не снята, меж струй — снята та же всего; у бока — по ноше волоска
        (`Край.с_тела`); в струе ноша ложится взамен лежащей (`обмен`)."""
        removed = removed / np.maximum(1.0 - between_steps(self.share_brush, i0, i1, f0, f1, g, ii, jj), 0.05) * (1.0 - by_samples(self.L, ii, jj))
        return (removed if edge is None else edge.with_body(removed, ii, jj)) + self.exchange

    @staticmethod
    def level(h, touches, blur):
        """Лежащая сырая краска под кистью после съёма: волосок в струе идёт в ней и её разглаживает — струя остаётся
        цветом, а не гребнем толщины (кисть сняла меж струй, в струе не сняла). Объём тот же."""
        return h + (blur(h, 2) - h) * touches

    @staticmethod
    def under_brush(touches):
        """Щель между прядями уже 2 px лежит под кистью: волоски идут по обе её стороны и сырую краску в ней двигают
        так же, как под собой. Без этого по сырому в щели стояла нетронутая стенка лежащей краски в 1 px (по бокам
        снято на четверть плёнки) и под углом к растру дробилась в пунктир вдоль гребня (6а-2, Движок 24).
        Наружу, за кромку следа, не идёт: сначала наибольшее по 3×3, потом наименьшее."""
        def window(a, f, edge):
            p = surround(a, 1, edge)
            r = p[1:-1, 1:-1]
            for dy in range(3):
                for dx in range(3):
                    r = f(r, p[dy:dy + a.shape[0], dx:dx + a.shape[1]])
            return r
        return np.maximum(touches, window(window(touches, np.maximum, 0.0), np.minimum, 0.0))

    def steps(self, ic, nr):
        self.share_brush = self.share[ic]
        self.in_wet = np.zeros((len(ic), nr))                    # струя открыта: волосок в лежащей сырой краске или несёт её
        self.carry = np.zeros((len(ic), nr))                      # такая доля струи заполнена подобранной краской
        self.ks = np.zeros((len(ic), nr, 2 * PResponseLResponseWith))

    def on_path(self, i0, i1, f0, f1, g):
        """(N, nm): доля своей краски, что волосок не оставил; доля плёнки, что легла в струю подобранной."""
        a = (self.in_wet[i0] * f0 + self.in_wet[i1] * f1)[:, g]
        g = (self.carry[i0] * f0 + self.carry[i1] * f1)[:, g]
        return self.L * a, self.L * g


class Edge:
    """Края мазка по сырому (6а-2): вход, бок и отрыв без дощечки. Кончик, севший в сырую краску, первые доли ширины
    кисти несёт её на себе: что он кладёт, меняется местами с лежащей (плёнка на этом месте — лежащая краска, своя
    осталась на кончике), свой тон набирается по ходу — у каждого гребня на своей длине. Крайние волоски несут мало
    краски и прижаты слабее: они идут в лежащей краске и двигают её, а своей поверх не кладут. Гребень, который рука
    поднимает, идёт в лежащей с того нажима, что у него остался сверх порога касания, — у каждого гребня со своего.
    Объём слоя тот же: сколько плёнки заменено лежащей, столько лежащей с этого места снято (`укладка.py`, «края по
    сырому»). Только со струями; по грунту и по сухому края прежние; пастозная на отрыве и у бока кроет телом."""

    def __init__(self, rng, with_, N, Nc, ds, W, thickly, spacing_brush, touch_brush, wet_films, share_pressure, reserve_g, ic):
        """`кас_к` (Nc, nr) — касание гребней по шагам переноса (шаг `шаг_к` px); `сыр_плёнок` — лежащая сырая краска
        под гребнем в толщинах плёнки мазка; `доля_нажима` (N,) и `запас_г` (N, nr) — нажим руки и запас нажима гребня
        сверх порога касания по шагам пути; `ic` — шаги пути, на которых считается перенос."""
        vkh, side, end, sweep = H._GROWING_TOGETHER_EDGE[:4]
        by_canvas = spacing_brush * np.cumsum(touch_brush, axis=0)             # сколько гребень прошёл, касаясь холста, px
        wet = clamp(wet_films / 0.06, 0.0, 1.0)              # 1 — под гребнем сырая краска
        # запас нажима гребня на последнем подъёме руки против того, что был в теле: 1 — до подъёма, 0 — гребень отошёл
        lift = np.arange(N) > N - 1 - int(np.argmax((share_pressure >= 0.97)[::-1]))
        reserve = np.where(lift[:, None], reserve_g / np.maximum(reserve_g.max(axis=0), 1e-6), 1.0)[ic]
        nr = with_.nr
        thin = 1.0 - thickly
        # вход: гребень набирает свой тон за 0,35–0,85 ширины кисти от своего первого касания; короткий удар —
        # не дольше трети своего хода по холсту: иначе он весь остаётся в лежащей и своего тона не доносит
        L = np.minimum(W * rng.uniform(*vkh, nr), sweep * by_canvas[-1])
        g = clamp(1.0 - by_canvas / np.maximum(L, 1e-6)[None, :], 0.0, 1.0)
        # отрыв: руку поднимают, нажим гребня падает к порогу касания — кончик уже не давит свою краску в холст,
        # а идёт в лежащей; с какой доли своего запаса нажима (`запас`: 1 — до подъёма, 0 — гребень отошёл) —
        # у каждого гребня своё: концы вразнобой, а не чертой
        g2 = clamp(1.0 - reserve / rng.uniform(*end, nr)[None, :], 0.0, 1.0)
        g2 = g2 * g2 * (3.0 - 2.0 * g2) * (by_canvas > 0.0) * float(clamp(2.0 * thin, 0.0, 1.0))
        self.ridge = 1.0 - (1.0 - g * thin) * (1.0 - g2)                              # (Nc, nr); сырость — в `доля`
        # отрыв, рельеф (Движок 24): в теле гребень продавливает подушку своей краски, снимает лежащую и кладёт свою;
        # на подъёме руки он продавливать перестаёт (`втир` падает) — и кладёт своей настолько же меньше: слабо прижатый
        # кончик на сырой краске своей не оставляет. Без этого последние 10 px гребень лежащую уже не снимал, а свою
        # клал полной плёнкой поверх: наплыв +0,4 плёнки с обрывом после канавки тела — крючок и тёмное пятно света.
        # Гребень, что и в теле лежал сверху (втирания не было), кладёт как клал
        self.lift = (reserve < 1.0) * float(clamp(2.0 * thin, 0.0, 1.0))
        # вход, рельеф (Движок 24): кончик, севший в сырую краску, уже несёт её на себе и больше не берёт — снимать
        # лежащую он начинает по мере того, как набирает свой тон (та же длина у гребня). Без этого канавка тела
        # начиналась уступом на линии первого касания: тёмная черта света поперёк входа
        self.takes = 1.0 - g * thin * wet
        # бок: крайние волоски несут до пятой части краски (тот же закон, что у капли: `укладка.py`, «полная») и
        # кладут свою, только пока прижаты; порог ноши, ниже которого волосок идёт в лежащей, гуляет вдоль хода
        m = float(clamp(W / 40.0, 0.6, 1.0))
        self.threshold = 0.2 + m * (side[0] - 0.2 + (side[1] - side[0]) * clamp(0.5 + 0.6 * _noise(rng, N, 2, max(0.5 * W, 8.0) / ds), 0.0, 1.0))   # (N, 2)
        # п. 13 (хвост 11ж): у кисти уже 16 px бок мельче вместе с шириной (как кромка по сырому, `укладка._узко`): иначе кисть
        # 6 px — вся «бок», в три гребня клала нить в 2 px, а в два гребня (без струй и краёв) — брусок, по сиду
        self.carries = H.carry_at_edges((with_.half_width - np.abs(with_.u)) / min(W / H._ResponseWithTRAYa[0], 1.0))    # (nm,)
        self.st = (with_.u > 0).astype(np.intp)
        self.wet = wet
        self.thin = thin

    @staticmethod
    def for_(has_paint, rng, *a):
        """Края мазка или None: по голому грунту (без струй и по сухому сюда не доходит)."""
        if not has_paint:
            return None
        return Edge(np.random.default_rng(rng.bit_generator.seed_seq.spawn(1)[0]), *a)

    @staticmethod
    def on_place(asked, V, h, mobile, ts_new, C):
        """По пикселям: столько своей плёнки заменено лежащей краской этого места (не больше, чем её там сырой);
        столько же лежащей снято — объём слоя тот же. Ответ — цвет плёнки и толщина лежащего."""
        changes = np.minimum(np.minimum(asked, V), np.maximum(h, 0.0) * mobile * 0.9)
        m = changes > 1e-7
        if m.any():
            g = (changes[m] / np.maximum(V[m], 1e-9))[:, None]
            ts_new[m] = from_ks(in_ks(ts_new[m]) * (1.0 - g) + in_ks(C[m]) * g)
            h = h - changes
        return ts_new, h

    def with_body(self, removed, ii, jj):
        """Бок, рельеф (Движок 25). Съём лежащей краски по отсчётам: крайний волосок несёт до пятой части ноши и прижат
        слабо — лежащую он не скребёт, а снимает её столько, сколько несёт (тот же закон `ноша_у_края`); что кисть
        подняла, она подняла телом: съём каждого шага пути тот же, поперёк кисти он делится по ноше волосков.
        Было: крайние 3 px клали 0,3–0,5 плёнки своей, а снимали 2–2,8 лежащей, как полные, — ров под кромкой глубже
        тела, и вал, что из него же вышел, стоял за кромкой стенкой в 1 px."""
        w = self.carries[jj] ** self.thin
        n = int(ii.max()) + 1
        return removed * w * (np.bincount(ii, weights=removed, minlength=n) / np.maximum(np.bincount(ii, weights=removed * w, minlength=n), 1e-12))[ii]

    def brush_film(self, touch, rub_in, pointed=1.0):
        """После переноса краски: касание по отсчётам; `втир` (Nc, nr) — как гребень продавливал подушку своей краски;
        `острая` — доля острого кончика у кисти (п. 12: 1 — ворс собран в остриё, подушки под ним нет, бок идёт в лежащей всегда)."""
        floor = H._GROWING_TOGETHER_EDGE[6]
        self.touch, self.reaches = touch, pointed + (1.0 - pointed) * (floor + (1.0 - floor) * clamp(rub_in / H._GROWING_TOGETHER_EDGE[5], 0.0, 1.0))
        c0 = np.maximum(np.argmax(self.lift > 0.0, axis=0) - 1, 0)
        self.melts = clamp(rub_in[c0, np.arange(rub_in.shape[1])][None, :] - rub_in, 0.0, 1.0) * self.lift
        return self

    def lays(self, i0, i1, f0, f1, g, ii, jj):
        """По отсчётам: какую долю своей плёнки гребень на подъёме руки по сырому ещё кладёт."""
        return 1.0 - by_hairs((self.melts[i0] * f0 + self.melts[i1] * f1) * (self.wet[i0] * f0 + self.wet[i1] * f1), g, ii, jj)

    def share(self, i0, i1, f0, f1, g, ii, jj):
        """По отсчётам: какая доля своей плёнки заменена лежащей краской на месте."""
        gr = between_steps(self.ridge, i0, i1, f0, f1, g, ii, jj)
        wet = between_steps(self.wet, i0, i1, f0, f1, g, ii, jj)
        side = clamp((self.threshold[ii, self.st[jj]] - self.carries[jj] * self.touch) / H._GROWING_TOGETHER_EDGE[4] + 0.5, 0.0, 1.0) * self.thin
        side = side * between_steps(self.reaches, i0, i1, f0, f1, g, ii, jj)   # п. 12: у щетины плашмя крайний волосок идёт в лежащей, насколько кисть до неё достаёт; лежит на подушке своей краски — бок ложится сверху своим цветом (подрезка)
        return (1.0 - (1.0 - gr) * (1.0 - side)) * wet


class Carry:
    """Протяжка: подобранная сырая краска ложится на кончик снаружи и сходит первой — раньше своей, что под ней.
    Ход стирает ношу с кончика за `Ч._СРАСТ_ТЯНЕТ[2]` ширин кисти (сколько бы её ни было: толстую полосу след несёт
    гуще, а не дальше), и гребень кладёт её вместо своей краски — не больше доли `[3]` того, что кладёт: тонкое место следа одной ношей
    не красится; кончилась — идёт своя. Так мазок тянет сырую полосу
    на ширину-полторы кисти, а не красит ею весь остаток хода."""

    def __init__(self, nr, steps):
        self.volume = np.zeros(nr)
        self.ks = np.zeros((nr, 2 * PResponseLResponseWith))
        self.behind_spacing = np.zeros(nr)
        self.steps = steps                                       # шагов переноса, за которые ноша сходит
        self.up = np.zeros(nr)                                # 6а-1: подобранное, ушедшее вверх по ворсу, в запас кисти
        self.up_ks = np.zeros((nr, 2 * PResponseLResponseWith))

    def take(self, foreign, ks_canvas, place=None, in_drop=1.0):
        """Подобранное за шаг: доля `Ч._СРАСТ_ТЯНЕТ[0]` — в ношу; ответ — что уходит в каплю, как раньше.
        6а-1 (струи): ношей — не больше `место` (что сойдёт в струи). Не вместившееся у тощей кисти мешается в каплю
        (доля `в_каплю`: на кончике место есть), у полной — уходит вверх по ворсу и мешается с запасом (`из_запаса`)."""
        in_carry = H._GROWING_TOGETHER_PULLS[0] * foreign if place is None else np.minimum(H._GROWING_TOGETHER_PULLS[0] * foreign, place)
        new = self.volume + in_carry
        self.ks = (self.ks * self.volume[:, None] + ks_canvas * in_carry[:, None]) / np.maximum(new, 1e-12)[:, None]
        self.volume = new
        self.behind_spacing = np.where(in_carry > 0.0, new / self.steps, self.behind_spacing)
        if place is None:
            return foreign - in_carry
        up = (1.0 - in_drop) * (H._GROWING_TOGETHER_PULLS[0] * foreign - in_carry)
        new = self.up + up
        self.up_ks = (self.up_ks * self.up[:, None] + ks_canvas * up[:, None]) / np.maximum(new, 1e-12)[:, None]
        self.up = new
        return foreign - in_carry - up

    def from_reserve(self, own, ks_own, reserve):
        """6а-1: своя краска идёт из запаса кисти вместе с тем, что ушло в него с холста (в их доле). Ответ — её цвет."""
        g = self.up / np.maximum(self.up + reserve, 1e-12)
        self.up = self.up - np.minimum(g * own, self.up)
        return ks_own * (1.0 - g[:, None]) + self.up_ks * g[:, None]

    def in_streams(self, c, agrees, streams, wet):
        """6а-1, тонкая краска: подобранное сходит в струи под волосками (сколько их у гребня), свою краску гребень
        в струе не оставляет. `сыро` — волосок идёт в лежащей сырой краске (0–1). Ответ — что гребень кладёт своей."""
        place = streams.share_brush[c] * agrees
        from_carry = np.minimum(np.minimum(self.volume, self.behind_spacing), place)
        self.volume = self.volume - from_carry
        g = from_carry / np.maximum(place, 1e-12)
        streams.carry[c] = g
        streams.in_wet[c] = np.maximum(wet, g)
        streams.ks[c] = self.ks
        # что не сошло в струи за свой срок (струя закрылась, гребень кладёт мало), остаётся на кончике под своей
        # краской: мешается в каплю (`в_каплю`), за кромку не уходит
        self.erased = np.minimum(self.volume, np.maximum(self.behind_spacing - from_carry, 0.0))
        self.volume = self.volume - self.erased
        return agrees * (1.0 - streams.share_brush[c] * streams.in_wet[c])

    def in_drop(self, drop, ks_drops):
        """6а-1: несошедшая ноша вмешана в каплю кончика (объём и цвет те же). Ответ — капля и её краска."""
        new = drop + self.erased
        ks = (ks_drops * drop[:, None] + self.ks * self.erased[:, None]) / np.maximum(new, 1e-12)[:, None]
        self.erased = np.zeros_like(self.erased)
        return new, np.where((self.volume + new > 1e-9)[:, None] & (new > 1e-9)[:, None], ks, ks_drops)

    def fade_out(self, agrees, ks_drops, share=1.0):
        """Гребень кладёт `уходит`: сколько может — ношей. Ответ: цвет того, что легло, и сколько из него взято у капли."""
        from_carry = np.minimum(np.minimum(self.volume, self.behind_spacing), H._GROWING_TOGETHER_PULLS[3] * share * agrees)
        self.volume = self.volume - from_carry
        g = (from_carry / np.maximum(agrees, 1e-12))[:, None]
        return ks_drops * (1.0 - g) + self.ks * g, agrees - from_carry


class Bank:
    """Сырая краска, которую кисть подняла и не унесла: кончик держит каплю не больше `Ч._СРАСТ_ВАЛ[0]` полной, лишнее
    обтекает кисть и ложится валом сразу за кромкой следа — с той стороны, к которой гребень ближе. Объём — тот,
    что снят с холста; цвет — лежавшей краски."""

    def __init__(self, Nc, with_, lean, steps_on_widths):
        self.carry = Carry(with_.nr, H._GROWING_TOGETHER_PULLS[2] * steps_on_widths) if H._GROWING_TOGETHER_PULLS[0] > 0 else None
        self.holds = H._GROWING_TOGETHER_BANK[0] * max(1.0, lean)         # тощая кисть (своей краски мало) чужой держит больше: место есть
        self.in_drop = float(clamp(lean - 1.0, 0.0, 1.0))    # 6а-1: у тощей кисти не вместившееся в струи — в каплю
        self.left = with_.center < 0
        self.width = 2.0 * with_.half_width
        self.volume = np.zeros((Nc, 2))
        self.ks = np.zeros((Nc, 2, 2 * PResponseLResponseWith))

    def spill(self, c, new, full, foreign, ks_canvas, place=None):
        """Шаг переноса c: подобранное, что не поместилось на кончике, записано в вал. Ответ — капля и подобранное без него.
        `место` (6а-1, струи) — сколько подобранного кончик несёт ношей; остальное — в каплю."""
        carries = 0.0 if self.carry is None else self.carry.volume + self.carry.up   # ноша (и ушедшее вверх по ворсу) — на той же кисти: место занимает
        the_excess = clamp(new + carries - self.holds * full, 0.0, foreign)
        new, foreign = new - the_excess, foreign - the_excess
        if self.carry is not None:
            in_drop = self.carry.take(foreign, ks_canvas, place, self.in_drop)
            new, foreign = new - foreign + in_drop, in_drop
        if the_excess.any():
            v = the_excess * self.width
            for st, m in ((0, self.left), (1, ~self.left)):
                self.volume[c, st] = v[m].sum()
                self.ks[c, st] = (ks_canvas[m] * v[m, None]).sum(axis=0) / max(self.volume[c, st], 1e-12)
        return new, foreign

    def fade_out(self, c, agrees, ks_drops, streams, wet):
        """Гребень кладёт `уходит`: ноша сходит первой — на гребень (как было) или, у тонкой краски по сырому, в струи."""
        if streams is None:
            return self.carry.fade_out(agrees, ks_drops)
        own = self.carry.in_streams(c, agrees, streams, wet)
        return self.carry.fade_out(own, ks_drops, streams.equal)

    def in_raster(self, xs, ys, weights, with_, W, on_canvas, lag, i0, bx0, by0):
        """Дописать отсчёты вала к отсчётам мазка: места, объём и цвет; касания и съёма у вала нет."""
        if not self.volume.any():
            return xs, ys, weights
        width = H._GROWING_TOGETHER_BANK[1] + H._GROWING_TOGETHER_BANK[2] * W
        # вал лежит откосом на обе стороны от кромки (Движок 25): наружу сходит на нет на своей ширине, внутрь — на
        # тех 7 px, где волоски несут меньше полной ноши и прижаты слабо (`ноша_у_края`): они его не сминают.
        # Было: вал начинался полной высотой сразу за кромкой — стенка в 1 px, тёмная черта света вдоль бока
        inward = min(7.0, 0.5 * with_.half_width)
        n = max(int(np.ceil((width + inward) / 0.5)), 2)
        behind = -inward + (np.arange(n) + 0.5) * ((width + inward) / n)   # px за краем кисти (минус — под краем)
        weight = np.where(behind > 0.0, 1.0 - behind / width, 1.0 + behind / inward)   # вал выше у кромки и сходит на нет
        weight = weight / weight.sum()
        rim = with_.half_width - H._YaDRResponse                         # кромка следа: самый край кисти краски не кладёт
        u = np.concatenate([-rim - behind, rim + behind])
        st = np.repeat([0, 1], n)
        edge = np.where(st == 0, 0, with_.nr - 1)
        X, Y = on_canvas(u, with_.strand[edge], lag[edge], np.zeros(2 * n))
        d = self.volume[i0][:, st] * (np.tile(weight, 2) / _ShBrush)[None, :]     # объём шага переноса — на его шаги пути поровну
        ii, jj = np.nonzero(d > 0.0)
        d = by_samples(d, ii, jj)
        ts = from_ks(self.ks)[i0[ii], st[jj]]
        own = [d, d * ts[:, 0], d * ts[:, 1], d * ts[:, 2]]
        weights = [np.concatenate([w, own[q] if q < 4 else np.zeros(len(d))]) for q, w in enumerate(weights)]
        return np.concatenate([xs, by_samples(X, ii, jj) - bx0]), np.concatenate([ys, by_samples(Y, ii, jj) - by0]), weights
