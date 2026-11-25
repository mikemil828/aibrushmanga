# READING VIEW of движок/кисть/скотч.py with names in English. Not runnable; edit the original.
"""Н-3: скотч-трафарет — малярная лента на холсте (`Холст.скотч`, `Холст.снять_скотч`).

Лента лежит на холсте там, где маска истинна. Краска мазков поверх неё ложится на ленту: под лентой слои
холста (все или один — `слой`) на это время пусты, а что в них было, лента держит у себя. Кромка ленты — ступенька
её высоты: кисть видит её тем же рельефом, что зерно грунта (`зерно.по_зерну`), — с ленты волосок сходит не
обрывом, у кромки на холсте он до дна не достаёт, пока нажим слаб; густая полная кисть кроет и тут.

Снятие: под лентой — что было до неё, бит в бит; краска на ленте уходит с ней. Физика у кромки:
- затёк — текучая краска (разбавитель, `ровно`), легшая у самой кромки, затекает под ленту на доли мм, тем глубже,
  чем больше нажим; густая и сухая не затекает. Глубину каждый мазок пишет ленте у своей кромки (`Лента.затекло`);
- разрыв — сырой пастоз, лёгший через кромку одной плёнкой, при снятии рвётся: у кромки часть его тела уходит с
  лентой, на глубину по толщине моста; сухое (`подсушить`) и схватившееся режется ровно.

Ни шума формы, ни случайного разброса здесь нет: неровность кромки — от борозд и толщины самой краски."""
import base64
import zlib

import numpy as np

from .canvas import MM_ON_H, Layer
from .mm import RResponseDNResponseY

HEIGHT_MM = 0.1        # толщина малярной ленты по умолчанию, мм (0,25 h)
_ZATYoBrush = (0.2, 0.5, 0.5, 0.08)   # затёк: до стольких мм под ленту; доля от нажима (0,5 + 0,5·нажим); текучесть (разбавитель, ровно), с которой затёк полный; столько h свежей краски у кромки — затёк полный
_RAZRYIn = (0.25, 0.30, 0.50, 0.7, 0.5)   # разрыв: до стольких мм от кромки; мост тоньше стольких h не рвётся; на столько h сверх — разрыв полный; столько свежего тела уходит с лентой у самой кромки; на столько падает покрытие там, где тело ушло целиком
_ResponseBRYIn = (0.15, 0.7, 0.35, 0.6)   # волосок, сходящий с ленты, касается холста не сразу: на стольких мм за кромкой при нажиме 0,6 и ленте 0,1 мм (с нажимом — короче, как 1,3 − нажим); столько тела и столько покрытия хода нет у самой кромки; густая краска висит на кончиках и мост держит, текучая затягивает щель — на эту долю короче
_InALAndBrush = (0.2, 0.35, 2.0, 0.15, 0.45)   # кисть, идущая на ленту, о её стенку сгребает краску кончиков: валик шириной столько мм у кромки на холсте; столько краски, что легла бы на ленту в первых px (их столько) за кромкой, сгребается при густой краске и нажиме 1; густота — тело свежей краски сверх стольких h, полная — ещё на столько
_PResponseLE = 6              # px вокруг ленты, что она держит у себя (сырой разрыв и сверка)
_BrushResponseLTsResponse = 2.0          # px снаружи у кромки: краска, легшая сюда, затекает под ленту
_InInERCanvas = 3.0           # px: откуда пришёл волосок — столько назад по ходу (на ленте ли он был)
_WithGLADAndT = 1.0        # σ px: кромка ленты — линия, а не лесенка пикселей маски (доля ленты в пикселе кромки)


def _edt(mask, indices=False):
    from scipy import ndimage
    return ndimage.distance_transform_edt(mask, return_indices=indices)


def mask_in_lines(mask):
    """bool H×W -> строка журнала: zlib(packbits) в base64."""
    return base64.b64encode(zlib.compress(np.packbits(np.asarray(mask, bool), axis=None).tobytes(), 9)).decode('ascii')


def mask_from_lines(row, shape):
    in_, sh = (int(v) for v in shape)
    bits = np.frombuffer(zlib.decompress(base64.b64decode(row)), np.uint8)
    return np.unpackbits(bits, count=in_ * sh).astype(bool).reshape(in_, sh)


def in_size(mask, height, width):
    """Маска журнала другого масштаба (повтор `--scale`) — ближайшим пикселем."""
    if mask.shape == (height, width):
        return mask
    iy = np.minimum((np.arange(height) + 0.5) * mask.shape[0] / height, mask.shape[0] - 1).astype(np.intp)
    ix = np.minimum((np.arange(width) + 0.5) * mask.shape[1] / width, mask.shape[1] - 1).astype(np.intp)
    return mask[iy][:, ix]


class Tape:
    """Одна лента: имя, маска (bool, весь холст), слой (None — все слои), высота (h), рамка поля (y0, y1, x0, x1 холста),
    что было в слоях в рамке до ленты (`под`), затёк по кромке (px, в рамке)."""

    def __init__(self, name, mask, layer, height_mm):
        self.name, self.layer = str(name), layer
        self.mask = np.asarray(mask, bool)
        self.height_mm = float(height_mm)
        self.height = self.height_mm / MM_ON_H
        ys, xs = np.nonzero(self.mask)
        in_, sh = self.mask.shape
        self.frame = (max(int(ys.min()) - _PResponseLE, 0), min(int(ys.max()) + 1 + _PResponseLE, in_),
                      max(int(xs.min()) - _PResponseLE, 0), min(int(xs.max()) + 1 + _PResponseLE, sh))
        y0, y1, x0, x1 = self.frame
        m = self.mask[y0:y1, x0:x1]
        d_outside, (iy, ix) = _edt(~m, indices=True)
        d_under = _edt(m)
        self.ring = ~m & (d_outside <= _BrushResponseLTsResponse)
        self.brush_tape = (iy.astype(np.int32), ix.astype(np.int32))      # снаружи — ближайший пиксель ленты
        from scipy import ndimage
        b = np.where(m, -(d_under - 0.5), d_outside - 0.5).astype(np.float32)   # px от кромки: снаружи +, под лентой −
        self.from_edges = b
        bs = ndimage.gaussian_filter(b, _WithGLADAndT, mode='nearest')
        gy, gx = np.gradient(bs)
        n = np.maximum(np.hypot(gx, gy), 1e-6)
        self.normal = ((-gx / n).astype(np.float32), (-gy / n).astype(np.float32))   # к ленте
        # доля ленты в пикселе кромки — по сглаженной линии, только в пикселях маски: краска за кромкой — вся, а пиксель
        # ленты на лесенке маски — часть его вне ленты, она несёт краску (кромка — линия, а не лесенка)
        self.share = np.where(m, np.clip(0.5 - bs, 0.0, 1.0), 0.0).astype(np.float32)
        self.seepage = np.zeros(m.shape, np.float32)
        self.under = {}          # имя слоя -> {поле: массив рамки}; слоя, которого не было, нет

    def __repr__(self):
        return (f'лента «{self.name}»: {int(self.mask.sum())} px, высота {self.height_mm:g} мм, '
                + ('все слои' if self.layer is None else f'слой «{self.layer}»'))

    def catches(self, layer):
        return self.layer is None or layer == self.layer

    def m(self):
        y0, y1, x0, x1 = self.frame
        return self.mask[y0:y1, x0:x1]

    def relief(self, x0, y0, x1, y1):
        """Высота ленты (h) в прямоугольнике холста (может выходить за край)."""
        t = np.zeros((y1 - y0, x1 - x0), np.float32)
        in_, sh = self.mask.shape
        xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, sh), min(y1, in_)
        if xb > xa and yb > ya:
            t[ya - y0:yb - y0, xa - x0:xb - x0] = self.mask[ya:yb, xa:xb] * np.float32(self.height)
        return t

    def seeped(self, rectangle, to, after, fluidity, pressure, px_on_mm):
        """После мазка: свежая краска (после − до, h) в прямоугольнике холста (y0, y1, x0, x1) у кромки снаружи
        затекает под ленту — глубина (px) в кольцо кромки."""
        if fluidity <= 0.0:
            return
        y0, y1, x0, x1 = rectangle
        ry0, ry1, rx0, rx1 = self.frame
        ya, yb, xa, xb = max(y0, ry0), min(y1, ry1), max(x0, rx0), min(x1, rx1)
        if yb <= ya or xb <= xa:
            return
        with_ = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        r = (slice(ya - ry0, yb - ry0), slice(xa - rx0, xb - rx0))
        fresh = np.maximum(after[with_] - to[with_], 0.0)
        depth = (_ZATYoBrush[0] * (px_on_mm or RResponseDNResponseY) * (0.5 + 0.5 * float(pressure)) * min(float(fluidity) / _ZATYoBrush[2], 1.0)
                 * np.clip(fresh / _ZATYoBrush[3], 0.0, 1.0))
        depth = np.where(self.ring[r], depth, 0.0).astype(np.float32)
        np.maximum(self.seepage[r], depth, out=self.seepage[r])


def _directions(points):
    t = np.asarray(points, np.float64)
    if t.ndim != 2 or len(t) < 2:
        return None
    d = np.gradient(t[:, :2], axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    if not (n > 1e-9).any():
        return None
    d = np.where(n[:, None] > 1e-9, d / np.maximum(n, 1e-12)[:, None], 0.0)
    return t, d


def at_edges(tape, with_, straight, to_h, to_a, points, px_on_mm, thinner, fluidity=0.0, angle=(0, 0)):
    """После мазка в слой `с`, у кромки ленты на холсте: обрыв — волосок, сошедший с ленты, касается холста не сразу
    (щель короче с нажимом и у густой краски); валик — кисть, идущая на ленту, сгребает краску кончиков о её стенку
    к кромке (больше с нажимом и густотой). Откуда и куда шёл волосок — по ходу кисти у этого места."""
    nd = _directions(points)
    if nd is None:
        return
    t, d = nd
    y0, y1, x0, x1 = straight
    ry0, ry1, rx0, rx1 = tape.frame
    ya, yb, xa, xb = max(y0, ry0), min(y1, ry1), max(x0, rx0), min(x1, rx1)
    if yb <= ya or xb <= xa:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    r = (slice(ya - ry0, yb - ry0), slice(xa - rx0, xb - rx0))
    m, b = tape.m()[r], tape.from_edges[r]
    band = ~m & (b < 3.0)
    if not band.any():
        return
    oy, ox = angle                                                     # угол массивов слоя на холсте (участок)
    m_ = (slice(ya - oy, yb - oy), slice(xa - ox, xb - ox))
    h = np.array(with_.thickness[m_]); a = np.array(with_.coverage[m_])
    h0, a0 = to_h[sl], to_a[sl]
    fresh = np.maximum(h - h0, 0.0)
    fresh_a = np.maximum(a - a0, 0.0)
    if not (fresh[band] > 0.0).any():
        return
    mm = px_on_mm or RResponseDNResponseY
    py, px = np.nonzero(band)
    X, Y = px + xa + 0.5, py + ya + 0.5
    # ближайшая точка хода — его направление и нажим здесь
    nearest = np.empty(len(X), np.intp)
    for i in range(0, len(X), 4096):
        dx = X[i:i + 4096, None] - t[None, :, 0]
        dy = Y[i:i + 4096, None] - t[None, :, 1]
        nearest[i:i + 4096] = np.argmin(dx * dx + dy * dy, axis=1)
    dx_, dy_ = d[nearest, 0], d[nearest, 1]
    pressure = t[nearest, 2] if t.shape[1] > 2 else np.full(len(X), 0.85)
    in_, sh = tape.mask.shape
    def on_tape(x, y):
        ix = np.clip(np.floor(x).astype(np.intp), 0, sh - 1)
        iy = np.clip(np.floor(y).astype(np.intp), 0, in_ - 1)
        return tape.mask[iy, ix] & (x >= 0) & (x < sh) & (y >= 0) & (y < in_)
    came = on_tape(X - dx_ * _InInERCanvas, Y - dy_ * _InInERCanvas)      # волосок пришёл с ленты
    agrees = on_tape(X + dx_ * _InInERCanvas, Y + dy_ * _InInERCanvas)      # волосок идёт на ленту
    nkh, nu = tape.normal[0][r][py, px], tape.normal[1][r][py, px]
    oblique = np.abs(dx_ * nkh + dy_ * nu)
    iy_l, ix_l = tape.brush_tape[0][r][py, px] - (ya - ry0), tape.brush_tape[1][r][py, px] - (xa - rx0)
    has_l = (iy_l >= 0) & (iy_l < yb - ya) & (ix_l >= 0) & (ix_l < xb - xa)
    iy_l, ix_l = np.clip(iy_l, 0, yb - ya - 1), np.clip(ix_l, 0, xb - xa - 1)
    body_l = np.where(has_l, fresh[iy_l, ix_l], 0.0)                 # краска хода на ленте у этого места
    body = np.maximum(fresh[py, px], body_l)
    thickly = np.clip((body - _InALAndBrush[3]) / _InALAndBrush[4], 0.0, 1.0) * (1.0 - float(thinner or 0.0))
    bb = b[py, px]
    dh = np.zeros(len(X)); da = np.zeros(len(X))
    # обрыв: сошедший с ленты волосок касается холста за щелью
    flows = min(float(fluidity) / _ZATYoBrush[2], 1.0)                  # текучая краска щель затягивает
    slit = (_ResponseBRYIn[0] * mm * (tape.height_mm / 0.1) * np.clip(1.3 - pressure, 0.1, 1.0) / 0.7 * (1.0 - _ResponseBRYIn[3] * thickly)
            * (1.0 - _ResponseBRYIn[3] * flows))
    s_response = np.where(came & ~agrees, np.clip(1.0 - bb / np.maximum(slit, 1e-6), 0.0, 1.0) * oblique, 0.0)
    dh -= fresh[py, px] * _ResponseBRYIn[1] * s_response
    da -= fresh_a[py, px] * _ResponseBRYIn[2] * s_response
    # валик: идущая на ленту кисть сгребает краску о стенку к кромке
    r = max(_InALAndBrush[0] * mm, 1.0)
    s_in = np.where(agrees & ~came & (bb < r), 2.0 * (1.0 - bb / r) * oblique, 0.0)
    bank = _InALAndBrush[1] * thickly * np.clip(pressure, 0.0, 1.0) * body_l * (_InALAndBrush[2] / r) * s_in
    dh += bank
    da += (1.0 - a[py, px]) * np.clip(bank / 0.15, 0.0, 1.0)
    if not (dh != 0.0).any():
        return
    ts = np.array(with_.color[m_])
    if (bank > 0).any():
        tsv = ts[iy_l, ix_l]
        w = (bank / np.maximum(h[py, px] + bank, 1e-9))[:, None]
        ts[py, px] = ts[py, px] * (1.0 - w) + tsv * w
        with_.color[m_] = ts.astype(np.float32)
    h[py, px] = np.maximum(h[py, px] + dh, h0[py, px])
    a[py, px] = np.clip(a[py, px] + da, a0[py, px], 1.0)
    with_.thickness[m_] = h.astype(np.float32)
    with_.coverage[m_] = a.astype(np.float32)


def _fields(with_):
    return [p for p in Layer._InWithE if getattr(with_, p) is not None]


def lay(canvas, tape):
    """Лента ложится: что было в её слоях в рамке — ей; под маской слои пусты."""
    y0, y1, x0, x1 = (tape.frame[0] - canvas.y0, tape.frame[1] - canvas.y0, tape.frame[2] - canvas.x0, tape.frame[3] - canvas.x0)
    where = (slice(y0, y1), slice(x0, x1))
    m = tape.m()
    for name, with_ in canvas._layers.items():
        if not tape.catches(name):
            continue
        tape.under[name] = {p: np.array(getattr(with_, p)[where]) for p in _fields(with_)}
        for p in _fields(with_):
            a = np.array(getattr(with_, p)[where])
            empty = Layer._PAtWithTResponse.get(p, 0.0)
            a[m] = empty
            getattr(with_, p)[where] = a


def _over(bottom, top, w):
    """Краска `верх` (поля слоя) долей w поверх `низ` — тот же слой: покрытие и цвет «поверх», тело в сумме."""
    a_in = top['покрытие'] * w
    a_n = bottom['покрытие']
    a = a_in + a_n * (1.0 - a_in)
    response = {'покрытие': a, 'толщина': bottom['толщина'] + top['толщина'] * w,
         'цвет': (top['цвет'] * a_in[..., None] + bottom['цвет'] * (a_n * (1.0 - a_in))[..., None]) / np.maximum(a, 1e-6)[..., None],
         'влажность': np.where(a_in > 0.0, np.maximum(bottom['влажность'], top['влажность']), bottom['влажность']),
         'схват': np.where(a_in > 0.0, np.minimum(bottom['схват'], top['схват']), bottom['схват'])}
    response['цвет'] = np.where((a > 0.0)[..., None], response['цвет'], bottom['цвет'])
    if 'пропуск' in bottom:
        response['пропуск'] = bottom['пропуск'] * (1.0 - a_in[..., None]) + top['пропуск'] * a_in[..., None]
    if 'текучая' in bottom:
        response['текучая'] = np.where(a_in > 0.0, np.maximum(bottom['текучая'], top['текучая']), bottom['текучая'])
    return response


def remove(canvas, tape):
    """Лента уходит: краска на ней — с ней; под ней — что было, бит в бит; у кромки — затёк и разрыв (`скотч.py`).
    Ответ — {'затёк': px площади под лентой с затёком, 'разрыв': px у кромки, где сырой пастоз ушёл с лентой}."""
    ry0, ry1, rx0, rx1 = tape.frame
    where = (slice(ry0 - canvas.y0, ry1 - canvas.y0), slice(rx0 - canvas.x0, rx1 - canvas.x0))
    m = tape.m()
    mm = canvas.px_on_mm or RResponseDNResponseY
    # до снятия: что лежит на ленте и у кромки; расстояния и ближайшие пиксели через кромку
    d_outside, (iy_in, ix_in) = _edt(~m, indices=True)          # снаружи — до ближайшего пикселя ленты
    d_under, (iy_n, ix_n) = _edt(m, indices=True)            # под лентой — до ближайшего пикселя снаружи
    result = {'затёк': 0, 'разрыв': 0}
    seepage = tape.seepage[iy_n, ix_n]                         # глубина затёка у ближайшей точки кромки снаружи
    in_seepage = m & (seepage > 0.0) & (d_under - 0.5 < seepage)
    strength_seepage = np.where(in_seepage, np.clip(1.0 - (d_under - 0.5) / np.maximum(seepage, 1e-6), 0.0, 1.0), 0.0)
    break_px = _RAZRYIn[0] * mm
    beside = ~m & (d_outside - 0.5 < break_px)
    affected_break = np.zeros(m.shape, bool)
    affected_seepage = np.zeros(m.shape, bool)
    for name, with_ in canvas._layers.items():
        if not tape.catches(name):
            continue
        fields = _fields(with_)
        now = {p: np.array(getattr(with_, p)[where]) for p in fields}
        was = tape.under.get(name)
        if was is None:                                     # слой заведён по ленте: под ней не было ничего
            was = {p: np.full(now[p].shape, Layer._PAtWithTResponse.get(p, 0.0), np.float32) for p in fields}
        else:
            was = {p: (was[p] if p in was else np.full(now[p].shape, Layer._PAtWithTResponse.get(p, 0.0), np.float32)) for p in fields}
        h, h0 = now['толщина'], was['толщина']
        fresh = np.where(m, h, np.maximum(h - h0, 0.0))     # что легло после ленты: снаружи — сверх прежнего, на ленте — всё
        fresh_a = np.where(m, now['покрытие'], (now['покрытие'] - was['покрытие']).clip(0.0))
        wet = fresh * now['влажность'] * (1.0 - now['схват'])
        new = {p: now[p].copy() for p in fields}
        # разрыв: сырой мост через кромку — тоньше из двух сторон
        if beside.any():
            bridge = np.minimum(wet, wet[iy_in, ix_in])
            depth = break_px * np.clip((bridge - _RAZRYIn[1]) / _RAZRYIn[2], 0.0, 1.0)
            f = np.where(beside & (depth > 0.0), np.clip(1.0 - (d_outside - 0.5) / np.maximum(depth, 1e-6), 0.0, 1.0), 0.0)
            if f.any():
                went = _RAZRYIn[3] * f
                new['толщина'] = h - fresh * went
                new['покрытие'] = now['покрытие'] - fresh_a * _RAZRYIn[4] * went * went
                affected_break |= f > 0.0
        # под лентой — что было; пиксель кромки — по доле ленты в нём (кромка — линия, не лесенка маски): его доля
        # вне ленты под маской несёт краску, что легла на ленту, поверх прежнего
        t = tape.share
        rim_under = m & (t < 1.0)
        if rim_under.any():
            laid = _over(was, now, 1.0)
            for p in fields:
                kp = rim_under[..., None] if new[p].ndim == 3 else rim_under
                new[p] = np.where(kp, laid[p], new[p])
        for p in fields:
            tt = t[..., None] if new[p].ndim == 3 else t
            blend = was[p] * tt + new[p] * (1.0 - tt)
            new[p] = np.where(tt >= 1.0, was[p], np.where(tt <= 0.0, new[p], blend))
        # затёк: свежая краска у кромки снаружи — под ленту, на долю по глубине
        if in_seepage.any():
            w = strength_seepage
            a_with = fresh_a[iy_n, ix_n] * w
            h_with = fresh[iy_n, ix_n] * w
            has = in_seepage & (h_with > 0.0) & (a_with > 0.0)
            if has.any():
                a = new['покрытие']
                a_new = a + a_with * (1.0 - a)
                ts_with = now['цвет'][iy_n, ix_n]
                ts = (new['цвет'] * (a * (1.0 - a_with))[..., None] + ts_with * a_with[..., None]) / np.maximum(a_new, 1e-6)[..., None]
                new['цвет'] = np.where(has[..., None], ts, new['цвет'])
                new['покрытие'] = np.where(has, a_new, a)
                new['толщина'] = np.where(has, new['толщина'] + h_with, new['толщина'])
                new['влажность'] = np.where(has, np.maximum(new['влажность'], now['влажность'][iy_n, ix_n]), new['влажность'])
                new['схват'] = np.where(has, np.minimum(new['схват'], now['схват'][iy_n, ix_n]), new['схват'])
                if 'пропуск' in new:
                    p_with = now['пропуск'][iy_n, ix_n]
                    new['пропуск'] = np.where(has[..., None], new['пропуск'] * (1.0 - a_with[..., None]) + p_with * a_with[..., None], new['пропуск'])
                if 'текучая' in new:
                    new['текучая'] = np.where(has, np.maximum(new['текучая'], now['текучая'][iy_n, ix_n]), new['текучая'])
                affected_seepage |= has
        for p in fields:
            getattr(with_, p)[where] = new[p].astype(np.float32)
    result['затёк'] = int(affected_seepage.sum())
    result['разрыв'] = int(affected_break.sum())
    return result
