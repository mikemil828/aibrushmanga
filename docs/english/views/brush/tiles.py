# READING VIEW of движок/кисть/плитки.py with names in English. Not runnable; edit the original.
"""2в, Е-6: поле слоя плитками — плитка заводится, когда в неё пишут не то, что в ней лежит; ровная — одним числом.

Поле (`Слой.цвет`, `покрытие`, `толщина`, `влажность`, `схват`, `пропуск`, `текучая`) — `Плитки`: читается и пишется,
как массив numpy (срезы, маски, индексы, арифметика, функции numpy). Срез — копия (собранная из плиток), не вид: менять
поле — только присваиванием `поле[срез] = …`. Значения те же, бит в бит.

Устройство: арена (плиток по высоте, по ширине, ПЛИТКА, ПЛИТКА[, 3]) — память, которую система даёт страницами, когда
в них пишут (`np.zeros` или общий mmap ядер), и таблица состояний плиток: 0 — нетронутая (`пусто`), 1 — ровная
(число в таблице чисел), 2 — своя (в арене). Нетронутую и ровную плитку чтение в арене не трогает — её страницы
не берутся. На ядрах арена и таблицы — в общей памяти: процесс счёта пишет мазок прямо в них (`окно`); два мазка,
что идут разом, плиток не делят (`ядра.py` сверяет рамки по сетке плиток)."""
import mmap

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

PLAndTBrushA = 64          # сторона плитки, px: «роса» стенда 2в — 11 МБ полей; меньше плитка — больше счёта склейки


def _bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _bits(v):
    return int(np.float32(v).view(np.uint32))


def _memory(shape, type, shared):
    """Нули: свои или в общей памяти процессов (анонимный mmap): страницы берутся, когда в них пишут."""
    n = int(np.prod(shape)) * np.dtype(type).itemsize
    if n < 1 << 16:
        return _memory_small(shape, type, shared)
    # своя — тоже mmap, не np.zeros: numpy советует системе большие страницы (2 МБ) — одна плитка взяла бы 2 МБ
    m = mmap.mmap(-1, n, flags=mmap.MAP_SHARED if shared else mmap.MAP_PRIVATE)
    return np.frombuffer(m, dtype=type, count=int(np.prod(shape))).reshape(shape)


def _memory_small(shape, type, shared):
    if not shared:
        return np.zeros(shape, type)
    n = int(np.prod(shape)) * np.dtype(type).itemsize
    return np.frombuffer(mmap.mmap(-1, max(n, 1)), dtype=type, count=int(np.prod(shape))).reshape(shape)


def by_grid(r):
    """Прямоугольник (x0, y0, x1, y1) -> он же, расширенный до границ плиток."""
    T = PLAndTBrushA
    return (r[0] // T * T, r[1] // T * T, -(-r[2] // T) * T, -(-r[3] // T) * T)


class Tiles(NDArrayOperatorsMixin):
    """Поле слоя (В, Ш) или (В, Ш, 3) float32 плитками ПЛИТКА×ПЛИТКА (шапка модуля). `байт` — сколько памяти полей
    занято на деле; `окно(гy, гx)` — вид на прямоугольник (та же память, свои координаты)."""

    __array_priority__ = 20
    dtype = np.dtype(np.float32)

    def __init__(self, shape, empty=0.0, shared=False):
        T = PLAndTBrushA
        shape = tuple(int(v) for v in shape)
        self.shape = shape
        self.empty = np.float32(empty)
        self.shared = bool(shared)
        self._oy = self._ox = 0
        self._full = shape[:2]
        n = (-(-shape[0] // T), -(-shape[1] // T))
        self._state = _memory(n, np.uint8, shared)
        self._number = _memory(n, np.float32, shared)
        self._arena = _memory(n + (T, T) + shape[2:], np.float32, shared)

    # ---------------------------------------------------------------- вид и копии
    def window(self, gy, gx):
        """Вид на прямоугольник поля (срезы без шага): пишет в ту же память."""
        y0, y1, _ = gy.indices(self.shape[0])
        x0, x1, _ = gx.indices(self.shape[1])
        in_ = object.__new__(Tiles)
        in_.__dict__.update(self.__dict__)
        in_._oy, in_._ox = self._oy + y0, self._ox + x0
        in_.shape = (max(y1 - y0, 0), max(x1 - x0, 0)) + self.shape[2:]
        return in_

    def __getstate__(self):
        if (self._oy, self._ox) != (0, 0) or self.shape[:2] != self._full:
            return Tiles.from_(np.asarray(self), self.empty).__getstate__()
        state = np.array(self._state)
        state[state == 3] = 1                          # копия: прежнего в арене у неё нет
        return {'shape': self.shape, 'пусто': float(self.empty), 'сост': state, 'число': np.array(self._number),
                'плитки': self._arena[state == 2]}

    def __setstate__(self, d):
        self.__init__(d['shape'], d['пусто'])
        self._state[...] = d['сост']
        self._number[...] = d['число']
        self._arena[d['сост'] == 2] = d['плитки']

    def __deepcopy__(self, memo):
        n = object.__new__(Tiles)
        n.__setstate__(self.__getstate__())
        return n

    def own(self, shared=False):
        """Копия поля: своя память (общая=False) или общая память процессов."""
        d = self.__getstate__()
        n = Tiles(d['shape'], d['пусто'], shared)
        n._state[...] = d['сост']
        n._number[...] = d['число']
        n._arena[d['сост'] == 2] = d['плитки']
        return n

    def cutout(self, gy, gx):
        """2г: копия прямоугольника (срезы без шага) плитками той же сетки: своя память — только его плитки (тронутые —
        копией, ровные — числом); ответ — окно на прямоугольник, координаты окна — как у среза."""
        T = PLAndTBrushA
        y0, y1, _ = gy.indices(self.shape[0])
        x0, x1, _ = gx.indices(self.shape[1])
        y1, x1 = max(y1, y0), max(x1, x0)
        Y0, Y1, X0, X1 = y0 + self._oy, y1 + self._oy, x0 + self._ox, x1 + self._ox
        ty0, tx0 = Y0 // T, X0 // T
        ty1, tx1 = max(-(-Y1 // T), ty0), max(-(-X1 // T), tx0)
        n = Tiles((max(min(ty1 * T, self._full[0]) - ty0 * T, 0), max(min(tx1 * T, self._full[1]) - tx0 * T, 0)) + self.shape[2:],
                   self.empty)
        with_ = np.array(self._state[ty0:ty1, tx0:tx1])
        with_[with_ == 3] = 1                                  # копия: прежнего в арене у неё нет
        n._state[...] = with_
        n._number[...] = self._number[ty0:ty1, tx0:tx1]
        n._arena[with_ == 2] = self._arena[ty0:ty1, tx0:tx1][with_ == 2]
        return n.window(slice(Y0 - ty0 * T, Y1 - ty0 * T), slice(X0 - tx0 * T, X1 - tx0 * T))

    def base(self):
        """2г: всё поле, на которое смотрит окно (своё — оно само): (поле, сдвиг окна по y, по x)."""
        response = object.__new__(Tiles)
        response.__dict__.update(self.__dict__)
        response._oy = response._ox = 0
        response.shape = tuple(self._full) + self.shape[2:]
        return response, self._oy, self._ox

    def has_outside(self, gy, gx):
        """2г: есть ли не ноль в поле вне прямоугольника (срезы без шага) — по плиткам, без сборки поля."""
        T = PLAndTBrushA
        y0, y1, _ = gy.indices(self.shape[0])
        x0, x1, _ = gx.indices(self.shape[1])
        H, W = self.shape[:2]
        oy, ox = self._oy, self._ox
        for ty in range(oy // T, (oy + H - 1) // T + 1) if H and W else ():
            for tx in range(ox // T, (ox + W - 1) // T + 1):
                ya, yb = max(ty * T, oy) - oy, min(ty * T + T, oy + H) - oy      # плитка в координатах окна
                xa, xb = max(tx * T, ox) - ox, min(tx * T + T, ox + W) - ox
                if y0 <= ya and yb <= y1 and x0 <= xa and xb <= x1:
                    continue                                                    # вся внутри прямоугольника
                with_ = self._state[ty, tx]
                if with_ != 2:
                    if (self.empty if with_ == 0 else self._number[ty, tx]) != 0:
                        return True
                    continue
                p = self._arena[ty, tx][ya + oy - ty * T:yb + oy - ty * T, xa + ox - tx * T:xb + ox - tx * T]
                inside = np.zeros(p.shape[:2], bool)
                inside[max(y0 - ya, 0):max(y1 - ya, 0), max(x0 - xa, 0):max(x1 - xa, 0)] = True
                if (p[~inside] != 0).any():
                    return True
        return False

    @classmethod
    def from_(cls, a, empty=0.0, shared=False):
        """Поле плитками из массива (В, Ш[, 3]): плитки, равные `пусто`, — без памяти, ровные — числом."""
        a = np.asarray(a, np.float32)
        p = cls(a.shape, empty, shared)
        p._write(0, a.shape[0], 0, a.shape[1], a)
        return p

    # ---------------------------------------------------------------- как массив
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    @property
    def nbytes(self):
        """Как у массива numpy: размер поля целиком. Занятое на деле — `байт`."""
        return 4 * self.size

    @property
    def bytes(self):
        return int((self._state >= 2).sum()) * self._arena[0, 0].nbytes + self._state.nbytes + self._number.nbytes

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f'Плитки({self.shape}, своих плиток {int((self._state == 2).sum())}, {self.bytes / 1e6:.1f} МБ)'

    def __array__(self, dtype=None, copy=None):
        a = self._collect(0, self.shape[0], 0, self.shape[1])
        return a if dtype is None else a.astype(dtype, copy=False)

    def copy(self):
        return np.asarray(self)

    def __getattr__(self, name):       # .sum(), .max(), .astype(), .any(axis=…) … — у собранного поля
        if name.startswith('_'):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __iter__(self):
        return iter(np.asarray(self))

    def __array_ufunc__(self, ufunc, method, *entries, out=None, **kw):
        entries = tuple(np.asarray(v) if isinstance(v, Tiles) else v for v in entries)
        if out is None:
            return getattr(ufunc, method)(*entries, **kw)
        out2 = tuple(np.asarray(o) if isinstance(o, Tiles) else o for o in out)
        r = getattr(ufunc, method)(*entries, out=out2, **kw)
        if not any(isinstance(o, Tiles) for o in out):
            return r
        for o, o2 in zip(out, out2):
            if isinstance(o, Tiles):
                o[...] = o2
        return out[0] if len(out) == 1 else out

    def any(self, *a, **kw):
        if a or kw or (self._oy, self._ox) != (0, 0) or self.shape[:2] != self._full:
            return bool(np.asarray(self).any(*a, **kw)) if not (a or kw) else np.asarray(self).any(*a, **kw)
        with_ = self._state
        if self.empty != 0 and (with_ == 0).any():
            return True
        if (self._number[(with_ == 1) | (with_ == 3)] != 0).any():
            return True
        return bool(self._arena[with_ == 2].any())

    # ---------------------------------------------------------------- плитки
    def _option(self, option):
        """Ключ среза -> (y0, y1, x0, x1, шаг, хвост, сжать) прямоугольника в координатах окна; None — не прямоугольник."""
        if not isinstance(option, tuple):
            option = (option,)
        if any(k is Ellipsis for k in option):
            i = next(i for i, k in enumerate(option) if k is Ellipsis)
            extra = self.ndim - (len(option) - 1 - sum(k is None for k in option))
            option = option[:i] + (slice(None),) * max(extra, 0) + option[i + 1:]
            if any(k is Ellipsis for k in option):
                return None
        axes, spacing, compress = [], [], []
        for axis in range(2):
            k = option[axis] if axis < len(option) else slice(None)
            n = self.shape[axis]
            if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
                k = int(k) + (n if k < 0 else 0)
                if not 0 <= k < n:
                    raise IndexError(f'индекс {k} вне поля {n}')
                axes.append((k, k + 1))
                spacing.append(1)
                compress.append(axis)
            elif isinstance(k, slice) and (k.step is None or k.step > 0):
                a, b, s = k.indices(n)
                axes.append((a, max(a, b)))
                spacing.append(s)
            else:
                return None
        tail = option[2:]
        if tail == (slice(None),) * len(tail) and len(tail) <= self.ndim - 2:
            tail = ()
        return axes[0][0], axes[0][1], axes[1][0], axes[1][1], tuple(spacing), tail, tuple(compress)

    def _collect(self, y0, y1, x0, x1):
        """Прямоугольник окна -> массив (своя память)."""
        T = PLAndTBrushA
        out = np.empty((max(y1 - y0, 0), max(x1 - x0, 0)) + self.shape[2:], np.float32)
        if y1 <= y0 or x1 <= x0:
            return out
        oy, ox = self._oy, self._ox
        Y0, Y1, X0, X1 = y0 + oy, y1 + oy, x0 + ox, x1 + ox
        state, number, arena, empty = self._state, self._number, self._arena, self.empty
        for ty in range(Y0 // T, (Y1 - 1) // T + 1):
            ya, yb = max(Y0, ty * T), min(Y1, ty * T + T)
            for tx in range(X0 // T, (X1 - 1) // T + 1):
                xa, xb = max(X0, tx * T), min(X1, tx * T + T)
                where_to = out[ya - Y0:yb - Y0, xa - X0:xb - X0]
                with_ = state[ty, tx]
                if with_ == 2:
                    where_to[...] = arena[ty, tx, ya - ty * T:yb - ty * T, xa - tx * T:xb - tx * T]
                else:
                    where_to[...] = empty if with_ == 0 else number[ty, tx]
        return out

    def __getitem__(self, option):
        brush = self._option(option)
        if brush is None:
            if isinstance(option, tuple) and len(option) == 2 and all(isinstance(k, np.ndarray) and k.dtype.kind in 'iu' for k in option):
                iy, ix = option                          # индексы точек: собрать их прямоугольник
                if iy.size == 0 or ix.size == 0:
                    return np.asarray(self)[option]
                y0, x0, y1, x1 = int(iy.min()), int(ix.min()), int(iy.max()) + 1, int(ix.max()) + 1
                if y0 >= 0 and x0 >= 0 and y1 <= self.shape[0] and x1 <= self.shape[1]:
                    return self._collect(y0, y1, x0, x1)[iy - y0, ix - x0]
            return np.asarray(self)[option]
        y0, y1, x0, x1, spacing, tail, compress = brush
        a = self._collect(y0, y1, x0, x1)
        if spacing != (1, 1):
            a = a[::spacing[0], ::spacing[1]]
        if tail:
            a = a[(slice(None), slice(None)) + tail]
        if compress:
            a = a.reshape(tuple(n for i, n in enumerate(a.shape) if i not in compress))
        return a

    def __setitem__(self, option, v):
        brush = self._option(option)
        if brush is None or brush[5] or brush[4] != (1, 1):          # маска, индексы, шаг, третья ось: через собранное поле
            a = np.asarray(self)
            a[option] = v
            self._write(0, self.shape[0], 0, self.shape[1], a)
            return
        y0, y1, x0, x1, _, _, compress = brush
        shape = (y1 - y0, x1 - x0) + self.shape[2:]
        if compress:
            narrowing = tuple(n for i, n in enumerate(shape) if i not in compress)
            v = np.broadcast_to(np.asarray(v, np.float32), narrowing).reshape(shape)
        else:
            v = np.broadcast_to(np.asarray(v, np.float32), shape)
        self._write(y0, y1, x0, x1, v)

    def _tiles(self, y0, y1, x0, x1):
        """Плитки прямоугольника окна: (ty, tx, срез в плитке, срез в прямоугольнике, плитка ли целиком)."""
        T = PLAndTBrushA
        oy, ox = self._oy, self._ox
        H, W = self._full
        Y0, Y1, X0, X1 = y0 + oy, y1 + oy, x0 + ox, x1 + ox
        for ty in range(Y0 // T, (Y1 - 1) // T + 1):
            ya, yb = max(Y0, ty * T), min(Y1, ty * T + T)
            for tx in range(X0 // T, (X1 - 1) // T + 1):
                xa, xb = max(X0, tx * T), min(X1, tx * T + T)
                all = ya == ty * T and xa == tx * T and yb == min(ty * T + T, H) and xb == min(tx * T + T, W)
                yield (ty, tx, (slice(ya - ty * T, yb - ty * T), slice(xa - tx * T, xb - tx * T)),
                       (slice(ya - Y0, yb - Y0), slice(xa - X0, xb - X0)), all)

    def _write(self, y0, y1, x0, x1, v):
        """Записать v (форма прямоугольника, float32) в прямоугольник окна."""
        if y1 <= y0 or x1 <= x0:
            return
        for ty, tx, in_tile, in_piece, all in self._tiles(y0, y1, x0, x1):
            self._in_tile(ty, tx, in_tile, v[in_piece], all)

    def _in_tile(self, ty, tx, in_tile, part, all):
        # состояния: 0 — нетронутая (в арене нули), 1 — ровная (в арене нули), 2 — своя, 3 — ровная, в арене прежнее
        with_ = self._state[ty, tx]
        if with_ != 2 or all:
            b = _bits(part)
            if all and (b == b.flat[0]).all():        # плитка переписана целиком и ровно: числом
                b0 = int(b.flat[0])
                if b0 == _bits(self.empty) and with_ == 0:
                    return
                self._number[ty, tx] = np.uint32(b0).view(np.float32)
                self._state[ty, tx] = 3 if with_ >= 2 else 1
                return
            if with_ != 2:
                was = _bits(self._number[ty, tx] if with_ in (1, 3) else self.empty)
                if (b == was).all():
                    return                              # пишут то, что в ровной плитке и лежит
                if with_ == 3 or was != 0:                  # в арене нетронутой и ровной (1) — нули: нулей не писать
                    self._arena[ty, tx] = np.uint32(was).view(np.float32)
                self._state[ty, tx] = 2
        self._arena[ty, tx][in_tile] = part

    def elementwise(self, where, f):
        """Применить к прямоугольнику `где` (срезы без шага) функцию f поэлементно (вычесть, обрезать…) — по плиткам,
        без сборки поля; ровная плитка остаётся числом."""
        y0, y1, x0, x1, spacing, tail, compress = self._option(where)
        assert spacing == (1, 1) and not tail and not compress
        if y1 <= y0 or x1 <= x0:
            return
        for ty, tx, in_tile, in_piece, all in self._tiles(y0, y1, x0, x1):
            with_ = self._state[ty, tx]
            if with_ == 2:
                p = self._arena[ty, tx][in_tile]
                p[...] = f(p)
                continue
            s = np.asarray(f(np.full((1, 1) + self.shape[2:], self.empty if with_ == 0 else self._number[ty, tx], np.float32)), np.float32)
            n = (in_tile[0].stop - in_tile[0].start, in_tile[1].stop - in_tile[1].start)
            self._in_tile(ty, tx, in_tile, np.broadcast_to(s, n + self.shape[2:]), all)


def elementwise(field, where, f):
    """f поэлементно на прямоугольнике `где` поля: у плиток — по плиткам, у массива (участок) — на месте."""
    if isinstance(field, Tiles):
        field.elementwise(where, f)
    else:
        w = field[where]
        w[...] = f(w)
