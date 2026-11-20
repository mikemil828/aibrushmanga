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

ПЛИТКА = 64          # сторона плитки, px: «роса» стенда 2в — 11 МБ полей; меньше плитка — больше счёта склейки


def _биты(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def _бит(v):
    return int(np.float32(v).view(np.uint32))


def _память(форма, тип, общая):
    """Нули: свои или в общей памяти процессов (анонимный mmap): страницы берутся, когда в них пишут."""
    n = int(np.prod(форма)) * np.dtype(тип).itemsize
    if n < 1 << 16:
        return _память_мал(форма, тип, общая)
    # своя — тоже mmap, не np.zeros: numpy советует системе большие страницы (2 МБ) — одна плитка взяла бы 2 МБ
    м = mmap.mmap(-1, n, flags=mmap.MAP_SHARED if общая else mmap.MAP_PRIVATE)
    return np.frombuffer(м, dtype=тип, count=int(np.prod(форма))).reshape(форма)


def _память_мал(форма, тип, общая):
    if not общая:
        return np.zeros(форма, тип)
    n = int(np.prod(форма)) * np.dtype(тип).itemsize
    return np.frombuffer(mmap.mmap(-1, max(n, 1)), dtype=тип, count=int(np.prod(форма))).reshape(форма)


def по_сетке(r):
    """Прямоугольник (x0, y0, x1, y1) -> он же, расширенный до границ плиток."""
    T = ПЛИТКА
    return (r[0] // T * T, r[1] // T * T, -(-r[2] // T) * T, -(-r[3] // T) * T)


class Плитки(NDArrayOperatorsMixin):
    """Поле слоя (В, Ш) или (В, Ш, 3) float32 плитками ПЛИТКА×ПЛИТКА (шапка модуля). `байт` — сколько памяти полей
    занято на деле; `окно(гy, гx)` — вид на прямоугольник (та же память, свои координаты)."""

    __array_priority__ = 20
    dtype = np.dtype(np.float32)

    def __init__(self, форма, пусто=0.0, общая=False):
        T = ПЛИТКА
        форма = tuple(int(v) for v in форма)
        self.shape = форма
        self.пусто = np.float32(пусто)
        self.общая = bool(общая)
        self._oy = self._ox = 0
        self._полн = форма[:2]
        н = (-(-форма[0] // T), -(-форма[1] // T))
        self._сост = _память(н, np.uint8, общая)
        self._число = _память(н, np.float32, общая)
        self._арена = _память(н + (T, T) + форма[2:], np.float32, общая)

    # ---------------------------------------------------------------- вид и копии
    def окно(self, гy, гx):
        """Вид на прямоугольник поля (срезы без шага): пишет в ту же память."""
        y0, y1, _ = гy.indices(self.shape[0])
        x0, x1, _ = гx.indices(self.shape[1])
        в = object.__new__(Плитки)
        в.__dict__.update(self.__dict__)
        в._oy, в._ox = self._oy + y0, self._ox + x0
        в.shape = (max(y1 - y0, 0), max(x1 - x0, 0)) + self.shape[2:]
        return в

    def __getstate__(self):
        if (self._oy, self._ox) != (0, 0) or self.shape[:2] != self._полн:
            return Плитки.из(np.asarray(self), self.пусто).__getstate__()
        сост = np.array(self._сост)
        сост[сост == 3] = 1                          # копия: прежнего в арене у неё нет
        return {'shape': self.shape, 'пусто': float(self.пусто), 'сост': сост, 'число': np.array(self._число),
                'плитки': self._арена[сост == 2]}

    def __setstate__(self, д):
        self.__init__(д['shape'], д['пусто'])
        self._сост[...] = д['сост']
        self._число[...] = д['число']
        self._арена[д['сост'] == 2] = д['плитки']

    def __deepcopy__(self, memo):
        н = object.__new__(Плитки)
        н.__setstate__(self.__getstate__())
        return н

    def своя(self, общая=False):
        """Копия поля: своя память (общая=False) или общая память процессов."""
        д = self.__getstate__()
        н = Плитки(д['shape'], д['пусто'], общая)
        н._сост[...] = д['сост']
        н._число[...] = д['число']
        н._арена[д['сост'] == 2] = д['плитки']
        return н

    def вырезка(self, гy, гx):
        """2г: копия прямоугольника (срезы без шага) плитками той же сетки: своя память — только его плитки (тронутые —
        копией, ровные — числом); ответ — окно на прямоугольник, координаты окна — как у среза."""
        T = ПЛИТКА
        y0, y1, _ = гy.indices(self.shape[0])
        x0, x1, _ = гx.indices(self.shape[1])
        y1, x1 = max(y1, y0), max(x1, x0)
        Y0, Y1, X0, X1 = y0 + self._oy, y1 + self._oy, x0 + self._ox, x1 + self._ox
        ty0, tx0 = Y0 // T, X0 // T
        ty1, tx1 = max(-(-Y1 // T), ty0), max(-(-X1 // T), tx0)
        н = Плитки((max(min(ty1 * T, self._полн[0]) - ty0 * T, 0), max(min(tx1 * T, self._полн[1]) - tx0 * T, 0)) + self.shape[2:],
                   self.пусто)
        с = np.array(self._сост[ty0:ty1, tx0:tx1])
        с[с == 3] = 1                                  # копия: прежнего в арене у неё нет
        н._сост[...] = с
        н._число[...] = self._число[ty0:ty1, tx0:tx1]
        н._арена[с == 2] = self._арена[ty0:ty1, tx0:tx1][с == 2]
        return н.окно(slice(Y0 - ty0 * T, Y1 - ty0 * T), slice(X0 - tx0 * T, X1 - tx0 * T))

    def основа(self):
        """2г: всё поле, на которое смотрит окно (своё — оно само): (поле, сдвиг окна по y, по x)."""
        о = object.__new__(Плитки)
        о.__dict__.update(self.__dict__)
        о._oy = о._ox = 0
        о.shape = tuple(self._полн) + self.shape[2:]
        return о, self._oy, self._ox

    def есть_вне(self, гy, гx):
        """2г: есть ли не ноль в поле вне прямоугольника (срезы без шага) — по плиткам, без сборки поля."""
        T = ПЛИТКА
        y0, y1, _ = гy.indices(self.shape[0])
        x0, x1, _ = гx.indices(self.shape[1])
        H, W = self.shape[:2]
        oy, ox = self._oy, self._ox
        for ty in range(oy // T, (oy + H - 1) // T + 1) if H and W else ():
            for tx in range(ox // T, (ox + W - 1) // T + 1):
                ya, yb = max(ty * T, oy) - oy, min(ty * T + T, oy + H) - oy      # плитка в координатах окна
                xa, xb = max(tx * T, ox) - ox, min(tx * T + T, ox + W) - ox
                if y0 <= ya and yb <= y1 and x0 <= xa and xb <= x1:
                    continue                                                    # вся внутри прямоугольника
                с = self._сост[ty, tx]
                if с != 2:
                    if (self.пусто if с == 0 else self._число[ty, tx]) != 0:
                        return True
                    continue
                п = self._арена[ty, tx][ya + oy - ty * T:yb + oy - ty * T, xa + ox - tx * T:xb + ox - tx * T]
                внутри = np.zeros(п.shape[:2], bool)
                внутри[max(y0 - ya, 0):max(y1 - ya, 0), max(x0 - xa, 0):max(x1 - xa, 0)] = True
                if (п[~внутри] != 0).any():
                    return True
        return False

    @classmethod
    def из(cls, a, пусто=0.0, общая=False):
        """Поле плитками из массива (В, Ш[, 3]): плитки, равные `пусто`, — без памяти, ровные — числом."""
        a = np.asarray(a, np.float32)
        п = cls(a.shape, пусто, общая)
        п._писать(0, a.shape[0], 0, a.shape[1], a)
        return п

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
    def байт(self):
        return int((self._сост >= 2).sum()) * self._арена[0, 0].nbytes + self._сост.nbytes + self._число.nbytes

    def __len__(self):
        return self.shape[0]

    def __repr__(self):
        return f'Плитки({self.shape}, своих плиток {int((self._сост == 2).sum())}, {self.байт / 1e6:.1f} МБ)'

    def __array__(self, dtype=None, copy=None):
        a = self._собрать(0, self.shape[0], 0, self.shape[1])
        return a if dtype is None else a.astype(dtype, copy=False)

    def copy(self):
        return np.asarray(self)

    def __getattr__(self, имя):       # .sum(), .max(), .astype(), .any(axis=…) … — у собранного поля
        if имя.startswith('_'):
            raise AttributeError(имя)
        return getattr(np.asarray(self), имя)

    def __iter__(self):
        return iter(np.asarray(self))

    def __array_ufunc__(self, ufunc, method, *входы, out=None, **kw):
        входы = tuple(np.asarray(v) if isinstance(v, Плитки) else v for v in входы)
        if out is None:
            return getattr(ufunc, method)(*входы, **kw)
        out2 = tuple(np.asarray(o) if isinstance(o, Плитки) else o for o in out)
        r = getattr(ufunc, method)(*входы, out=out2, **kw)
        if not any(isinstance(o, Плитки) for o in out):
            return r
        for o, o2 in zip(out, out2):
            if isinstance(o, Плитки):
                o[...] = o2
        return out[0] if len(out) == 1 else out

    def any(self, *a, **kw):
        if a or kw or (self._oy, self._ox) != (0, 0) or self.shape[:2] != self._полн:
            return bool(np.asarray(self).any(*a, **kw)) if not (a or kw) else np.asarray(self).any(*a, **kw)
        с = self._сост
        if self.пусто != 0 and (с == 0).any():
            return True
        if (self._число[(с == 1) | (с == 3)] != 0).any():
            return True
        return bool(self._арена[с == 2].any())

    # ---------------------------------------------------------------- плитки
    def _ключ(self, ключ):
        """Ключ среза -> (y0, y1, x0, x1, шаг, хвост, сжать) прямоугольника в координатах окна; None — не прямоугольник."""
        if not isinstance(ключ, tuple):
            ключ = (ключ,)
        if any(k is Ellipsis for k in ключ):
            i = next(i for i, k in enumerate(ключ) if k is Ellipsis)
            лишних = self.ndim - (len(ключ) - 1 - sum(k is None for k in ключ))
            ключ = ключ[:i] + (slice(None),) * max(лишних, 0) + ключ[i + 1:]
            if any(k is Ellipsis for k in ключ):
                return None
        оси, шаг, сжать = [], [], []
        for ось in range(2):
            k = ключ[ось] if ось < len(ключ) else slice(None)
            n = self.shape[ось]
            if isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
                k = int(k) + (n if k < 0 else 0)
                if not 0 <= k < n:
                    raise IndexError(f'индекс {k} вне поля {n}')
                оси.append((k, k + 1))
                шаг.append(1)
                сжать.append(ось)
            elif isinstance(k, slice) and (k.step is None or k.step > 0):
                a, b, s = k.indices(n)
                оси.append((a, max(a, b)))
                шаг.append(s)
            else:
                return None
        хвост = ключ[2:]
        if хвост == (slice(None),) * len(хвост) and len(хвост) <= self.ndim - 2:
            хвост = ()
        return оси[0][0], оси[0][1], оси[1][0], оси[1][1], tuple(шаг), хвост, tuple(сжать)

    def _собрать(self, y0, y1, x0, x1):
        """Прямоугольник окна -> массив (своя память)."""
        T = ПЛИТКА
        out = np.empty((max(y1 - y0, 0), max(x1 - x0, 0)) + self.shape[2:], np.float32)
        if y1 <= y0 or x1 <= x0:
            return out
        oy, ox = self._oy, self._ox
        Y0, Y1, X0, X1 = y0 + oy, y1 + oy, x0 + ox, x1 + ox
        сост, число, арена, пусто = self._сост, self._число, self._арена, self.пусто
        for ty in range(Y0 // T, (Y1 - 1) // T + 1):
            ya, yb = max(Y0, ty * T), min(Y1, ty * T + T)
            for tx in range(X0 // T, (X1 - 1) // T + 1):
                xa, xb = max(X0, tx * T), min(X1, tx * T + T)
                куда = out[ya - Y0:yb - Y0, xa - X0:xb - X0]
                с = сост[ty, tx]
                if с == 2:
                    куда[...] = арена[ty, tx, ya - ty * T:yb - ty * T, xa - tx * T:xb - tx * T]
                else:
                    куда[...] = пусто if с == 0 else число[ty, tx]
        return out

    def __getitem__(self, ключ):
        к = self._ключ(ключ)
        if к is None:
            if isinstance(ключ, tuple) and len(ключ) == 2 and all(isinstance(k, np.ndarray) and k.dtype.kind in 'iu' for k in ключ):
                iy, ix = ключ                          # индексы точек: собрать их прямоугольник
                if iy.size == 0 or ix.size == 0:
                    return np.asarray(self)[ключ]
                y0, x0, y1, x1 = int(iy.min()), int(ix.min()), int(iy.max()) + 1, int(ix.max()) + 1
                if y0 >= 0 and x0 >= 0 and y1 <= self.shape[0] and x1 <= self.shape[1]:
                    return self._собрать(y0, y1, x0, x1)[iy - y0, ix - x0]
            return np.asarray(self)[ключ]
        y0, y1, x0, x1, шаг, хвост, сжать = к
        a = self._собрать(y0, y1, x0, x1)
        if шаг != (1, 1):
            a = a[::шаг[0], ::шаг[1]]
        if хвост:
            a = a[(slice(None), slice(None)) + хвост]
        if сжать:
            a = a.reshape(tuple(n for i, n in enumerate(a.shape) if i not in сжать))
        return a

    def __setitem__(self, ключ, v):
        к = self._ключ(ключ)
        if к is None or к[5] or к[4] != (1, 1):          # маска, индексы, шаг, третья ось: через собранное поле
            a = np.asarray(self)
            a[ключ] = v
            self._писать(0, self.shape[0], 0, self.shape[1], a)
            return
        y0, y1, x0, x1, _, _, сжать = к
        форма = (y1 - y0, x1 - x0) + self.shape[2:]
        if сжать:
            суж = tuple(n for i, n in enumerate(форма) if i not in сжать)
            v = np.broadcast_to(np.asarray(v, np.float32), суж).reshape(форма)
        else:
            v = np.broadcast_to(np.asarray(v, np.float32), форма)
        self._писать(y0, y1, x0, x1, v)

    def _плитки(self, y0, y1, x0, x1):
        """Плитки прямоугольника окна: (ty, tx, срез в плитке, срез в прямоугольнике, плитка ли целиком)."""
        T = ПЛИТКА
        oy, ox = self._oy, self._ox
        H, W = self._полн
        Y0, Y1, X0, X1 = y0 + oy, y1 + oy, x0 + ox, x1 + ox
        for ty in range(Y0 // T, (Y1 - 1) // T + 1):
            ya, yb = max(Y0, ty * T), min(Y1, ty * T + T)
            for tx in range(X0 // T, (X1 - 1) // T + 1):
                xa, xb = max(X0, tx * T), min(X1, tx * T + T)
                вся = ya == ty * T and xa == tx * T and yb == min(ty * T + T, H) and xb == min(tx * T + T, W)
                yield (ty, tx, (slice(ya - ty * T, yb - ty * T), slice(xa - tx * T, xb - tx * T)),
                       (slice(ya - Y0, yb - Y0), slice(xa - X0, xb - X0)), вся)

    def _писать(self, y0, y1, x0, x1, v):
        """Записать v (форма прямоугольника, float32) в прямоугольник окна."""
        if y1 <= y0 or x1 <= x0:
            return
        for ty, tx, в_плитке, в_куске, вся in self._плитки(y0, y1, x0, x1):
            self._в_плитку(ty, tx, в_плитке, v[в_куске], вся)

    def _в_плитку(self, ty, tx, в_плитке, часть, вся):
        # состояния: 0 — нетронутая (в арене нули), 1 — ровная (в арене нули), 2 — своя, 3 — ровная, в арене прежнее
        с = self._сост[ty, tx]
        if с != 2 or вся:
            б = _биты(часть)
            if вся and (б == б.flat[0]).all():        # плитка переписана целиком и ровно: числом
                b0 = int(б.flat[0])
                if b0 == _бит(self.пусто) and с == 0:
                    return
                self._число[ty, tx] = np.uint32(b0).view(np.float32)
                self._сост[ty, tx] = 3 if с >= 2 else 1
                return
            if с != 2:
                был = _бит(self._число[ty, tx] if с in (1, 3) else self.пусто)
                if (б == был).all():
                    return                              # пишут то, что в ровной плитке и лежит
                if с == 3 or был != 0:                  # в арене нетронутой и ровной (1) — нули: нулей не писать
                    self._арена[ty, tx] = np.uint32(был).view(np.float32)
                self._сост[ty, tx] = 2
        self._арена[ty, tx][в_плитке] = часть

    def поэлементно(self, где, f):
        """Применить к прямоугольнику `где` (срезы без шага) функцию f поэлементно (вычесть, обрезать…) — по плиткам,
        без сборки поля; ровная плитка остаётся числом."""
        y0, y1, x0, x1, шаг, хвост, сжать = self._ключ(где)
        assert шаг == (1, 1) and not хвост and not сжать
        if y1 <= y0 or x1 <= x0:
            return
        for ty, tx, в_плитке, в_куске, вся in self._плитки(y0, y1, x0, x1):
            с = self._сост[ty, tx]
            if с == 2:
                п = self._арена[ty, tx][в_плитке]
                п[...] = f(п)
                continue
            s = np.asarray(f(np.full((1, 1) + self.shape[2:], self.пусто if с == 0 else self._число[ty, tx], np.float32)), np.float32)
            n = (в_плитке[0].stop - в_плитке[0].start, в_плитке[1].stop - в_плитке[1].start)
            self._в_плитку(ty, tx, в_плитке, np.broadcast_to(s, n + self.shape[2:]), вся)


def поэлементно(поле, где, f):
    """f поэлементно на прямоугольнике `где` поля: у плиток — по плиткам, у массива (участок) — на месте."""
    if isinstance(поле, Плитки):
        поле.поэлементно(где, f)
    else:
        w = поле[где]
        w[...] = f(w)
