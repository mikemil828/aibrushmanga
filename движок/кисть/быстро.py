"""Е-3: те же действия numpy без обёрток — на мазок их тысячи, и в ¼ обёртка стоит дороже счёта.

`зажать(a, lo, hi)` — `np.clip(a, lo, hi)` бит в бит: тот же ufunc `clip`, что зовёт `np.clip` (и `minimum` / `maximum`,
когда граница None), без проверок `_wrapfunc` и `_clip`. Целые массивы — через `np.clip` (у него свои правила границ)."""
import numpy as np

try:
    from numpy._core.umath import clip as _clip
except ImportError:                       # numpy 1.x
    from numpy.core.umath import clip as _clip


def зажать(a, lo, hi):
    if lo is None or hi is None or (type(a) is np.ndarray and a.dtype.kind in 'iub'):
        return np.clip(a, lo, hi)
    return _clip(a, lo, hi)


def _пары(ш, n):
    if np.ndim(ш) == 0:
        return ((int(ш), int(ш)),) * n
    ш = tuple(ш)
    if np.ndim(ш[0]) == 0:
        return ((int(ш[0]), int(ш[1])),) * n
    return tuple((int(д), int(п)) for д, п in ш)


def обложить(a, ш, значение=0, край=False):
    """`np.pad(a, ш)` (`constant_values=значение`) или `np.pad(a, ш, mode='edge')` при `край` — бит в бит, без разбора
    режимов `np.pad`: на мазок их сотни."""
    a = np.asarray(a)
    ш = _пары(ш, a.ndim)
    форма = tuple(n + д + п for n, (д, п) in zip(a.shape, ш))
    b = np.empty(форма, a.dtype) if край else np.full(форма, значение, a.dtype)
    b[tuple(slice(д, д + n) for n, (д, _) in zip(a.shape, ш))] = a
    if край:
        for ось, (n, (д, п)) in enumerate(zip(a.shape, ш)):
            до = [slice(None)] * a.ndim
            if д:
                до[ось] = slice(0, д)
                с = list(до); с[ось] = slice(д, д + 1)
                b[tuple(до)] = b[tuple(с)]
            if п:
                до[ось] = slice(д + n, None)
                с = list(до); с[ось] = slice(д + n - 1, д + n)
                b[tuple(до)] = b[tuple(с)]
    return b
