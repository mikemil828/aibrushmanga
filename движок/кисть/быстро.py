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


# Е-2: выборка полей хода по отсчётам следа (ii, jj — `np.nonzero` следа (N, nm), г — гребень волоска, i0/i1 — шаг
# переноса шага пути). `A[ii, jj]`, `A[i[ii], г[jj]]` и `(A[i0] * f0 + A[i1] * f1)[:, г][ii, jj]` — те же числа одним
# `take` по плоскому номеру, без промежуточного поля (N, nm): копия, а не счёт — бит в бит. Плоские номера одного хода
# считаются раз на ход (сверка по самим массивам, `is`; держится не больше 24 наборов).
_ОТС = {}


def _номера(ii, jj, ключ, массивы, счёт):
    к = (id(ii), id(jj)) + ключ
    з = _ОТС.get(к)
    if з is None or з[0] is not ii or з[1] is not jj or any(a is not b for a, b in zip(з[2], массивы)):
        if len(_ОТС) >= 24:
            _ОТС.clear()
        з = _ОТС[к] = (ii, jj, массивы, счёт())
    return з[3]


def по_отсчётам(A, ii, jj):
    """`A[ii, jj]` поля (N, m) или (N, m, C)."""
    if type(A) is not np.ndarray or A.ndim < 2 or A.shape[0] * A.shape[1] == 0:
        return A[ii, jj]
    m = A.shape[1]
    ф, вне = _номера(ii, jj, ('отс', m), (), lambda: (ii * m + jj, len(jj) and (int(jj.max()) >= m or int(jj.min()) < 0)))
    if вне:
        return A[ii, jj]
    if A.ndim == 2:
        return np.ravel(A).take(ф)
    return A.reshape((A.shape[0] * m,) + A.shape[2:]).take(ф, axis=0)


def по_гребню(A, i, г, ii, jj):
    """`A[i[ii], г[jj]]` поля шагов переноса (Nc, nr) или (Nc, nr, C)."""
    if type(A) is not np.ndarray or A.ndim < 2:
        return A[i[ii], г[jj]]
    nr = A.shape[1]
    ф, вне = _номера(ii, jj, ('гр', nr, id(i), id(г)), (i, г), lambda: (lambda гj: (i[ii] * nr + гj, len(гj) and (int(гj.max()) >= nr or int(гj.min()) < 0)))(г[jj]))
    if вне:
        return A[i[ii], г[jj]]
    if A.ndim == 2:
        return np.ravel(A).take(ф)
    return A.reshape((A.shape[0] * nr,) + A.shape[2:]).take(ф, axis=0)


def меж_шагов(A, i0, i1, f0, f1, г, ii, jj):
    """`(A[i0] * f0 + A[i1] * f1)[:, г][ii, jj]` поля шагов переноса A (Nc, nr): шаг пути — меж двух шагов переноса."""
    return по_волоскам(A[i0] * f0 + A[i1] * f1, г, ii, jj)


def по_волоскам(T, г, ii, jj):
    """`T[:, г][ii, jj]` поля шагов пути по гребням T (N, nr)."""
    if type(T) is not np.ndarray or T.ndim != 2:
        return T[:, г][ii, jj]
    nr = T.shape[1]
    ф, вне = _номера(ii, jj, ('меж', nr, id(г)), (г,), lambda: (lambda гj: (ii * nr + гj, len(гj) and (int(гj.max()) >= nr or int(гj.min()) < 0)))(г[jj]))
    return T[:, г][ii, jj] if вне else np.ravel(T).take(ф)
