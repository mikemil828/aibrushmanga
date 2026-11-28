"""Humanize one stroke record: the shared step of the journal patch (`manga.humanize.patch`) and of live drawing
(`Sheet.humanize()`), so the two give the same page."""
import copy
import math

import numpy as np

from .. import geom
from ..settings import configurable, section
from .motion import Person, arcs
from .state import LiveHand

FROM = 'humanize_from'          # field of a humanized stroke: the stroke's own fields before the hand drew it
OP = 'humanize'                 # journal line: who drew, with what strength (replay skips it)
SPARSE_PX = 0.2                 # a humanized path is written with fewer points: none dropped strays more than this
SHORT_MM = 0.5                  # strokes shorter than this are touches in place: left as they are
MAX_TARGETS = 10                # the hand aims at no more than this many targets along one stroke

# what each part of a drawing is worth to the hand when the settings name nothing (strength multiplier, interest)
PARTS = {'eyes': 0.25, 'face': 0.5, 'hair': 0.8, 'body': 1.0, 'background': 1.0, 'effects': 0.3}
INTEREST = {'eyes': 1.0, 'face': 0.85, 'hair': 0.65, 'body': 0.5, 'background': 0.25, 'effects': 0.35}
PENS = {'fineliner': 0.0, 'border': 0.0, 'marker': 0.5}


@configurable('humanize')
def config(enabled=False, strength=0.15, person=1, seed=None, max_drift_mm=2.0, ruled=0.0, parts=None,
           interest=None, pens=None):
    """The [humanize] settings, merged: the tables [humanize.parts], [humanize.interest] and [humanize.pens] are
    laid over the engine's own tables. `seed` None = the sheet's seed."""
    sec = section('humanize')
    tables = {}
    for name, base, given in (('parts', PARTS, parts), ('interest', INTEREST, interest), ('pens', PENS, pens)):
        t = dict(base)
        t.update({k: float(v) for k, v in (sec.get(name) or {}).items()})
        t.update({k: float(v) for k, v in (given or {}).items()})
        tables[name] = t
    if float(strength) < 0 or not math.isfinite(float(strength)):
        raise ValueError('humanize strength is a number from 0: 0 = off, 0.15 = default, 0.3 = the whole shape from the hand')
    return dict(enabled=bool(enabled), strength=float(strength), person=int(person),
                seed=None if seed is None else int(seed), max_drift_mm=float(max_drift_mm), ruled=float(ruled),
                **tables)


def header(cfg, seed, who):
    return {'op': OP, 'person': cfg['person'], 'seed': seed, 'strength': cfg['strength'],
            'max_drift_mm': cfg['max_drift_mm'], 'ruled': cfg['ruled'], 'parts': cfg['parts'],
            'interest': cfg['interest'], 'pens': cfg['pens'], 'who': who}


def cfg_from_header(d):
    return dict(enabled=True, strength=float(d['strength']), person=int(d['person']), seed=int(d['seed']),
                max_drift_mm=float(d['max_drift_mm']), ruled=float(d['ruled']), parts=dict(d['parts']),
                interest=dict(d['interest']), pens=dict(d['pens']))


# ---------------------------------------------------------------------------------------------- geometry
def _frac(xy):
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    return s / max(s[-1], 1e-9)


def _bends(P):
    """Targets read as a smooth curve through them: a leg's bend = the turn of the tangent between its ends, deg."""
    n = len(P)
    tan = np.zeros((n, 2))
    tan[0], tan[-1] = P[1] - P[0], P[-1] - P[-2]
    for i in range(1, n - 1):
        tan[i] = P[i + 1] - P[i - 1]
    a = np.arctan2(tan[:, 1], tan[:, 0])
    return [math.degrees((a[j + 1] - a[j] + math.pi) % (2 * math.pi) - math.pi) for j in range(n - 1)]


def _simplify(P, tol):
    """Indices of the points a polyline needs to stay within `tol` of the path (Douglas-Peucker)."""
    keep = np.zeros(len(P), bool)
    keep[[0, -1]] = True
    stack = [(0, len(P) - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        d = P[b] - P[a]
        L = np.hypot(*d)
        Q = P[a + 1:b] - P[a]
        off = np.abs(Q[:, 0] * d[1] - Q[:, 1] * d[0]) / L if L > 1e-9 else np.hypot(*Q.T)
        i = int(np.argmax(off))
        if off[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return np.nonzero(keep)[0]


def _densify(P, step=1.0):
    """A polyline with extra points on long legs (vertices kept), so a hand's drift can bend its legs."""
    out = [P[:1]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b[:2] - a[:2])) / step)))
        f = np.linspace(0, 1, n + 1)[1:, None]
        out.append(a + (b - a) * f)
    return np.vstack(out)


def _sparse(A, tol=SPARSE_PX):
    """Fewer points for a path (x, y, pressure): keep those without which the line between neighbours strays more
    than `tol` (pressure 0.01 counts as 0.2 px)."""
    n = len(A)
    if n < 4:
        return A
    F = np.column_stack([A[:, 0], A[:, 1], A[:, 2] * 20.0])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(A[:, :2], axis=0).T))])
    keep = np.zeros(n, bool)
    keep[[0, -1]] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        w = ((s[a + 1:b] - s[a]) / max(s[b] - s[a], 1e-9))[:, None]
        off = np.abs(F[a + 1:b] - (F[a] + w * (F[b] - F[a]))).max(axis=1)
        i = int(np.argmax(off))
        if off[i] > tol:
            m = a + 1 + i
            keep[m] = True
            stack += [(a, m), (m, b)]
    return A[keep]


def _limit(dx, dy, limit, drift=None, where=None):
    """Offsets (dx, dy) softly capped at `limit`; `drift` (the eye's memory) moves their mean toward the drift of
    strokes already lying under the path `where`."""
    if drift is not None and len(dx):
        m = np.array([dx.mean(), dy.mean()])
        m2 = drift(where, m)
        dx, dy = dx + (m2[0] - m[0]), dy + (m2[1] - m[1])
    r = np.hypot(dx, dy)
    if limit is None:
        return dx, dy, r
    limit = max(float(limit), 1e-9)
    k = np.where(r > 1e-9, limit * np.tanh(r / limit) / np.maximum(r, 1e-9), 1.0)
    return dx * k, dy * k, r * k


def _sway(Hp, u, g):
    """How much the hand rocks the pressure at path fraction u (0-1): around its own mean, 1 at the ends."""
    body = Hp > 0.5 * Hp.max()
    k = np.clip(Hp / max(float(np.median(Hp[body])), 1e-6), 0.6, 1.4)
    ends = np.clip(np.minimum(u, 1 - u) / 0.12, 0, 1)            # entry and exit stay as the pen draws them
    return (1 + (k - 1) * ends) ** g


def intent(E, ppm, pen_mm, pressure):
    """An engine path E (x, y, pressure; base px) as the hand's intent: a few targets (a polyline within 0.5 mm
    or coarser until at most MAX_TARGETS), bends from the neighbours. Returns (intent, target indices) or None."""
    P = E[:, :2] / ppm
    tol = 0.5
    idx = _simplify(P, tol)
    while len(idx) > MAX_TARGETS:
        tol *= 1.6
        idx = _simplify(P, tol)
    keep = np.concatenate([[True], np.hypot(*np.diff(P[idx], axis=0).T) > 0.05])
    if not keep[-1] and len(idx) > 2:                 # the stroke came back to its start: keep the end as a target
        keep[-1], keep[-2] = True, False
    idx = idx[keep]
    if len(idx) < 2 or idx[-1] != len(P) - 1 or (np.hypot(*np.diff(P[idx], axis=0).T) <= 0.05).any():
        return None
    T = P[idx]
    st = dict(points=[tuple(q) for q in T], bend=_bends(T), tolerance=float(np.clip(0.3 * pen_mm, 0.6, 4.0)),
              pressure=float(np.clip(pressure / 0.8, 0.3, 1.25)))
    return st, idx


def displace(E, idx, st, pieces, strength, ppm, limit_mm, drift=None):
    """The engine path E (base px) through the hand. Each point gets what the hand's stroke (pieces, mm) did
    differently from its own intent (arcs through its targets) at the same fraction of the path: drift, the
    lever's arc, a cut corner, falling short. The offset grows with strength up to 0.3 and is capped at
    `limit_mm`. Pressure rocks around the engine's (unchanged at the ends). Same number of points."""
    g = float(np.clip(strength / 0.3, 0.0, 1.0))
    H = np.vstack(pieces)
    P = E[:, :2] / ppm
    T, bend = np.asarray(st['points'], float), st['bend']
    n, m = len(P), len(idx) - 1
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
    j = np.clip(np.searchsorted(idx, np.arange(n), side='right') - 1, 0, m - 1)
    f = (s - s[idx[j]]) / np.maximum(s[idx[j + 1]] - s[idx[j]], 1e-9)
    chords = np.hypot(*np.diff(T, axis=0).T)
    half = np.radians(np.asarray(bend, float)) / 2
    Lc = chords * np.where(np.abs(half) > 1e-4, half / np.sin(np.where(np.abs(half) > 1e-4, half, 1.0)), 1.0)
    cum = np.concatenate([[0.0], np.cumsum(Lc)])
    u = (cum[j] + f * Lc[j]) / max(cum[-1], 1e-9)
    plan = np.empty((n, 2))
    for k in range(m):
        here = j == k
        if here.any():
            plan[here] = arcs(T[k], T[k + 1], bend[k], f[here])
    uH = _frac(H[:, :2])
    dx, dy, r = _limit(g * (np.interp(u, uH, H[:, 0]) - plan[:, 0]), g * (np.interp(u, uH, H[:, 1]) - plan[:, 1]),
                       limit_mm, drift, P)
    out = E.copy()
    out[:, 0] += dx * ppm
    out[:, 1] += dy * ppm
    out[:, 2] = np.clip(E[:, 2] * np.interp(u, uH, _sway(H[:, 2], uH, g)), 0, 1)
    return out, float(np.median(r))


# ---------------------------------------------------------------------------------------------- the hand
class Humanizer:
    """One hand for one sheet: the person, its live state, the parts of the drawing it is in."""

    def __init__(self, cfg, seed, ppm):
        self.cfg = cfg
        self.seed = int(seed)
        self.ppm = float(ppm)
        self.person = Person(cfg['person'])
        self.hand = LiveHand(self.person, seed=self.seed, strength=cfg['strength'])
        self.parts = []                    # innermost last
        self.map = None                    # (names, label image at base px): parts by place on the page
        self.drawn = 0
        self.left = 0
        self.offsets = []                  # median offset of each humanized stroke, mm

    def set_map(self, names, labels):
        self.map = (list(names), np.asarray(labels)) if names else None

    def part(self, rec=None):
        """The part a stroke belongs to: the innermost `sh.part()` block, else the part map at the stroke's points
        (the part most of its points lie in), else None."""
        if self.parts:
            return self.parts[-1]
        if self.map is None or rec is None:
            return None
        names, lab = self.map
        P = np.asarray(rec['points'], float)
        if P.ndim != 2 or not len(P):
            return None
        x = np.clip(np.round(P[:, 0]).astype(int), 0, lab.shape[1] - 1)
        y = np.clip(np.round(P[:, 1]).astype(int), 0, lab.shape[0] - 1)
        votes = np.bincount(lab[y, x].astype(int), minlength=len(names) + 1)
        k = int(np.argmax(votes))
        return names[k - 1] if k > 0 else None

    def enter(self, name):
        self.parts.append(name)
        self._pause()

    def leave(self):
        if self.parts:
            self.parts.pop()
        self._pause()

    def _pause(self):
        self.hand.s.pause(0.5)
        self.hand.hand.at = None

    def strength_for(self, rec, part):
        c = self.cfg
        pen = rec['pen']['name'] if isinstance(rec.get('pen'), dict) else str(rec.get('pen'))
        k = c['strength'] * c['pens'].get(pen, 1.0)
        if part is not None:
            k *= c['parts'].get(part, 1.0)
        taper = rec.get('taper')
        if len(rec['points']) == 2 and taper is not None and not any(taper):
            k *= c['ruled']                    # a straight line with blunt ends: ruled, not freehand
        return k

    def apply(self, rec):
        """A stroke record (as Sheet.stroke journals it) -> the record the hand drew, or None to leave it as is."""
        part = self.part(rec)
        s = self.strength_for(rec, part)
        if s <= 0:
            self.left += 1
            return None
        P = np.asarray(rec['points'], float)
        if P.ndim != 2 or len(P) < 2:
            self.left += 1
            return None
        if P.shape[1] < 3:
            P = np.hstack([P[:, :2], np.full((len(P), 1), float(rec['pressure']))])
        P = P[:, :3]
        E = geom.catmull(P, per=10) if rec.get('smooth', True) and len(P) > 2 else _densify(P)
        if float(np.sum(np.hypot(*np.diff(E[:, :2], axis=0).T))) < SHORT_MM * self.ppm:
            self.left += 1
            return None
        pen_mm = float(rec['pen'].get('max_mm', 0.5)) if isinstance(rec.get('pen'), dict) else 0.5
        p = E[:, 2]
        body = float(np.median(p[p > 0.5 * p.max()])) if p.max() > 0 else 0.8
        got = intent(E, self.ppm, pen_mm, body)
        if got is None:
            self.left += 1
            return None
        st, idx = got
        st['interest'] = self.cfg['interest'].get(part, 0.5) if part is not None else 0.5
        st['role'] = (part, rec['pen']['name'] if isinstance(rec.get('pen'), dict) else rec.get('pen'))
        lh = self.hand
        if lh.time_to_rest():
            lh.s.new_session()
            lh.hand.at = None
        lh.strength = s
        pieces, r = lh.stroke(st)
        lh.saw(r)
        if not pieces or not all(np.isfinite(q).all() for q in pieces):
            self.left += 1
            return None
        B, off = displace(E, idx, st, pieces, s, self.ppm, self.cfg['max_drift_mm'] * s, lh.drift)
        if not np.isfinite(B).all():
            self.left += 1
            return None
        B = _sparse(B)
        out = {k: v for k, v in rec.items()}
        out['points'] = np.round(B, 3).tolist()
        out['smooth'] = False
        out[FROM] = {'points': rec['points'], 'smooth': rec.get('smooth', True)}
        if part is not None:
            out['part'] = part
        self.drawn += 1
        self.offsets.append(off)
        return out


def strip(rec):
    """A humanized stroke record -> the record as it was before the hand drew it."""
    if FROM not in rec:
        return rec
    out = {k: v for k, v in rec.items() if k not in (FROM, 'part')}
    out.update(copy.deepcopy(rec[FROM]))
    return out


def in_part(name):
    """Decorator for a drawing helper whose first argument is the sheet: what it draws is part `name` (a part
    the caller named inside it still wins). Only humanize reads parts; the drawing doesn't change."""
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(sh, *args, **kw):
            part = getattr(sh, 'part', None)
            if part is None:
                return fn(sh, *args, **kw)
            with part(name):
                return fn(sh, *args, **kw)
        return wrapper
    return deco
