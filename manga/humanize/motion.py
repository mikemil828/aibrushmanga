"""How a hand moves the pen tip. Ported from the oil engine's hand module (движок/рука/движение.py, upstream
neokrasav4ik/aibrush). There is no ink physics here, only the path of the tip.

The hand gets an INTENT, not a path: targets to reach, how much each leg bends, where to press. It builds the
movement itself:

1. Throws. A stroke is a chain of ballistic throws toward the targets, each with a lognormal speed bell (the
   kinematic theory of fast human movement). The next throw starts before the last one ends: the more fluent
   the hand, the bigger the overlap, so corners round off and get cut, and the shape changes with tempo.
2. Body. A throw is a lever around a pivot (wrist, elbow), so a "straight" line comes out as an arc and the end
   drifts toward where the lever finds it easy. The pivot stays put while the hand can reach, then moves.
3. Eye. A slow stroke is not one throw but a chain of approaches of about 0.3 s, each re-aimed from where the
   hand actually is. The unevenness of a line comes from these corrections, not from added tremor. Throw error
   grows with speed: a fast line is clean but misses, a slow one is accurate but wavy.
4. Pen in hand. Pressure is not an envelope: landing and lift belong to the throws (a fluent hand lands and
   lifts on the move, a careful one sets down, presses, draws, stops); pulling toward yourself is heavier than
   pushing away, and a fast stroke is lighter.

Output: pieces of contact, arrays of (x, y, pressure, dt_ms) in mm.
"""
import math
from statistics import NormalDist

import numpy as np
from scipy.special import erf as _erf

DT = 0.002          # s
_Z = NormalDist().inv_cdf
_Q = 2.05           # a throw "lasts" between its 2 % and 98 % quantiles


def _angle(v):
    return math.atan2(v[1], v[0])


def _unit(a):
    return np.array([math.cos(a), math.sin(a)])


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Person:
    """What stays the same for one hand: which side the pivot is on, how the speed bell leans, which way is
    heavier, how it aims. `seed` picks the person; the same seed is the same artist on every page."""

    def __init__(self, seed, left_handed=None):
        r = np.random.default_rng([seed, 258])
        self.seed = seed
        self.left_handed = (r.random() < 0.15) if left_handed is None else left_handed
        psi = math.radians(50 + 25 * r.random())                # pivot below-right of the tip (y down)
        self.psi = math.pi - psi if self.left_handed else psi
        self.sigma = 0.22 + 0.20 * r.random()                   # skew of the speed bell
        self.kD = 0.03 + 0.04 * r.random()                      # spread of throw length (per unit of fluency)
        self.ktheta = math.radians(1.5 + 2.5 * r.random())      # spread of direction
        self.bias = math.radians(12 * r.random() - 6)           # every stroke turned a little: this hand's own
        self.reach = 0.85 + 0.27 * r.random()                   # falls short or overshoots: this hand's own
        self.body = 0.5 + 0.5 * r.random()                      # how much a stroke is a lever's arc
        self.lever = 0.8 + 0.5 * r.random()
        self.approach = 0.22 + 0.12 * r.random()                # s: one approach of a slow stroke
        self.toward = self.psi + math.radians(30 * r.random() - 15)
        self.heavy = 0.10 + 0.40 * r.random()                   # how much heavier a pull toward yourself is
        self.light = 0.10 + 0.25 * r.random()                   # how much lighter at speed
        self.press = 0.72 + 0.16 * r.random()
        self.tempo = 0.45 + 0.45 * r.random()                   # own fluency when no tempo is asked for

    def describe(self):
        return (f'person {self.seed}, {"left" if self.left_handed else "right"}-handed, pivot '
                f'{math.degrees(self.psi):.0f} deg, bell sigma {self.sigma:.2f}, bias '
                f'{math.degrees(self.bias):+.1f} deg, reach {self.reach:.2f}, pull heavier by {self.heavy:.2f}')


class Hand:
    def __init__(self, person, fluency=None, seed=0):
        self.p = person
        self.b = float(person.tempo if fluency is None else fluency)
        self.r = np.random.default_rng([person.seed, seed, 7])
        self.home = None
        self.R = None
        self.at = None           # where the hand was left after the last gesture (mm)
        self.form = 1.0          # how much tempo changes the shape: corner cutting, landing on the move (0 = none)
        self.miss = 0.0

    # ------------------------------------------------------------------ duration of a throw
    def _duration(self, D):
        b = self.b
        return (0.15 + 0.80 * (1 - b) ** 1.6) * (max(D, 2.0) / 30.0) ** 0.35

    # ------------------------------------------------------------------ body
    def _pivot(self, pts):
        """The pivot stays while the stroke is in reach; otherwise the hand is set down again under it."""
        P = np.asarray(pts, float)
        span = float(np.max(np.hypot(*(P - P[0]).T))) if len(P) > 1 else 0.0
        R = float(np.clip(3.5 * span, 60.0, 400.0)) * self.p.lever
        if self.home is not None and abs(R - self.R) < 0.3 * self.R:
            if np.max(np.hypot(*(P - self.home).T)) < 0.7 * self.R:
                return False
        self.home = P[0] + self.r.standard_normal(2) * 1.5
        self.R = R
        self.C = self.home + R * _unit(self.p.psi + math.radians(8) * self.r.standard_normal())
        return True

    def _easy_end(self, A, B):
        """Where the tip lands if a stroke of the same length is one lever swing around the pivot; the arc angle."""
        u = self.C - A
        R = max(np.hypot(*u), 1e-6)
        psi = _angle(u)
        d = B - A
        D = np.hypot(*d)
        a = _angle(d) - psi
        delta = -(D / R) * math.sin(a)
        rr = _rot(A - self.C, delta) * (1 - D * math.cos(a) / R)
        return self.C + rr, delta

    # ------------------------------------------------------------------ plan of throws
    def gesture(self, strokes):
        """strokes: [{'points': [(x, y), ...], 'bend': [deg per leg], 'accent': [0-1 per target]}, ...] in mm.
        Between strokes the hand travels through the air on its own. Returns the pieces of contact, a list of
        arrays (x, y, pressure, dt_ms), and the air path of the tip (t, x, y, pressure)."""
        p, b, r = self.p, self.b, self.r
        throws = []           # (t0, mu, sigma, D, theta, delta, on, T)
        touches = []          # (t_down, t_up, accents [(t, strength)], T_first, T_last)
        first_pt = np.asarray(strokes[0]['points'][0], float)
        self._pivot(strokes[0]['points'])
        start = (first_pt + 14.0 * _unit(p.psi + 0.4 * r.standard_normal()) if self.at is None
                 else np.array(self.at, float))
        plan = start.copy()         # where the hand thinks it is
        fact = start.copy()         # where it really is (sum of throws)
        t_on = 0.0
        overlap = 0.42 * b * self.form

        def throw(target, bend, on, eye_weight, in_air=False):
            nonlocal plan, fact
            believe = eye_weight * fact + (1 - eye_weight) * plan
            easy, d_body = self._easy_end(believe, target)
            aim = target + (0.05 + 0.30 * b) * p.body * (easy - target) if not in_air else target
            d = aim - believe
            D = np.hypot(*d)
            if D < 1e-6:
                plan = target.copy()
                return None
            T = self._duration(D)
            noise = 0.35 + b
            D2 = D * (p.reach ** b) * (1 + p.kD * noise * r.standard_normal())
            theta = _angle(d) + p.bias * b + p.ktheta * (0.9 + 0.5 * b) * r.standard_normal()
            delta = math.radians(bend) + 0.6 * d_body * p.body * (0.5 + 0.5 * b)
            half = delta / 2
            Darc = D2 * (half / math.sin(half) if abs(half) > 1e-4 else 1.0)
            emu = T / (math.exp(_Q * p.sigma) - math.exp(-_Q * p.sigma))
            mu = math.log(emu)
            t0 = on - math.exp(mu - _Q * p.sigma)
            throws.append((t0, mu, p.sigma, Darc, theta, delta, on, T))
            fact = fact + D2 * _unit(theta)
            plan = target.copy()
            return len(throws) - 1

        def t_q(k, q):
            t0, mu, sigma = throws[k][:3]
            return t0 + math.exp(mu + sigma * _Z(q))

        P = None
        for i, st in enumerate(strokes):
            P = [np.asarray(q, float) for q in st['points']]
            bends = list(st.get('bend') or [0.0] * (len(P) - 1))
            accents = list(st.get('accent') or [0.0] * len(P))
            # a fluent hand aims past a corner (a virtual target): the throws' overlap cuts the corner and the
            # trace comes out close to it
            V = [q.copy() for q in P]
            for j in range(1, len(P) - 1):
                a, c = P[j] - P[j - 1], P[j + 1] - P[j]
                la, lc = np.hypot(*a), np.hypot(*c)
                V[j] = P[j] + 0.75 * overlap * 0.22 * 0.5 * (la + lc) * (_rot(a / la, math.radians(bends[j - 1]) / 2)
                                                                        - _rot(c / lc, -math.radians(bends[j]) / 2))
            # --- through the air to the start of the stroke
            if i > 0 and self._pivot(st['points']):
                t_on += 0.10 + 0.25 * (1 - b) + 0.05 * r.random()
            k = throw(P[0], 0.0, t_on, 0.9 - 0.3 * b, in_air=True)
            if k is not None:
                T, Da = throws[k][7], throws[k][3]              # from afar the hand arrives first, then lands
                t_on += T * (1 - overlap * float(np.clip(12.0 / Da, 0.1, 1.0)))
                if np.hypot(*(P[0] - fact)) > 0.6:             # the eye brings the tip to its place: a short throw
                    k2 = throw(P[0], 0.0, t_on - 0.25 * T, 1.0 - 0.25 * b, in_air=True)
                    if k2 is not None:
                        t_on = max(t_on, t_on - 0.25 * T + throws[k2][7] * (1 - 0.6 * overlap))
                t_on += (1 - b) ** 2 * 0.12 * (0.5 + r.random())
            first = last = None
            acc = []
            # --- the stroke itself
            for j in range(1, len(P)):
                D = np.hypot(*(P[j] - P[j - 1]))
                T = self._duration(D)
                n = max(1, int(round(T / (p.approach * (1 + 1.2 * b)))))
                bend = bends[j - 1]
                for m in range(n):
                    target = V[j] if m == n - 1 else _on_arc(P[j - 1], P[j], bend, (m + 1) / n)
                    k = throw(target, bend / n, t_on, 0.9 - 0.4 * b)
                    if k is None:
                        continue
                    if first is None:
                        first = k
                    last = k
                    Tk = throws[k][7]
                    if m == n - 1 and j < len(P) - 1:          # a corner target: fluent cuts it, careful stops
                        t_on += Tk * (1 - overlap * (1 - 0.7 * accents[j])) + (1 - b) ** 2 * 0.10 * (0.5 + r.random())
                    else:
                        t_on += Tk * (1 - (0.42 if m < n - 1 else overlap))
                if accents[j] > 0 and last is not None:
                    acc.append((t_q(last, 0.9), accents[j]))
            if first is None:
                continue
            if accents[0] > 0:
                acc.append((t_q(first, 0.05), accents[0]))
            stand = (1 - b) ** 2 * 0.10
            t_down = t_q(first, 0.004 + 0.07 * b ** 1.5 * self.form) - stand * (0.6 + 0.8 * r.random())
            t_up = t_q(last, 0.996 - 0.09 * b ** 1.5 * self.form) + stand * (0.6 + 0.8 * r.random())
            touches.append((t_down, t_up, acc, throws[first][7], throws[last][7]))
            t_on = max(t_on, t_up - 0.5 * b * 0.1) + (1 - b) * 0.08

        self.miss = float(np.hypot(*(fact - P[-1])))          # how far the hand landed from the last target
        if not touches:
            return [], np.zeros((0, 4))
        # ------------------------------------------------------------------ movement: the sum of throws
        t_end = max(th[0] + math.exp(th[1] + 3.2 * th[2]) for th in throws) + 0.15
        t = np.arange(min(-0.2, touches[0][0] - 0.1), t_end, DT)
        v = np.zeros((len(t), 2))
        for t0, mu, sigma, D, theta, delta, on, T in throws:
            tau = np.maximum(t - t0, 1e-9)
            z = (np.log(tau) - mu) / sigma
            lam = np.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi) * tau)
            lam[t <= t0] = 0.0
            F = 0.5 * (1 + _erf(z / math.sqrt(2)))
            F[t <= t0] = 0.0
            phi = theta + delta * (F - 0.5)
            v[:, 0] += D * lam * np.cos(phi)
            v[:, 1] += D * lam * np.sin(phi)
        xy = start + np.cumsum(v, 0) * DT
        self.at = xy[-1].copy()
        speed = np.hypot(*v.T)

        # ------------------------------------------------------------------ pen in hand: pressure
        vref = 120.0
        touching = np.zeros(len(t))
        accent = np.ones(len(t))
        for t_down, t_up, acc, Tf, Tl in touches:
            tau_in = min(0.03 + 0.05 * (1 - b) * r.random(), 0.18 * Tf)    # a short throw lands quicker
            tau_out = min(0.04 + 0.05 * r.random(), 0.25 * Tl)
            a = np.clip((t - t_down) / tau_in, 0, 1)
            c = np.clip((t_up - t) / tau_out, 0, 1)
            touching = np.maximum(touching, (a * a * (3 - 2 * a)) * (c * c * (3 - 2 * c)))
            if b < 0.5:                                        # a careful hand, set down, presses in
                accent += 0.25 * (1 - 2 * b) * np.exp(-0.5 * ((t - t_down - tau_in) / 0.05) ** 2)
            for ta, s in acc:
                accent += 0.5 * s * np.exp(-0.5 * ((t - ta) / 0.06) ** 2)
        heading = np.arctan2(v[:, 1], v[:, 0])
        going = speed / (speed + 15.0)
        press = p.press * (1 + p.heavy * np.cos(heading - p.toward) * going)
        press *= 1 - p.light * np.clip(speed / (2.5 * vref), 0, 1) * 0.8
        press *= accent
        kern = np.exp(-0.5 * (np.arange(-30, 31) * DT / 0.012) ** 2)
        press = np.convolve(press, kern / kern.sum(), 'same')
        press = np.clip(press * touching, 0, 1)
        allc = np.column_stack([xy, press, np.full(len(t), DT * 1000)])

        pieces = []
        on = press > 0.02
        i = 0
        while i < len(t):
            if not on[i]:
                i += 1
                continue
            j = i
            while j < len(t) and on[j]:
                j += 1
            a, c = max(i - 1, 0), min(j + 1, len(t))
            if c - a >= 4:
                pieces.append(allc[a:c].copy())
            i = j
        return pieces, np.column_stack([t, xy, press])


def _on_arc(A, B, bend, frac):
    """A point on the arc from A to B whose tangent turns by `bend` degrees (0 = straight)."""
    delta = math.radians(bend)
    if abs(delta) < 1e-4:
        return A + (B - A) * frac
    chord = B - A
    L = np.hypot(*chord)
    R = L / (2 * math.sin(delta / 2))
    th0 = _angle(chord) - delta / 2              # tangent at the start
    C = A + R * _unit(th0 + math.pi / 2)
    return C + _rot(A - C, delta * frac)


def arcs(A, B, bend, f):
    """Points on the arc from A to B (tangent turns by `bend` degrees) at fractions f, all at once."""
    delta = math.radians(bend)
    if abs(delta) < 1e-4:
        return A + np.outer(f, B - A)
    chord = B - A
    R = np.hypot(*chord) / (2 * math.sin(delta / 2))
    th0 = math.atan2(chord[1], chord[0]) - delta / 2
    C = A + R * np.array([math.cos(th0 + math.pi / 2), math.sin(th0 + math.pi / 2)])
    v = A - C
    c, s = np.cos(delta * f), np.sin(delta * f)
    return C + np.column_stack([c * v[0] - s * v[1], s * v[0] + c * v[1]])
