"""A hand with a live state. Ported from the oil engine's hand module (движок/рука/живая.py, upstream
neokrasav4ik/aibrush).

There is no per-stroke seed. There is a person (what stays the same) and a state that runs from stroke to stroke:

  time       warm-up (the first strokes are stiff), fatigue (builds with path length and pressure, eases in a
             pause), attention (drifts slowly on its own, as a person's does);
  the task   how tight the target is and how long the stroke (a precise place is drawn slowly), how important
             the place is, whether the gesture is familiar (a repeat is bolder);
  luck       confidence grows with hits and falls with misses; after a miss the hand is more careful;
  pose       far from its comfortable place the hand is less precise.

The hand picks its own tempo."""
import copy
import math

import numpy as np

from .motion import Hand


class State:
    def __init__(self, person, seed):
        self.r = np.random.default_rng([person.seed, seed, 259])
        self.warm = 0.15
        self.fatigue = 0.0
        self.attention = 0.0           # -1..1, drifts on its own: three waves, over ten, a hundred, a thousand strokes
        self.waves = [0.0, 0.0, 0.0]
        self.boredom = 0.0             # grows on runs of the same stroke, broken by a change of gesture or a miss
        self.nerve = 0.0               # grows on a run of hits: the hand takes risks until it misses
        self.day = 0.0                 # mood of the sitting
        self.session = 1
        self.since_session = 0
        self.confidence = 0.55
        self.familiar = 0              # how many times in a row the same gesture
        self.last_role = None
        self.bias = 0.0                # slow drift of the hand's habits
        self.reach = 0.0
        self.after_miss = 0.0
        self.strokes = 0
        self.usual_miss = None

    def tick(self, path_mm, pressure):
        r = self.r
        self.warm += (1 - self.warm) * 0.05
        self.fatigue = min(1.0, self.fatigue + path_mm * (0.4 + pressure) * 0.00004)
        for i, (keep, kick) in enumerate([(0.88, 0.22), (0.988, 0.085), (0.9988, 0.030)]):
            self.waves[i] = keep * self.waves[i] + kick * r.standard_normal()
        self.attention = float(np.clip(sum(self.waves) + self.day - self.boredom, -1.5, 1.5))
        self.since_session += 1
        self.bias = 0.96 * self.bias + math.radians(0.8) * r.standard_normal()
        self.reach = 0.96 * self.reach + 0.012 * r.standard_normal()
        self.after_miss *= 0.6
        self.strokes += 1

    def pause(self, amount=1.0):
        """Stepped back from the page: fatigue eases, some warm-up is lost, attention is shaken up."""
        self.fatigue *= 1 - 0.55 * amount
        self.warm *= 1 - 0.35 * amount
        self.waves[0] = 0.6 * self.r.standard_normal()
        self.boredom *= 1 - 0.8 * amount
        self.familiar = 0

    def new_session(self):
        """Away for a long while and back: fresh, cold, in another mood."""
        self.fatigue *= 0.1
        self.warm = 0.2
        self.day = 0.22 * float(self.r.standard_normal())
        self.waves[0] = self.waves[1] = 0.0
        self.boredom = self.nerve = 0.0
        self.familiar = 0
        self.session += 1
        self.since_session = 0


class Drift:
    """The eye's memory of where strokes have already gone on the page. A person lays a line against what is
    already there, not against a ruler: if the earlier lines of a place went half a millimetre left of the plan,
    the next one aims there too. So a stroke's overall drift (the mean of its offset) takes the share `share`
    from the drift of strokes already lying under it and only the rest is its own; its shape (arc, cut corner,
    falling short) stays its own. Without this, neighbouring lines (hatching, an outline drawn in pieces) wander
    apart. Grid cells of `cell` mm; the first stroke of a place sets its drift."""

    def __init__(self, cell=5.0, share=0.7, memory=0.3):
        self.cell, self.share, self.memory, self.field = float(cell), float(share), float(memory), {}

    def __call__(self, pts, own):
        """pts: the stroke's path, mm (n, 2); own: its mean offset, mm. Returns the mean offset the eye corrects to."""
        cells = set(map(tuple, np.floor(np.asarray(pts, float)[:, :2] / self.cell).astype(int).tolist()))
        before = [self.field[c] for c in cells if c in self.field]
        own = np.asarray(own, float)
        if before:
            own = self.share * np.mean(before, axis=0) + (1 - self.share) * own
        for c in cells:
            was = self.field.get(c)
            self.field[c] = own if was is None else (1 - self.memory) * was + self.memory * own
        return own


class LiveHand:
    def __init__(self, person, seed=0, strength=1.0):
        self.p0 = person
        self.strength = float(strength)    # how visible the hand is: 0 = the path almost as planned, 1 = full
        self.s = State(person, seed)
        self.hand = Hand(person, person.tempo, seed=seed)
        self.hand.r = self.s.r             # one stream of chance for the hand's whole life
        self.drift = Drift()

    # ------------------------------------------------------------------ choosing a tempo
    def tempo(self, st):
        s, p = self.s, self.p0
        P = np.asarray(st['points'], float)
        D = float(np.sum(np.hypot(*np.diff(P, axis=0).T))) if len(P) > 1 else 1.0
        tol = float(st.get('tolerance', 3.0))
        hard = math.log2(D / tol + 1.0)                     # the longer the line and the tighter the aim, the slower
        role = st.get('role')                               # the same thing with the same pen: the same gesture
        s.familiar = s.familiar + 1 if role == s.last_role else 0
        if role == s.last_role:
            s.boredom = min(1.0, s.boredom + 0.006)         # the same again and again: the hand goes on autopilot
        else:
            s.boredom *= 0.55
        s.last_role = role
        interest = float(st.get('interest', 0.5))           # an important place: the hand is collected
        b = (p.tempo + 0.22 * (s.warm - 0.6) + 0.35 * (s.confidence - 0.6) - 0.25 * s.fatigue
             + 0.10 * s.attention - 0.13 * (hard - 2.3) + 0.035 * min(s.familiar, 6)
             - 0.30 * s.after_miss + 0.12 * s.boredom + 0.15 * s.nerve - 0.22 * (interest - 0.5))
        return float(np.clip(b, 0.03, 0.97))

    # ------------------------------------------------------------------ one stroke
    def stroke(self, st):
        """Returns the pieces of contact (mm) and a record of the stroke."""
        s, p0 = self.s, self.p0
        b = self.tempo(st)
        p = copy.copy(p0)
        pose = 0.0
        if self.hand.home is not None:
            pose = float(np.clip(np.hypot(*(np.asarray(st['points'][0], float) - self.hand.home))
                                 / max(self.hand.R, 1.0), 0, 1.5))
        interest = float(st.get('interest', 0.5))
        noise = ((1.35 - 0.5 * s.warm) * (1 + 0.8 * s.fatigue) * (1 - 0.15 * s.attention) * (1 + 0.35 * pose)
                 * (1 + 0.6 * s.boredom) * (1 + 0.5 * s.nerve) * (1.3 - 0.6 * interest))
        k = self.strength
        p.kD, p.ktheta = p0.kD * noise * k, p0.ktheta * noise * k
        p.bias = (p0.bias + s.bias) * k
        p.reach = 1.0 + (p0.reach + s.reach - 1.0) * k
        p.body = p0.body * k
        p.heavy, p.light = p0.heavy * (0.4 + 0.6 * k), p0.light * (0.4 + 0.6 * k)
        p.press = min(0.97, p0.press * (1 + 0.12 * s.fatigue) * float(st.get('pressure', 1.0)))
        self.hand.p, self.hand.b = p, b
        self.hand.form = math.sqrt(max(k, 0.0))         # at low strength tempo barely changes the shape
        pieces, _ = self.hand.gesture([st])
        if pieces:
            L = float(sum(np.sum(np.hypot(*np.diff(q[:, :2], axis=0).T)) for q in pieces))
            pr = float(np.mean(np.concatenate([q[:, 2] for q in pieces])))
        else:
            L, pr = 0.0, 0.0
        s.fatigue = min(1.0, s.fatigue + 0.00012 * interest)     # concentration tires even without a path
        rec = dict(b=b, miss=self.hand.miss, tolerance=float(st.get('tolerance', 3.0)), interest=interest, path=L)
        s.tick(L, pr)
        return pieces, rec

    # ------------------------------------------------------------------ what the hand learnt from the page
    def saw(self, rec):
        """After a stroke the eye sees whether it landed. Confidence and the next stroke follow from it."""
        s = self.s
        usual = rec['miss'] if s.usual_miss is None else s.usual_miss
        bad = rec['miss'] > max(2.0 * usual, 0.05) or rec['miss'] > rec['tolerance']
        luck = 1.0 - min(rec['miss'] / max(2.5 * usual, 0.05), 1.0)
        s.usual_miss = 0.97 * usual + 0.03 * rec['miss']
        s.confidence += 0.22 * (luck - s.confidence)
        if bad:
            s.after_miss = 1.0
            s.nerve = 0.0                     # a miss sobers and wakes up
            s.boredom *= 0.3
        elif s.confidence > 0.55:
            s.nerve = min(1.0, s.nerve + 0.008)

    def time_to_rest(self):
        """The hand ends the sitting itself: tired (threshold set by the day) after at least 300 strokes."""
        s = self.s
        return s.since_session > 300 and s.fatigue > 0.5 + 0.5 * s.day + 0.1 * s.attention
