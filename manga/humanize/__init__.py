"""Humanize: a hand draws the strokes. Ported from the oil engine's hand module (движок/рука, upstream
neokrasav4ik/aibrush) to the manga ink engine.

A stroke from the engine is exact: it goes through its points as a perfect curve. A person's pen doesn't.
Humanize gives every ink line to a simulated hand (`motion.Hand`): a stroke is a chain of ballistic throws
toward a few targets on the line, swung around a wrist pivot. So a line comes out with a slight arc, falls
short or overshoots a little, cuts a corner when the hand is fluent, and rocks its pressure (a pull toward
yourself is heavier). The hand has a live state (`state.LiveHand`): it warms up, tires, gains confidence after
clean lines and gets careful after a miss, and its eye lays a line against the lines already there, so
hatching and outlines drawn in pieces drift together instead of apart.

It is off by default, and with it off nothing changes: the page is byte for byte what it was.

    sh.humanize()                         # live: this sheet's strokes from here on (strength 0.15)
    sh.humanize(strength=0.1, parts={'eyes': 0.1})
    with sh.part('eyes'): ...             # name a part, so the hand is gentle where it matters

    python -m manga.humanize page.jsonl out.jsonl [--strength 0.15] [--person N] [--seed N]
                             [--part eyes=0.1 ...] [--pen maru=0.5 ...]      # a journal after the fact
    python -m manga.replay out.jsonl --out page.png

Settings: [humanize] in manga/settings.default.toml (copy lines into settings.local.toml to change them).
Strength: 0 = off; 0.15 = default, a quiet hand; by 0.3 the whole shape of a line is the hand's. Each part of a
drawing takes a share of it ([humanize.parts]: eyes 0.25, face 0.5, hair 0.8, body and background 1, effects
0.3), and so does each pen ([humanize.pens]: fineliner and border 0, they are ruled). How far a line may move is
capped at max_drift_mm x strength (0.3 mm at 0.15).
"""
from .core import FROM, OP, Humanizer, config, strip                     # noqa: F401
from .motion import Hand, Person                                          # noqa: F401
from .state import LiveHand                                               # noqa: F401
from .patch import rewrite, rewrite_file, strip_journal                   # noqa: F401
