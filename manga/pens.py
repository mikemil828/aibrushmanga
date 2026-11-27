"""Inking tools. Widths are in mm, so a pen draws the same line at any px_per_mm.

Pressure (0-1) maps to width as  w = min_mm + (max_mm - min_mm) * pressure ** gamma.
Every stroke gets an entry and exit taper (iri-nuki) unless the pen says otherwise.
"""
from dataclasses import dataclass, asdict, replace


@dataclass(frozen=True)
class Pen:
    name: str
    min_mm: float          # width at zero pressure
    max_mm: float          # width at full pressure
    gamma: float = 1.0     # pressure curve: >1 = needs a firm hand to open the nib
    taper_in_mm: float = 1.0
    taper_out_mm: float = 2.0
    wobble_mm: float = 0.02    # slow hand tremor, sideways
    rough: float = 0.0         # fast width noise along the edge (brush hair, paper tooth)
    dry: float = 0.0           # 0-1: how soon ink runs out toward the end (brush streaks)
    ink: float = 1.0           # opacity of the ink film

    def width_mm(self, pressure):
        return self.min_mm + (self.max_mm - self.min_mm) * (pressure ** self.gamma)

    def with_(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return Pen(**d)


PENS = {
    # G-pen: the main manga nib. Thin at rest, flares strongly under pressure.
    'g_pen': Pen('g_pen', 0.06, 0.85, gamma=1.7, taper_in_mm=1.2, taper_out_mm=3.0, wobble_mm=0.02),
    # Saji (spoon) pen: steadier, moderate swell. Good for clothes and backgrounds.
    'saji': Pen('saji', 0.08, 0.45, gamma=1.3, taper_in_mm=0.8, taper_out_mm=2.0, wobble_mm=0.015),
    # Maru (round) pen: hairline nib for faces, hatching and fine detail.
    'maru': Pen('maru', 0.03, 0.22, gamma=1.2, taper_in_mm=0.6, taper_out_mm=1.6, wobble_mm=0.01),
    # Brush pen: big swell, rough edges, runs dry at the tail.
    'brush': Pen('brush', 0.15, 4.0, gamma=1.4, taper_in_mm=2.0, taper_out_mm=6.0, wobble_mm=0.05,
                 rough=0.06, dry=0.35),
    # Technical fineliner: constant width, square-ish ends. Panel borders, rulers, buildings.
    'fineliner': Pen('fineliner', 0.30, 0.30, taper_in_mm=0.0, taper_out_mm=0.0, wobble_mm=0.0),
    'border': Pen('border', 0.8, 0.8, taper_in_mm=0.0, taper_out_mm=0.0, wobble_mm=0.0),
    # Marker for filling and heavy flat shapes.
    'marker': Pen('marker', 1.2, 1.2, taper_in_mm=0.0, taper_out_mm=0.0, wobble_mm=0.03),
    # Pencil for the sketch layer (use with mode='sketch', or Sheet.pencil). Soft grey, grainy, never printed.
    'pencil': Pen('pencil', 0.12, 0.32, gamma=1.0, taper_in_mm=0.6, taper_out_mm=1.2, wobble_mm=0.04,
                  rough=0.25, ink=0.6),
    # White-out brush (use with mode='white').
    'white': Pen('white', 0.2, 1.6, gamma=1.2, taper_in_mm=0.8, taper_out_mm=1.5, wobble_mm=0.02),
}


def get(pen):
    if isinstance(pen, Pen):
        return pen
    if isinstance(pen, dict):
        return Pen.from_dict(pen)
    try:
        return PENS[pen]
    except KeyError:
        raise ValueError(f'unknown pen {pen!r}; known: {", ".join(PENS)}') from None
