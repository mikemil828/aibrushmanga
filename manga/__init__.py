"""aibrush manga: an ink engine for manga pages (pens, hatching, screentone, effects, panels).

    from manga import Sheet, hatch, tone, effects
    sh = Sheet(2056, 2912, px_per_mm=8, journal='page.jsonl')
    sh.stroke([(100, 100), (400, 140), (700, 100)], pen='g_pen', pressure=0.8)
    sh.png('page.png')

Style defaults (pen widths, hatching, tone, soft shadows, borders) come from manga/settings.default.toml with
your own manga/settings.local.toml laid over it; see manga/settings.py.
"""
from .sheet import Sheet
from .pens import Pen, PENS
from .hatch import hatch
from .tone import tone
from . import effects, prep, geom, trace, nature, water, tools, face
from .frame import Frame
from . import settings

settings.load()

__all__ = ['Sheet', 'Pen', 'PENS', 'hatch', 'tone', 'effects', 'prep', 'geom', 'trace', 'nature', 'water', 'tools',
           'face', 'Frame', 'settings']
