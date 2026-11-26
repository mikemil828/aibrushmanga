# aibrush manga: an ink engine for manga pages

This fork adds a manga mode next to the oil engine in `движок/`. It draws in black ink on paper: nib pens with pressure and taper, hatching, screentone, white-out, focus lines, speed lines, flash and ruled panel borders. Every call goes into a JSONL journal, and the journal can be replayed at any scale. A page drafted at 8 px/mm can be re-rendered for print at 16 or 24 px/mm with the same strokes.

It is a separate package because ink needs none of the oil physics (wet mixing, paint relief, lighting). It keeps the parts of aibrush that carry over: pressure paths, a stroke journal, replay at scale, and stencil-style masks. It also follows the same advice: decide first, then draw.

Requires Python 3.10+, NumPy, SciPy and Pillow.

> **Provide a reference image for the best results.** The engine is at its best when it works from a
> drawing or illustration you supply (your own art, or art you have the rights to): the figure's shapes,
> materials and lines come from the reference, and the inker's decisions (pens, tones, blacks, cleanup)
> go on top, as in `examples/pool_page/`. Figures drawn without a reference, from coordinates alone, come
> out stiff and simplified, so they are not included as examples.

## Quick start

```python
from manga import Sheet, hatch, tone, effects

sh = Sheet(2056, 2912, px_per_mm=8, journal='page.jsonl')        # B4 manga page
rows = effects.layout(2056, 2912, [(1, [1]), (1, [1.5, 1])])        # panel polygons
p = rows[0][0]
with sh.clip(p):                                                    # everything inside is cut to the panel
    sh.stroke([(200, 300), (600, 260), (900, 320)], pen='g_pen', pressure=0.8)
    tone(sh, p, level=(0.3, 0.0, (0, 100), (0, 600)), lpi=32)       # gradient dot tone
    effects.focus_lines(sh, (1000, 500), (200, 120), (100, 100, 1950, 900))
effects.border(sh, p)
sh.png('page.png')
sh.close()
```

Replay larger for print: `python -m manga.replay page.jsonl --scale 3 --out page_print.png` (add `--bw` for pure 1-bit).

## Tools

| Module | What it does |
|-|-|
| `Sheet` | One ink-coverage buffer on paper. `stroke`, `fill` (solid black), `apply` (a precomputed layer), `clip` (panels, nestable), `png`. `mode='white'` turns any of these into white-out. |
| `pens` | `g_pen` (thin at rest, flares under pressure), `saji`, `maru` (hairline), `brush` (rough edge, runs dry at the tail), `fineliner`, `marker`, `white`. Widths are in mm, and the entry and exit taper is automatic. Make your own with `Pen(...)` or `PENS['g_pen'].with_(max_mm=1.2)`. |
| `hatch` | Hatching and cross-hatching driven by a darkness map. Layers add directions only where the tone gets darker. Lines are hand-length pen strokes with small gaps and overlaps. `field=` bends them along a direction map. |
| `tone` | Screentone: `kind='dot'`, `'line'` or `'noise'` (sand). `level` is a number, an array, or a gradient tuple. The grid is fixed to the sheet, so pieces cut from the same tone line up. `lpi` is physical: 50–65 for print, 27–42 for screen-size drafts. `mode='white'` scratches tone away. |
| `effects` | `focus_lines`, `speed_lines`, `flash`, `layout` (panel grid with manga gutters and slanted edges), `border`, `caption`. |
| `prep` | Measuring a reference photo: `darkness` (lightness, or `mode='value'` so red and blue read as mid-tones), `orientation`, `ink_outline` (inks a shape's contour, heavier on the side away from the light), `xdog_lines` (contour guides for figures). |
| `prep.ink_contour` | Inks chosen stretches of a shape's edge as hand-length strokes: a ridge where sky meets mountain, a cloud's lit underside, the top of a grove. `keep=` picks the stretches, `style=` varies pen and weight per stretch. |
| `Frame` | A reference image placed on the page. `P`/`pts` map reference px to page px, `up`/`upm` scale arrays and masks, `lay` lays tone from reference-sized masks and levels, `poly`/`disc`/`line` draw hand-placed shapes given in reference px. |
| `nature` | Drawn vegetation: `blade`, `edge_grass` (blades along a bank's edges, leaning outward), `tufts`, `seedlings` (plants in water, with reflections). `depth(y)` sets size and density by distance. |
| `water` | Still water: `calm_level` (line tone without streaks from things standing in it), `fill_under` (no halos around them), `mirror` (reflections), `ripples`. Use `angle=water.HORIZONTAL` for line tone. |
| `tools` | Working aids: `gridcrop` (a reference crop with a labelled grid, for reading coordinates), `fragment` (draw one part on a clipped sheet in seconds), `compare` (drawn area beside the reference). |
| `replay` | `python -m manga.replay journal.jsonl --scale S --out file.png [--bw] [--until N]` |

## Workflow that has worked

Learned on the pool page and the rice-field panel.

1. **Decide first, in writing.** Put the plan at the top of the page script: what is primary, which tone and
   pen each part gets, and what is drawn by hand. The rice-field plan is a good template.
2. **Region map from the reference** (`prep.py` in each example): colour tests first, then hand polygons for
   what colour can't separate (a sunlit slope as bright as the sky, a figure, a shadow). Look at an overlay.
3. **Read coordinates with `tools.gridcrop`** at 4-6x, then place shapes with `Frame.poly` and `Frame.line`.
4. **Try each part with `tools.fragment`** before running the whole sheet, and compare with `tools.compare`.
5. **Draw, don't trace, anything made of many small things** (grass, seedlings, thatch, needles): one tapered
   stroke each, from the helpers in `nature`. Traced, they become blobs. Watch the stroke count: density is
   the usual mistake.
6. **Keep black for the few things that should hold the eye.** Too much solid black makes a background part
   (houses, a grove, mud) jump forward; a dark tone sits back.
7. **Review area by area at print size**, write the weak spots down, and fix the worst first. Work in the
   example's folder in the repo from the start, so there is nothing to copy over later.

## Examples

- `examples/sampler.py`: every tool on one sheet.
- `examples/pool_page/`: a four-panel manga page drawn from a reference illustration (see its README). This is
  the recommended way to work: start from a reference image.
- `examples/rice_field/`: a single landscape panel (paddies, village, mountains, a walker) drawn from a reference
  image. It shows line tone for water, dot tone for land, and hand-drawn seedlings and figure (see its README).

## Tests

`python3 -m pytest manga/tests -q` (about 5 s). They check that strokes land where they should at the right
width in mm, that flat tones cover the right share of paper, that clips and white-out work, and that a journal
replays to the same sheet at x1 and x2. GitHub runs them on every push that touches `manga/`.

## What it can't do yet

- Real photos are not supported yet (planned for later).
- Figures traced from a reference are the weak point. Contours come from colour classes plus an XDoG line map, so faces and hands read as shapes, not as drawn anatomy. A real inker simplifies far more.
- Hatching follows straight rows, or a direction field without spacing control, so curved hatching can bunch up.
- Without a reference image, figures have to be built from hand-placed coordinates and come out stiff and
  simplified; detailed characters, hands and dynamic poses need a reference.
- Lettering is caption boxes and the bundled Comic Neue face; speech balloons are drawn per page, and there is no
  vertical text yet.
- Tone is laid flat. There is no moiré check when two dot tones overlap at different angles.
- The noise (sand) tone is only roughly scale-stable, because its grain size is rounded to whole pixels.
