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

For a home or office laser printer, render for its resolution and print the PNG at 100 % ("actual size"):
`python -m manga.replay page.jsonl --dpi 600 --fit letter --bw --out page_600dpi.png`. `--dpi` snaps every
screen to a whole-pixel cell with a fixed threshold order, so all dots come out the same; an unsnapped 1-bit
screen has dots of slowly drifting shape, which a laser printer shows as a woven pattern. `--fit` (letter, a4,
b4, legal) shrinks the page to the paper; leave it out to print at the page's own size.

## Your own style settings

Line thickness, hatching, screentone, hard or soft shadow edges, panel borders and paper colour all have defaults
in `manga/settings.default.toml`. Don't edit that file, since `git pull` replaces it. Instead create
`manga/settings.local.toml` (git ignores it, so a pull never touches it) and copy in only the lines you want to
change:

```toml
[lines]
width_scale = 1.3      # every pen 30 % heavier

[pens.g_pen]
max_mm = 1.1           # or change one pen

[hatch]
spacing_mm = 0.5       # denser hatching
layers = [[0.2, 0.0], [0.5, 90.0]]   # two directions at most

[tone]
soft_mm = 2.0          # soft shadows: tone fades out over 2 mm at its edge (0 = hard, knife-cut edges)
```

`python -m manga` lists the settings in effect and the files they came from. To keep several styles, put each in
its own file and pick one with `MANGA_SETTINGS=styles/noir.toml`, with `--settings styles/noir.toml` on any
script that imports manga, or with `manga.settings.use('styles/noir.toml')` in code. Files are laid on in that
order: defaults, local file, the environment variable, the flag, then code. Settings are only defaults, so a value
passed in a call (`hatch(..., spacing_mm=0.7)`) always wins. Journals store the values a page was drawn with, so
`manga.replay` redraws an old page exactly, whatever your settings are now. A misspelt name prints a warning.
`[tone] soft_mm` applies to every tone; for soft shadows only, pass `soft_mm=` in the shadow's own `tone()` call.

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
| `tools` | Working aids: `gridcrop` (a reference crop with a labelled grid, for reading coordinates), `fragment` (draw one part on a clipped sheet in seconds), `compare` (drawn area beside the reference), `pin` (the page round a review comment pinned at x%, y%, beside the same area of the reference), `cut_parts` (what a panel edge or cut takes off a figure: a sawn-off finger shows up as a small loss from a big piece). `layers` (reference, drawing and masks side by side), `before_after` (a review round on one sheet), `strip` (one part of every figure in a grid), `probe` (colour and classes along a line), `keep` (save a script's masks when `DUMP` is set). |
| `face` | The parts an expression is made of: `brow` (a tapered solid shape; `ribbon_of` measures one from a mask), `iris` (cel layers, rim, streaks, pupil, catch-lights), `eye_white` (the lid's shadow), `open_mouth` (black inside, toned tongue with a darker core, paper teeth with shaded corners, one lip line), `blush` (slanted hatching), `blush_tone` (a graded tone that fades out at its edge), `gloom` (vertical line tone for dread), `sweat`, `closed_eye`, `closed_smile`. See `examples/expressions`. |
| `replay` | `python -m manga.replay journal.jsonl --scale S --out file.png [--bw] [--until N]`, or `--dpi 600 [--fit letter] --bw` for a printer |
| `settings` | `python -m manga` shows the style settings in effect; see "Your own style settings" above. `tone(..., soft_mm=2)` gives one tone a soft edge |
| `animate` | `python -m manga.animate journal.jsonl --out drawing.mp4 [--gif drawing.gif] [--seconds 20]`: a film of the page being drawn, replayed from its journal in drawing order. Needs ffmpeg. |

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

Learned from review comments on the pool page:

8. **Start each comment with `tools.pin`.** It turns "68% from the left, 76% from the top" into the drawn area
   beside the reference, so you see at once whether the drawing or the reference is the odd one out.
9. **Check every cut with `tools.cut_parts`.** Leaving part of a figure out of a panel (a face sliver at the edge)
   can saw off something that belongs to the beat (the thumb). Look at each piece it reports.
10. **Look at the reference before closing a gap.** A fill that tidies an ugly cut-out may hide something real (the
   deck between her legs). If the cut-out looks bad, redraw the background through it instead.
11. **Fix in reference coordinates, not page pixels.** A fix keyed to page pixels breaks when the layout moves and
   can't be reused in another panel showing the same thing. Edit the material map, or map points with
   `to_page`/`Frame`.
12. **Keep hand-drawn features as data.** The eyes are point lists, so closing them was one switch (`content=True`)
   and an erase of that zone. A traced feature would have needed redrawing from scratch.
13. **An expression is three places.** Brows, eyes and mouth carry it; give them the black and keep the rest of
   the head quiet (hair in tone, not solid black, or black brows vanish against it). Check each face small: if
   the feeling doesn't read at thumbnail size, the brows or the mouth are too thin.

Learned on the expressions panel, over four review rounds:

14. **Look with three tools, not a new script each round.** `tools.strip` shows one part of every figure (all
    eyes), because a fix to one face often breaks another. `tools.layers` puts reference, drawing and the masks
    behind them side by side: a wrong drawing over a right mask is a drawing bug, a wrong mask is a detection
    bug. Leave `tools.keep(...)` at the end of the script and run it with `DUMP=m.npz` to get those masks.
    `examples/expressions/review.py` wires all three up; copy it for a new example.
15. **Send `tools.before_after` with every round.** Copy the output aside before rebuilding. The client sees each
    change without hunting for it, and you catch a regression before they do.
16. **Ask colour once, then stop.** `tools.probe` samples a line across a feature (down a mouth, along a faint
    brow). If the classes overlap there, no threshold will separate it: make it point data in reference px
    (like the sweat drops), say so in the README, and move on. Two rounds went into a brow tail that sat in the
    skin's shadow.
17. **Shape before colour for closed shapes.** An iris or an open mouth is convex: take the hull of its coloured
    pieces, then split it by position (teeth above the dark inside, tongue below). Colour alone left horns,
    gaps and doubled outlines.
18. **One owner per pixel.** When a helper draws a part (iris, teeth, brow wedge), take that area out of the traced
    blacks and line passes, and lay every paper ring before any ink. Otherwise a later pass redraws it: a dot
    beside the iris, a second lip line, a ring eating a neighbour.
19. **Gradients as graded tone, not bands.** A flush is `face.blush_tone` (fades with distance from its edge) over
    an area wider than the colour class, with `face.blush(..., soft=True)` hatching that fades with it.

Learned on the nine-expression grid, over five review rounds:

20. **A comment is one example of a class.** Twice a pinned comment ("the iris runs past its outline") turned out
    to hold on several faces. Before fixing, look at every instance: `tools.boxes_of(mask)` gives a box round each
    detected piece, and `tools.beside(ref, drawn, boxes)` puts each one next to the reference. Reply only after
    the whole set is right. `review.py --each iris` does it in one line.
21. **Open a pin with `tools.triptych`.** Reference, drawing and colour classes round the point, in one row. It
    says at once whether the classes are wrong (a prep or detection bug) or the drawing is (`review.py --pin X Y`).
22. **Try detection without drawing.** Masks are decided before the first ink, so `tools.run_until('panel.py',
    marker)` returns them in seconds instead of a full build. Print a line of stats per detection (area, colour
    share, what lies beside it); a false one (hair read as an iris, a cheek as an eye) stands out, and a gate on
    context (an amber core and an eye white beside it) removes it more surely than any colour threshold.
23. **A brow is one stroke, thick to thin.** `face.brow_stroke(wedge, cand)` follows the reference's thin tail
    from the wedge, measures width across the stroke (down a column overstates a sloping brow), drops a steep
    fringe tip with a level opening, and only narrows after the widest point. Lay `face.brow_ring` for all brows,
    then ink them. A brow drawn darker than the hair, or touching a fringe tip, is found by cutting the dark
    pixels free of the hair mass with a 3 px gap, not by colour.
24. **Use the reference's own outlines as borders.** `face.iris_to_rim` fills an iris out to its drawn rim and
    nothing past it; a hull of colours stops short where the iris is pale and runs over where the white is
    shaded. Likewise a shadow cast by one part comes from that part only: `face.eye_white` takes the lash band,
    not every dark pixel, or the iris's lower rim casts tone under the iris.
25. **Zones overlap: say which wins.** The eye zone reached the ears, and a dark lock over an ear was taken for a
    lash band and inked solid. Where two zones meet, decide the owner by what lies beside it (a lash band touches
    an eye white), not by which zone came first.

Learned from a published page (measured, not kept):

26. **Leave the colour art's shading out.** A manga inker keeps skin white (only the shadow under the chin and
    the one the hair casts stay), draws the nose as one stroke, and puts the weight into a few places: one
    continuous outline round the figure, heavier than the inner lines; the upper lash band, the heaviest mark on
    the face; a few blacks. `face.manga_iris` draws a mostly paper iris, `face.lid_dashes` a lower lid as
    dashes, and `effects.halo` the white gap that cuts a figure off a patterned background.

## Examples

- `examples/sampler.py`: every tool on one sheet.
- `examples/pool_page/`: a four-panel manga page drawn from a reference illustration (see its README). This is
  the recommended way to work: start from a reference image.
- `examples/expressions/`: eight expressions of one face as a single panel, drawn from a reference sheet. It shows
  how brows, eyes, mouths, blush, gloom and sweat are drawn with `manga.face` (see its README).
- `examples/expressions_grid/`: nine expressions of a boy in a 3 x 3 grid with a dark collar; irises layered by
  position and filled to their rim, brows as single tapered strokes, gritted teeth, a blush band (see its
  README). Its `review.py` has `--pin` and `--each`.
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
