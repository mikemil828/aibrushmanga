# Pencil sketch: a bust without a reference

Run from this folder: `python3 draw.py` (about 3 s). It writes:

- `thumbnails.png`: two poses in pencil, side by side. The first (turned to the viewer's right, leaning a
  little) was chosen.
- `sketch.png`: the pencil layer alone: head construction from `sketch.head`, neck and shoulder lines, the
  hair mass, rough eyes and irises, and the bangs as gesture lines from the parting.
- `sketch-under-ink.png`: the ink over the pencil (non-photo blue), to see where the ink left the decision.
- `page.png` and `page.jsonl`: the finished page. The pencil is not in it, and `--bw` or `--dpi` print files
  never show it. `python -m manga.replay page.jsonl --sketch on --out x.png` shows it again.

The plan is at the top of `draw.py`. Every inked feature is placed from the points `sketch.head` returned
(eyes, brows, nose, mouth, chin, jaw, ear) or from points named in head units (`Lp(u, v)`), so moving the
head or changing its turn moves the whole drawing with it.

## Weak spots (honest list)

- The face has no shading beyond the shadow under the chin; it reads flat next to the hair.
- The bangs are evenly spaced, stiff locks drawn from one parting; real bangs overlap and vary in width.
- The ring of light on the hair has a regular sawtooth edge.
- The neck is a little long and the shoulders plain; there is no body under the collar.
- The far side lock bulges outward just below the crown.
- The head construction is a simple sphere-and-jaw model: it is good for turns up to about three-quarter
  view and small nods, not for profiles or strong up/down views.
