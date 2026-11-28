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

## Second round: lessons from a face tutorial

From animeoutline.com's anime girl head tutorial (read, not copied): its construction matches `sketch.head`
(circle, jaw lines to a small pointed chin, ears between the head's middle and the half-way point to the chin,
nose half-way down, mouth between nose and chin). Applied here:

- Hair as clumps of uneven width and length, notches of uneven depth, locks curving away from the parting,
  split lines in the widest clumps and two loose strands at the temples.
- Cast shadows, light from the upper left: the fringe on the forehead (a lighter tone than the hair, so it
  reads as skin in shade), the upper lids on the eyes, the head on the neck. Skin is otherwise paper.
- The mouth as one line with a break at its lowest point; the nose as one small stroke (a toned nose shadow
  was tried and read as a smudge).

## Third round: the neck

The neck read as far too long. By the numbers it was only about a quarter of the head's height; it looked
long because its lines started up by the ears, it was narrow, and the shoulders met it at its foot, like a
tube. Now: chin to collar 0.4 R, the neck leaves the jaw about level with the mouth and widens a little
toward its base, and the shoulders rise into its sides (the trapezius) and meet it above the collar.
animeoutline.com's full-body anime girl tutorial (6.5 heads tall, legs about as long as head and body) gives
no neck or shoulder measurements.

## Fourth round: Loomis side planes

`sketch.head` is now built in 3D the Loomis way and then turned: the cranium ball is sliced flat on both sides
(the side planes, at 0.82 of the radius), the ear sits just behind the side plane's vertical line between the
brow and nose lines, and the jaw hangs from below the side plane to the chin. The face has a depth profile down
its centre line (brow, a small nose, lips, chin), so a turned head shows it on the far side. On the side
turned away, the outline is the outer envelope of cheek, jaw and profile; on the side that turns toward us,
the cheek moves inside the face and the outline becomes the jaw from under the ear. `turn` now reaches a
full profile (1.0), and `nod` tips the whole head up to 30 degrees, so the feature lines wrap round it.
`turnaround.py` draws ten poses (facing, 30 degrees, three-quarter, near profile, profile, looking down and up,
leaning) into `turnaround.png`. thedrawingcodex.com's Loomis manga page (read, not copied) gave the method
but no measurements; the numbers are mine.

On this page the turn reads more clearly: the far eye is narrower and closer to the far cheek, the far cheek
slopes in, and the near jaw runs from under the ear to the chin.

Eyes and jaw, second pass. Each eye is now an eyeball set into its socket, behind the face surface, with the
opening (lids) on the ball aimed forward and a little outward. `head` returns its outline as `eye_shape_l` /
`eye_shape_r`: an almond facing us, and in profile a wedge that opens toward the nose with the cornea rounding
its front, set back from the nose line by about a quarter of the cranium radius. The jaw now leaves its
corner steeply and rounds off toward the chin, so the near jaw bows a little instead of running ruler-straight
from the ear; the chin keeps its point. The example's neck now starts on the jaw line, so it tucks under it.

Third pass. `head` now also returns the lids themselves, `eye_upper_l/r` and `eye_lower_l/r` (inner corner to
outer): the upper lid full and arched, highest toward the outer corner, the lower one shallower. The construction
eyes are a little larger (manga proportions), and this example inks the opening, lash band, crease, lower lid
and iris straight from those curves, so the inked eye is the constructed one (the iris sits on the lower lid
and the catch-lights stay below the lash band). On the side turned away, the outline is now the limb of the
cheek: across each height the face rounds from the profile out to the cheek contour, and the outline is where
that stands out furthest. It follows the cheekbone and curves in to the chin. Near profile it hands back to the
contour, so nose and lips still show.

Fourth pass. Where the eyeball turns away and the lids stop short of their corner (the far eye's outer corner,
the near eye's inner corner in profile), `head` closes the gap with a rounded edge shared by both lids, so the
corner is soft instead of cut straight. The lash band is thinner inside the opening and the iris's solid top
is smaller, so the iris's dark-to-light top shows under the band. The far cheek is now the cheek's limb row by
row (no convex hull) whenever the head is not near profile, eased with a short moving average, and rounded
toward a smooth front instead of the nose and lips, so it can hollow a little below the cheekbone in a
three-quarter view without lumps.

Fifth pass. The rounded eye corner now runs along the eyeball's own edge on the page (the limb: a circle of
the ball's radius round its projected centre), so it wraps the ball. The cheek's cross-section is a flattened
quarter (a superellipse, `CHEEK_ROUND` 1.75) rather than a full ellipse, and from about 30 degrees the nose tip
alone may break the cheek line, ahead of the lips and chin. The eyes have their weight back: a larger solid
top on the iris, a denser tone and a larger pupil, with the gradient still showing under the lash band.

## Weak spots (honest list)

- Beyond those few shadows the face is flat; there is no form shading on cheek or jaw.
- The clumps still all start from one parting point, so the fringe fans out more than real hair does.
- The ring of light on the hair has a regular sawtooth edge.
- The shoulders are plain; there is no body under the collar.
- The far side lock bulges outward just below the crown.
- With this small manga nose the far cheek still covers the nose tip in a true three-quarter (turn 0.71); the
  tip breaks the outline only from about 0.8. Showing it earlier would take a more prominent nose, which is a
  style choice.
- Where a lid meets the eyeball's edge on a near-profile eye there is a slight kink.
- No hairline or neck in the construction itself; the neck is still placed by the example.

## Angles sheet (`angles.py`, `character.py`)

`character.py` draws the bust's girl at any turn, tilt and nod on the same `sketch.head()` construction: hair as
3D point sets (back hair, side locks, fringe clumps) projected and layered in painter's order, eyes inked from the
lids, neck and sailor collar turning half as far as the head. `angles.py` draws six poses (facing, three-quarter,
profile, looking down, looking up, the other side) into `angles.png` and `angles-sketch.png`.

Round 2: the side lock's front edge sits further back and bows over the cheekbone (the profile shows its face,
not a slab); past three-quarter the irises follow the head instead of looking at us; the hairline comes down at the
temples; the back hair's hem is a squared ring, so it curves seen from below and the bob flares at the back corners;
lock edges are inked only below the brow, and stubs shorter than 0.06R are dropped.

Weak spots: the far cheek at a 0.7 turn is still nearly straight (the nose tip doesn't break it yet; that's in
`sketch.head`); the profile eye's lash wedge sits a little apart from the lid; small steps where a lock meets the
back hair's outline ("looking down", "other side"); fringe strokes on the far edge end abruptly. `draw.py` is
unchanged.
