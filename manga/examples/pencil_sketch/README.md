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

## Weak spots (honest list)

- Beyond those few shadows the face is flat; there is no form shading on cheek or jaw.
- The clumps still all start from one parting point, so the fringe fans out more than real hair does.
- The ring of light on the hair has a regular sawtooth edge.
- The shoulders are plain; there is no body under the collar.
- The far side lock bulges outward just below the crown.
- The far cheek, from the cheekbone down to the chin, is still close to a straight line.
- The construction's eye opening is a plain lens with sharp corners; it gives the placement and the
  foreshortening, not the lid shapes, which are still inked by hand from it.
- No hairline or neck in the construction itself; the neck is still placed by the example.
