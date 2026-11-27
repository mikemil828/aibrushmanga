# Nine expressions, one boy (3 x 3 grid)

A single manga panel drawn from a flat-colour reference sheet: one boy's head nine times, divided 3 x 3
(neutral, laugh, anger / sad, shock, smug / worry, serious, embarrassed).

    python3 prep.py            # parts.npz: the reference separated by colour at panel resolution
    python3 panel.py           # panel.png and page.jsonl
    python3 review.py          # eyes.png, mouths.png; --old prev.png for before-after.png;
                               # DUMP=m.npz python3 panel.py, then --face N --part eyes --masks m.npz

`panel.py`'s plan is at the top of the file. It reuses the approach of `../expressions` and the review tools
in `manga.tools`.

## What is new compared with the girl's sheet

- A 3 x 3 grid with the reference's thin dividers kept as inner borders (`prep.COLS`, `prep.ROWS`).
- A dark collar: a darker tone than the hair, with its deepest folds black.
- The hair's shine as a lighter tone inside the hair tone, and the near-black hair shadow by the jaw treated as
  hair, not as ink lines.
- Irises layered by position (solid under the lid, tone fading down, pale amber low), because the
  reference's iris is a gradient whose middle tones fall between the colour classes.
- Gritted teeth: the opening is the white itself, with the reference's own tooth lines.
- The pupil is the black piece nearest the iris's middle, not the lash band lying across its top.
- Every iris colour piece within reach of the amber belongs to the iris (the pupil can split them).
- The eye white's rim: only its lower half is drawn, as a light lid line; pupils inside the white stay solid.
- The collar is one shape (its navy plus the near-black inside) with a tone graded by the reference's light.
- Near-black hair shadow behind the ears is hair (by darkness); blacks round the ears are hairlines.

## Weak spots

Third round: the ears have an outline against the background (G-pen, clipped to the ear zones away from the
hair), the blacks round the ears are hairlines, and the collar's folds are graded more strongly.

- Where the hair overlaps the ear there are still a few short dark strokes.
- The collar folds follow the reference's light, so they read as soft shading rather than drawn creases.

`ref.png` is not covered by the code's license: see `REF-LICENSE.txt`.
