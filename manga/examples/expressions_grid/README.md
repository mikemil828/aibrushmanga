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

## Weak spots

- The neck shadow is speckled where the reference uses two close skin tones.
- The ears are busy: their inner lines and shadow come out as small dark marks.
- The collar is one dark tone; the reference's folds and the light along its rim are mostly lost.
- The embarrassed face's left iris is darker and blotchier than the right.
- The angry face's lower lid lines run a little long at the outer corners.

`ref.png` is not covered by the code's license: see `REF-LICENSE.txt`.
