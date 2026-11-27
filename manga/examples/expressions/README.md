# Eight expressions

One head, eight feelings, drawn as a single manga panel from a reference sheet (`ref.png`): laughing, angry,
sad, surprised, frightened, smug, sulking, embarrassed. It is the example for `manga.face`, the engine's
expression parts.

    python3 prep.py      # reads the reference into named parts -> parts.npz, parts_preview.png
    python3 panel.py     # draws panel.png and the journal page.jsonl (about a minute)
    python3 -m manga.replay manga/examples/expressions/page.jsonl --scale 2 --out panel_x2.png   # print size

## How an expression is drawn

The head, hair and shirt are the same eight times, so they stay quiet: one dark tone for the hair with ink
strand lines and a scraped shine, one light tone for skin in shadow, a single line for the shirt. Everything
that changes gets the black and the care.

| Part | What carries the feeling | How it is drawn |
|---|---|---|
| Brows | slant, weight, how close they come | `face.brow`: a solid tapered shape through each brow's measured centre line; thin brows as G-pen lines |
| Eyes | how far the lids open, how much white shows, how small the iris is | lash bands in solid black; the white left as paper with the lid's shadow (`face.eye_white`); `face.iris` in layers: black under the lid, tone below, rim, streaks, pupil, catch-lights scraped last |
| Mouth | open or closed, its curve | `face.open_mouth`: black inside, toned tongue, paper teeth, one lip line; closed mouths as single G-pen lines |
| Cheeks | flush or embarrassment | `face.blush`: six light slanted strokes per cheek, or a tone band with dense hatching across the nose |
| Fear | dread | `face.gloom`: vertical line tone over the eyes, heavier at the top; `face.sweat` drops placed by hand |

`prep.py` separates the parts by colour (the reference is flat-coloured), at the panel's own resolution.
`panel.py` decides what each part becomes; its plan is at the top of the file.

## Weak spots

- On the smug face the right eye's catch-light sits under the lid; it is lost, and a small black dot is left
  beside the iris.
- The thin, hair-coloured brows of the smug face are mostly hidden by the fringe in the reference and come out
  as two small hooks.
- The half-closed irises are a flat light tone with a pupil: readable, but without the lid's shadow the big
  eyes have.
- The shirt is deliberately plain; its folds are a pale tone, not drawn cloth.

`ref.png` is not covered by the code's license: see `REF-LICENSE.txt`.
