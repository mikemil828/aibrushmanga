# Smug panel: Mio's "heh."

One panel (200 × 150 mm, 1600 × 1200 px at 8 px/mm) drawn without a reference image. Every line and tone
is placed by hand from points in the code.

Mio is an original character made for this example: dark bob with blunt bangs, twin pigtails on round
bobbles, freckles, striped T-shirt.

    python3 panel.py panel.jsonl                                 # writes panel.png and the stroke journal
    python3 -m manga.replay panel.jsonl --scale 2 --out x2.png   # from the repo root: print size

- `face.py`: the half-lidded smug eyes (heavy lid band, irises turned aside, catch-lights), raised brow,
  lopsided smirk with a fang, blush hatching, freckles, face outline.
- `mio.py`: black hair with a gloss ring and white strand lines, bangs, pigtails and bobbles, neck,
  striped shirt with a ribbed collar.
- `panel.py`: background gradient tone, focus lines, drawing order, lettering, border.

The plan that came before any code (what is primary, which pen, where it is thick) is in the example's
history: eyes first, then mouth and brows, face and hair, shirt, background last.
