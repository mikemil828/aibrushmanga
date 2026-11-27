# Pool page

A four-panel B4 manga page drawn from a reference illustration: the place (umbrella, sea, loungers), the wave
(hand close-up with speed lines), the face (flash, hand-drawn eyes, blush hatching, balloon) and the full figure
(G-pen contour, maru inner lines, two-tone cel shading per material).

The reference image, `ref.png`, is artwork by the fork's owner, all rights reserved (see `REF-LICENSE.txt`); it is
here only so the example runs.

## Run

One step, from any folder:

    python manga/examples/pool_page/run.py            # page/page.png and page/page.jsonl, about 1 min
    python manga/examples/pool_page/run.py --print 2  # also page/print.png at print size (16 px/mm)

`run.py` runs, in order: `masks.py` (figure, hair and line masks from the reference), then in `page/`
`materials.py` (material map: skin, suit, scales, hair, paper), `hairfix.py`, `eyes.py` (hand-placed fixes) and
`page.py`.

## What is drawn by hand

The reference supplies the figure's shapes and materials; the inker's decisions go on top:

- Eyes, mouth and cheek scales: points read off the reference (`page/eyes_hand.py`).
- Panel 1 (umbrella, sea, loungers), the big umbrella's ribs and the bushes in the tall panel
  (`page/panel1.py`, `page/umbrella3.py`, `page/bushes3.py`).
- Hair as strands following its flow (`page/hair.py`), an even-weight body contour (`page/contour.py`).
- Corrections to the material map where the reference was misread, one per review note (`page/eyes.py`).

The points and corrections fit this one image. For a new image, draw new ones.
