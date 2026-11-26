# Pool page (test 3)

A grey anime illustration redrawn as an inked, toned manga page, with decisions made by hand per region.

Run from a folder that holds the source image as `ref.png`:

    python path/to/masks.py    # figure, hair and pool masks (polygons drawn for this image), saves *.npy
    python path/to/draw3.py    # draws draw3_beta.png

- Figure: G-pen and maru lines with a 1 px white edge, stock tones a step lighter in the mid range.
- Background: maru only, lighter, no lines under 18 px, tones at 60 %.
- Hair: solid black, gloss scraped out, a few strands drawn back in white.
- Pool: one light tone, ripples scraped to paper.

The polygons fit one image only. For a new image, draw new ones.
