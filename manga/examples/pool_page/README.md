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

## page/ — a manga page using the image as reference only

Four panels on a B4 page: the place (umbrella, sea, loungers, drawn by hand from points read off the
reference), the wave (hand close-up with speed lines), the face (flash, hand-drawn eyes, blush hatching,
balloon) and the full figure (G-pen contour, maru inner lines, two-tone cel shading per material).

Run from `page/`, with the reference at `../ref.png` and test 3's `fig.npy`, `hair.npy`, `lines.npy` in `..`
(from `masks.py`):

    python materials.py && python hairfix.py && python eyes.py   # material map: skin, suit, scales, hair
    python page.py page.jsonl                                    # page.png, about 35 s
    python -m manga.replay page.jsonl --scale 2 --out print.png  # print size, about 1 min
