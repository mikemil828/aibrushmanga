# Rice-field panel

A single manga panel inked from `ref.png`: a walker on a path between flooded rice paddies at sunset, with a
village and misty mountains behind.

```
python3 manga/examples/rice_field/prep.py            # region map -> zones.npz (about 10 s)
python3 manga/examples/rice_field/panel.py page.jsonl # the panel -> panel.png + journal (about 20 s)
python3 -m manga.replay manga/examples/rice_field/page.jsonl --scale 2 --out panel_x2.png
```

The decisions behind each part (screens, pens, what is drawn by hand) are written at the top of `panel.py`.
What is drawn by hand rather than traced: the ridge lines of the left and middle mountains, the walker and his
shadow, and each rice seedling (leaves fanned up from the tuft's base, with a broken reflection).

`ref.png` is all rights reserved by the fork owner and is not under the AGPL; see `REF-LICENSE.txt`.
