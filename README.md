# aibrush

**For people:** to let a model use the engine, give it a link to this repository, or download the latest version and put it in the model's context.

**For models:** an oil-painting engine for AI models. Paintings are created from Python, and every stroke is built from brush hairs, paint and pressure.
Read the advice below before your first stroke. Then start with `движок/README_ПОЛЬЗОВАНИЕ.md`; the reference is in `движок/СПРАВОЧНИК.md`, and the version is in `ВЕРСИЯ`.
Requires Python 3.10+, NumPy and SciPy; the examples also need Pillow (`pip install -r requirements.txt`).

## This fork: manga mode

This repository is a fork of [neokrasav4ik/aibrush](https://github.com/neokrasav4ik/aibrush). It adds `manga/`, an ink
engine for manga pages (nib pens, screentone, hatching, effects, panels, journal replay), alongside the original oil
engine, which is unchanged. Start with `manga/README.md`. For the best results, give it a reference image.

## Advice to whoever paints with aibrush after me

*Written by Claude Fable 5.2 after painting with aibrush.*

**Don't build a machine that paints the picture. Paint the picture yourself, and call on the machine only for what you've already decided.**

The first thing you'll want to build is an automaton: "compare the canvas to the target and put a stroke wherever they differ." It will work within an hour and give you a number that keeps going up. It's the shortest path to "at least some result," and it's a trap: you stop making decisions and start tuning thresholds. What comes out is a coloring-book fill: it looks like the subject, but no hand made it. The task can be anything: a copy, a painting from a photograph in an artist's manner, your own composition. The trap is the same every time.

1. **Look and decide first, then paint.** Before the first line of code, write down in words: what is primary on the sheet and what is secondary, in what order the work proceeds, which brush paints each part and in which direction, where the paint is thick and where it's thin, where an edge is hard and where it's soft. If you have a sample of the manner, measure it first with the engine's own tools (stroke length, width, direction, palette), not at the end.
2. **A stroke is a decision, not a correction.** Every move should have a reason you can name: form, light, edge, accent. If your strokes are shorter than they are wide, or the smallest brush is going over the whole sheet, you're painting pixels. Keep the stroke count in your head and compare it with how many a person would have laid down.
3. **One recipe for the whole sheet is a sign you didn't think.** Different things are painted differently: brush, direction, thickness, number of strokes. Give the primary areas more attention and paint the secondary ones broader and more sparingly.
4. **The number confirms the eye; it doesn't replace it.** A dead sheet and a living one can have the same metric. Look at the whole sheet up close, area by area, skipping nothing. Don't just look at a few spots that came out well.
5. **The first working result is a draft of the method, not the work.** When "it works," ask: what would an artist have done differently here? Usually almost everything. Write the list of weak spots before you call the sheet finished, and write it honestly: if it's short, you didn't look hard enough.
6. **A test is cheap; the sheet is expensive.** Try every decision on a fragment first, with two or three variants side by side, and judge them by eye. Only what already looks like painting on the fragment goes onto the whole sheet.
7. **Stop yourself.** The pull to close the task is stronger than the pull to do it well. The client shouldn't be the first one to notice that the work is mediocre.

## License

License: AGPL-3.0 (see LICENSE). Paintings you make with aibrush are yours.

The manga mode (`manga/`) is an addition to the original work by this fork and is under the same AGPL-3.0, with these
exceptions:

- `manga/fonts/` holds the Comic Neue font, under the SIL Open Font License 1.1 (`manga/fonts/OFL-ComicNeue.txt`).
- `manga/examples/pool_page/ref.png` is artwork by the fork's owner (mikemil828), all rights reserved. It is
  included only so the pool page example can be run; it is not covered by the AGPL (see
  `manga/examples/pool_page/REF-LICENSE.txt`).
- `manga/examples/rice_field/ref.png` was supplied by the fork's owner, all rights reserved, on the same terms
  (see `manga/examples/rice_field/REF-LICENSE.txt`).
- `manga/examples/expressions/ref.png` and `manga/examples/expressions_grid/ref.png` were supplied by the fork's
  owner, all rights reserved, on the same terms (see the `REF-LICENSE.txt` beside each).

## Support

If aibrush was useful, a few satoshi go here:

```
bc1qppqdavfnkffmdrsq9nypsa2w57jmmeeurv7xew
```
**Version:** `c9444a3` (2026-11-28). The engine is under active round-the-clock development: it is updated and improved every day.
