# Reading the oil engine in English

The oil engine in `движок/` is written in Python with Russian names. Its handbooks (`движок/README_ПОЛЬЗОВАНИЕ.md`, `движок/СПРАВОЧНИК.md`) are already in English prose, but they quote the Russian names. This folder helps you read the code without changing it:

| File | What it is |
|-|-|
| `GUIDE.md` (this file) | How the engine is put together, what each module does, the main API in English, and the string values it accepts. |
| `GLOSSARY.md` | Every one of the ~3,700 Russian names in the code, with an English reading, how often it is used, and where. |
| `views/` | A copy of every module with the names replaced by English. **Reading aid only:** it doesn't run, and comments and messages stay Russian. Always edit the real file in `движок/`. |
| `ru2en.py` | The tool that makes the two files above. It can also translate any name: `python3 docs/english/ru2en.py нагрузка_для` prints `load_for`. |
| `words_extra.py` | The word list behind the translations, added to the engine's own handbook dictionary (`движок/кисть/словарь_справки.py`). Fix a word here and regenerate. |

Regenerate after the engine changes:

```
python3 docs/english/ru2en.py --glossary docs/english/GLOSSARY.md --views docs/english/views
```

The translation goes word by word, so read it as a gloss, not as a design. For example, `прямоугольник_мазка` comes out as `rectangle_stroke`; it means "the stroke's rectangle". A few different Russian names end up with the same English word (`холст`, `хол` and `х` all read as `canvas`).

## The engine on one page

aibrush simulates oil paint physically. It doesn't use textures or stamps. A stroke is built from brush hairs pulling a film of paint along a path, and the picture is whatever that film leaves.

1. **Canvas (`Холст`, `холст.py` → `canvas.py`).** The sheet. It holds named **layers** (`Слой`), each with per-pixel fields: colour (in linear light), coverage, film thickness, wetness, set (skin) and flow. The first layer created is the bottom one. The canvas also owns the ground colour, the canvas grain (`зерно`, woven threads as a height map), the journal and the palette.
2. **Brush (`Кисть`, `инструмент.py` → `tool.py`).** Loaded from a preset in `движок/пресеты/*.json` with a width (`ширина` in px, or `ширина_мм` in mm). It remembers the paint it picked up, so the next stroke starts dirty until you call `вытереть()` (wipe). Paint can be loaded per stroke (`нагрузка=`, load) or charged as live paint that runs out (`набрать(цвет, количество)`).
3. **Path (`путь.py` → `path.py`).** A stroke path is points `(x, y, pressure, dt_ms[, tilt, azimuth, roll])`. `жест` (gesture) and `путь_с_нажимом` (path with pressure) build one from a few anchor points. They spline it, resample it at an even step, and shape the pressure as entry, body and exit.
4. **Stroke (`Холст.мазок`, `мазок` = stroke).** The core call is `х.мазок(points, colour, brush, слой=layer, сид=seed, нагрузка=load, …)`. Inside it:
   - `кисть.py`, `пряди.py`, `набор.py` (→ `brush.py`, `strands.py`, `charge.py`): how the brush is built (leading hairs, ridges, strands) and how the paint is distributed on it.
   - `укладка.py` (→ `laying.py`, the biggest module): transfers paint along the ridges, rasterises it and writes it into the layer.
   - `краска.py` (→ `paint.py`): pigment mixing through a 12-band reflectance curve and Kubelka–Munk (`кс` = K/S), plus paint kinds (opaque … glazing) and thinner.
   - `срастание.py` (→ `growing_together.py`): neighbouring strands closing up, wet paint pushed into a roll at the edge, and the "carry" (`Ноша`) of wet paint dragged along.
   - Special tools: `мягкая.py` (soft brush), `нож.py` (palette knife), `снятие.py` (rag, scraper, handle end), `скотч.py` (masking tape), `правка.py` (re-wetting and pressing dried paint).
   - The call answers with a **response** (`Ответ`, `ответ.py` → `response.py`): the rectangle it wrote plus measurements of the trace (`цвет` colour, `закрыто` covered, `просветы` gaps, `ширина` width, `сырое` wet, `разрывы` breaks, `край` edge).
5. **Drying.** Time doesn't pass by itself. `подсушить(1.0)` dries fully, `схватить(0.5)` forms a skin. Wet paint in the same layer mixes with the next stroke; lower layers mix only with `мешать_ниже=True` (mix below).
6. **Light and output (`свет.py` → `light.py`).** Paint relief becomes a normal map lit from the upper left (matte, small highlight, optional varnish). `картинка()` (image) and `png()` render the sheet.
7. **Journal and replay (`журнал.py`, `повтор.py`, `задним.py`).** With `Холст(…, журнал='sheet.jsonl')`, every call that affects pixels is written as a JSONL line. `python -m кисть.повтор sheet.jsonl --out sheet.png [--scale 2]` replays it, at any scale. `кисть.задним` edits journal lines after the fact (delete, recolour, shift) and rebuilds only what changed.
8. **Speed (`ядра.py`, `плитки.py`, `быстро.py`).** Strokes whose rectangles don't overlap run on several cores with bit-identical results. Layers are stored as tiles, and a uniform tile is stored as a single number.
9. **Knowing before laying.** Calibration tables and test benches predict a stroke before you paint it. `нагрузка_для` (load for a coverage), `по_таблице` (what a load gives), `план_покрытия` (rows needed for a coverage), `борозды_для` (furrows), `смешение_для` (mixing over wet), `край_для` (edge softness), `предел_светлоты` (the lightness limit) and `к.хватит` (how many mm the paint on the brush will last).
10. **Eyes (`глаз.py` → `eye.py` and friends).** Read-only tools for judging the canvas: `взгляд` (look at a region), `обзор` (overview), `разность` (difference from a target), `окна` (worst windows), `виды` (views), `просвет` (gaps map), `манера` (measure a painting's manner against a reference), `однообразие` / `регулярность` (monotony and regularity of a series of strokes), `сводка` (stage summary) and `подсказки` (hints printed while you work).

## Module map

| Module | English | What it does |
|-|-|-|
| `холст.py` | canvas | The canvas: layers in linear light (float32), ground, saving, 16-bit PNG. Most public methods live here. |
| `инструмент.py` | tool | The brush as a tool: preset, load, charged paint, parsing the colour argument. |
| `кисть.py` | brush | The model of the brush and the stroke: the cross-section profile of the film the brush pulls along the path. |
| `путь.py` | path | Hand points → spline → even step; pressure, speed, brush angle; gestures. |
| `пряди.py` | strands | Brush makeup (leading hairs, ridges, strands), noise, the trace's edge, set-down and lift. |
| `набор.py` | charge | How paints lie on the brush: by ridges, uneven charge, veins, thin spots. |
| `укладка.py` | laying | Laying on the canvas: paint transfer along ridges, raster, writing into the layer. |
| `краска.py` | paint | Pigment mixing: colour → reflectance curve (12 bands) → Kubelka–Munk → colour; paint kinds and thinner. |
| `смешение.py` | mixing | Predicts one sweep over wet paint by number before laying (`смешение_для`). |
| `срастание.py` | growing together | Neighbouring strokes closing up, roll of paint at the edge, dragged wet paint. |
| `борозда.py` | furrow | Live furrow: depth and fill of the furrow under a leading hair change along the sweep. |
| `зерно.py` | grain | Canvas grain: the relief of the ground as woven threads. |
| `мягкая.py` | soft | Soft brush: an edge that fades out and a body without bare gaps. |
| `нож.py` | knife | Palette knife: a stiff smooth edge instead of bristle. |
| `снятие.py` | removal | Lifting wet paint along a sweep: rag, scraper, handle end. |
| `скотч.py` | tape | Masking-tape stencil (`скотч`, `снять_скотч`). |
| `правка.py` | edit | Correcting dried paint: re-wet (`размочить`) and press (`прижать`). |
| `свет.py` | light | Light on the paint relief: normals, light from the upper left, matte/semi/varnish, interreflection. |
| `мм.py`, `масштаб.py` | mm, scale | Physics in mm: what scales with px/mm and what still holds pixels. |
| `числа.py` | numbers | The model's numbers (thresholds, shares, sizes in px), all in one place. |
| `калибровка.py`, `ряды.py`, `фактура.py`, `край.py`, `предел.py`, `хватит.py` | calibration, rows, texture, edge, limit, lasts | Prediction tables and benches: coverage, rows, furrows, edge, lightness limit, how long paint lasts. |
| `ответ.py` | response | The stroke's answer: rectangle plus measurements of the trace. |
| `глаз.py` | eye | Looking at the canvas: sRGB view, OKLab, lightness, difference from a target, gauges. |
| `просвет.py`, `под_верхом.py`, `встреча.py` | gap, under top, encounter | What shows through, what is still visible under upper layers, what a sweep met. |
| `манера.py`, `разбор.py`, `рука.py`, `подпись.py`, `альбом.py`, `мерка_файл.py`, `скоро.py` | manner, parsing, hand, signature, album, gauge file, fast | Measuring a painting's manner from an image and comparing it to the canvas. |
| `однообразие.py`, `сводка.py`, `подсказки.py`, `подсказки_рельеф.py` | monotony, summary, hints, hints relief | Self-checks: repetitive strokes, stage summaries and hints. |
| `палитра.py` | palette | Named mixes: strokes can take a colour by name. |
| `перебор.py`, `варианты.py` | scan, variants | Trying several values or attempts side by side without touching the canvas. |
| `журнал.py`, `повтор.py`, `задним.py` | journal, replay, retroactive | Writing, replaying and editing the stroke journal. |
| `ядра.py`, `плитки.py`, `быстро.py` | cores, tiles, fast | Multi-core strokes, tiled layer storage, fast numpy helpers. |
| `распаковка.py` | unpickling | Safe pickle reading for variant files. |
| `справка.py`, `словарь_справки.py` | help, help dictionary | `python -m кисть.справка <word>` searches the handbook; the Russian→English dictionary. |

## Main API in English

**Canvas: `Холст(ширина, высота, грунт=…, px_на_мм=None, зерно=None, журнал=None, сид=0, ядра=None)`**
= `Canvas(width, height, ground, px_per_mm, grain, journal, seed, cores)`

| Method | English | Purpose |
|-|-|-|
| `мазок` | stroke | Lay one stroke (see above). |
| `пометка(имя, этап=, объект=)` / `заметка` | mark / note | Label the stage and object in the journal. Seeds derive from them. |
| `подсушить`, `схватить` | dry, set | Dry the paint fully or partly. |
| `размочить`, `прижать` | re-wet, press | Correct paint that has already dried. |
| `снять` | remove | Lift wet paint with a rag, scraper or handle end. |
| `скотч`, `снять_скотч`, `ленты` | tape, remove tape, tapes | Masking tape stencil. |
| `палитра`, `палитра_с`, `набрать` | palette, palette from, charge | Named mixes and live paint. |
| `снимок`, `откатить`, `проба`, `варианты`, `перебор`, `взять` | snapshot, roll back, trial, variants, scan, take | Try things and undo. |
| `править` | edit | Retroactive edits to journal lines. |
| `сохранить`, `загрузить`, `закрыть`, `досчитать` | save, load, close, finish computing | Files and the multi-core queue. |
| `картинка`, `png`, `png_слоя`, `png_толщины`, `png_высот`, `png_света` | image, png, layer/thickness/height/light PNGs | Output. |
| `взгляд`, `обзор`, `виды`, `сетка`, `разность`, `окна`, `серая`, `темнее`, `средний`, `переход`, `борозды`, `просвет`, `манера`, `однообразие`, `регулярность`, `сводка`, `подсказки` | look, overview, views, grid, difference, windows, gray, darker, mean, transition, furrows, gaps, manner, monotony, regularity, summary, hints | Eyes and self-checks. |
| `слои`, `слой`, `убрать_слой`, `порядок_слоёв` | layers, layer, remove layer, layer order | Layers. |

**Brush: `Кисть.из_файла('пресеты/плоская.json', ширина=48 | ширина_мм=6)`** = `Brush.from_file(preset, width | width_mm)`
Methods: `вытереть` wipe, `промыть` rinse, `набрать(цвет, количество)` charge, `осталось` left, `хватит` lasts, `копия` copy, `грязная` dirty, `в_файл` / `из_словаря` save/load.

**Paths:** `жест(ctrl, размер, вид, rng, нажим=, скорость=)` gesture; `путь_с_нажимом(…)` path with pressure; `нажим_вдоль` pressure along; `участки_касания` touch segments; `ломаная` polyline; `тычок(x, y, нажим)` jab.

**Colour:** `Краска(rgb, вид)` = Paint(rgb, kind); `смешать(a, b, доля_b)` mix; `цвет(L, C, h)` colour from OKLCh; `светлота` lightness; `шестн` hex; `разность_цветов` colour difference.

## Values passed as Russian strings

| Argument | Values → English |
|-|-|
| `нагрузка` (load) | `'протирка'` rub (0.10), `'обычная'` normal (0.28), `'пастозная'` impasto (0.90), `'сухая'` dry (half-empty, the trace breaks), or a number |
| `вид` of `Краска` (paint kind) | `'кроющая'` opaque, `'полукроющая'` semi-opaque, `'прозрачная'` transparent, `'лессировочная'` glazing, `'медиум'` medium (no pigment) |
| `раскладка` (how a set of paints lies) | `'полосы'` bands, `'ряд'` in order, `'смесь'` blend, `'кончик'` tip (second paint at one edge) |
| `жест` kinds | `'подъём'` lift, `'отрыв'` flick off, `'остановка'` stop (press and lift in place), `'смахивание'` sweep away |
| Presets (`пресеты/`) | `плоская` flat, `плоская_узкая` narrow flat, `круглая` round, `мягкая` soft, `сухая` dry, `веерная` fan, `растрёпанная` splayed, `лайнер` liner, `мастихин` palette knife |
| Layer names in examples | `'основа'` base, `'низ'` bottom, `'лессировка'` glaze |

Preset fields: `имя` name, `форма` shape, `ширина` width, `ряды` rows, `шаг` step, `неровность_шага` step unevenness, `разброс_длин` length spread, `разброс_жёсткости` stiffness spread, `запас` reserve, `вязкость` viscosity, `подбор` pickup, `толщина` thickness, `укрывистость` hiding power, `отставание` lag, `распластывание` splaying, `клочки` tufts, `сухость` dryness, `валик` roll, `вынос` carry-out, `нагрузка` load, `разброс_тона` tone spread, `остриё` point.

## Short names

Short local names follow loose habits: `х` is the canvas, `к` a brush, `о` a stroke response, `мм` millimetres, `кс` K/S (Kubelka–Munk), `ш` width, `в` height, `н`/`нж` pressure, `т` thickness, `ц` colour, `д` a share or a length, `сл` a layer, `кр` an edge. Other one- and two-letter names are local scratch variables; the glossary shows them as plain Latin letters.

## Where the manga engine fits

`manga/` (in English) is separate from all of this. It reuses no oil physics, only the same ideas: pressure paths, a journal, and replay at scale. See `manga/README.md`.
