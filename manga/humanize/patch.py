"""Humanize a finished journal: the same page, every line drawn by a hand.

    python -m manga.humanize page.jsonl out.jsonl [--strength 0.15] [--person N] [--seed N]
                             [--part eyes=0.1 ...] [--pen maru=0.5 ...] [--strip]

Defaults come from the [humanize] settings. The output is an ordinary journal: `python -m manga.replay` draws it
without the hand (the hand's points are in each stroke). A humanized stroke keeps its own fields in
"humanize_from", so a journal can be humanized again at another strength (the old hand is taken off first), and
`--strip` (or strength 0) gives the original journal back, byte for byte. The same input, settings, person and
seed give the same output byte for byte.
"""
import argparse
import json
import os
import shutil
import sys

from . import core


def _dump(d):
    return json.dumps(d, ensure_ascii=False, separators=(',', ':'))   # as Sheet._log writes it


def strip_journal(lines):
    """Journal lines with the hand taken off: humanized strokes back to their own fields, humanize lines gone.
    A journal without a hand comes back as the same lines."""
    out, had = [], False
    for l in lines:
        if '"humanize' not in l:
            out.append(l)
            continue
        d = json.loads(l)
        if d.get('op') == core.OP:
            had = True
            continue
        if core.FROM in d:
            had = True
            out.append(_dump(core.strip(d)))
            continue
        out.append(l)
    return out, had


def rewrite(lines, who='patch', jdir=None, **settings):
    """Journal lines (without newlines) -> (new lines, stats). Keywords override the [humanize] settings.
    `jdir`: the journal's folder of rasters (page.jsonl.d), where its part maps are."""
    lines, had = strip_journal([l.rstrip('\n') for l in lines])
    cfg = core.config(**settings)
    stats = dict(humanized=0, left=0, stripped=had, strength=cfg['strength'], person=cfg['person'], seed=None,
                 offset_mm=None)
    if cfg['strength'] <= 0:
        return lines, stats
    hum = None
    out = []
    for l in lines:
        s = l.strip()
        if not s:
            out.append(l)
            continue
        d = json.loads(s)
        op = d.get('op')
        if op == 'sheet' and hum is None:
            seed = int(d.get('seed', 0)) if cfg['seed'] is None else cfg['seed']
            hum = core.Humanizer(cfg, seed, float(d['px_per_mm']))
            stats['seed'] = seed
            out.append(l)
            out.append(_dump(core.header(cfg, seed, who)))
            continue
        if hum is not None and op == 'part':
            hum.enter(d['name'])
        elif hum is not None and op == 'unpart':
            hum.leave()
        elif hum is not None and op == 'part_map':
            if d.get('file') and jdir:
                from PIL import Image
                import numpy as np
                hum.set_map(d['names'], np.asarray(Image.open(os.path.join(jdir, d['file']))))
            else:
                hum.set_map([], None)
        elif hum is not None and op == 'stroke':
            drawn = hum.apply(d)
            if drawn is not None:
                out.append(_dump(drawn))
                continue
        out.append(l)
    if hum is not None:
        import numpy as np
        stats.update(humanized=hum.drawn, left=hum.left)
        if hum.offsets:
            stats['offset_mm'] = tuple(float(v) for v in np.percentile(hum.offsets, [50, 95]))
    return out, stats


def rewrite_file(src, dst, **settings):
    with open(src, encoding='utf-8') as f:
        lines = f.read().split('\n')
    end = lines[-1] == ''
    if end:
        lines = lines[:-1]
    out, stats = rewrite(lines, jdir=src + '.d', **settings)
    with open(dst, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + ('\n' if end else ''))
    # raster arguments (masks, tone levels) sit next to the journal: the new journal needs them too
    sd, dd = src + '.d', dst + '.d'
    if os.path.isdir(sd) and os.path.abspath(sd) != os.path.abspath(dd):
        shutil.copytree(sd, dd, dirs_exist_ok=True)
    return stats


def _pairs(items, what):
    out = {}
    for it in items or []:
        name, _, v = it.partition('=')
        try:
            out[name] = float(v)
        except ValueError:
            sys.exit(f'--{what} wants name=number, not {it!r}')
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('journal')
    ap.add_argument('out')
    ap.add_argument('--strength', type=float, default=None, help='0 = off, 0.15 = default, 0.3 = full')
    ap.add_argument('--person', type=int, default=None, help='which hand (the same number is the same artist)')
    ap.add_argument('--seed', type=int, default=None, help="this sitting's chance (default: the sheet's seed)")
    ap.add_argument('--part', action='append', metavar='NAME=SHARE', help='share of the strength for a part')
    ap.add_argument('--pen', action='append', metavar='NAME=SHARE', help='share of the strength for a pen')
    ap.add_argument('--strip', action='store_true', help='take the hand off: the original journal')
    a = ap.parse_args(argv)
    kw = {k: v for k, v in (('strength', a.strength), ('person', a.person), ('seed', a.seed)) if v is not None}
    if a.part:
        kw['parts'] = _pairs(a.part, 'part')
    if a.pen:
        kw['pens'] = _pairs(a.pen, 'pen')
    if a.strip:
        kw['strength'] = 0.0
    st = rewrite_file(a.journal, a.out, **kw)
    if st['strength'] <= 0:
        print(f'{a.out}: no hand' + (' (the old hand taken off)' if st['stripped'] else ' (unchanged)'))
        return
    off = st['offset_mm']
    print(f"{a.out}: {st['humanized']} strokes by hand (person {st['person']}, seed {st['seed']}, strength "
          f"{st['strength']:g}), {st['left']} as they were"
          + (f'; lines moved {off[0]:.3f} mm typically, {off[1]:.3f} mm at most (95 %)' if off else ''))


if __name__ == '__main__':
    main()
