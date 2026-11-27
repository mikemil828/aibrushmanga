"""Animate a page being drawn, from its journal:  python -m manga.animate page.jsonl --out drawing.mp4

The journal is replayed once and a frame is kept every so often, so the film shows the page in the order it
was drawn: tones and fills first where the script laid them first, then the strokes. Each tone or fill counts
as several strokes' worth of screen time (`--weight`), so the big areas don't flash past while thousands of
grass blades take minutes. Screen time is also shared with visible change (`--even`), so a run of tiny
strokes that barely alter the picture moves along faster. The last frame is held for `--hold` seconds.

Writes MP4 (H.264) and/or GIF with ffmpeg, which must be on the PATH.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np
from PIL import Image

from .replay import replay

HEAVY = ('tone', 'fill', 'apply', 'border', 'caption')


def pick(work, smalls, n_frames, even=0.5):
    """Choose n_frames candidate indices. Progress through the film is a blend of work done (strokes, with
    tones counted heavier) and visible change on screen, so big early tones don't flash past and long runs
    of tiny strokes that barely change the picture don't drag. even=1 is pure work, 0 pure change."""
    change = np.array([0.0] + [float(np.abs(b - a).mean()) for a, b in zip(smalls, smalls[1:])])
    seen = np.cumsum(change)
    w = work / (work[-1] or 1.0)
    c = seen / (seen[-1] or 1.0)
    prog = even * w + (1 - even) * c
    targets = np.linspace(0.0, 1.0, n_frames)
    return [int(min(np.searchsorted(prog, x), len(prog) - 1)) for x in targets]


def animate(journal, out, seconds=20.0, fps=24, width=720, weight=12, hold=2.5, gif=None, gif_width=480,
            gif_fps=12, even=0.5, progress=True):
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is needed to write the animation')
    with open(journal, encoding='utf-8') as f:
        ops = [json.loads(line)['op'] for line in f]
    cost = [weight if o in HEAVY else (1 if o == 'stroke' else 0) for o in ops]
    total = float(sum(cost)) or 1.0
    n_frames = max(int(seconds * fps), 2)
    step = total / (2 * n_frames)                        # candidates: about two per film frame
    tmp = tempfile.mkdtemp(prefix='manga-anim-')
    cand = []                                            # (cost so far, png path, small grey array)
    state = {'acc': 0.0, 'next': 0.0}

    def grab(sh):
        im = sh.image()
        h = int(round(im.height * width / im.width / 2)) * 2
        im = im.resize((width, h), Image.LANCZOS)
        path = os.path.join(tmp, f'c{len(cand):05d}.png')
        im.save(path)
        small = np.asarray(im.convert('L').resize((width // 4, h // 4), Image.BOX), np.float32)
        cand.append((state['acc'], path, small))

    def on_op(i, op, sh):
        if not cand:
            grab(sh)                                     # the blank sheet
        state['acc'] += cost[i]
        if ops[i] in HEAVY or state['acc'] >= state['next'] + step:
            state['next'] = state['acc']
            grab(sh)

    t = time.time()
    sh = replay(journal, on_op=on_op, progress=progress)
    grab(sh)
    frames = pick(np.array([c[0] for c in cand]), [c[2] for c in cand], n_frames, even)
    for k, j in enumerate(frames + [len(cand) - 1] * int(hold * fps)):
        os.link(cand[j][1], os.path.join(tmp, f'{k:05d}.png'))
    state['k'] = k + 1
    pattern = os.path.join(tmp, '%05d.png')
    written = []
    if out:
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern,
                        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', out],
                       check=True)
        written.append(out)
    if gif:
        vf = (f'fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];'
              f'[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(fps), '-i', pattern,
                        '-vf', vf, gif], check=True)
        written.append(gif)
    shutil.rmtree(tmp, ignore_errors=True)
    if progress:
        print(f'{", ".join(written)}: {state["k"]} frames, {time.time() - t:.0f}s')
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('journal')
    ap.add_argument('--out', help='MP4 file to write')
    ap.add_argument('--gif', help='GIF file to write (smaller, lower frame rate)')
    ap.add_argument('--seconds', type=float, default=20.0, help='length before the final hold')
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--width', type=int, default=720, help='MP4 width in px')
    ap.add_argument('--weight', type=float, default=12, help='screen time of a tone or fill, in strokes')
    ap.add_argument('--even', type=float, default=0.5,
                    help='0..1: share of screen time spread by work done; the rest follows visible change')
    ap.add_argument('--hold', type=float, default=2.5, help='seconds to hold the finished page')
    a = ap.parse_args()
    if not a.out and not a.gif:
        ap.error('give --out and/or --gif')
    animate(a.journal, a.out, a.seconds, a.fps, a.width, a.weight, a.hold, a.gif, even=a.even)


if __name__ == '__main__':
    main()
