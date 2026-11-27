"""Build the pool page in one step: masks, material map, page, and optionally the print-size replay.

    python manga/examples/pool_page/run.py            # writes page/page.png and page/page.jsonl
    python manga/examples/pool_page/run.py --print 2  # also page/print.png at 2x (16 px/mm)

Works from any folder; each script runs in the folder it expects, in the order it needs."""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, 'page')
ROOT = os.path.normpath(os.path.join(HERE, '..', '..', '..'))
STEPS = [
    (HERE, ['masks.py']),                    # figure, hair and line masks from ref.png -> ../*.npy
    (PAGE, ['materials.py']),                # material map: skin, suit, scales, hair, paper
    (PAGE, ['hairfix.py']),                  # hair vs. skin corrections
    (PAGE, ['eyes.py']),                     # hand-placed fixes: eyes, mouth, background gaps
    (PAGE, ['page.py', 'page.jsonl']),       # the page itself
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--print', dest='scale', type=float, default=None, help='also replay at this scale')
    a = ap.parse_args()
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''))
    for cwd, cmd in STEPS:
        t = time.time()
        print(f'-> {cmd[0]}', flush=True)
        subprocess.run([sys.executable] + cmd, cwd=cwd, env=env, check=True, stdout=subprocess.DEVNULL)
        print(f'   {time.time() - t:.0f} s', flush=True)
    if a.scale:
        print(f'-> replay at {a.scale}x', flush=True)
        subprocess.run([sys.executable, '-m', 'manga.replay', os.path.join(PAGE, 'page.jsonl'), '--scale', str(a.scale),
                        '--out', os.path.join(PAGE, 'print.png')], cwd=ROOT, env=env, check=True)
    print('done:', os.path.join(PAGE, 'page.png'))


if __name__ == '__main__':
    main()
