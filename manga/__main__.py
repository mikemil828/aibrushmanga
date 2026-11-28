"""python -m manga  -- show the style settings in effect and the files they came from."""
import json
import os

from . import settings as S
from .pens import PENS


def main():
    print('settings files, in the order they were laid on:')
    for p in S.files():
        print('  ' + p)
    if not os.path.isfile(S.LOCAL_FILE):
        print(f'  (no {S.LOCAL_FILE} yet: copy the lines you want to change from settings.default.toml into it)')
    print('\npens in effect (width and taper in mm):')
    for name, p in PENS.items():
        print(f'  {name:10s} {p.min_mm:.3f}-{p.max_mm:.3f}  taper {p.taper_in_mm:.2f}/{p.taper_out_mm:.2f}')
    print('\nfunction defaults set by the settings:')
    for name in sorted(S._registry):
        sec = {k: v for k, v in S.section(name).items() if not isinstance(v, dict)}
        if sec:
            print(f'  [{name}] ' + json.dumps(sec))


if __name__ == '__main__':
    main()
