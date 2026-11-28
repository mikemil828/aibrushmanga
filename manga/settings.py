"""Your own style settings, kept safe from `git pull`.

The engine reads its style knobs (pen widths, hatching, tone, soft or hard shadow edges, panel borders) from TOML
files, each one laid over the one before:

  1. manga/settings.default.toml   shipped with the repo: the engine's own values. Don't edit it, a pull
                                   would overwrite it.
  2. manga/settings.local.toml     yours. Ignored by git, so a pull never touches it. Copy only the lines you
                                   want to change from the default file.
  3. $MANGA_SETTINGS               a file named by this environment variable (a per-project style, say).
  4. --settings FILE               on the command line of any script that imports manga.
  5. manga.settings.use(FILE)      from code, at any point (or settings.override('hatch', spacing_mm=0.5)).

A setting is only a default: a value passed in the call itself (hatch(..., spacing_mm=0.5)) always wins. Each
section names the function whose defaults it sets ([hatch] -> hatch(), [effects.border] -> effects.border()),
except [lines] and [pens.*], which change the pens themselves.

Journals record the values a page was actually drawn with, so `manga.replay` redraws a page the same way
whatever your settings are now.

    python -m manga          # the settings in effect, and the files they came from
"""
import functools
import inspect
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(HERE, 'settings.default.toml')
LOCAL_FILE = os.path.join(HERE, 'settings.local.toml')
ENV_VAR = 'MANGA_SETTINGS'

_values = {}         # merged settings
_files = []          # files that were read, in order
_registry = {}       # section name ('hatch', 'effects.border') -> function whose defaults it sets


def _toml():
    try:
        import tomllib                      # Python 3.11+
        return tomllib
    except ModuleNotFoundError:
        try:
            import tomli                    # Python 3.10: pip install tomli
            return tomli
        except ModuleNotFoundError:
            return None


def _read(path):
    t = _toml()
    if t is None:
        raise ImportError(f'reading {path} needs Python 3.11+, or `pip install tomli` on Python 3.10')
    with open(path, 'rb') as f:
        return t.load(f)


def _merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _tuples(v):
    """TOML has lists only; the engine's defaults are tuples."""
    if isinstance(v, list):
        return tuple(_tuples(x) for x in v)
    return v


def section(name):
    """The merged values of one section ('hatch', 'effects.border'), or {}."""
    d = _values
    for part in name.split('.'):
        d = d.get(part) if isinstance(d, dict) else None
        if d is None:
            return {}
    return d


def configurable(name):
    """Decorator: keyword defaults of the function come from settings section `name` (call arguments win)."""
    def deco(fn):
        sig = inspect.signature(fn)
        _registry[name] = fn

        @functools.wraps(fn)
        def wrapper(*args, **kw):
            sec = section(name)
            if sec:
                given = sig.bind_partial(*args, **kw).arguments
                for k, v in sec.items():
                    if k not in given and k in sig.parameters and not isinstance(v, dict):
                        kw[k] = _tuples(v)
            return fn(*args, **kw)
        wrapper.__signature__ = sig
        return wrapper
    return deco


def _argv_file():
    """Take `--settings FILE` / `--settings=FILE` out of sys.argv, so the script's own argument parsing never
    sees it."""
    argv = sys.argv
    for i, a in enumerate(argv[1:], 1):
        if a == '--settings' and i + 1 < len(argv):
            path = argv[i + 1]
            del argv[i:i + 2]
            return path
        if a.startswith('--settings='):
            del argv[i]
            return a.split('=', 1)[1]
    return None


def load(argv=True, user=True):
    """Read the settings files in order and apply them. Called once when manga is imported.
    user=False reads the shipped defaults only (the tests use it, so your own settings can't fail them)."""
    _values.clear()
    _files.clear()
    paths = [DEFAULT_FILE]
    if user:
        paths += [LOCAL_FILE, os.environ.get(ENV_VAR) or None, _argv_file() if argv else None]
    for i, p in enumerate(paths):
        if not p:
            continue
        if i >= 2 and not os.path.isfile(p):
            raise FileNotFoundError(f'manga settings file not found: {p}')
        if not os.path.isfile(p):
            continue
        if i == 0 and _toml() is None:
            continue                         # the default file only repeats the engine's own values
        _merge(_values, _read(p))
        _files.append(p)
    _apply(', '.join(os.path.basename(p) for p in _files) or 'settings')


def use(path):
    """Lay another settings file over the current ones (from code: a style for one page)."""
    _merge(_values, _read(path))
    _files.append(path)
    _apply(os.path.basename(path))


def override(name, **values):
    """Change settings from code: override('hatch', spacing_mm=0.5), override('lines', width_scale=1.2)."""
    d = _values
    for part in name.split('.'):
        d = d.setdefault(part, {})
    d.update(values)
    _apply(f'override({name!r})')


def reset():
    """Back to the files only (drops use() and override())."""
    load(argv=False)


def _apply(origin):
    _check(origin)
    _apply_pens()


def _apply_pens():
    from .pens import PENS, BUILTIN_PENS, Pen
    fields = [f for f in inspect.signature(Pen).parameters if f != 'name']
    PENS.clear()
    PENS.update(BUILTIN_PENS)
    for name, over in section('pens').items():
        if not isinstance(over, dict):
            continue
        over = {k: v for k, v in over.items() if k in fields}
        if name in PENS:
            PENS[name] = PENS[name].with_(**over)
        else:
            try:
                PENS[name] = Pen(name, **over)
            except TypeError:
                warnings.warn(f'manga settings: new pen [pens.{name}] needs at least min_mm and max_mm')
    lines = section('lines')
    ws, ts, wb = (float(lines.get(k, 1.0)) for k in ('width_scale', 'taper_scale', 'wobble_scale'))
    if (ws, ts, wb) != (1.0, 1.0, 1.0):
        for name, p in list(PENS.items()):
            PENS[name] = p.with_(min_mm=p.min_mm * ws, max_mm=p.max_mm * ws, taper_in_mm=p.taper_in_mm * ts,
                                 taper_out_mm=p.taper_out_mm * ts, wobble_mm=p.wobble_mm * wb)


def _check(origin):
    """Warn about names the engine doesn't know (a typo would otherwise silently do nothing)."""
    from .pens import Pen
    known_top = {n.split('.')[0] for n in _registry} | {'lines', 'pens'}
    def warn(msg):
        warnings.warn(f'manga settings ({origin}): {msg}', stacklevel=4)

    for top, v in _values.items():
        if top not in known_top:
            warn(f'unknown section [{top}]')
    for name, fn in _registry.items():
        params = inspect.signature(fn).parameters
        for k, v in section(name).items():
            if isinstance(v, dict) and f'{name}.{k}' in _registry:
                continue
            if k not in params:
                warn(f'[{name}] has no setting {k!r}; known: {", ".join(p for p in params if params[p].default is not inspect.Parameter.empty)}')
    for k in section('lines'):
        if k not in ('width_scale', 'taper_scale', 'wobble_scale'):
            warn(f'[lines] has no setting {k!r}; known: width_scale, taper_scale, wobble_scale')
    pen_fields = set(inspect.signature(Pen).parameters) - {'name'}
    for pen, over in section('pens').items():
        for k in (over if isinstance(over, dict) else {}):
            if k not in pen_fields:
                warn(f'[pens.{pen}] has no field {k!r}; known: {", ".join(sorted(pen_fields))}')


def files():
    return list(_files)


def values():
    import copy
    return copy.deepcopy(_values)
