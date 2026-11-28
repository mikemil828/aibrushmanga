"""User settings (manga/settings.*.toml): shipped defaults change nothing, overrides change only defaults."""
import inspect
import json
import sys
import warnings

import numpy as np
import pytest

from manga import Sheet, tone, hatch, settings, PENS
from manga.pens import BUILTIN_PENS, Pen
from manga.replay import replay

PPM = 8.0


def _read_defaults():
    return settings._read(settings.DEFAULT_FILE)


def test_default_file_repeats_the_engines_own_values():
    """The shipped file must match the code, so loading it changes no page."""
    d = _read_defaults()
    for name, fn in settings._registry.items():
        params = inspect.signature(fn).parameters
        sec = settings.section(name)
        assert sec, f'[{name}] missing from settings.default.toml'
        for k, v in sec.items():
            if isinstance(v, dict):
                continue
            assert k in params, f'[{name}] {k} is not a parameter'
            assert settings._tuples(v) == params[k].default, f'[{name}] {k}: file {v} != code {params[k].default}'
    assert set(d['pens']) == set(BUILTIN_PENS)
    for name, over in d['pens'].items():
        assert BUILTIN_PENS[name].with_(**over) == BUILTIN_PENS[name], name
    assert dict(PENS) == BUILTIN_PENS


def test_override_changes_defaults_but_call_arguments_win(tmp_path):
    def strokes(**kw):
        sh = Sheet(200, 200, px_per_mm=PPM, seed=1)
        return hatch(sh, 0.3, mask=np.ones((200, 200), bool), layers=((0.0, 0.0),), seed=4, **kw)

    n0 = strokes()
    f = tmp_path / 'style.toml'
    f.write_text('[hatch]\nspacing_mm = 0.35\n')
    settings.use(str(f))
    assert strokes() > 1.6 * n0, 'half the spacing should lay about twice the lines'
    assert strokes(spacing_mm=0.7) == n0, 'a value passed in the call beats the setting'
    settings.reset()
    assert strokes() == n0


def test_line_width_scale_and_pen_tables():
    settings.override('lines', width_scale=1.5)
    assert PENS['g_pen'].max_mm == pytest.approx(BUILTIN_PENS['g_pen'].max_mm * 1.5)
    settings.override('pens', g_pen={'max_mm': 1.0}, mine={'min_mm': 0.1, 'max_mm': 0.2})
    assert PENS['g_pen'].max_mm == pytest.approx(1.5)          # table first, then the scale
    assert isinstance(PENS['mine'], Pen) and PENS['mine'].max_mm == pytest.approx(0.3)
    sh = Sheet(100, 100, px_per_mm=PPM, seed=1)
    sh.stroke([(10, 50), (90, 50)], pen='mine', pressure=1.0)
    assert sh.ink.sum() > 0
    settings.reset()
    assert dict(PENS) == BUILTIN_PENS


def test_soft_shadow_fades_tone_at_its_edge_and_replays(tmp_path):
    m = np.zeros((240, 240), bool)
    m[40:200, 40:200] = True
    j = str(tmp_path / 'p.jsonl')
    hard = Sheet(240, 240, px_per_mm=PPM, seed=1)
    tone(hard, m, level=0.5)
    sh = Sheet(240, 240, px_per_mm=PPM, seed=1, journal=j)
    settings.override('tone', soft_mm=3.0)
    tone(sh, m, level=0.5)
    sh.close()
    edge, mid = (slice(40, 200), slice(40, 46)), (slice(100, 140), slice(100, 140))
    assert sh.ink[edge].mean() < 0.5 * hard.ink[edge].mean(), 'the edge band should be much lighter'
    assert sh.ink[mid].mean() == pytest.approx(hard.ink[mid].mean(), abs=0.02), 'the middle keeps its level'
    settings.reset()
    assert np.array_equal(replay(j).ink, sh.ink), 'the journal keeps soft_mm, whatever the settings are now'


def test_settings_flag_is_taken_out_of_argv(tmp_path, monkeypatch):
    f = tmp_path / 'style.toml'
    f.write_text('[effects.border]\nwidth_mm = 1.5\n')
    monkeypatch.setattr(sys, 'argv', ['page.py', '--settings', str(f), 'out.jsonl'])
    settings.load()
    assert sys.argv == ['page.py', 'out.jsonl']
    assert settings.section('effects.border')['width_mm'] == 1.5


def test_env_var_file_is_read(tmp_path, monkeypatch):
    f = tmp_path / 'style.toml'
    f.write_text('[tone]\nlpi = 60.0\n')
    monkeypatch.setenv(settings.ENV_VAR, str(f))
    settings.load(argv=False)
    assert settings.section('tone')['lpi'] == 60.0


def test_typos_warn():
    with pytest.warns(UserWarning, match='spacing'):
        settings.override('hatch', spaceing_mm=0.5)
    with pytest.warns(UserWarning, match='unknown section'):
        settings.override('hatchh', spacing_mm=0.5)


def test_darkness_setting_shapes_only_what_scripts_ask_for():
    from manga import prep
    rgb = np.random.default_rng(0).random((40, 40, 3)).astype(np.float32)
    lines0 = prep.xdog_lines(rgb)
    settings.override('prep.darkness', mode='value', gamma=2.0)
    d = prep.darkness(rgb)
    assert d.shape == (40, 40)
    assert np.array_equal(prep.xdog_lines(rgb), lines0), "the engine's own darkness reads stay as they were"
