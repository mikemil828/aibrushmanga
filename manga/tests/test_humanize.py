"""Humanize: off changes nothing; on is deterministic, replays without the hand, and matches the journal patch."""
import json

import numpy as np

from manga import Sheet
from manga.humanize import rewrite_file
from manga.replay import replay


def draw(sh):
    for i in range(6):
        sh.stroke([(40, 40 + 30 * i), (200, 30 + 31 * i), (360, 45 + 30 * i)], pen='g_pen', pressure=0.7)
    with sh.part('eyes'):
        t = np.linspace(0, 2 * np.pi, 9)
        sh.stroke(np.c_[300 + 40 * np.cos(t), 250 + 15 * np.sin(t)], pen='maru', pressure=0.7)
    sh.stroke([(20, 280), (380, 280)], pen='fineliner')


def lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().splitlines()


def test_off_by_default_draws_exactly_as_before(tmp_path):
    a = Sheet(400, 300, px_per_mm=8, seed=1, journal=str(tmp_path / 'a.jsonl'))
    draw(a)
    a.close()
    assert not any('humanize' in l for l in lines(tmp_path / 'a.jsonl'))
    b = Sheet(400, 300, px_per_mm=8, seed=1)
    for i in range(6):
        b.stroke([(40, 40 + 30 * i), (200, 30 + 31 * i), (360, 45 + 30 * i)], pen='g_pen', pressure=0.7)
    t = np.linspace(0, 2 * np.pi, 9)
    b.stroke(np.c_[300 + 40 * np.cos(t), 250 + 15 * np.sin(t)], pen='maru', pressure=0.7)
    b.stroke([(20, 280), (380, 280)], pen='fineliner')
    assert np.array_equal(a.ink, b.ink), 'part() blocks must not change the drawing'


def test_live_hand_moves_lines_a_little_and_replays_without_it(tmp_path):
    j = str(tmp_path / 'h.jsonl')
    off = Sheet(400, 300, px_per_mm=8, seed=1)
    draw(off)
    on = Sheet(400, 300, px_per_mm=8, seed=1, journal=j)
    on.humanize(strength=0.3)
    draw(on)
    on.close()
    diff = np.abs(on.ink - off.ink)
    assert diff.max() > 0.5, 'the hand should move the lines'
    assert abs(on.ink.sum() - off.ink.sum()) < 0.15 * off.ink.sum(), 'about the same amount of ink'
    assert np.array_equal(on.ink[275:286], off.ink[275:286]), 'a fineliner (ruled) line stays put'
    again = replay(j)
    assert np.abs(again.ink - on.ink).max() < 1e-4


def test_patch_matches_live_strips_back_and_zero_is_identical(tmp_path):
    src, live, out = (str(tmp_path / n) for n in ('src.jsonl', 'live.jsonl', 'out.jsonl'))
    sh = Sheet(400, 300, px_per_mm=8, seed=1, journal=src)
    draw(sh)
    sh.close()
    lv = Sheet(400, 300, px_per_mm=8, seed=1, journal=live)
    lv.humanize(strength=0.2, who='patch')
    draw(lv)
    lv.close()
    rewrite_file(src, out, strength=0.2)
    assert lines(out) == lines(live), 'the journal patch draws what live drawing draws'
    rewrite_file(out, str(tmp_path / 'back.jsonl'), strength=0)
    assert lines(tmp_path / 'back.jsonl') == lines(src)
    rewrite_file(src, str(tmp_path / 'zero.jsonl'), strength=0)
    assert lines(tmp_path / 'zero.jsonl') == lines(src)


def test_eyes_get_a_gentler_hand(tmp_path):
    def moved(parts):
        src, out = str(tmp_path / 'e.jsonl'), str(tmp_path / 'eo.jsonl')
        sh = Sheet(400, 300, px_per_mm=8, seed=1, journal=src)
        draw(sh)
        sh.close()
        rewrite_file(src, out, strength=0.3, parts=parts)
        for l in lines(out):
            d = json.loads(l)
            if d.get('part') == 'eyes':
                a = np.asarray(d['humanize_from']['points'], float)
                b = np.asarray(d['points'], float)[:, :2]
                return float(np.min(np.hypot(*(b[:, None] - a[None]).transpose(2, 0, 1)), axis=1).max())
    assert moved({'eyes': 0.25}) < moved({'eyes': 1.0})
