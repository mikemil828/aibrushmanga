"""Read the Russian-named oil engine in English.

    python3 docs/english/ru2en.py нагрузка_для путь_с_нажимом      # translate names
    python3 docs/english/ru2en.py --file движок/кисть/холст.py      # print a module with English names
    python3 docs/english/ru2en.py --views docs/english/views        # write English-name views of every module
    python3 docs/english/ru2en.py --glossary docs/english/GLOSSARY.md

The views are a reading aid only: names are machine-translated word by word from the glossary, comments and
strings stay Russian, and the files are not meant to run. The real code is in движок/ and is unchanged.
"""
import argparse
import collections
import glob
import importlib.util
import io
import keyword
import os
import re
import sys
import tokenize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ENGINE = os.path.join(ROOT, 'движок')
CYR = re.compile('[А-Яа-яЁё]')


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dictionary():
    d = {}
    official = _load(os.path.join(ENGINE, 'кисть', 'словарь_справки.py'), 'slovar')
    for ru, en in official.ПАРЫ:
        if len(ru.split()) == 1:
            d.setdefault(ru.lower(), en.lower().replace(' ', '_').replace('-', '_'))
    extra = _load(os.path.join(HERE, 'words_extra.py'), 'words_extra')
    for ru, en in extra.WORDS.items():
        d[ru] = en
    return d


ENDINGS = sorted(['ами', 'ями', 'ого', 'его', 'ому', 'ему', 'ыми', 'ими', 'ых', 'их', 'ой', 'ей', 'ом', 'ем', 'ам',
                  'ям', 'ах', 'ях', 'ую', 'юю', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие', 'ый', 'ий', 'ов', 'ев', 'а', 'я',
                  'ы', 'и', 'у', 'ю', 'е', 'о', 'ь'], key=len, reverse=True)


def word(w, d):
    lw = w.lower()
    if not CYR.search(lw):
        return w
    if lw in d:
        return d[lw]
    m = re.match(r'^(.*?)(\d+)$', lw)
    if m and m.group(1) in d:
        return d[m.group(1)] + m.group(2)
    for e in ENDINGS:                      # crude stemming: try the dictionary form
        if lw.endswith(e) and len(lw) - len(e) >= 3:
            stem = lw[:-len(e)]
            for e2 in ('', 'а', 'о', 'ь', 'ы', 'й', 'ая', 'ый', 'ий', 'ое', 'ить', 'ать'):
                if stem + e2 in d:
                    return d[stem + e2]
    return translit(lw)


TR = dict(zip('абвгдеёжзийклмнопрстуфхцчшщъыьэюя',
              ['a', 'b', 'v', 'g', 'd', 'e', 'yo', 'zh', 'z', 'i', 'y', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't',
               'u', 'f', 'kh', 'ts', 'ch', 'sh', 'shch', '', 'y', '', 'e', 'yu', 'ya']))


def translit(w):
    t = ''.join(TR.get(c, c) for c in w)
    if len(w) <= 2:                    # short local names (з, п, кр...) read best as plain letters
        return t
    return '~' + t                     # ~ marks "not in the glossary, transliterated"


def name(n, d):
    """Translate one identifier: snake_case and CamelCase, keeping the shape (Python keywords get a trailing _)."""
    out = _name(n, d)
    return out + '_' if keyword.iskeyword(out.replace('~', '')) else out


def _name(n, d):
    lead = re.match(r'^_*', n).group(0)
    core = n[len(lead):]
    if '_' in core or core.islower() or not CYR.search(core):
        parts = core.split('_')
        out = '_'.join(word(p, d) if p else '' for p in parts)
        if core.isupper() and len(core) > 1:
            out = out.upper()
        return lead + out
    # CamelCase Cyrillic: ПустойМазок -> EmptyStroke
    parts = re.findall('[А-ЯЁA-Z][а-яёa-z0-9]*|[а-яёa-z0-9]+', core)
    if not parts:
        return lead + word(core, d)
    return lead + ''.join(word(p, d).replace('~', '').title().replace('_', '') for p in parts)


def view(path, d):
    src = open(path, encoding='utf-8').read()
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out = []
    for t in toks:
        if t.type == tokenize.NAME and CYR.search(t.string):
            out.append(t._replace(string=name(t.string, d).replace('~', '')))
        else:
            out.append(t)
    return tokenize.untokenize(out)


def engine_files():
    return sorted(glob.glob(os.path.join(ENGINE, 'кисть', '*.py'))) + sorted(
        glob.glob(os.path.join(ENGINE, 'примеры', '*.py')))


def all_names():
    cnt = collections.Counter()
    where = collections.defaultdict(set)
    for f in engine_files():
        for t in tokenize.generate_tokens(io.StringIO(open(f, encoding='utf-8').read()).readline):
            if t.type == tokenize.NAME and CYR.search(t.string):
                cnt[t.string] += 1
                where[t.string].add(os.path.relpath(f, ENGINE))
    return cnt, where


def glossary(path, d):
    cnt, where = all_names()
    rows = sorted(cnt, key=lambda n: (n.lstrip('_').lower(), n))
    miss = sum(1 for n in rows if '~' in name(n, d))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Oil engine glossary: every Russian name in the code\n\n')
        f.write('Generated by `python3 docs/english/ru2en.py --glossary docs/english/GLOSSARY.md`. English is a '
                'word-by-word reading aid, built from the engine\'s own handbook dictionary '
                '(`движок/кисть/словарь_справки.py`) plus `docs/english/words_extra.py`. A `~` marks a word that is '
                'not in the glossary and was transliterated. To improve a translation, add the word to '
                '`words_extra.py` and regenerate.\n\n')
        f.write(f'{len(rows)} names, {miss} with at least one transliterated word.\n\n')
        f.write('| Russian | English (reading aid) | Uses | Where |\n|-|-|-|-|\n')
        for n in rows:
            w = sorted(where[n])
            ws = ', '.join(x.replace('кисть/', '').replace('примеры/', 'ex/') for x in w[:3])
            if len(w) > 3:
                ws += f' +{len(w) - 3}'
            f.write(f'| `{n}` | `{name(n, d)}` | {cnt[n]} | {ws} |\n')
    return len(rows), miss


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*')
    ap.add_argument('--file')
    ap.add_argument('--views')
    ap.add_argument('--glossary')
    a = ap.parse_args()
    d = dictionary()
    for n in a.names:
        print(f'{n}  ->  {name(n, d)}')
    if a.file:
        sys.stdout.write(view(a.file, d))
    if a.views:
        for f in engine_files():
            rel = os.path.relpath(f, ENGINE)
            sub, base = os.path.split(rel)
            en_sub = {'кисть': 'brush', 'примеры': 'examples'}[sub]
            en_base = name(base[:-3], d).replace('~', 'x_') + '.py'
            dst = os.path.join(a.views, en_sub, en_base)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, 'w', encoding='utf-8') as fh:
                fh.write(f'# READING VIEW of движок/{rel} with names in English. Not runnable; edit the original.\n')
                fh.write(view(f, d))
        print('views written to', a.views)
    if a.glossary:
        n, miss = glossary(a.glossary, d)
        print(f'{a.glossary}: {n} names, {miss} with transliterated words')


if __name__ == '__main__':
    main()
