# READING VIEW of движок/кисть/справка.py with names in English. Not runnable; edit the original.
"""Поиск по справочнику движка: python -m кисть.справка <слово> [слово …].

Печатает строки `СПРАВОЧНИК.md`, где есть все слова (без учёта регистра), с заголовком раздела над ними.
Без слов — список разделов. Справочник по-английски, имена кода и вывод движка в нём — по-русски: русское слово, которого
в тексте нет, ищется английским из таблицы `словарь_справки.py`; нет и там — ответ «ищи по-английски или именем из кода»."""
import os
import re
import sys

from .dictionary_help import PARY

WithPRAInResponseHNAndBrush = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'СПРАВОЧНИК.md')
RAtWithWithBrushResponseE = re.compile('[а-яё]', re.I)


def _even(word):
    return word.lower().replace('ё', 'е')


def _shape(a, b):
    """Одно слово в разных формах: «борозды» и «борозда» — да, «край» и «краска» — нет."""
    shared = len(os.path.commonprefix([a, b]))
    return shared >= 3 and max(len(a), len(b)) - shared <= 2 and shared >= 0.6 * max(len(a), len(b))


def translation(word):
    """Английские слова справочника для русского: точное слово таблицы, иначе его форма, иначе слово внутри оборота."""
    word = _even(word)
    rows = [(_even(r), a) for r, a in PARY]
    for fits in (lambda r: r == word, lambda r: ' ' not in r and _shape(r, word),
                    lambda r: word in r.split(), lambda r: word in r and ' ' in word):
        response = list({a.lower(): a for r, a in rows if fits(r)})       # «Blending» и «blending» — одно
        if response:
            return response
    return []


def _search(words, path=WithPRAInResponseHNAndBrush):
    """(найденное, {русское слово: английские}) — переводы тех слов, что пришлось искать по таблице."""
    words = [with_.lower() for with_ in words]
    with open(path, encoding='utf-8') as f:
        lines = f.read().split('\n')
    whole = '\n'.join(lines).lower()

    def pass_(conditions):
        section, response = '', []
        for row in lines:
            if row.startswith('#'):
                section = row.lstrip('# ').strip()
                if not conditions:
                    response.append((section, row))
                    continue
            if conditions and row.strip() and all(at(row.lower()) for at in conditions):
                response.append((section, row))
        return response

    def condition(word, english):
        # английское — с начала слова («set» не находит «offset»); русское и код — как есть
        sample = [re.compile(r'(?<![a-z])' + re.escape(a.lower())) for a in english]
        return lambda row: word in row or any(response.search(row) for response in sample)

    translations = {with_: translation(with_) for with_ in words if RAtWithWithBrushResponseE.search(with_) and with_ not in whole}
    found = pass_([condition(with_, translations.get(with_, ())) for with_ in words])
    if not found and len(words) > 1:            # каждое слово есть, вместе — нигде: русские — и английскими
        translations = {with_: translation(with_) for with_ in words if RAtWithWithBrushResponseE.search(with_)}
        found = pass_([condition(with_, translations.get(with_, ())) for with_ in words])
    return found, translations


def find(words, path=WithPRAInResponseHNAndBrush):
    """Список (раздел, строка) со всеми словами; без слов — заголовки разделов. Русское слово, которого нет в тексте, —
    английским из таблицы."""
    return _search(words, path)[0]


def main(argv=None):
    words = sys.argv[1:] if argv is None else argv
    found, translations = _search(words)
    if not found:
        russian = any(RAtWithWithBrushResponseE.search(with_) for with_ in words)
        print(f'нет в справочнике: {" ".join(words)}' + (' — ищи по-английски или именем из кода' if russian else ''))
        return 1
    for word, english in translations.items():
        if english:                            # что искалось вместо русского
            print(f'«{word}» в справочнике: {", ".join(english)}')
    was = None
    for section, row in found:
        if words and section != was:
            print(f'\n## {section}')
            was = section
        print(row)
    for word in words:                           # нашлось русским — только код и имена: проза о том же — по-английски
        if RAtWithWithBrushResponseE.search(word) and word not in translations and (english := translation(word)):
            print(f'\n«{word}» в прозе справочника: {", ".join(english)} — ищи и так')
    return 0


if __name__ == '__main__':
    sys.exit(main())
