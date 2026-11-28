"""Поиск по справочнику движка: python -m кисть.справка <слово> [слово …].

Печатает строки `СПРАВОЧНИК.md`, где есть все слова (без учёта регистра), с заголовком раздела над ними.
Без слов — список разделов."""
import os
import sys

СПРАВОЧНИК = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'СПРАВОЧНИК.md')


def найти(слова, путь=СПРАВОЧНИК):
    """Список (раздел, строка) со всеми словами; без слов — заголовки разделов."""
    слова = [с.lower() for с in слова]
    раздел, ответ = '', []
    with open(путь, encoding='utf-8') as ф:
        for строка in ф:
            строка = строка.rstrip('\n')
            if строка.startswith('#'):
                раздел = строка.lstrip('# ').strip()
                if not слова:
                    ответ.append((раздел, строка))
                    continue
            if слова and строка.strip() and all(с in строка.lower() for с in слова):
                ответ.append((раздел, строка))
    return ответ


def main(argv=None):
    слова = sys.argv[1:] if argv is None else argv
    найдено = найти(слова)
    if not найдено:
        print(f'нет в справочнике: {" ".join(слова)}')
        return 1
    был = None
    for раздел, строка in найдено:
        if слова and раздел != был:
            print(f'\n## {раздел}')
            был = раздел
        print(строка)
    return 0


if __name__ == '__main__':
    sys.exit(main())
