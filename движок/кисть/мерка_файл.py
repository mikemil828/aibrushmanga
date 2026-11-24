"""Мн-4: мерка образца — файлом рядом с картинкой: `<картинка>.мерка-<ключ>.npz`. Второй вызов `мерка(путь, …)` не считает.

Ключ имени — доводы мерки (px/мм, окно, область, предел, сторона); внутри — хеш файла картинки и хеш кода мерки: сменилось
что-то одно — мерка считается заново и файл переписывается. В файле — числа и массивы без pickle: чужой файл кода не
выполнит. Карты, что из снимка считаются за доли секунды ('картинка', 'тон в пятне'), не пишутся — считаются, когда их
спросят. Папка только для чтения — мерка считается без файла."""
import hashlib
import json
import os
import tempfile
import zipfile
import zlib

import numpy as np

_КОД = []
_МОДУЛИ = ('манера', 'разбор', 'рука', 'скоро', 'глаз', 'мерка_файл')
_НЕ_ТОТ = (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile, zlib.error)


def _код():
    """Хеш кода мерки: числа от прежнего кода не берутся."""
    if not _КОД:
        h = hashlib.sha256()
        for и in _МОДУЛИ:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), и + '.py'), 'rb') as f:
                h.update(f.read())
        _КОД.append(h.hexdigest())
    return _КОД[0]


class Файл:
    """Место мерок одной картинки: `взять(что, доводы)` — записанное или None, `записать(что, доводы, данные)`."""

    def __init__(self, путь):
        self.путь = os.path.abspath(os.path.expanduser(str(путь)))
        with open(self.путь, 'rb') as f:
            self.хеш = hashlib.sha256(f.read()).hexdigest()

    def имя(self, что, доводы):
        ключ = hashlib.sha256(json.dumps([что, доводы], ensure_ascii=False).encode()).hexdigest()[:12]
        return f'{self.путь}.{что}-{ключ}.npz'

    def взять(self, что, доводы):
        try:
            with open(self.имя(что, доводы), 'rb') as f, np.load(f, allow_pickle=False) as z:
                д = json.loads(str(z['опись']))
                if д['картинка'] != self.хеш or д['код'] != _код():
                    return None
                return _назад(д['данные'], z)
        except _НЕ_ТОТ:
            return None

    def записать(self, что, доводы, данные):
        массивы, врем = {}, None
        опись = json.dumps({'картинка': self.хеш, 'код': _код(), 'доводы': доводы, 'данные': _вид(данные, массивы)}, ensure_ascii=False)
        try:
            д, врем = tempfile.mkstemp('.npz', '.мерка-', os.path.dirname(self.путь))
            with os.fdopen(д, 'wb') as f:
                np.savez_compressed(f, опись=np.array(опись), **массивы)
            os.chmod(врем, 0o644)
            os.replace(врем, self.имя(что, доводы))
        except OSError:
            if врем is not None and os.path.exists(врем):
                os.unlink(врем)


def _вид(о, массивы):
    """Данные для JSON: тип, которого у JSON нет, — [имя типа, значение]; массивы — отдельно, под номером."""
    if isinstance(о, np.ndarray):
        имя = f'м{len(массивы)}'
        строки = о.dtype == object
        массивы[имя] = о.astype(str) if строки else _уже(о)
        return ['массив', имя, 'object' if строки else о.dtype.str]
    if isinstance(о, np.generic):
        return ['numpy', о.dtype.str, о.item()]
    if isinstance(о, dict):
        return ['словарь', [[_вид(k, массивы), _вид(v, массивы)] for k, v in о.items()]]
    if isinstance(о, (list, tuple)):
        return ['список' if isinstance(о, list) else 'кортеж', [_вид(v, массивы) for v in о]]
    if о is None or isinstance(о, (bool, int, float, str)):
        return о
    raise TypeError(f'мерка файлом: {type(о).__name__}')


def _уже(a):
    """Целые — в самом узком типе, куда они входят."""
    if a.dtype == np.int64 and a.size:
        for т in (np.uint8, np.uint16, np.int32):
            if a.min() >= np.iinfo(т).min and a.max() <= np.iinfo(т).max:
                return a.astype(т)
    return a


def _назад(о, z):
    if not isinstance(о, list):
        return о
    if о[0] == 'массив':
        return z[о[1]].astype(object if о[2] == 'object' else np.dtype(о[2]))
    if о[0] == 'numpy':
        return np.dtype(о[1]).type(о[2])
    if о[0] == 'словарь':
        return {_назад(k, z): _назад(v, z) for k, v in о[1]}
    if о[0] in ('список', 'кортеж'):
        v = [_назад(v, z) for v in о[1]]
        return v if о[0] == 'список' else tuple(v)
    raise ValueError(о[0])
