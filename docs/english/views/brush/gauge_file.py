# READING VIEW of движок/кисть/мерка_файл.py with names in English. Not runnable; edit the original.
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

_BrushResponseD = []
_MResponseDAtLAnd = ('манера', 'разбор', 'рука', 'скоро', 'глаз', 'мерка_файл')
_NOT_THAT = (OSError, ValueError, KeyError, TypeError, IndexError, EOFError, zipfile.BadZipFile, zlib.error)


def _code():
    """Хеш кода мерки: числа от прежнего кода не берутся."""
    if not _BrushResponseD:
        h = hashlib.sha256()
        for and_ in _MResponseDAtLAnd:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), and_ + '.py'), 'rb') as f:
                h.update(f.read())
        _BrushResponseD.append(h.hexdigest())
    return _BrushResponseD[0]


class File:
    """Место мерок одной картинки: `взять(что, доводы)` — записанное или None, `записать(что, доводы, данные)`."""

    def __init__(self, path):
        self.path = os.path.abspath(os.path.expanduser(str(path)))
        with open(self.path, 'rb') as f:
            self.hash = hashlib.sha256(f.read()).hexdigest()

    def name(self, what, args):
        option = hashlib.sha256(json.dumps([what, args], ensure_ascii=False).encode()).hexdigest()[:12]
        return f'{self.path}.{what}-{option}.npz'

    def take(self, what, args):
        try:
            with open(self.name(what, args), 'rb') as f, np.load(f, allow_pickle=False) as z:
                d = json.loads(str(z['опись']))
                if d['картинка'] != self.hash or d['код'] != _code():
                    return None
                return _back(d['данные'], z)
        except _NOT_THAT:
            return None

    def write(self, what, args, data):
        arrays, temp = {}, None
        inventory = json.dumps({'картинка': self.hash, 'код': _code(), 'доводы': args, 'данные': _kind(data, arrays)}, ensure_ascii=False)
        try:
            d, temp = tempfile.mkstemp('.npz', '.мерка-', os.path.dirname(self.path))
            with os.fdopen(d, 'wb') as f:
                np.savez_compressed(f, inventory=np.array(inventory), **arrays)
            os.chmod(temp, 0o644)
            os.replace(temp, self.name(what, args))
        except OSError:
            if temp is not None and os.path.exists(temp):
                os.unlink(temp)


def _kind(response, arrays):
    """Данные для JSON: тип, которого у JSON нет, — [имя типа, значение]; массивы — отдельно, под номером."""
    if isinstance(response, np.ndarray):
        name = f'м{len(arrays)}'
        lines = response.dtype == object
        arrays[name] = response.astype(str) if lines else _already(response)
        return ['массив', name, 'object' if lines else response.dtype.str]
    if isinstance(response, np.generic):
        return ['numpy', response.dtype.str, response.item()]
    if isinstance(response, dict):
        return ['словарь', [[_kind(k, arrays), _kind(v, arrays)] for k, v in response.items()]]
    if isinstance(response, (list, tuple)):
        return ['список' if isinstance(response, list) else 'кортеж', [_kind(v, arrays) for v in response]]
    if response is None or isinstance(response, (bool, int, float, str)):
        return response
    raise TypeError(f'мерка файлом: {type(response).__name__}')


def _already(a):
    """Целые — в самом узком типе, куда они входят."""
    if a.dtype == np.int64 and a.size:
        for t in (np.uint8, np.uint16, np.int32):
            if a.min() >= np.iinfo(t).min and a.max() <= np.iinfo(t).max:
                return a.astype(t)
    return a


def _back(response, z):
    if not isinstance(response, list):
        return response
    if response[0] == 'массив':
        return z[response[1]].astype(object if response[2] == 'object' else np.dtype(response[2]))
    if response[0] == 'numpy':
        return np.dtype(response[1]).type(response[2])
    if response[0] == 'словарь':
        return {_back(k, z): _back(v, z) for k, v in response[1]}
    if response[0] in ('список', 'кортеж'):
        v = [_back(v, z) for v in response[1]]
        return v if response[0] == 'список' else tuple(v)
    raise ValueError(response[0])
