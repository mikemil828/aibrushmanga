# READING VIEW of движок/кисть/распаковка.py with names in English. Not runnable; edit the original.
"""С-1: чтение pickle с диска без выполнения чужого кода — для `вариант.pkl` (`варианты.прочесть`, М-8).

Обычный `pickle.loads` при чтении зовёт всё, что названо в файле: подложный файл выполняет любой код. `прочесть(байты)`
читает тот же поток, но пускает только то, что движок сам пишет в вариант:

- простые типы потока (числа, строки, байты, списки, кортежи, словари, множества) — они в pickle без имён;
- имя (GLOBAL) — только из списка `_имена()`: numpy (массив, dtype, скаляр) и классы-данные движка;
- вызов (REDUCE) — только из `_вызовы()`: сборка массива numpy, `Краска`, `Замес`, `complex`, `slice`, `range`…;
  класс с действием в конструкторе (`Журнал(путь)`, `Холст`…) в списке не стоит и не зовётся;
- новый объект (NEWOBJ) — только `cls.__new__` класса-данных движка, без `__init__`;
- состояние (BUILD) — только объекту разрешённого класса (не классу и не функции: иначе правился бы сам движок);
- INST, OBJ, EXT, PERSID и буферы вне потока — отказ.

Прочее — `Чужое` (ValueError) словами: какое имя. Свой класс в ответе куска читается обычным pickle по явному слову
(`х.взять(путь, своя=True)`). Чтец — питоновский `pickle._Unpickler`: у сишного `find_class` не знает, зачем имя взято
(значением, под вызов или под `__new__`), и разрешённый класс можно было бы позвать с чужими аргументами."""
import io
import pickle

import numpy as np


class Foreign(ValueError):
    """В pickle с диска — имя или действие, которых движок в вариант не пишет."""


_TABLAndTsY = None


def _tables():
    """(имена, вызовы, классы, массивы): что пускает чтец. Собирается при первом чтении — классы движка тянут модули."""
    global _TABLAndTsY
    if _TABLAndTsY is not None:
        return _TABLAndTsY
    import collections
    from .tool import Brush, Loading, _Paints
    from .paint import Paint
    from .monotony import Monotony
    from .response import Response
    from .palette import Mix
    from .tiles import Tiles
    from .path import Sweep
    from .canvas import Row
    try:
        from numpy._core import multiarray as M, numeric as H
        core = ('numpy._core', 'numpy.core')
    except ImportError:                              # numpy 1.x
        from numpy.core import multiarray as M, numeric as H
        core = ('numpy.core', 'numpy._core')
    # классы-данные движка: объект — через __new__ и состояние, конструктор не зовётся
    classes = (Brush, Loading, _Paints, Monotony, Response, Tiles)
    arrays = (np.ndarray, Sweep, Row)                 # собираются только через `_reconstruct`
    names = {(brush.__module__, brush.__qualname__): brush for brush in classes + arrays + (Paint, Mix)}
    names[('numpy', 'dtype')] = np.dtype
    for p in core:                                   # файл мог писать другой numpy: имя модуля ядра у 1.x и 2.x разное
        names[(p + '.multiarray', '_reconstruct')] = M._reconstruct
        names[(p + '.multiarray', 'scalar')] = M.scalar
        if hasattr(H, '_frombuffer'):
            names[(p + '.numeric', '_frombuffer')] = H._frombuffer
    for t in (complex, slice, range, float, set, frozenset, bytearray, type(None)):
        names[('builtins', t.__name__)] = t
    names[('collections', 'OrderedDict')] = collections.OrderedDict
    # что можно позвать (REDUCE): сборка массивов, краски (конструктор только проверяет и считает), простые типы;
    # float — `Нагрузка`, NoneType — метка учёта в `Ответ` (`подсказки._Учёт`)
    calls = {id(in_): in_ for in_ in (M._reconstruct, M.scalar, getattr(H, '_frombuffer', None), np.dtype, Paint, Mix,
                                 complex, slice, range, float, set, frozenset, bytearray, type(None),
                                 collections.OrderedDict) if in_ is not None}
    _TABLAndTsY = (names, calls, frozenset(classes), frozenset(arrays), M._reconstruct)
    return _TABLAndTsY


def _name(response):
    t = response if isinstance(response, type) else type(response)
    return f'{getattr(response, "__module__", None) or t.__module__}.{getattr(response, "__qualname__", None) or t.__qualname__}'


class _Reader(pickle._Unpickler):
    def find_class(self, module, name):
        try:
            return _tables()[0][(module, name)]
        except KeyError:
            raise Foreign(f'имя {module}.{name}') from None

    def persistent_load(self, pid):
        raise Foreign('ссылка вне файла (PERSID)')

    def _call(self):
        _, calls, _, arrays, collect = _tables()
        stack = self.stack
        arg, f = stack[-1], stack[-2]
        try:
            allowed = calls.get(id(f)) is f
        except TypeError:
            allowed = False
        if not allowed:
            raise Foreign(f'вызов {_name(f)}')
        if f is collect and not (isinstance(arg, tuple) and arg and type(arg[0]) is type and arg[0] in arrays):
            raise Foreign(f'массив класса {_name(arg[0]) if isinstance(arg, tuple) and arg else "?"}')
        pickle._Unpickler.load_reduce(self)

    def _new(self):
        brush = self.stack[-2]
        if not (isinstance(brush, type) and brush in _tables()[2]):
            raise Foreign(f'объект класса {_name(brush)}')
        pickle._Unpickler.load_newobj(self)

    def _new_with_names(self):
        brush = self.stack[-3]
        if not (isinstance(brush, type) and brush in _tables()[2]):
            raise Foreign(f'объект класса {_name(brush)}')
        pickle._Unpickler.load_newobj_ex(self)

    def _state(self):
        _, _, classes, arrays, _ = _tables()
        response = self.stack[-2]
        t = type(response)
        if isinstance(response, type) or not (t in classes or t in arrays or t is np.dtype or isinstance(response, np.dtype)):
            raise Foreign(f'состояние для {_name(response)}')
        pickle._Unpickler.load_build(self)

    def _forbidden(self):
        raise Foreign('старый вызов класса (INST, OBJ) или расширение (EXT)')

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.REDUCE[0]] = _call
    dispatch[pickle.NEWOBJ[0]] = _new
    dispatch[pickle.NEWOBJ_EX[0]] = _new_with_names
    dispatch[pickle.BUILD[0]] = _state
    for _code in (pickle.INST, pickle.OBJ, pickle.EXT1, pickle.EXT2, pickle.EXT4):
        dispatch[_code[0]] = _forbidden
    del _code


def read(bytes):
    """Объект из pickle `байты`, записанного движком в вариант. Чужое имя, вызов или действие — `Чужое` (сказано, что
    именно), до того, как оно выполнено; битый поток — `pickle.UnpicklingError`."""
    try:
        return _Reader(io.BytesIO(bytes)).load()
    except (Foreign, MemoryError, RecursionError):
        raise
    except Exception as e:                           # noqa: BLE001 — битый поток у питоновского чтеца падает чем угодно
        raise pickle.UnpicklingError(f'{type(e).__name__}: {e}') from None
