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


class Чужое(ValueError):
    """В pickle с диска — имя или действие, которых движок в вариант не пишет."""


_ТАБЛИЦЫ = None


def _таблицы():
    """(имена, вызовы, классы, массивы): что пускает чтец. Собирается при первом чтении — классы движка тянут модули."""
    global _ТАБЛИЦЫ
    if _ТАБЛИЦЫ is not None:
        return _ТАБЛИЦЫ
    import collections
    from .инструмент import Кисть, Загрузка, _Краски
    from .краска import Краска
    from .однообразие import Однообразие
    from .ответ import Ответ
    from .палитра import Замес
    from .плитки import Плитки
    from .путь import Ход
    from .холст import Ряд
    try:
        from numpy._core import multiarray as М, numeric as Ч
        ядро = ('numpy._core', 'numpy.core')
    except ImportError:                              # numpy 1.x
        from numpy.core import multiarray as М, numeric as Ч
        ядро = ('numpy.core', 'numpy._core')
    # классы-данные движка: объект — через __new__ и состояние, конструктор не зовётся
    классы = (Кисть, Загрузка, _Краски, Однообразие, Ответ, Плитки)
    массивы = (np.ndarray, Ход, Ряд)                 # собираются только через `_reconstruct`
    имена = {(к.__module__, к.__qualname__): к for к in классы + массивы + (Краска, Замес)}
    имена[('numpy', 'dtype')] = np.dtype
    for п in ядро:                                   # файл мог писать другой numpy: имя модуля ядра у 1.x и 2.x разное
        имена[(п + '.multiarray', '_reconstruct')] = М._reconstruct
        имена[(п + '.multiarray', 'scalar')] = М.scalar
        if hasattr(Ч, '_frombuffer'):
            имена[(п + '.numeric', '_frombuffer')] = Ч._frombuffer
    for т in (complex, slice, range, float, set, frozenset, bytearray, type(None)):
        имена[('builtins', т.__name__)] = т
    имена[('collections', 'OrderedDict')] = collections.OrderedDict
    # что можно позвать (REDUCE): сборка массивов, краски (конструктор только проверяет и считает), простые типы;
    # float — `Нагрузка`, NoneType — метка учёта в `Ответ` (`подсказки._Учёт`)
    вызовы = {id(в): в for в in (М._reconstruct, М.scalar, getattr(Ч, '_frombuffer', None), np.dtype, Краска, Замес,
                                 complex, slice, range, float, set, frozenset, bytearray, type(None),
                                 collections.OrderedDict) if в is not None}
    _ТАБЛИЦЫ = (имена, вызовы, frozenset(классы), frozenset(массивы), М._reconstruct)
    return _ТАБЛИЦЫ


def _имя(о):
    т = о if isinstance(о, type) else type(о)
    return f'{getattr(о, "__module__", None) or т.__module__}.{getattr(о, "__qualname__", None) or т.__qualname__}'


class _Чтец(pickle._Unpickler):
    def find_class(self, module, name):
        try:
            return _таблицы()[0][(module, name)]
        except KeyError:
            raise Чужое(f'имя {module}.{name}') from None

    def persistent_load(self, pid):
        raise Чужое('ссылка вне файла (PERSID)')

    def _вызов(self):
        _, вызовы, _, массивы, собрать = _таблицы()
        стек = self.stack
        арг, ф = стек[-1], стек[-2]
        try:
            можно = вызовы.get(id(ф)) is ф
        except TypeError:
            можно = False
        if not можно:
            raise Чужое(f'вызов {_имя(ф)}')
        if ф is собрать and not (isinstance(арг, tuple) and арг and type(арг[0]) is type and арг[0] in массивы):
            raise Чужое(f'массив класса {_имя(арг[0]) if isinstance(арг, tuple) and арг else "?"}')
        pickle._Unpickler.load_reduce(self)

    def _новый(self):
        к = self.stack[-2]
        if not (isinstance(к, type) and к in _таблицы()[2]):
            raise Чужое(f'объект класса {_имя(к)}')
        pickle._Unpickler.load_newobj(self)

    def _новый_с_именами(self):
        к = self.stack[-3]
        if not (isinstance(к, type) and к in _таблицы()[2]):
            raise Чужое(f'объект класса {_имя(к)}')
        pickle._Unpickler.load_newobj_ex(self)

    def _состояние(self):
        _, _, классы, массивы, _ = _таблицы()
        о = self.stack[-2]
        т = type(о)
        if isinstance(о, type) or not (т in классы or т in массивы or т is np.dtype or isinstance(о, np.dtype)):
            raise Чужое(f'состояние для {_имя(о)}')
        pickle._Unpickler.load_build(self)

    def _нельзя(self):
        raise Чужое('старый вызов класса (INST, OBJ) или расширение (EXT)')

    dispatch = dict(pickle._Unpickler.dispatch)
    dispatch[pickle.REDUCE[0]] = _вызов
    dispatch[pickle.NEWOBJ[0]] = _новый
    dispatch[pickle.NEWOBJ_EX[0]] = _новый_с_именами
    dispatch[pickle.BUILD[0]] = _состояние
    for _код in (pickle.INST, pickle.OBJ, pickle.EXT1, pickle.EXT2, pickle.EXT4):
        dispatch[_код[0]] = _нельзя
    del _код


def прочесть(байты):
    """Объект из pickle `байты`, записанного движком в вариант. Чужое имя, вызов или действие — `Чужое` (сказано, что
    именно), до того, как оно выполнено; битый поток — `pickle.UnpicklingError`."""
    try:
        return _Чтец(io.BytesIO(байты)).load()
    except (Чужое, MemoryError, RecursionError):
        raise
    except Exception as e:                           # noqa: BLE001 — битый поток у питоновского чтеца падает чем угодно
        raise pickle.UnpicklingError(f'{type(e).__name__}: {e}') from None
