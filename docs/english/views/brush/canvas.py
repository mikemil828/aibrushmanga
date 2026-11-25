# READING VIEW of движок/кисть/холст.py with names in English. Not runnable; edit the original.
"""Холст: слои краски в линейном свете (float32), грунт, сохранение, PNG 16 бит."""
import copy
import hashlib
import json
import mmap
import os
import warnings
import struct
import weakref
import zipfile
import zlib
from collections.abc import Mapping

import numpy as np

from .fast import clamp

from . import light as _light
from .grain import Grain
from .soft import release_along
from .tool import Loading
from .paint import in_linear, from_linear, thin, check_thinner, Paint
from .numbers import _ZERNResponse, _TEN
from .tiles import Tiles, PLAndTBrushA, elementwise
from .mm import in_mm, RResponseDNResponseY
from .hints import Hint, Hints, enabled as _hints_enabled, InYBrushL as _WITHOUT_HINTS

_InYTERTAYa = (128, 128, 128)   # У-1: краска-заглушка мазка без цвета вытертой кистью (нагрузка 0): в след не идёт (`мягкая.делят`, `пусты`)

SCALE_THICKNESSES = 4.0     # толщина h, которой в PNG толщины отвечает белый (65535)
MM_ON_H = _TEN[0] / 8.0   # мм краски в единице толщины h — 0,4 (как `глаз.ММ_НА_H`: 3,2 px высоты на h при 8 px/мм)
SCALE_THICKNESSES_MM = SCALE_THICKNESSES * MM_ON_H     # М-10 (в): 1,6 мм — белый карты толщины, одна шкала на все холсты
MM_ON_LEVEL_HEIGHTS = 0.001   # М-10 (г): карта высот — 1 мкм на уровень 16 бит, до 65,5 мм
ZALAndInBrushA = 0.7          # насколько верхний слой сглаживает рельеф нижних под собой
_PResponseLResponseWithA = 256           # высота полосы при выводе: память не зависит от размера холста
_REGION_ENTIRELY = 2_000_000   # М-9: px, до которых картинка со светом на одном потоке считается одним куском
_REGION_FLOW = 1_000_000   # М-9: px, больше которых картинка со светом идёт полосами в потоках (`_полосы_картинки`)
_FLOWS_OUTPUT = 4     # М-9: потолок потоков вывода и сжатия (`Холст._потоков`)
_BAND_GRAIN = 64      # М-9: полоса запаса высоты зерна (`Холст._высота_зерна`), строк
def _shared(shape, than=0.0):
    """Массив float32 в общей памяти (анонимный mmap, полон нулями или `чем`): процессы счёта мазков (`ядра.py`) пишут
    в него же. Занимает память, как обычный: страницы берутся, когда в них пишут."""
    n = 4 * int(np.prod(shape))
    if n == 0:
        return np.zeros(shape, np.float32)
    a = np.frombuffer(mmap.mmap(-1, n), dtype=np.float32).reshape(shape)
    if than:
        a[...] = than
    return a


def _in_shared(a):
    if isinstance(a, Tiles):
        return a.shared
    while isinstance(a, np.ndarray):         # вид вида …: в конце цепочки — владелец памяти
        a = a.base
    if isinstance(a, memoryview):
        a = a.obj
    return isinstance(a, mmap.mmap)


def _in_shared(a):
    """Копия массива в общей памяти (сам он — если уже там)."""
    if isinstance(a, Tiles):
        return a if a.shared else a.own(shared=True)
    if a is None or _in_shared(a) or a.nbytes == 0 or a.dtype != np.float32:
        return a
    n = _shared(a.shape)
    n[...] = a
    return n


def _field_layer(name, by_nadobnosti=False, doc=None):
    """Массив слоя — свойством: пока мазки холста считаются на ядрах (`ядра.py`), чтение массива сперва досчитывает их
    (у слоя холста с очередью — `_ждать`); у прочих слоёв — просто массив."""
    slot = '_' + name

    def take(self):
        try:
            wait = self._wait
        except AttributeError:
            wait = None
        if wait is not None:
            wait()
        return getattr(self, slot, None) if by_nadobnosti else getattr(self, slot)

    def dat(self, a):
        setattr(self, slot, a)

    return property(take, dat, doc=doc)


def write_frame(canvas, r):
    """К-3: рамка мазка — в `х._рамки` по метке строки журнала (`х._метка`), если повтор их собирает; обрезана по
    холсту, пустая — None (мазок холста не тронул)."""
    frames = canvas.__dict__.get('_рамки')
    if frames is None:
        return
    if r is not None:
        r = (max(int(r[0]), 0), max(int(r[1]), 0), min(int(r[2]), canvas.full_width), min(int(r[3]), canvas.full_height))
        if r[2] <= r[0] or r[3] <= r[1]:
            r = None
    frames.setdefault(canvas.__dict__.get('_метка'), []).append(None if r is None else list(r))


class _Frame(Exception):
    """Мазок дошёл до своей рамки и остановлен: её просили без самого мазка (`Холст.прямоугольник_мазка`, `ядра.py`)."""


class _OnlyFrame(dict):
    """Словарь «почему пусто» мазка, который считается только до рамки: как мазок её назвал — стоп, до чтения и записи холста."""

    def __setitem__(self, option, v):
        if option == 'рамка':
            raise _Frame(v)
        super().__setitem__(option, v)


class _InRegion(dict):
    """2г: «почему пусто» мазка варианта (`варианты`): рамка мазка (что он прочтёт и запишет) — только в области куска."""

    def __init__(self, region, sh, in_):
        super().__init__()
        self.region, self.sh, self.in_ = region, sh, in_

    def __setitem__(self, option, v):
        if option == 'рамка':
            x0, y0, x1, y1 = self.region
            r = (max(v[0], 0), max(v[1], 0), min(v[2], self.sh), min(v[3], self.in_))
            if r[2] > r[0] and r[3] > r[1] and not (x0 <= r[0] and y0 <= r[1] and r[2] <= x1 and r[3] <= y1):
                raise ValueError(f'варианты: мазок куска читает и пишет {r} (след с полями: смешение, вал, сырое под ним) — '
                                 f'это шире области {self.region}; область — шире на поле мазка (`х.прямоугольник_мазка` '
                                 'с доводами мазка), или мазки — внутрь')
        super().__setitem__(option, v)


class _InRegionToFrames(_InRegion):
    """У-2: разведка `варианты` — рамка мазка сверяется с областью куска (вышла — ValueError), и мазок стоит: дальше
    рамки не считается, холст не читается и не пишется."""

    def __setitem__(self, option, v):
        super().__setitem__(option, v)
        if option == 'рамка':
            raise _Frame(v)


def _strength_shadows(shadow):
    """П. 20: None, False и 0 — тени нет; True — рабочая сила; число — оно."""
    if shadow is True:
        return float(_light.SHADOW_WORKING)
    shadow = float(shadow or 0.0)
    if not 0.0 <= shadow <= 1.0:
        raise ValueError('тень — сила от 0 до 1')
    return shadow


class Layer:
    """Один слой краски: цвет (В,Ш,3), покрытие, толщина h, влажность, схват — всё float32.

    влажность — сколько в плёнке сырого (0 — сухо: мазок ложится поверх и ничего не берёт);
    схват — насколько сырая плёнка схватилась (0 — свежая, 1 — подсохшая корка): по схватившемуся
    мазок ложится резко и лишь чуть цепляет лежащее."""
    # `_сыро` — 2б: у среза слоя в процессе — заранее данный ответ «есть ли в слое сырое»; `_ждать` — у слоя холста, чьи
    # мазки считаются на ядрах: зовётся перед чтением любого массива слоя и досчитывает мазки (`ядра.py`)
    __slots__ = ('_цвет', '_покрытие', '_толщина', '_влажность', '_схват', '_пропуск', '_текучая', '_сыро', '_ждать')
    _PResponseLYa = ('цвет', 'покрытие', 'толщина', 'влажность', 'схват')      # массивы есть всегда; `пропуск` и `текучая` — по надобности
    _InWithE = _PResponseLYa + ('пропуск', 'текучая')
    _PAtWithTResponse = {'пропуск': 1.0, 'текучая': 0.0}                          # чем полон массив по надобности, пока в него не писали

    color, coverage, thickness, moisture, degree_of_set = (_field_layer(p) for p in _PResponseLYa)
    flowing = _field_layer('текучая', True, """К4: (В,Ш) — след текучей кисти (поле мазка "level"): насколько её ворс доставал до холста в этом месте (0 — не
        было её); None, пока текучих ходов в слое не было. По нему сырые текучие ходы находят друг друга и смыкаются.""")
    transmittance = _field_layer('пропуск', True, """П. 10, инструмент 3: (В,Ш,3) — пропуск света не кроющей краской слоя (1 — её нет); None, пока такой краски в слое не было.""")

    def __init__(self, height, width, shared=False):
        """общая=True — массивы в общей памяти: их пишут и процессы счёта мазков (`ядра.py`)."""
        zero = lambda shape: Tiles(shape, shared=shared)      # 2в: поля — плитками (`плитки.py`), на ядрах — в общей памяти
        self.color = zero((height, width, 3))
        self.coverage = zero((height, width))
        self.thickness = zero((height, width))
        self.moisture = zero((height, width))
        self.degree_of_set = zero((height, width))

    def __getstate__(self):
        return {p: getattr(self, p) for p in self._InWithE}      # без `_ждать`: очередь мазков — этого процесса

    def __setstate__(self, d):
        for p, a in d.items():
            if a is not None or p in self._PResponseLYa:
                setattr(self, p, a)


def _write_png(path, height, width, channel, bands, text=None):
    """Свой писатель PNG 16 бит (Pillow не пишет 16-битный RGB/RGBA).
    полосы — генератор массивов uint16 (строк, ширина, каналов). текст — {ключ: значение} ASCII: куски tEXt (М-10: мм на
    уровень серого у карт толщины и высот); пиксели от них не меняются."""
    type = {1: 0, 2: 4, 3: 2, 4: 6}[channel]

    def piece(name, data):
        return (struct.pack('>I', len(data)) + name + data
                + struct.pack('>I', zlib.crc32(name + data) & 0xffffffff))

    bpp = 2 * channel
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(piece(b'IHDR', struct.pack('>IIBBBBB', width, height, 16, type, 0, 0, 0)))
        f.write(piece(b'gAMA', struct.pack('>I', 45455)))
        f.write(piece(b'sRGB', b'\x00'))
        for brush, v in (text or {}).items():
            f.write(piece(b'tEXt', str(brush).encode('latin-1') + b'\x00' + str(v).encode('latin-1')))
        compression = zlib.compressobj(4)
        for p in bands:
            p = np.ascontiguousarray(p, dtype=np.uint16).reshape(p.shape[0], width * channel)
            b = p.astype('>u2').view(np.uint8).reshape(p.shape[0], width * bpp)
            lines = np.empty((b.shape[0], b.shape[1] + 1), np.uint8)
            lines[:, 0] = 1                                   # фильтр Sub
            lines[:, 1:bpp + 1] = b[:, :bpp]
            lines[:, bpp + 1:] = b[:, bpp:] - b[:, :-bpp]
            d = compression.compress(lines.tobytes())
            if d:
                f.write(piece(b'IDAT', d))
        d = compression.flush()
        if d:
            f.write(piece(b'IDAT', d))
        f.write(piece(b'IEND', b''))


def _by_order(f, pieces, n):
    """М-9: f(*кусок) для каждого куска — в n потоках (numpy отпускает GIL), ответы — по порядку кусков; вперёд считается
    не больше n + 1 кусков (память — как у n + 1 полос). n = 1 — по одному в этом потоке, как прежде. Пока вызывающий
    сжимает готовую полосу, потоки считают следующие: файл тот же байт в байт — сжатие идёт по порядку, одним потоком."""
    if n <= 1:
        for brush in pieces:
            yield f(*brush)
        return
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n) as pool:
        queue = deque()
        for brush in pieces:
            queue.append(pool.submit(f, *brush))
            if len(queue) > n:
                yield queue.popleft().result()
        while queue:
            yield queue.popleft().result()


class _Compressed:
    """М-9: поле .npy, уже сжатое (`_сжать_npy`): куски deflate, crc32 и длина несжатого."""
    __slots__ = ('куски', 'crc', 'длина', '_z')

    def __init__(self):
        self.pieces, self.crc, self.length = [], 0, 0

    def write(self, data):
        m = memoryview(data)
        self.crc = zlib.crc32(m, self.crc)
        self.length += m.nbytes
        self.pieces.append(self._z.compress(m))
        return m.nbytes


def _compress_npy(a):
    """М-9: массив -> `_Сжатое`: .npy (`np.lib.format.write_array`) тем же deflate, что пишет zipfile в ZIP_DEFLATED
    (уровень по умолчанию, сырой поток) — в своём потоке, пока прочие поля пишутся. zlib отпускает GIL."""
    with_ = _Compressed()
    with_._z = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15)
    np.lib.format.write_array(with_, np.asanyarray(a), allow_pickle=False)
    with_.pieces.append(with_._z.flush())
    del with_._z
    return with_



class _Otdat:
    """Вместо сжимателя записи zipfile: при закрытии отдаёт готовые сжатые байты."""
    def __init__(self, pieces):
        self._pieces = pieces

    def compress(self, data):          # не зовётся: в запись пишется уже сжатое
        raise RuntimeError('М-9: запись поля уже сжата')

    def flush(self):
        brush, self._pieces = self._pieces, []
        return b''.join(brush)


def _fit_in_compressed(f, ready):
    """М-9: в открытую запись zipfile — уже сжатое поле. Внутренности записи у zipfile CPython не на виду: если они не те,
    что ждём (другая версия Python), — False, и поле пишется прежним путём (то же, только дольше)."""
    if not (isinstance(ready, _Compressed) and type(getattr(f, '_compressor', None)) is _TYPE_SZHATIYA
            and getattr(f, '_crc', None) == 0 and getattr(f, '_file_size', None) == 0 and getattr(f, '_compress_size', None) == 0):
        return False
    f._compressor, f._crc, f._file_size = _Otdat(ready.pieces), ready.crc, ready.length
    return True


_TYPE_SZHATIYA = type(zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -15))


def in_16(gamma01):
    return (clamp(gamma01, 0.0, 1.0) * 65535.0 + 0.5).astype(np.uint16)


class Snapshot:
    """Снимок холста для `Холст.откатить` (п. 11б): список слоёв, счётчик сидов, что на кистях, и прежние пиксели
    всех слоёв там, куда с тех пор писали. Пиксели копируются не сразу, а перед каждой записью (мазок, сушка, схват,
    `вписать`) — только тронутый прямоугольник: снимок всего холста ничего не стоит, пока по холсту не пишут."""
    __slots__ = ('холст', 'номер', 'имена', 'убранные', 'счётчик', 'сиды', 'кисти', 'сразу', 'записи', 'старшие', 'журнал',
                 'подсказки', 'замесы', 'ленты', '__weakref__')


class _Layers(Mapping):
    """`Холст.слои` — только чтение: правка мимо журнала молча разводила бы лист и повтор (1в)."""
    ResponseShAndBBrushA = ('Холст.слои — только чтение: убрать слой — х.убрать_слой(имя), порядок — х.порядок_слоёв([имена снизу '
              'вверх]) (оба пишут строку в журнал); посмотреть без слоя — х.взгляд(…, без=[имя]), х.виды(…, без=[имя])')
    __slots__ = ('_д',)

    def __init__(self, d):
        self._d = d

    def __getitem__(self, name):
        return self._d[name]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __repr__(self):
        return f'слои({list(self._d)})'

    def __setitem__(self, *_):
        raise TypeError(self.ResponseShAndBBrushA)

    __delitem__ = __setitem__

    def __getattr__(self, name):
        if name in ('pop', 'popitem', 'clear', 'update', 'setdefault'):
            raise AttributeError(self.ResponseShAndBBrushA)
        raise AttributeError(name)


class EmptyStroke(UserWarning):
    """Е-7: мазок не оставил следа — с причиной."""


class Row(np.ndarray):
    """М-2+М-4-2: ответ `х.взгляд` — тот же массив uint8 (вид, не копия) и поле `подсказки`: тексты подсказок, новых с
    прошлой выдачи (`Подсказки.новые`); пусто — новых нет или подсказки выключены. У ответа `х.сетка` — и `укладка` (М-6)."""
    hints = ()
    laying = None


class Sokhraneno:
    """М-2+М-4-2: ответ `х.сохранить` — путь, мазков в холсте (`счётчик`), `подсказки`: новые с прошлой
    выдачи и `сводка` (Св-1) — сводка этапа строкой, пустая — в этапе мазков нет или подсказки выключены."""
    __slots__ = ('путь', 'счётчик', 'подсказки', 'сводка')

    def __init__(self, path, counter, hints, summary=''):
        self.path, self.counter, self.hints, self.summary = path, int(counter), list(hints), summary

    def __repr__(self):
        return (f'Сохранено({os.fspath(self.path) if isinstance(self.path, (str, os.PathLike)) else self.path!r}, '
                f'мазков {self.counter}' + ''.join(f'\n  {t}' for t in self.hints) + ')')


class _Eye:
    """5б: вызов глаза или мерки — в счёт подсказок (С4); внутри него вывод холста целиком не считается."""
    __slots__ = ('х', 'имя')

    def __init__(self, canvas, name):
        self.canvas, self.name = canvas, name

    def __enter__(self):
        d = self.canvas.__dict__
        if not d.get('_в_глазу'):
            p = self.canvas._p()
            if p is not None:
                p.eye(self.name)
        d['_в_глазу'] = d.get('_в_глазу', 0) + 1

    def __exit__(self, *_):
        self.canvas.__dict__['_в_глазу'] -= 1


def _find_journal(writing, path_npz):
    """М-10 (а): журнал, названный в `<.npz>.путь_журнала` — рядом с файлом (тем же относительным путём, папку можно переносить), иначе
    по полному пути. Нет ни того, ни другого — None (ошибка — словами при чтении, `Холст._путь_журнала`)."""
    if not isinstance(path_npz, (str, os.PathLike)):
        return None
    beside = os.path.join(os.path.dirname(os.path.abspath(path_npz)), writing.get('рядом', ''))
    for p in (beside, writing.get('путь')):
        if p and os.path.isfile(p):
            return os.path.normpath(p)
    return None


def _to_sokhraneniya(path_journal, path_npz, counter):
    """М-10 (а): байт конца строки {"saved": <этот файл>, "counter": счётчик} в журнале, если после неё в журнале есть
    ещё строки (лист писался дальше этого сохранения) — читать его надо до неё; иначе None (журнал как есть)."""
    name = os.path.basename(os.fspath(path_npz))
    end, place = None, 0
    with open(path_journal, 'rb') as f:
        for s in f:
            place += len(s)
            if s.startswith(b'{"saved"') and b'"counter"' in s:
                try:
                    z = json.loads(s)
                except ValueError:
                    continue
                if z.get('saved') == name and int(z.get('counter', -1)) == counter:
                    end = place
    return None if end is None or end >= place else end


class Canvas:
    measurement = True     # 3г (К-1): мазок отвечает замером следа (`ответ.Ответ`); False — только прямоугольником (`Холст(…, замер=)`)

    def __init__(self, width, height, ground=(150, 141, 126), px_on_mm=None, grain=None, journal=None, seed=0, cores=None, measurement=True,
                 hints=None):
        """ядра (Е-1) — сколько процессов считают мазки этого холста: мазки, чьи прямоугольники не пересекаются, идут разом,
        итог (все поля слоёв, кисти, журнал) — бит в бит как на одном; None — `KIST_CORES` или ядра машины, не больше 16;
        1 — всё в этом процессе, как прежде. Мазок на ядрах досчитывается позже вызова: `мазок` отвечает отложенным
        ответом, а любое чтение холста, слоя или кисти сперва досчитывает мазки (`досчитать`, `кисть/ядра.py`).
        журнал (А-1) — путь файла .jsonl: каждый мазок, снятие, сушка, схват, вытирание и набор кисти пишут в него
        строку сами, в том виде, в каком её читает `кисть.повтор` (`кисть/журнал.py`); первая строка — холст. Файл
        пишется заново. Откат снимка обрезает и журнал. None — журнала нет.
        px_на_мм — масштаб холста (п. 9): нужен, если ширина или шаг кисти заданы в миллиметрах
        (`Кисть(..., ширина_мм=, шаг_мм=)`); None — кисть только в пикселях, как раньше.
        зерно (п. 5б) — рельеф грунта, переплетение нитей: (шаг нити в мм, глубина в толщинах плёнки[, сид]) — нужен
        `px_на_мм`; None — грунт гладкий, всё как раньше.
        сид (А-9) — сид листа: мазок без своего сида берёт сид из листа, этапа и объекта (`пометка`) и своих точек, а
        не из номера мазка — вставка или удаление мазка не меняет сиды других (`сид_мазка`).
        замер (3г, К-1) — мазок отвечает замером следа (`ответ.Ответ`: цвет, закрыто, просветы, ширина, сырое, разрывы,
        край); False (или `х.замер = False` на время пачки) — только прямоугольник, как раньше: счёт на 5–10 % быстрее.
        На пиксели, кисть и журнал не влияет.
        подсказки (5б) — подсказки по ходу: строка `подсказка: <факт> — под рукой: <инструменты>` в stderr (`Подсказка`)
        на границе (пометка, сохранить, закрыть, конец процесса); None — по окружению `KIST_HINTS` (0 — нет), иначе
        включены; False — нет. Память — `<журнал>.подсказки.json`. На пиксели, кисть и журнал не влияет (`х.подсказки`)."""
        self.measurement = bool(measurement)
        self.width = int(width)
        self.height = int(height)
        self.ground_srgb = tuple(float(v) for v in ground)
        self.ground = in_linear(ground).astype(np.float32)
        self._layers = {}           # имя -> Слой, по порядку создания: первый — нижний; снаружи — `слои`, только чтение
        self._counter = 0
        self.seed = int(seed)
        self._stage = self._object = None   # от последней `пометка`: сид мазка без своего (А-9)
        self._repeats = {}        # А-9: сколько мазков с теми же этапом, объектом и точками уже легло; None — счёта нет
                                  # (повтор журнала без "seeds": "tree": сиды от номера строки, `.npz` — прежний)
        self.px_on_mm = None if px_on_mm is None else float(px_on_mm)
        # участок (п. 9): массивы — прямоугольник холста с углом (x0, y0); координаты везде — холста целиком
        self.x0 = self.y0 = 0
        self.full_width, self.full_height = self.width, self.height
        self._snapshots = []         # п. 11б: живые снимки (слабые ссылки) — им копия пикселей перед каждой записью
        self.grain = None         # п. 5б: (шаг нити, мм; глубина в толщинах плёнки; сид) или None
        self._grain = None
        if grain is not None:
            if self.px_on_mm is None:
                raise ValueError('зерно: шаг нити задан в мм — холсту нужен px_на_мм')
            spacing, depth, seed = (tuple(grain) + (0,))[:3]
            self.grain = (float(spacing), float(depth), int(seed))
            self._grain = Grain(float(spacing) * self.px_on_mm, float(depth) * _ZERNResponse[0], int(seed))
        self.journal = None
        self._outside_journal = False  # копия холста в другом процессе: её мазки в журнал не попадают
        self._cores = None if cores is None else int(cores)   # 2б: None — решится первым мазком (`ядра.число_ядер`)
        self._cores = None          # очередь мазков на ядрах (`ядра.Ядра`); заводится первым мазком
        if self._cores is not None and self._cores < 1:
            raise ValueError('ядра: число процессов от 1')
        if journal is not None:
            from .journal import Journal
            self.journal = Journal(journal, self)
        self._hints = Hints(self, _hints_enabled(hints), journal)

    def _writes(self):
        zh = getattr(self, 'журнал', None)
        return zh if zh is not None and not zh.pause else None

    @property
    def hints(self):
        """5б: подсказки по ходу (`кисть/подсказки.py`) — что сказано и что ждёт: `print(х.подсказки)`; выключить —
        `х.подсказки.выключить()` или один вид `х.подсказки.выключить('серия')`."""
        p = self.__dict__.get('_подсказки')
        return _WITHOUT_HINTS if p is None else p

    def _p(self):
        """Подсказки, если они слушают этот холст сейчас (не участок, не проба, не повтор), иначе None."""
        p = self.__dict__.get('_подсказки')
        return None if p is None or p.quiet(self) else p

    def _taken(self, name):
        """Св-1: позван инструмент, что стоит «под рукой» у подсказок (глаза считает `_Глаз`)."""
        p = self._p()
        if p is not None and not self.__dict__.get('_в_глазу'):
            p.taken(name)

    def summary(self, stage=None):
        """Св-1: сводка этапа строками (не больше 12): мазков и прочитанных ответов, рваные, просветы, тон следа от названного,
        невидимые под верхним слоем, без следа, самое однообразное место рамкой, подсказки, чьих инструментов не звали.
        этап — имя; None — этап, что пишется сейчас. Ответ — строка с `.числа`; пустая — мазков в этапе нет или подсказки
        выключены. Сама печатается на `сохранить` и смене этапа пометкой. Только читает."""
        from .summary import Summary
        self._boundary()
        p = self._p()
        if p is None:
            return Summary()
        return p.summary(p.stage_summary(self) if stage is None else stage)

    def _eye(self, name):
        return _Eye(self, name)

    def mark(self, text='', stage=None, object=None, intent=None, **more):
        """А-6: строка-пометка журнала {"mark": текст, "stage", "object", "intent", …}: начало этапа или объекта. Этап и
        объект пометки идут полями "stage" и "object" во все строки после неё, пока их не сменит следующая пометка
        (у мазка, снятия и заметки можно дать свои). ещё — любые поля JSON (замысел, мерка). Повтор строку не исполняет."""
        was = (self._stage, self._object)
        if stage is not None:
            self._stage = stage
        if object is not None:
            self._object = object
        p = self._p()
        if p is not None and (p._first or was != (self._stage, self._object)):
            if was[0] != self._stage:
                self._boundary()                        # Св-1: сводка прежнего этапа — по досчитанным мазкам
            p.mark(self, text, stage, object, was[0])       # 5б: смена этапа или объекта — граница подсказок
        zh = self._writes()
        if zh is None:
            return
        if stage is not None:
            zh.stage = stage
        if object is not None:
            zh.object = object
        from .journal import _in_json
        d = zh._intent({'mark': str(text)}, intent=intent)
        for brush, v in more.items():
            if brush in d:
                raise ValueError(f'пометка: поле {brush} уже есть')
            d[brush] = _in_json(v)
        zh.row(d)

    def note(self, what, stage=None, object=None):
        """А-6, К-11: строка {"note": что} — что угодно JSON: отвергнутые варианты, замеры до и после, что увидел на
        кропе и что поменял. Повтор её пропускает; `кисть.повтор --list` показывает."""
        zh = self._writes()
        if zh is not None:
            from .journal import _in_json
            zh.row(zh._intent({'note': _in_json(what)}, stage, object))

    # ------------------------------------------------------------------ ядра
    def _queue(self):
        """Очередь мазков на ядрах (`ядра.Ядра`) или None — холст считает сам: ядро одно, холст — участок или копия в
        процессе-потомке. Заводится первым мазком."""
        ya = self.__dict__.get('_ядра')
        if ya is not None and ya.canvas is not self:        # холст собран копией полей другого (`__dict__.update`): очередь — не его
            ya = self._cores = None
        if ya is not None:
            if ya.pid == os.getpid():
                return ya
            self._in_child()
            return None
        n = self.__dict__.get('_ядер', 1)
        if n == 1:
            return None
        from . import cores as Ya
        n = Ya.number_cores(n) if Ya.allowed() else 1
        self._cores = n
        if n == 1:
            return None
        self._cores = Ya.Cores(self, n)
        return self._cores

    def _in_child(self):
        """Холст с очередью оказался в процессе-потомке (fork): очередь — родителя, слои — в общей с ним памяти. Перед
        записью потомок берёт слои себе (копией) и считает сам, на одном ядре: родителю его мазки не видны, как и прежде."""
        ya = self._cores
        if ya._waiting:
            raise RuntimeError('холст в процессе-потомке: при ветвлении у родителя были недосчитанные мазки — до '
                               'ветвления звать х.досчитать()')
        self._cores, self._cores = None, 1
        for with_ in self._layers.values():
            with_._wait = None
            for field in Layer._InWithE:
                a = getattr(with_, '_' + field, None)
                if _in_shared(a):
                    setattr(with_, '_' + field, a.own() if isinstance(a, Tiles) else np.array(a))

    def _boundary(self, writing=False):
        """Перед тем, что читает или меняет холст мимо мазка: мазки очереди досчитаны. пишу=True — и в процессе-потомке
        слои свои (`_в_потомке`)."""
        ya = self.__dict__.get('_ядра')
        if ya is None or ya.canvas is not self:
            return
        if ya.pid == os.getpid():
            ya.boundary()
        elif writing or ya._waiting:
            self._in_child()

    def finish_computing(self):
        """Е-1: досчитать мазки, что ещё считаются на ядрах. Холст, слои и кисти делают это сами перед любым чтением
        (`картинка`, `х.слои[…].толщина`, `проба`, `сохранить`, сушка, `кисть.копия()`…); звать самому — перед замером
        времени, перед ветвлением процесса и если держишь массив слоя, взятый до мазков (он досчитанного не ждёт)."""
        self._boundary()

    def close(self):
        """Досчитать мазки и остановить процессы счёта (заведутся снова со следующим мазком); журнал — закрыть."""
        ya = self.__dict__.get('_ядра')
        if ya is not None and ya.canvas is self and ya.pid == os.getpid():
            ya.close()
        p = self._p()
        if p is not None:
            p.relief(self)                       # М-2: холст досчитан
            p.boundary(self)
        if getattr(self, 'журнал', None) is not None:
            self.journal.close()

    def __del__(self):
        ya = self.__dict__.get('_ядра')
        if ya is not None and ya.canvas is self and ya.pid == os.getpid():
            try:
                ya._stop()
            except Exception:       # noqa: BLE001 — интерпретатор закрывается
                pass

    # ------------------------------------------------------------------ слои
    def _layer(self, name):
        if name not in self._layers:
            self._layers[name] = Layer(self.height, self.width, shared=self.__dict__.get('_ядер', 1) != 1)
        return self._layers[name]

    def layer(self, name):
        self._boundary(writing=True)
        return self._layer(name)

    def memory(self):
        """Сколько памяти занимают поля каждого слоя на деле, байт (имя -> байт): у слоя плитками — только
        тронутые плитки (ровная плитка — одно число); так же и у слоя холста на ядрах (плитки в общей памяти)."""
        self._boundary()
        return {name: sum(getattr(a, 'байт', a.nbytes if isinstance(a, np.ndarray) else 0)
                         for a in (getattr(with_, p) for p in Layer._InWithE) if a is not None)
                for name, with_ in self._layers.items()}

    @property
    def layers(self):
        """Слои снизу вверх (имя -> Слой), только чтение: убрать слой — `убрать_слой`, порядок — `порядок_слоёв` (оба
        пишут строку в журнал; правка мимо них журнал не видел бы, и повтор молча разошёлся бы)."""
        return _Layers(self._layers)

    @layers.setter
    def layers(self, v):
        raise AttributeError(_Layers.ResponseShAndBBrushA)

    def remove_layer(self, name):
        """Убрать слой со всей его краской; в журнале — строка {"layer_drop": имя}. Откат снимка, взятого до этого,
        вернёт слой на место с его пикселями (журнал тоже вернётся). Посмотреть без слоя, ничего не меняя, —
        `х.взгляд(…, без=['лес'])`, `х.виды(…, без=['лес'])`."""
        if name not in self._layers:
            raise KeyError(f'убрать_слой: слоя {name!r} нет (есть: {list(self._layers)})')
        if (self.x0, self.y0, self.width, self.height) != (0, 0, self.full_width, self.full_height):
            raise ValueError('убрать_слой: на участке слой не убрать — `вписать` его не уберёт; убирать на холсте')
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        self._boundary(writing=True)
        zh = self._writes()
        if zh is not None:
            zh.row(zh._intent({'layer_drop': str(name)}))
        for r in getattr(self, '_снимки', []):     # снимку, при котором слой был, — сам слой: после этого в него не пишут
            with_ = r()
            if with_ is not None and with_.canvas is self and name in with_.names and name not in with_.removed:
                with_.removed[name] = self._layers[name]
        del self._layers[name]

    def order_layers(self, names):
        """Новый порядок слоёв снизу вверх — все имена, каждое раз; в журнале — строка {"layer_order": имена}."""
        self._taken('порядок_слоёв')
        names = [str(and_) for and_ in names]
        if sorted(names) != sorted(self._layers) or len(set(names)) != len(names):
            raise ValueError(f'порядок_слоёв: нужны все слои по разу — есть {list(self._layers)}, дано {names}')
        self._boundary(writing=True)
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        zh = self._writes()
        if zh is not None:
            zh.row(zh._intent({'layer_order': names}))
        self._layers = {and_: self._layers[and_] for and_ in names}

    def stroke(self, points, color=None, brush=None, layer='основа', seed=None, opacity=1.0,
              layout='полосы', load=None, mix_below=False, touch=None, bands=None, by_ridges=None, even=None, reserve=None, polyline=False, load_along=None, fibers=None, pressure_hand=False,
              stage=None, object=None, intent=None, thinner=None):
        """Положить мазок.

        этап, объект, намерение (А-6) — в журнал полями "stage", "object", "intent" (этап и объект без них — от
        последней `пометка`); на пиксели не влияют. Мазок, не оставивший следа, — предупреждение `ПустойМазок` с причиной.

        точки — (x, y, нажим 0–1, dt_мс[, наклон_°, азимут_рад, поворот_°]); можно только (x, y) —
        тогда нажим «вход — тело — сход» строится сам.
        цвет — sRGB 0..255: одна краска (r, g, b) или несколько [(rgb, доля), ...]; None — краска,
        набранная кистью (`кисть.набрать`). раскладка красок по кисти: 'полосы', 'смесь', 'кончик'.
        нагрузка — 'протирка', 'обычная', 'пастозная' или число; None — как у кисти.
        мешать_ниже — мешать и с сырой краской слоёв под этим (по умолчанию только свой слой).
        полосы (п. 13) — набор нескольких густых красок кусками: (ширина полосы, длина куска краски в ней), px холста;
        любое из двух — None (как было: по полосе на краску, полоса на весь ход).
        ровно (п. 15) — 0–1: текучая кроющая краска за ворсом смыкается в ровный слой (плоское тело по сырому); None — нет.
        запас (п. 17) — на сколько ширин кисти хватает краски: ход начинается плотно и рвётся островами (сухой срыв);
        None — как у кисти (полная: на ходе не кончается).
        волокна (К6) — 0–1, только с нагрузкой 'сухая': ворс полупустой кисти расчёсан — тело хода плотное, край рвётся
        волокнами вдоль хода (в журнале "fibers"); None — как было; с другой нагрузкой — ValueError.
        нажим_руки (К8.1) — нажим точек ведёт рука (в журнале — при "pressure_along"): плоская кисть плашмя касается
        шириной по самому нажиму — слабо поставленная садится серединой обоймы и раскрывается до всей ширины к нажиму
        0,55; False — как было: кисть плашмя ложится всей шириной и при лёгкой руке.
        касание — как поставлена плоская кисть: 1 — плашмя (вся ширина и при лёгком нажиме), 0 — на ребре:
        узкий след, ширина растёт с нажимом; None — как у кисти (плашмя).
        ломаная (К2) — точки — сырой путь руки: ведётся по самой ломаной, угол остаётся углом, нажим и углы между
        точками по прямой, входа и схода нет (`кисть.ломаная`); нужен нажим в точках. В журнале — "points".
        разбавитель (Г-1) — 0–1, в журнале "medium": краска с маслом и растворителем — ложится тоньше, прозрачнее, течёт
        (борозды смыкаются, как с `ровно`, и у не кроющей краски; дыры по зерну холста затягиваются), сухой срыв (`запас`)
        тянется дольше; густая (None, 0) держит рельеф. С нагрузкой 'сухая' и 0 — ValueError.
        Тот же сид и та же история кисти — те же пиксели."""
        color = self._by_palette(color)          # М-1: имя замеса — его цвет (`х.палитра`)
        z, zh = self._accept_stroke(points, color, brush, layer, seed, opacity, layout, load, mix_below, touch, bands,
                                   by_ridges, even, reserve, polyline, load_along, fibers, pressure_hand, stage, object, intent,
                                   thinner)
        if self.__dict__.get('_мимо'):          # К-3: пересборка затронутого — мазок вне него только учтён (сид, счётчик, слой)
            return None
        ya = self._queue()
        if ya is not None:
            return ya.stroke(z, brush, zh)
        return self._finish_computing_stroke(z, brush, zh)

    # ------------------------------------------------------------------ палитра (М-1)
    @property
    def palette(self):
        """М-1: палитра замесов холста (`кисть.палитра.Палитра`): `х.палитра({имя: цвет})`, `х.палитра.добавить(имя,
        цвет)`, `х.палитра['имя']`; мазок именем — `цвет='имя'` или списком имён (набор)."""
        self._taken('палитра')
        from .palette import Palette
        return Palette(self)

    def charge(self, brush, color, amount=None, layout='полосы'):
        """Д-1: кисть набирает краску — цвет (r, g, b), набор [(rgb, доля), …] или имя замеса палитры (список имён);
        количество — доля полной кисти (0–1]: краска живая, ход без цвета тратит её (`Кисть.набрать`); None — кисть полна
        на каждом ходе. Ответ — кисть."""
        return self.palette.charge(brush, color, amount, layout)

    def _by_palette(self, color):
        if color is None or not isinstance(color, (str, list, tuple)):
            return color
        from .palette import resolve, NoMix, from_file
        try:
            return resolve(color, self.__dict__.get('_замесы', {}))
        except NoMix as e:
            raise from_file(e, self) from None

    def palette_with(self, region=None, reference=None, n=12, path=None, light=False):
        """М-1: n замесов по образцу в области холста (образец — как цель глаз: картинка в координатах холста, путь,
        холст; `уложить`) — черновик `Черновик` «имя → (r, g, b)» с долями площади и замощением образца замесами;
        ничего не заводит и не кладёт: `х.палитра(ч)` или своей правкой. путь — PNG «образец | замощение | плашки»."""
        self._taken('палитра_с')
        from .eye import _frame, _target
        from .palette import palette_with
        if reference is None:
            raise ValueError('палитра_с: нужен образец (картинка, путь, холст) — в координатах холста, как цель глаз')
        frame = _frame(self, region)
        return palette_with(_target(reference, frame, light), n, path)

    @staticmethod
    def _parse_stroke(points, color, brush, layout, load, polyline, fibers):
        """Доводы `мазок` -> (точки, краски, раскладка): ломаная — в точки ядра, цвет — в краски; холст не меняется."""
        from .brush import parse_color, polyline as _polyline
        if polyline:
            points = _polyline(points)
        if brush is None:
            raise ValueError('мазок: не задана кисть')
        if fibers and ((brush.load if load is None else load) != 'сухая' or brush.shape == 'мастихин'):
            raise ValueError('мазок: волокна ("fibers") — только с нагрузкой \'сухая\' (полупустая кисть, не мастихин)')
        if color is None:
            if brush._paint is None:
                if brush.thickness_film(brush.load if load is None else load) > 0.0:
                    raise ValueError('мазок без цвета: сначала кисть.набрать(цвет) (без цвета — только вытертая кисть, нагрузка=0)')
                paints = parse_color(_InYTERTAYa, in_linear)   # У-1: у вытертой кисти (нагрузка 0) своей краски нет — цвет не нужен и в след не идёт
            else:
                paints, layout = brush._paint
        else:
            paints = parse_color(color, in_linear)
        return points, paints, layout

    def _accept_stroke(self, points, color, brush, layer, seed, opacity, layout, load, mix_below, touch, bands,
                       by_ridges, even, reserve, polyline, load_along, fibers, pressure_hand, stage, object, intent,
                       thinner=None):
        """Всё, что `мазок` делает с холстом и журналом до счёта, в порядке вызовов: сид, счёт повторов сида, счётчик,
        слой, строки журнала (кисть и мазок). Ответ: (запись мазка для `_положить` — её же берёт процесс счёта, журнал
        или None)."""
        polyline_given = bool(polyline)
        given = points
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        points, paints, layout = self._parse_stroke(points, color, brush, layout, load, polyline, fibers)
        thinner = check_thinner(thinner, brush.load if load is None else load)
        seed_sheet = self.seed_stroke(points, stage, object, take=True)   # и мазок со своим сидом в счёт повторов идёт
        if seed is None:
            seed = seed_sheet
        if thinner:
            self._taken('разбавитель')
        zh = self._writes()
        if zh is not None:
            name_brushes = zh.brush(brush)
        elif getattr(self, '_вне_журнала', False):
            self._outside_journal = 'мазки'
        self._counter += 1
        new_layer = layer not in self._layers
        self._layer(layer)
        p = self._p()
        if p is not None:
            from .palette import names_colors
            charge = brush._charge
            ts = color if color is not None else (charge[0] if charge is not None and charge[2] is brush._paint else None)
            mix = names_colors(ts) is not None
        label = None if p is None else p.stroke_accepted(self, points, brush, load, layer, self._object if object is None else object,
                                                       new_layer, paints, name_brushes if zh is not None else None, mix,
                                                       self._stage if stage is None else stage)
        if (brush.width_mm is not None or brush.spacing_mm is not None) and self.px_on_mm is None:
            raise ValueError('кисть в мм: у холста не задан px_на_мм')
        if zh is not None:
            from .journal import color_lines, _in_json
            d = {'brush': name_brushes, 'layer': str(layer)}
            d.update(color_lines(color))
            sweep = None if polyline_given or not hasattr(given, 'строка') else given.row()
            if sweep is None:
                d['core'] = _in_json(np.asarray(points, np.float64))
            else:                            # А-3: точки от `жест` / `путь_с_нажимом` / `тычок` — опорными точками
                if sweep.get('path_seed') == int(seed):
                    del sweep['path_seed']     # сид пути тот же, что мазка: повтор берёт "seed"
                d.update(sweep)
            d['seed'] = int(seed)
            for field, v, how_no in (('opacity', opacity, 1.0), ('layout', layout, 'полосы'), ('load', load, None),
                                     ('mix_below', bool(mix_below), False), ('touch', touch, None), ('ridges', by_ridges, None),
                                     ('level', even, None), ('reserve', reserve, None), ('load_along', load_along, None),
                                     ('fibers', fibers, None), ('hand', bool(pressure_hand), False), ('medium', thinner, None)):
                if v is not how_no and v != how_no:
                    d[field] = _in_json(v)
            if bands is not None:
                for field, v in zip(('stripe', 'stripe_len'), bands):
                    if v is not None:
                        d[field] = _in_json(v)
            if getattr(brush, 'просветы', 0.0) > 0 and brush.shape != 'мастихин':
                # К-14б: ход кистью из пучков — не одна полоса, а линии пучков; какие — в строку хода полем "lines"
                # (длина касания, ширина, нажим пучка): их читает зеркало однообразия. Счёт сухой — холст не тронут
                from .laying import lines_sweep
                with in_mm(self.px_on_mm):
                    lines = lines_sweep(points, brush.in_pixels(self.px_on_mm), int(seed), hand=bool(pressure_hand), touch=touch,
                                       load=thin(thinner, brush.load if load is None else load, 1.0, None, None)[0] if thinner else load)
                if lines:
                    d['lines'] = lines
            zh.row(zh._intent(d, stage, object, intent))
            self.__dict__.setdefault('_рамки', {})      # К-3: рамка мазка — по номеру его строки журнала (`править`)
            self._label = zh.lines
        return dict(points=points, paints=paints, layout=layout, layer=layer, seed=int(seed), opacity=float(opacity),
                    load=load, mix_below=mix_below, touch=touch, bands=bands, by_ridges=by_ridges, even=even,
                    reserve=reserve, load_along=load_along, fibers=fibers, pressure_hand=bool(pressure_hand), thinner=thinner,
                    measurement=bool(getattr(self, 'замер', True)), **({} if label is None else {'_п': label}),
                    **({'живая': True} if color is None and isinstance(brush._paint, Loading) else {})), zh   # Д-1: ход без цвета живой краской тратит её

    def _lay(self, z, brush, why):
        """Счёт мазка по записи `_принять_мазок` на этом холсте (у процесса счёта — на срезе холста): пиксели слоёв и
        что осталось на кисти. Ответ: (записанный прямоугольник с замером следа — `ответ.Ответ` — или None, кисть в пикселях)."""
        from .brush import lay_stroke
        layer, paints, even = z['слой'], z['краски'], z['ровно']
        target = self._layers[layer]
        names = list(self._layers)
        below = tuple(self._layers[and_] for and_ in names[:names.index(layer)])
        lower = below[::-1] if z['мешать_ниже'] else ()
        brush = brush.in_pixels(self.px_on_mm)
        lies = tuple(with_ for with_ in self._layers.values() if with_ is not target)   # п. 13: рельеф лежащей краски других слоёв — полупустой кисти
        turbidity = getattr(paints, 'мутность', 1.0)
        load, turbidity, even, reserve = thin(z.get('разбавитель'), brush.load if z['нагрузка'] is None else z['нагрузка'],
                                                     turbidity, even, z['запас']) if z.get('разбавитель') else (z['нагрузка'], turbidity, even, z['запас'])
        if turbidity < 1.0:                   # п. 10, инструмент 3: не кроющая краска — ей нужен рельеф лежащего под слоем
            kind = {'мутность': turbidity, 'под': below, 'пропуск': lambda: self._transmittance(target)}
        else:
            kind = {}
        if brush.shape == 'мастихин':            # «мастихин»: нож встаёт на твёрдое под своим слоем — ему нужен рельеф слоёв ниже
            kind['под'] = below
        if even:                            # п. 15: текучая краска заливает рельеф слоёв под своим — тот, что виден сквозь неё в `рельеф`
            kind['ровно_низ'] = (below, 1.0 - ZALAndInBrushA)
        grain = {} if self._grain is None else {'зерно': lambda x0, y0, x1, y1: self._grain_under(layer, x0, y0, x1, y1)}
        tapes = self._tapes_for(layer) if self.__dict__.get('_ленты') else ()
        before = self._to_records if getattr(self, '_снимки', None) else None
        if tapes:                            # Н-3: кромка ленты — ступенька рельефа под кистью; что легло у кромки — затёк
            grain = {'зерно': lambda x0, y0, x1, y1: self._relief_lent(tapes, layer, x0, y0, x1, y1)}
            to_tapes, snapshots = {}, before

            def before(p, _with=snapshots):
                if _with is not None:
                    _with(p)
                x0, y0, x1, y1 = (int(v) for v in p)       # холста целиком; массивы участка — от его угла
                to_tapes['п'] = (max(y0, self.y0), min(y1, self.y0 + self.height), max(x0, self.x0), min(x1, self.x0 + self.width))
                ya, yb, xa, xb = to_tapes['п']
                has = yb > ya and xb > xa
                to_tapes['h'] = np.array(target.thickness[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]) if has else None
                to_tapes['a'] = np.array(target.coverage[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]) if has else None
        fields = {}                           # 3г (К-1): поля следа — ответу мазка (`ответ.замерить`)
        lay_there = None
        if z.get('замер', True) and brush.thickness_film(brush.load if z['нагрузка'] is None else z['нагрузка']) > 0.0:
            from .encounter import LayThere      # Г-8: что лежало под кистью — до записи хода (только чтение)
            lay_there = before = LayThere(self, target, lower, getattr(brush, 'кайма', 0.0) > 0.0 and brush.shape != 'мастихин', before)
        loading = brush._paint if z.get('живая') else None      # Д-1: живая краска кисти — на миг счёта (на ядрах такой ход идёт в главном, по порядку)
        undermixing = brush.spread_tone               # Ф-2: медиум без пигмента — недомеса у него нет, пряди кисти его не красят
        if getattr(paints, 'медиум', False): brush.spread_tone = 0.0
        with in_mm(self.px_on_mm):          # Д-8: числа физики — в px этого холста (`мм.py`)
            try:
                measured = lay_stroke(target, z['точки'], paints, brush, z['сид'], z['непрозрачность'], why=why,
                                      layout=z['раскладка'], load=load, lower=lower, touch=z['касание'], bands=z['полосы'], neighboring=lies, by_ridges=z['по_гребням'], even=even, reserve=reserve, thinner=z.get('разбавитель') or 0.0, fibers=z['волокна'], hand=z['нажим_руки'], release=release_along(z['нагрузка_вдоль'], brush), loading=loading, **grain, **kind,
                                      window=(self.x0, self.y0, self.full_width, self.full_height),
                                      before_writing=before,
                                      measurement=fields.update if z.get('замер', True) else None)
                if tapes and to_tapes.get('h') is not None:
                    from .tape import at_edges
                    ya, yb, xa, xb = to_tapes['п']
                    t = np.asarray(z['точки'])
                    fluidity = max(float(z.get('разбавитель') or 0.0), float(even or 0.0))
                    for l in tapes:
                        at_edges(l, target, to_tapes['п'], to_tapes['h'], to_tapes['a'], t, self.px_on_mm, z.get('разбавитель'), fluidity,
                                 (self.y0, self.x0))
                    after = np.array(target.thickness[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0])
                    pressure = float(np.mean(t[:, 2])) if t.ndim == 2 and t.shape[1] > 2 else 0.85
                    for l in tapes:
                        l.seeped(to_tapes['п'], to_tapes['h'], after, fluidity, pressure, self.px_on_mm)
                consumption = brush.__dict__.pop('_расход', None)
                if consumption is not None:           # Д-1: что осталось после хода — кисти (и той, что задана в мм)
                    brush._change(loading.after(None, *consumption))
                if measured is not None and fields:
                    from .response import measure
                    measured = measure(self, measured, fields)
                    if lay_there is not None:
                        from .encounter import parse
                        parse(lay_there, measured, fields, paints, turbidity >= 1.0 and not z.get('разбавитель') and z['непрозрачность'] >= 1.0)
                    from .response import TELResponse
                    from .under_top import under_top
                    under_top(self, layer, measured, fields, TELResponse * measured.__dict__['тело'])      # Сл-1: что закрыли слои выше (только чтение)
                    if '_п' in z and '_след' in measured.__dict__:      # Св-1: сводка этапа — развёртка следа сразу, тем же счётом; у ответа — по запросу, как была
                        from .response import _unfold
                        measured.__dict__['_св'] = _unfold(measured.__dict__['_след'])
                    if loading is not None:
                        measured.__dict__['осталось'] = brush.left
            finally:
                brush.spread_tone = undermixing
                if brush is not brush:
                    brush._remainder = brush._remainder
                    brush._charged = getattr(brush, '_набрано', None)
                    brush._balance = getattr(brush, '_баланс', None)   # 4в-5: баланс краски кисти (`мягкая.Обмен.баланс`) — и у кисти, заданной в мм
        return measured, brush

    def _finish_computing_stroke(self, z, brush, zh):
        """Мазок, принятый `_принять_мазок`, — сразу, в этом процессе."""
        scouting = self.__dict__.get('_разведка')
        if scouting is not None:               # У-2: разведка `варианты` — только рамка мазка против области куска
            return scouting(self, z, brush)
        why = {} if self.__dict__.get('_только_в') is None else _InRegion(self._only_in, self.full_width, self.full_height)
        try:
            measured, brush = self._lay(z, brush, why)
        finally:
            if zh is not None:
                zh.after_stroke(brush)
        write_frame(self, why.get('рамка'))
        self._account_empty(measured, z['точки'], brush, z['непрозрачность'], z['нагрузка'], why, level=4, z=z)
        return measured

    def _account_empty(self, measured, points, brush, opacity, load, why, level=3, z=None, at_once=True):
        """Мазок без следа — в счёт `пустые` и словом `ПустойМазок` на первый каждого вида (v4, 1д). з — запись мазка:
        с меткой подсказок (`_п`) мазок идёт в их счёт (5б: С5, С6, С8); сразу — мазок досчитан на своём вызове."""
        kind = None
        if measured is None and not getattr(self, '_тихо', 0):
            reason = self._why_empty(points, brush, opacity, load, why)
            if reason is not None:
                kind = reason.split(':')[0].split(' — ')[0]
                if kind.startswith('нажим до'):
                    kind = 'нажим ниже порога касания'
                empty = self.__dict__.setdefault('_пустые', {})
                empty[kind] = empty.get(kind, 0) + 1
                if empty[kind] == 1:     # v4, 1д: одно слово на вид причины, дальше — счёт (`х.пустые`)
                    warnings.warn(EmptyStroke('мазок не оставил следа: ' + reason + f' (дальше «{kind}» — без слов, счёт — х.пустые)'), stacklevel=level)
        if z is not None and '_п' in z:
            p = self._p()
            if p is not None:
                p.stroke_laid(self, z['_п'], measured, kind, at_once)

    @property
    def empty(self):
        """v4, 1д: сколько мазков этого холста не оставили следа — по виду причины {вид: число}. Слово `ПустойМазок`
        звучит на первый мазок каждого вида, дальше — только счёт."""
        self._taken('пустые')
        self._boundary()
        return dict(getattr(self, '_пустые', {}))

    def seed_stroke(self, points, stage=None, object=None, take=False):
        """А-9: сид, который возьмёт `мазок(точки, …)` без своего: из сида листа, этапа и объекта (свои или от последней
        `пометка`), самих точек и того, который это мазок с такими точками в объекте. От номера мазка он не зависит:
        мазок, вставленный или убранный в другом месте, сиды прочих не трогает. взять=True — так делает `мазок`."""
        e = self._stage if stage is None else stage
        response = self._object if object is None else object
        t = np.ascontiguousarray(points, dtype=np.float64)
        h = hashlib.blake2b(json.dumps([self.seed, None if e is None else str(e), None if response is None else str(response),
                                        list(t.shape)]).encode('utf-8'), digest_size=8)
        h.update(t.tobytes())
        option = int.from_bytes(h.digest(), 'little') >> 1
        repeats = self._repeats if self._repeats is not None else {}
        n = repeats.get(option, 0)
        if take and self._repeats is not None:
            self._repeats[option] = n + 1
        return int.from_bytes(hashlib.blake2b(option.to_bytes(8, 'little') + n.to_bytes(4, 'little'),
                                              digest_size=4).digest(), 'little') & 0x7FFFFFFF

    def _why_empty(self, points, brush, opacity, load, why):
        """Е-7: почему мазок не оставил следа — по тому, где мазок кончился (`почему` от `положить_мазок`), и по вызову.
        None — молчать: след кисти лёг весь за краем холста, но ход подходил к краю ближе ширины кисти (заход за край —
        обычное дело, кусок кроют с запасом)."""
        t = np.asarray(points, np.float64).reshape(-1, np.shape(points)[-1] if np.ndim(points) == 2 else 2)
        x0, y0, x1, y1 = self.x0, self.y0, self.x0 + self.width, self.y0 + self.height
        W = float(brush.width)
        load = brush.load if load is None else load
        what = why.get('что')
        if float(opacity) <= 0.0:
            return f'непрозрачность {opacity}'
        if not isinstance(load, str) and float(load) <= 0.0 and what != 'вне':
            return ('нагрузка 0: на кисти нет своей краски — она только мешает сырую краску под собой (растушёвка), '
                    'а сырой краски этого слоя под ходом нет')
        if what == 'нажим 0' or (t.shape[1] >= 3 and float(t[:, 2].max()) <= 0.0):
            return 'нажим 0 по всему ходу: кисть над холстом'
        if what == 'вне':
            far = t[:, 0].max() < x0 - W or t[:, 0].min() > x1 + W or t[:, 1].max() < y0 - W or t[:, 1].min() > y1 + W
            if not far:
                return None
            sl = why.get('след')
            where = (f'след кисти x {sl[0]}…{sl[2]}, y {sl[1]}…{sl[3]}' if sl is not None else
                   f'x {t[:, 0].min():.0f}…{t[:, 0].max():.0f}, y {t[:, 1].min():.0f}…{t[:, 1].max():.0f}')
            return f'ход вне холста: {where} при холсте x {x0}…{x1}, y {y0}…{y1}'
        if what == 'пусто':
            return 'кисть коснулась, но краски не легло: краска на кисти кончилась (запас, сухая кисть; живая краска — `к.осталось`, добор — `набрать`)'
        if t.shape[1] >= 3:
            return f'нажим до {float(t[:, 2].max()):.3f} — ниже порога касания прядей кисти шириной {W:.1f} px'
        return 'кисть не коснулась холста (путь короче шага)'

    # ------------------------------------------------- проба мазка (п. 9)
    def remove(self, points, tool='тряпка', width=40.0, seed=0, layer=None, stage=None, object=None, intent=None):
        """К8.4: снять сырую краску по ходу. точки — (x, y, нажим[, …]), как у мазка (`путь_с_нажимом`, `жест`, сырые
        точки); инструмент — 'тряпка' (комок ткани: снимает сырое, оставляет тонкий тон и полосы нитей и складок вдоль
        хода, напитавшись — слабее) или 'скребок' (жёсткая кромка поперёк хода: до грунта, снятое — валиком за концы
        кромки и комком на подъёме), или 'черенок' (К-14а: круглый твёрдый конец ручки кисти процарапывает борозду —
        сырое не снято, а раздвинуто в валики у боков, объём тот же; нажим от 0,8 сажает конец на твёрдое — сухой слой
        или грунт, слабее — под ним плёнка, борозда в краске; сида не берёт; ответ — `снятие.Процарапано`);
        ширина — поперечник комка, длина кромки или поперечник конца черенка, px; слой — имя или None (все слои).
        Сухое и схватившееся не трогается. В журнале — строка с "tool"."""
        from .removal import remove
        if self.__dict__.get('_только_в') is not None:
            raise ValueError('варианты: снять в куске нельзя (рамку снятия с областью не сверить) — снимать до варианты или после взять')
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        zh = self._writes()
        if zh is not None:
            from .journal import _in_json
            d = {'tool': tool, 'size': _in_json(width), 'core': _in_json(np.asarray(points, np.float64)), 'seed': int(seed)}
            if layer is not None:
                d['layer'] = str(layer)
            zh.row(zh._intent(d, stage, object, intent))
        elif getattr(self, '_вне_журнала', False):
            self._outside_journal = 'мазки'
        return remove(self, points, tool, width, seed, layer)

    def _grain_under(self, layer, x0, y0, x1, y1):
        """П. 5б: зерно грунта в прямоугольнике холста (может выходить за край) — как его видит мазок в слой `слой`:
        (высота зерна под краской слоёв ниже этого, глубина зерна, функция заливки) — свой слой мазок зальёт сам."""
        g = self._grain.height(x0, y0, x1, y1)
        xa, ya, xb, yb = max(x0, self.x0), max(y0, self.y0), min(x1, self.x0 + self.width), min(y1, self.y0 + self.height)
        if xb > xa and yb > ya:
            h = None
            for name, with_ in self.layers.items():
                if name == layer:
                    break
                t = with_.thickness[ya - self.y0:yb - self.y0, xa - self.x0:xb - self.x0]
                h = t if h is None else h + t
            if h is not None:
                g[ya - y0:yb - y0, xa - x0:xb - x0] = self._grain.flooded(g[ya - y0:yb - y0, xa - x0:xb - x0], h)
        return g, self._grain.depth, self._grain.flooded

    def _relief_lent(self, tapes, layer, x0, y0, x1, y1):
        """Н-3: рельеф под мазком в слой `слой`, как его даёт `_зерно_под`, — со ступенькой лент: (высота, глубина,
        заливка); своя краска мазка на ленте ступеньку заливает, как зерно, по высоте ленты."""
        t = tapes[0].relief(x0, y0, x1, y1)
        for l in tapes[1:]:
            t = t + l.relief(x0, y0, x1, y1)
        gl_l = max(l.height for l in tapes)
        if self._grain is None:
            return t, gl_l, lambda g, h: g * np.exp(-h / (_ZERNResponse[6] * gl_l))
        g, gl, flooded = self._grain_under(layer, x0, y0, x1, y1)
        return g + t, max(gl, gl_l), lambda g, h: flooded(g - t, h) + t * np.exp(-h / (_ZERNResponse[6] * gl_l))

    def rectangle_stroke(self, points, brush, seed=None, touch=None, color=None, exactly=None, **how):
        """Прямоугольник (x0, y0, x1, y1) холста, вне которого мазок холста не читает и не пишет; None — мазок холста
        не тронет. С доводами мазка (цвет и `**как` — те же, что пойдут в `мазок`: слой, нагрузка, ровно, раскладка,
        запас…; без них — мазок с доводами по умолчанию, как `точно=True`) это рамка самого мазка: считается его же счётом —
        путь, кисть, пряди, их отставание и уход вбок, — до места, где мазок берёт холст (7–14 % времени мазка), пикселей не
        трогает; мазок с этими доводами за неё не выйдет никогда. Шире следа прядей она на поля мазка: 8 px и вал сырой
        краски (2 px + 0,15 ширины кисти); у мягкой кисти — и место её волосков на оси хода (там она смотрит сырое под
        собой). Сид — тот же, что пойдёт в `мазок` (None — сид листа, `сид_мазка`). Мазок с другими доводами (нагрузка,
        краска, раскладка…) ляжет иначе: его рамку просить с его доводами."""
        self._taken('прямоугольник_мазка')
        if brush is None:
            raise ValueError('прямоугольник_мазка: не задана кисть')
        if exactly is False:
            raise ValueError('прямоугольник_мазка: точно=False снято — рамка всегда счётом мазка')
        self._boundary()
        how = dict(how)
        stage, object = how.pop('этап', None), how.pop('объект', None)
        how.pop('намерение', None)
        layer = how.pop('слой', 'основа')
        color = self._by_palette(color)
        if color is None and brush._paint is None:
            color = (128, 128, 128)
        d = dict(opacity=1.0, layout='полосы', load=None, mix_below=False, bands=None, by_ridges=None, even=None,
                 reserve=None, polyline=False, load_along=None, fibers=None, pressure_hand=False, thinner=None)
        extra = [brush for brush in how if brush not in d]
        if extra:
            raise TypeError(f'прямоугольник_мазка: нет довода {extra[0]} (доводы — как у `мазок`)')
        d.update(how)
        points, paints, layout = self._parse_stroke(points, color, brush, d['раскладка'], d['нагрузка'], d.pop('ломаная'), d['волокна'])
        if seed is None:
            seed = self.seed_stroke(points, stage, object)
        d.update(points=points, paints=paints, layout=layout, layer=layer, seed=int(seed), touch=touch,
                 opacity=float(d['непрозрачность']), pressure_hand=bool(d['нажим_руки']))
        canvas = self
        if layer not in self._layers:               # слоя ещё нет: мазок завёл бы его пустым, верхним
            canvas = Canvas.__new__(Canvas)
            canvas.__dict__.update(self.__dict__)
            canvas._layers = dict(self._layers)
            canvas._layers[layer] = Layer(0, 0)
            canvas._snapshots, canvas._cores, canvas._cores = [], None, 1
        try:
            canvas._lay(d, brush.copy(), _OnlyFrame())
        except _Frame as e:
            r = e.args[0]
        else:
            return None
        if r[2] - r[0] < 3 or r[3] - r[1] < 3:
            return None
        p = (max(r[0], self.x0), max(r[1], self.y0), min(r[2], self.x0 + self.width), min(r[3], self.y0 + self.height))
        return p if p[2] > p[0] and p[3] > p[1] else None

    def subarea(self, region):
        """Дешёвая копия прямоугольника (x0, y0, x1, y1) холста: все слои (цвет, покрытие, толщина, сырость, схват),
        грунт, масштаб, счётчик сидов. Поля участка — плитками той же сетки: память — только его плитки. Мазки и картинки на участке — в координатах холста; мазок, который весь
        внутри участка (`прямоугольник_мазка` с доводами этого мазка), даёт на нём те же пиксели, что на холсте. Холст
        участок не трогает."""
        gy, gx = self._region(region)
        at = Canvas.__new__(Canvas)
        at.__dict__.update(self.__dict__)
        at.x0, at.y0 = self.x0 + gx.start, self.y0 + gy.start
        at._snapshots = []                         # журнал — тот же: мазки участка пишутся в журнал холста
        at._cores, at._cores = None, 1             # участок считает сам, на одном ядре
        at._hints = None                    # 5б: участок (варианты, перебор) молчит
        at._repeats = None if self._repeats is None else dict(self._repeats)
        at.width, at.height = gx.stop - gx.start, gy.stop - gy.start
        at._layers = {}
        for name, with_ in self.layers.items():
            n = Layer.__new__(Layer)
            for field in Layer._InWithE:
                a = getattr(with_, field)
                if a is not None or field in Layer._PResponseLYa:
                    setattr(n, field, a.cutout(gy, gx) if isinstance(a, Tiles) else Tiles.from_(a[gy, gx], Layer._PAtWithTResponse.get(field, 0.0)))
            wet = getattr(with_, '_сыро', None)
            if wet or (wet is None and (with_.moisture.has_outside(gy, gx) if isinstance(with_.moisture, Tiles)
                                          else bool(np.asarray(with_.moisture).any()))):
                n._wet = True             # 2г: сырое слоя есть и вне участка — мягкой кисти на участке ответ тот же, что на холсте
            at._layers[name] = n
        return at

    def fit_in(self, subarea):
        """Вернуть участок в холст: его слои (и новые) ложатся на своё место, счётчик сидов — его. Участок, писанный в
        другом процессе, у холста с журналом — предупреждение: его мазков в журнале нет."""
        if getattr(self, 'журнал', None) is not None and getattr(subarea, '_вне_журнала', False) == 'мазки':
            warnings.warn(EmptyStroke('вписать: мазки участка легли в другом процессе — в журнале их нет; '
                                      'класть участки в процессах — без журнала, или мазки — в этом процессе'), stacklevel=2)
        p = (subarea.x0, subarea.y0, subarea.x0 + subarea.width, subarea.y0 + subarea.height)
        self._boundary(writing=True)
        self._to_records(p)
        gy, gx = self._region(p)
        for name, with_ in subarea.layers.items():
            ts = self._layer(name)
            for field in Layer._PResponseLYa:
                getattr(ts, field)[gy, gx] = getattr(with_, field)
            if with_.transmittance is not None or ts.transmittance is not None:
                self._transmittance(ts)[gy, gx] = 1.0 if with_.transmittance is None else with_.transmittance
            if with_.flowing is not None or ts.flowing is not None:
                self._flowing(ts)[gy, gx] = 0.0 if with_.flowing is None else with_.flowing
        self._counter = subarea._counter
        self._repeats = subarea._repeats
        self._stage, self._object = subarea._stage, subarea._object

    def variants(self, piece, n, region, cores=None, in_folder=None):
        """n попыток одного куска разом. кусок(участок, i) кладёт мазки на копии участка `область` (x0, y0,
        x1, y1) — каждая копия в своём процессе (не больше ядер холста и 16; ядра — своё число), с холста, каким он
        был при вызове; i — номер варианта 0…n-1: чем варианты различаются, решает кусок (точки, цвета, объект пометки —
        сиды мазков — из сида листа, этапа, объекта пометки и точек). Ответ — список `Вариант` (`кисть/варианты.py`): .картинка (линейная, как
        `картинка`, — область), .ответ (что вернула кусок), .номер. Холст и кисти не трогаются; выбранный кладёт
        `взять`. Мазок куска, чья рамка (след с полями) выходит за область, — ValueError с подсказкой.
        в_папку (М-8) — каждый вариант ещё и на диск, `<в_папку>/<i>/` (картинка.png, журнал.jsonl, вариант.json,
        вариант.pkl), и лист `<в_папку>/рядом.png`: выбрать можно в другом процессе — `загрузить` тот же .npz с тем же
        журналом и `взять('<в_папку>/<i>')`. Холст перед этим — `сохранить`."""
        self._taken('варианты')
        from .variants import variants
        return variants(self, piece, n, region, cores, in_folder)

    def take(self, variant, own=False):
        """Выбранный вариант `варианты` — в холст и в журнал его мазками: поля области, строки журнала,
        кисти (подобранное, набранное), счётчик и сиды — те же, что дал бы кусок(х, i), писанный прямо в холст; прочие
        варианты забыты. Между `варианты` и `взять` холст и кисти не трогать — правка холста: ValueError.
        вариант — и путь `<в_папку>/<i>` (М-8): в холст, загруженный из того же .npz (отпечаток холста сверяется:
        не тот — ValueError, что не так); пиксели, журнал и .npz — как у куска прямо в этом холсте; кисти этого процесса
        не трогаются. С-1: файл варианта читается без выполнения кода из него — папку можно брать и чужую; в нём не
        данные движка — ValueError. своя=True — для своей папки, где кусок вернул объект своего класса (обычный pickle:
        чужой папке так не верить)."""
        self._taken('взять')
        from .variants import take
        return take(self, variant, own)

    def snapshot(self, region=None, brushes=()):
        """Запомнить холст для `откатить` (п. 11б): список слоёв, счётчик сидов и всё, что потом запишут мазки, сушка,
        схват и `вписать`, — по всему холсту, сколько бы мазков ни легло. кисти — их память (подобранное, набранное)
        тоже вернётся. область — прямоугольник, который копируется сразу (если в слои пишут мимо `мазок`); без неё
        снимок бесплатный: пиксели копируются перед записью, только тронутый прямоугольник. Снимок годен на
        много откатов; снимки, взятые после него, его откат гасит. Брошенный снимок (нет ссылок) холст забывает. Откат к нему обрезает записи старших
        снимков до их длины на миг его взятия: холст после отката тот же, что тогда."""
        self._taken('снимок')
        self._boundary(writing=True)
        with_ = Snapshot()
        with_.canvas = self
        with_.number = getattr(self, '_номер_снимка', 0) + 1
        self._number_snimka = with_.number
        with_.names = list(self.layers)
        with_.removed = {}
        with_.counter = self._counter
        with_.seeds = (None if self._repeats is None else dict(self._repeats), self._stage, self._object)
        with_.brushes = [(brush, copy.deepcopy(brush._remainder), copy.deepcopy(brush._paint)) for brush in brushes]
        with_.mixes = dict(self.__dict__.get('_замесы', {}))      # М-1: откат вернёт и палитру
        with_.tapes = [(l, l.seepage.copy()) for l in self.__dict__.get('_ленты', ())]   # Н-3: и ленты скотча
        with_.at_once = [] if region is None else [self._copy(region, with_.names)]
        with_.records = []
        live = [r for r in getattr(self, '_снимки', []) if r() is not None]
        with_.senior = [(r, len(r().records)) for r in live]         # длина записей старших на миг взятия (п. 14 (б))
        with_.journal = None if getattr(self, 'журнал', None) is None else self.journal.place()   # А-5: откат обрежет и журнал
        p = self._p()
        with_.hints = None if p is None else copy.deepcopy(p.count)   # 5б: откат вернёт и счёт подсказок
        self._snapshots = live + [weakref.ref(with_)]
        return with_

    def _copy(self, region, names):
        where = self._region(region)
        return where, {name: tuple(None if getattr(self.layers[name], field) is None else getattr(self.layers[name], field)[where].copy()
                                for field in Layer._InWithE) for name in names}

    @staticmethod
    def _transmittance(layer):
        """Массив пропуска слоя (п. 10, инструмент 3) — заводится с первой не кроющей краской в слое (в общей памяти,
        если там слой)."""
        if layer.transmittance is None:
            layer.transmittance = Tiles(layer.color.shape, 1.0, shared=_in_shared(layer.color))
        return layer.transmittance

    @staticmethod
    def _flowing(layer):
        """Массив следа текучей кисти слоя (К4) — заводится с первым текучим ходом в слое."""
        if layer.flowing is None:
            layer.flowing = Tiles(layer.thickness.shape, shared=_in_shared(layer.thickness))
        return layer.flowing

    def _to_records(self, region):
        """Перед записью в прямоугольник холста: прежние пиксели — каждому живому снимку (слои, что были при нём)."""
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        if not getattr(self, '_снимки', None):
            return
        if region is None:
            region = (self.x0, self.y0, self.x0 + self.width, self.y0 + self.height)
        live = []
        for r in self._snapshots:
            with_ = r()
            if with_ is None or with_.canvas is not self:
                continue
            live.append(r)
            where = self._region(region)
            if where[0].stop > where[0].start and where[1].stop > where[1].start:
                with_.records.append(self._copy(region, [and_ for and_ in with_.names if and_ in self._layers and and_ not in with_.removed]))
        self._snapshots = live

    def edit(self, cores=None, path=None, **edit):
        """К-3, А-7: правка задним числом — строки журнала этого холста (`журнал=`) по фильтру правятся, на холсте
        пересобирается только затронутое: после вызова холст и кисти — бит в бит как полный повтор правленого журнала.
        Фильтр и правка — доводы `кисть.задним.править_строки`: строки_с ("4-200"), слой, этап, объект, замес,
        область (x0, y0, x1, y1); удалить, цвет, новый_замес, dL, dC, dh, теплее, сдвиг (dx, dy), нагрузка, нажим,
        нажим_в, ширина. Прежний журнал — `<журнал>.до_правки`; снимки до правки гаснут. путь — PNG «до | после» по
        рамке затронутого 1:1. Ответ — `кисть.задним.Правка` (строк правлено, пересчитано, доля, площадь, время, как)."""
        from .retroactive import edit_canvas
        return edit_canvas(self, cores=cores, path_png=path, **edit)

    def roll_back(self, snapshot):
        """Вернуть холст к снимку бит в бит: все тронутые пиксели всех слоёв, новые слои убраны, счётчик и кисти снимка
        прежние. Снимок остаётся годен (можно класть и откатывать снова); снимки, взятые после него, гаснут."""
        self._taken('откатить')
        with_ = snapshot
        if not isinstance(with_, Snapshot) or with_.canvas is not self:
            raise ValueError('откатить: снимок не этого холста')
        if not any(r() is with_ for r in self._snapshots):      # взят после снимка, к которому уже откатили
            raise ValueError('откатить: снимок погас')
        self._boundary(writing=True)
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        for name in [and_ for and_ in self._layers if and_ not in with_.names or and_ in with_.removed]:
            del self._layers[name]            # новые слои; и новый слой того же имени, что убран после снимка
        self._layers.update(with_.removed)       # убранные после снимка — на место, с их краской
        self._layers = {and_: self._layers[and_] for and_ in with_.names}       # порядок снимка
        with_.removed = {}
        not_was = set()
        for where, arrays in reversed(with_.at_once + with_.records):
            for name, fields in arrays.items():
                for field, a in zip(Layer._InWithE, fields):
                    if a is not None:
                        getattr(self.layers[name], field)[where] = a
                    elif getattr(self.layers[name], field) is not None:      # пропуск, текучая: при снимке такого массива в слое не было
                        not_was.add((name, field))
        for name, field in not_was:          # и нет снова (2б): слой с пустым `пропуск` свет считал бы не кроющим, `сохранить` писал бы лишний массив
            setattr(self.layers[name], field, None)
        with_.records = []
        for r, length in with_.senior:                                # холст — как на миг взятия с: записи старших,
            senior = r()                                           # взятые позже, не нужны (п. 14 (б))
            if senior is not None:
                del senior.records[length:]
        self._counter = with_.counter
        if getattr(with_, 'замесы', None) is not None and (with_.mixes or self.__dict__.get('_замесы')):
            self._mixes = dict(with_.mixes)
        self._repeats, self._stage, self._object = (None if with_.seeds[0] is None else dict(with_.seeds[0])), with_.seeds[1], with_.seeds[2]
        if getattr(with_, 'ленты', None):
            self._tapes = [l for l, _ in with_.tapes]
            for l, seepage in with_.tapes:
                l.seepage = seepage.copy()
        else:
            self.__dict__.pop('_ленты', None)
        if with_.journal is not None and getattr(self, 'журнал', None) is not None:
            self.journal.roll_back(with_.journal)
            r = self.__dict__.get('_рамки')
            if r is not None:                                   # К-3: рамки мазков, чьи строки откат снял
                for n in [n for n in r if isinstance(n, int) and n > self.journal.lines]:
                    del r[n]
        p = self._p()
        if p is not None and getattr(with_, 'подсказки', None) is not None:
            p.count = copy.deepcopy(with_.hints)
        for brush, remainder, paint in with_.brushes:
            brush._remainder, brush._paint = copy.deepcopy(remainder), copy.deepcopy(paint)
        self._snapshots = [r for r in self._snapshots if r() is not None and r().number <= with_.number]

    def __getstate__(self):
        self._boundary()
        d = dict(self.__dict__)
        d['_снимки'] = []            # снимки — этого холста в этом процессе: в копию и в pickle не идут
        d['_ядра'] = None            # очередь мазков и её процессы — тоже: копия заведёт свои
        d['_подсказки'] = None       # 5б: подсказки — этого холста в этом процессе
        d.pop('_запас_зерна', None)  # М-9: запас высоты зерна — этого процесса, считается заново
        if d.get('журнал') is not None:   # файл журнала — этого процесса: копия в другом процессе пишет мимо него
            d['журнал'], d['_вне_журнала'] = None, True
        return d

    def __deepcopy__(self, memo):
        self._boundary()
        n = Canvas.__new__(Canvas)
        memo[id(self)] = n
        for brush, v in self.__dict__.items():
            setattr(n, brush, [] if brush == '_снимки' else None if brush in ('журнал', '_ядра', '_подсказки', '_запас_зерна') else copy.deepcopy(v, memo))
        if n.__dict__.get('_ядер', 1) != 1:          # копия считает на ядрах, как и холст: её слои — в общей памяти
            for with_ in n._layers.values():
                for field in Layer._InWithE:
                    setattr(with_, field, _in_shared(getattr(with_, field)))
        return n

    def trial(self, points, color=None, brush=None, region=None, light=True, **how):
        """Пробный мазок с откатом: положить, снять картинку области (по умолчанию — изменённого прямоугольника),
        вернуть холст как был. Кисть не пачкается (мазок идёт её копией). Ответ: (прямоугольник, картинка) —
        картинка линейная float32 (`картинка`), None — если мазок холста не тронул."""
        self._taken('проба')
        if brush is None:
            raise ValueError('проба: не задана кисть')
        self._boundary(writing=True)
        seed = how.pop('сид', None)
        seed = self.seed_stroke(points, how.get('этап'), how.get('объект')) if seed is None else seed
        p = self.rectangle_stroke(points, brush, seed, color=color, exactly=True, **how)
        if p is None:
            return None, None
        self.__dict__['_тишина'] = self.__dict__.get('_тишина', 0) + 1     # 5б: проба молчит
        with_ = self.snapshot()                  # 11б: копирует только то, что мазок тронет
        zh = getattr(self, 'журнал', None)
        if zh is not None:
            zh.pause += 1                     # проба в журнал не идёт
        try:
            measured = self.stroke(points, color, brush.copy(), seed=seed, **how)
            if measured is None:
                return None, None
            return measured, self.image(light=light, region=region or measured)
        finally:
            if zh is not None:
                zh.pause -= 1
            self.roll_back(with_)
            self.__dict__['_тишина'] -= 1

    def _region(self, region):
        """Прямоугольник холста (x0, y0, x1, y1) -> срезы массивов (у участка — со сдвигом на его угол)."""
        if region is None:
            return slice(0, self.height), slice(0, self.width)
        x0, y0, x1, y1 = (int(round(v)) for v in region)
        x0, x1, y0, y1 = x0 - self.x0, x1 - self.x0, y0 - self.y0, y1 - self.y0
        x0, y0 = min(max(x0, 0), self.width), min(max(y0, 0), self.height)
        return slice(y0, max(min(y1, self.height), y0)), slice(x0, max(min(x1, self.width), x0))

    # ------------------------------------------------------------------ скотч-трафарет (Н-3)
    @property
    def tapes(self):
        """Н-3: ленты скотча, что лежат на холсте, по порядку (`кисть.скотч.Лента`: имя, маска, слой, высота_мм)."""
        return list(self.__dict__.get('_ленты', ()))

    def _tapes_for(self, layer):
        return [l for l in self.__dict__.get('_ленты', ()) if l.catches(layer)]

    def tape(self, mask, layer=None, name=None, height_mm=None, stage=None, object=None, intent=None):
        """Н-3: малярная лента там, где `маска` (bool H×W холста) истинна. Краска мазков поверх ленты ложится на ленту,
        а не на холст и не на слой под ней; кромка — ступенька высоты ленты (0,1 мм — `высота_мм`), кисть идёт через неё
        по рельефу, как через гребень сухой краски: с ленты сходит не обрывом, у кромки по холсту до дна достаёт по
        нажиму и густоте. слой — None: лента на всех слоях; имя — лента только для мазков в этот слой (другие её не
        видят). имя — имя ленты (`снять_скотч(имя)`); None — «лента N». Пока лента лежит, `картинка` под ней — цвет
        грунта и краска на ленте; мазки считаются в этом процессе (без ядер). В журнале — строка {"tape": …}. Ответ — лента."""
        from . import tape as With
        if (self.x0, self.y0, self.width, self.height) != (0, 0, self.full_width, self.full_height):
            raise ValueError('скотч: только на холсте целиком (не на участке и не в варианте)')
        mask = np.asarray(mask)
        if mask.shape != (self.height, self.width):
            raise ValueError(f'скотч: маска {mask.shape} — нужна (высота, ширина) холста ({self.height}, {self.width})')
        mask = mask.astype(bool)
        if not mask.any():
            raise ValueError('скотч: маска пуста — ленте негде лечь')
        if layer is not None and not isinstance(layer, str):
            raise ValueError('скотч: слой — имя слоя или None (все слои)')
        tapes = self.__dict__.setdefault('_ленты', [])
        if name is None:
            n = len(tapes) + 1
            while any(l.name == f'лента {n}' for l in tapes):
                n += 1
            name = f'лента {n}'
        if any(l.name == str(name) for l in tapes):
            raise ValueError(f'скотч: лента «{name}» уже лежит (ленты: {", ".join(l.name for l in tapes)})')
        height_mm = With.HEIGHT_MM if height_mm is None else float(height_mm)
        if not 0.0 < height_mm <= 1.0:
            raise ValueError('скотч: высота_мм — от 0 до 1 мм (малярная лента — 0,1)')
        self._boundary(writing=True)
        zh = self._writes()
        if zh is not None:
            d = {'tape': str(name), 'shape': [self.height, self.width], 'mask': With.mask_in_lines(mask)}
            if layer is not None:
                d['layer'] = layer
            if height_mm != With.HEIGHT_MM:
                d['height'] = height_mm
            zh.row(zh._intent(d, stage, object, intent))
        l = With.Tape(name, mask, layer, height_mm)
        y0, y1, x0, x1 = l.frame
        self._to_records((x0, y0, x1, y1))
        With.lay(self, l)
        tapes.append(l)
        return l

    def remove_tape(self, name=None, stage=None, object=None, intent=None):
        """Н-3: снять ленту `имя` (None — все, сверху вниз): краска на ней уходит с ней; под ней — что было до неё, бит в
        бит; за кромкой — краска как легла. У кромки — физика: текучая краска (разбавитель, `ровно`), лёгшая у кромки,
        затекла под ленту на доли мм (глубже с нажимом), густая — нет; сырой пастоз, лёгший через кромку, рвётся (часть
        тела у кромки уходит с лентой), сухой (`подсушить`) режется ровно. Лента под другой лентой — ValueError: сперва
        верхнюю. В журнале — строка {"untape": …}. Ответ — {имя: {'затёк': px, 'разрыв': px}}."""
        from . import tape as With
        if (self.x0, self.y0, self.width, self.height) != (0, 0, self.full_width, self.full_height):
            raise ValueError('снять_скотч: только на холсте целиком (не на участке и не в варианте)')
        tapes = self.__dict__.get('_ленты') or []
        if not tapes:
            raise ValueError('снять_скотч: лент на холсте нет')
        if name is None:
            remove = list(reversed(tapes))
        else:
            found = [l for l in tapes if l.name == str(name)]
            if not found:
                raise ValueError(f'снять_скотч: ленты «{name}» нет (ленты: {", ".join(l.name for l in tapes)})')
            l = found[0]
            above = [in_ for in_ in tapes[tapes.index(l) + 1:] if (in_.mask & l.mask).any() and (in_.layer is None or l.layer is None or in_.layer == l.layer)]
            if above:
                raise ValueError(f'снять_скотч: на ленте «{l.name}» лежит «{above[0].name}» — сперва её')
            remove = [l]
        self._boundary(writing=True)
        zh = self._writes()
        if zh is not None:
            zh.row(zh._intent({'untape': None if name is None else str(name)}, stage, object, intent))
        result = {}
        for l in remove:
            y0, y1, x0, x1 = l.frame
            self._to_records((x0, y0, x1, y1))
            result[l.name] = With.remove(self, l)
            tapes.remove(l)
        if not tapes:
            del self.__dict__['_ленты']
        return result

    def _in_piece(self, what, region):
        """2г: на участке варианта сушка и схват — только в области куска (на холсте без области они шли бы по всему)."""
        response = self.__dict__.get('_только_в')
        if response is None:
            return
        if region is None or not (response[0] <= region[0] and response[1] <= region[1] and region[2] <= response[2] and region[3] <= response[3]):
            raise ValueError(f'варианты: {what} в куске — только с область= внутри области куска {response} (без неё на холсте '
                             'она шла бы по всему холсту); сушить весь холст — до варианты или после взять')

    def _in_journal(self, what, share, layer, region):
        zh = self._writes()
        if zh is None:
            return
        from .journal import _in_json
        d = {what: _in_json(share)}
        if layer is not None:
            d['layer'] = str(layer)
        if region is not None:
            d['area'] = _in_json(list(region))
        zh.row(zh._intent(d))

    def dry(self, share=1.0, layer=None, region=None):
        """Убавить влажность на долю (1.0 — досуха). Время сушки задаёт вызывающий.
        область = (x0, y0, x1, y1) — сушить только этот прямоугольник."""
        self._taken('подсушить')
        self._in_piece('подсушить', region)
        self._in_journal('dry', share, layer, region)
        self._to_records(region)
        where = self._region(region)
        for name, with_ in self.layers.items():
            if layer is None or name == layer:
                elementwise(with_.moisture, where, lambda w: clamp(np.subtract(w, np.float32(share)), 0.0, 1.0))

    def set(self, share=0.5, layer=None, region=None):
        """Дать сырой краске схватиться на долю (1.0 — корка): влажность та же, но по такому слою
        мазок ложится резко и лишь чуть цепляет лежащее. Свежая краска поверх не схвачена."""
        self._in_piece('схватить', region)
        self._in_journal('set', share, layer, region)
        self._to_records(region)
        where = self._region(region)
        for name, with_ in self.layers.items():
            if layer is None or name == layer:
                elementwise(with_.degree_of_set, where, lambda s: clamp(np.add(s, np.float32(share)), 0.0, 1.0))

    def re_wet(self, share=1.0, layer=None, region=None):
        """М-3: пара `подсушить` — масло поверх сухого: верх высохшей краски снова сырой, и растушёвка (`нагрузка=0`),
        тряпка, скребок и нож берут это место, как сырое. доля 0–1 (0 — ничего). Размокает только кожа плёнки — не
        глубже 0,25 h (0,1 мм) от поверхности: тонкое письмо оживает целиком, у пастозного вала тело под кожей стоит
        твёрдым; повторный вызов глубже не идёт. Схват (`схватить`) снимается в ту же меру. слой — имя или None (все,
        глубина — от поверхности); область = (x0, y0, x1, y1). Это правка места, не листа: размоченное заново мешается
        с тем, что кладут сверху. В журнале — строка {"wet": …}. Ответ — площадь, где влажность выросла, px."""
        from .edit import re_wet
        self._in_piece('размочить', region)
        self._in_journal('wet', share, layer, region)
        self._to_records(region)
        return re_wet(self, share, layer, region)

    def press(self, points, brush=None, width=None, layer=None, depth=None, stage=None, object=None, intent=None):
        """М-3: ход мастихина плашмя по высохшей краске — срезать то, что выше плоскости ножа: вершины сухого вала,
        гребни пастоза, резкий бугор. точки — (x, y, нажим[, …]), как у мазка (`жест`, `путь_с_нажимом`, сырые точки);
        кисть — мастихин (берётся его ширина) или ширина — длина кромки, px. Нажим до 0,3 — нож едет по вершинам и
        ничего не берёт; 0,6 — полпути к телу; от 0,9 лезвие садится на тело краски под собой (медиана высот краски под
        подошвой 4 мм × кромка; голый грунт рядом не в счёт) и ниже не идёт: гребни пастоза срезаются до его середины,
        и на косом ходе у кромки вала тоже; вал на ровном тонком слое срезается до слоя.
        глубина — мм (None — без предела): осторожный ход, ниже самой высокой точки под ножом больше чем на столько
        лезвие за ход не садится.
        Срезанное (сухое и сырое) уходит на нож и с холста; цвет слоя на месте, грунт и зерно не режутся. слой — имя
        (режется только он, и не под целой краской другого слоя) или None (все, сверху вниз). Кисть ворса сухое по-прежнему не мнёт. Это правка места:
        срезанная площадка гладкая, много таких — лист как пластик (подсказка «срезано ножом»). В журнале — строка с
        "tool": "нож". Ответ — `Прижато`: `срезано` px, `на_нож` (объём), `доля`, `глубже` мм, `почему` — если пусто."""
        from .edit import press
        if self.__dict__.get('_только_в') is not None:
            raise ValueError('варианты: прижать в куске нельзя (рамку хода с областью не сверить) — прижимать до варианты или после взять')
        if (brush is None) == (width is None):
            raise ValueError('прижать: нужен нож — кисть=Кисть мастихин или ширина= (длина кромки, px), одно из двух')
        if brush is not None:
            if brush.shape != 'мастихин':
                raise ValueError(f'прижать: это ход ножа — кисть формы «мастихин», не «{brush.shape}» (ворс сухое не режет)')
            if (brush.width_mm is not None or brush.spacing_mm is not None) and self.px_on_mm is None:
                raise ValueError('кисть в мм: у холста не задан px_на_мм')
            width = brush.in_pixels(self.px_on_mm).width
        self.__dict__['_правок'] = self.__dict__.get('_правок', 0) + 1   # 2г: правка холста — варианты, считанные до неё, не годны
        zh = self._writes()
        if zh is not None:
            from .journal import _in_json
            d = {'tool': 'нож', 'size': _in_json(width), 'core': _in_json(np.asarray(points, np.float64)), 'seed': 0}
            if depth is not None:
                d['depth'] = _in_json(depth)
            if layer is not None:
                d['layer'] = str(layer)
            zh.row(zh._intent(d, stage, object, intent))
        elif getattr(self, '_вне_журнала', False):
            self._outside_journal = 'мазки'
        response = press(self, points, float(width), layer, depth)
        p = self._p()
        if p is not None and response.cut_off:
            p.cut_off(self, response._cells)
        return response

    # ------------------------------------------------------------ сохранение
    def save(self, path, beyond=None):
        """Состояние холста в .npz: слои целиком (и след текучей кисти, К4), счётчик мазков, грунт, зерно.
        сверх (К3) — (словарь, массивы) того, кто пишет файл: словарь (JSON) — в мету, массивы — рядом под именами
        `x_<имя>`; так `кисть.повтор` кладёт в файл кисти и место в журнале. Тот же холст — тот же файл байт в байт.
        Ответ — `Сохранено`: путь, счётчик мазков, `подсказки` — новые с прошлой выдачи (М-2+М-4-2)."""
        self._boundary()
        meta = {'ширина': self.width, 'высота': self.height, 'грунт': self.ground_srgb,
                'слои': list(self.layers), 'счётчик': self._counter}
        if self._repeats is not None and (self.seed or self._stage is not None or self._object is not None):
            meta.update(seed=self.seed, stage=self._stage, object=self._object)
        if self.grain is not None:
            meta.update(grain=self.grain, px_on_mm=self.px_on_mm)
        elif self.px_on_mm is not None:
            meta.update(px_on_mm=self.px_on_mm)
        if self.__dict__.get('_замесы'):     # М-1: палитра — с холстом (повтор с точки, `загрузить`)
            from .journal import color_lines
            meta['палитра'] = [[and_, color_lines(Paint(tuple(z), z.kind))] for and_, z in self._mixes.items()]
        if self._writes() is not None:
            self.journal.saved(path, self._counter)       # А-1: строка {"saved": …} — `загрузить(…, журнал=)` продолжит с неё
        p, summary = self._p(), ''
        if p is not None:
            summary = p.saved(self, path)                          # 5б: счёт подсказок на это сохранение; граница
        if beyond is not None:
            meta['сверх'] = beyond[0]
        if self.__dict__.get('_ленты'):       # Н-3: ленты скотча — с холстом: маска, что под ней, затёк
            meta['ленты'], tapes_m = [], {}
            names = list(self.layers)
            for j, l in enumerate(self._tapes):
                meta['ленты'].append({'имя': l.name, 'слой': l.layer, 'высота_мм': l.height_mm,
                                      'под': {and_: sorted(fields) for and_, fields in l.under.items() if and_ in names}})   # слой убран — его нет
                tapes_m[f'L{j}m'] = l.mask
                tapes_m[f'L{j}z'] = l.seepage
                for and_, fields in l.under.items():
                    for field, a in (sorted(fields.items()) if and_ in names else ()):
                        tapes_m[f'L{j}_{names.index(and_)}_{field}'] = a
        arrays = {'мета': np.array(json.dumps(meta, ensure_ascii=False))}
        if self._repeats:                    # А-9: мазки, уже легшие с теми же точками в объекте
            brush = np.array(sorted(self._repeats), dtype=np.uint64)
            arrays['повторы'] = np.column_stack([brush, np.array([self._repeats[int(v)] for v in brush], dtype=np.uint64)])
        for i, with_ in enumerate(self.layers.values()):
            arrays[f'c{i}'] = with_.color
            arrays[f'a{i}'] = with_.coverage
            arrays[f'h{i}'] = with_.thickness
            arrays[f'w{i}'] = with_.moisture
            arrays[f's{i}'] = with_.degree_of_set
            if with_.transmittance is not None:
                arrays[f't{i}'] = with_.transmittance
            if with_.flowing is not None:
                arrays[f'l{i}'] = with_.flowing
        if meta.get('ленты'):
            arrays.update(tapes_m)
        if beyond is not None:
            arrays.update((f'x_{name}', a) for name, a in beyond[1].items())
        response = Sokhraneno(path, self._counter, [] if p is None else p.new(), summary)
        # как np.savez_compressed, но время записи в архив не идёт: один холст — один файл. Пишется рядом и встаёт на
        # место целиком (1г): обрыв не оставляет битого файла, а файл, на который смотрит жёсткая ссылка (точка повтора
        # под тем же `--save`), не переписывается на месте — ссылка рвётся, точка остаётся как была
        target = os.fspath(path) if isinstance(path, (str, os.PathLike)) else None
        where_to = path if target is None else f'{target}.часть{os.getpid()}'
        try:
            with zipfile.ZipFile(where_to, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                # М-9: поля сжимаются в потоках, по порядку и тем же zlib, что у zipfile, — файл тот же байт в байт
                compress = _by_order(_compress_npy, ((a,) for a in arrays.values()), self._flows())
                for (name, a), ready in zip(arrays.items(), compress):
                    writing = zipfile.ZipInfo(name + '.npy', date_time=(1980, 1, 1, 0, 0, 0))
                    writing.compress_type = zipfile.ZIP_DEFLATED
                    with z.open(writing, 'w', force_zip64=True) as f:
                        if not _fit_in_compressed(f, ready):
                            np.lib.format.write_array(f, np.asanyarray(a), allow_pickle=False)
            if target is not None:
                os.replace(where_to, target)
        except BaseException:
            if target is not None and os.path.exists(where_to):
                os.remove(where_to)
            raise
        self._write_path_journal(path)
        if p is not None and target is not None:      # Св-1: сводка — файлом рядом; пустая — прежний файл рядом убирается
            beside = target + '.сводка.txt'
            try:
                if summary:
                    with open(beside, 'w', encoding='utf-8') as f:
                        f.write(summary + '\n')
                elif os.path.exists(beside):
                    os.remove(beside)
            except OSError:
                pass
        return response

    def _write_path_journal(self, path):
        """М-10 (а): путь журнала, которым описан холст, — в файл рядом `<путь>.путь_журнала` (полный и относительно `.npz`):
        `загрузить` без `журнал=` возьмёт его для чтения. Не в сам `.npz`: тот же холст — тот же файл байт в байт, с каким
        бы журналом и где он ни писался (повтор, два процесса). Журнала нет — прежний файл рядом убирается."""
        if not isinstance(path, (str, os.PathLike)):
            return
        beside = os.fspath(path) + '.путь_журнала'
        zh = self.journal_path
        if zh is None:
            if os.path.exists(beside):
                os.remove(beside)
            return
        z = {'путь': os.path.abspath(zh), 'рядом': os.path.relpath(os.path.abspath(zh), os.path.dirname(os.path.abspath(path))),
             'счётчик': self._counter}
        with open(beside + '.часть', 'w', encoding='utf-8') as f:
            json.dump(z, f, ensure_ascii=False)
        os.replace(beside + '.часть', beside)

    @classmethod
    def load_file(cls, path, beyond=False, journal=None, cores=None, hints=None):
        """Холст из файла `сохранить`. сверх=True (К3) — ответ (холст, словарь, массивы): что положил в файл пишущий
        (`сохранить(…, сверх=)`); словарь None — в файле этого нет. журнал (А-1) — путь журнала, с которым холст писался:
        журнал обрезается до строки {"saved": …} этого файла и пишется дальше (правка этапа 5 с `этап4.npz` — строки
        этапов 5… уходят). Без `журнал=` (М-10 (а)) — журнал из `<путь>.путь_журнала` только для чтения (`журнал_путь`). В журнале такой строки нет — ValueError. ядра, подсказки — как у `Холст(…)`; с журналом подсказки
        берут свой счёт на этом сохранении (`<журнал>.подсказки.json`)."""
        with np.load(path) as d:
            meta = json.loads(str(d['мета']))
            fields = {}
            if 'px_на_мм' in meta:
                fields['px_на_мм'] = meta['px_на_мм']
            if 'зерно' in meta:
                fields['зерно'] = meta['зерно']
            canvas = cls(meta['ширина'], meta['высота'], meta['грунт'], cores=cores, hints=False if journal is not None else hints, **fields)
            shared = lambda a, empty=0.0: Tiles.from_(a, empty, shared=canvas._cores != 1)   # 2в: плитками; 2б: на ядрах — в общей памяти
            canvas._counter = meta.get('счётчик', 0)
            canvas.seed, canvas._stage, canvas._object = int(meta.get('сид', 0)), meta.get('этап'), meta.get('объект')
            if 'повторы' in d:
                canvas._repeats = {int(brush): int(n) for brush, n in d['повторы']}
            if meta.get('палитра'):
                from .palette import mix
                from .replay import _color
                canvas._mixes = {and_: mix(and_, _color(fields_colors)) for and_, fields_colors in meta['палитра']}
            for i, name in enumerate(meta['слои']):
                with_ = Layer.__new__(Layer)
                with_.color = shared(d[f'c{i}'])
                with_.coverage = shared(d[f'a{i}'])
                with_.thickness = shared(d[f'h{i}'])
                with_.moisture = shared(d[f'w{i}'])
                with_.degree_of_set = shared(d[f's{i}']) if f's{i}' in d else shared(np.zeros(with_.moisture.shape, np.float32))
                with_.transmittance = shared(d[f't{i}'], 1.0) if f't{i}' in d else None
                with_.flowing = shared(d[f'l{i}']) if f'l{i}' in d else None
                canvas._layers[name] = with_
            if meta.get('ленты'):                # Н-3: ленты скотча
                from .tape import Tape
                canvas._tapes = []
                for j, response in enumerate(meta['ленты']):
                    l = Tape(response['имя'], d[f'L{j}m'], response['слой'], response['высота_мм'])
                    l.seepage = np.array(d[f'L{j}z'])
                    l.under = {and_: {p: np.array(d[f'L{j}_{meta["слои"].index(and_)}_{p}']) for p in fields} for and_, fields in response['под'].items()}
                    canvas._tapes.append(l)
            canvas._loaded_from = os.path.basename(os.fspath(path)) if isinstance(path, (str, os.PathLike)) else None
            if journal is not None:
                from .journal import Journal
                canvas.journal = Journal(journal, canvas, resume=(path, canvas._counter))
                canvas._stage, canvas._object = canvas.journal.stage, canvas.journal.object
                canvas._hints = Hints(canvas, _hints_enabled(hints), journal, new=False)
                if canvas._hints.on:
                    canvas._hints.loaded(canvas, path, canvas._counter)    # 5б: С3 — пуск с этого сохранения
            elif isinstance(path, (str, os.PathLike)) and os.path.isfile(os.fspath(path) + '.путь_журнала'):
                with open(os.fspath(path) + '.путь_журнала', encoding='utf-8') as f:     # М-10 (а): только для чтения
                    writing = json.load(f)
                zh = _find_journal(writing, path)
                canvas._journal_reading = (zh, canvas._counter, writing.get('путь'),
                                    None if zh is None else _to_sokhraneniya(zh, path, canvas._counter))
            if beyond:
                return canvas, meta.get('сверх'), {name[2:]: d[name] for name in d.files if name.startswith('x_')}
        return canvas

    @property
    def journal_path(self):
        """М-10 (а): путь журнала, которым описан холст — тот, что он пишет (`журнал=`), или взятый для чтения из `.npz`
        (`загрузить` без `журнал=`, пока на холст не легло новых мазков); None — журнала нет. По нему читают
        `однообразие`, `регулярность`, `виды`, `манера`; его же — `python3 -m кисть.повтор <путь>`."""
        zh = getattr(self, 'журнал', None)
        if zh is not None:
            return zh.path
        h = self.__dict__.get('_журнал_чтения')
        if h is None or h[0] is None or h[1] != self._counter:
            return None
        return h[0]

    # ----------------------------------------------------------------- сборка
    def relief(self, y0=0, y1=None, x0=0, x1=None, full=False, grain=True):
        """Суммарная толщина краски по слоям: толщина копится от слоя к слою.

        Верхний слой заливает неровности нижнего: под его краской рельеф нижних слоёв берётся
        ослабленным (на ЗАЛИВКА = 70 % при полном покрытии) — иначе борозды нижнего мазка
        проступают сквозь верхний и он кажется прозрачным. полный=True — простая сумма."""
        y1 = self.height if y1 is None else y1
        x1 = self.width if x1 is None else x1
        h = np.zeros((y1 - y0, x1 - x0), np.float32)
        for i, with_ in enumerate(self.layers.values()):
            if i and not full:
                h *= 1.0 - ZALAndInBrushA * with_.coverage[y0:y1, x0:x1]
            h += with_.thickness[y0:y1, x0:x1]
        if grain and getattr(self, '_зерно', None) is not None:
            # п. 5б: зерно грунта под краской — тонкая плёнка идёт по нитям, толстая заливает ямки
            total = sum(with_.thickness[y0:y1, x0:x1] for with_ in self.layers.values()) if self.layers else 0.0
            h += self._grain.flooded(self._height_grain(y0, y1, x0, x1), total)
        return h

    def _height_grain(self, y0, y1, x0, x1):
        """М-9: высота зерна грунта под строками y0:y1, столбцами x0:x1 массива холста — из запаса полосами по
        `_ПОЛОСА_ЗЕРНА` строк во всю ширину: зерно не меняется, а вывод большого холста считал его заново на каждую
        полосу и каждый взгляд. Значения те же бит в бит (`Зерно.высота` поточечна). Запас — этого процесса: в копию и
        в pickle не идёт."""
        z = self.__dict__.get('_запас_зерна')
        if z is None or z[0] is not self._grain or z[1] != (self.x0, self.y0, self.width, self.height):
            z = self._reserve_grain = (self._grain, (self.x0, self.y0, self.width, self.height), {})
        bands, P = z[2], _BAND_GRAIN
        parts = []
        for p in range(y0 // P, -(-y1 // P)):
            a = bands.get(p)
            if a is None:
                a = bands[p] = self._grain.height(self.x0, self.y0 + p * P, self.x0 + self.width,
                                                   self.y0 + min(self.height, (p + 1) * P))
            parts.append(a[max(y0 - p * P, 0):min(y1 - p * P, P), x0:x1])
        return parts[0].copy() if len(parts) == 1 else np.concatenate(parts)

    def _without_grooves(self, y0, y1, x0, x1):
        """К9. Поправка к наклону рельефа (по x, по y) для света, или None, если слои не лежат друг на друге.

        `рельеф` берёт толщу нижних слоёв под краской верхнего ослабленной — так заливаются их борозды. Но у кромки
        верхнего слоя от этого встаёт ступень, которой на холсте нет: тонкая плёнка будто роет в толстом слое канаву
        во всю свою ширину, и стенка канавы под светом — белый контур вокруг бледного хода. Наклон рельефа — это
        наклон толщины слоёв (ослабленный под краской выше) и та самая ступень: толщина слоя на перепад его
        закрытости. Ступень здесь и считается, чтобы свет её вычел; под самой краской и на открытом месте она —
        ровно ноль, там свет прежний бит в бит. Поле счёта — тот же 1 px, что у света. Ступень от кромки некроющего слоя
        (у слоя есть `пропуск`) не вычитается."""
        layers = list(self.layers.values())
        open, veil, kx, ky = None, None, None, None    # доля рельефа, не залитая кроющими слоями выше и некроющими
        for i in range(len(layers) - 1, 0, -1):
            a = layers[i].coverage[y0:y1, x0:x1]
            if layers[i].transmittance is not None:      # некроющий слой (лессировка, вуаль): его ступень остаётся как была —
                if a.any():                      # с ней мерены пороги п. 10, инструмент 3 (`test_лессировка`); долг
                    veil = 1.0 - ZALAndInBrushA * a if veil is None else veil * (1.0 - ZALAndInBrushA * a)
            elif open is not None or a.any():
                open = 1.0 - ZALAndInBrushA * a if open is None else open * (1.0 - ZALAndInBrushA * a)
            t = layers[i - 1].thickness[y0:y1, x0:x1]
            if open is None or not t.any():
                continue
            if kx is None:
                kx, ky = np.zeros(t.shape, np.float32), np.zeros(t.shape, np.float32)
            # те же разности, что в `свет.множители`: середина — полуразность соседей, край массива — односторонняя
            in_ = t if veil is None else t * veil
            kx[:, 1:-1] += (in_[:, 2:] + in_[:, :-2]) * (open[:, 2:] - open[:, :-2]) * 0.25
            kx[:, 0] += (in_[:, 1] + in_[:, 0]) * (open[:, 1] - open[:, 0]) * 0.5
            kx[:, -1] += (in_[:, -1] + in_[:, -2]) * (open[:, -1] - open[:, -2]) * 0.5
            ky[1:-1] += (in_[2:] + in_[:-2]) * (open[2:] - open[:-2]) * 0.25
            ky[0] += (in_[1] + in_[0]) * (open[1] - open[0]) * 0.5
            ky[-1] += (in_[-1] + in_[-2]) * (open[-1] - open[-2]) * 0.5
        return None if kx is None else (kx, ky)

    def image(self, light=True, strength=None, region=None, shadow=0.0, surface=None):
        """Готовая картинка, линейный свет, float32 (В, Ш, 3). область = (x0, y0, x1, y1) холста: считается только она
        (со светом — с полем 2 · `свет.поле_перелива()` + 1 px), и она равна тому же месту полной картинки. У участка свет
        у его края считается без соседей: брать область не ближе этого поля к краю участка (если тот не край холста).
        тень — сила падающей тени по рельефу (п. 20): 0 — нет, картинка как была; True — рабочая сила
        (`свет.ТЕНЬ_РАБОЧАЯ`); с тенью поле области — `свет.ПОЛЕ_ТЕНИ` px больше.
        Г-2: со светом светлая краска возвращает свет, ушедший с наклонов (`свет.ПЕРЕОТРАЖЕНИЕ`), а свет выше белого
        переливается к соседям (`свет.перелить`) — белила под светом не серят. поверхность — ключ вывода
        ('матовое', 'полуматовое', 'лак'; None — умолчание: слабый узкий блик без дымки): блик и насыщение под плёнкой масла; без света ключ
        ничего не меняет; журнал, слои и холст не трогает."""
        _light.surface_output(surface)
        if region is None and not self.__dict__.get('_в_глазу'):
            p = self._p()
            if p is not None:
                p.output(self)                   # 5б: С4 — холст выведен целиком
        return self._image(light, strength, region, shadow, surface)

    def _image(self, light, strength, region, shadow, surface, in_bands=True):
        """`картинка` без подсказки вывода; полосами=False — область одним куском (полоса `png`)."""
        gy, gx = self._region(region)
        x0, y0, x1, y1 = gx.start, gy.start, gx.stop, gy.stop
        if not (light and (self.layers or getattr(self, '_зерно', None) is not None)):
            return self._color(y0, y1, x0, x1)
        shadow = _strength_shadows(shadow)
        bands = self._bands_image(y0, y1, x1 - x0) if in_bands else None
        if bands is not None:          # М-9: большая область — полосами в потоках: память полосы, а не холста
            self._boundary()
            brush = np.empty((y1 - y0, x1 - x0, 3), np.float32)

            def band(a, b):
                return self._image_light(x0, a, x1, b, strength, shadow, surface)
            with in_mm(self.px_on_mm):
                for (a, b), p in zip(bands, _by_order(band, bands, self._flows())):
                    brush[a - y0:b - y0] = p
            return brush
        return self._image_light(x0, y0, x1, y1, strength, shadow, surface)

    def _image_light(self, x0, y0, x1, y1, strength, shadow, surface):
        """Картинка со светом строк y0:y1, столбцов x0:x1 массива холста одним куском (`картинка`)."""
        with in_mm(self.px_on_mm):          # Д-8: высота рельефа и перелив в px этого холста
            pp = _light.field_spill()
            field = (_light.FIELD_SHADOWS if shadow else 1) + 2 * pp
            ya, yb = max(0, y0 - field), min(self.height, y1 + field)
            xa, xb = max(0, x0 - field), min(self.width, x1 + field)
            h = self.relief(ya, yb, xa, xb)
            m, b = _light.multiplier(h, _light.STRENGTH_BY_DEFAULT if strength is None else strength, shadow,
                                    self._without_grooves(ya, yb, xa, xb), surface)
            D = _light.shortfall(m)
            # перелив берёт соседей не дальше `пп` px: считается на области с этим полем, затем вырезается
            ca, cb = max(ya, y0 - pp), min(yb, y1 + pp)
            da, db = max(xa, x0 - pp), min(xb, x1 + pp)
            in_ = (slice(ca - ya, cb - ya), slice(da - xa, db - xa))
            brush = _light.overflow(_light.output(self._color(ca, cb, da, db), m[in_], b[in_], D[in_], surface))
        return np.ascontiguousarray(brush[y0 - ca:y0 - ca + (y1 - y0), x0 - da:x0 - da + (x1 - x0)])

    def _color(self, y0, y1, x0, x1):
        """Цвет красок без света, линейный, float32 (В, Ш, 3), строки y0:y1, столбцы x0:x1 массива холста."""
        brush = np.empty((y1 - y0, x1 - x0, 3), np.float32)
        brush[:] = self.ground
        for with_ in self.layers.values():
            a = with_.coverage[y0:y1, x0:x1, None]
            if with_.transmittance is not None:        # п. 10, инструмент 3: не кроющая краска слоя темнит то, что под ним, на просвет
                brush *= with_.transmittance[y0:y1, x0:x1]
            brush += (with_.color[y0:y1, x0:x1] - brush) * a
        return brush

    # -------------------------------------------------------------------- глаза (кисть/глаз.py): только читают
    def look(self, region=None, scale=1.0, light=True, strength=None, beside=None, path=None, without=None, to=None,
               surface=None):
        """Холст глазами: sRGB 8 бит (uint8) области, увеличенный в `масштаб` раз; рядом — образец тем же местом и
        масштабом; путь — PNG 8 бит; без=[слои] / до='слой' — как без этих слоёв или только ниже слоя (Б-5);
        поверхность — Г-2, ключ вывода ('матовое', 'полуматовое', 'лак'). Ответ — `Ряд` (тот же uint8) с полем `подсказки`:
        новые с прошлой выдачи (М-2+М-4-2). Подробно — `кисть.глаз.взгляд`."""
        from . import eye
        top = not self.__dict__.get('_в_глазу')
        with self._eye('взгляд'):
            row = eye.look(self, region, scale, light, strength, beside, path, without, to, surface)
        if not top:
            return row
        row = row.view(Row)                     # М-2+М-4-2: те же байты, и поле `подсказки` — новые с прошлой выдачи
        p = self._p()
        row.hints = [] if p is None else p.new()
        return row

    def grid(self, reference=None, spacing=None, region=None, scale=1.0, laying='как есть', beside=False, path=None, light=True):
        """М-6: сетка в px холста с подписями поверх холста или образца, уложенного на холст (укладка 'как есть' — как
        `рядом` у `взгляд`, 'вписать', 'обрезать', 'растянуть'); рядом=True — «холст | образец». Тем же масштабом и
        областью, что `взгляд`. Только картинка. Ответ — `Ряд` uint8 с полем `укладка`. Подробно — `кисть.глаз.сетка`."""
        from . import eye
        with self._eye('сетка'):
            return eye.grid(self, reference, spacing, region, scale, laying, beside, path, light)

    def views(self, region=None, journal=None, without=None, to=None, size=500, path=None, light=True, target=None):
        """К-6: один лист видов холста — прищур, тон, насыщенность, тепло-холод, края, толщина, сырость, направления и
        размеры мазков (из журнала) — и словарь их массивов; цель — её прищур рядом. Подробно — `кисть.глаз.виды`."""
        from . import eye
        with self._eye('виды'):
            return eye.views(self, region, journal, without, to, size, path, light, target)

    def overview(self, target, region=None, crops=4, window=300, size=800, path=None, light=True, without=None, to=None):
        """Б-10: лист — область издали с рамками мест наибольшей разности с целью и эти места 1:1 «холст | цель».
        Подробно — `кисть.глаз.обзор`."""
        from . import eye
        with self._eye('обзор'):
            return eye.overview(self, target, region, crops, window, size, path, light, without, to)

    def monotony(self, object=None, stage=None, region=None, strokes=None, layer=None, reference=None, journal=None,
                    path=None, cell=None):
        """К-12: зеркало однообразия — что у мазков зоны одинаково (длины, наклоны, ширины, профили нажима, кисти,
        цвета, нагрузка) и оценка «один приём с дрожанием» 0–1 со строкой словами; Б-9 — похожи ли места на сетку или
        на равномерный случай. Зона — по журналу холста (или `журнал` — путь .jsonl): объект, этап, слой, область
        (центр мазка), мазки (номера строк или строки). Без отбора (область — можно) — и места: каждый объект по клеткам
        `клетка` мм (50), худшие с рамкой и словами, что узко; `путь` — PNG-карта мест поверх холста. Подробно —
        `кисть.однообразие`."""
        from . import monotony as Response
        from . import eye
        with self._eye('однообразие'):
            background = None
            if path is not None and object is None and stage is None and layer is None and strokes is None:
                background = eye._kind(self, eye._frame(self, None), True)
            return Response.monotony(self._path_journal(journal, 'однообразие'), object, stage, region, strokes, layer, reference,
                                 path, cell, background)

    def regularity(self, object=None, stage=None, region=None, strokes=None, layer=None, journal=None):
        """Б-9: мерка регулярности мест, наклонов, длин и шага мазков зоны — 'сетка', 'случай' 0–1 и 'слово'.
        Подробно — `кисть.однообразие.регулярность_мазков`."""
        from . import monotony as Response
        with self._eye('регулярность'):
            return Response.regularity(self._path_journal(journal, 'регулярность'), object, stage, region, strokes, layer)

    def _journal_for_reading(self):
        """Путь журнала холста или, если холст взял журнал из `.npz`, а лист писался дальше этого сохранения, — байты
        журнала до строки этого сохранения (мазков, которых на холсте нет, мерки не видят). None — журнала нет."""
        zh = self.journal_path
        h = self.__dict__.get('_журнал_чтения')
        if zh is None or getattr(self, 'журнал', None) is not None or h is None or h[3] is None:
            return zh
        with open(zh, 'rb') as f:
            return f.read(h[3])

    def _path_journal(self, journal, what):
        if journal is not None:
            return journal
        zh = self._journal_for_reading()
        if zh is None:
            h = self.__dict__.get('_журнал_чтения')
            why = ''
            if h is not None and h[0] is None:
                why = f' (файл <имя>.npz.путь_журнала рядом с .npz называет журнал {h[2]}, а его файла нет — ни по этому пути, ни рядом с .npz)' if len(h) > 2 else ''
            elif h is not None and h[1] != self._counter:
                why = (f' (журнал {h[0]} взят из .npz для чтения и описывает {h[1]} мазков, а после загрузки легло ещё '
                          f'{self._counter - h[1]} без журнала — Холст.загрузить(…, журнал=) пишет их)')
            raise ValueError(f'{what}: у холста нет журнала{why} — Холст(…, журнал=\'лист.jsonl\') или {what}(журнал=путь .jsonl)')
        return zh

    def gap(self, region=None, layer=None, threshold=0.5, area=4, holes=50, path=None, scale=1, without=None, to=None):
        """Б-3: карта просвета — доля грунта (или того, что под слоем `слой`), видная сквозь краску, по пикселю — и дыры
        (место от `порог`, от `площадь` px) списком с рамкой, центром и площадью; `путь` — PNG «холст | карта».
        Подробно — `кисть.просвет`."""
        from .gap import gap
        with self._eye('просвет'):
            return gap(self, region, layer, threshold, area, holes, path, scale, without, to)

    def scan(self, strokes, region=None, target=None, path=None, scale=1, cores=None, measure=None, **param):
        """Б-8: пачка мазков на нескольких значениях параметра (`нагрузка=[0.006, 0.012, 0.025]`) — варианты в процессах
        (`варианты`), лист «до | варианты» и замеры каждого; холст не тронут, выбранный — `х.взять(п[k])`.
        Подробно — `кисть.перебор`."""
        from .scan import scan
        with self._eye('перебор'):
            return scan(self, strokes, region, target, path, scale, cores, measure, **param)

    def manner(self, region=None, reference=None, region_sample=None, px_on_mm_sample=None, object=None, stage=None,
               layer=None, journal=None, light=True, window=8.0, path=None, without=None, to=None):
        """К-4: мерка манеры области по картинке — палитра и замесы, разброс тона и цвета в пятне, края жёсткие и
        мягкие, направления и крупность фактуры; мазки — по журналу; с `образец` — его мерка той же линейкой и разница.
        Подробно — `кисть.манера`."""
        from .manner import manner
        with self._eye('манера'):
            m = manner(self, region, reference, region_sample, px_on_mm_sample, object, stage, layer, journal, light,
                       window, path, without, to)
        p = self.__dict__.get('_подсказки')
        if reference is not None and p is not None:
            p.manner(m)                            # 5г: С9 называет «цвет в пятне» холста и образца
        return m

    def mean(self, region=None, light=False):
        """Средний цвет области: {'L', 'C', 'h', 'rgb', 'hex'}, L — серая шкала 0–255. Подробно — `кисть.глаз.средний`."""
        from . import eye
        with self._eye('средний'):
            return eye.mean(self, region, light)

    def transition(self, region, axis='x', spacing=8, light=False):
        """Спад тона поперёк стыка (У-1): профиль L, ширина 10–90 %, наибольшая ступень на `шаг` px. Подробно — `кисть.глаз.переход`."""
        from . import eye
        with self._eye('переход'):
            return eye.transition(self, region, axis, spacing, light)

    def difference(self, target, region=None, light=False, window=None, worst=5, path=None, scale=10.0):
        """Холст против цели (картинка, холст, `.npz`, функция f(x, y)): карта dL и сводка — среднее, σ, худшие окна.
        Подробно — `кисть.глаз.разность`."""
        from . import eye
        with self._eye('разность'):
            return eye.difference(self, target, region, light, window, worst, path, scale)

    def windows(self, target, width, n=3, region=None, tolerance=2.0, light=False):
        """Мерка «окно N ширин» против цели. Подробно — `кисть.глаз.окна`."""
        from . import eye
        with self._eye('окна'):
            return eye.windows(self, target, width, n, region, tolerance, light)

    def gray(self, target, n=200, region=None, light=False):
        """Мерка «серая N» против цели. Подробно — `кисть.глаз.серая`."""
        from . import eye
        with self._eye('серая'):
            return eye.gray(self, target, n, region, light)

    def darker(self, region, background, light=False, to=None):
        """Мерка «объект темнее фона на X»: фон — прямоугольник холста или цель; до='слой' — фон на холсте только со
        слоями ниже него (У-2). Подробно — `кисть.глаз.темнее`."""
        from . import eye
        with self._eye('темнее'):
            return eye.darker(self, region, background, light, to)

    def furrows(self, region=None, width=12, light=True, strength=None):
        """Мерка «борозды, L»: то, что уже `ширина` px, со светом рельефа. Подробно — `кисть.глаз.борозды`."""
        from . import eye
        with self._eye('борозды'):
            return eye.furrows(self, region, width, light, strength)

    def _in_bands(self):
        for y0 in range(0, self.height, _PResponseLResponseWithA):
            yield y0, min(self.height, y0 + _PResponseLResponseWithA)

    # -------------------------------------------------------------------- PNG
    def png(self, path, light=True, strength=None, shadow=0.0):
        """Готовая картинка, PNG RGB 16 бит (sRGB). тень — сила падающей тени (п. 20), 0 — нет."""
        p = self._p()
        if p is not None and not self.__dict__.get('_в_глазу'):
            p.output(self)                       # 5б: С4 — холст выведен целиком
        self._boundary()
        def band(y0, y1):
            return in_16(from_linear(self._image(light, strength, (self.x0, self.y0 + y0, self.x0 + self.width, self.y0 + y1), shadow,
                                                   None, in_bands=False)))
        with in_mm(self.px_on_mm):      # М-9: числа масштаба — один раз здесь, полосы в потоках их не подменяют
            _write_png(path, self.height, self.width, 3, _by_order(band, self._in_bands(), self._flows()))

    def _bands_image(self, y0, y1, width):
        """М-9: строки области картинки со светом — полосами [(a, b), …] или None (одним куском, как прежде). Полосой —
        если область больше `_ОБЛАСТЬ_ЦЕЛИКОМ` px (счёт света держит в памяти около 180 байт на px: холст 6400×4000
        целиком — 4,6 ГБ) или, при потоках больше одного, больше `_ОБЛАСТЬ_ПОТОКАМ` (меньшую быстрее посчитать одним
        куском: у каждой полосы своё поле света). Полоса — та же картинка того же места (поля света берёт сама),
        значения бит в бит."""
        lines, n = y1 - y0, self._flows()
        if lines < 2 or width < 1 or lines * width <= (_REGION_ENTIRELY if n == 1 else _REGION_FLOW):
            return None
        h = max(_PResponseLResponseWithA // 2, min(_REGION_ENTIRELY // width, -(-lines // n)))
        if h >= lines:
            return None
        return [(a, min(y1, a + h)) for a in range(y0, y1, h)]

    def _flows(self):
        """М-9: сколько потоков считают полосы вывода и сжимают поля — ядра холста (как у мазков), не больше ядер, что
        даны процессу, и не больше `_ПОТОКОВ_ВЫВОДА`: каждый держит рабочие массивы своей полосы (и арену памяти),
        а сверх четырёх вывод упирается в сжатие, которое идёт одним потоком."""
        n = self.__dict__.get('_ядер')
        if n is None:
            from .cores import number_cores
            n = number_cores(None)
        try:
            n = min(n, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            pass
        return max(1, min(int(n), _FLOWS_OUTPUT))

    def png_layer(self, path, layer):
        """Слой цвета с альфой (покрытие), PNG RGBA 16 бит, альфа прямая."""
        with_ = self.layers[layer]

        def bands():
            for y0, y1 in self._in_bands():
                p = np.empty((y1 - y0, self.width, 4), np.uint16)
                p[..., :3] = in_16(from_linear(with_.color[y0:y1]))
                p[..., 3] = in_16(with_.coverage[y0:y1])
                yield p
        _write_png(path, self.height, self.width, 4, bands())

    def png_thicknesses(self, path, layer=None, legend=True):
        """Карта толщины, PNG серый 16 бит, шкала абсолютная — одна на все холсты: 0 — нет краски, 65535 =
        `ШКАЛА_ТОЛЩИНЫ` единиц h = `ШКАЛА_ТОЛЩИНЫ_ММ` (1,6 мм); толще — белый. Мм на уровень — в куске tEXt
        `kist:mm_per_level`. слой=None — рельеф всех слоёв (тот, по которому считается свет), иначе толщина одного слоя.
        легенда (М-10 (в)) — рядом файл `<имя>.легенда.png` (8 бит, глазам): та же карта той же шкалой и полоса шкалы с
        подписями в мм, сколько места толще шкалы и наибольшая толщина. Ответ — путь легенды или None (легенда=False)."""
        sheets = [] if legend else None

        def bands():
            for y0, y1 in self._in_bands():
                h = self.relief(y0, y1) if layer is None else self.layers[layer].thickness[y0:y1]
                p = in_16(h / SCALE_THICKNESSES)
                if sheets is not None:
                    sheets.append((p, float(h.max()) if h.size else 0.0, int((h > SCALE_THICKNESSES).sum())))
                yield p[..., None]
        _write_png(path, self.height, self.width, 1, bands(),
                    {'kist:mm_per_level': f'{SCALE_THICKNESSES_MM / 65535:.9g}', 'kist:mm_at_white': f'{SCALE_THICKNESSES_MM:g}',
                     'kist:what': 'paint relief, all layers' if layer is None else 'layer thickness'})
        if sheets is None:
            return None
        from .eye import legend_thicknesses
        return legend_thicknesses(path, np.concatenate([p for p, _, _ in sheets]) if sheets else np.zeros((0, self.width), np.uint16),
                               max([m for _, m, _ in sheets] + [0.0]) * MM_ON_H, sum(n for _, _, n in sheets),
                               'рельеф всех слоёв' if layer is None else f'слой «{layer}»')

    def png_heights(self, path, without=None, to=None, full=False):
        """М-10 (г), карта высот для печати (РЕШЕНИЕ DEN 19.11 00:23 +04 «ок»): PNG серый 16 бит, пиксель в пиксель с
        `х.png(путь)` — высота краски над грунтом в мм, `ММ_НА_УРОВЕНЬ_ВЫСОТ` (0,001 мм) на уровень серого, 0 — грунт
        (до 65,5 мм). Высота — та, что видит свет (`х.рельеф`: слой выше заливает борозды нижних), без зерна грунта;
        полный=True — простая сумма толщин слоёв. без=[слои] / до='слой' — как без этих слоёв (Б-5). Мм на уровень и
        `px_на_мм` — в кусках tEXt PNG и в соседнем `<имя>.json` (там же размер в px и мм, наибольшая высота, слои).
        Принтер не задан — общий формат. Холст, журнал и сиды не трогает. Ответ — словарь того же `.json`."""
        self._taken('png_высот')
        from .eye import without_layers
        canvas = without_layers(self, without, to)
        px_mm = RResponseDNResponseY if self.px_on_mm is None else float(self.px_on_mm)
        max = [0.0]

        def bands():
            for y0, y1 in self._in_bands():
                h = canvas.relief(y0, y1, full=full, grain=False) * MM_ON_H
                if h.size:
                    max[0] = max(max[0], float(h.max()))
                yield (np.clip(h / MM_ON_LEVEL_HEIGHTS, 0.0, 65535.0) + 0.5).astype(np.uint16)[..., None]
        _write_png(path, self.height, self.width, 1, bands(),
                    {'kist:mm_per_level': f'{MM_ON_LEVEL_HEIGHTS:g}', 'kist:px_per_mm': f'{px_mm:g}',
                     'kist:what': 'paint height above ground, mm; 0 = ground'})
        layers = [and_ for and_ in canvas.layers]
        d = {'мм_на_уровень': MM_ON_LEVEL_HEIGHTS, 'px_на_мм': px_mm, 'ширина_px': self.width, 'высота_px': self.height,
             'ширина_мм': round(self.width / px_mm, 3), 'высота_мм': round(self.height / px_mm, 3),
             'наибольшая_высота_мм': round(max[0], 4), 'выше_шкалы': max[0] > 65535 * MM_ON_LEVEL_HEIGHTS,
             'слои': layers, 'высота': 'простая сумма толщин' if full else 'как для света (слой выше заливает борозды нижних)',
             'зерно_грунта': 'не входит', 'файл': os.path.basename(os.fspath(path))}
        with open(os.path.splitext(os.fspath(path))[0] + '.json', 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=1)
        return d

    def png_light(self, path, strength=None, shadow=0.0):
        """Слой света отдельно: серый 16 бит, нейтраль 50 %, для режима «Жёсткий свет». тень — сила падающей
        тени (п. 20), 0 — нет: тень идёт в тот же слой."""
        strength = _light.STRENGTH_BY_DEFAULT if strength is None else strength
        shadow = _strength_shadows(shadow)
        field = _light.FIELD_SHADOWS if shadow else 1

        def bands():
            for y0, y1 in self._in_bands():
                ya, yb = max(0, y0 - field), min(self.height, y1 + field)
                with in_mm(self.px_on_mm):
                    with_ = _light.layer_light(self.relief(ya, yb), strength, shadow, self._without_grooves(ya, yb, 0, self.width))
                yield in_16(with_[y0 - ya:y0 - ya + (y1 - y0)])[..., None]
        _write_png(path, self.height, self.width, 1, bands())
