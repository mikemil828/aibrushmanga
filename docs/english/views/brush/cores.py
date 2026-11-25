# READING VIEW of движок/кисть/ядра.py with names in English. Not runnable; edit the original.
"""Е-1 (v4, 2б): мазки на ядрах — бит в бит как на одном. И в повторе журнала, и в письме из Python (`х.мазок`).

Мазок читает и пишет холст только в своей рамке (`Холст.прямоугольник_мазка`: след прядей с полями — смешение, сырое
под ним, рельеф соседних слоёв, вал; у мягкой кисти — и место её волосков на оси хода). Два мазка, чьи рамки не
пересекаются, друг друга не видят — их можно считать разом. `Ядра` держит очередь мазков в порядке вызовов и отдаёт
мазок процессу, когда его рамка не пересекает ни одного более раннего мазка, который ещё не досчитан; порядок мазков,
что пересекаются, — прежний. Итог — те же байты всех полей холста и кистей, что на одном ядре, потому что каждый мазок
видит под собой ровно то, что видел бы в свой черёд.

Рамка мазка — не оценка заранее, а число самого мазка: отставание прядей вдоль хода и уход вбок в мазке нормально
распределены, формулой с запасом их не накрыть. Поэтому мазок идёт в процесс дважды: сперва — счёт до рамки (холста
не трогает; 7–14 % времени мазка), потом, когда рамка известна и свободна, — сам мазок на срезе холста по этой рамке.
Пока рамка мазка неизвестна, мазки после него не кладутся (их рамки считаются). Мазок, раньше которого в очереди
недосчитанных нет, ждать некого: он идёт в процесс один раз, на весь холст, — называет главному свою рамку и кладётся
дальше (так идут ходы по одному месту — ореол, подмалёвок внахлёст: второго счёта до рамки у них нет).

Что ещё мазок берёт не из рамки — и как это держится:
- кисть несёт подобранное прошлым мазком (`_остаток`): мазок той же кистью ждёт прошлого, если кисть между ними не
  вытерта (вытертая, с заданной грязью — не ждёт); счёт до рамки подобранного не берёт и не ждёт;
- мягкая кисть спрашивает, есть ли в слое сырое вообще (`мягкая.сырое_под`): ответ даётся мазку заранее, когда
  известны рамки всех более ранних мазков его слоя (до ответа мазок своей рамки не считает, а значит, и мазки после
  него не кладутся). «Да» — если в слое есть сырой пиксель-свидетель вне рамок всех мазков этого слоя, что стоят в
  очереди (им его не тронуть, а досчитанные все раньше этого мазка): свидетели берутся редкой сеткой по слою и
  убывают, когда на них ложится рамка нового мазка слоя. Свидетеля нет — мазок ждёт, пока досчитаются более ранние
  мазки слоя, и ответ — есть ли в слое сырой пиксель. По сырому слою она ещё до рамки смотрит сырое под волосками на
  оси хода: её счёт до рамки идёт на срезе холста по этой оси и ждёт более ранних мазков, чьи рамки ось пересекают;
- массивы слоя по надобности (`пропуск`, `текучая`) заводит только главный процесс: такой мазок считается в нём;
  новый слой или массив — процессы заводятся заново (им виден только холст, каким он был при их рождении);
- слои лежат в общей памяти (анонимный mmap), процессы — ветвлением (fork); мазок в процессе видит только свою рамку
  (срезы массивов), и рамка, которую он назовёт, сверяется со срезом: вышла за него — ошибка, а не «почти равно».

Граница (`граница`, `досчитать`) — любое чтение или правка холста мимо мазка: до неё всё досчитано. Холст зовёт её сам
(`Холст._граница`, чтение массива слоя, чтение подобранного кистью). Память: как на одном ядре (слои — те же, общие) плюс
на каждый процесс рабочие массивы одного мазка; в очереди — точки не больше `ОЧЕРЕДЬ` мазков.
"""
import copy
import multiprocessing
import os
import pickle
import time
import traceback
import warnings
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from .paint import thin
from .tool import wet_field
from .tiles import by_grid

PResponseTResponseLResponseBrush = 16            # процессов — не больше (правило машины)
ResponseHERED = 512           # мазков в очереди: дальше вызов мазка ждёт счёта
_WithETBrushA = 16             # px: шаг сетки, которой ищутся сырые пиксели-свидетели слоя
_WithInAndDETELEY = 256       # столько свидетелей держится на слой
WITH_STROKE = 48            # первые мазки холста — в главном процессе: короткий лист процессов не заводит
_PREZhNEE = '\0прежнее'   # поле кисти (`_остаток`, `_набрано`, `_баланс`), которого мазок не тронул
_NET = object()         # рамка мазка ещё не считана
_ZhDYoT = object()        # мягкой кисти ещё не сказано, есть ли в слое сырое
_WithLResponseTY = ('_цвет', '_покрытие', '_толщина', '_влажность', '_схват', '_пропуск', '_текучая')


def number_cores(cores=None):
    """Сколько процессов брать: число из вызова; None — переменная окружения `KIST_CORES`, без неё — ядра машины, не
    больше 16. 1 — счёт на одном ядре, как прежде."""
    if cores is None:
        cores = os.environ.get('KIST_CORES') or os.cpu_count() or 1
    cores = int(cores)
    if cores < 1:
        raise ValueError('ядра: число процессов от 1')
    return min(cores, PResponseTResponseLResponseBrush)


def allowed():
    """Можно ли этому процессу заводить процессы счёта: потомку-демону (процесс пула) — нельзя, он считает сам."""
    return not multiprocessing.current_process().daemon


class TightRectangle(RuntimeError):
    """Мазок назвал рамку шире среза холста, который ему дан по его же рамке: на ядрах он не бит в бит."""


class _Why(dict):
    """Словарь «почему пусто» мазка в процессе: рамку мазка (вне неё мазок холста не читает и не пишет) сверяет со срезом
    холста, как только мазок её назвал, — до первой записи. Шире среза — мазок бросается нетронутым."""

    def __init__(self, canvas, at):
        super().__init__()
        self.canvas = (canvas.x0, canvas.y0, canvas.x0 + canvas.width, canvas.y0 + canvas.height)
        self.cut = (at.x0, at.y0, at.x0 + at.width, at.y0 + at.height)

    def __setitem__(self, option, v):
        if option == 'рамка':
            canvas, with_ = self.canvas, self.cut
            r = (max(v[0], canvas[0]), max(v[1], canvas[1]), min(v[2], canvas[2]), min(v[3], canvas[3]))
            if r[2] > r[0] and r[3] > r[1] and (r[0] < with_[0] or r[1] < with_[1] or r[2] > with_[2] or r[3] > with_[3]):
                raise TightRectangle(f'рамка мазка {r} шире среза {with_}')
        super().__setitem__(option, v)


class _TellFrame(dict):
    """Словарь «почему пусто» мазка, что идёт в процессе один раз, без счёта до рамки заранее: рамку, как только мазок
    её назвал, — главному процессу (по ней он пустит мазки после этого), и мазок кладётся дальше."""

    def __init__(self, tube, number):
        super().__init__()
        self.tube, self.number = tube, number

    def __setitem__(self, option, v):
        if option == 'рамка':
            self.tube.send_bytes(pickle.dumps({'номер': self.number, 'рамка': v, 'дальше': True}, pickle.HIGHEST_PROTOCOL))
        super().__setitem__(option, v)


class _Stroke:
    """Мазок очереди. рамка: `_НЕТ` — не считана; None — мазок кончился до рамки (холста не тронул); иначе — прямоугольник
    холста. до — что было на кисти перед мазком: подобранное или более ранний мазок очереди (тогда этот его ждёт);
    после — что на кисти после (когда досчитан). ось — у мягкой кисти по сырому: где её счёт до рамки читает холст."""
    __slots__ = ('очередь', 'номер', 'з', 'слои', 'слой', 'кисть', 'хозяин', 'до', 'после', 'сыро', 'ось', 'рамка', 'рамка_идёт',
                 'идёт', 'готов', 'ответ', 'метка', 'мешают', 'кому')      # мешают — более ранние недосчитанные мазки, чьи рамки пересекают эту; кому — кому мешает этот

    @property
    def waits_brush(self):
        return self.to if isinstance(self.to, _Stroke) and not self.to.ready else None


class Deferred:
    """Ответ `Холст.мазок` на ядрах: мазок принят (журнал, сид, счётчик — как на одном ядре), пиксели досчитаются
    позже. `взять()` — досчитать этот мазок и дать его ответ: записанный прямоугольник (x0, y0, x1, y1) или None
    (следа нет). Как кортеж — то же: `tuple(изм)`, `изм[0]`, `x0, y0, x1, y1 = изм`, `bool(изм)`, `изм == (…)`.
    Только `изм is None` не сработает: пустой мазок — `not изм` или `изм.взять() is None`."""
    __slots__ = ('_м',)

    def __init__(self, m):
        self._m = m

    def take(self):
        m = self._m
        while not m.ready:
            m.queue._accept()
        return m.response['изм']

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        measured = self.take()                 # 3г (К-1): замер следа — `изм.цвет`, `изм.закрыто`… — досчитает этот мазок
        if measured is None:
            raise AttributeError(f'мазок не оставил следа: его ответ — None, поля «{name}» нет (проверка — `not изм`)')
        return getattr(measured, name)

    def __str__(self):
        return str(self.take())

    def _tuple(self):
        measured = self.take()
        if measured is None:
            raise TypeError('мазок не оставил следа: его ответ — None (проверка — `not изм` или `изм.взять() is None`)')
        return measured

    def __bool__(self):
        return self.take() is not None

    def __iter__(self):
        return iter(self._tuple())

    def __len__(self):
        return len(self._tuple())

    def __getitem__(self, i):
        return self._tuple()[i]

    def __eq__(self, other):
        if isinstance(other, Deferred):
            other = other.take()
        return self.take() == other

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __repr__(self):
        return repr(self.take())


def _kind(canvas, p, layers, wet, target):
    """Холст-срез: прямоугольник `п` холста `х` без копии — мазок на нём пишет прямо в общую память, но только здесь."""
    from .canvas import Canvas, Layer
    from .tiles import Tiles
    gy, gx = canvas._region(p)
    if gy.stop <= gy.start or gx.stop <= gx.start:
        # Рамка (ось) мимо холста — мазок холста не трогает, но идёт ради кисти. На одном ядре он идёт на целом холсте и
        # читает под волосками по маске «внутри холста» с индексом 0 вместо чужих мест; у пустого среза индекса 0 нет.
        # Срез — один пиксель холста у угла: волоски вне холста — вне и его, читается он только под маской, не пишется.
        y0, x0 = min(gy.start, canvas.height - 1), min(gx.start, canvas.width - 1)
        gy, gx = slice(y0, y0 + 1), slice(x0, x0 + 1)
    at = Canvas.__new__(Canvas)
    at.__dict__.update(canvas.__dict__)
    at.x0, at.y0 = canvas.x0 + gx.start, canvas.y0 + gy.start
    at.width, at.height = gx.stop - gx.start, gy.stop - gy.start
    at._snapshots, at._repeats, at.journal, at._outside_journal, at._quiet, at._empty, at._cores, at._cores = [], None, None, False, 1, {}, None, 1
    at._hints = None
    at._layers = {}
    for name in layers:
        with_ = canvas._layers[name]
        n = Layer.__new__(Layer)
        for slot in _WithLResponseTY:
            a = getattr(with_, slot, None)
            if a is not None:
                setattr(n, slot, a.window(gy, gx) if isinstance(a, Tiles) else a[gy, gx])
        if name == target and wet is not None:
            n._wet = wet
        at._layers[name] = n
    return at


def _worker(canvas, tube, foreign=()):
    """Процесс: берёт мазок и считает его на срезе холста — до рамки (отдаёт рамку) или целиком (отдаёт, что стало с
    кистью). Мазок, кончившийся до рамки, досчитан уже первым счётом."""
    from .canvas import _Frame, _OnlyFrame
    for t in foreign:               # концы труб главного, взятые ветвлением: без них смерть главного видна как конец трубы
        t.close()
    while True:
        try:
            task = tube.recv_bytes()
        except (EOFError, OSError):
            return
        if task == b'':
            return
        number, to_frames, p, layers, wet, brush, z = pickle.loads(task)
        response = {'номер': number, 'до_рамки': to_frames}
        t0 = time.perf_counter()
        try:
            at = _kind(canvas, p, layers, wet, z['слой'])
            arrays = [(with_, slot, getattr(with_, slot, None)) for with_ in at._layers.values() for slot in _WithLResponseTY]
            why = _OnlyFrame() if to_frames else _TellFrame(tube, number) if p is None else _Why(canvas, at)
            brush._charged = brush._balance = _PREZhNEE
            with warnings.catch_warnings(record=True) as words:
                warnings.simplefilter('always')
                try:
                    measured, _ = at._lay(z, brush, why)
                except _Frame as e:
                    response['рамка'] = e.args[0]
                else:
                    for with_, slot, a in arrays:
                        if getattr(with_, slot, None) is not a:
                            raise TightRectangle(f'мазок завёл или подменил массив слоя «{slot}» в процессе')
                    response.update(measured=measured, why=dict(why), remainder=wet_field(brush, '_остаток'), charged=wet_field(brush, '_набрано'),
                                 balance=wet_field(brush, '_баланс'), words=[(str(w.message), w.category) for w in words])
        except BaseException as e:      # noqa: BLE001 — ошибка мазка идёт в главный процесс и звучит там
            response['ошибка'] = (e if isinstance(e, (ValueError, TightRectangle)) else None, traceback.format_exc())
        response['сек'] = time.perf_counter() - t0       # сколько шёл счёт: для замеров очереди
        try:
            tube.send_bytes(pickle.dumps(response, pickle.HIGHEST_PROTOCOL))
        except (BrokenPipeError, OSError):
            return


class Cores:
    """Очередь мазков холста на ядрах. `мазок(запись, кисть, журнал)` — мазок, принятый холстом (`Холст._принять_мазок`):
    считается, когда дойдёт черёд; `граница()` — после неё холст и кисти — как после тех же мазков на одном ядре.
    Повтор журнала даёт холсту `_готово(метка, изм, слова)` — зовётся в порядке мазков, когда мазок досчитан (метка —
    `холст._метка` на миг вызова мазка, слова — предупреждения мазка, [(текст, вид)])."""

    def __init__(self, canvas, cores=None):
        self.canvas = canvas
        self.cores = number_cores(cores)
        self.pid = os.getpid()
        self._waiting = []           # мазки по порядку вызовов: не отданные и идущие; досчитанные с головы уходят
        self._processes = []       # [процесс, труба, мазок или None]
        self._signature = None      # слои и их массивы, какими их видят процессы
        self._number = 0
        self._wet = {}          # слой -> (ys, xs): сырые пиксели-свидетели, которых не тронул ни один мазок слоя с тех пор
        self._tail = {}          # id(кисти) -> последний её мазок в очереди
        self._brushes = {}         # id(кисти) -> сколько её мазков в очереди
        self._broken = None
        self.strokes = self.at_once = 0     # сколько мазков ушло процессам и сколько из них — когда уже шёл другой

    # ------------------------------------------------------------ процессы
    def _layers_te_zhe(self):
        layers = self.canvas._layers
        if len(layers) != len(self._signature):
            return False
        for (name, with_), (was, layer, arrays) in zip(layers.items(), self._signature):
            if name != was or with_ is not layer or any(getattr(with_, slot, None) is not a for slot, a in zip(_WithLResponseTY, arrays)):
                return False
        return True

    def _pustit_processes(self):
        if self._processes:
            return
        from .canvas import _in_shared
        for with_ in self.canvas._layers.values():
            for slot in _WithLResponseTY:
                a = getattr(with_, slot, None)
                if a is not None:
                    setattr(with_, slot, _in_shared(a))
            with_._wait = self.boundary
        self._signature = [(name, with_, [getattr(with_, slot, None) for slot in _WithLResponseTY]) for name, with_ in self.canvas._layers.items()]
        environment = get_context('fork')
        for _ in range(self.cores):
            ours, its = environment.Pipe()
            p = environment.Process(target=_worker, args=(self.canvas, its, [z[1] for z in self._processes] + [ours]), daemon=True)
            p.start()
            its.close()
            self._processes.append([p, ours, None])

    def _stop(self):
        for p, tube, _ in self._processes:
            try:
                tube.send_bytes(b'')
            except (BrokenPipeError, OSError):
                pass
            tube.close()
        for p, _, _ in self._processes:
            p.join(5)
            if p.is_alive():
                p.kill()
        self._processes = []

    # --------------------------------------------------------------- мазок
    def stroke(self, z, brush, zh=None):
        """Поставить принятый мазок в очередь. Ответ — `Отложенный`; мазок, посчитанный сразу (первые `С_МАЗКА`, при
        живом снимке, новый массив слоя), — его ответ как на одном ядре."""
        canvas = self.canvas
        if self._broken:
            raise RuntimeError(self._broken)
        self._number += 1
        layer = z['слой']
        target = canvas._layers[layer]
        _, turbidity, even, _ = thin(z.get('разбавитель'), 0.0, getattr(z['краски'], 'мутность', 1.0), z['ровно'], None)
        if (self._number <= WITH_STROKE or z.get('живая') or canvas._snapshots or canvas.__dict__.get('_ленты') or (canvas.x0, canvas.y0, canvas.width, canvas.height) != (0, 0, canvas.full_width, canvas.full_height)
                or (turbidity < 1.0 and getattr(target, '_пропуск', None) is None)      # Г-1: разбавитель — тоже не кроющая и текучая
                or (even and getattr(target, '_текучая', None) is None)):
            return self._itself(z, brush, zh)                 # новый массив слоя, снимок холста — в главном; Н-3: и мазок по ленте скотча (затёк пишется ленте); Д-1: и ход живой краской — он начинает с остатка прошлого хода кисти, идёт по порядку
        if self._processes and not self._layers_te_zhe():     # новый слой или массив: процессам он не виден — завести их заново
            self.finish_computing(keep=True)
            self._stop()
        to = wet_field(brush, '_остаток')
        if isinstance(to, _Stroke) and to.queue is not self:
            to = brush._remainder                           # кисть с недосчитанным мазком другого холста: досчитать его
        m = _Stroke()
        m.queue, m.number, m.z, m.layer, m.label = self, self._number, z, layer, canvas.__dict__.get('_метка')
        m.layers = tuple(canvas._layers)
        m.frame, m.frame_goes, m.goes, m.ready, m.response, m.after, m.wet, m.axis = _NET, False, False, False, None, None, None, None
        m.mix, m.to_whom = set(), []
        if getattr(brush, 'кайма', 0.0) > 0.0 and brush.shape != 'мастихин':        # мягкая: есть ли в слое сырое вообще — ответ позже
            m.wet = _ZhDYoT
        m.to = to
        m.owner = brush
        m.brush = copy.copy(brush)                  # кисть как она есть на этом вызове; подобранное — перед счётом
        m.brush.__dict__.pop('_ждёт_ядра', None)
        m.brush._remainder = _PREZhNEE
        brush._remainder = m                          # до счёта: «что останется после этого мазка»
        brush.__dict__['_ждёт_ядра'] = self
        self._tail[id(brush)] = m
        self._brushes[id(brush)] = self._brushes.get(id(brush), 0) + 1
        if zh is not None:
            zh.after_stroke(brush)
        self._waiting.append(m)
        self._distribute()
        while len(self._waiting) >= ResponseHERED:
            self._accept()
        return Deferred(m)

    def _wet_li(self, i):
        """Ответить мягкой кисти мазка `_ждут[i]`, есть ли в его слое сырое на его черёд (шапка модуля). Ответ: дан ли."""
        m, canvas = self._waiting[i], self.canvas
        layer = m.layer
        earlier = [d for d in self._waiting[:i] if d.layer == layer and not d.ready]
        if any(d.frame is _NET for d in earlier):      # рамки более ранних мазков слоя ещё не все известны: свидетели под ними не сняты
            return False
        with_ = self._wet.get(layer)
        w = canvas._layers[layer]._moisture
        if with_ is None or not len(with_[0]):
            ys, xs = np.nonzero(np.array(w[::_WithETBrushA, ::_WithETBrushA]))      # копия: под рамками недосчитанных мазков слой сейчас пишется
            ys, xs = ys * _WithETBrushA, xs * _WithETBrushA
            for d in self._waiting:                      # под рамкой мазка слоя из очереди — ещё пишется или будет: не свидетель
                if d.layer == layer and d.frame is not _NET and d.frame is not None and not (d.ready and d.number < m.number) and len(ys):
                    r = d.frame
                    outside = (xs < r[0]) | (xs >= r[2]) | (ys < r[1]) | (ys >= r[3])
                    ys, xs = ys[outside], xs[outside]
            spacing = len(ys) // _WithInAndDETELEY + 1
            with_ = self._wet[layer] = (ys[::spacing], xs[::spacing])
        if len(with_[0]):
            m.wet = True
        elif earlier:                                  # свидетеля нет: ждать, пока более ранние мазки слоя досчитаются
            return False
        else:                                         # все мазки слоя до этого досчитаны, после него — не клались: слой как на его черёд
            m.wet = bool(w.any())
        if m.wet:                                    # по сырому мягкая кисть до рамки читает холст под волосками на оси хода
            from .path import _path
            brush = m.brush.in_pixels(canvas.px_on_mm)
            path = _path(m.z['точки'], brush, np.random.default_rng(m.z['сид']))
            r = 0.5 * float(brush.width) + 1.0
            m.axis = (max(int(np.floor(path['x'].min() - r)), 0), max(int(np.floor(path['y'].min() - r)), 0),
                     min(int(np.ceil(path['x'].max() + r)) + 1, canvas.width), min(int(np.ceil(path['y'].max() + r)) + 1, canvas.height))
            if m.axis[2] <= m.axis[0] or m.axis[3] <= m.axis[1]:
                m.axis = (0, 0, 0, 0)
        return True

    def _bind(self, m):
        """Рамка мазка стала известна: кто из более ранних недосчитанных ему мешает, кому из более поздних мешает он;
        сырые свидетели слоя под рамкой — сняты."""
        r = m.frame
        with_ = self._wet.get(m.layer)
        if with_ is not None and len(with_[0]):
            outside = (with_[1] < r[0]) | (with_[1] >= r[2]) | (with_[0] < r[1]) | (with_[0] >= r[3])
            self._wet[m.layer] = (with_[0][outside], with_[1][outside])
        if r[2] <= r[0] or r[3] <= r[1]:
            return
        r = by_grid(r)                  # 2в: мазки, что идут разом, плиток полей не делят (плитку заводит тот, кто в неё пишет)
        earlier = True
        for d in self._waiting:
            if d is m:
                earlier = False
                continue
            q = d.frame
            if d.ready or q is _NET or q is None:
                continue
            q = by_grid(q)
            if q[0] < r[2] and q[2] > r[0] and q[1] < r[3] and q[3] > r[1]:
                if earlier:
                    m.mix.add(d)
                    d.to_whom.append(m)
                else:
                    d.mix.add(m)
                    m.to_whom.append(d)

    def _itself(self, z, brush, zh):
        """Мазок в главном процессе, как на одном ядре (до него всё досчитано)."""
        self.finish_computing(keep=True)
        canvas = self.canvas
        ready = canvas.__dict__.get('_готово')
        why = {}
        try:
            with warnings.catch_warnings(record=True) as words:
                warnings.simplefilter('always')
                try:
                    measured, brush = canvas._lay(z, brush, why)
                finally:
                    if zh is not None:
                        zh.after_stroke(brush)
                from .canvas import write_frame
                write_frame(canvas, why.get('рамка'))
                canvas._account_empty(measured, z['точки'], brush, z['непрозрачность'], z['нагрузка'], why, z=z)
        finally:
            self._wet.clear()
        if ready is not None:
            ready(canvas.__dict__.get('_метка'), measured, [(w.message, w.category) for w in words])
        else:
            for w in words:
                warnings.warn(w.message, w.category, stacklevel=4)
        return measured

    def _task(self, m, to_frames):
        brush = m.brush
        if to_frames:
            p = m.axis if m.axis is not None else (0, 0, 0, 0)
            wet_to = _PREZhNEE
        else:
            p = None if m.frame is _NET else m.frame       # рамка не считана: мазок идёт один раз, на весь холст
            wet_to = m.to.after if isinstance(m.to, _Stroke) else m.to
        brush._remainder = wet_to
        return pickle.dumps((m.number, to_frames, p, m.layers, m.wet, brush, m.z), pickle.HIGHEST_PROTOCOL)

    def _distribute(self):
        """Отдать свободным процессам работу: счёт до рамки — мазкам по порядку; мазок — тому, чья рамка известна, как и
        рамки всех более ранних недосчитанных, и ни одной из них не пересекает."""
        if not self._waiting:
            return
        if not self._processes:
            self._pustit_processes()
        free = [z for z in self._processes if z[2] is None]
        if not free:
            return
        waiting = self._waiting
        n = len(waiting)
        known = n                                   # до этого места очереди рамки всех недосчитанных известны
        waiting_response = set()
        for i, m in enumerate(waiting):
            if m.ready or m.frame is not _NET:
                continue
            if known == n:
                known = i
            if m.frame_goes:
                continue
            if m.wet is _ZhDYoT and (m.layer in waiting_response or not self._wet_li(i)):      # ответы мягким кистям слоя — по порядку
                waiting_response.add(m.layer)
                continue
            if m.axis is not None:                      # читает холст по оси хода: все раньше него — с рамками, и ось свободна
                if known != i:
                    continue
                response = m.axis
                if any(not d.ready and d.frame is not None and d.frame[0] < response[2] and d.frame[2] > response[0] and d.frame[1] < response[3]
                       and d.frame[3] > response[1] for d in waiting[:i]):
                    continue
            z = free.pop()
            if all(d.ready for d in waiting[:i]):         # раньше него недосчитанных нет: сразу весь мазок, рамку скажет по дороге
                z[1].send_bytes(self._task(m, False))
                self.strokes += 1
                self.at_once += any(d[2] is not None and not d[2].frame_goes for d in self._processes)
                z[2], m.goes, m.frame_goes = m, True, True
            else:
                z[1].send_bytes(self._task(m, True))
                z[2], m.frame_goes = m, True
            if not free:
                return
        for i in range(known):
            m = waiting[i]
            if m.ready or m.goes or m.mix or m.frame is None or m.waits_brush is not None:
                continue
            z = free.pop()
            z[1].send_bytes(self._task(m, False))
            self.strokes += 1
            self.at_once += any(d[2] is not None and not d[2].frame_goes for d in self._processes)
            z[2], m.goes = m, True
            if not free:
                return

    def _finished(self, m):
        """Мазок досчитан: что осталось на кисти — кисти (если после него её не трогали) и журналу."""
        response = m.response['остаток']
        if isinstance(response, str) and response == _PREZhNEE:
            response = m.to.after if isinstance(m.to, _Stroke) else m.to
        m.after, m.goes, m.ready = response, False, True
        for d in m.to_whom:
            d.mix.discard(m)
        m.to_whom = []
        owner = m.owner
        if wet_field(owner, '_остаток') is m:
            owner._remainder = response
        zh = getattr(self.canvas, 'журнал', None)
        if zh is not None:
            zh.finished(owner, m, response)
        if self._tail.get(id(owner)) is m:
            del self._tail[id(owner)]

    def _accept(self):
        """Дождаться ответа процесса: рамка мазка или досчитанный мазок; сказать `готово` по порядку."""
        if self._broken:
            raise RuntimeError(self._broken)
        go = {z[1]: z for z in self._processes if z[2] is not None}
        if not go:
            self._distribute()
            go = {z[1]: z for z in self._processes if z[2] is not None}
            if not go:
                raise RuntimeError('ядра: очередь стоит — мазки ждут, а ни один не идёт')
        canvas = self.canvas
        for tube in wait(list(go)):
            z = go[tube]
            m = z[2]
            try:
                response = pickle.loads(tube.recv_bytes())
            except (EOFError, OSError):
                self._broken = f'ядра: процесс мазка {m.number} умер (память?) — холст недосчитан; писать заново с ядра=1'
                self._stop()
                raise RuntimeError(self._broken) from None
            if response.get('дальше'):                    # мазок, что идёт один раз, назвал рамку и кладётся дальше
                r = response['рамка']
                r = (max(r[0], 0), max(r[1], 0), min(r[2], canvas.width), min(r[3], canvas.height))
                m.frame, m.frame_goes = (r if r[2] > r[0] and r[3] > r[1] else (0, 0, 0, 0)), False
                self._bind(m)
                continue
            z[2] = None
            if 'ошибка' in response:
                e, trace = response['ошибка']
                m.frame, m.frame_goes, m.goes = None, False, False       # мазок с ошибкой следа не оставил: дальше очередь идёт без него
                m.response = dict(measured=None, why={}, remainder=_PREZhNEE, charged=_PREZhNEE, balance=_PREZhNEE, words=[], error=True)
                if isinstance(e, TightRectangle):
                    self._broken = f'{e}: на ядрах этот лист не бит в бит — считать с ядра=1 и сказать Движку'
                    self._stop()
                    raise TightRectangle(self._broken) from None
                if e is not None:
                    e.label = m.label
                    if m.label is None:           # письмо из Python: ошибка звучит позже вызова мазка — сказать, какого
                        e.args = (f'{e} — это мазок № {m.number} холста (слой «{m.layer}»), он считался на ядрах, ошибка звучит позже '
                                  'его вызова; с Холст(…, ядра=1) — сразу на нём',) + e.args[1:]
                    raise e
                self._broken = f'ядра: мазок {m.number} упал в процессе:\n{trace}'
                self._stop()
                raise RuntimeError(self._broken)
            if response['до_рамки']:
                m.frame_goes = False
                if 'рамка' in response:
                    r = response['рамка']
                    r = (max(r[0], 0), max(r[1], 0), min(r[2], canvas.width), min(r[3], canvas.height))
                    m.frame = r if r[2] > r[0] and r[3] > r[1] else (0, 0, 0, 0)
                    self._bind(m)
                else:                                 # мазок кончился до рамки: холста не тронул, досчитан
                    m.frame, m.response = None, response
            else:
                if m.frame is _NET:                   # шёл один раз и кончился до рамки: холста не тронул
                    m.frame, m.frame_goes = None, False
                m.response = response
                self._finished(m)
        for m in self._waiting:                          # кончившиеся до рамки: досчитаны, когда досчитан прошлый мазок их кисти
            if not m.ready and m.frame is None and m.response is not None and m.waits_brush is None:
                self._finished(m)
        ready = canvas.__dict__.get('_готово')
        while self._waiting and self._waiting[0].ready:
            m = self._waiting.pop(0)
            response = m.response
            self._brushes[id(m.owner)] -= 1
            if not self._brushes[id(m.owner)]:           # мазков этой кисти в очереди больше нет
                del self._brushes[id(m.owner)]
                if m.owner.__dict__.get('_ждёт_ядра') is self:
                    del m.owner.__dict__['_ждёт_ядра']
            for field in ('набрано', 'баланс'):       # что мазок оставил в кисти помимо остатка — в порядке вызовов
                if not (isinstance(response[field], str) and response[field] == _PREZhNEE):
                    setattr(m.owner, '_' + field, response[field])
            if response.get('ошибка'):
                continue
            if canvas.__dict__.get('_рамки') is not None:     # К-3: рамка мазка — повтору, по метке строки, в порядке мазков
                from .canvas import write_frame
                label, canvas._label = canvas.__dict__.get('_метка'), m.label
                write_frame(canvas, None if m.frame is None or m.frame is _NET else m.frame)
                canvas._label = label
            brush = m.brush.in_pixels(canvas.px_on_mm)
            with warnings.catch_warnings(record=True) as words:
                warnings.simplefilter('always')
                canvas._account_empty(response['изм'], m.z['точки'], brush, m.z['непрозрачность'], m.z['нагрузка'], response['почему'], z=m.z, at_once=False)
            words = list(response['слова']) + [(w.message, w.category) for w in words]
            if ready is not None:
                ready(m.label, response['изм'], words)
            else:
                for text, kind in words:
                    warnings.warn(text, kind, stacklevel=4)
        self._distribute()

    def brush_brushes(self, brush):
        """Досчитать мазки этой кисти: её подобранное, набранное и баланс — как после них на одном ядре."""
        if os.getpid() != self.pid:
            raise RuntimeError('кисть в процессе-потомке: при ветвлении её мазки ещё считались — до ветвления звать х.досчитать()')
        while self._brushes.get(id(brush)):          # набранное и баланс идут кисти по порядку вызовов: до её последнего мазка
            self._accept()
        if brush.__dict__.get('_ждёт_ядра') is self:
            del brush.__dict__['_ждёт_ядра']

    def finish_computing(self, keep=False):
        """Все мазки очереди досчитаны, кисти — со своим остатком. оставить=False — и процессы остановлены (заведутся
        со следующим мазком)."""
        while self._waiting:
            self._accept()
        if not keep:
            self._stop()

    close = finish_computing

    def boundary(self):
        """Перед тем, что читает или правит холст мимо мазка (картинка, массив слоя, сушка, снятие, сохранение): всё
        досчитано, счёт сырого слоёв забыт."""
        if os.getpid() != self.pid:
            if self._waiting:
                raise RuntimeError('холст в процессе-потомке: при ветвлении у родителя были недосчитанные мазки — до '
                                   'ветвления звать х.досчитать()')
            return
        if self._waiting:
            self.finish_computing(keep=True)
        if self._wet:
            self._wet.clear()
