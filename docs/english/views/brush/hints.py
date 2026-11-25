# READING VIEW of движок/кисть/подсказки.py with names in English. Not runnable; edit the original.
"""v4, 5б (В-40 «б», РЕШЕНИЕ DEN 18.11 12:35 +04): подсказки по ходу — факт холста или журнала числами и что под рукой.

Спецификация — `проверка/КАРТА_АБ_подсказки.md`, раздел «Б». Подсказка — строка в stderr (`warnings.warn(Подсказка(…))`,
как `ПустойМазок`) вида `подсказка: <факт> — под рукой: <инструмент>, <инструмент>[, …]`: факт без оценки, не меньше
двух инструментов, порядок их переставлен от сида листа и номера подсказки, что делать — не сказано. Всё сказанное и
отложенное — `х.подсказки` (`print(х.подсказки)`).

Только читает: пиксели, сиды и журнал от подсказок не зависят (журнал — продукт, подсказка в него не идёт). Счёт —
на мазок по уже известным полям (доводы мазка, `закрыто`, `цвет`, `сырое` ответа); по пикселям нового счёта нет.

Триггеры (пороги — спецификации):
    С1 «серия»            — в объекте (без пометки — в слое) ≥ 40 ходов одной кистью (поля без имени и ширины; ширина
                            ступенью вдвое), в одном секторе наклона 30° и одной нагрузкой (ступень 0,1); вытертая не в счёт;
    С3 «пуск этапа»       — этап пишется с того же сохранения (или журнал — с той же первой пометкой) в 3-й раз;
    С4 «глаза»            — с последнего `взгляд`, `виды`, `обзор` ≥ 1000 мазков, а холст выведен целиком (`png`,
                            `картинка` без области); раз на этап, на самом выводе; вывод целиком счёт не сбрасывает
                            (М-10 (б): С4 звучит на нём самом) — строка так и говорит;
    С5 «ответы»           — в объекте ≥ 10 мазков без следа, а ответ мазка прочитан меньше чем у 1 % ходов;
    С6 «первый мазок слоя» — первый мазок нового слоя: `закрыто` < 0,1; на своём мазке (на ядрах — на границе);
    С8 «по одному месту»  — мягкая кисть с краской ≥ 20 раз прошла по сырому своего слоя (`о.сырое['по сырому']` ≥ 0,5)
                            через одну плитку 64 px (центр плитки — под следом кисти); раз на слой;
    С10 «толстый слой», С11 «валы по границам», С12 «пастозно» — (v4, М-2) счёт по слоям, `подсказки_рельеф.py`; только
                            на `х.сохранить` и `х.закрыть` (холст досчитан), только слои с новыми мазками;
    С13 «грязь на кисти»  — (v4, М-4) кисть подобрала сырую краску чужого тона (`о.сырое['к положенному']` ≥ 0,1, тон
                            следа того хода от названного — ΔL ≥ 8 или Δa, b ≥ 0,02) и за `ГРЯЗЬ_ХОДОВ` следующих ходов,
                            не вытертая, легла дальше `ГРЯЗЬ_ШИРИН` ширин от места подбора — в объекте ≥ `ПОРОГ_ГРЯЗИ`
                            таких ходов; в факт — сколько из ходов этой кисти в объекте и тон следа против названного;
    С14 «однообразие места» — (v4, зеркало с адресом) у зоны (объект, без пометки — слой) с прошлой мерки ≥ 30 новых
                            ходов: журнал листа — местами (`однообразие.места_мазков`: зона × клетка 50 мм, от 30 ходов);
                            худшее место с оценкой ≥ 0,6 — рамкой и тем, что узко (кисть, нагрузка, ширина, длина, наклон,
                            вытирание), и сколько таких мест у зоны; холст без журнала — молчит;
    С17 «раскраска точками» — (v4, М-12) у зоны (объект, без пометки — слой) с прошлой мерки ≥ 30 новых ходов, а из всех её
                            ходов по журналу листа от `однообразие.ДОЛЯ_ТОЧЕК` (0,3) короче `однообразие.КОРОТКИЙ` (1,5)
                            своей ширины — сколько и длина/ширина p50; раз на зону; холст без журнала — молчит;
    С15 «масштаб»         — (v4, М-9) холст не на `мм.РОДНОЙ` (8) px/мм, первый мазок кистью, что держит px
                            (`масштаб.держит_px`: узкая уже 24 px здесь или на 8 px/мм, мягкая, сухая, кромка густой) — раз на
                            холст, на этом мазке (на ядрах — на границе): какие кисти и ширины уходят от настройки и чем эта;
    С16 «цвет формулой»   — (v4, М-1) из первых `ФОРМУЛА_ХОДОВ` ходов объекта (без пометки — слоя) с краской не с палитры
                            (цвет без имени замеса) и разных цветов среди них — от `ФОРМУЛА_ДОЛЯ`; раз на объект, на сотом
                            ходе (на ядрах — на границе); под рукой — `х.палитра_с`, `х.палитра`;
    С19 «сырое чужое», С20 «серый мягкой» — (v4, Г-8) на самом мазке (на ядрах — на границе) по `о.чужое` и `о.серый` ответа
                            (`встреча.py`: пороги и счёт): ход по сырому чужому цвету — подобрано и сдвиг тона; мягкая по
                            сухому в новом слое — доля пути тона; каждая — раз на этап (без этапа — на объект, без него —
                            на слой);
    С21 «под верхним слоем» — (v4, Сл-1) на самом мазке (на ядрах — на границе): ход лёг под краской слоёв выше и виден на
                            листе меньше чем на `под_верхом.ЗАКРЫТ` (0,25) следа (`о.видно`, `о.закрыл`); раз на этап, как С19;
    С9 «разброс цвета»    — (v4, 5г) в объекте ≥ 40 ходов с краской, а названные цвета соседних ходов (центр хода в одной
                            плитке 16 мм) разошлись по a, b OKLab не больше σ 0,010 и по L не больше σ 4 (серая шкала);
                            счёт — по названным цветам (довод `цвет` или набор), не по пикселям; в факт — и мерка
                            `х.манера(…, образец=)` этого листа «цвет в пятне» холста и образца, если её звали.

Когда: на границе — смена объекта или этапа пометкой, `х.сохранить`, `х.закрыть`, конец процесса; С3 — на пуске
(`Холст.загрузить(…, журнал=)` или первая пометка нового журнала), С4 и С6 — на своём вызове. Каждая пара «вид ×
объект (слой, этап)» звучит раз на лист; на границу — одна строка, старшая по порядку `СТАРШИНСТВО` (прочие
ждут, строка кончается «(ещё N: х.подсказки)»); на процесс — не больше трёх строк.

Память: у холста с журналом — файл `<журнал>.подсказки.json` (сказанное, ждущее, пуски, счёт на каждое сохранение:
`загрузить` с этапа берёт счёт того сохранения); файл стёрт — подсказки с начала. Холст без журнала помнит в себе.

Канал (М-2+М-4-2, РЕШЕНИЕ DEN 18.11 23:23 +04 «А»): сказанная строка идёт и в stdout (если stdout — не тот же файл, что
stderr: в терминале строка одна; в конце процесса — только stderr), а все новые с прошлой выдачи — сказанные и ждущие — полем `подсказки` ответа
`х.взгляд` и `х.сохранить` (каждая раз на лист, `Подсказки.новые`). Память сверяется со счётом мазков при пуске
(`загрузить(…, журнал=)` — счётчик сохранения, новый журнал — ноль) и на границе: ждущие и сказанные, чей факт сосчитан по
мазкам, которых в журнале больше нет, забываются (`забыть`; ключ остаётся — пара не звучит чаще прежнего).

Выключатель: `Холст(…, подсказки=False)`, окружение `KIST_HINTS=0`, на ходу `х.подсказки.выключить()` или один вид
`х.подсказки.выключить('серия')` (выключенное помнит файл листа); `warnings.filterwarnings('ignore', category=кисть.Подсказка)` в скрипте глушит только stderr (ключ
`-W ignore::кисть.Подсказка` Python отвергает: имя модуля не латиницей). Молчат всегда:
`кисть.повтор`, `х.проба`, участки `варианты` и `перебор`.
"""
import atexit
import copy
import json
import math
import os
import random
import re
import time
import warnings
import weakref

import numpy as np

from . import scale as M_

THRESHOLD_SERIES = 40        # С1
THRESHOLD_LAUNCH = 3         # С3
THRESHOLD_EYE = 1000       # С4
THRESHOLD_EMPTY = 10       # С5
SHARE_READING = 0.01      # С5: ответ прочитан меньше чем у стольких ходов объекта
THRESHOLD_COVERED = 0.1     # С6
THRESHOLD_PLACES = 20        # С8
BY_WET = 0.5         # С8
PLAndTBrushA = 64             # С8: px
BrushLETBrushA = 32             # С1: доля холста под серией — клетками столько px
SPREAD_SWEEPS = 40      # С9: ходов с краской в объекте
SPREAD_AB = 0.010      # С9: σ a, b OKLab (длина) названных цветов в плитке — не больше
SPREAD_L = 4.0         # С9: σ L серой шкалы — не больше
SPREAD_MM = 16.0       # С9: плитка соседства, мм (без px/мм — 128 px)
ON_PROCESS = 3
WithNAndMBrushResponseIn = 40            # счёт на столько последних сохранений в файле памяти
DIRT_BRUSH_LAID = 0.1   # С13: подобрано к положенному — подбор
DIRT_ΔL = 8.0          # С13: тон следа хода-подбора от названного: L серой шкалы
DIRT_ΔAB = 0.02        # С13: … или a, b OKLab (длина)
DIRT_SWEEPS = 3         # С13: подобранное видно в следе стольких следующих ходов (замер: ΔL 16 → 4 → 2 → 0)
DIRT_WIDTHS = 8.0       # С13: «далеко» — дальше стольких ширин кисти (и не ближе 64 px)
THRESHOLD_DIRT = 2         # С13: ходов с чужим подобранным в объекте
FORMULA_SWEEPS = 100     # С16: первые столько ходов объекта с краской
CUT_OFF_SHARE = 0.10     # С18 (М-3): ножом (`х.прижать`) срезано больше такой доли холста
FORMULA_SHARE = 0.8      # С16: из них не с палитры и разных цветов среди них — не меньше этой доли

InAndDY = {'С1': 'серия', 'С3': 'пуск этапа', 'С4': 'глаза', 'С5': 'ответы', 'С6': 'первый мазок слоя', 'С8': 'по одному месту',
        'С9': 'разброс цвета', 'С10': 'толстый слой', 'С11': 'валы по границам', 'С12': 'пастозно', 'С13': 'грязь на кисти',
        'С14': 'однообразие места', 'С15': 'масштаб', 'С16': 'цвет формулой', 'С17': 'раскраска точками', 'С18': 'срезано ножом',
        'С19': 'сырое чужое', 'С20': 'серый мягкой', 'С21': 'под верхним слоем'}
WithTARShAndNWithTInResponse = ('С21', 'С19', 'С20', 'С15', 'С3', 'С6', 'С16', 'С17', 'С13', 'С11', 'С10', 'С14', 'С1', 'С12', 'С18', 'С8', 'С9', 'С5', 'С4')
GLAZA = ('взгляд', 'виды', 'обзор', 'разность', 'переход', 'средний', 'окна', 'серая', 'темнее', 'просвет', 'борозды',
         'однообразие', 'регулярность', 'манера', 'перебор', 'сетка')
_WithMResponseTRYaT = ('взгляд', 'виды', 'обзор')           # С4: после них счёт мазков — с нуля
_PAPBrushA = os.path.dirname(os.path.abspath(__file__))
_in_end = [False]                                  # идёт `_конец_процесса`
_in_process = [0]                                   # строк, сказанных этим процессом
_live = weakref.WeakSet()


class Hint(UserWarning):
    """5б: подсказка по ходу — факт холста и что под рукой (`х.подсказки`, выключить — `Холст(…, подсказки=False)`)."""


_format_previous = warnings.formatwarning


def _format(message, category, filename, lineno, line=None):
    """5г (хвост 5б): в stderr подсказка — одной своей строкой `подсказка: …`, без «Подсказка:» перед ней и без строки
    исходника под ней; прочие предупреждения — как были."""
    if isinstance(category, type) and issubclass(category, Hint):
        return f'{message}\n'
    return _format_previous(message, category, filename, lineno, line)


warnings.formatwarning = _format


def enabled(hints=None):
    """Довод `подсказки=` холста: None — по окружению `KIST_HINTS` (0 — выключены), иначе как дано."""
    if hints is None:
        return os.environ.get('KIST_HINTS', '1').strip() not in ('0', 'нет', 'false', 'False', 'off')
    return bool(hints)


def _sbros_process():
    """Для тестов: счёт строк процесса — с нуля (новый процесс)."""
    _in_process[0] = 0


def _h(v, n=1):
    return f'{v:.{n}f}'.replace('.', ',')


def _z(text):
    """Слово ответа мазка — запятыми в числах, как все подсказки."""
    return re.sub(r'(?<=\d)\.(?=\d)', ',', text)


def _mn(n, one, two, five):
    n = int(n)
    if n % 10 == 1 and n % 100 != 11:
        return f'{n} {one}'
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return f'{n} {two}'
    return f'{n} {five}'


def _time(with_):
    return f'{_h(with_ / 60.0)} мин' if with_ >= 90 else f'{with_:.0f} с'


def _fresh_count():
    return {'о': {}, 'с': {}, 'после_глаз': 0, 'выводов': 0, 'к': {}, 'новые_слои': {}}


def _in_json(v):
    if isinstance(v, set):
        return sorted(v)
    raise TypeError(type(v))


def _from_json(with_):
    for response in with_.get('о', {}).values():
        for s in response['серии'].values():
            s[4] = set(s[4])
    return with_


class _Accounting:
    """Метка в ответе мазка: первое чтение его полей — в счёт «ответ прочитан» объекта (С5). В pickle не идёт."""
    __slots__ = ('п', 'объект', 'этап')

    def __init__(self, p, object, stage):
        self.p, self.object, self.stage = weakref.ref(p), object, stage

    def read(self):
        p = self.p()
        if p is not None:
            response = p.count['о'].get(self.object)
            if response is not None:
                response['прочтено'] += 1
            e = p.count.get('э', {}).get(self.stage)           # Св-1
            if e is not None:
                e['прочитано'] += 1
            p.taken('ответ')

    def __reduce__(self):
        return (type(None), ())


def function_taken(name):
    """Св-1: позвана функция вне холста (`нагрузка_для`, `смешение_для`, `мерка`) — инструмент подсказок в счёт."""
    for p in list(_live):
        if p.on and p._pid == os.getpid():
            p.taken(name)


class Hints:
    """`х.подсказки`: что сказано и что ждёт; `выключить()`, `выключить('серия')`. `print(х.подсказки)` — всё строками."""

    def __init__(self, canvas=None, on=True, journal=None, new=True):
        self._canvas = None if canvas is None else weakref.ref(canvas)
        self.on = bool(on)
        self.disabled = set()
        self.stated = []           # [{'вид', 'ключ', 'текст', 'счётчик'}]: счётчик — мазков холста, когда факт сосчитан
        self.waiting = []
        self.forgotten = []            # ключи забытых (`забыть`): пара не звучит снова
        self.released = set()         # ключи, уже отданные полем `подсказки` (`новые`)
        self.runs = {}             # ключ пуска -> {'n', 'прошлый': [мазков, с], 'идёт': [мазков, с]}
        self.snapshots = {}            # 'файл#счётчик' -> счёт на миг сохранения
        self.taken = {}             # Св-1: вызовов инструментов подсказок за лист (`взят`)
        self.eyes = {}             # вызовов каждого глаза и мерки за лист
        self.reference = None         # С9: «цвет в пятне» p50 [холст, образец] последней `х.манера(…, образец=)`
        self.count = _fresh_count()
        self.path = None
        self._journal = None         # С14: зеркало мест читает журнал листа
        self._launch = None           # (ключ, счётчик холста, время)
        self._first = False        # новый журнал: пуск считается с первой пометки
        if journal is not None and self.on:
            self._journal = os.path.abspath(os.fspath(journal))
            self.path = self._journal + '.подсказки.json'   # конец процесса может прийти из другой папки
            self._read()
            self._first = True
            if self.on and new:
                self.forgotten(0)          # журнал пишется с начала: всё сосчитанное по прежним мазкам — стёрто
        self._pid = os.getpid()     # процесс-ветвь (ядра, варианты) с копией холста молчит и файл не пишет
        if self.on:
            _live.add(self)

    # ------------------------------------------------------------ снаружи
    def vyklyuchit(self, kind=None):
        """Выключить все подсказки (вид None) или один вид: 'серия', 'пуск этапа', 'глаза', 'ответы', 'первый мазок
        слоя', 'по одному месту', …, 'срезано ножом' (или 'С1'…'С18'). У холста с журналом выключенное помнит файл листа."""
        if kind is None:
            self.on = False
        else:
            brush = kind if kind in InAndDY else {v: k for k, v in InAndDY.items()}.get(kind)
            if brush is None:
                raise ValueError(f'подсказки.выключить: вида {kind!r} нет (есть: {", ".join(InAndDY.values())})')
            self.disabled.add(brush)
            self.waiting = [p for p in self.waiting if p['вид'] != brush]
        self._write(always=True)

    def new(self):
        """Канал: тексты подсказок (сказанных и ждущих), которых ещё не было в поле `подсказки` ответа; каждая — раз на
        лист. Поле `подсказки` ответа `х.взгляд` и `х.сохранить`."""
        if not self.on:
            return []
        n = [p for p in self.stated + self.waiting if p['ключ'] not in self.released]
        if n:
            for p in n:
                p.setdefault('взято', self._how_much(p))
            self.released.update(p['ключ'] for p in n)
            self._write()
        return [p['текст'] for p in n]

    # ------------------------------------------------------------ Св-1: сводка этапа
    def taken(self, name):
        self.taken[name] = self.taken.get(name, 0) + 1

    def _how_much(self, p):
        from .summary import tools
        return sum(self.taken.get(and_, 0) for and_ in tools(p['текст']))

    def unanswered(self):
        """Сказанные (или отданные полем `подсказки`), после которых ни одного их инструмента «под рукой» не позвано."""
        return [p for p in self.stated + self.waiting if 'взято' in p and self._how_much(p) == p['взято']]

    def strokes_sheet(self):
        """Мазки журнала листа (дочитанного) и его мета: (мазки, мета); журнала нет или он не читается — None."""
        if self._journal is None:
            return None
        from . import monotony as Response
        d = self.__dict__.get('_дочитка')
        if d is None or d.path != self._journal:
            d = self.__dict__['_дочитка'] = Response.ReadRest(self._journal)
        try:
            return d.strokes(), d.meta
        except (OSError, ValueError):
            self.__dict__.pop('_дочитка', None)
            return None

    def summary(self, stage, aloud=False):
        """Св-1: сводка этапа (None — без этапа). вслух — и в stdout, если с прошлой сводки этапа легли новые мазки."""
        from . import summary as With
        option = '' if stage is None else str(stage)
        t = With.collect(self, option)
        if t and aloud:
            e = self.count['э'][option]
            if e['мазков'] > e['сказано']:
                e['сказано'] = e['мазков']
                try:
                    print(t, flush=True)
                except (OSError, ValueError):
                    pass
        return t

    def stage_summary(self, canvas):
        """Этап, о котором сводка сейчас: текущий; в нём мазков нет (этап дан у самих мазков) — этап последнего мазка, если
        о нём ещё не сказано."""
        e = self.count.get('э', {})
        option = '' if canvas._stage is None else str(canvas._stage)
        if not e.get(option, {}).get('мазков'):
            brush = self.count.get('э последний')
            if brush in e and e[brush]['мазков'] > e[brush]['сказано']:
                return brush
        return option

    def forgotten(self, counter):
        """Сверка с журналом: подсказки, сосчитанные после `счётчик` мазков (или без отметки счёта), — по мазкам, которых
        в журнале больше нет (перезапуск этапа с сохранения, лист с начала): из сказанных и ждущих уходят, ключ — в
        `забыты`; счёт на сохранения позже этого — тоже."""
        erased = lambda p: p.get('счётчик') is None or p['счётчик'] > counter
        for sp in (self.stated, self.waiting):
            for p in [p for p in sp if erased(p)]:
                sp.remove(p)
                if p['ключ'] not in self.forgotten:
                    self.forgotten.append(p['ключ'])
                self.released.discard(p['ключ'])
        for brush in [brush for brush in self.snapshots if int(brush.rsplit('#', 1)[1]) > counter]:
            del self.snapshots[brush]

    def __iter__(self):
        return iter([p['текст'] for p in self.stated + self.waiting])

    def __len__(self):
        return len(self.stated) + len(self.waiting)

    def __str__(self):
        if not self.on and not len(self):
            return 'подсказки выключены'
        h = [f'подсказки: сказано {len(self.stated)}, ждут {len(self.waiting)}' + ('' if self.on else ' (выключены)')
             + (f'; выключены виды: {", ".join(InAndDY[brush] for brush in sorted(self.disabled))}' if self.disabled else '')]
        h += ['  сказана: ' + p['текст'] for p in self.stated]
        h += ['  ждёт:    ' + p['текст'] for p in self.waiting]
        return '\n'.join(h)

    __repr__ = __str__

    # ------------------------------------------------------------ память
    def _read(self):
        try:
            with open(self.path, encoding='utf-8') as f:
                d = json.load(f)
        except (OSError, ValueError):
            return
        self.stated, self.waiting = list(d.get('сказано', [])), list(d.get('ждут', []))
        self.runs, self.eyes = dict(d.get('пуски', {})), dict(d.get('глаза', {}))
        self.reference = d.get('образец')
        self.taken = dict(d.get('взято', {}))
        self.snapshots = {brush: _from_json(v) for brush, v in d.get('снимки', {}).items()}
        self.disabled = set(d.get('выключены', []))
        self.forgotten, self.released = list(d.get('забыты', [])), set(d.get('отдано', []))
        if d.get('выключены_все'):
            self.on = False

    def _write(self, always=False):
        if self.path is None or (not self.on and not always) or self._pid != os.getpid():
            return
        d = {'сказано': self.stated, 'ждут': self.waiting, 'пуски': self.runs, 'глаза': self.eyes,
             'выключены': sorted(self.disabled), 'выключены_все': not self.on, 'снимки': self.snapshots, 'образец': self.reference,
             'забыты': self.forgotten, 'отдано': sorted(self.released), 'взято': self.taken}
        part = f'{self.path}.часть{os.getpid()}'
        try:
            with open(part, 'w', encoding='utf-8') as f:
                json.dump(d, f, ensure_ascii=False, default=_in_json, separators=(',', ':'))
            os.replace(part, self.path)
        except OSError:
            pass

    # ------------------------------------------------------------ служебное
    def _canvas(self):
        return None if self._canvas is None else self._canvas()

    def quiet(self, canvas):
        return not self.on or self._pid != os.getpid() or canvas.__dict__.get('_тишина', 0) or canvas.__dict__.get('_тихо', 0)

    def _add(self, kind, option, fact, tools):
        if kind in self.disabled or option in self.forgotten or any(p['ключ'] == option for p in self.stated + self.waiting):
            return
        canvas = self._canvas()
        and_ = list(tools)
        random.Random(f'{0 if canvas is None else canvas.seed}:{len(self)}').shuffle(and_)   # правило 3: первый не становится тропинкой
        self.waiting.append({'вид': kind, 'ключ': option, 'текст': f'подсказка: {fact} — под рукой: ' + ', '.join(and_),
                          'счётчик': 0 if canvas is None else int(canvas._counter)})

    def _say(self):
        """Граница: одна строка — старшая из ждущих; не больше `НА_ПРОЦЕСС` строк на процесс."""
        if not self.on or not self.waiting or _in_process[0] >= ON_PROCESS:
            return
        self.waiting.sort(key=lambda p: WithTARShAndNWithTInResponse.index(p['вид']))
        p = self.waiting.pop(0)
        self.stated.append(p)
        p.setdefault('взято', self._how_much(p))
        _in_process[0] += 1
        text = p['текст'] + (f' (ещё {len(self.waiting)}: х.подсказки)' if self.waiting else '')
        warnings.warn(Hint(text), stacklevel=2, skip_file_prefixes=(_PAPBrushA,))
        if not _in_end[0]:            # конец процесса: stdout скрипта уже кончился (его последнюю строку читают) —
            _in_stdout(text)           # там только stderr, а полю ответа строка достанется в следующем процессе

    @staticmethod
    def _name(option):
        return f'объект «{option[2:]}»' if option.startswith('о:') else f'слой «{option[2:]}»'

    @staticmethod
    def _selection(option):
        return f'объект={option[2:]!r}' if option.startswith('о:') else f'слой={option[2:]!r}'

    # ------------------------------------------------------------ мазок
    def stroke_accepted(self, canvas, points, brush, load, layer, object, new_layer, paints, name_brushes=None, mix=False, stage=None):
        """`Холст._принять_мазок`: С1 (серия), С4 (мазков без глаз); слои и места объекта (С10–С12); метка мазка для
        `мазок_лёг` (С5, С6, С8, С13)."""
        with_ = self.count
        with_['после_глаз'] += 1
        ko = ('о:' + str(object)) if object is not None else ('с:' + str(layer))
        response = with_['о'].setdefault(ko, {'ходов': 0, 'серии': {}, 'пустые': {}, 'прочтено': 0})
        response['ходов'] += 1
        n = brush.load if load is None else load
        try:
            wiped = brush.thickness_film(n) <= 0.0
        except Exception:                                    # noqa: BLE001 — нагрузка, которой мазок всё равно не примет
            wiped = False
        t = np.asarray(points, np.float64)
        t = t[:, :2] if t.ndim == 2 and t.shape[1] >= 2 else np.zeros((0, 2))
        sh = float(brush.width_mm * canvas.px_on_mm) if brush.width_mm is not None and canvas.px_on_mm else float(brush.width)
        soft = getattr(brush, 'кайма', 0.0) > 0.0 and brush.shape != 'мастихин'
        if not wiped and len(t):
            from dataclasses import asdict
            fields = {brush: v for brush, v in asdict(brush).items() if brush not in ('имя', 'ширина', 'ширина_мм')}
            chord = t[-1] - t[0]
            if math.hypot(*chord) > 1e-6:
                slant = math.degrees(math.atan2(chord[1], chord[0])) % 180.0
                sector = int(((slant + 15.0) % 180.0) // 30.0)
            else:
                sector = -1
            ng = f'{int(math.floor(float(n) / 0.1))}' if isinstance(n, (int, float)) else str(n)
            option = json.dumps([fields, int(math.floor(math.log2(max(sh, 1e-3)))), sector, ng], sort_keys=True, ensure_ascii=False)
            s = response['серии'].get(option)
            if s is None:
                word_ng = (f'нагрузкой {_h(int(ng) / 10)}–{_h(int(ng) / 10 + 0.1)}' if isinstance(n, (int, float))
                            else f'нагрузкой «{n}»')
                s = response['серии'][option] = [0, f'{"мягкая" if soft else brush.shape} {sh:.0f} px', sector, word_ng, set()]
            s[0] += 1
            kl = _cells(t, sh / 2.0, canvas.full_width, canvas.full_height)
            s[4].update(kl)
            if paints:                                         # М-2, С12: места и слои объекта
                nx = math.ceil(canvas.full_width / BrushLETBrushA)
                mask = int(response.get('места', '0'), 16)
                for c in kl:
                    mask |= 1 << ((c // 100000) * nx + c % 100000)
                response['места'] = format(mask, 'x')
                response.setdefault('слои', {})[str(layer)] = 1
                with_.setdefault('новые_слои', {})[str(layer)] = 1
        if paints and not wiped and len(t):                # С9: названный цвет хода — в плитку его центра
            t_mm = SPREAD_MM * canvas.px_on_mm if canvas.px_on_mm else 128.0
            ts = t.mean(axis=0)
            pl = response.setdefault('цвета', {}).setdefault(f'{int(ts[1] // t_mm)}:{int(ts[0] // t_mm)}', [0.0] * 7)
            response['с краской'] = response.get('с краской', 0) + 1
            for x, d in _colors(paints):
                pl[0] += d
                for i in range(3):
                    pl[1 + i] += d * x[i]
                    pl[4 + i] += d * x[i] * x[i]
        label = {'о': ko, 'слой': str(layer)}
        label['э'] = '' if stage is None else str(stage)       # Св-1
        if new_layer:
            self.taken('новый слой')
        label['этап'] = ('э:' + str(stage)) if stage is not None else ko      # Г-8: раз на этап
        if paints and not wiped and len(t) and response.get('формула') != 0:     # М-1, С16: первые 100 ходов с краской
            f = response.setdefault('формула', [0, 0, []])
            f[0] += 1
            if not mix:
                f[1] += 1
                option_colors = ':'.join(f'{float(v):.4f}' for c, d in paints for v in (*np.asarray(c).ravel(), d))
                if option_colors not in f[2]:
                    f[2].append(option_colors)
            if f[0] >= FORMULA_SWEEPS:
                different = len(f[2])
                if f[1] >= FORMULA_SHARE * f[0] and different >= FORMULA_SHARE * f[1]:
                    self._add('С16', 'С16:' + ko, fact_formulas(self._name(ko), f[0], f[1], different),
                                   TOOLS_FORMULAS + [f'х.однообразие({self._selection(ko)})'])
                    label['формула'] = 1
                response['формула'] = 0                               # раз на объект: первая сотня сосчитана
        if canvas.px_on_mm is not None and float(canvas.px_on_mm) != M_.RResponseDNResponseY and 'С15:холст' not in self._options():   # М-9
            what = M_.holds_px(brush, n, canvas.px_on_mm)
            if what:
                self._scale(canvas, brush, name_brushes, what, sh)
                label['масштаб'] = 1
        if new_layer and not wiped and paints:
            label['новый'] = _lightness(paints)
        if soft and not wiped and len(t):
            label['плитки'] = _tiles(t, sh / 2.0)
        if paints and not wiped and len(t):                # С13: кисть, место хода, названный тон
            from .tool import wet_field
            ts = _colors(paints)
            label['кисть'] = [id(brush), str(name_brushes or brush.name), wet_field(brush, '_остаток') is None]
            label['где'] = [float(t[:, 0].mean()), float(t[:, 1].mean()), max(DIRT_WIDTHS * sh, 64.0)]
            label['тон'] = [sum(x[i] * d for x, d in ts) for i in range(3)]
            brush = self.__dict__.setdefault('_кисти', {})
            z = brush.get(id(brush))
            if z is None or z[0]() is not brush:
                brush[id(brush)] = [weakref.ref(brush), []]
        return label

    def stroke_laid(self, canvas, label, measured, emptiness, at_once):
        """Мазок досчитан (`Холст._учесть_пустой`): С5 (пустые, чтение ответа), С6, С8."""
        with_ = self.count
        response = with_['о'].get(label['о'])
        from . import summary as With                             # Св-1
        e = with_.setdefault('э', {}).setdefault(label['э'], With.new())
        with_['э последний'] = label['э']
        With.account(e, measured, label.get('тон'), emptiness)
        if measured is None:
            if response is not None and emptiness is not None:
                response['пустые'][emptiness] = response['пустые'].get(emptiness, 0) + 1
        elif hasattr(measured, '__dict__') and 'закрыто' in measured.__dict__:
            measured.__dict__['_учёт'] = _Accounting(self, label['о'], label['э'])
        d = getattr(measured, '__dict__', {}) if measured is not None else {}
        encounter = self._encounter(canvas, label, d)
        if encounter and at_once:
            self._write()
            self._say()
        if 'новый' in label and 'закрыто' in d and 'серый' not in d:     # С6; мазок без следа говорит `ПустойМазок`, без замера (`замер=False`) — не судим
            covered = float(d['закрыто'])
            if covered < THRESHOLD_COVERED:
                layer = label['слой']
                color = d.get('цвет')
                wet = (d.get('сырое') or {}).get('по сырому', 0.0)
                tone = ('следа не видно' if color is None
                       else f'тон следа L {color["L"]:.0f} при названном {label["новый"]:.0f}')
                under = 'сырого под ним нет' if not wet else f'по сырому {_h(wet, 2)}'
                self._add('С6', 'С6:' + layer, f'первый мазок слоя «{layer}» — закрыто {_h(covered, 2)}, {tone}, {under}',
                               ['х.проба', 'print(о)', 'х.перебор(мазки, нагрузка=[…])', f'х.взгляд(область, без=[{layer!r}])'])
                if at_once:
                    self._write()
                    self._say()
        if ('масштаб' in label or 'формула' in label) and at_once:     # М-9: на первом мазке такой кисти; С16 — на сотом
            self._write()
            self._say()
        if 'кисть' in label:
            self._dirt(label, d if measured is not None else {})
        if 'плитки' in label and measured is not None and d.get('цвет') is not None:      # С8
            if (d.get('сырое') or {}).get('по сырому', 0.0) >= BY_WET:
                p = with_['с'].setdefault(label['слой'], {})
                L = round(float(d['цвет']['L']), 1)
                for t in label['плитки']:
                    brush = str(t)
                    z = p.get(brush)
                    if z is None:
                        p[brush] = [1, L, L]
                    else:
                        z[0] += 1
                        z[2] = L

    def _encounter(self, canvas, label, d):
        """Г-8: С19 и С20, Сл-1: С21 — по полям ответа `чужое` и `серый` (`встреча.разобрать`). True — добавлена новая."""
        from . import encounter as In
        was = len(self.waiting)
        if 'чужое' in d:
            layer = label['слой']
            self._add('С19', 'С19:' + label['этап'], f'слой «{layer}» — {_z(In.word_foreign(d["чужое"], True))}',
                           ['смешение_для(к, место, холст=х)', 'х.подсушить(…, область=)', 'х.мазок(…, слой=)'])
        if 'серый' in d:
            self._add('С20', 'С20:' + label['этап'], f'слой «{label["слой"]}» — {_z(In.word_gray(d["серый"], d["цвет"]["L"]))}',
                           ['нагрузка_для(к, покрытие=)', 'х.подсушить(…, область=)', 'х.проба'])
        if d.get('закрыл') is not None:
            from . import under_top as N
            if d['видно'] < N.ZABrushRYT:
                self._add('С21', 'С21:' + label['этап'], f'ход в слой «{label["слой"]}»: {_z(N.word(d["видно"], d["закрыл"]))}',
                               ['х.порядок_слоёв(имена)', 'х.мазок(…, слой=новый_сверху)', f'х.взгляд(область, без=[{d["закрыл"]!r}])'])
        return len(self.waiting) > was

    def _options(self):
        return {p['ключ'] for p in self.stated + self.waiting} | set(self.forgotten)

    def _scale(self, canvas, brush, name_brushes, what, sh):
        """С15 «масштаб» (М-9): холст не на `мм.РОДНОЙ` px/мм и первый мазок кистью, которая держит px (`масштаб.держит_px`):
        раз на холст — какие кисти и ширины здесь уходят от настройки и чем эта."""
        p = float(canvas.px_on_mm)
        h = lambda v: _h(v, 0) if abs(v - round(v)) < 0.05 else _h(v)
        name = str(name_brushes or getattr(brush, 'имя', '') or brush.shape)
        self._add('С15', 'С15:холст',
                       f'холст {h(p)} px/мм, физика на {M_.RResponseDNResponseY:.0f}: px держат узкая уже {h(M_.NARROW_PX / p)} мм '
                       f'({M_.NARROW_PX:.0f} px; при {M_.RResponseDNResponseY:.0f} px/мм — уже {h(M_.NARROW_PX / M_.RResponseDNResponseY)} мм), мягкая, сухая, '
                       f'кромка густой; «{name}» {h(sh / p)} мм = {sh:.0f} px: {", ".join(what)}',
                       ['х.проба(точки, кисть=к)', 'х.взгляд(область, масштаб=4)', 'х.просвет(область)'])

    def _dirt(self, label, d):
        """С13: подбор чужого тона кистью и ходы, что после него легли далеко (по полям ответа: `сырое`, `цвет`)."""
        id, name, clean = label['кисть']
        z = self.__dict__.get('_кисти', {}).get(id)
        if z is None:
            return
        pickups = z[1]
        if clean:                                       # вытерта (или новая) до этого хода
            if pickups:
                self.taken('кисть')                       # Св-1: инструменты С13 `к.вытереть()`, `к.с()`
            pickups.clear()
        cx, cy, far = label['где']
        L, a, b = label['тон']
        brush = self.count.setdefault('к', {}).setdefault(json.dumps([name, label['о']], ensure_ascii=False), [0, 0, 0.0, 0.0, 0.0])
        brush[1] += 1
        color = d.get('цвет')
        if color is not None:
            C, h = float(color['C']), math.radians(float(color['h']))
            dL, dab = abs(float(color['L']) - L), math.hypot(C * math.cos(h) - a, C * math.sin(h) - b)
        further = [math.hypot(p[0] - cx, p[1] - cy) for p in pickups]
        further = [r for r in further if r > far]
        if further and color is not None:
            brush[0] += 1
            brush[2], brush[3], brush[4] = max(brush[2], dL), max(brush[3], dab), max(brush[4], max(further))
        for p in pickups:
            p[2] += 1
        pickups[:] = [p for p in pickups if p[2] < DIRT_SWEEPS]
        if color is not None and (d.get('сырое') or {}).get('к положенному', 0.0) >= DIRT_BRUSH_LAID \
                and (dL >= DIRT_ΔL or dab >= DIRT_ΔAB):
            pickups.append([cx, cy, 0])

    # ------------------------------------------------------------ рельеф
    def relief(self, canvas):
        """М-2: С10–С12 по слоям, в которые с прошлого счёта легли мазки; `х.сохранить`, `х.закрыть` (холст досчитан)."""
        if not self.on or self._pid != os.getpid() or not {'С10', 'С11', 'С12'} - self.disabled:
            return
        new = self.count.get('новые_слои') or {}
        if not new:
            return
        from . import hints_relief as R
        was = {p['ключ'] for p in self.stated + self.waiting}
        layers = {s for s in new if s in canvas.layers and not {'С10:с:' + s, 'С11:с:' + s} <= was}
        nx = math.ceil(canvas.full_width / BrushLETBrushA)
        objects = {}
        for ko, response in self.count['о'].items():
            if 'места' in response and 'С12:' + ko not in was and any(s in new for s in response.get('слои', {})):
                mask = int(response['места'], 16)
                bits = np.nonzero(np.unpackbits(np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, 'little'), np.uint8),
                                                bitorder='little'))[0]
                objects[ko] = (list(response['слои']), set(((bits // nx) * 100000 + bits % nx).tolist()))
        for kind, option, fact, and_ in R.count(canvas, layers, objects):
            self._add(kind, option, fact, and_)
        self.count['новые_слои'] = {}

    def cut_off(self, canvas, cells):
        """М-3: С18 — какая доля холста срезана ножом (`х.прижать`). клетки — {номер клетки `КЛЕТКА` px: срезано px} этого
        хода; по клетке помнится наибольшее за ход (повторный ход по месту площадь не удваивает)."""
        if not self.on or self._pid != os.getpid() or 'С18' in self.disabled:
            return
        with_ = self.count.setdefault('срезано', {})
        for brush, n in cells.items():
            with_[str(brush)] = max(int(n), int(with_.get(str(brush), 0)))
        share = sum(with_.values()) / max(canvas.full_width * canvas.full_height, 1)
        if share > CUT_OFF_SHARE:
            self._add('С18', 'С18:холст', f'ножом (`х.прижать`) срезано {share:.0%} холста — срез гладкий, фактуры мазка на нём нет',
                           ['х.взгляд(область)', 'х.борозды(область)', 'х.png_высот(путь)'])

    # ------------------------------------------------------------ глаза
    def eye(self, name):
        self.eyes[name] = self.eyes.get(name, 0) + 1
        self.taken(name)
        if name in _WithMResponseTRYaT:
            self.count['после_глаз'] = 0
            self.count['выводов'] = 0

    def output(self, canvas):
        """Холст выведен целиком (`png`, `картинка` без области): С4 — на этом вызове."""
        with_ = self.count
        with_['выводов'] += 1
        if with_['после_глаз'] < THRESHOLD_EYE:
            return
        stage = canvas._stage
        count = ', '.join(f'{and_} {self.eyes.get(and_, 0)}' for and_ in GLAZA if and_ in ('взгляд', 'виды', 'обзор', 'разность')
                         or self.eyes.get(and_, 0))
        zeros = {'взгляд': 'х.взгляд(область, рядом=)', 'виды': 'х.виды(цель=)', 'обзор': 'х.обзор(цель)', 'разность': 'х.разность(цель)'}
        and_ = [in_ for brush, in_ in zeros.items() if not self.eyes.get(brush, 0)]
        for brush, in_ in zeros.items():
            if len(and_) >= 2:
                break
            if in_ not in and_:
                and_.append(in_)
        self._add('С4', f'С4:{stage}', f'{_mn(with_["после_глаз"], "мазок", "мазка", "мазков")} без взгляд, виды, обзор (png, картинка — не в '   # М-10 (б): что
                       f'счёт); выведен целиком {_mn(with_["выводов"], "раз", "раза", "раз")}; за лист {count}', and_)   # в счёте — словами
        self._write()
        self._say()

    def manner(self, m):
        """`х.манера(…, образец=)`: «цвет в пятне» p50 холста и образца — в факт С9."""
        try:
            self.reference = [float(m.canvas['цвет в пятне']['p50']), float(m.reference['цвет в пятне']['p50'])]
        except Exception:                                    # noqa: BLE001 — мерка без образца или без раздела
            return

    # ------------------------------------------------------------ пуски, сохранения
    def launch(self, canvas, option, from_where):
        """С3: `Холст.загрузить(…, журнал=)` (ключ — сохранение) или первая пометка нового журнала."""
        z = self.runs.setdefault(option, {'n': 0})
        z['n'] += 1
        if 'идёт' in z:
            z['прошлый'] = z['идёт']
        z['идёт'] = [0, 0.0]
        self._launch = (option, canvas._counter, time.monotonic())
        if z['n'] >= THRESHOLD_LAUNCH and 'прошлый' in z:
            m, seconds = z['прошлый']
            self._add('С3', 'С3:' + option, f'{from_where} пишется в {z["n"]}-й раз; прошлый пуск — '
                           f'{_mn(m, "мазок", "мазка", "мазков")}, {_time(seconds)} счёта',
                           ['х.перебор', 'х.варианты и х.взять', 'х.снимок и х.откатить', 'python -m кисть.повтор … --scale',
                            'python -m кисть.повтор … --resume'])
        self._write()
        self._say()

    def mark(self, canvas, text, stage, object, previous=None):
        if previous != canvas._stage:                          # Св-1: этап кончился — его сводка, до строк границы
            self.summary(previous, aloud=True)
        if self._first:
            self._first = False
            self.launch(canvas, 'начало#' + json.dumps([str(text), stage, object], ensure_ascii=False, default=str),
                      f'лист с первой пометкой «{text or stage or object}»')
        self.boundary(canvas)

    def loaded(self, canvas, path, counter):
        self.forgotten(counter)            # журнал обрезан по этому сохранению: сосчитанное по мазкам после него — стёрто
        name = os.path.basename(os.fspath(path))
        option = f'{name}#{counter}'
        with_ = self.snapshots.get(option)
        self.count = copy.deepcopy(with_) if with_ is not None else _fresh_count()
        self._first = False
        self.launch(canvas, option, f'с «{name}» лист')

    def saved(self, canvas, path):
        """Ответ — сводка этапа (Св-1): сказана до строк границы, по подсказкам, что звучали раньше неё."""
        summary = self.summary(self.stage_summary(canvas), aloud=True)
        self.relief(canvas)
        option = f'{os.path.basename(os.fspath(path))}#{canvas._counter}'
        self.snapshots.pop(option, None)
        self.snapshots[option] = copy.deepcopy(self.count)
        while len(self.snapshots) > WithNAndMBrushResponseIn:
            del self.snapshots[next(iter(self.snapshots))]
        self.boundary(canvas)
        return summary

    # ------------------------------------------------------------ С14: зеркало с адресом
    def _places(self):
        """С14 (зеркало с адресом, РЕШЕНИЕ DEN 18.11 23:33 +04): зоны (объект, без пометки — слой), у которых с прошлой
        мерки легло от `однообразие.МЕСТО_ХОДОВ` ходов, — по журналу листа местами (`однообразие.места_мазков`); худшее
        место зоны с оценкой от `однообразие.ПОРОГ_МЕСТА` — в подсказку. Холст без журнала — молчит."""
        if self._journal is None or ('С14' in self.disabled and 'С17' in self.disabled):
            return
        from . import monotony as Response
        was = {p['ключ'] for p in self.stated + self.waiting} | set(self.forgotten)
        needed = lambda kind, ko: kind not in self.disabled and f'{kind}:{ko}' not in was
        zones, zones14 = {}, set()
        for ko, response in self.count['о'].items():
            if (needed('С14', ko) or needed('С17', ko)) and response['ходов'] - response.get('мер', 0) >= Response.PLACE_SWEEPS:
                z = ('объект' if ko.startswith('о:') else 'слой', ko[2:])
                zones[z] = ko
                if needed('С14', ko):
                    zones14.add(z)
        if not zones:
            return
        r = self.strokes_sheet()
        if r is None:
            return
        strokes, meta = r
        for ko in zones.values():
            self.count['о'][ko]['мер'] = self.count['о'][ko]['ходов']
        self._points(Response, strokes, {z: ko for z, ko in zones.items() if needed('С17', ko)})
        if not zones14:
            return
        places, objects, _ = Response.places_strokes(strokes, Response._cell_px(meta, None), zones14)
        stated = set()
        for d in places:                                  # худшие первыми: первое место зоны — её худшее
            z = ('объект', d['объект']) if 'объект' in d else ('слой', d['слой'])
            if z in stated or d['оценка'] < Response.THRESHOLD_PLACES:
                continue
            stated.add(z)
            ko, ob = zones[z], objects[z]
            narrow = ([t for t in d['узко по'] if t[0] not in ('вытерта', 'точки')][:3]     # точки — своей подсказкой С17
                    + [t for t in d['узко по'] if t[0] == 'вытерта'])
            self._add('С14', 'С14:' + ko, f'{self._name(ko)}, рамка {tuple(d["рамка"])}: {d["мазков"]} ходов — '
                           + ', '.join(t[2] for t in narrow) + f'; мест от {_h(Response.THRESHOLD_PLACES)} — {ob["худших"]} из {ob["мест"]}',
                           ['х.однообразие(путь=…)', f'х.однообразие({self._selection(ko)})', 'х.виды(область)'])
            if any(p['ключ'] == 'С1:' + ko for p in self.stated + self.waiting):      # о серии уже говорит С1: адрес —
                p = next((p for p in self.waiting if p['ключ'] == 'С14:' + ko), None)   # полем `подсказки` и в
                if p is not None:                                                    # `х.подсказки`, без второй
                    self.waiting.remove(p)                                              # строки о той же зоне
                    self.stated.append(p)

    def _points(self, Response, strokes, zones):
        """С17 (М-12): зона (вид, имя) → ключ счёта; у зоны от `О.МЕСТО_ХОДОВ` ходов в журнале доля ходов короче
        `О.КОРОТКИЙ` своей ширины от `О.ДОЛЯ_ТОЧЕК` — факт тем же словом, что у зеркала."""
        if not zones:
            return
        gr = {}
        for m in strokes:
            z = Response._name_zones(m)
            if z in zones:
                gr.setdefault(z, []).append(m)
        for z, mm in gr.items():
            if len(mm) < Response.PLACE_SWEEPS:
                continue
            canvas = Response._sweep([Response._features_stroke(m) for m in mm])
            share = canvas['короче 1,5 ширины']
            if share < Response.SHARE_POINTS:
                continue
            ko = zones[z]
            self._add('С17', 'С17:' + ko, f'{self._name(ko)} — {Response.word_points(share)} (всего {len(mm)}), '
                           f'длина/ширина p50 {_h(canvas["длина/ширина"][1], 2)}',
                           [f'х.однообразие({self._selection(ko)})', 'х.однообразие(путь=…)', 'х.манера(область, образец=)'])

    # ------------------------------------------------------------ граница
    def boundary(self, canvas=None):
        """Смена объекта или этапа, сохранение, закрытие, конец процесса: С1, С5, С8 — по счёту; одна строка."""
        canvas = self._canvas() if canvas is None else canvas
        if canvas is None or not self.on or self._pid != os.getpid():
            return
        self.forgotten(canvas._counter)         # сверка: факт не может быть сосчитан по мазкам, которых у холста нет
        if self._launch is not None:
            brush, c0, t0 = self._launch
            if brush in self.runs:
                self.runs[brush]['идёт'] = [int(canvas._counter - c0), round(time.monotonic() - t0, 1)]
        for ko, response in self.count['о'].items():
            if response['серии']:
                s = max(response['серии'].values(), key=lambda s: s[0])
                if s[0] >= THRESHOLD_SERIES:
                    sector = 'без наклона' if s[2] < 0 else f'в секторе {s[2] * 30}°±15°'
                    cells = math.ceil(canvas.full_width / BrushLETBrushA) * math.ceil(canvas.full_height / BrushLETBrushA)
                    share = 100.0 * len(s[4]) / max(cells, 1)
                    self._add('С1', 'С1:' + ko, f'{self._name(ko)} — {_mn(response["ходов"], "ход", "хода", "ходов")}, из них {s[0]} '
                                   f'одной кистью ({s[1]}), {sector}, {s[3]}; под ними {share:.0f} % холста',
                                   [f'х.однообразие({self._selection(ko)})', 'х.виды(область)', 'х.манера(область, образец=)'])   # 5г: строка до ~220 знаков
            r = _spread(response)
            if r is not None and r[1] <= SPREAD_AB and r[2] <= SPREAD_L:
                at = ''
                if self.reference:
                    at = f'; цвет в пятне (х.манера): холст {_h(self.reference[0], 4)}, образец {_h(self.reference[1], 4)}'
                self._add('С9', 'С9:' + ko, f'{self._name(ko)} — {_mn(response["с краской"], "ход", "хода", "ходов")} с краской; '
                               f'σ названных цветов в плитке {SPREAD_MM:.0f} мм: a, b {_h(r[1], 3)}, L {_h(r[2])}{at}',
                               ['х.манера(область, образец=)', 'мерка(картинка)', 'х.разность(цель)'])
            empty = sum(response['пустые'].values())
            if empty >= THRESHOLD_EMPTY and response['прочтено'] < SHARE_READING * response['ходов']:
                views = sorted(response['пустые'].items(), key=lambda t: -t[1])[:2]
                self._add('С5', 'С5:' + ko, f'{self._name(ko)} — {_mn(empty, "мазок", "мазка", "мазков")} без следа ('
                               + ', '.join(f'{in_[:40]} {n}' for in_, n in views) + f'), ответ мазка прочитан у {response["прочтено"]} из {response["ходов"]}',
                               ['х.пустые', 'print(о)', 'о.разрывы', 'х.прямоугольник_мазка'])
        self._places()                   # после С1: о зоне, где С1 уже сказала серию, С14 строкой не звучит
        for option, brush in self.count.get('к', {}).items():
            if brush[0] >= THRESHOLD_DIRT:
                name, ko = json.loads(option)
                self._add('С13', 'С13:' + option, f'кисть «{name}», {self._name(ko)} — подобранное за {brush[4]:.0f} px отсюда '
                               f'в {_mn(brush[0], "ходе", "ходах", "ходах")} из {brush[1]}; тон следа против названного — до ΔL {_h(brush[2])}, '
                               f'Δa, b {_h(brush[3], 3)}', ['к.вытереть()', 'х.взгляд(область)', 'к.с()'])
        for layer, p in self.count['с'].items():
            if not p:
                continue
            brush, z = max(p.items(), key=lambda t: t[1][0])
            if z[0] >= THRESHOLD_PLACES:
                tx, ty = int(brush) % 100000, int(brush) // 100000
                from_ = sum(1 for v in p.values() if v[0] >= THRESHOLD_PLACES)
                self._add('С8', 'С8:' + layer, f'мягкая с краской по сырому слоя «{layer}» через одно место '     # 5г: строка до ~220 знаков
                               f'(плитка {PLAndTBrushA} px у {tx * PLAndTBrushA}, {ty * PLAndTBrushA}) {_mn(z[0], "раз", "раза", "раз")}, плиток от {THRESHOLD_PLACES} — {from_}; '
                               f'тон первого хода L {z[1]:.0f}, последнего L {z[2]:.0f}',
                               ['о.сырое', 'х.переход(область)', 'х.подсушить(…, область=)'])
        self._write()
        self._say()


TOOLS_FORMULAS = ['х.палитра_с(область, образец, n)', 'х.палитра({имя: цвет})']


def fact_formulas(zone, sweeps, not_with_palette, different):
    """С16 (М-1): факт «цвет формулой» — и подсказке, и зеркалу однообразия."""
    return (f'{zone} — из первых {_mn(sweeps, "хода", "ходов", "ходов")} с краской не с палитры {not_with_palette}, '
            f'цветов у них {different} разных: цвет формулой на каждый ход, не замесами')


def _in_stdout(text):
    """Канал: сказанная строка — и в stdout, если он не тот же файл, что stderr (иначе строка уже видна)."""
    import sys
    try:
        a, b = os.fstat(sys.stdout.fileno()), os.fstat(sys.stderr.fileno())
        if (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino):
            return
    except (AttributeError, OSError, ValueError):       # stdout подменён (тесты, блокнот) — пишем в него
        pass
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


def _colors(paints):
    """С9: названные краски хода [(линейный RGB, доля)] -> [((L серой шкалы, a, b OKLab), доля)], доли в сумме 1."""
    from .paint import from_linear
    from .eye import oklab, _L
    in_ = sum(float(d) for _, d in paints) or 1.0
    result = []
    for c, d in paints:
        lab = oklab(np.clip(from_linear(np.asarray(c, np.float64)) * 255.0, 0.0, 255.0))
        result.append(((float(_L(lab[0])), float(lab[1]), float(lab[2])), float(d) / in_))
    return result


def _spread(response):
    """С9: σ названных цветов внутри плиток соседства, сведённая по всем плиткам: (ходов с краской, σ a, b, σ L) или None
    (ходов меньше порога, или соседей в плитках меньше четверти ходов — ходы врозь, счёт не о соседях)."""
    n = response.get('с краской', 0)
    if n < SPREAD_SWEEPS:
        return None
    ss, st = np.zeros(3), 0.0
    for pl in response.get('цвета', {}).values():
        w = pl[0]
        if w <= 1.0 + 1e-9:
            continue
        sr = np.asarray(pl[1:4]) / w
        ss += np.maximum(np.asarray(pl[4:7]) - w * sr * sr, 0.0)
        st += w - 1.0
    if st < 0.25 * n:
        return None
    σ = np.sqrt(ss / st)
    return n, float(math.hypot(σ[1], σ[2])), float(σ[0])


def _lightness(paints):
    """Тон названной краски — L серой шкалы (как `цвет['L']` ответа)."""
    from .paint import from_linear
    from .eye import lightness
    try:
        lin = sum(np.asarray(c, np.float64) * float(d) for c, d in paints)
        return float(lightness(np.clip(from_linear(lin) * 255.0, 0.0, 255.0)))
    except Exception:                                        # noqa: BLE001
        return float('nan')


def _along(t, spacing):
    """Точки пути через не больше чем `шаг` px."""
    if len(t) < 2:
        return t
    d = np.hypot(*np.diff(t, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return t[:1]
    u = np.linspace(0.0, s[-1], max(2, int(math.ceil(s[-1] / spacing)) + 1))
    return np.column_stack([np.interp(u, s, t[:, 0]), np.interp(u, s, t[:, 1])])


def _cells(t, r, width, height=None):
    """Клетки `КЛЕТКА` px под путём шириной 2r (номер — y · 100000 + x)."""
    p = _along(t, BrushLETBrushA / 2.0)
    k = int(math.ceil(r / BrushLETBrushA))
    o = np.arange(-k, k + 1)
    cx = np.floor(p[:, 0] / BrushLETBrushA).astype(np.int64)
    cy = np.floor(p[:, 1] / BrushLETBrushA).astype(np.int64)
    X = (cx[:, None, None] + o[None, None, :]).repeat(len(o), 1)
    Y = (cy[:, None, None] + o[None, :, None]).repeat(len(o), 2)
    m = (X >= 0) & (Y >= 0) & (X * BrushLETBrushA < width) & ((Y * BrushLETBrushA < height) if height else True)
    return set(np.unique(Y[m] * 100000 + X[m]).tolist())


def _tiles(t, r):
    """Плитки `ПЛИТКА` px, через которые прошёл след кисти шириной 2r (центр плитки — не дальше r от пути)."""
    p = _along(t, PLAndTBrushA / 4.0)
    k = int(math.ceil(r / PLAndTBrushA)) + 1
    o = np.arange(-k, k + 1)
    cx = np.floor(p[:, 0] / PLAndTBrushA).astype(np.int64)
    cy = np.floor(p[:, 1] / PLAndTBrushA).astype(np.int64)
    X = (cx[:, None, None] + o[None, None, :]).repeat(len(o), 1)
    Y = (cy[:, None, None] + o[None, :, None]).repeat(len(o), 2)
    dx = (X + 0.5) * PLAndTBrushA - p[:, 0, None, None]
    dy = (Y + 0.5) * PLAndTBrushA - p[:, 1, None, None]
    m = (X >= 0) & (Y >= 0) & (dx * dx + dy * dy <= max(r, PLAndTBrushA / 2.0) ** 2)
    return sorted(set((Y[m] * 100000 + X[m]).tolist()))


@atexit.register
def _end_process():
    _in_end[0] = True
    for p in list(_live):
        try:
            p.boundary()
        except Exception:                                    # noqa: BLE001 — интерпретатор закрывается
            pass


InYBrushL = Hints(None, on=False)
