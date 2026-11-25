# READING VIEW of движок/кисть/варианты.py with names in English. Not runnable; edit the original.
"""2г, К-2: варианты куска — несколько попыток одного места разом, в нескольких процессах; выбранная ложится в холст
и в журнал своими мазками, как если бы кусок писался прямо в холст.

`х.варианты(кусок, n, область)`: функция `кусок(участок, i)` кладёт мазки на копии участка `область` (`Холст.участок`);
n копий считаются в процессах-ветвях (fork; не больше ядер холста и 16), каждая — со своим номером i, на холсте, каким
он был при вызове. Ответ — n `Вариант`: `картинка` (линейная, как `Холст.картинка`, — участок `область`) и `ответ` (что
вернула `кусок`). `х.взять(вариант)` кладёт выбранный: поля слоёв области (`вписать`), строки журнала, которые кусок
написал бы (те же байты), что стало с кистями (подобранное, набранное — и в журнале кисть помнит то же), счётчик и
сиды. Прочие варианты забыты. Повтор журнала после `взять` — тот же `.npz`, что у холста.

Почему это то же, что писать прямо: мазок читает и пишет холст только в своей рамке (`ядра.py`); рамка каждого мазка
куска сверяется с областью (вышла — ошибка `варианты` с подсказкой), а участок несёт ответ «есть ли сырое в слое вне
участка» (мягкая кисть, `Холст.участок`). Ветвь начинает с того, что было у родителя при вызове: кисти, журнал (копией
в памяти), сиды; что кусок изменил в кистях, ветвь узнаёт, сравнив все кисти процесса до и после куска.

Между `варианты` и `взять` холст не трогать: правка холста (мазок, сушка, откат, `вписать`…) делает варианты негодными
— `взять` скажет. Кисти между ними тоже не трогать (набрать, вытереть): `взять` вернёт им то, что было после куска.

М-8: `х.варианты(…, в_папку='варианты/фигура')` — каждый вариант ещё и на диск, в `<папка>/<i>/`: `вариант.pkl` (поля
слоёв области, строки журнала, счётчик и сиды, ответ, слова), `картинка.png` (sRGB 8 бит области — для глаз),
`журнал.jsonl` (строки куска, как лягут в журнал), `вариант.json` (что читать человеку и модели); в `<папка>/` —
`рядом.png` (холст сейчас и все варианты одним листом) и `варианты.json` (отпечаток холста). Выбор — в другом
процессе: `х = Холст.загрузить(<тот же .npz>, журнал=<тот же журнал>)`, `х.взять('<папка>/<i>')` — то же, что дал бы
кусок(х, i) прямо в этот холст: пиксели, строки журнала, `.npz` байт в байт. Холст не тот (отпечаток: размер, грунт,
зерно, сид, счётчик, этап и объект, повторы, поля слоёв в области и сырое вне её) или журнал не на том месте — ValueError словами, что не так.
Кисти другого процесса вариант не трогает: кусок для диска заводит или набирает свои кисти сам (грязь и краска,
что были на кистях до куска, в другой процесс не едут; в журнале варианта они — своими строками, повтор верен).
Журнал варианта на диске пишет кисти заново (строки `def`), как после `загрузить`: так байты те же, что у куска прямо
в загруженном холсте; `взять(вв[i])` в том же процессе кладёт те же строки.

С-1: `вариант.pkl` с диска читает `распаковка.прочесть` — код из файла не выполняется, папку можно брать и чужую; в
файле не данные движка — ValueError, что за имя. `взять(путь, своя=True)` — обычный pickle, для своей папки со своим
классом в ответе куска. Новое поле варианта своего класса движка — вписать класс в `распаковка._таблицы`.
"""
import copy
import gc
import hashlib
import io
import json
import os
import pickle
import shutil
import zlib
import warnings
from multiprocessing import get_context
from multiprocessing.connection import wait

import numpy as np

from . import unpickling
from .tool import Brush
from .tiles import Tiles

PResponseTResponseLResponseBrush = 16
_WithInResponseYo = '\0своё'           # в строке кисти журнала — то самое поле кисти, что у неё сейчас (тот же объект)
_NOT_TRANSFER = ('_ждёт_ядра',)


class Variant:
    """Один вариант куска: `номер` (i в `кусок(участок, i)`), `картинка` — линейная float32 (В, Ш, 3) области,
    `ответ` — что вернула `кусок`, `область`, `слова` — предупреждения, что звучали в куске (повторятся при `взять`),
    `однообразие` — К-12, зеркало однообразия мазков куска (`кисть.однообразие.Однообразие`: оценка, слова, сводка,
    регулярность; номера — строки куска с 1); считается всегда, и у холста без журнала."""
    __slots__ = ('номер', 'картинка', 'ответ', 'область', 'слова', 'однообразие', 'папка', '_данные', '_группа', '_диск')

    def __repr__(self):
        return f'Вариант({self.number}, область {self.region}' + (f', {self.folder}' if self.folder else '') + ')'


class _Group:
    __slots__ = ('холст', 'правок', 'кисти', 'взят', '__weakref__')


def _outward(a):
    if isinstance(a, Tiles):
        response, oy, ox = a.base()
        return (response, oy, ox, a.shape)
    return a


def _inward(z):
    if isinstance(z, tuple):
        response, oy, ox, shape = z
        return response.window(slice(oy, oy + shape[0]), slice(ox, ox + shape[1]))
    return z


def _brushes_process():
    return [o for o in gc.get_objects() if isinstance(o, Brush)]


def _cast(brush):
    """Поля кисти и их байты. Метка мазка очереди (`ядра._Мазок`: мазок кисти, что ещё считается или посчитан на ядрах
    какого-то холста) не пиклится — она тянет за собой очередь и её холст, у чужого холста с недосчитанными мазками
    это ошибка в ветви (`test_ядра` перед `test_варианты_2г` одним процессом); кусок её не правит, а заменяет —
    сравнивается по `is`."""
    from .cores import _Stroke
    d = dict(brush.__dict__)
    return d, {p: None if isinstance(v, _Stroke) else pickle.dumps(v, 5) for p, v in d.items() if p not in _NOT_TRANSFER}


def _changed(v, v0, b0):
    return v is not v0 or (b0 is not None and pickle.dumps(v, 5) != b0)


def _branch(canvas, piece, i, region, on_disk=False):
    """В процессе-ветви: кусок на участке, ответ — pickle всего, что нужно `взять`. на_диск (М-8) — реестр кистей
    журнала ветви пуст, как после `загрузить`: кисти куска названы заново своими строками."""
    from .canvas import Layer
    at = canvas.subarea(region)
    at.__dict__.pop('_правок', None)
    zh = getattr(canvas, 'журнал', None)
    mirror = None
    if zh is None:                            # К-12: холст без журнала — ветвь пишет строки в память только для зеркала
        from .journal import Journal
        mirror = Journal.__new__(Journal)
        mirror.path, mirror.pause, mirror.brushes, mirror.number, mirror.lines = None, 0, {}, 0, 0
        mirror.stage, mirror.object = canvas.__dict__.get('_этап'), canvas.__dict__.get('_объект')
        mirror._f = io.BytesIO()
        at.journal = mirror
    if zh is not None:                        # журнал ветви — копия журнала холста в памяти: строки идут родителю
        zh2 = copy.copy(zh)
        zh2._f = io.BytesIO()
        zh2.brushes = {} if on_disk else {k: list(z) for k, z in zh.brushes.items()}
        at.journal = zh2
        registry_to = {k: tuple(z) for k, z in zh2.brushes.items()}
    at._outside_journal = False
    at._only_in = tuple(region)
    at._empty = dict(getattr(canvas, '_пустые', {}))
    to = {id(brush): (brush, *_cast(brush)) for brush in _brushes_process()}
    with warnings.catch_warnings(record=True) as words:
        warnings.simplefilter('always')
        response = piece(at, i)
        at.finish_computing()
    if mirror is not None:
        at.journal = None
    response = _monotony(at.journal if mirror is None else mirror)
    brushes = []
    for oid, (brush, d0, b0) in to.items():
        d1 = brush.__dict__
        measured = {p: v for p, v in d1.items() if p not in _NOT_TRANSFER
               and (p not in d0 or _changed(v, d0[p], b0[p]))}
        removed = [p for p in d0 if p not in d1]
        if measured or removed:
            brushes.append((oid, measured, removed, None))
    lines = None
    if zh is not None:
        zh2 = at.journal
        lines = {'байты': zh2._f.getvalue(), 'номер': zh2.number, 'этап': zh2.stage, 'объект': zh2.object, 'кисти': []}
        for k, z in zh2.brushes.items():
            if k in registry_to and all(a is b for a, b in zip(z, registry_to[k])):
                continue
            brush = z[0]
            if k not in to:                  # кисть заведена в куске: едет целиком
                brushes.append((k, {}, [], brush))
            d = brush.__dict__
            lines['кисти'].append((k, z[1], z[2], _WithInResponseYo if z[3] is d.get('_остаток·') else z[3],
                                    _WithInResponseYo if z[4] is d.get('_краска') else z[4]))
    layers = {name: {p: _outward(getattr(with_, p)) for p in Layer._InWithE} for name, with_ in at._layers.items()}
    data = {'слои': layers, 'счётчик': at._counter, 'повторы': at._repeats, 'этап': at._stage, 'объект': at._object,
              'пустые': dict(getattr(at, '_пустые', {})), 'кисти': brushes, 'журнал': lines}
    try:
        response_b = pickle.dumps(response, 5)
    except Exception as e:                   # noqa: BLE001
        raise TypeError(f'варианты: то, что вернула кусок ({type(response).__name__}), не уходит из процесса (pickle): {e}; '
                        'возвращать числа, строки, массивы, словари') from None
    return pickle.dumps((at.image(), response_b, [(type(w.message).__name__, str(w.message)) for w in words],
                         pickle.dumps(data, 5), response), 5)


def _monotony(zh):
    """К-12: отчёт зеркала по строкам, что кусок написал в журнал ветви; кисти, заведённые до куска, — из реестра."""
    import json
    from dataclasses import asdict
    from .journal import _in_json
    from . import monotony as Response
    brushes = {}
    for z in zh.brushes.values():
        fields = z[2] if z[2] is not None else asdict(z[0])
        fields = _in_json(fields)
        brushes[z[1]] = (json.dumps({brush: v for brush, v in fields.items() if brush not in Response._BEZ}, sort_keys=True), fields.get('ширина'))
    return Response.report(Response.strokes_journal(zh._f.getvalue(), brushes), 'кусок')


# У-2: разведка — мазок, чей ход с запасом (ширин кисти и полей _ПОЛЕ_Р px) весь в области, рамкой в неё же ложится:
# на всех пресетах, ширинах 4–160 px, нажимах 0,1–1 рамка шире хода не больше 1,2 ширины и 10 px + 0,15 ширины, с
# наклоном 70° — 2,4 ширины; прочие — рамкой счётом мазка до места, где он берёт холст (как `прямоугольник_мазка`).
# Ошибётся запас — мазок шире области всё равно остановит вариант при счёте (`холст._ВОбласти`)
_ZAPAWith, _RESERVE_SLANT = 1.6, 3.0
_FIELD_R = 12.0


def _stroke_scouting(at, z, brush):
    from .canvas import _Frame, _InRegionToFrames
    x0, y0, x1, y1 = at._only_in
    t = np.asarray(z['точки'], np.float64)
    W = float(brush.in_pixels(at.px_on_mm).width)
    slant = t.shape[1] >= 5 and bool(np.any(t[:, 4] != 0.0))
    z_ = _FIELD_R + ((_RESERVE_SLANT if slant else _ZAPAWith) + 0.15) * W
    if t.size and x0 <= t[:, 0].min() - z_ and y0 <= t[:, 1].min() - z_ and t[:, 0].max() + z_ <= x1 and t[:, 1].max() + z_ <= y1:
        return None
    at.__dict__['_мазков_рамкой'] = at.__dict__.get('_мазков_рамкой', 0) + 1
    try:
        at._lay(z, brush, _InRegionToFrames(at._only_in, at.full_width, at.full_height))
    except _Frame:
        pass
    return None


def _scout(canvas, piece, n, region):
    """У-2, в процессе-ветви: все n кусков на участке без счёта — каждый мазок только рамкой против области. Ответ:
    None — все рамки в области (или кусок без мазков счёта не пройти: тогда скажут варианты); текст ошибки — первый
    мазок шире области."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(n):
            at = canvas.subarea(region)           # каждый вариант — с холста, каким он был при вызове: свои счёт сидов, пометка
            at.__dict__.pop('_правок', None)
            at.journal = None
            at._outside_journal = False
            at._only_in = tuple(region)
            at._scouting = _stroke_scouting
            at._quiet = 1
            try:
                piece(at, i)
            except ValueError as e:
                if 'шире области' in str(e) or str(e).startswith('варианты:'):
                    return i, str(e)
                return None
            except Exception:                # noqa: BLE001 — кусок, читающий холст, без счёта может пойти не так: скажут варианты
                return None
    return None


def _scout(canvas, piece, n, region, tube):
    try:
        tube.send_bytes(pickle.dumps(_scout(canvas, piece, n, region), 5))
    except BaseException:                    # noqa: BLE001
        pass
    finally:
        tube.close()


def _worker(canvas, piece, i, region, tube, on_disk=False):
    try:
        tube.send_bytes(b'+' + _branch(canvas, piece, i, region, on_disk))
    except BaseException as e:               # noqa: BLE001 — родителю: что упало и где
        import traceback
        try:
            tube.send_bytes(b'-' + pickle.dumps((type(e).__name__, str(e), traceback.format_exc()), 5))
        except Exception:                    # noqa: BLE001
            pass
    finally:
        tube.close()


def variants(canvas, piece, n, region, cores=None, in_folder=None):
    from . import cores as Ya
    n = int(n)
    if n < 1:
        raise ValueError('варианты: n — от 1')
    if not callable(piece):
        raise TypeError('варианты: кусок — функция кусок(участок, i)')
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    x0, y0 = max(x0, canvas.x0), max(y0, canvas.y0)
    x1, y1 = min(x1, canvas.x0 + canvas.width), min(y1, canvas.y0 + canvas.height)
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise ValueError(f'варианты: область {tuple(region)} вне холста')
    region = (x0, y0, x1, y1)
    if not Ya.allowed():
        raise RuntimeError('варианты: этот процесс — сам процесс счёта (демон), ветвей ему не завести')
    canvas.finish_computing()
    if cores is None:
        cores = canvas.__dict__.get('_ядер')
        cores = Ya.number_cores(None) if cores is None else cores
    k = max(1, min(int(cores), PResponseTResponseLResponseBrush, n))
    g = _Group()
    g.canvas, g.edits, g.taken = canvas, canvas.__dict__.get('_правок', 0), False
    g.brushes = {id(brush): brush for brush in _brushes_process()}     # держим: id кисти в ветви — тот же объект здесь
    environment = get_context('fork')
    scouting, its = environment.Pipe(duplex=False)  # У-2: рамки всех мазков куска — разведкой рядом со счётом: шире области — стоп сразу
    scout = environment.Process(target=_scout, args=(canvas, piece, n, region, its), daemon=True)
    scout.start()
    its.close()
    waiting, go, ready, error = list(range(n)), {}, {}, None
    while (waiting and error is None) or go:
        while waiting and len(go) < k - (scouting is not None and k >= PResponseTResponseLResponseBrush) and error is None:   # с разведкой — не больше 16 процессов
            i = waiting.pop(0)
            ours, its = environment.Pipe(duplex=False)
            p = environment.Process(target=_worker, args=(canvas, piece, i, region, its, in_folder is not None), daemon=True)
            p.start()
            its.close()
            go[ours] = (i, p, [])
        for tube in wait(list(go) + ([scouting] if scouting is not None else [])):
            if tube is scouting:
                try:
                    result = pickle.loads(scouting.recv_bytes())
                except EOFError:
                    result = None
                scouting.close()
                scout.join()
                scouting = None
                if result is not None:
                    error = (result[0], ('ValueError', result[1] + ' (рамки мазков сверены до счёта вариантов)', ''))
                    break
                continue
            i, p, pieces = go[tube]
            try:
                pieces.append(tube.recv_bytes())
            except EOFError:
                pass
            else:
                continue
            tube.close()
            p.join()
            del go[tube]
            if not pieces:
                error = error or (i, ('RuntimeError', f'процесс варианта упал (код {p.exitcode})', ''))
            elif pieces[0][:1] == b'-':
                error = error or (i, pickle.loads(pieces[0][1:]))
            else:
                ready[i] = pieces[0][1:]
            if error is not None:
                break
        if error is not None:               # У-2: прочие варианты не досчитывать — ошибка сразу
            for t, (_, p2, _) in list(go.items()):
                p2.terminate()
                p2.join()
                t.close()
            go.clear()
    if scouting is not None:
        scout.terminate()
        scout.join()
        scouting.close()
    if error is not None:
        i, (name, text, trace) = error
        type = {'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError}.get(name, RuntimeError)
        raise type(f'варианты: вариант {i}: {name}: {text}' + (f'\n--- в процессе варианта:\n{trace}' if trace else ''))
    response = []
    for i in range(n):
        image, response_b, words, data, response = pickle.loads(ready.pop(i))
        in_ = Variant()
        in_.number, in_.image, in_.response, in_.region, in_.words = i, image, pickle.loads(response_b), region, words
        in_.monotony, in_.folder, in_._disk = response, None, None
        in_._data, in_._group = data, g
        response.append(in_)
    identical = {}
    for in_ in response:
        identical.setdefault(in_.image.tobytes(), []).append(in_.number)
    if n > 1 and len(identical) < n:
        pairs = [v for v in identical.values() if len(v) > 1]
        warnings.warn(f'варианты: одинаковые картинки у вариантов {pairs} — кусок не различает i (сиды мазков — из '
                      'сида листа, этапа, объекта пометки и точек: разные i — разные точки, цвета или объект пометки)', stacklevel=3)
    if in_folder is not None:
        _on_disk(canvas, response, in_folder)
    return response


# ------------------------------------------------------------------ М-8: варианты на диске
FResponseRMAT = 2                  # вариант.pkl — pickle словаря, сжатый zlib; 2 (М-8-2) — отпечаток по области


def _hash(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode() + a.tobytes()).hexdigest()[:20]


def _json(v):
    return json.loads(json.dumps(v, ensure_ascii=False, default=str))


def imprint(canvas, region=None):
    """М-8: отпечаток холста — то, от чего зависят мазки куска и что `сохранить` кладёт в `.npz`: размер, грунт,
    масштаб, зерно, сид, счётчик мазков, этап и объект пометки, повторы сидов, слои по порядку и хеш каждого их поля.
    М-8-2: с `область` (x0, y0, x1, y1) поля слоёв хешируются только в ней — кусок вне области не читает и не пишет
    (рамки сверены до счёта), `взять` кладёт только её; вне области от слоя берётся одно — есть ли там сырое (ответ
    участка мягкой кисти, `Холст.участок`). Без `область` — поля целиком. Тот же холст после `сохранить` и
    `загрузить` — тот же отпечаток. Словарь (JSON)."""
    from .canvas import Layer
    canvas.finish_computing()
    repeats = sorted((canvas._repeats or {}).items())
    if region is None:
        layers = [[name, {p: (None if getattr(with_, p) is None else _hash(getattr(with_, p))) for p in Layer._InWithE}]
                for name, with_ in canvas.layers.items()]
    else:
        at = canvas.subarea(region)          # срезы плиток той же сетки — память и время только области
        layers = [[name, dict({p: (None if getattr(with_, p) is None else _hash(getattr(with_, p))) for p in Layer._InWithE},
                           wet_outside=bool(getattr(with_, '_сыро', False)))]
                for name, with_ in at.layers.items()]
    return _json({'размер': [canvas.full_width, canvas.full_height], 'грунт': list(canvas.ground_srgb), 'px_на_мм': canvas.px_on_mm,
                  'зерно': None if canvas.grain is None else list(canvas.grain), 'сид': canvas.seed, 'счётчик': canvas._counter,
                  'этап': canvas._stage, 'объект': canvas._object,
                  'повторы': _hash(np.array(repeats, np.uint64).reshape(-1, 2)), 'область': region and list(region),
                  'слои': layers})


_WithLResponseInA = {'размер': 'размер холста', 'грунт': 'грунт', 'px_на_мм': 'px_на_мм', 'зерно': 'зерно', 'сид': 'сид листа',
          'счётчик': 'счётчик мазков', 'этап': 'этап пометки', 'объект': 'объект пометки', 'повторы': 'повторы сидов'}


def _difference(was, has):
    """Что не так словами: у вариантов … — у холста …"""
    r = []
    for brush, word in _WithLResponseInA.items():
        if was.get(brush) != has.get(brush):
            r.append(f'{word}: у вариантов {was.get(brush)!r}, у холста {has.get(brush)!r}' if brush != 'повторы' else word + ' иные')
    sb, se = dict((and_, p) for and_, p in was['слои']), dict((and_, p) for and_, p in has['слои'])
    if list(sb) != list(se):
        r.append(f'слои: у вариантов {list(sb)}, у холста {list(se)}')
    for name in sb:
        if name in se and sb[name] != se[name]:
            r.append(f'слой «{name}»: иные поля ' + ', '.join('сырое вне области' if p == 'сыро_вне' else p
                                                           for p in sb[name] if sb[name][p] != se[name].get(p)))
    return r


def _write_atomically(path, bytes):
    with open(path + '.часть', 'wb') as f:
        f.write(bytes)
    os.replace(path + '.часть', path)


def _png(path, a8):
    from PIL import Image
    Image.fromarray(a8).save(path + '.часть.png')
    os.replace(path + '.часть.png', path)


def _brief(v, n=40):
    t = repr(v)
    return t if len(t) <= n else t[:n - 1] + '…'


def _on_disk(canvas, vv, in_folder):
    """М-8: варианты — в `<папка>/<i>/`, лист `рядом.png` и `варианты.json` — в папку."""
    from .eye import _in_8, _sheet, look
    from .paint import from_linear
    folder = os.path.abspath(os.path.expanduser(os.fspath(in_folder)))
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(folder):           # прежние варианты этой папки с номером ≥ n — убрать: лист и номера — одного вызова
        if name.isdigit() and int(name) >= len(vv) and os.path.isfile(os.path.join(folder, name, 'вариант.pkl')):
            shutil.rmtree(os.path.join(folder, name))
    release = imprint(canvas, vv[0].region)
    zh = getattr(canvas, 'журнал', None)
    journal_to = None if zh is None else _json({'кистей': zh.number, 'этап': zh.stage, 'объект': zh.object})
    empty_to = dict(getattr(canvas, '_пустые', {}))
    panels, signatures = [look(canvas, vv[0].region)], ['холст сейчас']
    for in_ in vv:
        where_to = os.path.join(folder, str(in_.number))
        os.makedirs(where_to, exist_ok=True)
        lines = pickle.loads(in_._data)['журнал']
        _write_atomically(os.path.join(where_to, 'вариант.pkl'), zlib.compress(pickle.dumps(
            {'формат': FResponseRMAT, 'номер': in_.number, 'область': in_.region, 'картинка': in_.image,
             'ответ': pickle.dumps(in_.response, 5), 'слова': in_.words, 'однообразие': in_.monotony, 'данные': in_._data,
             'отпечаток': release, 'журнал_до': journal_to, 'пустые_до': empty_to}, 5), 3))
        a8 = _in_8(np.clip(from_linear(in_.image) * 255.0, 0.0, 255.0))
        _png(os.path.join(where_to, 'картинка.png'), a8)
        zh_file = os.path.join(where_to, 'журнал.jsonl')
        if lines is not None:
            _write_atomically(zh_file, lines['байты'])
        elif os.path.exists(zh_file):
            os.remove(zh_file)
        response = in_.monotony
        _write_atomically(os.path.join(where_to, 'вариант.json'), json.dumps(_json(
            {'номер': in_.number, 'область': list(in_.region), 'ответ': in_.response, 'слова': [t for _, t in in_.words],
             'однообразие': None if response is None else round(float(response.score), 3),
             'строк_журнала': None if lines is None else lines['байты'].count(b'\n'),
             'взять': f"х = Холст.загрузить(<тот же .npz>, журнал=<тот же журнал>); х.взять({where_to!r})"}),
            ensure_ascii=False, indent=1).encode('utf-8'))
        in_.folder = where_to
        panels.append(a8)
        signatures.append(f'{in_.number}' + ('' if in_.response is None else ': ' + _brief(in_.response)))
    _sheet(panels, signatures, os.path.join(folder, 'рядом.png'))
    _write_atomically(os.path.join(folder, 'варианты.json'), json.dumps(_json(
        {'n': len(vv), 'область': list(vv[0].region), 'рядом': 'рядом.png', 'журнал_до': journal_to, 'отпечаток': release}),
        ensure_ascii=False, indent=1).encode('utf-8'))


def _with_disk(bytes, what, file, own):
    """С-1: pickle из файла варианта. Чужая папка (своя=False) — чтецом `распаковка`: только то, что движок сам пишет
    в вариант; иное имя или вызов в файле — ValueError до того, как оно выполнено. своя=True — обычный pickle (он
    выполняет то, что названо в файле): только для папки, что считал сам."""
    if own:
        return pickle.loads(bytes)
    try:
        return unpickling.read(bytes)
    except unpickling.Foreign as e:
        raise ValueError(
            f'взять: {file} — {what}: {e}; такого движок в вариант не пишет, файл не прочитан: обычное чтение pickle '
            'выполнило бы код из файла. Папку считал сам (кусок вернул объект своего класса или положил его в кисть) — '
            'х.взять(путь, своя=True); папка чужая (прислана, скачана, из архива) — не брать, посчитать варианты у себя'
        ) from None


def read(path, own=False):
    """М-8: вариант с диска (`<папка>/<i>` от `варианты(…, в_папку=)`) — `Вариант` с картинкой, ответом, словами и
    однообразием, как в том процессе; положить — `х.взять(путь)` или `х.взять(прочесть(путь))`.
    С-1: файл читается без выполнения кода из него — папку можно брать и чужую; в файле что-то кроме данных движка
    (числа, строки, массивы, словари, кисти, краски, ответы мазков) — ValueError, какое имя. своя=True — читать обычным
    pickle: для своей папки, где кусок вернул объект своего класса; чужой папке так не верить."""
    where_to = os.path.abspath(os.path.expanduser(os.fspath(path)))
    file = os.path.join(where_to, 'вариант.pkl') if os.path.isdir(where_to) else where_to
    if not os.path.isfile(file):
        raise FileNotFoundError(f'взять: варианта нет — {file} не найден (папка — `<в_папку>/<номер>`, как у '
                                'варианты(…, в_папку=))')
    with open(file, 'rb') as f:
        try:
            z = _with_disk(zlib.decompress(f.read()), 'в файле', file, own)
        except (zlib.error, pickle.UnpicklingError, EOFError):
            z = None
    if not isinstance(z, dict) or z.get('формат') != FResponseRMAT:
        raise ValueError(f'взять: {file} — не вариант этого движка (формат {z.get("формат") if isinstance(z, dict) else "?"}, '
                         f'нужен {FResponseRMAT}); посчитать варианты заново')
    in_ = Variant()
    try:
        in_.number, in_.region, in_.image, in_.words = z['номер'], tuple(z['область']), z['картинка'], z['слова']
        in_.response, in_.monotony, in_.folder = _with_disk(z['ответ'], 'в ответе куска', file, own), z['однообразие'], os.path.dirname(file)
        in_._data, in_._group = z['данные'], None
        in_._disk = {'отпечаток': z['отпечаток'], 'журнал_до': z['журнал_до'], 'пустые_до': z['пустые_до'],
                   'своя': bool(own), 'файл': file}
        if not isinstance(in_._data, bytes):
            raise TypeError('данные')
    except (KeyError, TypeError, pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f'взять: {file} — файл варианта не цел ({type(e).__name__}: {e}); посчитать варианты заново') from None
    return in_


def _in_canvas(canvas, in_, d):
    """Поля слоёв области и счётчик сидов варианта — в холст (`вписать`)."""
    from .canvas import Canvas, Layer
    at = Canvas.__new__(Canvas)
    x0, y0, x1, y1 = in_.region
    at.x0, at.y0, at.width, at.height = x0, y0, x1 - x0, y1 - y0
    at._layers = {}
    for name, fields in d['слои'].items():
        with_ = Layer.__new__(Layer)
        for p, z in fields.items():
            if z is not None or p in Layer._PResponseLYa:
                setattr(with_, p, _inward(z))
        at._layers[name] = with_
    at._counter, at._repeats, at._stage, at._object = d['счётчик'], d['повторы'], d['этап'], d['объект']
    at._outside_journal = False
    canvas.fit_in(at)


def _say(in_):
    from .canvas import EmptyStroke
    for kind, text in in_.words:
        warnings.warn(EmptyStroke(text) if kind == 'ПустойМазок' else UserWarning(text), stacklevel=4)


def _take_with_disk(canvas, in_):
    z = in_._disk
    diff = _difference(z['отпечаток'], imprint(canvas, in_.region))
    if diff:
        raise ValueError(f'взять: вариант {in_.folder} считан с другого холста — ' + '; '.join(diff) + '. Варианты на диск '
                         'считать с сохранённого холста (`сохранить`, затем `варианты(…, в_папку=)`) и брать в холст, '
                         '`загрузить`-нутый из того же .npz, до новых мазков; иначе — посчитать варианты заново')
    try:
        d = _with_disk(in_._data, 'в данных варианта (слои, кисти куска)', z['файл'], z['своя'])
    except (pickle.UnpicklingError, EOFError) as e:
        raise ValueError(f'взять: {z["файл"]} — файл варианта не цел ({e}); посчитать варианты заново') from None
    zh = getattr(canvas, 'журнал', None)
    lines, to = d['журнал'], z['журнал_до']
    if zh is not None:
        if lines is None:
            raise ValueError('взять: варианты считаны с холста без журнала, а этот холст пишет журнал — мазков варианта '
                             'в нём не будет и повтор разойдётся; посчитать варианты с холстом, что пишет журнал')
        has = _json({'кистей': zh.number, 'этап': zh.stage, 'объект': zh.object})
        if has != to:
            raise ValueError(f'взять: журнал холста не на том месте, где был при варианты (кистей названо, этап, объект: '
                             f'у вариантов {to}, у журнала {has}) — `загрузить` с тем журналом, что писал холст, и '
                             'брать до новых мазков и пометок')
    elif lines is not None:
        warnings.warn('взять: холст не пишет журнал (загружен без журнал=) — строк варианта в журнале не будет; писать '
                      'журнал дальше — Холст.загрузить(…, журнал=)', UserWarning, stacklevel=3)
    _in_canvas(canvas, in_, d)
    p = dict(getattr(canvas, '_пустые', {}))     # мазки без следа, что дал кусок, — к счёту этого холста
    for kind, k in d['пустые'].items():
        p[kind] = p.get(kind, 0) + k - z['пустые_до'].get(kind, 0)
    canvas._empty = {kind: k for kind, k in p.items() if k}
    if zh is not None:
        zh._f.write(lines['байты'])
        zh._f.flush()
        zh.lines += lines['байты'].count(b'\n')      # К-3: номер строки журнала; рамок мазков варианта холст не знает
        zh.number, zh.stage, zh.object = lines['номер'], lines['этап'], lines['объект']
    _say(in_)
    return None


def take(canvas, in_, own=False):
    if isinstance(in_, (str, os.PathLike)):
        in_ = read(in_, own)
    if not isinstance(in_, Variant):
        raise TypeError('взять: нужен вариант из х.варианты(…) или папка варианта на диске (`<в_папку>/<i>`)')
    if in_._group is None:
        return _take_with_disk(canvas, in_)
    g = in_._group
    if g.canvas is not canvas:
        if in_.folder is not None:
            raise ValueError(f'взять: вариант не этого холста — с диска его берут путём: х.взять({in_.folder!r})')
        raise ValueError('взять: вариант не этого холста')
    if g.taken:
        raise ValueError('взять: из этих вариантов уже взят один — прочие забыты; новые — х.варианты(…) заново')
    if canvas.__dict__.get('_правок', 0) != g.edits:
        raise ValueError('взять: холст менялся после варианты (мазок, сушка, откат, вписать…) — варианты считаны с '
                         'прежнего холста; посчитать их заново')
    d = pickle.loads(in_._data)
    g.taken = True
    _in_canvas(canvas, in_, d)
    canvas._empty = d['пустые']
    brushes = {}
    for oid, measured, removed, new in d['кисти']:
        brush = new if new is not None else g.brushes[oid]
        for p in removed:
            brush.__dict__.pop(p, None)
        brush.__dict__.update(measured)
        brushes[oid] = brush
    zh = getattr(canvas, 'журнал', None)
    if zh is not None and d['журнал'] is not None:
        with_ = d['журнал']
        zh._f.write(with_['байты'])
        zh._f.flush()
        zh.number, zh.stage, zh.object = with_['номер'], with_['этап'], with_['объект']
        for oid, name, fields, rest, paint in with_['кисти']:
            brush = brushes.get(oid) or g.brushes[oid]
            zh.brushes.pop(oid, None)
            zh.brushes[id(brush)] = [brush, name, fields, brush.__dict__.get('_остаток·') if isinstance(rest, str) and rest == _WithInResponseYo else rest,
                             brush.__dict__.get('_краска') if isinstance(paint, str) and paint == _WithInResponseYo else paint]
    g.brushes = {}
    _say(in_)
    return None
