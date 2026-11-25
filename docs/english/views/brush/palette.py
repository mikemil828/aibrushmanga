# READING VIEW of движок/кисть/палитра.py with names in English. Not runnable; edit the original.
"""v4, М-1: палитра замесов — именованные цвета холста; мазок берёт цвет именем замеса.

`х.палитра` — палитра холста (`Палитра`): `х.палитра({'охра свет': (214, 168, 96), 'тень': цвет(60, 0.03, 40)})` —
завести или поправить замесы разом, `х.палитра.добавить(имя, цвет)` — один; каждая правка — строка журнала
{"palette": имя, "color": [r, g, b], "paint": …} (этап и объект — от пометки), её исполняет повтор. Замес — любой цвет
мазка одной краской: (r, g, b), `Краска(rgb, вид)`, `цвет(L, C, h)`, `смешать_srgb(…)`.

Мазок именем: `цвет='охра свет'` в `мазок`, `проба`, `перебор`, `варианты` (у участка — та же палитра); список имён —
набор (смесь замесов на кисти, `раскладка` — как у набора); элементы списка — имя, (имя, доля), rgb, (rgb, доля).
`к.набрать(х.палитра['охра свет'])` или `х.палитра.набрать(к, 'охра свет')` — кисть набирает замес. Нет имени —
KeyError с ближайшими именами. Мазок именем ложится теми же пикселями, что тем же цветом напрямую: имя меняет только
журнал — строка мазка несёт и цвет, и "swatch" (имя или список имён, null — краска без имени), а повтор берёт цвет
из строки палитры выше: правка строки замеса и `--resume` пересобирают мазки этого замеса, остальные — бит в бит.

`х.палитра_с(область, образец, n)` — n замесов по образцу черновиком (`Черновик`: словарь «имя → (r, g, b)»,
доли площади, замощение образца замесами в прищуре); ничего не заводит и не кладёт. Считается так: образец в
области слегка размыт, k средних по площади в светлоте серой шкалы (/255) и a, b OKLab; начало — ступени
светлоты поровну от тёмного края до светлого, так тёмное и светлое пятно малой площади остаются своими замесами.

Палитра не меняет цвет мазка сама: оттенок даёт кисть (грязь, `разброс_тона`, сырое под ходом).
"""
import difflib
import math
import os

import numpy as np

from .paint import Paint


class Mix(Paint):
    """Цвет замеса палитры: `Краска` с именем (`имя`). Идёт везде, где цвет; журнал пишет рядом с цветом имя."""
    def __new__(cls, rgb, kind='кроющая', name=''):
        with_ = super().__new__(cls, rgb, kind)
        with_.name = str(name)
        return with_

    def __reduce__(self):
        return (Mix, (tuple(self), self.kind, self.name))

    def __repr__(self):
        return f'Замес({self.name!r}, ({", ".join(_number_line(v) for v in self)}))'


def _number_line(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))


def mix(name, color):
    """Цвет одной краской -> `Замес` с именем (вид краски — от `Краска`). Набор и не цвет — ValueError."""
    if isinstance(name, str) is False or not name.strip():
        raise ValueError(f'палитра: имя замеса — непустая строка, а не {name!r}')
    if isinstance(color, str):
        raise ValueError(f'палитра: замес «{name}» — цвет (r, g, b), Краска или цвет(L, C, h), а не имя {color!r}')
    try:
        ts = [float(v) for v in np.asarray(color, np.float64).ravel()]
    except (TypeError, ValueError):
        ts = []
    if len(ts) != 3 or np.ndim(color) != 1:
        raise ValueError(f'палитра: замес «{name}» — одна краска (r, g, b) 0–255, а не {color!r} (смесь двух замесов — '
                         'списком имён в мазке)')
    if not all(-1e-9 <= v <= 255.0 + 1e-9 for v in ts):
        raise ValueError(f'палитра: замес «{name}» — sRGB 0–255, а не {tuple(ts)}')
    return Mix(ts, getattr(color, 'вид', 'кроющая'), name)


def nearest(name, names, n=3):
    """Имена палитры, похожие на имя (для ошибки)."""
    names = list(names)
    b = difflib.get_close_matches(str(name), names, n=n, cutoff=0.4)
    if not b:
        b = [and_ for and_ in names if str(name).split()[0] in and_][:n] if str(name).split() else []
    return b


class NoMix(KeyError):
    """Имени нет в палитре холста (KeyError с ближайшими именами)."""
    def __str__(self):
        return str(self.args[0])


def from_file(e, canvas):
    """`НетЗамеса` у холста из `Холст.загрузить`: сказать, из какого `.npz` взята палитра (её замесы — какими были при
    `сохранить`; новые имена из скрипта там не появятся)."""
    from_where = canvas.__dict__.get('_загружен_из')
    if not from_where:
        return e
    return NoMix(f'{e}; палитра взята из {from_where} — какой была при сохранить; новые замесы — х.палитра({{…}}) после загрузки')


def _no(name, mixes):
    b = nearest(name, mixes)
    if not mixes:
        return NoMix(f'палитра: замеса «{name}» нет — палитра пуста (х.палитра({{имя: цвет}}), черновик — '
                         'х.палитра_с(область, образец, n))')
    return NoMix(f'палитра: замеса «{name}» нет' + (f'; ближайшие: {", ".join(repr(and_) for and_ in b)}' if b else '')
                     + f' (в палитре {len(mixes)}: х.палитра)')


def has_names(color):
    """В цвете мазка есть имя замеса (строка или список с именами)."""
    if isinstance(color, str):
        return True
    if isinstance(color, (list, tuple)) and not isinstance(color, Paint):
        for e in color:
            if isinstance(e, str) or (isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], str)):
                return True
    return False


def resolve(color, mixes):
    """Цвет мазка с именами замесов -> тот же цвет замесами (`Замес`); без имён — как есть (тот же объект)."""
    if not has_names(color):
        return color

    def one(name):
        z = mixes.get(name)
        if z is None:
            raise _no(name, mixes)
        return z
    if isinstance(color, str):
        return one(color)
    result = []
    for e in color:
        if isinstance(e, str):
            result.append(one(e))
        elif isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], str):
            result.append((one(e[0]), e[1]))
        else:
            result.append(e)
    return result


def names_colors(color):
    """Имя замеса одной краски (str), список имён набора по краскам (None — краска без имени) или None — имён нет."""
    if color is None:
        return None
    if isinstance(color, Mix):
        return color.name
    if isinstance(color, Paint):
        return None
    ts = list(color)
    from .tool import _number
    if ts and _number(ts[0]):
        return None
    names = []
    for e in ts:
        if isinstance(e, Mix):
            names.append(e.name)
        elif isinstance(e, (list, tuple)) and len(e) == 2 and isinstance(e[0], Mix):
            names.append(e[0].name)
        else:
            names.append(None)
    return names if any(and_ is not None for and_ in names) else None


def color_from_lines(z, mixes, color):
    """Повтор: цвет строки мазка или набора с "swatch" — замесы палитры повтора по имени; доли и краски без имени — из
    строки (`цвет` — то, что прочёл `повтор._цвет`)."""
    names = z.get('swatch')
    if names is None:
        return color
    if isinstance(names, str):
        if names not in mixes:
            raise _no(names, mixes)
        return mixes[names]
    if not isinstance(color, list) or len(color) != len(names):
        raise ValueError(f'"swatch" списком — по имени (или null) на каждую краску "colors" ({len(names)} имён)')
    result = []
    for e, name in zip(color, names):
        if name is None:
            result.append(e)
            continue
        if name not in mixes:
            raise _no(name, mixes)
        if isinstance(e, (list, tuple)) and len(e) == 2 and not isinstance(e[0], (int, float)):
            share = e[1]
        elif isinstance(e, (list, tuple)) and len(e) == 4:
            share = e[3]
        else:
            share = 1.0
        result.append((mixes[name], share))
    return result


class Palette:
    """`х.палитра`: замесы холста по имени. `х.палитра({имя: цвет, …})` — завести или поправить разом (строки журнала),
    `добавить(имя, цвет)` — один, `х.палитра['имя']` — `Замес` (годится в `мазок`, `к.набрать`), `набрать(к, имя
    или [имена])` — кисть набирает замес; `имена`, `len`, `in`; `print(х.палитра)` — таблица: имя, rgb, L C h."""

    def __init__(self, canvas):
        self._canvas = canvas

    @property
    def _mixes(self):
        return self._canvas.__dict__.setdefault('_замесы', {})

    def __call__(self, mixes, stage=None, object=None):
        if not isinstance(mixes, dict):
            raise ValueError('палитра: замесы — словарь {имя: цвет}')
        ready = [mix(name, ts) for name, ts in mixes.items()]     # сперва все проверены: кривой — ни один не заведён
        for z in ready:
            self._create(z, stage, object)
        return self

    def add(self, name, color, stage=None, object=None):
        """Завести замес или поправить: строка журнала {"palette": имя, "color": …}; мазки после неё — новым цветом,
        легшие раньше — прежним (повтор берёт цвет из строки выше мазка)."""
        self._create(mix(name, color), stage, object)
        return self._mixes[str(name)]

    def _create(self, z, stage, object):
        canvas = self._canvas
        self._mixes[z.name] = z          # мазки, принятые раньше, уже взяли свой цвет: досчитывать их не нужно
        zh = canvas._writes()
        if zh is not None:
            from .journal import color_lines
            d = {'palette': z.name}
            fields = color_lines(Paint(tuple(z), z.kind))
            d.update(fields)
            zh.row(zh._intent(d, stage, object))

    def __getitem__(self, name):
        z = self._mixes.get(name)
        if z is None:
            raise from_file(_no(name, self._mixes), self._canvas)
        return z

    def __contains__(self, name):
        return name in self._mixes

    def __len__(self):
        return len(self._mixes)

    def __iter__(self):
        return iter(list(self._mixes))

    @property
    def names(self):
        return list(self._mixes)

    def items(self):
        return list(self._mixes.items())

    def charge(self, brush, color, amount=None, layout='полосы'):
        """`кисть.набрать` именем замеса или списком имён (набор); количество (Д-1) — доля полной кисти, краска живая
        (то же — `х.набрать(кисть, цвет, количество)`)."""
        if isinstance(amount, str):       # прежний вызов: третий довод был раскладкой
            amount, layout = None, amount
        try:
            color = resolve(color, self._mixes)
        except NoMix as e:
            raise from_file(e, self._canvas) from None
        return brush.charge(color, amount, layout)

    def __repr__(self):
        if not self._mixes:
            return 'палитра пуста: х.палитра({имя: цвет}); черновик по образцу — х.палитра_с(область, образец, n)'
        from .eye import lch
        with_ = [f'палитра: {len(self._mixes)} замесов']
        for name, z in self._mixes.items():
            L, C, h = lch(np.asarray(tuple(z)))
            kind = '' if z.kind == 'кроющая' else f', {z.kind}'
            with_.append(f'  {name}: ({", ".join(_number_line(round(v, 2)) for v in z)}) L {L:.0f} C {C:.3f} h {h:.0f}{kind}')
        return '\n'.join(with_)

    __str__ = __repr__


# ---------------------------------------------------------------- черновик по образцу

WEIGHT_L = 1.0          # светлота серой шкалы / 255 против a, b OKLab: тон и оттенок поровну
TResponseHEBrush = 40000        # образец для счёта — не больше стольких пикселей (уменьшен средним)
AndTERATsAndY = 40


class Draft(dict):
    """Ответ `палитра_с`: словарь «имя → (r, g, b)», светлые первыми (замеса, которому в образце не досталось площади, —
    нет: замесов бывает меньше n); `доли` — доля площади области у каждого,
    `замощение` — (В, Ш, 3) образец области, где каждый пиксель — ближайший замес, `прищур_dL` — средняя разность
    тона образца и замощения в прищуре (как `х.виды`), L; `лист` — uint8 «образец | замощение | плашки» (`путь` —
    PNG). В палитру — `х.палитра(ч)` или своей правкой."""

    def __repr__(self):
        from .eye import lch
        with_ = [f'черновик палитры: {len(self)} замесов, замощение образца в прищуре |dL| {self.squint_dL:.1f}'.replace('.', ',')]
        for name, ts in self.items():
            L, C, h = lch(np.asarray(ts, np.float64))
            with_.append(f'  {name!r}: {tuple(ts)},  L {L:.0f} C {C:.3f} h {h:.0f}, площадь {100 * self.shares[name]:.0f} %')
        return '\n'.join(with_)

    __str__ = __repr__


_TResponseNA = ((20, 'красный'), (45, 'оранжевый'), (75, 'охристый'), (110, 'жёлтый'), (165, 'зелёный'), (215, 'бирюзовый'),
         (290, 'синий'), (335, 'фиолетовый'), (361, 'красный'))


def name_colors(rgb):
    """Имя по тону и оттенку: «тёмный серо-синий 54» (число — L серой шкалы)."""
    from .eye import lch
    L, C, h = lch(np.asarray(rgb, np.float64))
    light = 'тёмный' if L < 85 else 'средний' if L < 170 else 'светлый'
    if C < 0.025:
        hue = 'серый тёплый' if (C > 0.008 and (h < 110 or h > 335)) else 'серый холодный' if C > 0.008 else 'серый'
    else:
        hue = next(word for limit, word in _TResponseNA if h < limit)
        if C < 0.06:
            hue = 'серо-' + hue
    return f'{light} {hue} {L:.0f}'


def _reduce(a, k):
    if k <= 1:
        return a
    in_, sh = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:in_, :sh].reshape(in_ // k, k, sh // k, k, a.shape[2]).mean(axis=(1, 3))


def _squint_L(rgb):
    """L прищура, как у `х.виды`: светлота, трижды размытая на ~1/120 длинной стороны."""
    from .eye import oklab, _L, _blur
    L = _L(oklab(rgb)[..., 0])
    r = max(1, int(round(max(L.shape) / 120)))
    for _ in range(3):
        L = _blur(L, r)
    return L


def _in_points(rgb):
    """sRGB -> точки кластеров: L серой шкалы / 255 (весом `ВЕС_L`), a, b OKLab."""
    from .eye import oklab, _L
    lab = oklab(rgb)
    return np.concatenate([_L(lab[..., :1]) / 255.0 * WEIGHT_L, lab[..., 1:]], -1)


def clusters(p, n, iterations=AndTERATsAndY):
    """k средних по точкам `_в_точки` (N, 3); начало — n ступеней светлоты поровну от p1 до p99 (не по площади: светлое
    и тёмное пятно малой площади получают свой центр); пустой кластер берёт точку, дальнюю от своих центров. Ответ —
    центры (n, 3), номер центра каждой точки."""
    L = p[:, 0]
    lo, hi = np.percentile(L, [1, 99])
    steps = lo + (hi - lo) * (np.arange(n) + 0.5) / n
    ts = np.empty((n, 3))
    for k, s in enumerate(steps):                    # центр ступени — средняя точка у этой светлоты
        m = np.abs(L - s) <= max((hi - lo) / (2 * n), 1e-3)
        ts[k] = p[m].mean(0) if m.any() else p[np.abs(L - s).argmin()]
    for _ in range(iterations):
        d = ((p[:, None, :] - ts[None, :, :]) ** 2).sum(-1)
        j = d.argmin(1)
        new = ts.copy()
        for k in range(n):
            m = j == k
            if m.any():
                new[k] = p[m].mean(0)
            else:
                far = int(d[np.arange(len(p)), j].argmax())
                new[k] = p[far]
                j[far] = k
        if np.allclose(new, ts, atol=1e-6):
            ts = new
            break
        ts = new
    d = ((p[:, None, :] - ts[None, :, :]) ** 2).sum(-1)
    return ts, d.argmin(1)


def _in_rgb(ts):
    from .eye import from_oklab, _from_L
    lab = np.concatenate([_from_L(np.clip(ts[..., :1] / WEIGHT_L, 0, 1) * 255.0), ts[..., 1:]], -1)
    return np.clip(np.round(from_oklab(lab)), 0, 255).astype(int)


def palette_with(reference, n=12, path=None):
    """Черновик палитры: n замесов по картинке образца (sRGB 0–255 (В, Ш, 3)) — `Черновик`. Ничего не заводит."""
    from .eye import oklab, _blur
    a = np.asarray(reference, np.float64)[..., :3]
    n = int(n)
    if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError(f'палитра_с: образец — картинка (В, Ш, 3), а не {a.shape}')
    if not 1 <= n <= 64:
        raise ValueError(f'палитра_с: n — от 1 до 64 замесов, а не {n}')
    k = max(1, int(math.ceil(math.sqrt(a.shape[0] * a.shape[1] / TResponseHEBrush))))
    m = _reduce(a, k)
    r = max(1, int(round(max(m.shape[:2]) / 240)))      # мелкий шум не делает замеса: размыто вчетверо слабее прищура
    m = np.stack([_blur(m[..., i], r) for i in range(3)], -1)
    center, _ = clusters(_in_points(m).reshape(-1, 3), min(n, m.shape[0] * m.shape[1]))
    rgb = _in_rgb(center)
    t_ts = _in_points(rgb.astype(np.float64))
    j = np.empty(a.shape[:2], np.intp)
    for y0 in range(0, a.shape[0], 256):                # ближайший замес каждому пикселю образца — кусками
        j[y0:y0 + 256] = ((_in_points(a[y0:y0 + 256])[..., None, :] - t_ts) ** 2).sum(-1).argmin(-1)
    shares = np.bincount(j.ravel(), minlength=len(rgb)) / j.size
    order = sorted(range(len(rgb)), key=lambda i: -float(oklab(rgb[i].astype(np.float64))[0]))
    h = Draft()
    h.shares = {}
    for i in order:
        if shares[i] <= 0 and len(h) >= 1:
            continue
        name = name_colors(rgb[i])
        while name in h:
            name += "'"
        h[name] = tuple(int(v) for v in rgb[i])
        h.shares[name] = round(float(shares[i]), 4)
    h.tiling = rgb[j].astype(np.float64)
    h.squint_dL = round(float(np.abs(_squint_L(a) - _squint_L(h.tiling)).mean()), 2)
    h.sheet = sheet_draft(a, h)
    if path is not None:
        from PIL import Image
        Image.fromarray(h.sheet).save(os.path.expanduser(os.fspath(path)))
    return h


def sheet_draft(a, h, width_swatches=260):
    """uint8: образец 1:1 | замощение замесами 1:1 | плашки замесов высотой по площади (не уже 18 px) с именем."""
    from PIL import Image, ImageDraw
    from .eye import _font, lightness
    In, Sh = a.shape[:2]
    sheet = Image.new('RGB', (2 * Sh + width_swatches + 16, max(In, 18 * len(h) + 8)), (128, 128, 128))
    sheet.paste(Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8)), (0, 0))
    sheet.paste(Image.fromarray(np.clip(h.tiling + 0.5, 0, 255).astype(np.uint8)), (Sh + 8, 0))
    d = ImageDraw.Draw(sheet)
    font = _font(12)
    heights = [max(18.0, h.shares[and_] * (sheet.height - 8)) for and_ in h]
    s = (sheet.height - 8) / max(sum(heights), 1.0)
    y = 4.0
    for (name, ts), h in zip(h.items(), heights):
        h *= s if sum(heights) > sheet.height - 8 else 1.0
        d.rectangle((2 * Sh + 16, int(y), 2 * Sh + 16 + width_swatches - 4, int(y + h) - 2), fill=tuple(ts))
        tone = (0, 0, 0) if lightness(np.asarray(ts, np.float64)) > 140 else (255, 255, 255)
        d.text((2 * Sh + 20, int(y) + 2), f'{name} {100 * h.shares[name]:.0f}%', fill=tone, font=font)
        y += h
    return np.asarray(sheet)


# ---------------------------------------------------------------- кисть.набрать с именем
def _wrap_charge():
    """`к.набрать('имя')` — ValueError словами (имя знает холст). Обёртка здесь, а не в `инструмент.py`: тот входит в
    хеш физики калибровки (`калибровка._ФИЗИКА`), правка его кода сделала бы таблицы чужими."""
    import functools
    from .tool import Brush
    previous = Brush.charge
    if getattr(previous, '_палитра', False):
        return

    @functools.wraps(previous)
    def charge(self, color, amount=None, layout='полосы'):
        if has_names(color):
            raise ValueError(f'набрать: имя замеса {color!r} — у холста: х.набрать(к, имя или [имена][, количество]) или '
                             'к.набрать(х.палитра[имя])')
        return previous(self, color, amount, layout)
    charge._palette = True
    Brush.charge = charge


_wrap_charge()
