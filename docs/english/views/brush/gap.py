# READING VIEW of движок/кисть/просвет.py with names in English. Not runnable; edit the original.
"""v4, 3д (Б-3): карта просвета — какая доля грунта или нижнего слоя видна сквозь краску, по пикселю; дыры списком.

Только читает: холст, кисти, сиды и журнал не меняются. Счёт — тот же, что у `Холст.картинка`: слой кладётся на то,
что под ним, как `к = к · пропуск; к = к + (цвет − к) · покрытие`, — значит, свет того, что лежит под слоем, доходит
до глаза долей `(1 − покрытие) · пропуск`, а сквозь несколько слоёв — произведением этих долей. Это и есть карта:
1 — видно только нижнее (краски нет), 0 — нижнее закрыто целиком. `пропуск` (не кроющая краска: лессировка, вуаль)
трёхцветный — берётся его яркость (вес каналов линейного света 0,2126, 0,7152, 0,0722).

`слой=None` — сквозь все слои, видно грунт; `слой='луг'` — сквозь этот слой и все выше: видно то, что лежит под
ним. Дыра — связное (8 соседей) место, где карта не ниже `порог`, площадью от `площадь` px.
"""
import numpy as np

YaRBrushResponseWithT = np.array([0.2126, 0.7152, 0.0722], np.float32)


class Gap:
    """Ответ `х.просвет`: `.карта` — float32 (В, Ш) области, 0–1; `.дыры` — список словарей по убыванию площади:
    'рамка' (x0, y0, x1, y1 холста), 'центр' (x, y), 'площадь' px, 'просвет' (среднее карты в дыре), 'у края' (дыра
    касается края области — может быть просто краем зоны); `.сводка` — 'просвет' (среднее карты), 'доля' (доля
    пикселей от порога), 'дыр', 'в дырах' px, 'сквозь' (слои), 'видно' ('грунт' или имя слоя под ними), 'порог',
    'площадь', 'область'; `.лист` — uint8 «холст | карта» с рамками дыр (`путь` — PNG)."""
    __slots__ = ('карта', 'дыры', 'сводка', 'область', '_лист')

    def __getitem__(self, option):
        return self.summary[option]

    @property
    def sheet(self):
        return self._sheet() if callable(self._sheet) else self._sheet

    def __repr__(self):
        with_ = self.summary
        return (f'Просвет({with_["область"]}: видно {with_["видно"]} {with_["просвет"]:.3f} в среднем, от {with_["порог"]} — '
                f'{with_["доля"]:.3f} места, дыр {with_["дыр"]}' +
                (f', крупная {self.holes[0]["площадь"]} px у {self.holes[0]["центр"]})' if self.holes else ')'))


def map(canvas, region=None, layer=None):
    """Карта просвета области (float32 (В, Ш)) и слои, сквозь которые она взята, и что под ними видно."""
    from .eye import _frame
    canvas.finish_computing()
    x0, y0, x1, y1 = _frame(canvas, region)
    names = list(canvas.layers)
    if layer is not None and layer not in names:
        raise KeyError(f'просвет: слоя {layer!r} нет (слои снизу вверх: {names})')
    see_through = names[names.index(layer):] if layer is not None else names
    visible = 'грунт' if layer is None or names.index(layer) == 0 else f'под «{layer}»'
    ly, lx = slice(y0 - canvas.y0, y1 - canvas.y0), slice(x0 - canvas.x0, x1 - canvas.x0)
    brush = np.ones((y1 - y0, x1 - x0), np.float32)
    for and_ in see_through:
        with_ = canvas.layers[and_]
        brush *= 1.0 - np.asarray(with_.coverage[ly, lx], np.float32)
        if with_.transmittance is not None:
            brush *= np.asarray(with_.transmittance[ly, lx], np.float32) @ YaRBrushResponseWithT
    np.clip(brush, 0.0, 1.0, out=brush)
    return brush, (x0, y0, x1, y1), see_through, visible


def gap(canvas, region=None, layer=None, threshold=0.5, area=4, holes=50, path=None, scale=1, without=None, to=None):
    """Б-3: карта просвета области и дыры. Подробно — модуль и `Просвет`."""
    from scipy import ndimage
    from .eye import without_layers
    if not 0.0 < threshold <= 1.0:
        raise ValueError('просвет: порог — от 0 (не включая) до 1')
    canvas = without_layers(canvas, without, to)
    brush, frame, see_through, visible = map(canvas, region, layer)
    x0, y0 = frame[:2]
    labels, n = ndimage.label(brush >= threshold, structure=np.ones((3, 3), bool))
    holes = []
    if n:
        areas = np.bincount(labels.ravel(), minlength=n + 1)
        sums = np.bincount(labels.ravel(), weights=brush.ravel().astype(np.float64), minlength=n + 1)
        places = ndimage.find_objects(labels)
        for j in np.argsort(-areas[1:], kind='stable') + 1:
            s = int(areas[j])
            if s < area:
                break
            sy, sx = places[j - 1]
            yy, xx = np.nonzero(labels[sy, sx] == j)
            holes.append({'рамка': (x0 + sx.start, y0 + sy.start, x0 + sx.stop, y0 + sy.stop),
                         'центр': (round(x0 + sx.start + float(xx.mean()), 1), round(y0 + sy.start + float(yy.mean()), 1)),
                         'площадь': s, 'просвет': round(float(sums[j] / s), 3),
                         'у края': bool(sy.start == 0 or sx.start == 0 or sy.stop == brush.shape[0] or sx.stop == brush.shape[1])})
    p = Gap()
    p.map, p.holes, p.region = brush, holes[:holes] if holes is not None else holes, frame
    p.summary = {'просвет': round(float(brush.mean()), 4), 'доля': round(float((brush >= threshold).mean()), 4), 'дыр': len(holes),
                'в дырах': int(sum(d['площадь'] for d in holes)), 'сквозь': see_through, 'видно': visible, 'порог': threshold,
                'площадь': area, 'область': frame}
    p._sheet = lambda: _sheet(canvas, p, scale, None)
    if path is not None:
        p._sheet = _sheet(canvas, p, scale, path)
    return p


def _sheet(canvas, p, scale, path):
    """«холст | карта»: карта серым (белое — видно нижнее), рамки дыр красным — на обеих половинах."""
    from PIL import Image, ImageDraw
    from .eye import look, _scale, _sheet as sheet_panel
    left = look(canvas, p.region, scale=scale)
    right = _scale((np.repeat(p.map[..., None], 3, -1) * 255.0 + 0.5).astype(np.uint8), scale)
    right = right[:left.shape[0], :left.shape[1]]
    x0, y0 = p.region[:2]
    m = left.shape[1] / max(1, p.map.shape[1])
    panels = []
    for a in (left, right):
        im = Image.fromarray(np.ascontiguousarray(a))
        d = ImageDraw.Draw(im)
        for dd in p.holes[:20]:
            a0, b0, a1, b1 = dd['рамка']
            d.rectangle([(a0 - x0) * m - 2, (b0 - y0) * m - 2, (a1 - x0) * m + 1, (b1 - y0) * m + 1], outline=(230, 30, 30))
        panels.append(np.asarray(im))
    with_ = p.summary
    signatures = [f'холст {p.region}, ×{scale}',
               f'просвет: видно {with_["видно"]} (белое) · среднее {with_["просвет"]:.3f} · дыр {with_["дыр"]}'.replace('.', ',')]
    return sheet_panel(panels, signatures, path)
