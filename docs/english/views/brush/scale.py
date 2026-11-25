# READING VIEW of движок/кисть/масштаб.py with names in English. Not runnable; edit the original.
"""М-7, М-9: что на холсте не в `мм.РОДНОЙ` px/мм держит px и уходит от настройки физики — для строки черновика
`кисть.повтор --scale` < 1 и подсказки С15 «масштаб». Только читает кисть; физику (`мм.py`, `числа.py`) не трогает и в хеш
калибровки не входит."""
from .mm import RResponseDNResponseY, _k

NARROW_PX = 24.0     # px: у кисти уже стольких px холста работает физика узкой кисти, пороги которой в px (`числа._ТЫЧОК`
                    # 24, `_КРАЙ_ОДНА` 24, `_КРАЙ_УЗКАЯ` 20, `_ОСТРАЯ` 16, `_ПОЛ_ПЛЁНКИ` 14, `_ТРИ_УЗК` 12, `_ЩЕЛЬ_УЗК` 8)
GAtWithTAYa = 0.6        # нагрузка (число): с этой — кромка густой кисти (`_БОК` 1,3 px, `_УГОЛ`) в px; 'пастозная' — 0,9
HOLDS_PX = ('узкая', 'мягкая', 'сухая', 'кромка густой')


def width_px(brush, px_on_mm):
    """Ширина кисти в px холста `px_на_мм` (кисть в мм — по нему, кисть в px — как есть)."""
    if getattr(brush, 'ширина_мм', None) is not None and px_on_mm:
        return float(brush.width_mm) * float(px_on_mm)
    return float(brush.width)


def holds_px(brush, load=None, px_on_mm=None):
    """М-7, М-9: чем мазок этой кистью на холсте `px_на_мм` уходит от настройки физики (8 px/мм) — кортеж слов из
    `ДЕРЖИТ_PX`; на родном масштабе (и без `px_на_мм`) — пусто. 'узкая' — кисть уже `УЗКАЯ_PX` px на этом холсте или на
    родном (ширина в мм та же); 'мягкая' — `кайма` > 0, не мастихин; 'сухая' — нагрузка 'сухая'; 'кромка густой' —
    нагрузка 'пастозная' или число от `ГУСТАЯ`. Только читает кисть."""
    k = _k(px_on_mm)
    if k == 1.0:
        return ()
    n = getattr(brush, 'нагрузка', None) if load is None else load
    sh = width_px(brush, px_on_mm)
    what = []
    if min(sh, sh / k) < NARROW_PX:
        what.append('узкая')
    if getattr(brush, 'кайма', 0.0) > 0.0 and getattr(brush, 'форма', '') != 'мастихин':
        what.append('мягкая')
    if n == 'сухая':
        what.append('сухая')
    if n == 'пастозная' or (isinstance(n, (int, float)) and not isinstance(n, bool) and float(n) >= GAtWithTAYa):
        what.append('кромка густой')
    return tuple(what)
