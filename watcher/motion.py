"""Движение на столе: деталь оторвалась и ездит за соплом.

Сеть смотрит на один кадр и училась на искусственном «отрыве» (деталь лежит на боку).
Настоящий отрыв 03.10.2026 выглядел иначе: тонкая деталь первых слоёв оторвалась, сопло
таскало её по столу и в конце ударило ею по голове. Сообщений не было. Такое видно не
по форме, а по тому, что неподвижная картина стола поменялась.

Голова в живых снимках ездит по всему кадру, и простая разница соседних снимков ловит
её, а не деталь (на таймлапсах голова у переднего края закрывала половину стола).
Поэтому сравниваются не снимки, а «фон»: медиана последних `window` снимков по каждому
пикселю. Голова в разных местах - медиана её убирает, а неподвижное (стол, детали)
остаётся. Если фон сейчас и фон `window` снимков назад заметно различаются в нижней
части кадра - на столе что-то сдвинулось: деталь уехала со своего места, по столу
размазан пластик. Обычная печать меняет фон медленно (стол опускается на слой).

Без обучения и без данных брака: только яркость, выровненная на медиану кадра
(подсветка и автоэкспозиция камеры меняют весь кадр сразу).
"""
from collections import deque

import numpy as np
from PIL import Image, ImageFilter

SMALL = (160, 90)          # ширина, высота уменьшенного кадра
BED_TOP = 0.57             # ниже этой доли высоты - стол (линия сопла ~0.46-0.53)
PIXEL_THR = 0.25           # пиксель фона «изменился», если яркость сдвинулась на 25% от медианы кадра


def prepare(img):
    """PIL RGB -> уменьшенная серая нижняя часть кадра, нормированная на медиану всего кадра."""
    g = img.convert("L").resize(SMALL, Image.BILINEAR).filter(ImageFilter.GaussianBlur(1.0))
    a = np.asarray(g, dtype=np.float32)
    med = float(np.median(a))
    return a[int(SMALL[1] * BED_TOP):] / max(med, 8.0)


def changed_fraction(a, b, thr=PIXEL_THR):
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    return float((np.abs(a - b) > thr).mean())


class MotionWatch:
    """Фон сейчас против фона window снимков назад. Тревога: need подряд оценок >= threshold."""

    def __init__(self, threshold=0.07, window=5, need=2):
        self.threshold = threshold
        self.window = max(2, int(window))
        self.need = max(1, int(need))
        self.frames = deque(maxlen=2 * self.window)
        self.streak = 0

    def reset(self):
        self.frames.clear()
        self.streak = 0

    def ready(self):
        return len(self.frames) >= 2 * self.window

    def update(self, img):
        """PIL RGB -> (доля изменившегося фона 0..1, тревога: bool)."""
        self.frames.append(prepare(img))
        if not self.ready():
            return 0.0, False
        old = np.median(np.stack(list(self.frames)[:self.window]), axis=0)
        new = np.median(np.stack(list(self.frames)[self.window:]), axis=0)
        ch = changed_fraction(old, new)
        self.streak = self.streak + 1 if ch >= self.threshold else 0
        return ch, self.streak >= self.need
