"""Тепловая карта «где брак» поверх кадра и рамка вокруг самого горячего места."""
import io

import numpy as np
from PIL import Image, ImageDraw

from watcher.classes import CLASSES


def normalize_cam(cam_cls):
    """Карта одного класса -> 0..1 (отрицательный вклад = 0)."""
    m = np.maximum(np.asarray(cam_cls, dtype=np.float32), 0)
    hi = float(m.max()) if m.size else 0.0
    return m / hi if hi > 1e-9 else np.zeros_like(m)


def colorize(v):
    """0..1 -> RGB: тёмно-синий - жёлтый - красный (без matplotlib)."""
    v = np.clip(v, 0, 1)[..., None]
    stops = np.array([[0, 0, 80], [0, 120, 255], [255, 230, 0], [255, 40, 0]], dtype=np.float32)
    pos = v * (len(stops) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(stops) - 2)
    f = pos - i
    return (stops[i[..., 0]] * (1 - f) + stops[i[..., 0] + 1] * f).astype(np.uint8)


def hot_box(heat, frac=0.6):
    """Рамка вокруг клеток не холоднее frac от максимума: (x0, y0, x1, y1) в долях 0..1 или None."""
    h = np.asarray(heat)
    if h.size == 0 or h.max() <= 0:
        return None
    ys, xs = np.nonzero(h >= frac * h.max())
    gh, gw = h.shape
    return (xs.min() / gw, ys.min() / gh, (xs.max() + 1) / gw, (ys.max() + 1) / gh)


def overlay(img, cam, cls, alpha=0.45):
    """PIL RGB + cam [5, h, w] -> новая PIL RGB с картой выбранного класса и рамкой."""
    img = img.convert("RGB")
    if cam is None or cls not in CLASSES:
        return img.copy()
    heat = normalize_cam(cam[CLASSES.index(cls)])
    big = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR),
                     dtype=np.float32) / 255.0
    base = np.asarray(img, dtype=np.float32)
    col = colorize(big).astype(np.float32)
    a = (alpha * big)[..., None]
    out = Image.fromarray((base * (1 - a) + col * a).clip(0, 255).astype(np.uint8))
    box = hot_box(heat)
    if box:
        w, h = img.size
        d = ImageDraw.Draw(out)
        d.rectangle([box[0] * w, box[1] * h, box[2] * w - 1, box[3] * h - 1], outline=(255, 40, 40),
                    width=max(2, w // 200))
    return out


def zoom(img, cam, cls, out_w=960, min_frac=0.45, pad=0.6):
    """Крупный план места брака для Telegram (17.10, Алексей: «область ошибки и просто фото, а то так не видно»).

    Вырезка 16:9 вокруг горячей рамки (с запасом pad с каждой стороны, не меньше min_frac ширины кадра),
    увеличенная до out_w; место брака - тонкая красная рамка, без цветной карты. Нет карты - весь кадр.
    """
    img = img.convert("RGB")
    W, H = img.size
    oh = round(out_w * H / W)
    box = hot_box(normalize_cam(cam[CLASSES.index(cls)])) if cam is not None and cls in CLASSES else None
    if not box:
        return img.resize((out_w, oh), Image.LANCZOS)
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    w = min(W, max((x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad) * W / H, min_frac * W))
    h = w * H / W
    left = min(max((x0 + x1) / 2 - w / 2, 0), W - w)
    top = min(max((y0 + y1) / 2 - h / 2, 0), H - h)
    out = img.crop((round(left), round(top), round(left + w), round(top + h))).resize((out_w, oh), Image.LANCZOS)
    k = out_w / w
    ImageDraw.Draw(out).rectangle([(x0 - left) * k, (y0 - top) * k, (x1 - left) * k - 1, (y1 - top) * k - 1],
                                  outline=(255, 40, 40), width=max(2, out_w // 320))
    return out


def jpeg(img, quality=85):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=quality)
    return buf.getvalue()
