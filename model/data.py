"""Загрузка кадров для обучения: index.csv -> тензоры с аугментацией.

Аугментация повторяет то, чем живой снимок отличается от кадра таймлапса: другое
сжатие JPEG, яркость и баланс света в камере, лёгкая размытость движения, сдвиг и
масштаб (голова в живом кадре ездит), зеркало.
"""
import csv
import io
import random

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

from watcher.classes import CLASSES
from watcher.model_runtime import MEAN, STD

from model.net import INPUT_SIZE


def read_index(path, split):
    rows = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == split:
                rows.append(r)
    return rows


def to_tensor(img):
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def augment(img, rng):
    w, h = img.size
    s = rng.uniform(0.8, 1.0)
    cw, ch = int(w * s), int(h * s * rng.uniform(0.92, 1.08))
    ch = min(ch, h)
    x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x0, y0, x0 + cw, y0 + ch)).resize(INPUT_SIZE, Image.BILINEAR)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.4)))
    if rng.random() < 0.6:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(25, 90))
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return img


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, rows, train):
        self.rows = rows
        self.train = train
        self.labels = [CLASSES.index(r["cls"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        if self.train:
            img = augment(img, random.Random(hash((i, random.random()))))
        else:
            img = img.resize(INPUT_SIZE, Image.BILINEAR)
        return to_tensor(img), self.labels[i]


def balanced_weights(rows, normal_share=0.4, skz_share_of_spaghetti=0.3):
    """Веса выборки: норма ~40% эпохи (ложные тревоги важнее всего), брак поровну.
    Внутри спагетти кадры SKZ (чужая камера) - не больше трети."""
    n = len(CLASSES)
    groups = {}
    for i, r in enumerate(rows):
        key = (r["cls"], r["source"] == "skz")
        groups.setdefault(key, []).append(i)
    w = np.zeros(len(rows), dtype=np.float64)
    fail_share = (1 - normal_share) / (n - 1)
    for (cls, is_skz), idx in groups.items():
        share = normal_share if cls == "normal" else fail_share
        if cls == "spaghetti":
            has_both = (("spaghetti", True) in groups) and (("spaghetti", False) in groups)
            if has_both:
                share *= skz_share_of_spaghetti if is_skz else (1 - skz_share_of_spaghetti)
        w[idx] = share / len(idx)
    return w
