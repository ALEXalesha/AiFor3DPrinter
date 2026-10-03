"""Сборка набора: кадры таймлапсов (норма) + искусственный брак + спагетти SKZ -> index.csv.

    python -m dataset.build [--rate 0.3] [--workers 4]

Каждая строка index.csv: путь, класс, часть (train/val/test), печать, источник. Часть
берётся у печати (а печать - у модели, dataset/common.group_split): искусственный брак,
нарисованный поверх кадра тестовой печати, тоже тестовый. Кадры SKZ - в той части,
которую дали авторы набора. Плюс кадры, отмеченные кнопкой «Всё нормально» в Telegram
(feedback/normal) - они идут в обучение как норма.
"""
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dataset.common import dataset_dir
from watcher import paths

SYNTH_CLASSES = ("spaghetti", "blob", "detached", "no_extrusion")


def load_patches(split, limit=600):
    """Куски настоящих спагетти из кадров SKZ своей части."""
    import cv2

    from dataset.skz import spaghetti_boxes
    d = os.path.join(paths.data_dir(), "dataset", "skz", split)
    out = []
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".jpg"):
            continue
        txt = os.path.join(d, fn[:-4] + ".txt")
        if not os.path.exists(txt):
            continue
        img = cv2.cvtColor(cv2.imread(os.path.join(d, fn)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        for (x0, y0, x1, y1) in spaghetti_boxes(open(txt, encoding="utf-8").read(), w, h):
            out.append(img[y0:y1, x0:x1].copy())
            if len(out) >= limit:
                return out
    return out


_PATCHES = {}


def _work(job):
    """Один кадр -> несколько искусственных. Запускается в отдельном процессе."""
    import cv2
    cv2.setNumThreads(1)
    from PIL import Image

    from dataset import synth
    src, out_dir, split, pid, idx, seed, want = job
    if split not in _PATCHES:
        _PATCHES[split] = load_patches(split)
    rng = np.random.default_rng(seed)
    img = synth.as_rgb(Image.open(src).convert("RGB"))
    rows = []
    for cls in want:
        if cls == "normal-edit":
            out = synth.neutral_edit(img, rng)
            label = "normal"
        elif cls == "spaghetti":
            out = synth.spaghetti(img, rng, _PATCHES[split])
            label = cls
        else:
            out = synth.GENERATORS[cls](img, rng)
            label = cls
        if out is None:
            continue
        name = f"{pid}_{idx:05d}_{cls}.jpg"
        dst = os.path.join(out_dir, label, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(out).save(dst, quality=90)
        rows.append((dst, label, split, pid, "synth" if label != "normal" else "timelapse-edit", idx))
    return rows


def build(rate=0.3, edit_rate=0.15, workers=4, seed=1):
    root = dataset_dir()
    prints = json.load(open(os.path.join(root, "prints.json"), encoding="utf-8"))
    synth_dir = dataset_dir("synth")
    rng = np.random.default_rng(seed)
    rows, jobs = [], []
    for pid in sorted(prints):
        p = prints[pid]
        fdir = os.path.join(root, "frames", pid)
        frames = sorted(f for f in os.listdir(fdir) if f.endswith(".jpg"))
        for i, f in enumerate(frames):
            src = os.path.join(fdir, f)
            rows.append((src, "normal", p["split"], pid, "timelapse", i))
            if i < 2:                     # первые кадры - пустой стол без головы
                continue
            want = [c for c in SYNTH_CLASSES if rng.random() < rate]
            if rng.random() < edit_rate:
                want.append("normal-edit")
            if want:
                jobs.append((src, synth_dir, p["split"], pid, i, int(rng.integers(1 << 31)), want))
    print(f"нормальных кадров {len(rows)}, кадров под подделки {len(jobs)}", flush=True)
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_work, jobs, chunksize=16):
            rows.extend(r)
            done += 1
            if done % 1000 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    for split in ("train", "val", "test"):
        d = os.path.join(root, "skz", split)
        if os.path.isdir(d):
            for fn in sorted(os.listdir(d)):
                if fn.endswith(".jpg"):
                    rows.append((os.path.join(d, fn), "spaghetti", split, "skz:" + fn.split("-frame_")[0], "skz", 0))
    fb = os.path.join(paths.data_dir(), "feedback", "normal")
    if os.path.isdir(fb):
        for fn in sorted(os.listdir(fb)):
            if fn.endswith(".jpg"):
                rows.append((os.path.join(fb, fn), "normal", "train", "feedback", "telegram", 0))
    from dataset.labels import load_labels
    known = {r[0]: i for i, r in enumerate(rows)}
    for path, cls in load_labels().items():       # ручная разметка (инструмент разметки)
        if not os.path.exists(path):
            continue
        if path in known:
            r = rows[known[path]]
            rows[known[path]] = (r[0], cls, r[2], r[3], "label", r[5])
        else:
            rows.append((path, cls, "train", "label", "label", 0))
    with open(os.path.join(root, "index.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "cls", "split", "print", "source", "frame"])
        w.writerows(rows)
    summary = {}
    for _, cls, split, *_ in rows:
        summary.setdefault(split, {}).setdefault(cls, 0)
        summary[split][cls] += 1
    json.dump(summary, open(os.path.join(root, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--rate" in a:
        kw["rate"] = float(a[a.index("--rate") + 1])
    if "--workers" in a:
        kw["workers"] = int(a[a.index("--workers") + 1])
    build(**kw)
