"""Настоящие «спагетти» из открытого набора SKZ (Hugging Face), лицензия CC BY-SA 4.0.

    python -m dataset.skz [--train 700 --val 120 --test 160]

Набор: DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Plattform_Cam - камера сверху
на весь стол другого принтера, разметка YOLO (класс 3 = Spaghetti). Берём часть кадров,
равномерно по печатям, и уменьшаем до 960x540 JPEG. Из рамок «Spaghetti» потом режутся
куски настоящих спагетти, которые накладываются на кадры Centauri (dataset/synth.py).

Разбиение train/val/test - то, что дали авторы набора. Кадры в репозиторий не кладутся;
источник и лицензия записаны в dataset/LICENSES.md.
"""
import io
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from dataset.common import dataset_dir

REPO = "DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Plattform_Cam"
API = f"https://huggingface.co/api/datasets/{REPO}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
LICENSE = "CC BY-SA 4.0"
SPAGHETTI = 3


def list_files():
    with urllib.request.urlopen(API, timeout=60) as r:
        info = json.load(r)
    return [s["rfilename"] for s in info.get("siblings", [])]


def print_of(path):
    return os.path.basename(path).split("-frame_")[0]


def choose(files, split, n):
    imgs = sorted(f for f in files if f.startswith(f"images/{split}/") and f.endswith(".png"))
    by_print = defaultdict(list)
    for f in imgs:
        by_print[print_of(f)].append(f)
    out, k = [], 0
    while len(out) < n and any(by_print.values()):     # по кругу по печатям: всех понемногу
        for p in sorted(by_print):
            if by_print[p]:
                lst = by_print[p]
                out.append(lst.pop((k * 7919) % len(lst)))
                if len(out) >= n:
                    break
        k += 1
    return out


def fetch(path, timeout=120):
    with urllib.request.urlopen(RAW + urllib.request.quote(path), timeout=timeout) as r:
        return r.read()


def download_one(img_path, out_dir):
    name = os.path.splitext(os.path.basename(img_path))[0]
    jpg = os.path.join(out_dir, name + ".jpg")
    txt = os.path.join(out_dir, name + ".txt")
    if os.path.exists(jpg) and os.path.exists(txt):
        return name, "есть"
    lbl_path = img_path.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
    try:
        labels = fetch(lbl_path, 60).decode("utf-8", "replace")
    except Exception:
        labels = ""
    img = Image.open(io.BytesIO(fetch(img_path))).convert("RGB")
    img.thumbnail((960, 540), Image.LANCZOS)
    img.save(jpg + ".tmp", "JPEG", quality=90)
    os.replace(jpg + ".tmp", jpg)
    open(txt, "w", encoding="utf-8").write(labels)
    return name, "ok"


def build(counts):
    files = list_files()
    root = dataset_dir("skz")
    meta = {"source": f"https://huggingface.co/datasets/{REPO}", "license": LICENSE, "splits": {}}
    for split, n in counts.items():
        chosen = choose(files, split, n)
        out = dataset_dir("skz", split)
        with ThreadPoolExecutor(4) as ex:
            for i, (name, st) in enumerate(ex.map(lambda p: download_one(p, out), chosen), 1):
                if i % 50 == 0 or i == len(chosen):
                    print(f"{split}: {i}/{len(chosen)}", flush=True)
        meta["splits"][split] = len(chosen)
    json.dump(meta, open(os.path.join(root, "source.json"), "w", encoding="utf-8"), indent=1)
    return meta


def spaghetti_boxes(label_text, w, h, min_px=24):
    """Рамки класса Spaghetti в пикселях (x0, y0, x1, y1); слишком мелкие отброшены."""
    out = []
    for line in label_text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            c = int(parts[0])
            cx, cy, bw, bh = (float(v) for v in parts[1:])
        except ValueError:
            continue
        if c != SPAGHETTI:
            continue
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 - x0 >= min_px and y1 - y0 >= min_px:
            out.append((x0, y0, x1, y1))
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    counts = {"train": 700, "val": 120, "test": 160}
    for i, a in enumerate(args):
        if a.startswith("--") and a[2:] in counts and i + 1 < len(args):
            counts[a[2:]] = int(args[i + 1])
    print(build(counts))
