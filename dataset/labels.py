"""Ручная разметка кадров: dataset/labels.csv в папке данных (путь, вид). Без Qt."""
import csv
import os

from watcher import paths
from watcher.classes import CLASSES


def labels_path():
    return paths.data_path("dataset", "labels.csv")


def load_labels(path=None):
    path = path or labels_path()
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) == 2 and row[1] in CLASSES:
                    out[row[0]] = row[1]
    except OSError:
        pass
    return out


def save_labels(labels, path=None):
    path = path or labels_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for k in sorted(labels):
            w.writerow([k, labels[k]])
    os.replace(tmp, path)
