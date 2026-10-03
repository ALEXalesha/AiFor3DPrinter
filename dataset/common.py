"""Общее для инструментов набора данных: где лежит набор, как зовутся печати, разбиение."""
import hashlib
import os
import re

from watcher import paths

TIMELAPSES = os.environ.get("AIFP_TIMELAPSES", "")       # папка таймлапсов (задаётся при запуске)

# ECC_0.4_<модель>_<материал><слой>_<время>-<номер>.mp4
_DUR_RE = re.compile(r"_(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?-\d+\.mp4$", re.I)
_TAIL_RE = re.compile(r"_(?:Generic )?(?:PLA|PET|PETG|TPU|ABS|ASA|PLApod)[^_]*_?0\.\d+_[0-9hms]+-\d+\.mp4$", re.I)


def dataset_dir(*parts):
    d = os.path.join(paths.data_dir(), "dataset", *parts)
    os.makedirs(d, exist_ok=True)
    return d


def duration_s(filename):
    """Время печати из имени таймлапса Elegoo: ..._33m14s-123.mp4 -> 1994. Нет - None."""
    m = _DUR_RE.search(os.path.basename(filename))
    if not m or not any(m.groups()):
        return None
    h, mi, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mi * 60 + s
    return total or None


def model_name(filename):
    """Модель печати без материала, времени и номера: одна модель = одна группа при разбиении."""
    base = os.path.basename(filename)
    base = re.sub(r"^ECC_0\.\d_", "", base)
    stem = _TAIL_RE.sub("", base)
    if stem == base:
        stem = re.sub(r"-\d+\.mp4$", "", base)
    return stem.strip(" _").lower()


def print_id(filename):
    return hashlib.sha1(os.path.basename(filename).encode("utf-8")).hexdigest()[:10]


def group_split(group, val=0.15, test=0.2, salt="aifp-1"):
    """train / val / test по хешу группы (модели печати), а не кадра.

    Кадры одной печати - и все печати одной модели - попадают в одну часть: иначе сеть
    видела бы в проверке почти те же кадры, что в обучении, и цифры врали бы."""
    h = int(hashlib.sha1((salt + "|" + group).encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < test:
        return "test"
    if h < test + val:
        return "val"
    return "train"
