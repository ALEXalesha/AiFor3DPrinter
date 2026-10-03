"""Снимки и оценки по печатям: чтобы после брака было что посмотреть.

03.10.2026 деталь оторвалась, сопло таскало её по столу, сообщений не было - а в журнале
службы были только «новая печать» и «печать не идёт». Ни одного кадра и ни одной оценки.
Теперь у каждой печати своя папка в %LOCALAPPDATA%\\AiForPrinter\\prints:

    20261003-204932_tornado-tower/
        meta.json       имя файла печати и время начала
        scores.jsonl    одна строка на снимок: время, слой, оценки сети, движение, тревога
        frames/*.jpg    сами снимки (последние max_frames_per_print)

Кадры - полные снимки камеры: камера стоит внутри закрытой камеры принтера и видит
только стол и голову, а полный кадр нужен, чтобы позже прогнать по нему новую сеть
(main.py review --rescore). Папка вне репозитория, в .gitignore на всякий случай тоже.
Размер ограничен: старые кадры печати уходят первыми (оценки остаются), старые печати -
целиком, когда их больше max_prints или все кадры весят больше max_bytes.
"""
import json
import os
import re
import shutil
import time

from watcher.storage import load_json, save_bytes, save_json


def slug(name, n=40):
    base = os.path.splitext(os.path.basename(str(name or "print")))[0]
    base = re.sub(r"^ECC_0\.\d_", "", base)
    s = re.sub(r"[^0-9A-Za-z]+", "-", base).strip("-").lower()
    return (s or "print")[:n]


class FrameStore:
    def __init__(self, root, max_frames_per_print=1500, max_prints=20, max_bytes=2 * 1024 ** 3):
        self.root = root
        self.max_frames = max_frames_per_print
        self.max_prints = max_prints
        self.max_bytes = max_bytes
        self.dir = None

    def start(self, key, t):
        name = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + "_" + slug(key)
        d = os.path.join(self.root, name)
        k = 1
        while os.path.exists(d):
            k += 1
            d = os.path.join(self.root, f"{name}-{k}")
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        save_json(os.path.join(d, "meta.json"), {"key": key, "start": t})
        self.dir = d
        self.prune()
        return d

    def add(self, t, frame_jpeg, record):
        if self.dir is None:
            self.start(record.get("key", "?"), t)
        name = time.strftime("%H%M%S", time.localtime(t)) + f"-{int((t % 1) * 1000):03d}.jpg"
        path = os.path.join(self.dir, "frames", name)
        k = 1
        while os.path.exists(path):
            path = os.path.join(self.dir, "frames", name[:-4] + f"_{k}.jpg")
            k += 1
        save_bytes(path, frame_jpeg)
        rec = dict(record, frame=os.path.relpath(path, self.dir))
        try:
            with open(os.path.join(self.dir, "scores.jsonl"), "a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass
        self._cap_frames(self.dir)
        self.prune()
        return path

    def _cap_frames(self, d):
        fd = os.path.join(d, "frames")
        try:
            frames = sorted(os.listdir(fd))
        except OSError:
            return
        for fn in frames[:max(0, len(frames) - self.max_frames)]:
            _remove(os.path.join(fd, fn))

    def prune(self):
        """Старые печати - прочь, пока их не больше max_prints и кадры не тяжелее max_bytes.
        Текущая печать не удаляется никогда; в ней при нехватке места уходят старые кадры."""
        prints = list_prints(self.root)
        while len(prints) > self.max_prints and prints[0]["dir"] != self.dir:
            _remove_tree(prints.pop(0)["dir"])
        while prints and _frames_bytes(prints) > self.max_bytes:
            if prints[0]["dir"] != self.dir:
                _remove_tree(prints.pop(0)["dir"])
                continue
            fd = os.path.join(self.dir, "frames")
            frames = sorted(os.listdir(fd)) if os.path.isdir(fd) else []
            if not frames:
                break
            _remove(os.path.join(fd, frames[0]))
            prints = list_prints(self.root)


def _remove(path):
    try:
        os.remove(path)          # свой кэш кадров с ограничением размера, не файлы пользователя
    except OSError:
        pass


def _remove_tree(path):
    shutil.rmtree(path, ignore_errors=True)


def _frames_bytes(prints):
    total = 0
    for p in prints:
        fd = os.path.join(p["dir"], "frames")
        try:
            total += sum(e.stat().st_size for e in os.scandir(fd) if e.is_file())
        except OSError:
            pass
    return total


def list_prints(root):
    """Печати по времени начала: [{dir, key, start, frames}]. Папки без meta.json не считаются."""
    out = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for n in names:
        d = os.path.join(root, n)
        meta = load_json(os.path.join(d, "meta.json"))
        if not isinstance(meta, dict):
            continue
        fd = os.path.join(d, "frames")
        try:
            nf = sum(1 for e in os.scandir(fd) if e.name.endswith(".jpg"))
        except OSError:
            nf = 0
        out.append({"dir": d, "key": str(meta.get("key", "?")), "start": float(meta.get("start", 0) or 0),
                    "frames": nf})
    out.sort(key=lambda p: (p["start"], p["dir"]))
    return out


def load_scores(d):
    out = []
    try:
        with open(os.path.join(d, "scores.jsonl"), encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and "t" in r:
                    if r.get("frame"):
                        r["frame"] = os.path.join(d, r["frame"])
                    out.append(r)
    except OSError:
        pass
    return out
