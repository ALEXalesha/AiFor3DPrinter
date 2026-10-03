"""Нарезка таймлапсов Elegoo в кадры: одна папка на печать, плюс prints.json.

    python -m dataset.frames <папка таймлапсов>

Таймлапс Centauri Carbon - кадр на слой, 640x360, 20 кадров/с. Время печати берётся из
имени файла (..._1h3m-123.mp4), по нему потом считаются ложные тревоги на час печати.
Кадры кладутся в папку данных (%LOCALAPPDATA%\\AiForPrinter\\dataset\\frames) - в
репозиторий они не попадают.
"""
import json
import os
import subprocess
import sys

from dataset.common import dataset_dir, duration_s, group_split, model_name, print_id


def ffmpeg_extract(video, out_dir, quality=3):
    os.makedirs(out_dir, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video, "-q:v", str(quality), os.path.join(out_dir, "%05d.jpg")]
    subprocess.run(cmd, check=True, timeout=600, stdin=subprocess.DEVNULL)
    return sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))


def build(src_dir, force=False):
    root = dataset_dir("frames")
    index_path = os.path.join(dataset_dir(), "prints.json")
    try:
        prints = json.load(open(index_path, encoding="utf-8"))
    except (OSError, ValueError):
        prints = {}
    videos = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".mp4"))
    for i, v in enumerate(videos, 1):
        pid = print_id(v)
        out = os.path.join(root, pid)
        if pid in prints and not force and os.path.isdir(out):
            continue
        frames = ffmpeg_extract(os.path.join(src_dir, v), out)
        group = model_name(v)
        prints[pid] = {"file": v, "group": group, "split": group_split(group), "frames": len(frames),
                       "duration_s": duration_s(v)}
        print(f"[{i}/{len(videos)}] {pid} {len(frames)} кадров {prints[pid]['split']} {group}", flush=True)
    json.dump(prints, open(index_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return prints


if __name__ == "__main__":
    build(sys.argv[1], force="--force" in sys.argv)
