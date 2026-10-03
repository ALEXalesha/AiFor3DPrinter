"""JSON на диске: битый файл не роняет службу, запись целиком через временный файл."""
import json
import os
from pathlib import Path


def load_json(path):
    """Содержимое файла или None: нет файла, мусор, обрезанная запись."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None


def save_json(path, data):
    """Через временный файл: убитый посреди записи процесс оставит прежний файл целым."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def save_bytes(path, data):
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return True
    except OSError:
        return False
