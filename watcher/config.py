"""Настройки службы и окна: config.json в папке данных.

Любое значение из файла проходит normalize(): мусор, чужой тип, выход за пределы -
значение по умолчанию или ближайшая граница. Особое правило у автопаузы: она включена,
только если в файле стоит ровно true (не "true", не 1). Пока Алексей не сказал «да»
по цифрам ложных тревог, она выключена.
"""
import math
import os

from watcher import paths
from watcher.storage import load_json, save_json

DEFAULTS = {
    "camera_url": "http://127.0.0.1:8099",
    "interval_s": 15,          # снимок раз в N секунд во время печати
    "threshold": 0.7,          # уверенность, с которой кадр считается «брак»
    "window": 4,               # решение по последним N кадрам ...
    "need": 3,                 # ... из них столько должны быть «брак»
    "cooldown_s": 600,         # после тревоги столько секунд не тревожить снова
    "auto_pause": False,       # пауза без человека: ВЫКЛЮЧЕНА, пока Алексей не скажет «да»
    "telegram": True,          # класть тревоги боту n8n
    "bot_dir": "",             # пусто = %USERPROFILE%\n8n_pc\run\aifp
    "theme": "glass",
    "lang": "ru",
}

LIMITS = {
    "interval_s": (5, 300),
    "threshold": (0.3, 0.99),
    "window": (1, 10),
    "need": (1, 10),
    "cooldown_s": (0, 24 * 3600),
}

THEMES = ("glass", "formal", "light", "night", "warm")
LANGS = ("ru", "en")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def normalize(raw):
    cfg = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return cfg
    for key, (lo, hi) in LIMITS.items():
        v = raw.get(key)
        if _num(v):
            v = min(max(v, lo), hi)
            cfg[key] = int(round(v)) if isinstance(DEFAULTS[key], int) else float(v)
    cfg["need"] = min(cfg["need"], cfg["window"])
    cfg["auto_pause"] = raw.get("auto_pause") is True
    cfg["telegram"] = raw.get("telegram") is not False
    for key in ("camera_url", "bot_dir"):
        if isinstance(raw.get(key), str) and len(raw[key]) < 500:
            cfg[key] = raw[key].strip()
    if not cfg["camera_url"]:
        cfg["camera_url"] = DEFAULTS["camera_url"]
    if raw.get("theme") in THEMES:
        cfg["theme"] = raw["theme"]
    if raw.get("lang") in LANGS:
        cfg["lang"] = raw["lang"]
    return cfg


def config_path():
    return paths.data_path("config.json")


def load():
    return normalize(load_json(config_path()))


def save(cfg):
    return save_json(config_path(), normalize(cfg))


def bot_dir(cfg):
    """Папка обмена с ботом n8n: туда служба кладёт тревоги, оттуда берёт ответы кнопок."""
    if cfg.get("bot_dir"):
        return cfg["bot_dir"]
    home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(home, "n8n_pc", "run", "aifp")
