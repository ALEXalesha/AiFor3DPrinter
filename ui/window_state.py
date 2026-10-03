"""Окно открывается там, где его закрыли.

Правило выбора места (restore) - строка в строку из TicTacToeAi (оттуда же AiCar tk_window_state.restore), оно же
в калькуляторах, Paint Pro и SyncGlass: что бы ни лежало в файле - окно на отключённом
мониторе, размер больше экрана или меньше минимума, мусор, обрезанный файл, - окно
открывается там, где его видно и за заголовок можно взяться. Рабочие области экранов
берутся у Qt (availableGeometry - без панели задач), основной первым.

В файл пишутся левый верхний угол рамки окна (pos) и размер клиентской области - ровно то,
что потом принимают move() и resize().
"""
from __future__ import annotations

import math

from watcher.storage import load_json, save_json

# Сколько окна должно остаться на экране, чтобы за него можно было взяться: полоса
# заголовка высотой 38 и хотя бы 80 по ширине.
GRIP_HEIGHT = 38
GRIP_WIDTH = 80


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _round(v) -> int:
    """Как Math.round в JS: половина - вверх. round() Python округлял бы 2.5 до 2."""
    return math.floor(v + 0.5)


def _overlap(a, b):
    w = min(a["x"] + a["width"], b["x"] + b["width"]) - max(a["x"], b["x"])
    h = min(a["y"] + a["height"], b["y"] + b["height"]) - max(a["y"], b["y"])
    return (w, h) if w > 0 and h > 0 else None


def restore(saved, areas, opts) -> dict:
    """Где и какого размера открыть окно.

    saved - что лежало в файле (что угодно); areas - рабочие области экранов
    {x, y, width, height}, основной первым; opts - {width, height, minWidth, minHeight}.
    Без x и y в ответе - окно ставится по центру основного экрана."""
    by_default = {"width": opts["width"], "height": opts["height"], "maximized": False}
    screens = [a for a in (areas if isinstance(areas, list) else [])
               if isinstance(a, dict) and all(_finite(a.get(k)) for k in ("x", "y", "width", "height"))
               and a["width"] > 0 and a["height"] > 0]
    if not isinstance(saved, dict) or not _finite(saved.get("width")) or not _finite(saved.get("height")):
        return by_default

    maximized = saved.get("maximized") is True
    big_w = max([opts["width"]] + [a["width"] for a in screens])
    big_h = max([opts["height"]] + [a["height"] for a in screens])
    width = _round(min(max(saved["width"], opts["minWidth"]), max(big_w, opts["minWidth"])))
    height = _round(min(max(saved["height"], opts["minHeight"]), max(big_h, opts["minHeight"])))

    if not _finite(saved.get("x")) or not _finite(saved.get("y")):
        return {"width": width, "height": height, "maximized": maximized}
    x, y = _round(saved["x"]), _round(saved["y"])

    grip = {"x": x, "y": y, "width": width, "height": GRIP_HEIGHT}
    best, best_area = None, 0
    for a in screens:
        o = _overlap(grip, a)
        if o and o[0] * o[1] > best_area:
            best, best_area = a, o[0] * o[1]
    if best is None or best_area < GRIP_WIDTH * GRIP_HEIGHT / 2:
        return {"width": width, "height": height, "maximized": maximized}

    width = min(width, max(best["width"], opts["minWidth"]))
    height = min(height, max(best["height"], opts["minHeight"]))
    nx = max(best["x"], min(x, best["x"] + best["width"] - width))
    ny = max(best["y"], min(y, best["y"] + best["height"] - height))
    return {"x": nx, "y": ny, "width": width, "height": height, "maximized": maximized}


def work_areas() -> list:
    """Рабочие области экранов по Qt, основной первым."""
    from PySide6.QtGui import QGuiApplication

    primary = QGuiApplication.primaryScreen()
    screens = [primary] + [s for s in QGuiApplication.screens() if s is not primary]
    out = []
    for s in screens:
        if s is None:
            continue
        g = s.availableGeometry()
        out.append({"x": g.x(), "y": g.y(), "width": g.width(), "height": g.height()})
    return out


class WindowMemory:
    """Ставит окно на место до показа и запоминает обычные границы (у развёрнутого - те,
    к которым оно вернётся). track() звать из moveEvent и resizeEvent окна, save() -
    из closeEvent."""

    def __init__(self, window, path, opts):
        self.window = window
        self.path = path
        self.normal = None
        placed = restore(load_json(path), work_areas(), opts)
        window.resize(placed["width"], placed["height"])
        if "x" in placed:
            window.move(placed["x"], placed["y"])
        else:
            areas = work_areas()
            if areas:
                a = areas[0]
                window.move(a["x"] + max(0, (a["width"] - placed["width"]) // 2),
                            a["y"] + max(0, (a["height"] - placed["height"]) // 2))
        self.maximize = placed["maximized"]

    def show(self):
        """Показать окно так, как его оставили: развёрнутым или обычным."""
        if self.maximize:
            self.window.showMaximized()
        else:
            self.window.show()

    def track(self):
        w = self.window
        if not (w.isMaximized() or w.isMinimized() or w.isFullScreen()):
            self.normal = {"x": w.x(), "y": w.y(), "width": w.width(), "height": w.height()}

    def save(self):
        self.track()
        if self.normal is None:
            return False
        return save_json(self.path, {**self.normal, "maximized": self.window.isMaximized()})
