import json

import pytest

from ui import window_state as ws

AREAS = [{"x": 0, "y": 0, "width": 1920, "height": 1040}]
OPTS = {"width": 900, "height": 700, "minWidth": 640, "minHeight": 560}


@pytest.mark.parametrize("saved", [None, 5, "x", [], {}, {"width": "a", "height": 3},
                                   {"width": float("nan"), "height": 500}, {"width": True, "height": 600}])
def test_garbage_gives_default_size_centered(saved):
    assert ws.restore(saved, AREAS, OPTS) == {"width": 900, "height": 700, "maximized": False}


def test_normal_place_is_kept():
    got = ws.restore({"x": 100, "y": 50, "width": 1000, "height": 800}, AREAS, OPTS)
    assert got == {"x": 100, "y": 50, "width": 1000, "height": 800, "maximized": False}


def test_window_on_a_disconnected_monitor_opens_centered():
    got = ws.restore({"x": 3000, "y": 100, "width": 1000, "height": 800}, AREAS, OPTS)
    assert "x" not in got and got["width"] == 1000


def test_too_big_window_is_cut_to_the_screen():
    got = ws.restore({"x": 0, "y": 0, "width": 5000, "height": 4000}, AREAS, OPTS)
    assert got["width"] == 1920 and got["height"] == 1040


def test_too_small_window_grows_to_the_minimum():
    got = ws.restore({"x": 10, "y": 10, "width": 10, "height": 10}, AREAS, OPTS)
    assert got["width"] == 640 and got["height"] == 560


def test_window_half_off_screen_is_pulled_in():
    got = ws.restore({"x": 1500, "y": 900, "width": 1000, "height": 800}, AREAS, OPTS)
    assert got["x"] == 920 and got["y"] == 240


def test_title_bar_above_the_screen_does_not_count():
    got = ws.restore({"x": 100, "y": -500, "width": 1000, "height": 800}, AREAS, OPTS)
    assert "x" not in got


def test_second_monitor_is_used():
    areas = AREAS + [{"x": 1920, "y": 0, "width": 1280, "height": 1000}]
    got = ws.restore({"x": 2000, "y": 100, "width": 1000, "height": 800}, areas, OPTS)
    assert got["x"] == 2000


def test_maximized_flag_survives():
    got = ws.restore({"x": 0, "y": 0, "width": 1000, "height": 800, "maximized": True}, AREAS, OPTS)
    assert got["maximized"] is True


def test_qt_window_opens_where_it_was_closed(qapp, tmp_path):
    from PySide6.QtWidgets import QWidget

    path = tmp_path / "window.json"
    opts = {"width": 500, "height": 400, "minWidth": 300, "minHeight": 200}
    w = QWidget()
    mem = ws.WindowMemory(w, path, opts)
    w.show()
    w.move(60, 30)
    w.resize(520, 410)
    qapp.processEvents()
    mem.track()
    assert mem.save()
    w.close()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert (saved["x"], saved["y"], saved["width"], saved["height"]) == (60, 30, 520, 410)

    w2 = QWidget()
    ws.WindowMemory(w2, path, opts)
    w2.show()
    qapp.processEvents()
    assert (w2.x(), w2.y(), w2.width(), w2.height()) == (60, 30, 520, 410)
    w2.close()


def test_qt_window_with_broken_file_opens_with_default_size(qapp, tmp_path):
    from PySide6.QtWidgets import QWidget

    path = tmp_path / "window.json"
    path.write_text("{обрыв", encoding="utf-8")
    w = QWidget()
    ws.WindowMemory(w, path, {"width": 500, "height": 400, "minWidth": 300, "minHeight": 200})
    w.show()
    qapp.processEvents()
    assert (w.width(), w.height()) == (500, 400)
    w.close()
