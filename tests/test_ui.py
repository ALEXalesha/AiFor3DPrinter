"""Окно без экрана (Qt offscreen): строится во всех темах и языках, показывает состояние службы,
автопауза только с подтверждением, «Снимок сейчас» только читает camera_server."""
import json
import os
import time

import pytest
from fake_camera import make_jpeg

from ui import theme
from ui.app import MainWindow
from watcher import alerts, config, paths
from watcher.decision import Alarm
from watcher.status import PrinterStatus

PROBS = {"normal": 0.1, "spaghetti": 0.8, "detached": 0.05, "blob": 0.03, "no_extrusion": 0.02}


def write_state(**kw):
    s = {"t": time.time(), "state": "printing", "filename": "cube.gcode", "layer": 5, "total": 50, "probs": PROBS,
         "series": [[time.time() - 60 + i * 15, 0.1 * i, "spaghetti"] for i in range(5)]}
    s.update(kw)
    json.dump(s, open(paths.data_path("state.json"), "w", encoding="utf-8"))


@pytest.fixture
def win(qapp):
    w = MainWindow(remember=False, confirm=lambda text: False)
    yield w
    w.close()


@pytest.mark.parametrize("th", list(theme.TOKENS))
@pytest.mark.parametrize("lang", ["ru", "en"])
def test_builds_in_every_theme_and_language(qapp, th, lang):
    config.save(dict(config.DEFAULTS, theme=th, lang=lang))
    w = MainWindow(remember=False)
    assert w.tabs.count() == 3 and w.theme_key == th and w.lang == lang
    assert "QComboBox::drop-down" in w.styleSheet()
    w.close()


def test_state_printing_shows_probs_and_file(win):
    write_state()
    win.refresh()
    assert "cube.gcode" in win.state_label.text() and "5/50" in win.state_label.text()
    assert win.bars["spaghetti"].value() == 800 and win.bars["normal"].value() == 100
    assert len(win.chart.series) == 5


def test_stale_state_means_service_not_running(win):
    write_state(t=time.time() - 3600)
    win.refresh()
    assert win.state_label.text() == "Служба не запущена"
    assert all(b.value() == 0 for b in win.bars.values()) and win.chart.series == []


def test_offline_state_shown(win):
    write_state(state="offline", error="/status: timed out")
    win.refresh()
    assert "camera_server не отвечает" in win.state_label.text()


def test_autopause_declined_stays_off(win):
    win.autopause.setChecked(True)
    assert not win.autopause.isChecked()
    win.save_settings()
    assert config.load()["auto_pause"] is False


def test_autopause_confirmed_saved(qapp):
    w = MainWindow(remember=False, confirm=lambda text: "паузу" in text)
    w.autopause.setChecked(True)
    assert w.autopause.isChecked()
    assert w.save_settings() and config.load()["auto_pause"] is True
    w.close()


def test_settings_saved_normalized(win):
    win.interval.setValue(20)
    win.window_spin.setValue(3)
    win.need_spin.setValue(3)
    win.cooldown.setValue(5)
    win.telegram.setChecked(False)
    assert win.save_settings()
    c = config.load()
    assert c["interval_s"] == 20 and c["window"] == 3 and c["need"] == 3 and c["cooldown_s"] == 300
    assert c["telegram"] is False and c["auto_pause"] is False


def test_bad_camera_url_not_saved(win):
    win.url.setText("http://127.0.0.1:8099/stopprint?x=")
    assert win.save_settings() is False
    assert config.load()["camera_url"] == config.DEFAULTS["camera_url"]


def test_theme_and_language_switch_persist(win):
    win.theme_box.setCurrentIndex(list(theme.TOKENS).index("warm"))
    assert win.theme_key == "warm" and config.load()["theme"] == "warm"
    win.lang_box.setCurrentIndex(1)
    assert win.tabs.tabText(0) == "Now" and config.load()["lang"] == "en"


def test_alert_history_listed_with_answer(win):
    st = PrinterStatus("printing", "печатает", 3, 10, "a.gcode")
    aid = alerts.save_alert(Alarm("blob", 0.91, 3, 4, time.time()), make_jpeg(), make_jpeg(), st, time.time(),
                            dict(config.DEFAULTS, telegram=False))
    alerts.append_history({"id": aid, "answer_for": aid, "answer": "ok", "t": time.time()})
    win.refresh()
    assert win.alert_list.count() == 1
    text = win.alert_list.item(0).text()
    assert "ком на сопле" in text and "91%" in text and "всё нормально" in text


def test_check_now_reads_only(win, fake_camera):
    c = config.load()
    c["camera_url"] = fake_camera.url
    config.save(c)
    win.check_now(wait=True)
    assert win.last_check and "frame" in win.last_check and not win.last_check.get("error")
    assert set(fake_camera.routes) <= {"status", "snap"}


def test_check_now_camera_down_shows_error(win):
    c = config.load()
    c["camera_url"] = "http://127.0.0.1:1"
    config.save(c)
    win.check_now(wait=True)
    assert win.last_check.get("error")


def test_wrapped_labels_never_fixed_height(win):
    from PySide6.QtWidgets import QLabel
    for lab in win.findChildren(QLabel):
        if lab.wordWrap():
            assert lab.maximumHeight() == 16777215, lab.text()


def test_min_size_with_bigger_fonts_fits(win):
    win.resize(win.minimumSize())
    win.setStyleSheet(win.styleSheet() + "* { font-size: 15px; }")
    win.tabs.setCurrentIndex(2)
    win.show()
    from PySide6.QtWidgets import QApplication
    QApplication.processEvents()
    for lab in [win.autopause_hint] + list(win.set_labels.values()):
        if lab.isVisible() and lab.width() > 0:
            assert lab.heightForWidth(lab.width()) <= lab.height() + 1, lab.text()
    win.hide()


def test_window_never_shown_by_tests(win):
    assert not win.isVisible()
    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
