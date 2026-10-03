"""Окно AiForPrinter: живой снимок и тепловая карта, уверенность по видам, график за печать,
история тревог, настройки службы. Темы Paint, RU/EN, окно запоминает место и размер.

Окно не управляет принтером. Оно читает то, что пишет служба (state.json, last*.jpg,
история тревог), правит config.json и по кнопке «Снимок сейчас» само берёт /status и
/snap у camera_server - только чтение.
"""
import os
import sys
import time

from PySide6.QtCore import QObject, QPointF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

from ui import theme
from ui.i18n import tr
from ui.window_state import WindowMemory
from watcher import alerts, config, paths
from watcher.camera_client import check_base_url
from watcher.classes import CLASSES, FAILURES, name
from watcher.storage import load_json

STALE_S = 180          # state.json старше - служба не запущена


class Combo(QComboBox):
    """Список со своей стрелкой: без правила ::drop-down Fusion рисует квадратную кнопку."""

    def paintEvent(self, e):
        super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        c = self.palette().color(self.foregroundRole())
        p.setBrush(c)
        p.setPen(Qt.NoPen)
        x, y = self.width() - 17, self.height() / 2 - 2
        p.drawPolygon(QPolygonF([QPointF(x, y), QPointF(x + 9, y), QPointF(x + 4.5, y + 5)]))
        p.end()


def card():
    f = QFrame()
    f.setObjectName("card")
    return f


def wrap_label(text="", obj=None):
    lab = QLabel(text)
    lab.setWordWrap(True)
    if obj:
        lab.setObjectName(obj)
    return lab


class Chart(QWidget):
    """Брак за печать: максимум по видам брака на каждом снимке и линия порога."""

    def __init__(self):
        super().__init__()
        self.series = []
        self.threshold = 0.7
        self.colors = theme.palette("glass")
        self.setMinimumHeight(140)

    def set_data(self, series, threshold, colors):
        self.series = [s for s in series if isinstance(s, (list, tuple)) and len(s) >= 2]
        self.threshold = threshold
        self.colors = colors
        self.update()

    def points(self, w, h, pad=8):
        if len(self.series) < 2:
            return []
        t0, t1 = self.series[0][0], self.series[-1][0]
        span = max(t1 - t0, 1e-9)
        return [QPointF(pad + (t - t0) / span * (w - 2 * pad), h - pad - max(0.0, min(1.0, float(v))) * (h - 2 * pad))
                for t, v, *_ in self.series]

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h, pad = self.width(), self.height(), 8
        p.setPen(QPen(QColor(self.colors["grid"]), 1))
        for k in (0.25, 0.5, 0.75):
            y = h - pad - k * (h - 2 * pad)
            p.drawLine(QPointF(pad, y), QPointF(w - pad, y))
        ty = h - pad - self.threshold * (h - 2 * pad)
        pen = QPen(QColor(self.colors["danger"]), 1, Qt.DashLine)
        p.setPen(pen)
        p.drawLine(QPointF(pad, ty), QPointF(w - pad, ty))
        pts = self.points(w, h, pad)
        if pts:
            path = QPainterPath(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.setPen(QPen(QColor(self.colors["accent"]), 2))
            p.drawPath(path)
        p.end()


class CheckWorker(QObject):
    """«Снимок сейчас»: /status и /snap у camera_server и оценка сетью в фоне."""
    done = Signal(dict)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        from watcher.camera_client import CameraClient, CameraError
        from watcher.service import load_model
        out = {}
        try:
            c = CameraClient(self.url)
            st = c.status()
            out["status"] = st.state
            frame = c.snapshot()
            out["frame"] = frame
            model = load_model()
            if model is not None:
                pred = model.predict(frame)
                out["probs"] = pred.probs
                from watcher.heatmap import jpeg, overlay
                from watcher.model_runtime import to_rgb
                top = max(FAILURES, key=lambda k: pred.probs.get(k, 0))
                out["heat"] = jpeg(overlay(to_rgb(frame), pred.cam, top))
        except (CameraError, ValueError) as e:
            out["error"] = str(e)
        except Exception as e:      # сеть или кадр - показать, не ронять окно
            out["error"] = f"{type(e).__name__}: {e}"
        self.done.emit(out)


class MainWindow(QMainWindow):
    def __init__(self, remember=True, confirm=None):
        super().__init__()
        self.cfg = config.load()
        self.lang = self.cfg["lang"]
        self.confirm = confirm or self._confirm_dialog
        self.check_thread = None
        self.last_check = None
        self.setWindowTitle("AiForPrinter")
        icon = paths.resource("ui", "icon.png")
        if os.path.exists(icon):
            self.setWindowIcon(QIcon(icon))
        self.setMinimumSize(760, 560)
        self.build()
        self.apply_theme(self.cfg["theme"])
        self.memory = WindowMemory(self, paths.data_path("window.json"),
                                   {"width": 1100, "height": 760, "minWidth": 760, "minHeight": 560}) if remember else None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(3000)
        self.refresh()

    # ---------- построение ----------
    def build(self):
        page = QWidget()
        page.setObjectName("page")
        root = QVBoxLayout(page)
        root.setContentsMargins(18, 14, 18, 14)
        top = QHBoxLayout()
        self.title = QLabel("AiForPrinter")
        self.title.setObjectName("title")
        self.state_label = QLabel()
        self.state_label.setObjectName("stateIdle")
        top.addWidget(self.title)
        top.addSpacing(16)
        top.addWidget(self.state_label, 1)
        self.theme_box = Combo()
        for key in theme.TOKENS:
            self.theme_box.addItem(theme.NAMES[self.lang][key], key)
        self.theme_box.setCurrentIndex(list(theme.TOKENS).index(self.cfg["theme"]))
        self.theme_box.currentIndexChanged.connect(lambda _i: self.set_theme(self.theme_box.currentData()))
        self.lang_box = Combo()
        self.lang_box.addItem("Русский", "ru")
        self.lang_box.addItem("English", "en")
        self.lang_box.setCurrentIndex(0 if self.lang == "ru" else 1)
        self.lang_box.currentIndexChanged.connect(lambda _i: self.set_lang(self.lang_box.currentData()))
        top.addWidget(self.theme_box)
        top.addWidget(self.lang_box)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_live(), "")
        self.tabs.addTab(self.build_alerts(), "")
        self.tabs.addTab(self.build_settings(), "")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(page)
        self.retranslate()

    def build_live(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        left = QVBoxLayout()
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(480, 270)
        self.image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.image, 1)
        row = QHBoxLayout()
        self.heat_box = QCheckBox()
        self.heat_box.setChecked(True)
        self.heat_box.toggled.connect(lambda _v: self.refresh())
        self.check_btn = QPushButton()
        self.check_btn.setObjectName("primary")
        self.check_btn.clicked.connect(self.check_now)
        row.addWidget(self.heat_box)
        row.addStretch(1)
        row.addWidget(self.check_btn)
        left.addLayout(row)
        self.chart_title = wrap_label(obj="muted")
        left.addWidget(self.chart_title)
        self.chart = Chart()
        left.addWidget(self.chart)
        lay.addLayout(left, 3)

        side = card()
        sl = QVBoxLayout(side)
        self.conf_title = wrap_label()
        sl.addWidget(self.conf_title)
        self.bars, self.bar_labels = {}, {}
        grid = QGridLayout()
        for i, cls in enumerate(CLASSES):
            lab = QLabel()
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setTextVisible(False)
            if cls != "normal":
                bar.setObjectName("fail")
            grid.addWidget(lab, i, 0)
            grid.addWidget(bar, i, 1)
            self.bars[cls], self.bar_labels[cls] = bar, lab
        sl.addLayout(grid)
        self.detail = wrap_label(obj="muted")
        sl.addWidget(self.detail)
        sl.addStretch(1)
        self.passport_label = wrap_label(obj="muted")
        sl.addWidget(self.passport_label)
        lay.addWidget(side, 2)
        return w

    def build_alerts(self):
        w = QWidget()
        lay = QHBoxLayout(w)
        self.alert_list = QListWidget()
        self.alert_list.setIconSize(QPixmap(160, 90).size())
        self.alert_list.currentRowChanged.connect(self.show_alert)
        lay.addWidget(self.alert_list, 2)
        self.alert_image = QLabel()
        self.alert_image.setAlignment(Qt.AlignCenter)
        self.alert_image.setMinimumSize(360, 200)
        lay.addWidget(self.alert_image, 3)
        self.alert_ids = []
        return w

    def build_settings(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        w = QWidget()
        w.setObjectName("page")
        outer.setWidget(w)
        form = QGridLayout(w)
        form.setColumnStretch(1, 1)
        self.set_labels = {}

        def row(key, widget, r):
            lab = wrap_label()
            self.set_labels[key] = lab
            form.addWidget(lab, r, 0)
            form.addWidget(widget, r, 1)

        self.interval = QSpinBox()
        self.interval.setRange(*config.LIMITS["interval_s"])
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(*config.LIMITS["threshold"])
        self.threshold.setSingleStep(0.05)
        self.threshold.setDecimals(2)
        self.window_spin = QSpinBox()
        self.window_spin.setRange(*config.LIMITS["window"])
        self.need_spin = QSpinBox()
        self.need_spin.setRange(*config.LIMITS["need"])
        self.cooldown = QSpinBox()
        self.cooldown.setRange(0, 24 * 60)
        self.url = QLineEdit()
        self.telegram = QCheckBox()
        self.autopause = QCheckBox()
        self.autopause.toggled.connect(self.on_autopause)
        for r, (key, wdg) in enumerate([("interval", self.interval), ("threshold", self.threshold),
                                        ("window", self.window_spin), ("need", self.need_spin),
                                        ("cooldown", self.cooldown), ("camera_url", self.url)]):
            row(key, wdg, r)
        form.addWidget(self.telegram, 6, 0, 1, 2)
        form.addWidget(self.autopause, 7, 0, 1, 2)
        self.autopause_hint = wrap_label(obj="muted")
        form.addWidget(self.autopause_hint, 8, 0, 1, 2)
        self.save_btn = QPushButton()
        self.save_btn.setObjectName("primary")
        self.save_btn.clicked.connect(self.save_settings)
        form.addWidget(self.save_btn, 9, 0)
        self.save_msg = wrap_label(obj="muted")
        form.addWidget(self.save_msg, 9, 1)
        form.setRowStretch(10, 1)
        self.load_settings()
        return outer

    # ---------- тексты, тема ----------
    def retranslate(self):
        L = self.lang
        for i, key in enumerate(("tab_live", "tab_alerts", "tab_settings")):
            self.tabs.setTabText(i, tr(key, L))
        self.heat_box.setText(tr("heatmap", L))
        self.check_btn.setText(tr("check_now", L))
        self.chart_title.setText(tr("chart", L))
        self.conf_title.setText(tr("confidence", L))
        for cls, lab in self.bar_labels.items():
            lab.setText(name(cls, L))
        for key, lab in self.set_labels.items():
            lab.setText(tr(key, L))
        self.telegram.setText(tr("telegram", L))
        self.autopause.setText(tr("autopause", L))
        self.autopause_hint.setText(tr("autopause_hint", L))
        self.save_btn.setText(tr("save", L))
        for i in range(self.theme_box.count()):
            self.theme_box.setItemText(i, theme.NAMES[L][self.theme_box.itemData(i)])
        self.update_passport()

    def apply_theme(self, key):
        self.theme_key = key if key in theme.TOKENS else "glass"
        self.colors = theme.palette(self.theme_key)
        self.setStyleSheet(theme.qss(self.theme_key))

    def set_theme(self, key):
        self.apply_theme(key)
        self.cfg = config.load()
        self.cfg["theme"] = self.theme_key
        config.save(self.cfg)
        self.refresh()

    def set_lang(self, lang):
        self.lang = lang if lang in config.LANGS else "ru"
        self.cfg = config.load()
        self.cfg["lang"] = self.lang
        config.save(self.cfg)
        self.retranslate()
        self.refresh()

    # ---------- настройки ----------
    def load_settings(self):
        c = config.load()
        self.interval.setValue(c["interval_s"])
        self.threshold.setValue(c["threshold"])
        self.window_spin.setValue(c["window"])
        self.need_spin.setValue(c["need"])
        self.cooldown.setValue(c["cooldown_s"] // 60)
        self.url.setText(c["camera_url"])
        self.telegram.setChecked(c["telegram"])
        self.autopause.blockSignals(True)
        self.autopause.setChecked(c["auto_pause"] is True)
        self.autopause.blockSignals(False)

    def _confirm_dialog(self, text):
        return QMessageBox.question(self, "AiForPrinter", text) == QMessageBox.Yes

    def on_autopause(self, on):
        if on and not self.confirm(tr("autopause_confirm", self.lang)):
            self.autopause.blockSignals(True)
            self.autopause.setChecked(False)
            self.autopause.blockSignals(False)

    def save_settings(self):
        try:
            check_base_url(self.url.text())
        except ValueError:
            self.save_msg.setText(tr("bad_url", self.lang))
            return False
        c = config.load()
        c.update({"interval_s": self.interval.value(), "threshold": round(self.threshold.value(), 2),
                  "window": self.window_spin.value(), "need": self.need_spin.value(),
                  "cooldown_s": self.cooldown.value() * 60, "camera_url": self.url.text().strip(),
                  "telegram": self.telegram.isChecked(), "auto_pause": self.autopause.isChecked() is True})
        ok = config.save(c)
        self.cfg = config.load()
        self.load_settings()
        self.save_msg.setText(tr("saved", self.lang) if ok else "!")
        return ok

    # ---------- обновление ----------
    def state(self):
        s = load_json(paths.data_path("state.json"))
        return s if isinstance(s, dict) else {}

    def refresh(self):
        L = self.lang
        st = self.state()
        age = time.time() - float(st.get("t", 0) or 0)
        kind = st.get("state") if st and age < STALE_S else "none"
        key = {"printing": "st_printing", "idle": "st_idle", "paused": "st_paused", "preparing": "st_preparing",
               "offline": "st_offline", "none": "st_none"}.get(kind, "st_unknown")
        text = tr(key, L)
        if kind == "printing" and st.get("filename"):
            text += f" · {st['filename']}"
            if st.get("layer") is not None and st.get("total"):
                text += f" · {st['layer']}/{st['total']}"
        if st.get("error") and kind != "none":
            text += f" · {st['error']}"
        self.state_label.setText(text)
        self.state_label.setObjectName({"printing": "stateOk", "offline": "stateBad"}.get(kind, "stateIdle"))
        self.state_label.setStyle(self.state_label.style())

        probs = st.get("probs") if kind == "printing" else None
        if self.last_check and self.last_check.get("probs") and (not probs):
            probs = self.last_check["probs"]
        for cls in CLASSES:
            v = float((probs or {}).get(cls, 0) or 0)
            self.bars[cls].setValue(int(max(0.0, min(1.0, v)) * 1000))
            self.bars[cls].setToolTip(f"{v * 100:.1f}%")
        if probs:
            top = max(CLASSES, key=lambda c: probs.get(c, 0))
            self.detail.setText(f"{name(top, L)}: {probs.get(top, 0) * 100:.0f}%")
        else:
            self.detail.setText("")
        self.chart.set_data((st.get("series") or []) if kind == "printing" else [], self.cfg["threshold"], self.colors)
        self.show_frame()
        self.refresh_alerts()

    def show_frame(self):
        data = None
        if self.last_check and self.last_check.get("frame"):
            data = self.last_check.get("heat" if self.heat_box.isChecked() else "frame") or self.last_check["frame"]
        else:
            path = paths.data_path("last_heat.jpg" if self.heat_box.isChecked() else "last.jpg")
            if os.path.exists(path):
                try:
                    data = open(path, "rb").read()
                except OSError:
                    data = None
        if not data:
            self.image.setPixmap(QPixmap())
            self.image.setText(tr("no_frame", self.lang))
            return
        img = QImage.fromData(data)
        if img.isNull():
            return
        pm = QPixmap.fromImage(img).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image.setPixmap(pm)

    def refresh_alerts(self):
        hist = list(reversed(alerts.load_history(200)))
        ids = [h["id"] for h in hist]
        if ids == self.alert_ids and self.alert_list.count() == max(1, len(ids)):
            return
        self.alert_ids = ids
        self.alert_list.blockSignals(True)
        self.alert_list.clear()
        if not hist:
            self.alert_list.addItem(tr("no_alerts", self.lang))
        for h in hist:
            ans = {"ok": "answer_ok", "pause": "answer_pause"}.get(h.get("answer"), "answer_none")
            when = time.strftime("%d.%m %H:%M", time.localtime(float(h.get("t", 0) or 0)))
            text = f"{when}  {name(h.get('cls', ''), self.lang)} {float(h.get('score', 0)) * 100:.0f}%\n{tr(ans, self.lang)}"
            if h.get("filename"):
                text += f" · {h['filename']}"
            item = QListWidgetItem(text)
            thumb = os.path.join(alerts.alerts_dir(), h["id"], "heat.jpg")
            if os.path.exists(thumb):
                item.setIcon(QIcon(QPixmap(thumb).scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            self.alert_list.addItem(item)
        self.alert_list.blockSignals(False)

    def show_alert(self, row):
        if not (0 <= row < len(self.alert_ids)):
            return
        path = os.path.join(alerts.alerts_dir(), self.alert_ids[row], "heat.jpg")
        pm = QPixmap(path)
        if not pm.isNull():
            self.alert_image.setPixmap(pm.scaled(self.alert_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_passport(self):
        P = load_json(paths.resource("model", "passport.json"))
        if not isinstance(P, dict):
            self.passport_label.setText(tr("passport_none", self.lang))
            return
        fa = P.get("false_alarms_per_hour")
        self.passport_label.setText(f"{tr('passport', self.lang)}: {fa} {tr('fa_per_hour', self.lang)} "
                                    f"({P.get('test_hours')} h, {P.get('evaluated', '')})")

    # ---------- «Снимок сейчас» ----------
    def check_now(self, wait=False):
        if self.check_thread is not None:
            return
        self.check_btn.setEnabled(False)
        self.check_btn.setText(tr("checking", self.lang))
        self.worker = CheckWorker(config.load()["camera_url"])
        self.check_thread = QThread(self)
        self.worker.moveToThread(self.check_thread)
        self.check_thread.started.connect(self.worker.run)
        self.worker.done.connect(self.on_checked)
        self.check_thread.start()
        if wait:
            deadline = time.time() + 60
            while self.check_thread is not None and time.time() < deadline:
                QApplication.processEvents()
                time.sleep(0.02)

    def on_checked(self, out):
        self.last_check = out
        if self.check_thread is not None:
            self.check_thread.quit()
            self.check_thread.wait(5000)
            self.check_thread = None
        self.check_btn.setEnabled(True)
        self.check_btn.setText(tr("check_now", self.lang))
        if out.get("error"):
            self.detail.setText(out["error"])
        self.refresh()
        if out.get("error"):
            self.detail.setText(out["error"])

    # ---------- окно ----------
    def moveEvent(self, e):
        if self.memory:
            self.memory.track()
        super().moveEvent(e)

    def resizeEvent(self, e):
        if self.memory:
            self.memory.track()
        super().resizeEvent(e)
        self.show_frame()

    def closeEvent(self, e):
        self.timer.stop()
        if self.check_thread is not None:
            self.check_thread.quit()
            self.check_thread.wait(3000)
        if self.memory:
            self.memory.save()
        super().closeEvent(e)


def run():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    if w.memory:
        w.memory.show()
    else:
        w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
