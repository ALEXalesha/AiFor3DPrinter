"""Инструмент разметки: кадр на экране, клавиша 0-4 - вид, ←/→ - листать.

    python -m dataset.label_tool [папка с кадрами]

По умолчанию открывает кадры из кнопок Telegram (feedback/normal) и кадры тревог
(alerts/*/frame.jpg). Разметка пишется в dataset/labels.csv в папке данных:
путь, вид. dataset/build.py добавляет её в обучение (часть train).
"""
import glob
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from watcher import paths
from dataset.labels import load_labels, save_labels
from watcher.classes import CLASSES, name


def default_frames():
    d = paths.data_dir()
    return sorted(glob.glob(os.path.join(d, "feedback", "normal", "*.jpg")) +
                  glob.glob(os.path.join(d, "alerts", "*", "frame.jpg")))


class LabelTool(QWidget):
    def __init__(self, files, labels_file=None):
        super().__init__()
        self.setWindowTitle("AiForPrinter - разметка")
        self.files = files
        self.labels_file = labels_file
        self.labels = load_labels(labels_file)
        self.i = 0
        self.image = QLabel(alignment=Qt.AlignCenter)
        self.image.setMinimumSize(640, 360)
        self.info = QLabel()
        lay = QVBoxLayout(self)
        lay.addWidget(self.image, 1)
        lay.addWidget(self.info)
        row = QHBoxLayout()
        self.buttons = []
        for k, cls in enumerate(CLASSES):
            b = QPushButton(f"{k} {name(cls)}")
            b.clicked.connect(lambda _=False, c=cls: self.set_label(c))
            row.addWidget(b)
            self.buttons.append(b)
            QShortcut(QKeySequence(str(k)), self, activated=lambda c=cls: self.set_label(c))
        lay.addLayout(row)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=lambda: self.go(1))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=lambda: self.go(-1))
        self.show_current()

    def current(self):
        return self.files[self.i] if self.files else None

    def set_label(self, cls):
        f = self.current()
        if f is None:
            return
        self.labels[f] = cls
        save_labels(self.labels, self.labels_file)
        self.go(1)

    def go(self, d):
        if self.files:
            self.i = max(0, min(len(self.files) - 1, self.i + d))
        self.show_current()

    def show_current(self):
        f = self.current()
        if f is None:
            self.info.setText("Нет кадров для разметки.")
            return
        pm = QPixmap(f)
        if not pm.isNull():
            self.image.setPixmap(pm.scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lab = self.labels.get(f)
        self.info.setText(f"{self.i + 1}/{len(self.files)}  {os.path.basename(f)}  "
                          f"{'- ' + name(lab) if lab else '- не размечен'}")


def main(argv):
    files = sorted(glob.glob(os.path.join(argv[1], "*.jpg"))) if len(argv) > 1 else default_frames()
    app = QApplication.instance() or QApplication(argv)
    w = LabelTool(files)
    w.resize(900, 600)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
