"""Qt без экрана - для тестов, самопроверки и кадров README.

У платформы offscreen своя база шрифтов, и пустая: вместо букв рисуются квадратики.
Ей показывается папка шрифтов Windows - тогда текст на кадрах тот же, что в окне.
Звать до создания QApplication.
"""
import os


def setup(force=False):
    if force:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fonts = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    if os.path.isdir(fonts):
        os.environ.setdefault("QT_QPA_FONTDIR", fonts)
