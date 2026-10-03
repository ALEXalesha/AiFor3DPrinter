"""Где лежат данные: настройки, журнал, кадры тревог, набор данных, чекпойнты.

Всё - в %LOCALAPPDATA%\\AiForPrinter, вне репозитория и вне синхронизируемых папок:
кадры тяжёлые и на них видно камеру принтера. Переменная AIFP_HOME подменяет папку
целиком - так тесты не трогают настоящие данные.
"""
import os
import sys

APP_NAME = "AiForPrinter"
ENV = "AIFP_HOME"


def frozen():
    return getattr(sys, "frozen", False)


def repo_dir():
    """Папка с кодом (или распакованной сборкой)."""
    if frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource(*parts):
    return os.path.join(repo_dir(), *parts)


def pick_data_dir(env):
    if env.get(ENV):
        return env[ENV]
    base = env.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, APP_NAME)


def data_dir():
    return pick_data_dir(os.environ)


def data_path(*parts, make_parent=True):
    path = os.path.join(data_dir(), *parts)
    if make_parent:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def model_path():
    """Обученная сеть: своя в папке данных (после дообучения) важнее той, что в репозитории."""
    own = os.path.join(data_dir(), "model", "aifp.onnx")
    if os.path.exists(own):
        return own
    return resource("model", "aifp.onnx")
