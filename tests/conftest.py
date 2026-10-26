import os
import sys

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest  # noqa: E402
from hypothesis import settings  # noqa: E402

import offscreen  # noqa: E402

# Без экрана: Qt рисует в память. Ставится до любого импорта Qt.
offscreen.setup()

settings.register_profile("default", max_examples=400, deadline=None)
settings.register_profile("thorough", max_examples=4000, deadline=None)
settings.register_profile("quick", max_examples=60, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(autouse=True)
def _own_data_dir(tmp_path, monkeypatch):
    """Каждый тест пишет настройки, журнал и кадры в свою временную папку, а не в профиль."""
    home = tmp_path / "home"
    monkeypatch.setenv("AIFP_HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    return home


@pytest.fixture
def fake_camera():
    from fake_camera import FakeCameraServer
    srv = FakeCameraServer()
    yield srv
    srv.close()


_app = None


@pytest.fixture
def qapp():
    """Одно QApplication на весь прогон: второе Qt создать не даст."""
    global _app
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance() or QApplication([])
    return _app


def pytest_sessionfinish(session, exitstatus):
    session.config._aifp_exit = int(exitstatus)


def pytest_unconfigure(config):
    """CI GitHub 26.10: после «182 passed» Python падал при выходе (Fatal Python error: Aborted, код 3, на медленной
    машине) - окна и потоки Qt доживали до закрытия интерпретатора. После итогов pytest процесс выходит сразу
    с кодом pytest, если загружен Qt; без Qt - обычный выход."""
    code = getattr(config, "_aifp_exit", None)
    if code is None or "PySide6.QtCore" not in sys.modules:
        return
    import faulthandler
    faulthandler.disable()                # потоки Qt, падающие при выходе, - не шум в логе
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
