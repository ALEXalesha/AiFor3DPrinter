"""Клиент camera_server против поддельного сервера: снимок, статус, пауза, тайм-ауты, мусор."""
import time

import pytest
from fake_camera import STATUS_IDLE, make_jpeg

from watcher.camera_client import CameraClient, CameraError, ForbiddenEndpoint, check_base_url
from watcher.status import parse_status


def test_snapshot_returns_jpeg(fake_camera):
    c = CameraClient(fake_camera.url)
    assert c.snapshot() == fake_camera.snap
    assert fake_camera.routes == ["snap"]


def test_status_printing_and_idle(fake_camera):
    c = CameraClient(fake_camera.url)
    s = c.status()
    assert s.printing and s.layer == 50 and s.total_layers == 200 and s.filename == "cube.gcode"
    fake_camera.status_text = STATUS_IDLE
    assert c.status().state == "idle"


def test_pause_hits_pause_only(fake_camera):
    assert CameraClient(fake_camera.url).pause() is True
    assert fake_camera.routes == ["pause"]


@pytest.mark.parametrize("mode", ["503", "garbage", "drop"])
def test_bad_answers_raise_camera_error(fake_camera, mode):
    fake_camera.mode = mode
    c = CameraClient(fake_camera.url)
    with pytest.raises(CameraError):
        c.snapshot()


def test_status_garbage_is_unknown_not_crash(fake_camera):
    fake_camera.mode = "garbage"
    assert CameraClient(fake_camera.url).status().state == "unknown"


def test_hang_times_out_quickly(fake_camera):
    fake_camera.mode = "hang"
    fake_camera.hang_s = 3
    c = CameraClient(fake_camera.url, timeouts={"snap": 0.5, "status": 0.5, "pause": 0.5})
    t0 = time.monotonic()
    with pytest.raises(CameraError):
        c.status()
    assert time.monotonic() - t0 < 2.5


def test_server_down_is_camera_error():
    c = CameraClient("http://127.0.0.1:1", timeouts={"snap": 1, "status": 1, "pause": 1})
    with pytest.raises(CameraError):
        c.status()


def test_truncated_jpeg_rejected(fake_camera):
    fake_camera.snap = make_jpeg()[:-50]
    with pytest.raises(CameraError):
        CameraClient(fake_camera.url).snapshot()


def test_unknown_endpoint_is_refused_before_network(fake_camera):
    c = CameraClient(fake_camera.url)
    for ep in ("stopprint", "resume", "print", "files", "snap/../stopprint", ""):
        with pytest.raises(ForbiddenEndpoint):
            c.url_for(ep)
    assert fake_camera.routes == []


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8099/stopprint?x=", "http://127.0.0.1:8099/stopprint", "http://127.0.0.1:8099?a",
    "http://127.0.0.1:8099/#x", "https://127.0.0.1:8099", "ftp://h", "127.0.0.1:8099", "", "http://",
    "http://u:p@127.0.0.1:8099", "http://127.0.0.1:99999",
])
def test_bad_base_urls_rejected(url):
    with pytest.raises(ValueError):
        check_base_url(url)


def test_good_base_urls_normalized():
    assert check_base_url("http://127.0.0.1:8099") == "http://127.0.0.1:8099"
    assert check_base_url("http://127.0.0.1:8099/") == "http://127.0.0.1:8099"
    assert check_base_url(" http://localhost ") == "http://localhost:80"


@pytest.mark.parametrize("text,state", [
    ("Статус: печатает", "printing"), ("Статус: пауза", "paused"), ("Статус: ставится на паузу", "paused"),
    ("Статус: нагрев", "preparing"), ("Статус: калибровка стола", "preparing"), ("Статус: простаивает", "idle"),
    ("Статус: завершено", "idle"), ("Статус: остановлено", "idle"), ("Статус: код 42", "unknown"),
    ("Ошибка: нет связи", "unknown"), ("", "unknown"),
])
def test_status_words(text, state):
    assert parse_status(text).state == state


def test_status_non_string():
    assert parse_status(None).state == "unknown"
    assert parse_status(b"x").state == "unknown"
