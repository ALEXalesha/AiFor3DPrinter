"""Хранилище снимков по печатям, журнал каждого снимка и разбор печати (main.py review).

03.10.2026 печать оторвалась, а в журнале были только «новая печать» и «печать не идёт» -
понять, что видела служба, было не по чему. Теперь каждый снимок пишется в журнал одной
строкой и кладётся в хранилище (локально, вне репозитория, с ограничением размера)."""
import json
import os

from fake_camera import STATUS_IDLE, STATUS_PRINTING, make_jpeg
from helpers import Clock, FakeModel, bad, cfg_loader, quiet_log

from watcher import paths
from watcher.camera_client import CameraClient
from watcher.framestore import FrameStore, list_prints, load_scores
from watcher.review import render_timeline, review
from watcher.service import Service


def rec(t, **kw):
    return dict({"t": t, "layer": 1, "total": 10, "probs": {"normal": 0.9, "spaghetti": 0.1}, "motion": 0.01}, **kw)


def test_store_keeps_frames_and_scores_per_print(tmp_path):
    fs = FrameStore(str(tmp_path))
    fs.start("cube.gcode", 1000.0)
    for i in range(3):
        fs.add(1000.0 + i, make_jpeg(), rec(1000.0 + i))
    prints = list_prints(str(tmp_path))
    assert len(prints) == 1 and prints[0]["key"] == "cube.gcode" and prints[0]["frames"] == 3
    scores = load_scores(prints[0]["dir"])
    assert [s["t"] for s in scores] == [1000.0, 1001.0, 1002.0] and all(os.path.exists(s["frame"]) for s in scores)


def test_store_caps_frames_per_print_but_keeps_scores(tmp_path):
    fs = FrameStore(str(tmp_path), max_frames_per_print=5)
    fs.start("a", 0.0)
    for i in range(12):
        fs.add(float(i), make_jpeg(), rec(float(i)))
    p = list_prints(str(tmp_path))[0]
    assert p["frames"] == 5 and len(load_scores(p["dir"])) == 12
    kept = [s for s in load_scores(p["dir"]) if os.path.exists(s["frame"])]
    assert [s["t"] for s in kept] == [7.0, 8.0, 9.0, 10.0, 11.0]


def test_store_prunes_old_prints_by_count_and_size(tmp_path):
    fs = FrameStore(str(tmp_path), max_prints=3, max_bytes=10**9)
    for k in range(5):
        fs.start(f"p{k}", float(k * 100))
        fs.add(float(k * 100), make_jpeg(), rec(float(k * 100)))
    assert [p["key"] for p in list_prints(str(tmp_path))] == ["p2", "p3", "p4"]
    one = len(make_jpeg())
    fs2 = FrameStore(str(tmp_path), max_prints=50, max_bytes=one * 4)
    fs2.start("big", 999.0)
    for i in range(3):
        fs2.add(999.0 + i, make_jpeg(), rec(999.0 + i))
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fl in os.walk(str(tmp_path)) for f in fl
                if f.endswith(".jpg"))
    assert total <= one * 4 and list_prints(str(tmp_path))[-1]["key"] == "big"


def test_store_ignores_junk_dirs_and_broken_scores(tmp_path):
    os.makedirs(tmp_path / "junk")
    fs = FrameStore(str(tmp_path))
    fs.start("x", 1.0)
    fs.add(1.0, make_jpeg(), rec(1.0))
    d = list_prints(str(tmp_path))[0]["dir"]
    open(os.path.join(d, "scores.jsonl"), "a", encoding="utf-8").write("{broken\n")
    assert len(load_scores(d)) == 1 and len(list_prints(str(tmp_path))) == 1


def make(fake, model=None, **cfg):
    cfg.setdefault("camera_url", fake.url)
    return Service(model or FakeModel(), client=CameraClient(fake.url, {"snap": 3, "status": 3, "pause": 3}),
                   cfg_loader=cfg_loader(**cfg), clock=Clock(), log=quiet_log())


def test_service_logs_every_snapshot_and_stores_it(fake_camera):
    svc = make(fake_camera)
    lines = []
    svc.log.info = lambda msg, *a: lines.append(msg % a if a else msg)
    for _ in range(4):
        svc.step()
    snaps = [ln for ln in lines if ln.startswith("снимок")]
    assert len(snaps) == 4 and "н.97" in snaps[0] and "слой 50/200" in snaps[0]
    p = list_prints(paths.data_path("prints", "x").rsplit(os.sep, 1)[0])
    assert len(p) == 1 and p[0]["frames"] == 4 and p[0]["key"] == "cube.gcode"


def test_new_print_gets_its_own_folder(fake_camera):
    svc = make(fake_camera)
    svc.step()
    fake_camera.status_text = STATUS_IDLE
    svc.step()
    fake_camera.status_text = STATUS_PRINTING
    svc.step()
    root = os.path.join(paths.data_dir(), "prints")
    assert len(list_prints(root)) == 2


def test_early_layers_fast_cadence(fake_camera):
    """Первые слои - снимок раз в 5 с: отрыв 03.10 случился в первые минуты."""
    fake_camera.status_text = STATUS_PRINTING.replace("Слой 50/200", "Слой 3/200")
    svc = make(fake_camera)
    assert svc.step() == (5, "ok")
    fake_camera.status_text = STATUS_PRINTING
    assert svc.step() == (15, "ok")


def test_review_timeline_marks_alarm_and_scores(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad("detached", 0.93)))
    for _ in range(4):
        svc.step()
    root = os.path.join(paths.data_dir(), "prints")
    text = review(root, "cube")
    assert "cube.gcode" in text and "ТРЕВОГА" in text and "от.93" in text
    assert "1." in review(root, None)            # список печатей


def test_review_rescore_uses_given_model(fake_camera):
    svc = make(fake_camera)
    for _ in range(3):
        svc.step()
    root = os.path.join(paths.data_dir(), "prints")
    text = review(root, "1", rescore_model=FakeModel(default=bad("spaghetti", 0.95)))
    assert "сп.95" in text and "ТРЕВОГА" in text


def test_render_timeline_handles_empty():
    assert "нет снимков" in render_timeline({"key": "x", "dir": "", "start": 0}, [])


def test_frames_dir_is_outside_repo_and_gitignored():
    assert not os.path.abspath(paths.data_dir()).startswith(paths.repo_dir())
    ign = open(os.path.join(paths.repo_dir(), ".gitignore"), encoding="utf-8").read()
    assert "prints/" in ign
    json.dumps({})
