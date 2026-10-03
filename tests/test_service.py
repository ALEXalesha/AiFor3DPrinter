"""Служба против поддельного camera_server: только во время печати, тревога, файл боту, ответы кнопок."""
import json
import os

from fake_camera import STATUS_IDLE, STATUS_PRINTING
from helpers import Clock, FakeModel, bad, cfg_loader, quiet_log

from watcher import alerts, paths
from watcher.camera_client import CameraClient
from watcher.service import Service


def make(fake, model=None, **cfg):
    cfg.setdefault("camera_url", fake.url)
    load = cfg_loader(**cfg)
    return Service(model or FakeModel(), client=CameraClient(fake.url, {"snap": 3, "status": 3, "pause": 3}),
                   cfg_loader=load, clock=Clock(), log=quiet_log())


def test_idle_printer_no_snapshots(fake_camera):
    fake_camera.status_text = STATUS_IDLE
    svc = make(fake_camera)
    for _ in range(5):
        wait, what = svc.step()
        assert what == "idle" and wait == 30
    assert set(fake_camera.routes) == {"status"}


def test_printing_takes_snapshot_and_writes_state(fake_camera):
    svc = make(fake_camera)
    wait, what = svc.step()
    assert what == "ok" and wait == 15
    assert fake_camera.routes == ["status", "snap"]
    st = json.load(open(paths.data_path("state.json"), encoding="utf-8"))
    assert st["state"] == "printing" and st["filename"] == "cube.gcode" and st["auto_pause"] is False
    assert os.path.exists(paths.data_path("last_heat.jpg"))


def test_offline_waits_and_backs_off(fake_camera):
    fake_camera.mode = "503"
    svc = make(fake_camera)
    waits = [svc.step()[0] for _ in range(8)]
    assert waits == sorted(waits) and waits[-1] == 120 and waits[0] == 20


def test_alarm_goes_to_bot_outbox_and_history(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad("blob", 0.95)))
    results = [svc.step()[1] for _ in range(4)]
    assert results == ["ok", "ok", "alarm", "ok"]
    out = os.path.join(os.environ["USERPROFILE"], "n8n_pc", "run", "aifp", "outbox")
    files = sorted(os.listdir(out))
    assert len(files) == 2 and files[0].endswith(".jpg") and files[1].endswith(".json")
    rec = json.load(open(os.path.join(out, files[1]), encoding="utf-8"))
    assert rec["cls"] == "blob" and "ком на сопле" in rec["caption"] and alerts.ID_RE.match(rec["id"])
    assert len(alerts.load_history()) == 1
    assert "pause" not in fake_camera.routes


def test_telegram_off_no_outbox(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad()), telegram=False)
    for _ in range(4):
        svc.step()
    assert not os.path.exists(os.path.join(os.environ["USERPROFILE"], "n8n_pc", "run", "aifp", "outbox"))
    assert len(alerts.load_history()) == 1


def test_autopause_on_pauses_once_per_alarm(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad()), auto_pause=True)
    for _ in range(6):
        svc.step()
    assert fake_camera.routes.count("pause") == 1


def test_ok_button_moves_frame_to_feedback(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad()))
    for _ in range(3):
        svc.step()
    aid = alerts.load_history()[-1]["id"]
    inbox = os.path.join(os.environ["USERPROFILE"], "n8n_pc", "run", "aifp", "inbox")
    os.makedirs(inbox)
    open(os.path.join(inbox, aid + ".ok"), "w").write("1")
    open(os.path.join(inbox, "x.ok"), "w").write("1")
    svc.step()
    assert os.path.exists(os.path.join(alerts.feedback_dir(), aid + ".jpg"))
    assert alerts.load_history()[-1]["answer"] == "ok"
    assert sorted(os.listdir(os.path.join(inbox, "bad"))) == ["x.ok"]
    assert os.listdir(os.path.join(inbox, "done")) == [aid + ".ok"]


def test_new_print_resets_votes(fake_camera):
    svc = make(fake_camera, FakeModel(seq=[bad(), bad()], default=bad()))
    svc.step()
    svc.step()
    fake_camera.status_text = STATUS_PRINTING.replace("cube.gcode", "other.gcode")
    assert svc.step()[1] == "ok"       # голоса старой печати не считаются


def test_print_end_resets(fake_camera):
    svc = make(fake_camera, FakeModel(default=bad()))
    svc.step()
    svc.step()
    fake_camera.status_text = STATUS_IDLE
    svc.step()
    fake_camera.status_text = STATUS_PRINTING
    assert [svc.step()[1] for _ in range(3)] == ["ok", "ok", "alarm"]


def test_no_model_still_watches_status(fake_camera):
    svc = make(fake_camera)
    svc.model = None
    assert svc.step()[1] == "no-model"


def test_broken_model_does_not_crash(fake_camera):
    class Boom:
        def predict(self, f):
            raise RuntimeError("boom")
    svc = make(fake_camera, Boom())
    assert svc.step()[1] == "bad-frame"


def test_run_stops_with_max_steps(fake_camera):
    fake_camera.status_text = STATUS_IDLE
    svc = make(fake_camera)
    waits = []
    svc.run(max_steps=3, sleep=waits.append)
    assert waits == [30, 30]
