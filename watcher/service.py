"""Служба: раз в N секунд статус -> снимок -> сеть -> решение по нескольким кадрам -> тревога.

Правила, которые проверяют тесты:
- работает только при «Статус: печатает»; в остальное время спрашивает статус раз в
  IDLE_POLL_S и камеру не трогает;
- camera_server не отвечает - ждёт (с растущей паузой до OFFLINE_MAX_S) и пишет в
  журнал один раз на серию сбоев, а не падает;
- пауза без человека - только если в настройках auto_pause ровно true (по умолчанию нет);
- остановку печати служба не вызывает никогда: у клиента такого запроса нет.
"""
import json
import logging
import logging.handlers
import os
import threading
import time

from watcher import alerts, config, paths
from watcher.camera_client import CameraClient, CameraError
from watcher.classes import FAILURES
from watcher.decision import Decider
from watcher.heatmap import jpeg, overlay
from watcher.model_runtime import to_rgb
from watcher.storage import save_bytes, save_json

IDLE_POLL_S = 30
OFFLINE_MIN_S = 20
OFFLINE_MAX_S = 120


def make_logger(path=None):
    log = logging.getLogger("aifp.watcher")
    if not log.handlers:
        log.setLevel(logging.INFO)
        path = path or paths.data_path("watcher.log")
        h = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(h)
    return log


class Service:
    def __init__(self, model, client=None, cfg_loader=config.load, clock=time.time, log=None):
        self.model = model
        self.cfg_loader = cfg_loader
        self.cfg = cfg_loader()
        self.client = client or CameraClient(self.cfg["camera_url"])
        self.clock = clock
        self.log = log or make_logger()
        self.decider = self._new_decider()
        self.print_key = None
        self.offline_streak = 0
        self.last_state = None
        self.last_layer = None
        self.series = []            # [(t, fail_score, top)] за текущую печать - для графика в окне

    def _new_decider(self):
        c = self.cfg
        return Decider(c["threshold"], c["window"], c["need"], c["cooldown_s"],
                       max_age_s=max(c["interval_s"] * c["window"] * 3, 60))

    def _reload_config(self):
        new = self.cfg_loader()
        keys = ("threshold", "window", "need", "cooldown_s", "interval_s")
        changed = any(new[k] != self.cfg[k] for k in keys)
        if new["camera_url"] != self.cfg["camera_url"]:
            try:
                self.client = CameraClient(new["camera_url"])
            except ValueError as e:
                self.log.warning("адрес camera_server не принят (%s), остаётся прежний", e)
                new["camera_url"] = self.cfg["camera_url"]
        self.cfg = new
        if changed:
            old = self.decider
            self.decider = self._new_decider()
            self.decider.last_alarm_t = old.last_alarm_t

    def _state(self, **kw):
        kw.setdefault("t", self.clock())
        kw["auto_pause"] = self.cfg["auto_pause"]
        save_json(paths.data_path("state.json"), kw)
        return kw

    def step(self):
        """Один шаг. -> (секунд до следующего шага, что произошло)."""
        self._reload_config()
        try:
            alerts.process_inbox(self.cfg, self.log)
        except OSError as e:
            self.log.warning("ответы бота не прочитаны: %s", e)
        now = self.clock()
        try:
            status = self.client.status()
        except CameraError as e:
            self.offline_streak += 1
            if self.offline_streak == 1:
                self.log.warning("camera_server не отвечает: %s - жду", e)
            self._state(state="offline", error=str(e)[:200])
            return min(OFFLINE_MIN_S * self.offline_streak, OFFLINE_MAX_S), "offline"
        if self.offline_streak:
            self.log.info("camera_server снова отвечает (после %d сбоев)", self.offline_streak)
            self.offline_streak = 0

        if not status.printing:
            if self.last_state == "printing":
                self.log.info("печать не идёт (%s) - наблюдение остановлено", status.text or status.state)
                self.decider.reset()
                self.print_key = None
            self.last_state = status.state
            self._state(state=status.state, status=status.text)
            return IDLE_POLL_S, "idle"

        key = status.filename or "?"
        if key != self.print_key or (status.layer is not None and self.last_layer is not None
                                     and status.layer < self.last_layer):
            self.log.info("новая печать: %s - наблюдаю", key)
            self.decider.reset()
            self.series = []
            self.print_key = key
        self.last_state = "printing"
        self.last_layer = status.layer

        try:
            frame = self.client.snapshot()
        except CameraError as e:
            self.log.warning("снимок не получен: %s", e)
            self._state(state="printing", status=status.text, error=str(e)[:200])
            return self.cfg["interval_s"], "no-frame"
        if self.model is None:
            self._state(state="printing", status=status.text, error="нет модели (model/aifp.onnx)")
            return self.cfg["interval_s"], "no-model"
        try:
            pred = self.model.predict(frame)
        except Exception as e:      # битый кадр не должен ронять службу
            self.log.warning("кадр не оценён: %s", e)
            return self.cfg["interval_s"], "bad-frame"

        top_fail = max(FAILURES, key=lambda c: pred.probs.get(c, 0.0))
        img = to_rgb(frame)
        heat = jpeg(overlay(img, pred.cam, top_fail))
        save_bytes(paths.data_path("last.jpg"), frame)
        save_bytes(paths.data_path("last_heat.jpg"), heat)
        self.series.append((now, round(pred.fail_score(), 4), top_fail))
        self.series = self.series[-2000:]
        self._state(state="printing", status=status.text, filename=status.filename, layer=status.layer,
                    total=status.total_layers, probs=pred.probs, top=pred.top, top_fail=top_fail,
                    series=self.series[-400:])

        alarm = self.decider.update(pred.probs, now)
        if alarm is None:
            return self.cfg["interval_s"], "ok"
        heat = jpeg(overlay(img, pred.cam, alarm.cls))
        aid = alerts.save_alert(alarm, frame, heat, status, now, self.cfg)
        self.log.warning("ТРЕВОГА %s: %s %.0f%% (%d из %d кадров)", aid, alarm.cls, alarm.score * 100,
                         alarm.votes, alarm.window)
        if self.cfg["auto_pause"] is True:
            try:
                ok = self.client.pause()
                self.log.warning("автопауза (%s): %s", aid, "поставлена" if ok else "принтер не подтвердил")
            except CameraError as e:
                self.log.warning("автопауза не удалась: %s", e)
        return self.cfg["interval_s"], "alarm"

    def run(self, stop=None, max_steps=None, sleep=None):
        """Цикл службы. stop - threading.Event; ожидание прерывается сразу по stop."""
        stop = stop or threading.Event()
        steps = 0
        self.log.info("служба запущена: %s, автопауза %s", self.cfg["camera_url"],
                      "ВКЛ" if self.cfg["auto_pause"] else "выкл")
        while not stop.is_set():
            try:
                wait, _ = self.step()
            except Exception as e:      # последняя линия: служба не падает молча
                self.log.exception("ошибка шага: %s", e)
                wait = OFFLINE_MIN_S
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
            (sleep or stop.wait)(wait)
        self.log.info("служба остановлена")


def load_model():
    path = paths.model_path()
    if not os.path.exists(path):
        return None
    from watcher.model_runtime import Model
    return Model(path)


def read_state():
    from watcher.storage import load_json
    s = load_json(paths.data_path("state.json"))
    return s if isinstance(s, dict) else {}


def main(argv=None):
    model = load_model()
    svc = Service(model)
    if model is None:
        svc.log.warning("нет обученной сети %s - служба будет только смотреть статус", paths.model_path())
    print(json.dumps({"data": paths.data_dir(), "model": paths.model_path() if model else None,
                      "auto_pause": svc.cfg["auto_pause"]}, ensure_ascii=False))
    try:
        svc.run()
    except KeyboardInterrupt:
        pass
