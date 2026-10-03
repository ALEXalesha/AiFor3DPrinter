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
from watcher.framestore import FrameStore
from watcher.motion import MotionWatch
from watcher.review import fmt_probs
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
        self.early_decider = self._new_decider(early=True)
        self.print_key = None
        self.offline_streak = 0
        self.last_state = None
        self.last_layer = None
        self.series = []            # [(t, fail_score, top)] за текущую печать - для графика в окне
        self.store = FrameStore(os.path.join(paths.data_dir(), "prints"))
        self.motion = MotionWatch()

    def _new_decider(self, early=False):
        c = self.cfg
        if early:
            return Decider(c["early_threshold"], c["early_window"], c["early_need"], c["cooldown_s"],
                           max_age_s=max(c["early_interval_s"] * c["early_window"] * 3, 30))
        return Decider(c["threshold"], c["window"], c["need"], c["cooldown_s"],
                       max_age_s=max(c["interval_s"] * c["window"] * 3, 60))

    def _deciders(self):
        return (self.decider, self.early_decider)

    def _reload_config(self):
        new = self.cfg_loader()
        keys = ("threshold", "window", "need", "cooldown_s", "interval_s", "early_interval_s", "early_threshold",
                "early_window", "early_need")
        changed = any(new[k] != self.cfg[k] for k in keys)
        if new["camera_url"] != self.cfg["camera_url"]:
            try:
                self.client = CameraClient(new["camera_url"])
            except ValueError as e:
                self.log.warning("адрес camera_server не принят (%s), остаётся прежний", e)
                new["camera_url"] = self.cfg["camera_url"]
        self.cfg = new
        if changed:
            for attr, early in (("decider", False), ("early_decider", True)):
                old = getattr(self, attr)
                new_d = self._new_decider(early)
                new_d.last_alarm_t = old.last_alarm_t
                new_d.muted = set(old.muted)
                setattr(self, attr, new_d)

    def _state(self, **kw):
        kw.setdefault("t", self.clock())
        kw["auto_pause"] = self.cfg["auto_pause"]
        save_json(paths.data_path("state.json"), kw)
        return kw

    def _apply_answer(self, aid, kind):
        """«Всё нормально» на тревогу текущей печати - этот вид брака молчит до её конца."""
        if kind != "ok":
            return
        rec = alerts.load_alert(aid)
        if rec and self.print_key and (rec.get("filename") or "?") == self.print_key:
            for d in self._deciders():
                d.mute(rec.get("cls"))
            self.log.info("«%s» заглушён до конца печати %s (ответ на %s)", rec.get("cls"), self.print_key, aid)

    def step(self):
        """Один шаг. -> (секунд до следующего шага, что произошло)."""
        self._reload_config()
        try:
            answers = alerts.process_inbox(self.cfg, self.log)
        except OSError as e:
            answers = []
            self.log.warning("ответы бота не прочитаны: %s", e)
        for aid, kind in answers:
            self._apply_answer(aid, kind)
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
                for d in self._deciders():
                    d.reset()
                self.motion.reset()
                self.print_key = None
            self.last_state = status.state
            self._state(state=status.state, status=status.text)
            return IDLE_POLL_S, "idle"

        key = status.filename or "?"
        if key != self.print_key or (status.layer is not None and self.last_layer is not None
                                     and status.layer < self.last_layer):
            self.log.info("новая печать: %s - наблюдаю", key)
            for d in self._deciders():
                d.reset()
            self.motion.reset()
            self.series = []
            self.print_key = key
            self.store.start(key, now)
        self.last_state = "printing"
        self.last_layer = status.layer

        early = status.layer is not None and status.layer <= self.cfg["early_layers"]
        interval = self.cfg["early_interval_s"] if early else self.cfg["interval_s"]
        layer_txt = f"{status.layer}/{status.total_layers}" if status.layer is not None else "-"
        try:
            frame = self.client.snapshot()
        except CameraError as e:
            self.log.warning("снимок не получен (слой %s): %s", layer_txt, e)
            self._state(state="printing", status=status.text, error=str(e)[:200])
            return interval, "no-frame"
        rec = {"t": now, "layer": status.layer, "total": status.total_layers}
        try:
            img = to_rgb(frame)
            rec["motion"] = round(self.motion.update(img)[0], 4)
        except Exception as e:      # битый кадр не должен ронять службу
            self.store.add(now, frame, dict(rec, error=str(e)[:200]))
            self.log.warning("снимок слой %s: кадр не читается: %s", layer_txt, e)
            return interval, "bad-frame"
        if self.model is None:
            self.store.add(now, frame, rec)
            self.log.info("снимок слой %s: нет сети, дв.%02d", layer_txt, round(rec["motion"] * 100))
            self._state(state="printing", status=status.text, error="нет модели (model/aifp.onnx)")
            return interval, "no-model"
        try:
            pred = self.model.predict(frame)
        except Exception as e:      # битый кадр не должен ронять службу
            self.store.add(now, frame, dict(rec, error=str(e)[:200]))
            self.log.warning("снимок слой %s: кадр не оценён: %s", layer_txt, e)
            return interval, "bad-frame"
        rec["probs"] = {k: round(v, 4) for k, v in pred.probs.items()}

        top_fail = max(FAILURES, key=lambda c: pred.probs.get(c, 0.0))
        heat = jpeg(overlay(img, pred.cam, top_fail))
        save_bytes(paths.data_path("last.jpg"), frame)
        save_bytes(paths.data_path("last_heat.jpg"), heat)
        self.series.append((now, round(pred.fail_score(), 4), top_fail))
        self.series = self.series[-2000:]
        self._state(state="printing", status=status.text, filename=status.filename, layer=status.layer,
                    total=status.total_layers, probs=pred.probs, top=pred.top, top_fail=top_fail,
                    muted=sorted(self.decider.muted | self.early_decider.muted), series=self.series[-400:])

        # первые слои - своё, более чувствительное правило; пауза между тревогами общая
        alarm = (self.early_decider if early else self.decider).update(pred.probs, now)
        if alarm:
            for d in self._deciders():
                d.last_alarm_t = now
        rec["alarm"] = alarm.cls if alarm else None
        self.store.add(now, frame, rec)
        self.log.info("снимок слой %s %s дв.%02d%s", layer_txt, fmt_probs(pred.probs), round(rec["motion"] * 100),
                      f" ТРЕВОГА {alarm.cls}" if alarm else "")
        if alarm is None:
            return interval, "ok"
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
        return interval, "alarm"

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
