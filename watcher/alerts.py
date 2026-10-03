"""Тревоги: кадры на диск, файл для бота n8n, ответы кнопок обратно.

Обмен с ботом - через папку (по умолчанию n8n_pc\\run\\aifp), как у ответа Claude в боте:
токен Telegram и chat id живут только в n8n, служба их не знает и знать не должна.

    outbox/<id>.jpg   кадр с тепловой картой (пишется первым)
    outbox/<id>.json  тревога; появление json = «можно отправлять»
    inbox/<id>.ok     владелец нажал «Всё нормально» -> кадр уходит в набор как норма
    inbox/<id>.pause  владелец нажал «Пауза» (паузу ставит сам бот через camera_server)

Бот переименовывает json в .sent ДО отправки (не больше одного раза, как Telegram
offset в боте). id - только [0-9a-z-]: из него бот строит имя файла.
"""
import json
import os
import re
import secrets
import shutil
import time

from watcher import paths
from watcher.classes import name
from watcher.storage import load_json, save_bytes, save_json

ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-z]{4}$")


def new_id(t=None):
    return time.strftime("%Y%m%d-%H%M%S", time.localtime(t if t is not None else time.time())) + "-" + \
        "".join(secrets.choice("0123456789abcdefghijklmnopqrstuvwxyz") for _ in range(4))


def caption(alarm, status, t, lang="ru"):
    pct = round(alarm.score * 100)
    lines = [f"⚠️ Похоже на брак: {name(alarm.cls, lang)} ({pct}%, {alarm.votes} из {alarm.window} кадров)"]
    if status is not None and status.filename:
        where = f"🖨 {status.filename}"
        if status.layer is not None and status.total_layers:
            where += f", слой {status.layer}/{status.total_layers}"
        lines.append(where)
    lines.append(time.strftime("%H:%M", time.localtime(t)))
    lines.append("⏸ Пауза - поставить печать на паузу. ✅ Всё нормально - кадр пойдёт в обучение, "
                 "а этот вид брака не будет тревожить до конца печати.")
    return "\n".join(lines)[:1000]


def _dir(*parts):
    d = os.path.join(paths.data_dir(), *parts)
    os.makedirs(d, exist_ok=True)
    return d


def alerts_dir():
    return _dir("alerts")


def save_alert(alarm, frame_jpeg, heat_jpeg, status, t, cfg):
    """Кадр и запись о тревоге в папку данных, при включённом Telegram - файл боту. -> id."""
    aid = new_id(t)
    adir = os.path.join(alerts_dir(), aid)
    save_bytes(os.path.join(adir, "frame.jpg"), frame_jpeg)
    save_bytes(os.path.join(adir, "heat.jpg"), heat_jpeg)
    rec = {
        "id": aid, "t": t, "cls": alarm.cls, "name": name(alarm.cls, "ru"), "score": round(alarm.score, 4),
        "votes": alarm.votes, "window": alarm.window,
        "filename": status.filename if status else "", "layer": status.layer if status else None,
        "total": status.total_layers if status else None,
        "caption": caption(alarm, status, t), "answer": None, "sent_to_bot": False,
    }
    if cfg.get("telegram"):
        from watcher.config import bot_dir
        out = os.path.join(bot_dir(cfg), "outbox")
        ok = save_bytes(os.path.join(out, aid + ".jpg"), heat_jpeg) and save_json(
            os.path.join(out, aid + ".json"),
            {k: rec[k] for k in ("id", "t", "cls", "name", "score", "caption")})
        rec["sent_to_bot"] = bool(ok)
    save_json(os.path.join(adir, "alert.json"), rec)
    append_history(rec)
    return aid


def load_alert(aid):
    if not ID_RE.match(str(aid)):
        return None
    rec = load_json(os.path.join(alerts_dir(), aid, "alert.json"))
    return rec if isinstance(rec, dict) else None


def history_path():
    return paths.data_path("alerts", "history.jsonl")


def append_history(rec):
    try:
        with open(history_path(), "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_history(limit=200):
    out = []
    try:
        with open(history_path(), encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and ID_RE.match(str(rec.get("id", ""))):
                    out.append(rec)
    except OSError:
        return []
    answers = {}
    for rec in out:
        if rec.get("answer_for"):
            answers[rec["answer_for"]] = rec.get("answer")
    alerts = [r for r in out if not r.get("answer_for")]
    for r in alerts:
        if r["id"] in answers:
            r["answer"] = answers[r["id"]]
    return alerts[-limit:]


def feedback_dir():
    """Кадры, отмеченные владельцем «Всё нормально»: пример без брака для дообучения."""
    return _dir("feedback", "normal")


def process_inbox(cfg, log=None):
    """Ответы кнопок из бота. -> список (id, ответ). Чужие и битые файлы - в inbox/bad."""
    from watcher.config import bot_dir
    inbox = os.path.join(bot_dir(cfg), "inbox")
    done = []
    try:
        names = sorted(os.listdir(inbox))
    except OSError:
        return done
    for fn in names:
        full = os.path.join(inbox, fn)
        if not os.path.isfile(full):
            continue
        aid, _, kind = fn.rpartition(".")
        if not ID_RE.match(aid) or kind not in ("ok", "pause"):
            _move(full, os.path.join(inbox, "bad", fn))
            continue
        adir = os.path.join(alerts_dir(), aid)
        if kind == "ok":
            src = os.path.join(adir, "frame.jpg")
            if os.path.exists(src):
                os.makedirs(feedback_dir(), exist_ok=True)
                shutil.copyfile(src, os.path.join(feedback_dir(), aid + ".jpg"))
        rec = load_json(os.path.join(adir, "alert.json"))
        if isinstance(rec, dict):
            rec["answer"] = kind
            save_json(os.path.join(adir, "alert.json"), rec)
        append_history({"id": aid, "answer_for": aid, "answer": kind, "t": time.time()})
        _move(full, os.path.join(inbox, "done", fn))
        if log:
            log.info("ответ на тревогу %s: %s", aid, "всё нормально (кадр в набор)" if kind == "ok" else "пауза")
        done.append((aid, kind))
    return done


def _move(src, dst):
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.replace(src, dst)
    except OSError:
        pass
