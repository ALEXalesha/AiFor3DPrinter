"""Паспорт модели: проверка ONNX (ровно того, что запускает служба) на печатях вне обучения.

    python -m model.evaluate [--model model/aifp.onnx]

Главная цифра - ложные тревоги на час нормальной печати. Считается так: тестовые печати
(их моделей не было ни в обучении, ни в выборе чекпойнта) прогоняются в реальном
времени печати. Кадр таймлапса - один на слой, поэтому каждый кадр «держится» столько
секунд, сколько в среднем шёл слой (время печати из имени файла / число кадров), а
служба смотрит раз в interval_s. Соседние снимки службы тогда - один и тот же кадр:
это худший случай для правила «3 из 4», ошибка одного кадра повторяется. Поверх -
то же решение по кадрам (watcher.decision.Decider), что в службе.

Скорость обнаружения: с 40% печати кадры заменяются искусственным браком этого вида,
считается время от начала брака до тревоги.
"""
import argparse
import csv
import json
import os
import time
import zlib
from collections import defaultdict

import numpy as np
from PIL import Image

from dataset.common import dataset_dir
from watcher import paths
from watcher.classes import CLASSES, FAILURES, name
from watcher.decision import Decider
from watcher.model_runtime import Model, preprocess


def predict_paths(model, items, batch=32):
    """items: пути или массивы -> np.ndarray [N, 5] вероятностей."""
    out = []
    for i in range(0, len(items), batch):
        xs = []
        for it in items[i:i + batch]:
            img = Image.open(it).convert("RGB") if isinstance(it, str) else Image.fromarray(it)
            xs.append(preprocess(img, model.size)[0])
        logits = model.session.run(["logits"], {model.input_name: np.stack(xs)})[0]
        z = logits - logits.max(1, keepdims=True)
        e = np.exp(z)
        out.append(e / e.sum(1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def simulate(probs_seq, duration_s, interval_s, thr, window, need, cooldown_s, start_frame=None):
    """Печать в реальном времени: -> (число тревог, [время тревог от начала])."""
    n = len(probs_seq)
    if n == 0 or not duration_s:
        return 0, []
    period = duration_s / n
    d = Decider(thr, window, need, cooldown_s, max_age_s=max(interval_s * window * 3, 60))
    alarms = []
    t = 0.0
    while t < duration_s:
        i = min(int(t / period), n - 1)
        p = dict(zip(CLASSES, probs_seq[i]))
        if d.update(p, t):
            alarms.append(t)
        t += interval_s
    return len(alarms), alarms


def frame_metrics(y, p, thr):
    pred = p.argmax(1)
    res = {}
    for c, cls in enumerate(CLASSES):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        n = int((y == c).sum())
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / (tp + fn) if tp + fn else None
        r = {"n": n, "precision": prec, "recall": rec}
        if c:
            r["detected_at_thr"] = float((p[y == c][:, 1:].max(1) >= thr).mean()) if n else None
        res[cls] = r
    normal = y == 0
    res["normal"]["false_positive_rate_at_thr"] = float((p[normal][:, 1:].max(1) >= thr).mean()) if normal.any() else None
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    return res, conf.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=paths.resource("model", "aifp.onnx"))
    ap.add_argument("--interval", type=float, default=15)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t_start = time.time()
    model = Model(a.model, threads=a.threads)
    root = dataset_dir()
    prints = json.load(open(os.path.join(root, "prints.json"), encoding="utf-8"))
    rows = [r for r in csv.DictReader(open(os.path.join(root, "index.csv"), encoding="utf-8")) if r["split"] == "test"]
    print(f"тестовых кадров {len(rows)}", flush=True)
    probs = predict_paths(model, [r["path"] for r in rows])
    y = np.array([CLASSES.index(r["cls"]) for r in rows])
    cfg = {"threshold": 0.7, "window": 4, "need": 3, "cooldown_s": 600}

    own = np.array([r["source"] != "skz" for r in rows])
    metrics, conf = frame_metrics(y[own], probs[own], cfg["threshold"])
    skz = ~own
    skz_rec = {"n": int(skz.sum()),
               "detected_at_thr": float((probs[skz][:, 1:].max(1) >= cfg["threshold"]).mean()) if skz.any() else None,
               "argmax_spaghetti": float((probs[skz].argmax(1) == 1).mean()) if skz.any() else None}

    # нормальные кадры тестовых печатей по порядку
    seq = defaultdict(list)
    for r, p in zip(rows, probs):
        if r["source"] == "timelapse":
            seq[r["print"]].append((int(r["frame"]), p))
    test_prints = sorted(seq)
    hours = sum(prints[pid]["duration_s"] for pid in test_prints) / 3600

    def false_alarms(thr, window, need):
        total = 0
        per = {}
        for pid in test_prints:
            s = [p for _, p in sorted(seq[pid], key=lambda t: t[0])]
            k, _ = simulate(s, prints[pid]["duration_s"], a.interval, thr, window, need, cfg["cooldown_s"])
            total += k
            if k:
                per[prints[pid]["group"]] = per.get(prints[pid]["group"], 0) + k
        return total, per

    fa, fa_per = false_alarms(cfg["threshold"], cfg["window"], cfg["need"])
    sweep = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        for window, need in ((4, 3), (3, 2), (5, 4), (6, 5)):
            k, _ = false_alarms(thr, window, need)
            sweep.append({"threshold": thr, "rule": f"{need}/{window}", "false_alarms": k,
                          "per_hour": round(k / hours, 3) if hours else None})

    # скорость обнаружения на искусственном браке
    from dataset import synth
    delays = {c: [] for c in FAILURES}
    missed = {c: 0 for c in FAILURES}
    frames_root = os.path.join(root, "frames")
    for pid in test_prints:
        files = sorted(os.listdir(os.path.join(frames_root, pid)))
        n = len(files)
        if n < 12:
            continue
        k0 = int(n * 0.4)
        base = [p for _, p in sorted(seq[pid], key=lambda t: t[0])]
        for ci, cls in enumerate(FAILURES):
            rng = np.random.default_rng(zlib.crc32(f"{pid}|{cls}".encode()))
            fakes, idx = [], []
            for i in range(k0, n):
                img = synth.as_rgb(Image.open(os.path.join(frames_root, pid, files[i])).convert("RGB"))
                out = synth.spaghetti(img, rng) if cls == "spaghetti" else synth.GENERATORS[cls](img, rng)
                if out is not None:
                    fakes.append(out)
                    idx.append(i)
            s = list(base)
            if fakes:
                pf = predict_paths(model, fakes)
                for i, p in zip(idx, pf):
                    s[i] = p
            dur = prints[pid]["duration_s"]
            start_t = k0 * dur / n
            _, al = simulate(s, dur, a.interval, cfg["threshold"], cfg["window"], cfg["need"], cfg["cooldown_s"])
            after = [t for t in al if t >= start_t]
            if after:
                delays[cls].append(after[0] - start_t)
            else:
                missed[cls] += 1

    def dstat(v, m):
        if not v and not m:
            return None
        v = sorted(v)
        return {"prints": len(v) + m, "detected": len(v), "median_s": round(v[len(v) // 2]) if v else None,
                "within_5min": round(sum(1 for d in v if d <= 300) / (len(v) + m), 3)}

    passport = {
        "model": os.path.basename(a.model), "meta": model.meta, "evaluated": time.strftime("%Y-%m-%d %H:%M"),
        "service_rule": cfg, "interval_s": a.interval,
        "test_prints": len(test_prints), "test_hours": round(hours, 2),
        "false_alarms": fa, "false_alarms_per_hour": round(fa / hours, 3) if hours else None,
        "false_alarm_groups": fa_per,
        "frames": metrics, "confusion": conf, "skz_real_spaghetti": skz_rec,
        "detection_delay": {c: dstat(delays[c], missed[c]) for c in FAILURES},
        "sweep": sweep, "seconds": round(time.time() - t_start),
    }
    out_json = paths.resource("model", "passport.json")
    json.dump(passport, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    open(paths.resource("model", "PASSPORT.md"), "w", encoding="utf-8", newline="\n").write(render(passport))
    print(render(passport))


def _pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def render(P):
    r = P["service_rule"]
    L = ["# Паспорт модели", "",
         f"Модель `{P['model']}`, проверена {P['evaluated']}. Правило службы: брак при уверенности "
         f"от {r['threshold']}, {r['need']} из {r['window']} кадров, снимок раз в {P['interval_s']:.0f} с, "
         f"пауза между тревогами {r['cooldown_s'] // 60} мин.", "",
         "## Ложные тревоги на час нормальной печати", "",
         f"**{P['false_alarms_per_hour']} в час** - {P['false_alarms']} тревог за {P['test_hours']} ч "
         f"нормальной печати ({P['test_prints']} тестовых печатей, модели которых сеть не видела).", "",
         "Каждый кадр таймлапса держится всё время своего слоя, служба смотрит раз в "
         f"{P['interval_s']:.0f} с - это худший случай для правила «{r['need']} из {r['window']}».", ""]
    if P["false_alarm_groups"]:
        L.append("Где были ложные тревоги: " + ", ".join(f"{g} ({k})" for g, k in P["false_alarm_groups"].items()) + ".")
        L.append("")
    L += ["## Кадры: точность и полнота по видам", "",
          "Норма - настоящие кадры тестовых печатей; брак - искусственный, нарисованный поверх них "
          "(цифры по браку поэтому оптимистичны).", "",
          "| Вид | Кадров | Точность | Полнота | Замечен при пороге |", "|---|---:|---:|---:|---:|"]
    for cls in CLASSES:
        m = P["frames"][cls]
        det = _pct(m.get("detected_at_thr")) if cls != "normal" else "-"
        L.append(f"| {name(cls)} | {m['n']} | {_pct(m['precision'])} | {_pct(m['recall'])} | {det} |")
    L += ["", f"Нормальный кадр принят за брак (порог {r['threshold']}): "
          f"{_pct(P['frames']['normal']['false_positive_rate_at_thr'])} кадров.", ""]
    s = P["skz_real_spaghetti"]
    if s["n"]:
        L += ["## Настоящие спагетти (набор SKZ, другая камера)", "",
              f"{s['n']} тестовых кадров: брак замечен на {_pct(s['detected_at_thr'])}, "
              f"как «спагетти» - {_pct(s['argmax_spaghetti'])}.", ""]
    L += ["## Как быстро замечается брак", "",
          "С 40% печати кадры заменены искусственным браком; время от начала брака до тревоги.", "",
          "| Вид | Печатей | Замечен | Медиана | За 5 минут |", "|---|---:|---:|---:|---:|"]
    for cls, d in P["detection_delay"].items():
        if d:
            med = "-" if d["median_s"] is None else f"{d['median_s'] // 60} мин {d['median_s'] % 60} с"
            L.append(f"| {name(cls)} | {d['prints']} | {d['detected']} | {med} | {_pct(d['within_5min'])} |")
    L += ["", "## Порог и правило: ложные тревоги в час", "", "| Порог | 3/4 | 2/3 | 4/5 | 5/6 |", "|---|---:|---:|---:|---:|"]
    by = defaultdict(dict)
    for x in P["sweep"]:
        by[x["threshold"]][x["rule"]] = x["per_hour"]
    for thr in sorted(by):
        L.append(f"| {thr} | " + " | ".join(str(by[thr].get(k, "-")) for k in ("3/4", "2/3", "4/5", "5/6")) + " |")
    L += ["", "Проверка на настоящих печатях с камеры (неделя наблюдений, шаг 8 плана) ещё впереди: "
          "живой снимок отличается от таймлапса (голова ездит, другая выдержка)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
