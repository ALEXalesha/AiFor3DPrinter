"""Разбор печати: что видела служба на каждом снимке.

    python main.py review              список печатей в хранилище
    python main.py review 3            печать №3 из списка (или часть имени файла)
    python main.py review 3 --rescore  прогнать по сохранённым кадрам текущую сеть и правило

Строка на снимок: время, слой, оценки по видам (н - норма, сп - спагетти, от - отрыв,
ком - ком на сопле, нп - нет пластика, в процентах), «дв» - доля стола, сдвинувшаяся
относительно фона (пока только пишется, порог подбирается по живым снимкам), и тревога.
"""
import os
import time

from watcher.classes import CLASSES
from watcher.decision import Decider
from watcher.framestore import list_prints, load_scores

SHORT = {"normal": "н", "spaghetti": "сп", "detached": "от", "blob": "ком", "no_extrusion": "нп"}


def fmt_probs(p):
    p = p or {}
    return " ".join(f"{SHORT[c]}.{round(float(p.get(c, 0) or 0) * 100):02d}" for c in CLASSES)


def bar(v, width=10):
    n = int(round(max(0.0, min(1.0, float(v))) * width))
    return "#" * n + "." * (width - n)


def render_timeline(meta, scores):
    head = f"{meta.get('key', '?')}  (начало {time.strftime('%d.%m %H:%M:%S', time.localtime(meta.get('start', 0)))})"
    if not scores:
        return head + "\nнет снимков"
    lines = [head, "время     слой      " + " ".join(f"{SHORT[c]:>4}" for c in CLASSES) + "   дв   брак"]
    alarms = []
    best = {c: (0.0, None) for c in CLASSES[1:]}
    for r in scores:
        p = r.get("probs") or {}
        fail = max(float(p.get(c, 0) or 0) for c in CLASSES[1:])
        layer = f"{r.get('layer')}/{r.get('total')}" if r.get("layer") is not None else "-"
        mark = f"  ТРЕВОГА {r['alarm']}" if r.get("alarm") else ""
        if r.get("alarm"):
            alarms.append((r["t"], r["alarm"]))
        for c in CLASSES[1:]:
            v = float(p.get(c, 0) or 0)
            if v > best[c][0]:
                best[c] = (v, r["t"])
        lines.append(f"{time.strftime('%H:%M:%S', time.localtime(r['t']))}  {layer:<8}  {fmt_probs(p)}  "
                     f"дв.{round(float(r.get('motion', 0) or 0) * 100):02d}  {bar(fail)}{mark}")
    dur = scores[-1]["t"] - scores[0]["t"]
    lines.append("")
    lines.append(f"снимков {len(scores)} за {int(dur // 60)} мин {int(dur % 60)} с, тревог {len(alarms)}")
    lines.append("максимум: " + ", ".join(
        f"{SHORT[c]} {v * 100:.0f}%" + (f" в {time.strftime('%H:%M:%S', time.localtime(t))}" if t else "")
        for c, (v, t) in best.items()))
    return "\n".join(lines)


def rescore(scores, model, cfg=None):
    """Те же снимки через другую сеть и правило службы: новые оценки и тревоги."""
    from watcher.model_runtime import to_rgb
    cfg = cfg or {"threshold": 0.7, "window": 4, "need": 3, "cooldown_s": 600}
    d = Decider(cfg["threshold"], cfg["window"], cfg["need"], cfg["cooldown_s"])
    out = []
    for r in scores:
        f = r.get("frame")
        if not f or not os.path.exists(f):
            continue
        with open(f, "rb") as fh:
            pred = model.predict(to_rgb(fh.read()))
        a = d.update(pred.probs, r["t"])
        out.append(dict(r, probs={k: round(v, 4) for k, v in pred.probs.items()}, alarm=a.cls if a else None))
    return out


def pick(prints, which):
    if which is None:
        return None
    w = str(which).strip()
    if w.isdigit() and 1 <= int(w) <= len(prints):
        return prints[int(w) - 1]
    hits = [p for p in prints if w.lower() in p["key"].lower() or w.lower() in os.path.basename(p["dir"]).lower()]
    return hits[-1] if hits else None


def review(root, which=None, rescore_model=None, cfg=None):
    prints = list_prints(root)
    if not prints:
        return "В хранилище нет печатей: " + root
    if which is None:
        rows = []
        for i, p in enumerate(prints, 1):
            sc = load_scores(p["dir"])
            fail = max((max(float((s.get("probs") or {}).get(c, 0) or 0) for c in CLASSES[1:]) for s in sc), default=0)
            al = sum(1 for s in sc if s.get("alarm"))
            rows.append(f"{i}. {time.strftime('%d.%m %H:%M', time.localtime(p['start']))}  {p['key']}  "
                        f"снимков {len(sc)} (кадров {p['frames']}), максимум брака {fail * 100:.0f}%, тревог {al}")
        return "\n".join(rows)
    p = pick(prints, which)
    if p is None:
        return f"Нет печати «{which}». Список: main.py review"
    sc = load_scores(p["dir"])
    if rescore_model is not None:
        sc = rescore(sc, rescore_model, cfg)
    return render_timeline(p, sc)


def main(argv):
    from watcher import config, paths
    root = os.path.join(paths.data_dir(), "prints")
    args = [a for a in argv if not a.startswith("--")]
    model = None
    if "--rescore" in argv:
        from watcher.service import load_model
        model = load_model()
    print(review(root, args[0] if args else None, model, config.load()))
