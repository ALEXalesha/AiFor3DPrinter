"""Закон по отрыву 03.10.2026 на настоящих кадрах (только на ПК с набором данных, в CI - пропуск).

Деталь первых слоёв оторвалась, сопло таскало её по столу - служба промолчала. Теперь:
на тестовых печатях (их модели сеть не видела) с 5% печати деталь «таскают» по столу,
и служба с правилом 3 из 4 и снимком раз в 5 с на первых слоях должна поднять тревогу
не позже чем через DRAG_MAX_S после начала - на большинстве печатей; и молчать до начала.
"""
import csv
import json
import os
import zlib

import numpy as np
import pytest
from PIL import Image

from watcher import paths

DRAG_MAX_S = 120
ROOT = os.path.join(os.environ.get("LOCALAPPDATA", ""), "AiForPrinter", "dataset")
MODEL = paths.resource("model", "aifp.onnx")
need = pytest.mark.skipif(not (os.path.exists(os.path.join(ROOT, "index.csv")) and os.path.exists(MODEL)),
                          reason="нет локального набора данных или сети")


@need
def test_law_dragged_part_alarms_within_two_minutes():
    from dataset import synth
    from model.evaluate import predict_paths, simulate
    from watcher.model_runtime import Model
    prints = json.load(open(os.path.join(ROOT, "prints.json"), encoding="utf-8"))
    test = sorted(pid for pid, p in prints.items() if p["split"] == "test")
    model = Model(MODEL, threads=4)
    ok = total = early_false = 0
    for pid in test:
        files = sorted(os.listdir(os.path.join(ROOT, "frames", pid)))
        n = len(files)
        if n < 20:
            continue
        k0 = max(3, int(n * 0.05))
        end = min(n, k0 + 12)
        normal = predict_paths(model, [os.path.join(ROOT, "frames", pid, f) for f in files[:end]])
        rng = np.random.default_rng(zlib.crc32(pid.encode()))
        seq = None
        fakes = []
        for f in files[k0:end]:
            img = synth.as_rgb(Image.open(os.path.join(ROOT, "frames", pid, f)).convert("RGB"))
            seq = seq or synth.DragSequence(img, rng)
            fakes.append(seq.apply(img))
        probs = list(normal[:k0]) + list(predict_paths(model, fakes))
        dur = prints[pid]["duration_s"] * end / n
        _, al = simulate(probs, dur, 15, 0.7, 4, 3, 600)
        start = prints[pid]["duration_s"] * k0 / n
        early_false += any(t < start for t in al)
        total += 1
        ok += any(start <= t <= start + DRAG_MAX_S for t in al)
    assert total >= 15
    assert ok / total >= 0.8, f"{ok} из {total}"
    assert early_false <= 1, early_false


def test_rows_are_csv_readable_when_present():
    p = os.path.join(ROOT, "index.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            assert next(csv.reader(f))[:3] == ["path", "cls", "split"]
