"""Чекпойнт PyTorch -> model/aifp.onnx (два выхода: logits и cam) и сверка с onnxruntime.

    python -m model.export <путь к best.pt> [--out model/aifp.onnx]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from model.net import INPUT_SIZE, Exportable, load_checkpoint
from watcher import paths
from watcher.classes import CLASSES
from watcher.version import VERSION


def export(ckpt, out):
    net, ck = load_checkpoint(ckpt)
    model = Exportable(net).eval()
    x = torch.randn(1, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    torch.onnx.export(model, x, out, input_names=["image"], output_names=["logits", "cam"],
                      dynamic_axes={"image": {0: "n"}, "logits": {0: "n"}, "cam": {0: "n"}},
                      opset_version=17, dynamo=False)
    import onnx
    m = onnx.load(out)
    meta = {"classes": json.dumps(list(CLASSES)), "input_size": json.dumps(list(INPUT_SIZE)),
            "version": VERSION, "epoch": str(ck.get("epoch")), "trained": time.strftime("%Y-%m-%d"),
            "license": "weights CC BY-SA 4.0 (trained on SKZ CC BY-SA 4.0 data), code MIT"}
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.save(m, out)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    xs = torch.randn(2, 3, INPUT_SIZE[1], INPUT_SIZE[0])
    with torch.no_grad():
        lt, ct = model(xs)
    lo, co = s.run(None, {"image": xs.numpy()})
    err = max(float(np.abs(lt.numpy() - lo).max()), float(np.abs(ct.numpy() - co).max()))
    print(f"ONNX: {out} ({os.path.getsize(out) / 1e6:.1f} МБ), расхождение с torch {err:.2e}")
    assert err < 1e-3, err
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=paths.resource("model", "aifp.onnx"))
    a = ap.parse_args()
    export(a.ckpt, a.out)
