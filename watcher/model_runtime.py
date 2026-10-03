"""Сеть в службе: ONNX через onnxruntime, без PyTorch.

У сети два выхода: logits [1, 5] и cam [1, 5, h, w] - карта «где брак» по каждому виду.
Голова сети - свёртка 1x1 по признакам и среднее по карте (см. model/net.py), поэтому
карта класса - это ровно Grad-CAM для линейной головы: вклад каждой клетки в логит.
Отдельный проход с градиентами не нужен, и служба обходится onnxruntime.
"""
import io
import json

import numpy as np
from PIL import Image

from watcher.classes import CLASSES

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_SIZE = (288, 160)   # ширина, высота: 16:9 камеры, кратно 32


def to_rgb(frame):
    """JPEG-байты, PIL.Image или массив HxWx3 -> PIL RGB."""
    if isinstance(frame, (bytes, bytearray)):
        img = Image.open(io.BytesIO(frame))
        img.load()
        return img.convert("RGB")
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    return Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")


def preprocess(img, size=DEFAULT_SIZE):
    """PIL RGB -> float32 [1, 3, H, W], нормировка ImageNet (как при обучении)."""
    x = np.asarray(img.resize(size, Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


class Prediction:
    def __init__(self, probs, cam):
        self.probs = probs          # {класс: вероятность}
        self.cam = cam              # np.ndarray [5, h, w] или None

    @property
    def top(self):
        return max(CLASSES, key=lambda c: self.probs.get(c, 0.0))

    def fail_score(self):
        return max(self.probs.get(c, 0.0) for c in CLASSES[1:])


class Model:
    def __init__(self, path, threads=2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads      # на ПК работают и другие программы
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map or {}
        self.classes = tuple(json.loads(meta["classes"])) if "classes" in meta else CLASSES
        if self.classes != CLASSES:
            raise ValueError(f"сеть обучена на других видах: {self.classes}")
        size = json.loads(meta["input_size"]) if "input_size" in meta else DEFAULT_SIZE
        self.size = (int(size[0]), int(size[1]))
        self.input_name = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.meta = meta

    def predict(self, frame):
        img = to_rgb(frame)
        res = self.session.run(None, {self.input_name: preprocess(img, self.size)})
        out = dict(zip(self.outputs, res))
        logits = out.get("logits", res[0])[0]
        p = softmax(logits)
        cam = out["cam"][0] if "cam" in out else None
        return Prediction({c: float(v) for c, v in zip(self.classes, p)}, cam)
