"""Общие подделки для тестов службы: сеть с заданными ответами, часы, конфиг."""
import itertools
import logging

import numpy as np

from watcher import config
from watcher.classes import CLASSES
from watcher.model_runtime import Prediction

NORMAL = {"normal": 0.97, "spaghetti": 0.01, "detached": 0.01, "blob": 0.005, "no_extrusion": 0.005}


def bad(cls="spaghetti", p=0.9):
    d = {c: (1 - p) / 4 for c in CLASSES}
    d[cls] = p
    return d


class FakeModel:
    def __init__(self, seq=None, default=None):
        self.seq = list(seq or [])
        self.default = default or NORMAL
        self.calls = 0

    def predict(self, frame):
        self.calls += 1
        p = self.seq.pop(0) if self.seq else self.default
        cam = np.zeros((5, 5, 9), np.float32)
        cam[:, 2, 4] = 1.0
        return Prediction(dict(p), cam)


class Clock:
    def __init__(self, t=1_800_000_000.0, step=15.0):
        self.t = t
        self.step = step

    def __call__(self):
        self.t += self.step
        return self.t


def cfg_loader(**over):
    cfg = config.normalize(dict(config.DEFAULTS, **over))
    return lambda: dict(cfg)


def quiet_log():
    log = logging.getLogger("aifp.test." + str(next(_n)))
    log.addHandler(logging.NullHandler())
    log.propagate = False
    return log


_n = itertools.count()
