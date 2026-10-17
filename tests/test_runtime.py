"""Сеть в службе (onnxruntime) и тепловая карта."""
import os

import numpy as np
import pytest
from fake_camera import make_jpeg
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from watcher import paths
from watcher.classes import CLASSES
from watcher.heatmap import colorize, hot_box, normalize_cam, overlay, zoom
from watcher.model_runtime import preprocess, softmax, to_rgb

MODEL = paths.resource("model", "aifp.onnx")


def test_preprocess_shape_and_norm():
    x = preprocess(Image.new("RGB", (640, 360), (124, 116, 104)))
    assert x.shape == (1, 3, 160, 288) and x.dtype == np.float32 and abs(float(x.mean())) < 0.05


def test_to_rgb_accepts_jpeg_pil_array():
    for f in (make_jpeg(), Image.new("L", (10, 10)), np.zeros((5, 6, 3), np.uint8)):
        assert to_rgb(f).mode == "RGB"


@given(st.lists(st.floats(-50, 50), min_size=5, max_size=5))
def test_law_softmax_is_a_distribution(z):
    p = softmax(z)
    assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


@given(st.integers(1, 12), st.integers(1, 12), st.integers(0, 2**31 - 1))
def test_law_heat_normalized_and_box_inside(h, w, seed):
    cam = np.random.default_rng(seed).normal(0, 1, (h, w))
    n = normalize_cam(cam)
    assert n.min() >= 0 and n.max() <= 1 + 1e-6
    box = hot_box(n)
    if box:
        x0, y0, x1, y1 = box
        assert 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1


def test_colorize_range():
    c = colorize(np.linspace(0, 1, 11))
    assert c.shape == (11, 3) and c.dtype == np.uint8 and tuple(c[0]) == (0, 0, 80) and tuple(c[-1]) == (255, 40, 0)


def test_overlay_marks_hot_place():
    img = Image.new("RGB", (640, 360), (40, 40, 40))
    cam = np.zeros((5, 5, 9), np.float32)
    cam[1, 1, 7] = 5.0                    # спагетти справа вверху
    out = np.asarray(overlay(img, cam, "spaghetti")).astype(int)
    assert out[80:136, 505:555, 0].mean() > out[250:, :150, 0].mean() + 30
    assert overlay(img, None, "spaghetti").size == img.size


def test_zoom_is_a_close_up_of_the_hot_place():
    # 17.10 Алексей: «пусть скидывает 2 фото: область ошибки и просто фото, а то так не видно»
    img = Image.new("RGB", (640, 360), (40, 40, 40))
    img.paste((250, 250, 250), (520, 90, 540, 110))          # белая метка в месте брака
    cam = np.zeros((5, 5, 9), np.float32)
    cam[1, 1, 7] = 5.0                                        # спагетти справа вверху
    z = zoom(img, cam, "spaghetti")
    a = np.asarray(z).astype(int)
    assert z.size == (960, 540)
    white = a[..., 0] > 200
    assert white.sum() > (20 * 20) * 4, "метка увеличена"     # кадр 640 -> вырезка не шире 0,45*640 -> 960
    red = (a[..., 0] > 200) & (a[..., 1] < 80)
    assert red.any(), "место отмечено рамкой"
    spread = a.max(axis=2) - a.min(axis=2)
    assert spread[~red].max() < 60, "без цветной тепловой карты: вне рамки кадр серый, как был"
    full = zoom(img, None, "spaghetti")
    assert full.size == (960, 540) and np.asarray(full).astype(int)[..., 0].max() > 200


@given(st.integers(1, 9), st.integers(1, 9), st.integers(0, 2**31 - 1))
def test_law_zoom_keeps_16x9_and_the_hot_place_inside(h, w, seed):
    cam = np.zeros((5, h, w), np.float32)
    cam[3] = np.random.default_rng(seed).normal(0, 1, (h, w))
    img = Image.new("RGB", (640, 360), (40, 40, 40))
    z = np.asarray(zoom(img, cam, CLASSES[3])).astype(int)
    assert z.shape == (540, 960, 3)
    red = (z[..., 0] > 200) & (z[..., 1] < 80)
    if hot_box(normalize_cam(cam[3])):
        assert red.any(), "рамка места видна в вырезке"


@pytest.mark.skipif(not os.path.exists(MODEL), reason="сеть ещё не обучена")
def test_shipped_model_loads_and_predicts():
    from watcher.model_runtime import Model
    m = Model(MODEL)
    assert m.classes == CLASSES
    pred = m.predict(np.full((360, 640, 3), 50, np.uint8))
    assert abs(sum(pred.probs.values()) - 1) < 1e-4 and pred.cam.shape[0] == 5
    assert pred.cam.shape[1:] == (5, 9)
