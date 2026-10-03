"""Инструменты набора данных: имена печатей, разбиение по печатям, подделки брака, разметка SKZ."""
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dataset import synth
from dataset.common import duration_s, group_split, model_name, print_id
from dataset.labels import load_labels, save_labels
from dataset.skz import choose, spaghetti_boxes


@pytest.mark.parametrize("fn,sec", [
    ("ECC_0.4_3DBenchy_PET0.2_33m14s-13842401.mp4", 1994),
    ("ECC_0.4_Vase_TPU0.2_8h0m-413652.mp4", 8 * 3600),
    ("ECC_0.4_sosiskas_PET0.2_1m10s-48348411.mp4", 70),
    ("ECC_0.4_Body_15_TPU0.2_8h23m-1044766.mp4", 8 * 3600 + 23 * 60),
    ("random.mp4", None),
])
def test_duration_from_name(fn, sec):
    assert duration_s(fn) == sec


def test_same_model_same_group_any_material_and_run():
    a = model_name("ECC_0.4_papa_noel_Generic PLA _0.2_1h2m-223116.mp4")
    b = model_name("ECC_0.4_papa_noel_Generic PLA _0.2_1h54m-184399.mp4")
    c = model_name("ECC_0.4_Lid Type B (Circular)_Generic PLA _0.2_33m37s-23321159.mp4")
    d = model_name("ECC_0.4_konuuus_PET0.2_24m55s-19388524.mp4")
    e = model_name("ECC_0.4_konuuus_TPU0.2_1h1m-31178594.mp4")
    assert a == b == "papa_noel" and c == "lid type b (circular)" and d == e == "konuuus"


@given(st.text(min_size=1, max_size=40))
def test_law_split_is_a_function_of_group(group):
    s = group_split(group)
    assert s in ("train", "val", "test") and s == group_split(group)


def test_split_shares_roughly_right():
    parts = [group_split(f"g{i}") for i in range(4000)]
    assert 0.17 < parts.count("test") / 4000 < 0.23 and 0.12 < parts.count("val") / 4000 < 0.18


def test_print_id_stable_and_short():
    assert print_id("a.mp4") == print_id("C:/x/a.mp4") and len(print_id("a.mp4")) == 10


def fake_frame(color=(220, 40, 40), obj=True):
    """Похоже на кадр Centauri: тёмный стол, серая голова сверху, деталь у линии сопла."""
    img = np.full((360, 640, 3), 45, np.uint8)
    img[:150, 180:420] = (90, 90, 85)                       # голова
    img[300:] = 55
    rng = np.random.default_rng(0)
    img = np.clip(img + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    if obj:
        img[175:260, 280:360] = color
    return img


def test_object_mask_finds_the_part():
    m, box = synth.object_mask(fake_frame())
    assert box is not None and abs(box[0] - 280) < 6 and abs(box[1] - 175) < 6 and abs(box[2] - 360) < 6


def test_object_mask_empty_bed_none():
    assert synth.object_mask(fake_frame(obj=False)) == (None, None)


@pytest.mark.parametrize("cls", list(synth.GENERATORS))
def test_generators_change_the_frame_and_keep_shape(cls):
    img = fake_frame()
    rng = np.random.default_rng(3)
    out = synth.spaghetti(img, rng) if cls == "spaghetti" else synth.GENERATORS[cls](img, rng)
    assert out is not None and out.shape == img.shape and out.dtype == np.uint8
    assert np.abs(out.astype(int) - img.astype(int)).sum() > 2000
    assert (img == fake_frame()).all(), "исходный кадр не испорчен"


def test_no_extrusion_opens_a_gap_under_the_nozzle_line():
    img = fake_frame()
    out = synth.no_extrusion(img, np.random.default_rng(5))
    _, box = synth.object_mask(out)
    assert box is not None and box[1] > 175 + 10


def test_detached_moves_the_part():
    img = fake_frame()
    out = synth.detached(img, np.random.default_rng(1))
    red = (out[..., 0] > 150) & (out[..., 1] < 90)
    ys, xs = np.nonzero(red)
    assert red.sum() > 1000 and (abs(xs.mean() - 320) > 25 or abs(ys.mean() - 217) > 20)


def test_generators_without_part_return_none_where_needed():
    img = fake_frame(obj=False)
    rng = np.random.default_rng(0)
    assert synth.detached(img, rng) is None and synth.no_extrusion(img, rng) is None
    assert synth.spaghetti(img, rng) is not None and synth.blob(img, rng) is not None


def test_neutral_edit_keeps_the_part():
    img = fake_frame()
    out = synth.neutral_edit(img, np.random.default_rng(2))
    assert (out[175:260, 280:360] == img[175:260, 280:360]).mean() > 0.95


def test_deterministic_with_seed():
    img = fake_frame()
    a = synth.spaghetti(img, np.random.default_rng(9))
    b = synth.spaghetti(img, np.random.default_rng(9))
    assert (a == b).all()


def test_skz_boxes_only_spaghetti_class_and_clipped():
    txt = "3 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.1 0.1\n3 0.99 0.99 0.2 0.2\nbad line\n3 0.5 0.5 0.001 0.001"
    boxes = spaghetti_boxes(txt, 960, 540)
    assert boxes[0] == (384, 216, 576, 324) and all(b[2] <= 960 and b[3] <= 540 for b in boxes) and len(boxes) == 2


def test_skz_choose_round_robin_over_prints():
    files = [f"images/train/P{p}-frame_{i:05d}.png" for p in range(3) for i in range(10)] + ["images/val/X-frame_1.png"]
    got = choose(files, "train", 6)
    assert len(got) == 6 and {g.split("/")[-1].split("-")[0] for g in got} == {"P0", "P1", "P2"}


def test_labels_roundtrip(tmp_path):
    p = str(tmp_path / "labels.csv")
    save_labels({"a.jpg": "blob", "b.jpg": "normal"}, p)
    open(p, "a", encoding="utf-8").write("c.jpg,unknown\n")
    assert load_labels(p) == {"a.jpg": "blob", "b.jpg": "normal"}
    assert load_labels(str(tmp_path / "none.csv")) == {}


def test_no_frames_tracked_in_git():
    """Кадры камеры (комната, личные вещи) в репозиторий не попадают."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import subprocess
    try:
        out = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        pytest.skip("нет git")
    imgs = [f for f in out.splitlines() if f.lower().endswith((".jpg", ".jpeg", ".png", ".mp4"))]
    assert all(f.startswith("docs/img/") for f in imgs), imgs


@given(st.integers(2, 900), st.integers(2, 900), st.integers(0, 2**31 - 1))
def test_law_paste_patch_any_size_never_crashes(pw, ph, seed):
    rng = np.random.default_rng(seed)
    patch = rng.integers(0, 255, (ph, pw, 3)).astype(np.uint8)
    patch[ph // 3: 2 * ph // 3 + 1, pw // 3: 2 * pw // 3 + 1] = (20, 200, 30)
    img = fake_frame()
    out = synth.paste_patch(img, rng, patch, (rng.uniform(0, 640), rng.uniform(0, 360)), (200, 40, 40))
    assert out.shape == img.shape


@given(st.integers(0, 600), st.integers(150, 350), st.integers(1, 640), st.integers(1, 200),
       st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)), st.integers(0, 2**31 - 1),
       st.sampled_from(["spaghetti", "blob", "detached", "no_extrusion", "neutral"]))
def test_law_generators_never_crash(x, y, w, h, color, seed, cls):
    img = fake_frame(obj=False)
    img[y:y + h, x:x + w] = color
    rng = np.random.default_rng(seed)
    gen = synth.neutral_edit if cls == "neutral" else synth.GENERATORS[cls]
    out = gen(img, rng)
    assert out is None or (out.shape == img.shape and out.dtype == np.uint8)


def test_split_same_in_another_process():
    """Разбиение - функция имени модели, а не случая: в другом процессе то же самое."""
    import subprocess
    import sys
    groups = [f"model {i}" for i in range(120)]
    code = "import sys; from dataset.common import group_split; print(','.join(group_split(g) for g in sys.argv[1:]))"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = subprocess.run([sys.executable, "-c", code, *groups], cwd=root, capture_output=True, text=True, timeout=60)
    assert out.stdout.strip().split(",") == [group_split(g) for g in groups]


def test_build_keeps_every_print_in_one_part(_own_data_dir):
    """Сборка набора: все кадры печати и все подделки поверх них - в части этой печати."""
    import csv
    import json

    from PIL import Image

    from dataset import build
    from dataset.common import dataset_dir
    root = dataset_dir()
    prints = {}
    for k, split in enumerate(["train", "val", "test"]):
        pid = f"p{k}"
        d = os.path.join(root, "frames", pid)
        os.makedirs(d)
        for i in range(6):
            Image.fromarray(fake_frame()).save(os.path.join(d, f"{i + 1:05d}.jpg"))
        prints[pid] = {"file": f"{pid}.mp4", "group": pid, "split": split, "frames": 6, "duration_s": 600}
    json.dump(prints, open(os.path.join(root, "prints.json"), "w"))
    build.build(rate=1.0, edit_rate=1.0, workers=1)
    rows = list(csv.DictReader(open(os.path.join(root, "index.csv"), encoding="utf-8")))
    by_print = {}
    for r in rows:
        by_print.setdefault(r["print"], set()).add(r["split"])
    assert {p: s for p, s in by_print.items()} == {"p0": {"train"}, "p1": {"val"}, "p2": {"test"}}
    assert {r["cls"] for r in rows if r["print"] == "p2"} == {"normal", "spaghetti", "blob", "detached", "no_extrusion"}


def test_extra_parts_adds_upright_copies_on_the_bed():
    """Несколько деталей на столе - это норма (сеть путала их со спагетти)."""
    img = fake_frame()
    out = synth.extra_parts(img, np.random.default_rng(4))
    red0 = ((img[..., 0] > 150) & (img[..., 1] < 90)).sum()
    red = (out[..., 0] > 120) & (out[..., 1] < 100)
    assert red.sum() > red0 * 1.3
    ys, _ = np.nonzero(red)
    assert ys.min() >= synth.HEAD_Y - 5
    assert (out[175:260, 280:360] == img[175:260, 280:360]).mean() > 0.9


def test_drag_sequence_moves_part_and_leaves_a_trail():
    """Отрыв 03.10.2026: деталь таскает по столу - она уходит со своего места, за ней след."""
    img = fake_frame()
    seq = synth.DragSequence(img, np.random.default_rng(2))
    outs = [seq.apply(img) for _ in range(5)]
    red_home = [((o[175:260, 280:360, 0] > 150) & (o[175:260, 280:360, 1] < 90)).mean() for o in outs]
    assert max(red_home) < 0.7, red_home                     # на своём месте детали больше нет целиком
    diff = [np.abs(a.astype(int) - b.astype(int)).sum() for a, b in zip(outs, outs[1:])]
    assert min(diff) > 1000                                    # каждый шаг кадр меняется


@settings(max_examples=80)
@given(st.integers(0, 2**31 - 1), st.booleans())
def test_law_drag_never_crashes(seed, with_part):
    img = fake_frame(obj=with_part)
    rng = np.random.default_rng(seed)
    out = synth.dragged(img, rng)
    assert out.shape == img.shape and out.dtype == np.uint8
