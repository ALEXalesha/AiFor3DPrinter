"""Обучение на видеокарте.

    python -m model.train --trial              короткий замер: шагов/с и оценка времени эпохи
    python -m model.train --epochs 12          полное обучение -> runs/<время>/best.pt

Потоки процессора ограничены (по умолчанию 4 на загрузку кадров и 4 на torch): на ПК
работают другие агенты и программы. Чекпойнты - в папке данных (%LOCALAPPDATA%), не в
репозитории.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from model.data import FrameDataset, balanced_weights, read_index
from model.net import AifpNet, count_params, param_groups
from watcher import paths
from watcher.classes import CLASSES


def evaluate(net, loader, device, thr=0.7):
    net.eval()
    conf = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    normal_fp = normal_n = 0
    fail_hit = np.zeros(len(CLASSES))
    fail_n = np.zeros(len(CLASSES))
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for x, y in loader:
            logits, _ = net(x.to(device, non_blocking=True))
            p = torch.softmax(logits.float(), 1).cpu().numpy()
            y = y.numpy()
            pred = p.argmax(1)
            for a, b in zip(y, pred):
                conf[a, b] += 1
            fail = p[:, 1:].max(1) >= thr
            normal_fp += int((fail & (y == 0)).sum())
            normal_n += int((y == 0).sum())
            for c in range(1, len(CLASSES)):
                m = y == c
                fail_n[c] += m.sum()
                fail_hit[c] += (m & (p[:, c] >= thr)).sum()
    recall = {CLASSES[c]: float(fail_hit[c] / fail_n[c]) if fail_n[c] else None for c in range(1, len(CLASSES))}
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    fp = normal_fp / max(normal_n, 1)
    rs = [v for v in recall.values() if v is not None]
    score = (sum(rs) / len(rs) if rs else 0) - 5 * fp
    return {"acc": acc, "normal_fp_rate": fp, "recall_at_thr": recall, "score": score, "confusion": conf.tolist()}


def make_loaders(index, batch, workers, steps_per_epoch=None):
    train_rows = read_index(index, "train")
    val_rows = read_index(index, "val")
    ds_tr, ds_va = FrameDataset(train_rows, True), FrameDataset(val_rows, False)
    w = torch.as_tensor(balanced_weights(train_rows))
    n = steps_per_epoch * batch if steps_per_epoch else len(train_rows)
    sampler = torch.utils.data.WeightedRandomSampler(w, n, replacement=True)
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch, sampler=sampler, drop_last=True, **kw)
    va = torch.utils.data.DataLoader(ds_va, batch_size=batch * 2, shuffle=False, **kw)
    return tr, va, len(train_rows), len(val_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--trial", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="шагов на эпоху (0 = размер train)")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = os.path.join(paths.data_dir(), "dataset", "index.csv")
    tr, va, n_tr, n_va = make_loaders(index, a.batch, a.workers, a.steps or None)
    net = AifpNet().to(device).to(memory_format=torch.channels_last)
    print(f"устройство {device}, параметров {count_params(net):,}, train {n_tr}, val {n_va}", flush=True)

    opt = torch.optim.AdamW(param_groups(net, a.lr * 0.3, a.lr), weight_decay=1e-4)
    steps_per_epoch = len(tr)
    total = steps_per_epoch * (1 if a.trial else a.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr * 0.3, a.lr], total_steps=total, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    if a.trial:
        net.train()
        t0, seen = None, 0
        for i, (x, y) in enumerate(tr):
            if i == 10:
                torch.cuda.synchronize()
                t0 = time.time()
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = net(x.to(device, non_blocking=True).to(memory_format=torch.channels_last))
                loss = loss_fn(logits.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if i >= 10:
                seen += x.shape[0]
            if i >= 160:
                break
        torch.cuda.synchronize()
        dt = time.time() - t0
        ips = seen / dt
        ve0 = time.time()
        m = evaluate(net, va, device)
        vt = time.time() - ve0
        epoch_s = n_tr / ips + vt
        est = {"images_per_s": round(ips, 1), "train_images": n_tr, "val_s": round(vt, 1),
               "epoch_min": round(epoch_s / 60, 1), "for_epochs": {e: round(e * epoch_s / 60) for e in (8, 12, 16)},
               "loss": loss.item(), "val_acc_after_trial": round(m["acc"], 3)}
        print("ОЦЕНКА:", json.dumps(est, ensure_ascii=False), flush=True)
        json.dump(est, open(paths.data_path("runs", "trial.json"), "w"), indent=1)
        return

    run = paths.data_path("runs", time.strftime("%Y%m%d-%H%M%S"), "x").rsplit(os.sep, 1)[0]
    best, history = -math.inf, []
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, tl, nb = time.time(), 0.0, 0
        for x, y in tr:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, _ = net(x.to(device, non_blocking=True).to(memory_format=torch.channels_last))
                loss = loss_fn(logits.float(), y.to(device, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item()
            nb += 1
        m = evaluate(net, va, device)
        rec = {"epoch": ep, "loss": round(tl / max(nb, 1), 4), "min": round((time.time() - t0) / 60, 1),
               **{k: m[k] for k in ("acc", "normal_fp_rate", "recall_at_thr", "score")}}
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        ck = {"model": net.state_dict(), "epoch": ep, "val": m, "args": vars(a), "classes": list(CLASSES)}
        torch.save(ck, os.path.join(run, "last.pt"))
        if m["score"] > best:
            best = m["score"]
            torch.save(ck, os.path.join(run, "best.pt"))
        json.dump(history, open(os.path.join(run, "history.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("лучший чекпойнт:", os.path.join(run, "best.pt"), "score", round(best, 4), flush=True)


if __name__ == "__main__":
    main()
