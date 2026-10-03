"""Искусственный брак поверх нормальных кадров Centauri Carbon.

Настоящих кадров брака с этой камеры почти нет, а нормальных - тысячи (таймлапсы).
Поэтому брак рисуется поверх нормальных кадров тех же печатей:

- spaghetti     - путаные нити цвета пластика (случайные блуждания) и куски настоящих
                  спагетти из набора SKZ (CC BY-SA 4.0), перекрашенные в цвет пластика;
- blob          - неровный ком пластика с бликом там, где сопло касается детали;
- detached      - деталь вырезана со своего места (заливка cv2.inpaint) и положена
                  рядом на бок, иногда с нитями;
- no_extrusion  - стол «уехал» вниз, а деталь не выросла: щель между плоскостью сопла
                  и верхом детали (у Centauri стол опускается на слой, голова - на месте),
                  плюс редкие висящие нити.

У камеры Centauri голова печатает в одной горизонтальной плоскости, поэтому верх
растущей детали на кадре всегда у «линии сопла» (~y=165 из 360). Это и есть то, что
сеть должна заметить при «печати в воздух».

Чтобы сеть не училась на следах самой подделки (размытые пятна заливки), такие же
«нейтральные правки» иногда делаются и на нормальных кадрах (neutral_edit).
Цифры по искусственному браку оптимистичны - главная проверка остаётся на настоящих
печатях (шаг 8 плана) и на кадрах из кнопок Telegram.
"""
import math

import cv2
import numpy as np

W, H = 640, 360
HEAD_Y = 165          # ниже этой строки - стол и деталь (голова и задняя стенка выше)
TOP_MAX = 215         # верх растущей детали - у линии сопла, не ниже этой строки
PALETTE = [(230, 40, 40), (250, 110, 20), (240, 240, 235), (40, 40, 45), (60, 110, 230), (150, 70, 200),
           (40, 170, 70), (250, 210, 40), (230, 70, 150), (150, 150, 155), (120, 70, 40)]


def as_rgb(img):
    img = np.asarray(img)
    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img[..., :3]).astype(np.uint8)


def bed_color(img):
    strip = img[H - 40:H - 5, 120:520].reshape(-1, 3)
    return np.median(strip, axis=0)


def object_mask(img, min_area=250, max_area=90000):
    """Маска печатаемой детали. -> (mask, bbox) или (None, None).

    Деталь - то, что ярче своего окружения (белый, серый пластик) или заметно цветное,
    ниже головы, и её верх у линии сопла (растущая деталь касается сопла). Блеск стола
    широкий и уходит в фон медианного фильтра, искры стола - мелкие и срезаются открытием."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[..., 2]
    bg = cv2.medianBlur(v, 61).astype(np.int16)
    bright = (v.astype(np.int16) - bg) > 45
    colorful = (hsv[..., 1] > 110) & (v > 70)
    m = (bright | colorful).astype(np.uint8)
    m[:HEAD_Y] = 0
    m[:, :45] = 0                                 # надпись ELEGOO на краю стола
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best, best_score = None, 0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a < min_area or a > max_area or y > TOP_MAX:
            continue
        score = a - 3 * abs((x + w / 2) - W / 2)
        if score > best_score:
            best, best_score = i, score
    if best is None:
        return None, None
    x, y, w, h, _ = stats[best]
    return (lbl == best).astype(np.uint8), (int(x), int(y), int(x + w), int(y + h))


def filament_color(img, mask, rng):
    if mask is not None and mask.sum() > 50:
        c = np.median(img[mask > 0], axis=0)
        return tuple(int(v) for v in c)
    return PALETTE[rng.integers(len(PALETTE))]


def _jitter(color, rng, k=25):
    return tuple(int(np.clip(c + rng.integers(-k, k + 1), 0, 255)) for c in color)


def _strand(rng, x, y, n, step, curl):
    pts, a = [], rng.uniform(0, 2 * math.pi)
    da = 0.0
    for _ in range(n):
        da = 0.85 * da + rng.normal(0, curl)
        a += da
        x += step * math.cos(a)
        y += step * math.sin(a) * 0.55            # стол виден под углом - петли сплющены
        pts.append((x, y))
    return np.array(pts, np.int32)


def draw_strands(img, rng, center, spread, color, count, thick=(1, 4), length=(40, 220)):
    out = img.copy()
    cx, cy = center
    for _ in range(count):
        x = cx + rng.normal(0, spread)
        y = cy + rng.normal(0, spread * 0.35)
        pts = _strand(rng, x, y, int(rng.integers(*length)), rng.uniform(1.5, 4.0), rng.uniform(0.15, 0.6))
        pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1], HEAD_Y - 20, H - 1)
        t = int(rng.integers(*thick))
        col = _jitter(color, rng)
        dark = tuple(int(c * 0.7) for c in col)
        cv2.polylines(out, [pts + [1, 2]], False, dark, t + 1, cv2.LINE_AA)
        cv2.polylines(out, [pts], False, col, t, cv2.LINE_AA)
        if t >= 2:
            hi = tuple(int(min(255, c * 0.6 + 110)) for c in col)
            cv2.polylines(out, [pts - [0, 1]], False, hi, 1, cv2.LINE_AA)
    return out


def recolor_patch(patch, mask, color):
    """Кусок спагетти SKZ -> цвет нашего пластика (тон и насыщенность, яркость своя)."""
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_RGB2HSV)[0, 0].astype(np.float32)
    hsv[..., 0] = target[0]
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.3 + target[1] * 0.7, 0, 255)
    v_mean = hsv[..., 2][mask > 0].mean() if mask.sum() else 128
    hsv[..., 2] = np.clip(hsv[..., 2] * (target[2] / max(v_mean, 1)) * 0.9 + 10, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def patch_mask(patch):
    """Спагетти на светлом столе SKZ: всё, что заметно отличается от краёв куска."""
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]]).astype(np.float32)
    ref = np.median(border, axis=0)
    d = np.sqrt(((patch.astype(np.float32) - ref) ** 2).sum(-1))
    m = (d > 45).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def paste_patch(img, rng, patch, center, color):
    out = img.copy()
    target_w = int(rng.uniform(70, 260))
    s = target_w / patch.shape[1]
    ph = max(8, int(patch.shape[0] * s * rng.uniform(0.45, 0.8)))   # вид сбоку: сплющить по высоте
    ph = min(ph, H - (HEAD_Y - 10))                                  # высокий кусок не шире области стола
    p = cv2.resize(patch, (target_w, ph), interpolation=cv2.INTER_AREA)
    m = patch_mask(p)
    if m.sum() < 40:
        return out
    p = recolor_patch(p, m, color)
    x0 = int(np.clip(center[0] - target_w / 2 + rng.normal(0, 40), 0, W - target_w))
    y0 = int(np.clip(center[1] - ph / 2 + rng.normal(0, 15), HEAD_Y - 10, H - ph))
    a = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]
    roi = out[y0:y0 + ph, x0:x0 + target_w].astype(np.float32)
    out[y0:y0 + ph, x0:x0 + target_w] = (roi * (1 - a) + p.astype(np.float32) * a).astype(np.uint8)
    return out


def spaghetti(img, rng, patches=()):
    mask, box = object_mask(img)
    color = filament_color(img, mask, rng)
    if box:
        center = ((box[0] + box[2]) / 2, max(box[3] - 10, HEAD_Y + 30))
    else:
        center = (rng.uniform(200, 440), rng.uniform(HEAD_Y + 40, H - 60))
    out = img
    mode = rng.random()
    if mode < 0.7 or not patches:
        out = draw_strands(out, rng, center, rng.uniform(25, 90), color, int(rng.integers(4, 14)))
    if (mode >= 0.5) and patches:
        for _ in range(int(rng.integers(1, 3))):
            out = paste_patch(out, rng, patches[rng.integers(len(patches))], center, color)
    return out


def blob(img, rng):
    mask, box = object_mask(img)
    color = filament_color(img, mask, rng)
    if box:
        cx = (box[0] + box[2]) / 2 + rng.normal(0, 6)
        cy = box[1] + rng.uniform(-6, 4)
    else:
        cx, cy = rng.uniform(220, 420), rng.uniform(HEAD_Y + 5, HEAD_Y + 25)
    r = rng.uniform(10, 34)
    k = 28
    ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
    noise = np.convolve(rng.normal(0, 0.18, k * 3), np.ones(3) / 3, "same")[k:2 * k]
    rr = r * (1 + noise)
    pts = np.stack([cx + rr * np.cos(ang) * 1.15, cy + rr * np.sin(ang) * 0.9], 1).astype(np.int32)
    out = img.copy()
    layer = np.zeros((H, W), np.uint8)
    cv2.fillPoly(layer, [pts], 255)
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1.0 - 0.45 * np.clip(((xx - cx + r * 0.4) ** 2 + (yy - cy + r * 0.5) ** 2) / (2.2 * r * r), 0, 1)
    col = np.array(_jitter(color, rng, 15), np.float32)
    blob_rgb = np.clip(col[None, None] * shade[..., None] * rng.uniform(0.75, 1.05), 0, 255)
    a = cv2.GaussianBlur(layer.astype(np.float32) / 255, (5, 5), 0)[..., None]
    out = (out.astype(np.float32) * (1 - a) + blob_rgb * a).astype(np.uint8)
    hx, hy = int(cx - r * 0.35), int(cy - r * 0.35)
    cv2.ellipse(out, (hx, hy), (max(2, int(r * 0.25)), max(1, int(r * 0.12))), -20, 0, 360,
                tuple(int(min(255, c * 0.5 + 140)) for c in col), -1, cv2.LINE_AA)
    if rng.random() < 0.4:
        out = draw_strands(out, rng, (cx, cy + r), 10, color, int(rng.integers(1, 4)), (1, 3), (15, 60))
    return out


def detached(img, rng):
    mask, box = object_mask(img, min_area=500)
    if box is None:
        return None
    color = filament_color(img, mask, rng)
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1].copy()
    cmask = mask[y0:y1, x0:x1].copy()
    hole = cv2.dilate(mask, np.ones((9, 9), np.uint8))
    out = cv2.inpaint(img, hole, 5, cv2.INPAINT_TELEA)
    if rng.random() < 0.35:                      # остаток нижних слоёв на столе
        keep = int((y1 - y0) * rng.uniform(0.05, 0.2))
        sub = mask[y1 - keep:y1, x0:x1] > 0
        out[y1 - keep:y1, x0:x1][sub] = img[y1 - keep:y1, x0:x1][sub]
    angle = rng.choice([-1, 1]) * rng.uniform(55, 125)
    s = rng.uniform(0.75, 1.0)
    h, w = crop.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, s)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    rc = cv2.warpAffine(crop, M, (nw, nh), flags=cv2.INTER_LINEAR)
    rm = cv2.warpAffine(cmask * 255, M, (nw, nh), flags=cv2.INTER_LINEAR)
    nh_squash = max(4, int(nh * rng.uniform(0.7, 1.0)))
    rc = cv2.resize(rc, (nw, nh_squash))
    rm = cv2.resize(rm, (nw, nh_squash))
    dx = rng.choice([-1, 1]) * rng.uniform(40, 170)
    px = int(np.clip((x0 + x1) / 2 + dx - nw / 2, 0, W - nw)) if nw < W else 0
    py = int(np.clip(y1 + rng.uniform(-10, 50) - nh_squash, HEAD_Y, H - nh_squash)) if nh_squash < H - HEAD_Y else HEAD_Y
    nw, nh_squash = min(nw, W - px), min(nh_squash, H - py)
    a = (cv2.GaussianBlur(rm[:nh_squash, :nw], (3, 3), 0).astype(np.float32) / 255)[..., None]
    roi = out[py:py + nh_squash, px:px + nw].astype(np.float32)
    shade = rng.uniform(0.7, 1.0)
    out[py:py + nh_squash, px:px + nw] = (roi * (1 - a) + rc[:nh_squash, :nw].astype(np.float32) * shade * a).astype(np.uint8)
    if rng.random() < 0.5:
        out = draw_strands(out, rng, ((x0 + x1) / 2, y0 + 10), 30, color, int(rng.integers(2, 7)))
    return out


def no_extrusion(img, rng):
    mask, box = object_mask(img, min_area=300)
    if box is None:
        return None
    color = filament_color(img, mask, rng)
    top = box[1]
    d = int(rng.uniform(14, 48))
    y0 = max(HEAD_Y, top - 2)
    out = img.copy()
    out[y0 + d:] = img[y0:H - d]
    band = np.zeros((H, W), np.uint8)
    band[y0:y0 + d + 1, max(0, box[0] - 40):min(W, box[2] + 40)] = 1
    out = cv2.inpaint(out, band, 4, cv2.INPAINT_TELEA)
    if rng.random() < 0.6:
        nx = (box[0] + box[2]) / 2 + rng.normal(0, 8)
        for _ in range(int(rng.integers(1, 4))):
            n = int(rng.integers(6, 20))
            pts = np.array([(nx + rng.normal(0, 3) * i ** 0.5, y0 + i * d / n) for i in range(n)], np.int32)
            cv2.polylines(out, [pts], False, _jitter(color, rng), 1, cv2.LINE_AA)
    return out


def neutral_edit(img, rng):
    """Та же заливка и вставка, что у подделок, но без брака: стол остаётся столом."""
    out = img.copy()
    mask, box = object_mask(img)
    for _ in range(int(rng.integers(1, 3))):
        r = int(rng.uniform(8, 35))
        for _try in range(10):
            cx, cy = int(rng.uniform(60, W - 60)), int(rng.uniform(HEAD_Y + 60, H - 15))
            if box is None or not (box[0] - r < cx < box[2] + r and box[1] - r < cy < box[3] + r):
                break
        m = np.zeros((H, W), np.uint8)
        cv2.ellipse(m, (cx, cy), (r, max(3, r // 2)), 0, 0, 360, 1, -1)
        out = cv2.inpaint(out, m, 5, cv2.INPAINT_TELEA)
    return out


GENERATORS = {"spaghetti": spaghetti, "blob": blob, "detached": detached, "no_extrusion": no_extrusion}
