"""Synthetic before/after demo for the building-footprint pipeline.

Simulates a dense housing scene, a blurry binary U-Net output ("before"),
and the proposed 3-class output followed by marker watershed + adaptive
polygon regularization ("after"). The post-processing is the real
OpenCV code planned for the project; only the network outputs are simulated.
"""
import json
import sys

import cv2
import numpy as np

OUT = sys.argv[1]
H = W = 400
rng = np.random.default_rng(7)


def rotated_rect(cx, cy, w, h, ang):
    return cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)


# ---------- 1. synthetic scene + instance ground truth ----------
img = np.zeros((H, W, 3), np.uint8)
img[:] = (88, 120, 104)  # ground (BGR)
img += rng.integers(0, 18, (H, W, 3), dtype=np.uint8)
cv2.rectangle(img, (0, 182), (W, 214), (150, 150, 146), -1)  # road
cv2.rectangle(img, (192, 0), (222, 182), (150, 150, 146), -1)

gt = np.zeros((H, W), np.int32)
houses = []
# two rows of touching row-houses (dense settlement)
for i in range(6):
    houses.append((22 + i * 28, 30, 28, 46, 0))
    houses.append((22 + i * 28, 82, 28, 44, 0))
for i in range(5):
    houses.append((250 + i * 27, 40, 27, 50, 0))
# rotated cluster below the road
for i in range(4):
    houses.append((70 + i * 31, 270, 31, 52, 12))
for i in range(3):
    houses.append((280 + i * 34, 300, 34, 60, -8))
houses.append((90, 360, 70, 34, 12))

roof_cols = [(60, 70, 170), (70, 90, 190), (110, 110, 120), (150, 90, 70), (80, 100, 160)]
for k, (x, y, w, h, a) in enumerate(houses, start=1):
    if a == 0:
        pts = np.array([[x, y - h // 2], [x + w, y - h // 2], [x + w, y + h // 2], [x, y + h // 2]], np.int32)
    else:
        pts = rotated_rect(x, y, w, h, a)
    cv2.fillPoly(gt, [pts], k)
    col = roof_cols[k % len(roof_cols)]
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, tuple(int(c * 0.7) for c in col), 1)
# trees overlapping some roofs
for (tx, ty, r) in [(180, 120, 14), (360, 100, 12), (40, 320, 16), (230, 360, 15)]:
    cv2.circle(img, (tx, ty), r, (40, 95, 45), -1)
img = cv2.GaussianBlur(img, (3, 3), 0)

n_gt = len(houses)
binary_gt = (gt > 0).astype(np.float32)


# contact label: pixels inside the dilation of >= 2 different houses
def contact_label(inst, r=3):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    count = np.zeros(inst.shape, np.int32)
    for i in range(1, inst.max() + 1):
        count += cv2.dilate((inst == i).astype(np.uint8), k) > 0
    return (count >= 2).astype(np.float32)


contact_gt = contact_label(gt)


# ---------- 2. simulated network outputs ----------
def smooth_noise(scale):
    n = rng.normal(0, 1, (H, W)).astype(np.float32)
    return cv2.GaussianBlur(n, (0, 0), scale)


# "before": binary U-Net -> blurry, jagged, bleeds across shared walls
p_before = cv2.GaussianBlur(binary_gt, (0, 0), 3.2) + 0.9 * smooth_noise(1.2)
mask_before = (p_before > 0.45).astype(np.uint8)
# coarse decoder output: stair-step edges after upsampling
small = cv2.resize(mask_before, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
mask_before = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)

# "after": 3-class U-Net++ with boundary-aware loss -> sharper, plus contact map
p_build = cv2.GaussianBlur(binary_gt, (0, 0), 1.6) + 0.5 * smooth_noise(2.0)
p_contact = np.clip(cv2.GaussianBlur(contact_gt, (0, 0), 1.2) * 1.6, 0, 1)


# ---------- 3. post-processing (the real planned code) ----------
def naive_polygons(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c.reshape(-1, 2) for c in cnts if cv2.contourArea(c) > 60]


def split_instances(p_b, p_c, tau=0.5):
    building = (p_b > 0.5).astype(np.uint8)
    marker_mask = ((p_b - p_c) > tau).astype(np.uint8)
    marker_mask = cv2.erode(marker_mask, np.ones((3, 3), np.uint8))
    n, markers = cv2.connectedComponents(marker_mask)
    markers = markers + 1
    markers[(building == 1) & (marker_mask == 0)] = 0  # unknown band -> watershed decides
    ws_input = cv2.cvtColor((np.clip(p_b, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(ws_input, markers.astype(np.int32))
    markers[building == 0] = 1
    return markers  # 1 = background, -1 = ridge, >=2 = instances


def regularize(inst_mask, k=0.04, rect_thr=0.75, snap_deg=15):
    inst_mask = cv2.morphologyEx(inst_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    inst_mask = cv2.morphologyEx(inst_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 60:
        return None
    eps = k * np.sqrt(area)  # adaptive tolerance: scales with building size
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if area / max(rw * rh, 1) > rect_thr:  # rectangular building -> snap to its oriented box
        return cv2.boxPoints(((cx, cy), (rw, rh), ang)).astype(np.int32)
    # otherwise snap each edge direction to the dominant orientation (0/90 deg)
    theta = np.deg2rad(ang)
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    q = (poly - [cx, cy]) @ R.T
    for i in range(len(q)):
        j = (i + 1) % len(q)
        d = q[j] - q[i]
        a = np.degrees(np.arctan2(d[1], d[0])) % 90
        if min(a, 90 - a) < snap_deg:
            if abs(d[0]) > abs(d[1]):
                q[j, 1] = q[i, 1]
            else:
                q[j, 0] = q[i, 0]
    return (q @ R + [cx, cy]).astype(np.int32)


before_polys = naive_polygons(mask_before)
markers = split_instances(p_build, p_contact)
inst_ids = [i for i in np.unique(markers) if i >= 2]
after_polys = [p for p in (regularize((markers == i).astype(np.uint8)) for i in inst_ids) if p is not None]

# ---------- 4. render panels ----------
S = 3  # upscale for crisp lines
big = lambda a: cv2.resize(a, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"{OUT}/p1_citra.png", cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_CUBIC))

p2 = big(img).copy()
overlay = p2.copy()
overlay[big(mask_before) == 1] = (60, 60, 230)
p2 = cv2.addWeighted(overlay, 0.45, p2, 0.55, 0)
for p in before_polys:
    cv2.polylines(p2, [p * S], True, (40, 40, 255), 3)
cv2.imwrite(f"{OUT}/p2_sebelum.png", p2)

palette = rng.integers(60, 255, (markers.max() + 2, 3)).astype(np.uint8)
p3 = big(img).copy()
col = np.zeros_like(p3)
mk = big(markers.astype(np.int32))
for i in inst_ids:
    col[mk == i] = palette[i]
sel = mk >= 2
p3[sel] = (0.35 * p3[sel] + 0.65 * col[sel]).astype(np.uint8)
p3[big((markers == -1).astype(np.uint8)) == 1] = (255, 255, 255)
cv2.imwrite(f"{OUT}/p3_instance.png", p3)

p4 = big(img).copy()
for p in after_polys:
    cv2.polylines(p4, [p * S], True, (40, 40, 255), 3)
    for v in p:
        cv2.circle(p4, tuple(int(t) for t in v * S), 5, (255, 255, 255), -1)
cv2.imwrite(f"{OUT}/p4_sesudah.png", p4)

stats = {
    "gt_buildings": n_gt,
    "before_polygons": len(before_polys),
    "before_mean_vertices": round(float(np.mean([len(cv2.approxPolyDP(p.reshape(-1, 1, 2), 1.0, True)) for p in before_polys])), 1),
    "after_polygons": len(after_polys),
    "after_mean_vertices": round(float(np.mean([len(p) for p in after_polys])), 1),
}
json.dump(stats, open(f"{OUT}/stats.json", "w"), indent=2)
print(stats)
