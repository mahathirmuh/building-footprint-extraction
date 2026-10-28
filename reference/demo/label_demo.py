"""Label types for building extraction, rendered on the same synthetic scene.

Panels: binary mask, instance mask, orientation label derived from the
instance mask (no polygons needed), and a COCO-style polygon annotation
(what HiSup / PolyR-CNN / Pix2Poly require).
"""
import sys

import cv2
import numpy as np

OUT = sys.argv[1]
H = W = 400
S = 3
rng = np.random.default_rng(7)

img = np.zeros((H, W, 3), np.uint8)
img[:] = (88, 120, 104)
img += rng.integers(0, 18, (H, W, 3), dtype=np.uint8)
cv2.rectangle(img, (0, 182), (W, 214), (150, 150, 146), -1)
cv2.rectangle(img, (192, 0), (222, 182), (150, 150, 146), -1)

houses = []
for i in range(6):
    houses.append((22 + i * 28, 30, 28, 46, 0))
    houses.append((22 + i * 28, 82, 28, 44, 0))
for i in range(5):
    houses.append((250 + i * 27, 40, 27, 50, 0))
for i in range(4):
    houses.append((70 + i * 31, 270, 31, 52, 12))
for i in range(3):
    houses.append((280 + i * 34, 300, 34, 60, -8))
houses.append((90, 360, 70, 34, 12))

gt = np.zeros((H, W), np.int32)
polys = []
roof_cols = [(60, 70, 170), (70, 90, 190), (110, 110, 120), (150, 90, 70), (80, 100, 160)]
for k, (x, y, w, h, a) in enumerate(houses, start=1):
    if a == 0:
        pts = np.array([[x, y - h // 2], [x + w, y - h // 2], [x + w, y + h // 2], [x, y + h // 2]], np.int32)
    else:
        pts = cv2.boxPoints(((x, y), (w, h), a)).astype(np.int32)
    polys.append(pts)
    cv2.fillPoly(gt, [pts], k)
    col = roof_cols[k % len(roof_cols)]
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, tuple(int(c * 0.7) for c in col), 1)
img = cv2.GaussianBlur(img, (3, 3), 0)

big = lambda a: cv2.resize(a, (W * S, H * S), interpolation=cv2.INTER_NEAREST)

# 1. binary mask: 1 = building, 0 = background (touching houses merge)
binary = ((gt > 0) * 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/l1_biner.png", cv2.cvtColor(big(binary), cv2.COLOR_GRAY2BGR))

# 2. instance mask: one id per building (still a raster)
pal = rng.integers(70, 255, (gt.max() + 1, 3)).astype(np.uint8)
pal[0] = (30, 27, 20)
cv2.imwrite(f"{OUT}/l2_instance.png", big(pal[gt]))

# 3. orientation label derived automatically from the instance mask
theta_img = np.zeros((H, W, 3), np.uint8)
theta_img[:] = (30, 27, 20)
for i in range(1, gt.max() + 1):
    m = (gt == i).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (_, _), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    hue = int(((ang % 90) / 90.0) * 179)  # 90-degree periodic -> sin(4θ), cos(4θ) in training
    color = cv2.cvtColor(np.uint8([[[hue, 200, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
    theta_img[m == 1] = color
th = big(theta_img)
for i in range(1, gt.max() + 1):  # small direction ticks
    ys, xs = np.nonzero(gt == i)
    cnts, _ = cv2.findContours((gt == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(max(cnts, key=cv2.contourArea))
    t = np.deg2rad(ang)
    L = 18
    p1 = (int((cx - L * np.cos(t)) * S), int((cy - L * np.sin(t)) * S))
    p2 = (int((cx + L * np.cos(t)) * S), int((cy + L * np.sin(t)) * S))
    cv2.line(th, p1, p2, (255, 255, 255), 3)
cv2.imwrite(f"{OUT}/l3_orientasi.png", th)

# 4. polygon annotation: ordered vertices per building (COCO)
pl = cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_CUBIC)
pl = (pl * 0.55).astype(np.uint8)
for j, p in enumerate(polys):
    cv2.polylines(pl, [p * S], True, (80, 220, 255), 3)
    for n, v in enumerate(p):
        cv2.circle(pl, tuple(int(c) for c in v * S), 7, (80, 220, 255), -1)
        if j in (0, 13, 21):
            cv2.putText(pl, str(n + 1), (int(v[0] * S) + 8, int(v[1] * S) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(f"{OUT}/l4_poligon.png", pl)
print("ok", len(polys), "buildings")
