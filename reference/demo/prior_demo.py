"""Classical morphological priors on the synthetic dense-housing scene.

MBI-style brightness top-hat (Huang & Zhang 2011, simplified with linear
structuring elements) and a 'gap map': multi-directional black top-hat that
highlights narrow dark lines between adjoining roofs. Also a classical
baseline B0: MBI -> Otsu -> morphology -> gap-aware watershed.
"""
import sys

import cv2
import numpy as np

OUT = sys.argv[1]
H = W = 400
S = 3
rng = np.random.default_rng(7)

# same scene as label_demo.py / before_after.py, plus a 1-2 px dark gap between row houses
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
roof_cols = [(60, 70, 170), (70, 90, 190), (110, 110, 120), (150, 90, 70), (80, 100, 160)]
gt = np.zeros((H, W), np.int32)
for k, (x, y, w, h, a) in enumerate(houses, start=1):
    if a == 0:
        pts = np.array([[x, y - h // 2], [x + w, y - h // 2], [x + w, y + h // 2], [x, y + h // 2]], np.int32)
    else:
        pts = cv2.boxPoints(((x, y), (w, h), a)).astype(np.int32)
    cv2.fillPoly(gt, [pts], k)
    col = roof_cols[k % len(roof_cols)]
    cv2.fillPoly(img, [pts], col)
    cv2.polylines(img, [pts], True, tuple(int(c * 0.55) for c in col), 1)  # shared wall / shadow line
img = cv2.GaussianBlur(img, (3, 3), 0)


def line_se(length, angle):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    t = np.deg2rad(angle)
    dx, dy = np.cos(t) * c, np.sin(t) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


ANGLES = [0, 45, 90, 135]
bright = cv2.medianBlur(img.max(axis=2), 5).astype(np.float32)

# MBI (simplified): mean over scales of the increase in min-over-directions white top-hat
scales = [31, 51, 71, 91]
th_prev = np.zeros_like(bright)
mbi = np.zeros_like(bright)
for s in scales:
    th = np.min([cv2.morphologyEx(bright, cv2.MORPH_TOPHAT, line_se(s, a)) for a in ANGLES], axis=0)
    mbi += np.abs(th - th_prev)
    th_prev = th
mbi /= len(scales)

# gap map: max over directions of black top-hat with short lines -> narrow dark gaps between roofs
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
gap = np.max([cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line_se(7, a)) for a in ANGLES], axis=0)


def norm(a):
    a = a - a.min()
    return (255 * a / max(a.max(), 1e-6)).astype(np.uint8)


mbi_u8, gap_u8 = norm(mbi), norm(gap)

# classical baseline B0: Otsu on MBI, morphology, watershed with gap map as barrier
_, bmask = cv2.threshold(cv2.GaussianBlur(mbi_u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bmask = cv2.morphologyEx(bmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
bmask = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
_, gapbin = cv2.threshold(gap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
core = cv2.erode(((bmask > 0) & (gapbin == 0)).astype(np.uint8), np.ones((3, 3), np.uint8))
n, markers = cv2.connectedComponents(core)
markers = markers + 1
markers[(bmask > 0) & (core == 0)] = 0
markers = cv2.watershed(cv2.cvtColor(gap_u8, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
ids = [i for i in np.unique(markers) if i >= 2 and (markers == i).sum() > 80]

big = lambda a: cv2.resize(a, (W * S, H * S), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f'{OUT}/m1_citra.png', cv2.resize(img, (W * S, H * S), interpolation=cv2.INTER_CUBIC))
cv2.imwrite(f'{OUT}/m2_mbi.png', big(cv2.applyColorMap(mbi_u8, cv2.COLORMAP_MAGMA)))
cv2.imwrite(f'{OUT}/m3_celah.png', big(cv2.applyColorMap(gap_u8, cv2.COLORMAP_MAGMA)))
pal = rng.integers(70, 255, (markers.max() + 2, 3)).astype(np.uint8)
b0 = (cv2.resize(img, (W * S, H * S)) * 0.4).astype(np.uint8)
mk = big(markers.astype(np.int32))
for i in ids:
    b0[mk == i] = pal[i]
cv2.imwrite(f'{OUT}/m4_b0.png', b0)
print('B0 instances:', len(ids), 'of', len(houses))
