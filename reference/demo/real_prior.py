"""Classical morphology on a REAL tile: SpaceNet 6 (Rotterdam), via the solaris repo sample data.

Steps: high-pass pansharpening (MS 300x300 + PAN 600x600) -> MBI -> gap map
(multi-directional black top-hat) -> classical baseline B0 (Otsu + morphology + watershed).
"""
import sys

import cv2
import numpy as np
import tifffile

SRC, OUT = sys.argv[1], sys.argv[2]
ms = tifffile.imread(f'{SRC}/ms1.tif').astype(np.float32)        # 300x300x4
pan = tifffile.imread(f'{SRC}/pan1.tif').astype(np.float32)      # 600x600

ms_up = cv2.resize(ms, (pan.shape[1], pan.shape[0]), interpolation=cv2.INTER_CUBIC)
rgb = ms_up[..., :3]
pan_n = (pan - pan.mean()) / pan.std() * rgb.mean(axis=2).std()
detail = pan_n - cv2.GaussianBlur(pan_n, (0, 0), 2.0)
sharp = rgb + detail[..., None]                                   # high-pass detail injection (keeps colour)


def stretch(a):
    out = np.zeros(a.shape, np.uint8)
    for c in range(a.shape[2]):
        lo, hi = np.percentile(a[..., c], [1, 99.5])
        out[..., c] = np.clip((a[..., c] - lo) / (hi - lo) * 255, 0, 255)
    return out


img = stretch(sharp)                       # BGR order as rendered earlier looked natural
H, W = img.shape[:2]


def line_se(length, angle):
    k = np.zeros((length, length), np.uint8)
    c = length // 2
    t = np.deg2rad(angle)
    dx, dy = np.cos(t) * c, np.sin(t) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


ANGLES = [0, 45, 90, 135]
bright = cv2.medianBlur(img.max(axis=2), 5).astype(np.float32)
th_prev = np.zeros_like(bright)
mbi = np.zeros_like(bright)
for s in [21, 41, 61, 81]:
    th = np.min([cv2.morphologyEx(bright, cv2.MORPH_TOPHAT, line_se(s, a)) for a in ANGLES], axis=0)
    mbi += np.abs(th - th_prev)
    th_prev = th
mbi /= 4

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
gap = np.max([cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, line_se(9, a)) for a in ANGLES], axis=0)


def norm(a, p=99.5):
    a = a - a.min()
    return np.clip(255 * a / max(np.percentile(a, p), 1e-6), 0, 255).astype(np.uint8)


mbi_u8, gap_u8 = norm(mbi), norm(gap)

_, bmask = cv2.threshold(cv2.GaussianBlur(mbi_u8, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bmask = cv2.morphologyEx(bmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
bmask = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
_, gapbin = cv2.threshold(gap_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
core = cv2.erode(((bmask > 0) & (gapbin == 0)).astype(np.uint8), np.ones((3, 3), np.uint8))
_, markers = cv2.connectedComponents(core)
markers = markers + 1
markers[(bmask > 0) & (core == 0)] = 0
markers = cv2.watershed(cv2.cvtColor(gap_u8, cv2.COLOR_GRAY2BGR), markers.astype(np.int32))
ids = [i for i in np.unique(markers) if i >= 2 and (markers == i).sum() > 150]

rng = np.random.default_rng(3)
pal = rng.integers(70, 255, (markers.max() + 2, 3)).astype(np.uint8)
b0 = (img * 0.45).astype(np.uint8)
for i in ids:
    b0[markers == i] = (0.35 * img[markers == i] + 0.65 * pal[i]).astype(np.uint8)
    cnts, _ = cv2.findContours((markers == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(b0, cnts, -1, (60, 60, 255), 2)

Y0, Y1, X0, X1 = 30, 260, 370, 600          # crop: dense row-house blocks (top right)
S = 3
up = lambda a: cv2.resize(a[Y0:Y1, X0:X1], ((X1 - X0) * S, (Y1 - Y0) * S), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(f'{OUT}/r0_full.png', cv2.resize(img, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC))
cv2.imwrite(f'{OUT}/r1_citra.png', up(img))
cv2.imwrite(f'{OUT}/r2_mbi.png', up(cv2.applyColorMap(mbi_u8, cv2.COLORMAP_MAGMA)))
cv2.imwrite(f'{OUT}/r3_celah.png', up(cv2.applyColorMap(gap_u8, cv2.COLORMAP_MAGMA)))
cv2.imwrite(f'{OUT}/r4_b0.png', up(b0))
print('B0 segments (full tile):', len(ids))
