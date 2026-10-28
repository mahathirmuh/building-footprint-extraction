"""Cek cepat klaim METHOD.md (dibuat saat menyusun dokumen, 28 Okt 2026). BUKAN kode produksi.

- bolak-balik kontur marching squares -> IoU 1,000; cv2.findContours apa adanya -> 0,84 (0,81 untuk 10x10 px)
- filter kontak max/min -> pita 2 px di kedua sisi dinding bersama, rumah terisolasi tanpa kontak
- tanda sin 4theta: berbalik saat flip H/V dan transpose, tetap saat rot90
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.measure import find_contours
from skimage.morphology import disk
from skimage.draw import polygon as draw_polygon
from pycocotools import mask as cocomask

rng = np.random.default_rng(0)
H = W = 64


def shoelace(c):
    x, y = c[:, 1], c[:, 0]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def mask_to_exterior_ms(mask):
    padded = np.pad(mask.astype(np.uint8), 1)
    cs = find_contours(padded, 0.5)
    ring = max(cs, key=lambda c: abs(shoelace(c)))
    return ring[:, ::-1] - 1 + 0.5


def mask_to_exterior_cv2(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    return c  # raw pixel indices as (x, y), the usual practice


def rasterize(xy):
    rle = cocomask.frPyObjects([xy.ravel().tolist()], H, W)
    return cocomask.decode(cocomask.merge(rle)).astype(bool)


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def random_mask():
    m = np.zeros((H, W), bool)
    kind = rng.integers(3)
    if kind == 0:  # axis-aligned rectangle
        h, w = rng.integers(6, 20, 2)
        r, c = rng.integers(5, 30, 2)
        m[r:r + h, c:c + w] = True
    elif kind == 1:  # rotated rectangle
        cx, cy = rng.uniform(20, 44, 2)
        w, h = rng.uniform(8, 22, 2)
        ang = rng.uniform(0, 90)
        box = cv2.boxPoints(((cx, cy), (w, h), ang))
        rr, cc = draw_polygon(box[:, 1], box[:, 0], (H, W))
        m[rr, cc] = True
    else:  # L-shape
        r, c = rng.integers(8, 30, 2)
        m[r:r + 18, c:c + 8] = True
        m[r + 10:r + 18, c:c + 18] = True
    return m


ious_ms, ious_cv = [], []
for _ in range(300):
    m = random_mask()
    if m.sum() < 10:
        continue
    ious_ms.append(iou(rasterize(mask_to_exterior_ms(m)), m))
    ious_cv.append(iou(rasterize(mask_to_exterior_cv2(m)), m))
print(f"round-trip IoU marching squares: mean {np.mean(ious_ms):.4f} min {np.min(ious_ms):.4f}")
print(f"round-trip IoU cv2 raw indices : mean {np.mean(ious_cv):.4f} min {np.min(ious_cv):.4f}")

# small 10x10 building
m = np.zeros((H, W), bool); m[20:30, 20:30] = True
print("10x10 ms IoU", round(iou(rasterize(mask_to_exterior_ms(m)), m), 4),
      "| cv2 IoU", round(iou(rasterize(mask_to_exterior_cv2(m)), m), 4))

# contact filter
L = np.zeros((H, W), np.uint16)
L[10:30, 10:20] = 1
L[10:30, 20:30] = 2          # touching building 1
L[40:50, 40:50] = 3          # isolated
r = 2
fp = disk(r)
big = np.iinfo(L.dtype).max
lmax = ndi.maximum_filter(L, footprint=fp)
lmin = ndi.minimum_filter(np.where(L > 0, L, big), footprint=fp)
contact = (L > 0) & ((lmax != L) | (lmin != L))
cols = np.where(contact.any(0))[0]
print("contact cols:", cols.tolist(), "| isolated has contact:", bool(contact[40:50, 40:50].any()))


# theta and flips
def theta_of(mask):
    rows, cols_ = np.nonzero(mask)
    pts = np.column_stack([cols_, rows]).astype(np.float32)
    box = cv2.boxPoints(cv2.minAreaRect(pts))
    ex, ey = box[1] - box[0]
    return np.degrees(np.arctan2(ey, ex)) % 90


def rect_mask(angle_deg):
    big_m = np.zeros((128, 128), bool)
    box = cv2.boxPoints(((64, 64), (60, 30), angle_deg))
    rr, cc = draw_polygon(box[:, 1], box[:, 0], (128, 128))
    big_m[rr, cc] = True
    return big_m


for a in [10, 30, 55, 80]:
    m = rect_mask(a)
    t = theta_of(m)
    s, c = np.sin(np.radians(4 * t)), np.cos(np.radians(4 * t))
    th = theta_of(m[:, ::-1]); sh = np.sin(np.radians(4 * th)); ch = np.cos(np.radians(4 * th))
    tv = theta_of(m[::-1, :]); sv = np.sin(np.radians(4 * tv))
    tt = theta_of(m.T); st = np.sin(np.radians(4 * tt))
    tr = theta_of(np.rot90(m)); sr = np.sin(np.radians(4 * tr)); cr = np.cos(np.radians(4 * tr))
    print(f"angle {a:>2}: theta={t:5.1f} | hflip sin {s:+.2f}->{sh:+.2f} cos {c:+.2f}->{ch:+.2f} | "
          f"vflip sin {sv:+.2f} | transpose sin {st:+.2f} | rot90 sin {sr:+.2f} cos {cr:+.2f}")
