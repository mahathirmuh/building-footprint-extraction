"""Prototype regularisasi poligon (METHOD.md §7), dibuat saat menyusun dokumen (28 Okt 2026).

BUKAN kode produksi. Dipakai untuk membuktikan bahwa spesifikasi bisa dicapai:
- kotak lewat momen orde-2 (bukan bounding box) -> IoU 0,98-0,99 vs kontur mentah 0,95-0,97 (tepi berderau);
- bentuk L / T / satu dinding miring tanpa derau -> 6 / 8 / 5 verteks tepat di 8 sudut rotasi.
Jalankan: python regularize_proto.py [--debug]
"""
import sys
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely import affinity
from skimage.measure import find_contours
from skimage.draw import polygon as draw_polygon

rng = np.random.default_rng(1)
S = 160
DEBUG = "--debug" in sys.argv


def shoelace(c):
    x, y = c[:, 1], c[:, 0]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def mask_to_exterior(mask):
    cs = find_contours(np.pad(mask.astype(np.uint8), 1), 0.5)
    ring = max(cs, key=lambda c: abs(shoelace(c)))
    return ring[:, ::-1] - 1 + 0.5


def theta_mar(mask):
    rows, cols = np.nonzero(mask)
    b = cv2.boxPoints(cv2.minAreaRect(np.column_stack([cols, rows]).astype(np.float32)))
    ex, ey = b[1] - b[0]
    return np.arctan2(ey, ex) % (np.pi / 2)


def rot(pts, th, c):
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (pts - c) @ R.T + c


def moment_rect(mask, th):
    rows, cols = np.nonzero(mask)
    pts = np.column_stack([cols + 0.5, rows + 0.5])
    c = pts.mean(0)
    q = rot(pts, -th, c)
    w = np.sqrt(12 * q[:, 0].var() + 1)
    h = np.sqrt(12 * q[:, 1].var() + 1)
    cq = q.mean(0)
    corners = np.array([[cq[0] - w / 2, cq[1] - h / 2], [cq[0] + w / 2, cq[1] - h / 2],
                        [cq[0] + w / 2, cq[1] + h / 2], [cq[0] - w / 2, cq[1] + h / 2]])
    return Polygon(rot(corners, th, c))


def edge_angle(a, b):
    d = b - a
    return np.degrees(np.arctan2(d[1], d[0])) % 180


def classify(a, b, delta, l_obl):
    ang = edge_angle(a, b)
    dh, dv = min(ang, 180 - ang), abs(ang - 90)
    if dh <= delta:
        return "H"
    if dv <= delta:
        return "V"
    if np.hypot(*(b - a)) < l_obl:            # short oblique edge = staircase / chamfer noise
        return "H" if dh < dv else "V"
    return "O"


def offset(cls, a, b):
    return (a[1] + b[1]) / 2 if cls == "H" else (a[0] + b[0]) / 2


def regularize_rectilinear(pts, delta, l_min, l_obl, eps):
    """pts: DP polygon in the theta-aligned frame. Returns new vertices."""
    n = len(pts)
    # 1. edges with class and length-weighted offset
    E = []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        cls = classify(a, b, delta, l_obl)
        L = float(np.hypot(*(b - a)))
        E.append({"cls": cls, "a": a, "b": b, "len": L,
                  "off": offset(cls, a, b) if cls != "O" else None})

    def merge(E):
        out = []
        for e in E:
            if out and out[-1]["cls"] == e["cls"] and e["cls"] in "HV":
                p = out[-1]
                tot = p["len"] + e["len"]
                p["off"] = (p["off"] * p["len"] + e["off"] * e["len"]) / max(tot, 1e-9)
                p["len"], p["b"] = tot, e["b"]
            else:
                out.append(dict(e))
        if len(out) > 1 and out[0]["cls"] == out[-1]["cls"] and out[0]["cls"] in "HV":
            last = out.pop()
            f = out[0]
            tot = f["len"] + last["len"]
            f["off"] = (f["off"] * f["len"] + last["off"] * last["len"]) / max(tot, 1e-9)
            f["len"], f["a"] = tot, last["a"]
        return out

    E = merge(E)
    # 2. iteratively delete the shortest edge below l_min, then re-merge neighbours
    while len(E) > 4:
        i = int(np.argmin([e["len"] for e in E]))
        if E[i]["len"] >= l_min:
            break
        del E[i]
        E = merge(E)
    # 3. lines and intersections
    def line(e):
        if e["cls"] == "H":
            return np.array([0.0, e["off"]]), np.array([1.0, 0.0])
        if e["cls"] == "V":
            return np.array([e["off"], 0.0]), np.array([0.0, 1.0])
        d = e["b"] - e["a"]
        return e["a"], d / max(np.hypot(*d), 1e-9)

    out = []
    for i in range(len(E)):
        (p1, d1), (p2, d2) = line(E[i - 1]), line(E[i])
        M = np.array([d1, -d2]).T
        if abs(np.linalg.det(M)) < 1e-6:
            p = E[i]["a"]
        else:
            t = np.linalg.solve(M, p2 - p1)
            p = p1 + t[0] * d1
            if np.hypot(*(p - E[i]["a"])) > 2 * eps + 2:
                p = E[i]["a"]
        out.append(p)
    if DEBUG:
        print("     edges:", [(e["cls"], round(e["len"], 1)) for e in E])
    return np.array(out)


def regularize(mask, th, rho=0.9, k=0.04, delta=15):
    raw = Polygon(mask_to_exterior(mask)).buffer(0)
    A = raw.area
    rect = moment_rect(mask, th)
    r_iou = rect.intersection(raw).area / rect.union(raw).area
    if r_iou >= rho:
        return rect, raw, "rect"
    eps = k * np.sqrt(A)
    l_min, l_obl = max(2.0, 2 * eps), max(6.0, 4 * eps)
    c = np.array(raw.centroid.coords[0])
    r_pts = rot(np.array(raw.exterior.coords)[:-1], -th, c)
    dp = np.array(Polygon(r_pts).simplify(eps, preserve_topology=True).exterior.coords)[:-1]
    if DEBUG:
        print(f"   theta={np.degrees(th):.1f} eps={eps:.2f} dp={len(dp)} rect_iou={r_iou:.3f}")
    pts = regularize_rectilinear(dp, delta, l_min, l_obl, eps)
    poly = Polygon(rot(pts, th, c)).simplify(1e-3)
    if (not poly.is_valid) or poly.area <= 0 or poly.intersection(raw).area / poly.union(raw).area < 0.7:
        return Polygon(rot(dp, th, c)), raw, "dp-fallback"
    return poly, raw, "snap"


def raster(poly):
    xs, ys = poly.exterior.xy
    m = np.zeros((S, S), bool)
    # +0.5 shift: draw_polygon treats coordinates as pixel indices (centres)
    rr, cc = draw_polygon(np.array(ys) - 0.5, np.array(xs) - 0.5, (S, S))
    m[rr, cc] = True
    return m


def jagged(mask):
    noise = rng.random(mask.shape) < 0.25
    edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
    m = mask.copy()
    m[edge & noise] = ~m[edge & noise]
    return m


def nv(p):
    return len(p.exterior.coords) - 1


def iou(a, b):
    return a.intersection(b).area / a.union(b).area


shapes = {
    "rect": (Polygon(cv2.boxPoints(((80, 80), (70, 40), 0))), 4),
    "L": (Polygon([[40, 40], [110, 40], [110, 70], [70, 70], [70, 120], [40, 120]]), 6),
    "T": (Polygon([[30, 50], [130, 50], [130, 80], [95, 80], [95, 125], [65, 125], [65, 80], [30, 80]]), 8),
    "chamfer": (Polygon([[40, 40], [120, 40], [120, 85], [95, 115], [40, 115]]), 5),
    "small": (Polygon(cv2.boxPoints(((80, 80), (12, 9), 0))), 4),
}
for name, (base, expect) in shapes.items():
    res = []
    for ang in [0, 10, 20, 33, 45, 58, 70, 85]:
        gt = affinity.rotate(base, ang, origin=(80, 80))
        m = raster(gt)
        mj = jagged(m) if name != "small" else m
        if DEBUG and name in ("L",):
            print(f" {name} ang={ang}")
        out, raw, path = regularize(mj, theta_mar(mj))
        res.append((ang, path, nv(out), round(iou(out, gt), 3), round(iou(raw, gt), 3)))
    ok = sum(r[2] == expect for r in res)
    mean_gain = np.mean([r[3] - r[4] for r in res])
    print(f"{name:>8}: expect {expect} verts -> {ok}/{len(res)} | mean IoU(out)-IoU(raw) = {mean_gain:+.3f}")
    for r in res:
        print("          ", r)
