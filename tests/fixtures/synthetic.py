"""Seeded synthetic scenes with exactly known ground truth (PLAN.md Phase 0).

The source of truth is a list of COCO polygons in continuous (x, y) coordinates, where pixel (r, c) covers
[c, c+1] × [r, r+1] (docs/METHOD.md §0). The instance map is rasterised from those polygons with pycocotools and
painted by decreasing area, like the CrowdAI annotations (docs/DATA.md §2.3); the RGB image is then drawn from the
instance map. Polygons have no repeated closing point; the final convention follows the CrowdAI ground truth,
which is checked in Phase 1.

A scene has four groups, one per quadrant in random order:
- ``rect``: 2-3 isolated rotated rectangles, at least 4 px from any other building (never "contact");
- ``row``: 3-5 touching houses of equal depth whose shared walls use identical vertices, rotated as a whole;
- ``L``: one L-shaped house (6 vertices);
- ``gap_row``: 3-4 axis-aligned houses on integer coordinates, separated by dark gaps of 1-2 px.
Roofs are brighter than the textured ground; gap pixels are darker than both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi
from shapely.geometry import Polygon

SCENE_SIZE = 300  # CrowdAI tile side
QUADRANT_MARGIN = 3  # px between a group and its quadrant edge, so groups are >= 6 px apart
ISOLATION_PX = 4  # minimum polygon distance between isolated houses

_GROUND_RGB = np.array([96.0, 112.0, 88.0])
_ROOF_RGB = np.array(
    [[214, 120, 96], [226, 222, 210], [190, 80, 66], [206, 196, 150], [176, 182, 194], [228, 170, 110]], dtype=float
)
_GAP_LEVEL = (22.0, 40.0)


@dataclass(frozen=True, eq=False)
class Building:
    id: int  # value in the instance map
    kind: str  # rect | row | L | gap_row
    polygon: np.ndarray  # (k, 2) float (x, y) exterior ring, no repeated closing point
    theta_deg: float  # orientation in [0, 90), image coordinates with y down (docs/METHOD.md §3.3)
    neighbors: tuple[int, ...] = ()  # ids of buildings sharing a wall with this one


@dataclass(eq=False)
class Scene:
    seed: int
    image: np.ndarray  # (H, W, 3) uint8 RGB
    instance_map: np.ndarray  # (H, W) uint16, 0 = background
    buildings: list[Building]
    gap_mask: np.ndarray  # (H, W) bool, dark ground between consecutive gap_row houses
    gap_pairs: list[tuple[int, int, int]]  # (id_a, id_b, gap width in px)


def rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (x, y) points about the origin; positive angles turn +x towards +y (clockwise on screen)."""
    t = np.radians(angle_deg)
    return points @ np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])


def rect_polygon(cx: float, cy: float, w: float, h: float, angle_deg: float = 0.0) -> np.ndarray:
    """w × h rectangle centred at (cx, cy); its first edge points along angle_deg, so theta = angle_deg mod 90."""
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return rotate(local, angle_deg) + (cx, cy)


def l_polygon(cx: float, cy: float, w: float, h: float, arm: float, angle_deg: float = 0.0) -> np.ndarray:
    """L-shaped house (6 vertices) covering the top and left sides of a w × h box centred at (cx, cy)."""
    local = np.array([[0, 0], [w, 0], [w, arm], [arm, arm], [arm, h], [0, h]], dtype=float) - (w / 2, h / 2)
    return rotate(local, angle_deg) + (cx, cy)


def polygon_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    """Rasterise one (x, y) polygon the way pycocotools annToMask does."""
    rles = mask_utils.frPyObjects([polygon.ravel().tolist()], height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def instance_map_from_polygons(polygons: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    """Instance map with id i + 1 for polygons[i], painted by decreasing pixel area (small overlaps stay visible)."""
    masks = [polygon_mask(p, height, width) for p in polygons]
    out = np.zeros((height, width), np.uint16)
    for i in sorted(range(len(masks)), key=lambda i: (-int(masks[i].sum()), i)):
        out[masks[i]] = i + 1
    return out


def make_scene(seed: int, size: int = SCENE_SIZE) -> Scene:
    """Build one deterministic scene: the same seed always gives the same polygons, raster and image."""
    if size < 256:
        raise ValueError("scenes need at least 256 px so that every group fits in its quadrant")
    rng = np.random.default_rng(seed)
    half, m = size // 2, QUADRANT_MARGIN
    corners = [(0, 0), (half, 0), (0, half), (half, half)]
    boxes = [(x + m, y + m, x + half - m, y + half - m) for x, y in (corners[k] for k in rng.permutation(4))]

    polygons, kinds, thetas = [], [], []

    def add(polygon: np.ndarray, kind: str, theta: float) -> int:
        polygons.append(polygon)
        kinds.append(kind)
        thetas.append(float(theta) % 90.0)
        return len(polygons)

    for polygon, angle in _isolated_rects(rng, boxes[0]):
        add(polygon, "rect", angle)
    houses, angle = _touching_row(rng, boxes[1])
    row_ids = [add(p, "row", angle) for p in houses]
    add(*_l_house(rng, boxes[2]))
    gap_houses, strips, widths = _gap_row(rng, boxes[3])
    gap_ids = [add(p, "gap_row", 0.0) for p in gap_houses]

    neighbors: dict[int, set[int]] = {i: set() for i in range(1, len(polygons) + 1)}
    for a, b in pairwise(row_ids):
        neighbors[a].add(b)
        neighbors[b].add(a)
    gap_pairs = [(a, b, g) for (a, b), g in zip(pairwise(gap_ids), widths, strict=True)]

    instance_map = instance_map_from_polygons(polygons, size, size)
    gap_mask = np.zeros((size, size), bool)
    for r0, r1, c0, c1 in strips:
        gap_mask[r0:r1, c0:c1] = True
    adjacency = {i: set(n) for i, n in neighbors.items()}
    for a, b, _ in gap_pairs:
        adjacency[a].add(b)
        adjacency[b].add(a)
    image = _render(instance_map, gap_mask, adjacency, rng)

    buildings = [
        Building(i, kinds[i - 1], polygons[i - 1], thetas[i - 1], tuple(sorted(neighbors[i])))
        for i in range(1, len(polygons) + 1)
    ]
    return Scene(seed, image, instance_map, buildings, gap_mask, gap_pairs)


def to_coco(scenes: Sequence[Scene], category_id: int = 100, file_names: Sequence[str] | None = None) -> dict:
    """COCO ground truth for a list of scenes: image ids start at 1, one annotation per building in id order.

    category_id 100 follows the Pix2Poly WHU conversion (docs/DATA.md §3.2); the CrowdAI value is checked in Phase 1.
    """
    if file_names is None:
        file_names = [f"synthetic_{scene.seed:04d}.png" for scene in scenes]
    images, annotations = [], []
    for image_id, (scene, file_name) in enumerate(zip(scenes, file_names, strict=True), start=1):
        height, width = scene.instance_map.shape
        images.append({"id": image_id, "file_name": file_name, "height": height, "width": width})
        for b in scene.buildings:
            rle = mask_utils.merge(mask_utils.frPyObjects([b.polygon.ravel().tolist()], height, width))
            (x0, y0), (x1, y1) = b.polygon.min(axis=0), b.polygon.max(axis=0)
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [b.polygon.ravel().tolist()],
                    "area": float(mask_utils.area(rle)),
                    "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                    "iscrowd": 0,
                }
            )
    categories = [{"id": category_id, "name": "building", "supercategory": "building"}]
    return {"images": images, "annotations": annotations, "categories": categories}


def _shift_into(rng: np.random.Generator, points: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Random translation keeping every point inside box = (x0, y0, x1, y1)."""
    (lx, ly), (hx, hy) = points.min(axis=0), points.max(axis=0)
    x0, y0, x1, y1 = box
    if hx - lx > x1 - x0 or hy - ly > y1 - y0:
        raise ValueError("building group does not fit in its quadrant")
    return np.array([rng.uniform(x0 - lx, x1 - hx), rng.uniform(y0 - ly, y1 - hy)])


def _isolated_rects(rng, box):
    target = int(rng.integers(2, 4))
    placed, out = [], []
    for _ in range(200):
        if len(out) == target:
            break
        angle = rng.uniform(5, 85)
        local = rect_polygon(0, 0, rng.uniform(16, 40), rng.uniform(14, 32), angle)
        polygon = np.round(local + _shift_into(rng, local, box), 2)
        shape = Polygon(polygon)
        if all(shape.distance(other) >= ISOLATION_PX for other in placed):
            placed.append(shape)
            out.append((polygon, angle))
    if len(out) < 2:
        raise RuntimeError("could not place two isolated rectangles")
    return out


def _touching_row(rng, box):
    """Equal-depth houses side by side. Wall corners are computed once and shared by both neighbours, so the
    shared walls are bit-identical and pycocotools splits their pixels without overlap."""
    n = int(rng.integers(3, 6))
    widths = rng.uniform(14, 22, n)
    depth = rng.uniform(24, 36)
    angle = rng.uniform(0, 90)
    xs = np.concatenate([[0.0], np.cumsum(widths)]) - widths.sum() / 2
    corners = rotate(np.array([[x, -depth / 2] for x in xs] + [[x, depth / 2] for x in xs]), angle)
    corners = np.round(corners + _shift_into(rng, corners, box), 2)
    top, bottom = corners[: n + 1], corners[n + 1 :]
    return [np.array([top[i], top[i + 1], bottom[i + 1], bottom[i]]) for i in range(n)], angle


def _l_house(rng, box):
    angle = rng.uniform(0, 90)
    local = l_polygon(0, 0, rng.uniform(36, 50), rng.uniform(36, 50), rng.uniform(14, 20), angle)
    return np.round(local + _shift_into(rng, local, box), 2), "L", angle


def _gap_row(rng, box):
    """Axis-aligned houses separated by 1-2 px gaps. Returns polygons, gap strips (r0, r1, c0, c1) and gap widths."""
    n = int(rng.integers(3, 5))
    widths = rng.integers(14, 23, n)
    gaps = rng.integers(1, 3, n - 1)
    depth = int(rng.integers(24, 37))
    starts = np.concatenate([[0], np.cumsum(widths[:-1] + gaps)])
    length = int(starts[-1] + widths[-1])
    vertical = bool(rng.integers(2))  # stack along y instead of x
    extent_x, extent_y = (depth, length) if vertical else (length, depth)
    x0, y0, x1, y1 = box
    ox = int(rng.integers(x0, x1 - extent_x + 1))
    oy = int(rng.integers(y0, y1 - extent_y + 1))

    polygons, strips = [], []
    for i in range(n):
        a, b = int(starts[i]), int(starts[i] + widths[i])  # span of house i along the row
        if vertical:
            polygons.append(np.array([[ox, oy + a], [ox + depth, oy + a], [ox + depth, oy + b], [ox, oy + b]], float))
        else:
            polygons.append(np.array([[ox + a, oy], [ox + b, oy], [ox + b, oy + depth], [ox + a, oy + depth]], float))
        if i < n - 1:
            g = int(gaps[i])
            strips.append((oy + b, oy + b + g, ox, ox + depth) if vertical else (oy, oy + depth, ox + b, ox + b + g))
    return polygons, strips, [int(g) for g in gaps]


def _render(instance_map, gap_mask, adjacency, rng):
    """Textured ground, one roof colour per building (different from its neighbours), dark gap pixels."""
    height, width = instance_map.shape
    smooth = ndi.gaussian_filter(rng.normal(0.0, 1.0, (height, width)), sigma=12)
    smooth *= 14.0 / (np.abs(smooth).max() + 1e-9)
    image = _GROUND_RGB + smooth[..., None] + rng.normal(0.0, 6.0, (height, width, 3))
    colours: dict[int, int] = {}
    for bid in sorted(adjacency):
        taken = {colours[j] for j in adjacency[bid] if j in colours}
        colours[bid] = int(rng.choice([k for k in range(len(_ROOF_RGB)) if k not in taken]))
        roof = instance_map == bid
        image[roof] = _ROOF_RGB[colours[bid]] + rng.uniform(-8, 8, 3) + rng.normal(0.0, 4.0, (int(roof.sum()), 3))
    image[gap_mask] = rng.uniform(*_GAP_LEVEL) + rng.normal(0.0, 3.0, (int(gap_mask.sum()), 3))
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)
