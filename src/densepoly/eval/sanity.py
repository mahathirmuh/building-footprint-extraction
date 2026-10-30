"""Sanity checks of the evaluation pipeline on real ground truth (docs/EVALUATION.md §6.1, §6.2, §6.4, §6.6) and
their report, docs/sanity_report.md, rendered from the JSON the checks write (no number is typed by hand).

§6.1 evaluates the GT as predictions twice: (i) without the GT whose mask is empty, where every ideal value must
hold, and (ii) as released, where exactly those empty masks stay unmatched. With overlapping GT polygons, GT as
predictions also merges and splits some houses; the expected sets follow from the GT alone (gt_overlap_sets).
§6.6 resamples a perturbed copy of the GT (shifted houses, some dropped, random scores) to check the bootstrap AP
against COCOeval and to show what passing duplicate imgIds to COCOeval would do instead. §6.4 compares the saved
evaluation of HiSup's released val predictions with the numbers of its paper.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import platform
from collections import defaultdict

import numpy as np
import shapely
from omegaconf import OmegaConf
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

from densepoly import paths
from densepoly.eval import coco_io, instance, run_eval, stats
from densepoly.utils.io import read_json, write_json

COVER = instance.COVER_THRESHOLD
PACKAGES = ("numpy", "pandas", "pyarrow", "scipy", "shapely", "pycocotools", "opencv-python-headless", "multiprocess")
AP_TOLERANCE = 0.3  # §6.4: AP points between our evaluation of HiSup's release and its paper
# §6.4 rows: (key of configs/paper_numbers.yaml, our value in the paper's unit). HiSup's C-IoU and IoU come from
# cIoU.py, which averages over images with predictions, hence CIoU_p2p and IoU_p2p.
HISUP_ROWS = (
    ("AP", lambda r: 100 * r["overall"]["AP"]),
    ("AP50", lambda r: 100 * r["overall"]["AP50"]),
    ("AP75", lambda r: 100 * r["overall"]["AP75"]),
    ("AR", lambda r: 100 * r["coco_stats"]["segm"]["AR100"]),
    ("AP_boundary", lambda r: 100 * r["overall"]["AP_boundary"]),
    ("PoLiS", lambda r: r["overall"]["PoLiS"]),
    ("CIoU", lambda r: 100 * r["overall"]["CIoU_p2p"]),
    ("IoU", lambda r: 100 * r["overall"]["IoU_p2p"]),
)


def environment() -> dict:
    """Versions that the numbers of the report depend on."""
    versions = {"python": platform.python_version(), "geos": shapely.geos_version_string}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"versions": versions, "platform": platform.platform(), "processor": platform.processor()}


def check(section: str, case: str, name: str, value, target: str, passed: bool) -> dict:
    return {"section": section, "case": case, "name": name, "value": value, "target": target, "passed": bool(passed)}


def gt_overlap_sets(gt: dict) -> tuple[set[int], set[int]]:
    """(covered, covering): ids of GT covered for ≥ 50% of their pixels by another GT of the same image, and of GT
    that cover ≥ 50% of another GT. With the GT used as predictions, instance.py must find exactly covered ∪ covering
    merged and covering split (its rules with p = another GT polygon)."""
    sizes = {img["id"]: (img["height"], img["width"]) for img in gt["images"]}
    by_image = defaultdict(list)
    for ann in gt["annotations"]:
        by_image[ann["image_id"]].append(ann)
    covered, covering = set(), set()
    for image_id, anns in by_image.items():
        rles = [instance.instance_rle(a["segmentation"], *sizes[image_id]) for a in anns]
        areas = mask_utils.area(rles)
        ious = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)), float).reshape(len(rles), len(rles))
        for i, j in zip(*np.nonzero(np.triu(ious > 0, k=1)), strict=True):
            inter = mask_utils.area(mask_utils.merge([rles[i], rles[j]], intersect=True))
            for big, small in ((i, j), (j, i)):
                if areas[small] > 0 and inter >= COVER * areas[small]:
                    covering.add(anns[big]["id"])
                    covered.add(anns[small]["id"])
    return covered, covering


def perturbed_predictions(gt: dict, seed: int = 0, drop: float = 0.1, max_shift: int = 2) -> list[dict]:
    """Results for §6.6: each GT polygon (empty masks skipped) shifted by whole pixels (|dx|, |dy| ≤ max_shift), with
    a random score in [0.5, 1); a fraction drop is left out. Random numbers are drawn for every annotation in order."""
    rng = np.random.default_rng(seed)
    empty = set(coco_io.empty_mask_ids(gt))
    results = []
    for ann in gt["annotations"]:
        dropped, (dx, dy), score = rng.random() < drop, rng.integers(-max_shift, max_shift + 1, 2), rng.uniform(0.5, 1)
        if dropped or ann["id"] in empty:
            continue
        rings = [(np.reshape(ring, (-1, 2)) + [dx, dy]).ravel().tolist() for ring in ann["segmentation"]]
        points = np.concatenate([np.reshape(ring, (-1, 2)) for ring in rings]).astype(float)
        results.append(
            {
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "segmentation": rings,
                "score": float(score),
                "bbox": coco_io.ring_bbox(points),
                "theta_deg": None,
                "kappa": None,
            }
        )
    return results


def _case_name(dataset: str, split: str, variant: str | None) -> str:
    return f"{dataset}/{split}" + (f" ({variant})" if variant else "")


def gt_as_prediction(dataset: str, split: str, variant: str | None = None, workers: int = 1) -> dict:
    """§6.1 on one ground truth: the evaluations (ii) as released and, when it has empty masks, (i) without them."""
    case = _case_name(dataset, split, variant)
    gt = coco_io.load_gt(dataset, split, variant)
    empty = coco_io.empty_mask_ids(gt)
    as_released = run_eval.evaluate(
        dataset, split, "sanity_gt", "ref", coco_io.gt_as_predictions(gt), variant, released=True, workers=workers
    )
    tables = stats.load_run(dataset, "sanity_gt", "ref", split, variant)
    if empty:
        clean = coco_io.without_annotations(gt, empty)
        name = split if variant is None else f"{split}_{variant}"
        clean_path = paths.out_root() / "sanity" / "gt" / dataset / f"{name}_noempty.json"
        write_json(clean_path, clean, indent=None)
        results = coco_io.gt_as_predictions(clean)
        without_empty = run_eval.evaluate(
            dataset, split, "sanity_gt_noempty", "ref", results, variant, released=True, workers=workers,
            gt_path=clean_path,
        )  # fmt: skip
        clean_tables = stats.load_run(dataset, "sanity_gt_noempty", "ref", split, variant)
    else:
        clean, without_empty, clean_tables = gt, as_released, tables
    covered, covering = gt_overlap_sets(clean)
    merged, split_gt = int(clean_tables.per_image["n_merged"].sum()), int(clean_tables.per_image["n_split"].sum())
    unmatched = int((tables.per_image["n_gt"] - tables.per_image["n_matched"]).sum())
    min_iou = float(tables.pairs["iou"].min()) if len(tables.pairs) else None
    o = without_empty["overall"]
    checks = [
        check("6.1", case, "(i) AP", o["AP"], "1 ± 0,001", abs(o["AP"] - 1) <= 1e-3),
        check("6.1", case, "(i) AP_boundary", o["AP_boundary"], "1 ± 0,001", abs(o["AP_boundary"] - 1) <= 1e-3),
        check("6.1", case, "(i) CIoU", o["CIoU"], "1 (± 1e-6)", o["CIoU"] >= 1 - 1e-6),
        check("6.1", case, "(i) CIoU_p2p", o["CIoU_p2p"], "1 (± 1e-6)", o["CIoU_p2p"] >= 1 - 1e-6),
        check("6.1", case, "(i) PoLiS vendor", o["PoLiS"], "None, 0 citra valid",
              o["PoLiS"] is None and o["PoLiS_valid_images"] == 0),
        check("6.1", case, "(i) PoLiS_paired", o["PoLiS_paired"], "0", o["PoLiS_paired"] <= 1e-9),
        check("6.1", case, "(i) MTA_inst (°)", o["MTA_inst"], "≈ 0 (≤ 0,01)", o["MTA_inst"] <= 0.01),
        check("6.1", case, "(i) MTA vendor (°)", o["MTA"], "≤ 0,5", o["MTA"] <= 0.5),
        check("6.1", case, "(i) GT merged", merged, f"= {len(covered | covering)} (GT tumpang tindih)",
              merged == len(covered | covering)),
        check("6.1", case, "(i) GT split", split_gt, f"= {len(covering)} (GT tumpang tindih)",
              split_gt == len(covering)),
        check("6.1", case, "(ii) GT tak ter-match", unmatched, f"= {len(empty)} (GT bermask kosong)",
              unmatched == len(empty)),
        check("6.1", case, "(ii) IoU pasangan terkecil", min_iou, "1", min_iou is not None and min_iou >= 1 - 1e-9),
    ]  # fmt: skip
    keep = ("AP", "AP_boundary", "CIoU", "CIoU_p2p", "PoLiS", "PoLiS_valid_images", "MTA", "PoLiS_paired", "MTA_inst",
            "merge_rate", "split_rate", "recall_inst50", "vertex_F1", "n_images", "n_gt", "n_pairs")  # fmt: skip
    return {
        "case": case,
        "images": len(gt["images"]),
        "annotations": len(gt["annotations"]),
        "empty_masks": len(empty),
        "pred_check_as_released": as_released["pred_check"],
        "without_empty": {k: without_empty["overall"][k] for k in keep},
        "as_released": {k: as_released["overall"][k] for k in keep},
        "seconds": {"without_empty": without_empty["seconds"], "as_released": as_released["seconds"]},
        "checks": checks,
    }


def determinism(dataset: str = "crowdai", split: str = "val_clean", workers: int = 1) -> dict:
    """§6.2: the same predictions evaluated twice give identical numbers (all but provenance and timings)."""
    gt = coco_io.load_gt(dataset, split)
    results = perturbed_predictions(gt, seed=1)
    first, second = (
        run_eval.evaluate(dataset, split, "sanity_repeat", "ref", results, released=True, workers=workers)
        for _ in range(2)
    )
    same = all(first[key] == second[key] for key in ("overall", "coco_stats", "per_stratum", "pred_check"))
    case = _case_name(dataset, split, None)
    return {"case": case, "checks": [check("6.2", case, "evaluasi dua kali identik", same, "identik", same)]}


def bootstrap_ap(
    dataset: str = "crowdai",
    split: str = "val_clean",
    n_rep: int = 1000,
    n_naive: int = 200,
    seed: int = 0,
    workers: int = 1,
) -> dict:
    """§6.6 on perturbed GT predictions: identity resample = stats[0]; a CI containing the point estimate; and the
    same resamples passed as duplicate imgIds straight to COCOeval, which keeps one copy of each image."""
    case = _case_name(dataset, split, None)
    gt = coco_io.load_gt(dataset, split)
    results = perturbed_predictions(gt, seed=seed)
    record = run_eval.evaluate(dataset, split, "sanity_perturbed", "ref", results, released=True, workers=workers)
    run = stats.load_run(dataset, "sanity_perturbed", "ref", split)
    cache = run.caches["segm"]
    ids = cache.img_ids
    point = record["coco_stats"]["segm"]["AP"]
    identity = stats.coco_ap(cache, cache.positions(ids))
    weights = stats.bootstrap_weights(len(ids), n_rep, seed)
    values = stats.metric_values(run, "AP", ids, weights)
    ci = [float(v) for v in np.percentile(values, [2.5, 97.5])]

    kept, _ = coco_io.check_predictions(results, gt, strict=False)
    coco_gt = coco_io.coco_from_dict(gt)
    coco_dt = coco_io.load_results(coco_gt, kept)
    naive, unique = [], []
    for w in weights[:n_naive]:
        evaluator = COCOeval(coco_gt, coco_dt, "segm")
        evaluator.params.imgIds = np.repeat(ids, w).tolist()
        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        naive.append(float(evaluator.stats[0]))
        unique.append(len(evaluator.params.imgIds) / len(ids))
    naive_ci = [float(v) for v in np.percentile(naive, [2.5, 97.5])]
    proper_ci = [float(v) for v in np.percentile(values[:n_naive], [2.5, 97.5])]
    width_ratio = (naive_ci[1] - naive_ci[0]) / (proper_ci[1] - proper_ci[0])
    checks = [
        check("6.6", case, "resampel identitas = stats[0]", identity, f"= {point}", identity == point),
        check("6.6", case, "CI 95% memuat nilai titik", ci, f"∋ {point:.4f}", ci[0] <= point <= ci[1]),
        check("6.6", case, "lebar CI cara naif / cara benar", width_ratio, "≠ 1 (naif membuang duplikat)",
              abs(width_ratio - 1) > 0.05),
    ]  # fmt: skip
    return {
        "case": case,
        "point_AP": point,
        "n_rep": n_rep,
        "ci95": ci,
        "sd": float(np.std(values, ddof=1)),
        "n_naive": n_naive,
        "proper_ci95_first_n_naive": proper_ci,
        "naive_ci95": naive_ci,
        "naive_unique_fraction_mean": float(np.mean(unique)),
        "width_ratio": width_ratio,
        "pred_check": record["pred_check"],
        "checks": checks,
    }


def load_paper_numbers() -> dict:
    return OmegaConf.to_container(OmegaConf.load(paths.configs_dir() / "paper_numbers.yaml"))


def hisup_reproduction() -> dict | None:
    """§6.4: our evaluation of HiSup's released val predictions (scripts/evaluate.py, exp hisup_release, val_full)
    against the numbers of its paper; None until that evaluation exists."""
    file = paths.metrics_file("crowdai", "hisup_release", "ref", "val_full")
    if not file.exists():
        return None
    record = read_json(file)
    paper = load_paper_numbers()["hisup"]["crowdai_val"]
    rows = []
    for key, ours in HISUP_ROWS:
        value = ours(record)
        rows.append(
            {"metric": key, "paper": paper["values"][key], "ours": value, "delta": value - paper["values"][key]}
        )
    ap = rows[0]
    case = "crowdai/val_full"
    return {
        "case": case,
        "metrics_file": file.relative_to(paths.out_root()).as_posix(),
        "created": record["created"],
        "git": record["git"],
        "source": paper["source"],
        "rows": rows,
        "other": {
            "CIoU (semua citra)": 100 * record["overall"]["CIoU"],
            "IoU (ΣI/ΣU)": 100 * record["overall"]["IoU"],
            "MTA (°)": record["overall"]["MTA"],
        },
        "pred_check": record["pred_check"],
        "seconds": record["seconds"],
        "checks": [
            check(
                "6.4",
                case,
                "AP HiSup: rilisan dievaluasi ulang vs paper",
                ap["ours"],
                f"{ap['paper']} ± {AP_TOLERANCE}",
                abs(ap["delta"]) <= AP_TOLERANCE,
            ),
        ],  # fmt: skip
    }


def all_checks(report: dict) -> list[dict]:
    parts = [*report["gt_as_prediction"], report["determinism"], report["bootstrap_ap"]]
    if report.get("hisup_reproduction"):
        parts.append(report["hisup_reproduction"])
    return [c for part in parts for c in part["checks"]]


def _fmt(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "ya" if value else "tidak"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _printed_decimals(value) -> int:
    """Decimals of a number as a paper printed it: 79.4 → 1, 0.726 → 3."""
    text = str(value)
    return len(text.split(".")[1]) if "." in text else 0


def render_report(report: dict) -> str:
    """docs/sanity_report.md from the report JSON."""
    checks = all_checks(report)
    failed = [c for c in checks if not c["passed"]]
    lines = [
        "# Laporan uji kewarasan pipeline evaluasi",
        "",
        "Dibuat otomatis oleh `scripts/sanity_checks.py` dari `outputs/metrics/sanity_report.json`; jangan diedit "
        "tangan. Uji dan targetnya: `docs/EVALUATION.md` §6.",
        "",
        f"- Dibuat: {report['created']}, commit `{report['git']}`.",
        f"- Mesin: {report['environment']['platform']}, prosesor {report['environment']['processor']}.",
        "- Versi: " + ", ".join(f"{k} {v}" for k, v in report["environment"]["versions"].items()) + ".",
        f"- **Hasil: {len(checks) - len(failed)} dari {len(checks)} uji lolos.**",
        "",
        "## §6.1 GT sebagai prediksi",
        "",
        "(i) GT bermask kosong dikeluarkan dari GT dan prediksi; (ii) GT apa adanya.",
    ]
    for part in report["gt_as_prediction"]:
        lines += [
            "",
            f"### {part['case']}",
            "",
            f"{part['images']} citra, {part['annotations']} anotasi GT, {part['empty_masks']} GT bermask kosong. "
            f"Pemeriksaan prediksi (ii): {part['pred_check_as_released']['too_few_points']} ring dengan < 3 titik "
            "berbeda (mask kosong bagi pycocotools, tidak dipakai PoLiS/MTA), "
            f"{part['pred_check_as_released']['dropped']} ring tak terbaca dibuang.",
            "",
            "| Uji | Nilai | Target | Hasil |",
            "|---|---|---|---|",
        ]
        lines += [f"| {c['name']} | {_fmt(c['value'])} | {c['target']} | {'lolos' if c['passed'] else '**gagal**'} |"
                  for c in part["checks"]]  # fmt: skip
    lines += ["", "## §6.2 Kasus tepi", "", "Tes unit (data sintetis) yang dijalankan skrip ini:", ""]
    lines += [f"- `{t['test']}`: {'lolos' if t['passed'] else '**gagal**'}" for t in report["edge_case_tests"]]
    lines += [""] + [f"- {c['case']}: {c['name']}: {'lolos' if c['passed'] else '**gagal**'}"
                     for c in report["determinism"]["checks"]]  # fmt: skip
    h = report.get("hisup_reproduction")
    lines += ["", "## §6.4 Reproduksi HiSup", ""]
    if h is None:
        lines.append("Belum ada: evaluasi `hisup_release` di `crowdai/val_full` belum dijalankan.")
    else:
        check_ = h["pred_check"]
        lines += [
            f"Prediksi rilisan HiSup dievaluasi ulang dengan pipeline ini (`{h['metrics_file']}`, "
            f"dibuat {h['created']}, commit `{h['git']}`). Pemeriksaan longgar: {check_['results']} prediksi, "
            f"{check_['dropped']} dibuang, {check_['too_few_points']} ring dua titik (ikut AP sebagai mask kosong, "
            "dikeluarkan dari PoLiS/MTA). C-IoU dan IoU paper mengikuti aturan `cIoU.py` (citra yang punya prediksi), "
            "jadi dibandingkan dengan `CIoU_p2p` dan `IoU_p2p`. Angka pipeline ini ditulis dengan satu desimal lebih "
            "banyak daripada paper.",
            "",
            f"Angka paper (`configs/paper_numbers.yaml`): {h['source']['paper'].strip()}; {h['source']['tables']}; "
            f"hlm. {h['source']['page']}.",
            "",
            "| Metrik | Paper | Pipeline ini | Selisih |",
            "|---|---|---|---|",
        ]
        for r in h["rows"]:
            d = _printed_decimals(r["paper"]) + 1
            lines.append(f"| {r['metric']} | {_fmt(r['paper'])} | {r['ours']:.{d}f} | {r['delta']:+.{d}f} |")
        other = ", ".join(f"{k} {v:.2f}" for k, v in h["other"].items())
        lines += [
            "",
            f"Varian lain dari pipeline ini: {other}.",
            "",
            "| Uji | Nilai | Target | Hasil |",
            "|---|---|---|---|",
        ]
        lines += [f"| {c['name']} | {_fmt(c['value'])} | {c['target']} | {'lolos' if c['passed'] else '**gagal**'} |"
                  for c in h["checks"]]  # fmt: skip
    b = report["bootstrap_ap"]
    lines += [
        "",
        "## §6.6 Bootstrap AP",
        "",
        f"Prediksi buatan di {b['case']}: GT digeser acak ±2 px, 10% dibuang, skor acak. "
        f"AP titik {_fmt(b['point_AP'])}; {b['n_rep']} replikasi: CI 95% {_fmt(b['ci95'])}, SD {_fmt(b['sd'])}.",
        f"Cara naif (imgIds duplikat langsung ke COCOeval, {b['n_naive']} replikasi pertama): rata-rata hanya "
        f"{b['naive_unique_fraction_mean']:.1%} citra unik yang tersisa, CI 95% {_fmt(b['naive_ci95'])} versus "
        f"{_fmt(b['proper_ci95_first_n_naive'])} dengan cara yang benar (rasio lebar {b['width_ratio']:.2f}).",
        "",
        "| Uji | Nilai | Target | Hasil |",
        "|---|---|---|---|",
    ]
    lines += [f"| {c['name']} | {_fmt(c['value'])} | {c['target']} | {'lolos' if c['passed'] else '**gagal**'} |"
              for c in b["checks"]]  # fmt: skip
    return "\n".join(lines) + "\n"


def load_report() -> dict:
    return read_json(paths.out_root() / "metrics" / "sanity_report.json")
