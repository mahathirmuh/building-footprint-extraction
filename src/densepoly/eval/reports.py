"""Tables of the comparison with other methods and of the density analysis (PLAN.md Phase 8; docs/EXPERIMENTS.md §6,
§8 points 4-5, §9), built like those of eval/tables.py from the saved evaluations:

- sota: the FULL final against HiSup's released predictions on crowdai/val_clean, with the decision of §8 point 5.
- paper: numbers reported by other papers (configs/paper_numbers.yaml), never mixed with ours.
- strata: every evaluated experiment per density stratum of test (N2, descriptive).
- n2: the effect of each claim per stratum and Δdense − Δsparse (stratified bootstrap), with the decision of §8 point 4.
- p2p: WHU test against Pix2Poly's ground truth, next to the numbers of the Pix2Poly paper (approximate comparison).
- efficiency: parameters, network speed and post-processing time (scripts/benchmark.py).

Importing this module registers the tables in eval/tables.py (BUILDERS), so scripts/tables.py writes them too.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from densepoly import paths
from densepoly.eval import stats, tables
from densepoly.eval.tables import BOOTSTRAP_SEED, CI_REPLICATES, LABELS, SEEDS, cell
from densepoly.utils.io import read_json

ALL_IMAGES = "semua"
MAIN_STRATA = ("jarang", "sedang", "padat")
SECONDARY_STRATA = ("bersentuhan", "tidak_bersentuhan")
SPARSE_STRATUM, DENSE_STRATUM = "jarang", "padat"

SOTA_SPLIT = "val_clean"
SOTA_REFERENCE = ("hisup_release", "ref")
SOTA_AP_GAP = 0.03  # §8 point 5: at most 3 AP points behind HiSup
SOTA_METRICS = ("AP", "AP50", "AP_boundary", "IoU", "PoLiS", "MTA", "CIoU", "CIoU_p2p", "N_ratio")
SOTA_HEADER = ("AP", "AP50", "AP_b", "IoU", "PoLiS ↓", "MTA ↓", "C-IoU", "C-IoU_p2p", "N-ratio")
SOTA_DELTAS = ("AP", "AP_boundary", "IoU", "CIoU", "PoLiS", "MTA")

PAPER_BLOCKS = {"crowdai": "crowdai_val", "whu": "whu_test"}
PAPER_COLUMNS = (
    "AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L", "AR", "AR50", "AR75", "AR_S", "AR_M", "AR_L",
    "AP_boundary", "IoU", "CIoU", "N_ratio", "PoLiS", "MTA", "FPS",
)  # fmt: skip
PAPER_HEADER = {"AP_boundary": "AP_b", "CIoU": "C-IoU", "N_ratio": "N-ratio", "PoLiS": "PoLiS ↓", "MTA": "MTA ↓"}

STRATA_METRICS = (
    "AP", "AP_boundary", "IoU", "CIoU", "PoLiS_paired", "MTA_inst", "merge_rate", "split_rate", "recall_inst50",
)  # fmt: skip
STRATA_SHOWN = ("AP", "merge_rate", "split_rate", "PoLiS_paired", "MTA_inst", "CIoU")

# The comparisons whose effect is followed over the strata: those of the claims on CrowdAI (§8 points 1-2); on WHU only the
# step that needs no second training, as a description (one seed).
N2_COMPARISONS = {
    "crowdai": (
        {"name": "N1", "a": "b5", "b": "full_p0", "metrics": ("PoLiS_paired", "MTA_inst"), "seeds": SEEDS, "base": "n1"},
        {"name": "N3", "a": "full_p0", "b": "full_p2", "metrics": ("merge_rate",), "seeds": SEEDS, "base": "n3"},
    ),
    "whu": (
        {
            "name": "Pemanduan θ",
            "a": "b5p",
            "b": "full_p0",
            "metrics": ("PoLiS_paired", "MTA_inst"),
            "seeds": (0,),
            "base": None,
        },
    ),
}

P2P_VARIANT = "p2p"
P2P_METRICS = ("IoU_p2p", "CIoU_p2p", "N_ratio_p2p", "PoLiS", "MTA")  # Pix2Poly's rules (docs/EVALUATION.md §2.2)
P2P_PAPER_KEYS = ("IoU", "CIoU", "N_ratio", "PoLiS", "MTA")  # the same metrics as the papers name them
P2P_HEADER = ("IoU", "C-IoU", "N-ratio", "PoLiS ↓", "MTA ↓")
P2P_PERCENT = {"IoU", "CIoU"}  # printed in percent by the papers


def strata_image_ids(dataset: str, split: str, splits_dir: Path | None = None) -> dict[str, list[int]]:
    """{stratum: image ids} of the frozen density file: the three density strata, then touching and not touching."""
    density = read_json(paths.density_file(dataset, paths.cache_split(split), splits_dir))
    groups = {**density.get("secondary", {}).get("strata", {}), **density["main"]["strata"]}
    return {name: [int(i) for i in groups[name]] for name in (*MAIN_STRATA, *SECONDARY_STRATA) if groups.get(name)}


def load_paper_numbers() -> dict:
    return OmegaConf.to_container(OmegaConf.load(paths.configs_dir() / "paper_numbers.yaml"))


def _printed(value) -> str:
    """A number of another paper as it was printed there, with a decimal comma."""
    return "—" if value is None else str(value).replace(".", ",")


def _runs_of(dataset: str, exp: str, split: str, variant: str | None = None) -> list[str]:
    candidates = [paths.run_name(None), *map(paths.run_name, SEEDS)]
    return [run for run in candidates if paths.metrics_file(dataset, exp, run, split, variant).exists()]


def _records(dataset: str, exp: str, runs, split: str, variant: str | None = None) -> list[dict]:
    return [read_json(paths.metrics_file(dataset, exp, run, split, variant)) for run in runs]


def _group_values(records: list[dict], group: str, metrics) -> dict:
    """{metric: per-run values and their mean ± SD} of all images or of one stratum, from the metric records."""
    rows = [r["overall"] if group == ALL_IMAGES else r["per_stratum"].get(group, {}) for r in records]
    values = {metric: [row.get(metric) for row in rows] for metric in metrics}
    return {metric: {"per_seed": v, "summary": tables.summarize(v)} for metric, v in values.items()}


def _efficiency(dataset: str, exp: str) -> dict | None:
    """The saved benchmark of an experiment (seed 0, or the seedless run), with its file; None when it was not run."""
    for run in (paths.run_name(0), paths.run_name(None)):
        file = paths.efficiency_file(dataset, exp, run)
        if file.exists():
            return {**read_json(file), "file": file.relative_to(paths.out_root()).as_posix()}
    return None


# --- sota: FULL final against HiSup on the clean val tiles


def decide_sota(ap: dict) -> dict:
    """§8 point 5, from stats.compare of AP (ours − HiSup, seed-averaged): at most 3 points behind."""
    return {
        "rule": "EXPERIMENTS §8 point 5",
        "delta": ap["delta"],
        "ci95": ap["ci95"],
        "limit": -SOTA_AP_GAP,
        "supported": ap["delta"] >= -SOTA_AP_GAP,
    }


def sota(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """HiSup (released predictions) and the FULL final on val_clean: overall, per density stratum, and Δ with its CI."""
    split, (ref_exp, ref_run) = SOTA_SPLIT, SOTA_REFERENCE
    final = tables.full_final(dataset)[0]["decision"]["choice"]
    runs = list(map(paths.run_name, SEEDS))
    ref_records, records = _records(dataset, ref_exp, [ref_run], split), _records(dataset, final, runs, split)
    reference = stats.load_run(dataset, ref_exp, ref_run, split)
    ours = tables.load_runs(dataset, (final,), split)[final]
    groups = {ALL_IMAGES: [int(i) for i in reference.per_image.index]}
    groups.update({name: ids for name, ids in strata_image_ids(dataset, split, splits_dir).items() if name in MAIN_STRATA})
    values = {
        group: {
            "n_images": len(ids),
            ref_exp: _group_values(ref_records, group, SOTA_METRICS),
            final: _group_values(records, group, SOTA_METRICS),
        }
        for group, ids in groups.items()
    }
    deltas = {
        group: {
            metric: stats.compare([reference] * len(ours), ours, metric, ids, n_rep, BOOTSTRAP_SEED)
            for metric in SOTA_DELTAS
        }
        for group, ids in groups.items()
    }
    decision = decide_sota(deltas[ALL_IMAGES]["AP"])
    payload = {
        "split": split,
        "n_images": len(groups[ALL_IMAGES]),
        "final": final,
        "reference": ref_exp,
        "seeds": list(SEEDS),
        "n_rep": n_rep,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "values": values,
        "deltas": deltas,
        "decision": decision,
        "efficiency": _efficiency(dataset, final),
        "sources": {
            ref_exp: tables._sources(dataset, ref_exp, [ref_run], split),
            final: tables._sources(dataset, final, runs, split),
        },
    }
    label = LABELS[final]

    def row(name: str, group: str, exp: str) -> list[str]:
        return [name, *(tables._mean_sd(metric, values[group][exp][metric]["summary"]) for metric in SOTA_METRICS)]

    main = [
        ["HiSup", "ya", *row("", ALL_IMAGES, ref_exp)[1:], "prediksi rilisan, dievaluasi ulang"],
        [f"{label} (kita)", "tidak", *row("", ALL_IMAGES, final)[1:], f"run kita, {len(runs)} seed"],
    ]
    per_stratum = [
        [group, str(values[group]["n_images"]), *row(name, group, exp)]
        for group in groups
        if group != ALL_IMAGES
        for name, exp in (("HiSup", ref_exp), (label, final))
    ]
    delta_rows = [
        [
            group,
            metric,
            cell(metric, result["delta"], True),
            tables._ci(metric, result["ci95"]),
            ", ".join(cell(metric, delta, True) for delta in result["delta_per_seed"]),
        ]
        for group, by_metric in deltas.items()
        for metric, result in by_metric.items()
    ]
    verdict = "didukung" if decision["supported"] else "tidak didukung"
    lines = [
        f"# SOTA: {dataset}/{split} ({payload['n_images']} tile; FULL final = {label}, {len(runs)} seed)",
        "",
        "HiSup dievaluasi dari file prediksi rilisannya dengan pipeline yang sama. HiSup dilatih dengan anotasi poligon dan",
        "dengan seluruh train v1.0; metode kita hanya memakai mask dan train yang sudah dideduplikasi.",
        "AP, IoU, dan C-IoU dalam persen; PoLiS dalam piksel; MTA dalam derajat. Baris kita: rata-rata ± SD antar-seed.",
        "",
        *tables._markdown_table(("Metode", "Anotasi poligon?", *SOTA_HEADER, "Sumber"), main),
        "",
        "## Per strata kepadatan",
        "",
        *tables._markdown_table(("Strata", "Tile", "Metode", *SOTA_HEADER), per_stratum),
        "",
        f"## Selisih ({label} − HiSup, rata-rata seed; CI 95% bootstrap per tile, {n_rep} replikasi)",
        "",
        *tables._markdown_table(("Tile", "Metrik", "Δ", "CI 95%", "Δ per seed"), delta_rows),
        "",
        f"## Keputusan ({decision['rule']})",
        "",
        f"- Selisih AP terhadap HiSup: {cell('AP', decision['delta'], True)} poin, CI 95% "
        f"{tables._ci('AP', decision['ci95'])}; batas {cell('AP', decision['limit'], True)} poin = "
        f"{tables._yes(decision['supported'])}",
        f'- **"Mendekati SOTA" {verdict}**',
    ]
    return payload, lines


def _missing_sota(dataset: str) -> list[str]:
    if dataset != "crowdai":
        return [f"the sota table is defined on crowdai/{SOTA_SPLIT} only"]
    missing = tables.missing_runs(dataset, tables.FULL_CANDIDATES, "val_select")
    if missing:
        return missing
    final = tables.full_final(dataset)[0]["decision"]["choice"]
    reference = paths.metrics_file(dataset, SOTA_REFERENCE[0], SOTA_REFERENCE[1], SOTA_SPLIT)
    return [*([] if reference.exists() else [str(reference)]), *tables.missing_runs(dataset, (final,), SOTA_SPLIT)]


# --- paper: numbers reported elsewhere


def paper(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """The rows of configs/paper_numbers.yaml for this dataset, as printed in the papers, each with its source."""
    block = PAPER_BLOCKS[dataset]
    entries = {key: entry[block] for key, entry in load_paper_numbers().items() if block in entry}
    columns = [c for c in PAPER_COLUMNS if any(c in entry["values"] for entry in entries.values())]
    rows = [
        [
            entry["method"],
            *(_printed(entry["values"].get(column)) for column in columns),
            f"{entry['source']['tables']}; hlm. {entry['source']['page']}",
        ]
        for entry in entries.values()
    ]
    notes = list(dict.fromkeys(" ".join(str(entry["split_note"]).split()) for entry in entries.values()))
    lines = [
        f"# Angka dilaporkan di paper: {dataset} ({block})",
        "",
        "Disalin dari PDF paper masing-masing (`configs/paper_numbers.yaml`), bukan dari pipeline kita. Tidak sebanding",
        "langsung dengan tabel kita kecuali split dan protokolnya sama (lihat catatan).",
        "",
        *tables._markdown_table(("Metode", *(PAPER_HEADER.get(c, c) for c in columns), "Sumber"), rows),
        "",
        "## Catatan split",
        "",
        *(f"- {note}" for note in notes),
    ]
    payload = {"block": block, "columns": columns, "rows": entries, "sources": [{"file": "configs/paper_numbers.yaml"}]}
    return payload, lines


def _missing_paper(dataset: str) -> list[str]:
    block = PAPER_BLOCKS[dataset]
    has_rows = any(block in entry for entry in load_paper_numbers().values())
    return [] if has_rows else [f"configs/paper_numbers.yaml has no {block} rows"]


# --- strata: every experiment per density stratum (N2, descriptive)


def strata(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Metrics of every evaluated experiment in each stratum of test, mean ± SD over its runs."""
    split, groups = "test", (*MAIN_STRATA, *SECONDARY_STRATA)
    rows, sizes = {}, {}
    for exp in LABELS:
        runs = _runs_of(dataset, exp, split)
        if not runs:
            continue
        records = _records(dataset, exp, runs, split)
        rows[exp] = {
            "runs": runs,
            "strata": {group: _group_values(records, group, STRATA_METRICS) for group in groups},
            "sources": tables._sources(dataset, exp, runs, split),
        }
        for group in groups:
            stratum = records[0]["per_stratum"].get(group, {})
            sizes.setdefault(group, {"n_images": stratum.get("n_images"), "n_gt": stratum.get("n_gt")})
    header = [f"{group} ({sizes[group]['n_images']} tile)" for group in groups]
    lines = [
        f"# Per strata kepadatan: {dataset}/{split} (rata-rata ± SD antar-seed)",
        "",
        "Strata dihitung dari GT dan dibekukan sebelum evaluasi (`splits/density_*.json`). Tiga kolom pertama membagi tile",
        "menurut jumlah bangunan; dua kolom terakhir menurut ada tidaknya bangunan yang bersentuhan.",
        "AP, C-IoU, dan laju dalam persen; PoLiS dalam piksel; MTA dalam derajat.",
    ]
    for metric in STRATA_SHOWN:
        table = [
            [LABELS[exp], *(tables._mean_sd(metric, row["strata"][group][metric]["summary"]) for group in groups)]
            for exp, row in rows.items()
        ]
        lines += ["", f"## {metric}", "", *tables._markdown_table(("Kode", *header), table)]
    return {"split": split, "sizes": sizes, "rows": rows}, lines


def _missing_strata(dataset: str) -> list[str]:
    if any(_runs_of(dataset, exp, "test") for exp in LABELS):
        return []
    return [str(paths.metrics_file(dataset, tables.FULL_CANDIDATES[0], paths.run_name(0), "test"))]


# --- n2: the effect of the claims per stratum, and dense against sparse


def larger_in_dense(result: dict) -> dict:
    """A lower-is-better metric of stats.compare_strata: the gain is larger in the dense stratum when the CI of
    Δdense − Δsparse lies below 0 (§8 point 4)."""
    ci = result["ci95"]
    below = ci is not None and ci[1] < 0
    return {"difference": result["difference"], "ci95": ci, "ci_below_zero": below, "passed": below}


def decide_n2(base_supported: bool | None, checks: dict) -> dict:
    """§8 point 4: "the effect is largest in dense areas" needs the claim itself and, for each of its main metrics, a CI
    of Δdense − Δsparse that excludes 0 on the side of the dense stratum. base_supported None: no claim was tested."""
    supported = bool(base_supported) and all(check["passed"] for check in checks.values())
    return {"rule": "EXPERIMENTS §8 point 4", "base_supported": base_supported, "checks": checks, "supported": supported}


def n2(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Δ of each comparison in every stratum of test, and Δdense − Δsparse with a stratified bootstrap."""
    split = "test"
    groups = strata_image_ids(dataset, split, splits_dir)
    base_tables = {"n1": tables.n1, "n3": tables.n3}
    comparisons, lines = {}, [
        f"# N2: efek per strata kepadatan, {dataset}/{split}",
        "",
        f"Δ = kanan − kiri, rata-rata seed, dengan CI 95% bootstrap per tile ({n_rep} replikasi). Semua metrik di sini lebih",
        "baik bila turun. Δpadat − Δjarang memakai bootstrap berstrata: tile disampel ulang di dalam tiap strata.",
    ]
    for spec in N2_COMPARISONS[dataset]:
        a, b, seeds = spec["a"], spec["b"], spec["seeds"]
        runs = tables.load_runs(dataset, (a, b), split, seeds)
        per_stratum = {
            group: {m: stats.compare(runs[a], runs[b], m, ids, n_rep, BOOTSTRAP_SEED) for m in spec["metrics"]}
            for group, ids in groups.items()
        }
        dense_sparse = {
            m: stats.compare_strata(
                runs[a], runs[b], m, groups[DENSE_STRATUM], groups[SPARSE_STRATUM], n_rep, BOOTSTRAP_SEED
            )
            for m in spec["metrics"]
        }
        base = None
        if spec["base"]:
            base = base_tables[spec["base"]](dataset, n_rep, splits_dir)[0]["decision"]["supported"]
        decision = decide_n2(base, {m: larger_in_dense(result) for m, result in dense_sparse.items()})
        name = f"{a}->{b}"
        comparisons[name] = {
            "name": spec["name"],
            "seeds": list(seeds),
            "metrics": list(spec["metrics"]),
            "n_images": {group: len(ids) for group, ids in groups.items()},
            "per_stratum": per_stratum,
            "dense_minus_sparse": dense_sparse,
            "decision": decision,
            "sources": {
                exp: tables._sources(dataset, exp, map(paths.run_name, seeds), split) for exp in (a, b)
            },
        }
        rows = [
            [
                group,
                str(len(groups[group])),
                *(
                    f"{cell(m, per_stratum[group][m]['delta'], True)} {tables._ci(m, per_stratum[group][m]['ci95'])}"
                    for m in spec["metrics"]
                ),
            ]
            for group in groups
        ]
        lines += [
            "",
            f"## {spec['name']}: {LABELS[a]} → {LABELS[b]} ({len(seeds)} seed)",
            "",
            *tables._markdown_table(("Strata", "Tile", *(f"Δ {m} [CI 95%]" for m in spec["metrics"])), rows),
            "",
        ]
        for m, result in dense_sparse.items():
            lines.append(
                f"- Δpadat − Δjarang, `{m}`: {cell(m, result['difference'], True)}, CI 95% {tables._ci(m, result['ci95'])}; "
                f"CI di bawah 0 = {tables._yes(decision['checks'][m]['ci_below_zero'])}"
            )
        if base is None:
            lines.append("- Deskriptif saja: perbandingan ini bukan klaim §8, dan hanya ada satu seed.")
        else:
            verdict = "didukung" if decision["supported"] else "tidak didukung"
            lines.append(f"- Klaim dasarnya ({spec['name']}, strata padat) didukung = {tables._yes(base)}")
            lines.append(f'- **"Efek {spec["name"]} terbesar di area padat" {verdict}** ({decision["rule"]})')
    payload = {"split": split, "n_rep": n_rep, "bootstrap_seed": BOOTSTRAP_SEED, "comparisons": comparisons}
    return payload, lines


def _missing_n2(dataset: str) -> list[str]:
    missing = []
    for spec in N2_COMPARISONS[dataset]:
        exps = (spec["a"], spec["b"], *(tables.REQUIRED[spec["base"]][0] if spec["base"] else ()))
        missing += tables.missing_runs(dataset, dict.fromkeys(exps), "test", spec["seeds"])
    return list(dict.fromkeys(missing))


# --- p2p: WHU against Pix2Poly's ground truth, next to the paper


def p2p(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Our WHU test numbers under Pix2Poly's rules against Pix2Poly's ground truth (and, for contrast, against the main
    one), next to the numbers the Pix2Poly paper reports. The comparison is approximate (docs/DATA.md §3.2)."""
    split = "test"
    ours, table = {}, []
    for exp in tables.FULL_CANDIDATES:
        for variant, gt_name in ((P2P_VARIANT, "p2p"), (None, "utama")):
            runs = _runs_of(dataset, exp, split, variant)
            if not runs:
                continue
            values = _group_values(_records(dataset, exp, runs, split, variant), ALL_IMAGES, P2P_METRICS)
            ours[f"{exp}/{gt_name}"] = {
                "runs": runs,
                "metrics": values,
                "sources": tables._sources(dataset, exp, runs, split, variant),
            }
            shown = [tables._mean_sd(m.removesuffix("_p2p"), values[m]["summary"]) for m in P2P_METRICS]
            table.append([f"{LABELS[exp]} (kita, {len(runs)} run)", gt_name, *shown, "run kita, test direkonstruksi"])
    block = PAPER_BLOCKS[dataset]
    reported = {key: entry[block] for key, entry in load_paper_numbers().items() if block in entry}
    for entry in reported.values():
        shown = [_printed(entry["values"].get(key)) for key in P2P_PAPER_KEYS]
        source = f"dilaporkan di paper: {entry['source']['tables']}; hlm. {entry['source']['page']}"
        table.append([entry["method"], "p2p (paper)", *shown, source])
    lines = [
        f"# WHU, aturan Pix2Poly: {dataset}/{split}",
        "",
        "IoU dan C-IoU dalam persen (hanya citra yang punya prediksi, aturan Pix2Poly); PoLiS dalam piksel; MTA dalam derajat.",
        "Baris GT `p2p` memakai GT cara Pix2Poly (connected component dari label raster), baris GT utama memakai bangunan",
        "resmi yang terpisah per bangunan.",
        "",
        "**Perbandingan dengan angka paper hanya kira-kira:** test kita direkonstruksi dari rilisan resmi (tile 1024² dipotong",
        "menjadi kuadran 512²), sehingga daftar tile-nya tidak sama persis dengan test Pix2Poly.",
        "",
        *tables._markdown_table(("Metode", "GT", *P2P_HEADER, "Sumber"), table),
    ]
    payload = {"split": split, "ours": ours, "paper": reported, "paper_file": [{"file": "configs/paper_numbers.yaml"}]}
    return payload, lines


def _missing_p2p(dataset: str) -> list[str]:
    if dataset != "whu":
        return ["the p2p table is defined on whu/test only"]
    if any(_runs_of(dataset, exp, "test", P2P_VARIANT) for exp in tables.FULL_CANDIDATES):
        return []
    return [str(paths.metrics_file(dataset, tables.FULL_CANDIDATES[0], paths.run_name(0), "test", P2P_VARIANT))]


# --- efficiency


def efficiency(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Parameters, network speed (batch 1, AMP) and post-processing time (one CPU thread) of the benchmarked models.

    Seeded experiments (FULL+Y) also carry the detector's time per image, which enters the full-pipeline FPS.
    """
    rows, table = {}, []
    for exp, label in LABELS.items():
        record = _efficiency(dataset, exp)
        if record is None:
            continue
        rows[exp] = record
        post = record["postproc"]
        detector = record.get("detector")
        detector_ms = 1000 * detector["seconds_per_image"] if detector else 0.0
        total_ms = record["network_ms"] + detector_ms + post["total_ms"]
        table.append(
            [
                label,
                f"{record['parameters'] / 1e6:.2f}".replace(".", ","),
                "×".join(str(n) for n in record["input"][2:]),
                f"{record['fps']:.1f}".replace(".", ","),
                f"{record['network_ms']:.2f}".replace(".", ","),
                f"{detector_ms:.2f}".replace(".", ",") if detector else "–",
                f"{post['total_ms']:.2f}".replace(".", ","),
                f"{1000 / total_ms:.1f}".replace(".", ","),
            ]
        )
    devices = list(dict.fromkeys(f"GPU {r['gpu']}; CPU {r['cpu']}" for r in rows.values()))
    lines = [
        f"# Efisiensi: {dataset}",
        "",
        "Jaringan: batch 1, AMP, 20 iterasi pemanasan lalu 200 iterasi terukur. Post-processing: satu thread CPU, rata-rata",
        "per citra (pemisahan instance dan poligonisasi). Detektor (hanya skenario +Y): YOLOv8n-OBB, waktu `predict` per citra",
        "di GPU dengan batch 32, termasuk membaca citra dari disk; parameternya tidak termasuk kolom Parameter.",
        "FPS alur penuh = 1000 / (ms jaringan + ms detektor + ms post-processing).",
        "",
        *tables._markdown_table(
            (
                "Kode",
                "Parameter (juta)",
                "Input",
                "FPS jaringan",
                "Jaringan (ms)",
                "Detektor (ms)",
                "Post-processing (ms)",
                "FPS alur penuh",
            ),
            table,
        ),
        "",
        *(f"- Perangkat: {device}" for device in devices),
    ]
    return {"rows": rows}, lines


def _missing_efficiency(dataset: str) -> list[str]:
    if any(_efficiency(dataset, exp) for exp in LABELS):
        return []
    return [str(paths.efficiency_file(dataset, tables.FULL_CANDIDATES[0], paths.run_name(0)))]


tables.BUILDERS.update({"sota": sota, "paper": paper, "strata": strata, "n2": n2, "p2p": p2p, "efficiency": efficiency})
tables.MISSING.update(
    {
        "sota": _missing_sota,
        "paper": _missing_paper,
        "strata": _missing_strata,
        "n2": _missing_n2,
        "p2p": _missing_p2p,
        "efficiency": _missing_efficiency,
    }
)
