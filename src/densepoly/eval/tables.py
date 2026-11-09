"""Result tables of docs/EXPERIMENTS.md §9 and the decisions of §8, from the saved evaluations of scripts/evaluate.py.

The tables: ablation (test, mean ± SD over the seeds), N1 (B5 → B5′ → FULL-P0), N3 (FULL-P0 → FULL-P2) and the attribution
of the priors (FULL-P0 → FULL-P1 → FULL-P2) in the dense stratum of test, and the choice of the FULL final on val_select. The comparisons are those of
eval/stats.py (cluster bootstrap by image, Δ per seed and then averaged); no metric is recomputed from predictions. Every
table is written as Markdown for reading and as JSON with its numbers and the metric files behind them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from densepoly import paths
from densepoly.eval import stats
from densepoly.utils.io import read_json, write_json, write_lines
from densepoly.utils.provenance import git_revision, timestamp

SEEDS = (0, 1, 2)
DENSE_STRATUM = "padat"
CI_REPLICATES = 1000  # docs/EVALUATION.md §4
BOOTSTRAP_SEED = 0
AP_DROP_LIMIT = 0.005  # §8 points 1 and 2: AP may fall by at most 0.5 points (mean over the seeds, dense stratum)
FULL_TIE_MARGIN = 0.002  # §8 point 6: under 0.2 AP of difference, the variant with fewer input channels
FULL_CANDIDATES = ("full_p0", "full_p2")  # fewest input channels first

LABELS = {
    "b0": "B0",
    "b1": "B1",
    "b2": "B2",
    "b3": "B3",
    "b4": "B4",
    "b5": "B5",
    "b5p": "B5′",
    "full_p0": "FULL-P0",
    "full_p1": "FULL-P1",
    "full_p2": "FULL-P2",
}
ABLATION_METRICS = ("AP", "AP50", "AP_boundary", "IoU", "PoLiS", "MTA", "CIoU", "CIoU_p2p", "N_ratio", "merge_rate")
ABLATION_HEADER = (
    "AP",
    "AP50",
    "AP_b",
    "IoU",
    "PoLiS ↓",
    "MTA ↓",
    "C-IoU",
    "C-IoU_p2p",
    "N-ratio",
    "merge rate ↓",
)

N1_METHODS = ("b5", "b5p", "full_p0")
N1_STEPS = (("b5", "b5p"), ("b5p", "full_p0"), ("b5", "full_p0"))
N1_STEP_NOTES = {"b5->b5p": "efek pelatihan θ", "b5p->full_p0": "efek pemanduan θ", "b5->full_p0": "efek total"}
N1_METRICS = ("PoLiS_paired", "MTA_inst", "AP")
N3_METHODS = ("full_p0", "full_p2")
N3_STEPS = (("full_p0", "full_p2"),)
N3_METRICS = ("merge_rate", "split_rate", "recall_inst50", "AP")
ATTRIBUTION_METHODS = ("full_p0", "full_p1", "full_p2")
ATTRIBUTION_STEPS = (("full_p0", "full_p1"), ("full_p1", "full_p2"), ("full_p0", "full_p2"))
ATTRIBUTION_STEP_NOTES = {
    "full_p0->full_p1": "sumbangan MBI, deskriptif",
    "full_p1->full_p2": "sumbangan peta celah",
    "full_p0->full_p2": "prior gabungan (N3)",
}
THETA_STATS = ("theta_err_median", "theta_err_p90")

PERCENT = {
    "AP",
    "AP50",
    "AP_boundary",
    "IoU",
    "CIoU",
    "CIoU_p2p",
    "merge_rate",
    "split_rate",
    "recall_inst50",
}
DECIMALS = {"PoLiS": 3, "PoLiS_paired": 3}


def cell(metric: str, value: float | None, signed: bool = False) -> str:
    """A number as the report shows it: AP, IoU and rates in percent, decimal comma, "—" when there is none."""
    if value is None or value != value:
        return "—"
    shown = 100 * value if metric in PERCENT else value
    return f"{shown:{'+' if signed else ''}.{DECIMALS.get(metric, 2)}f}".replace(".", ",")


def summarize(values: Iterable[float | None]) -> dict | None:
    """Mean ± SD over the seeds; None when a seed has no value."""
    values = list(values)
    if not values or any(v is None or v != v for v in values):
        return None
    return stats.seed_summary(values)


def _mean_sd(metric: str, summary: dict | None) -> str:
    if summary is None:
        return "—"
    text = cell(metric, summary["mean"])
    return text if summary["sd"] is None else f"{text} ± {cell(metric, summary['sd'])}"


def _markdown_table(header: Iterable[str], rows: Iterable[Iterable[str]]) -> list[str]:
    header = list(header)
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _yes(flag: bool) -> str:
    return "ya" if flag else "tidak"


def _clean(value):
    """The same structure with NaN as None, so that the JSON is valid."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    return value


def missing_runs(dataset: str, exps: Iterable[str], split: str, seeds: Iterable[int] = SEEDS) -> list[str]:
    """The metric files a table needs and that do not exist."""
    files = (paths.metrics_file(dataset, exp, paths.run_name(seed), split) for exp in exps for seed in seeds)
    return [str(file) for file in files if not file.exists()]


def _sources(dataset: str, exp: str, runs: Iterable[str], split: str) -> list[dict]:
    """Which metric files a table read, with the commit that wrote each."""
    entries = []
    for run in runs:
        file = paths.metrics_file(dataset, exp, run, split)
        record = read_json(file)
        entries.append(
            {
                "file": file.relative_to(paths.out_root()).as_posix(),
                "git": record.get("git"),
                "created": record.get("created"),
            }
        )
    return entries


# --- decisions of docs/EXPERIMENTS.md §8


def improved(result: dict) -> dict:
    """A lower-is-better metric of stats.compare: the CI lies below 0 and Δ is negative in every seed (§8 points 1, 2)."""
    ci, per_seed = result["ci95"], result["delta_per_seed"]
    ci_below_zero = ci is not None and ci[1] < 0
    lower_in_every_seed = bool(per_seed) and all(delta < 0 for delta in per_seed)
    return {
        "ci_below_zero": ci_below_zero,
        "lower_in_every_seed": lower_in_every_seed,
        "passed": ci_below_zero and lower_in_every_seed,
    }


def ap_kept(result: dict) -> dict:
    """AP of stats.compare, seed-averaged Δ: it may fall by at most 0.5 points (§8 points 1, 2)."""
    return {"delta": result["delta"], "passed": result["delta"] >= -AP_DROP_LIMIT}


def _decision(rule: str, checks: dict) -> dict:
    return {"rule": rule, "checks": checks, "supported": all(check["passed"] for check in checks.values())}


def decide_n1(total: dict) -> dict:
    """§8 point 1, from the comparisons {metric: stats.compare result} of B5 → FULL-P0 in the dense stratum."""
    checks = {
        "PoLiS_paired": improved(total["PoLiS_paired"]),
        "MTA_inst": improved(total["MTA_inst"]),
        "AP": ap_kept(total["AP"]),
    }
    return _decision("EXPERIMENTS §8 point 1", checks)


def decide_n3(step: dict) -> dict:
    """§8 point 2, from the comparisons of FULL-P0 → FULL-P2 in the dense stratum."""
    return _decision("EXPERIMENTS §8 point 2", {"merge_rate": improved(step["merge_rate"]), "AP": ap_kept(step["AP"])})


def decide_attribution(steps: dict) -> dict:
    """§8 point 3: "the gap map helps" needs FULL-P2 to pass criterion 2 against FULL-P0 and also against FULL-P1."""
    checks = {}
    for name, title in (("full_p0->full_p2", "FULL-P0 → FULL-P2"), ("full_p1->full_p2", "FULL-P1 → FULL-P2")):
        checks[f"merge_rate ({title})"] = improved(steps[name]["merge_rate"])
        checks[f"AP ({title})"] = ap_kept(steps[name]["AP"])
    return _decision("EXPERIMENTS §8 point 3", checks)


def choose_full_final(ap_per_seed: dict[str, list[float]]) -> dict:
    """§8 point 6: the higher AP on val_select (mean over the seeds); under 0.2 AP of difference, FULL-P0."""
    fewer, more = FULL_CANDIDATES
    means = {exp: float(np.mean(ap_per_seed[exp])) for exp in FULL_CANDIDATES}
    difference = means[more] - means[fewer]
    if abs(difference) < FULL_TIE_MARGIN:
        choice, reason = fewer, "tie"
    else:
        choice, reason = (more if difference > 0 else fewer), "higher_ap"
    return {"rule": "EXPERIMENTS §8 point 6", "mean_ap": means, "difference": difference, "choice": choice, "reason": reason}


# --- numbers behind the tables


def dense_image_ids(dataset: str, split: str, splits_dir: Path | None = None) -> list[int]:
    """The images of the dense stratum, from the frozen density file."""
    density = read_json(paths.density_file(dataset, paths.cache_split(split), splits_dir))
    return [int(image_id) for image_id in density["main"]["strata"][DENSE_STRATUM]]


def load_runs(dataset: str, exps: Iterable[str], split: str, seeds: Iterable[int] = SEEDS) -> dict:
    """{exp: [RunTables per seed]}."""
    return {exp: [stats.load_run(dataset, exp, paths.run_name(seed), split) for seed in seeds] for exp in exps}


def matched_by_all(runs: list[stats.RunTables], metric: str) -> set[int]:
    """GT matched by every run; for MTA_inst also with a value in every run (as stats.common_gt)."""
    return set.intersection(*(stats.common_gt(run, run, metric) for run in runs))


def theta_errors(run: stats.RunTables, image_ids: list[int], gt_ids: set[int]) -> dict:
    """Median and 90th percentile of the orientation error (degrees) of the matched pairs of these images and GT."""
    pairs = run.pairs[run.pairs["image_id"].isin(image_ids) & run.pairs["gt_id"].isin(gt_ids)]
    errors = pairs["theta_err"].dropna().to_numpy(float)
    if not errors.size:
        return dict.fromkeys(THETA_STATS)
    return {"theta_err_median": float(np.median(errors)), "theta_err_p90": float(np.percentile(errors, 90))}


def method_values(runs: dict, metrics: Iterable[str], image_ids: Iterable[int], theta: bool = False) -> dict:
    """{method: {metric: per-seed values and their mean ± SD}} on image_ids. Paired metrics and θ errors use, per seed, the
    GT matched by every method of the table (§9)."""
    metrics = list(metrics)
    ids = list(dict.fromkeys(int(i) for i in image_ids))
    ones = np.ones((1, len(ids)), int)
    names = [*metrics, *(THETA_STATS if theta else ())]
    values = {exp: {name: [] for name in names} for exp in runs}
    for k in range(len(next(iter(runs.values())))):
        group = [seed_runs[k] for seed_runs in runs.values()]
        common = {metric: matched_by_all(group, metric) for metric in metrics if metric in stats.PAIRED_METRICS}
        matched = matched_by_all(group, "PoLiS_paired")
        for exp, seed_runs in runs.items():
            run = seed_runs[k]
            for metric in metrics:
                values[exp][metric].append(float(stats.metric_values(run, metric, ids, ones, common.get(metric))[0]))
            if theta:
                for name, value in theta_errors(run, ids, matched).items():
                    values[exp][name].append(value)
    return {
        exp: {name: {"per_seed": v, "summary": summarize(v)} for name, v in row.items()} for exp, row in values.items()
    }


def compare_methods(runs: dict, steps: Iterable, metrics: Iterable[str], image_ids, n_rep: int) -> dict:
    """{"a->b": {metric: stats.compare result}}, Δ = metric(b) − metric(a), the same resamples everywhere."""
    metrics = list(metrics)
    return {
        f"{a}->{b}": {metric: stats.compare(runs[a], runs[b], metric, image_ids, n_rep, BOOTSTRAP_SEED) for metric in metrics}
        for a, b in steps
    }


def _comparison(dataset, methods, steps, metrics, n_rep, splits_dir, theta=False) -> dict:
    image_ids = dense_image_ids(dataset, "test", splits_dir)
    runs = load_runs(dataset, methods, "test")
    return {
        "split": "test",
        "stratum": DENSE_STRATUM,
        "n_images": len(image_ids),
        "seeds": list(SEEDS),
        "n_rep": n_rep,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "methods": method_values(runs, metrics, image_ids, theta),
        "steps": compare_methods(runs, steps, metrics, image_ids, n_rep),
        "sources": {exp: _sources(dataset, exp, map(paths.run_name, SEEDS), "test") for exp in methods},
    }


# --- the tables: each builder returns (payload of the JSON, lines of the Markdown)


def _existing_runs(dataset: str, exp: str, split: str) -> list[str]:
    candidates = [paths.run_name(None), *map(paths.run_name, SEEDS)]
    return [run for run in candidates if paths.metrics_file(dataset, exp, run, split).exists()]


def ablation(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """Overall metrics on test of every experiment evaluated so far (§9); experiments without a run are left out."""
    split = "test"
    rows, table = {}, []
    for exp, label in LABELS.items():
        runs = _existing_runs(dataset, exp, split)
        if not runs:
            continue
        records = [read_json(paths.metrics_file(dataset, exp, run, split)) for run in runs]
        metrics = {metric: summarize(r["overall"].get(metric) for r in records) for metric in ABLATION_METRICS}
        rows[exp] = {"runs": runs, "metrics": metrics, "sources": _sources(dataset, exp, runs, split)}
        table.append([label, str(len(runs)), *(_mean_sd(metric, metrics[metric]) for metric in ABLATION_METRICS)])
    lines = [
        f"# Ablasi: {dataset}/{split} (rata-rata ± SD antar-seed)",
        "",
        "AP, IoU, C-IoU, dan merge rate dalam persen; PoLiS dalam piksel; MTA dalam derajat.",
        "",
        *_markdown_table(("Kode", "Run", *ABLATION_HEADER), table),
    ]
    return {"split": split, "rows": rows}, lines


def full_final(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """The choice of the FULL final on val_select (§8 point 6)."""
    split, runs = "val_select", list(map(paths.run_name, SEEDS))
    ap = {
        exp: [read_json(paths.metrics_file(dataset, exp, run, split))["overall"]["AP"] for run in runs]
        for exp in FULL_CANDIDATES
    }
    decision = choose_full_final(ap)
    payload = {
        "split": split,
        "ap_per_seed": ap,
        "decision": decision,
        "sources": {exp: _sources(dataset, exp, runs, split) for exp in FULL_CANDIDATES},
    }
    fewer, more = FULL_CANDIDATES
    if decision["reason"] == "tie":
        why = f"selisih AP kurang dari 0,2 poin, jadi dipilih varian dengan kanal input lebih sedikit ({LABELS[fewer]})"
    else:
        why = "AP rata-rata tertinggi, dengan selisih minimal 0,2 poin"
    rows = [
        [LABELS[exp], *(cell("AP", value) for value in ap[exp]), cell("AP", decision["mean_ap"][exp])]
        for exp in FULL_CANDIDATES
    ]
    lines = [
        f"# Pemilihan FULL final: {dataset}/{split} (§8 poin 6)",
        "",
        *_markdown_table(("Kode", "AP seed 0", "AP seed 1", "AP seed 2", "AP rata-rata"), rows),
        "",
        f"- Selisih {LABELS[more]} − {LABELS[fewer]}: {cell('AP', decision['difference'], signed=True)} poin AP",
        f"- **FULL final: {LABELS[decision['choice']]}** ({why})",
    ]
    return payload, lines


def _ci(metric: str, ci: list[float] | None) -> str:
    return "—" if ci is None else f"[{cell(metric, ci[0], True)}; {cell(metric, ci[1], True)}]"


def _step_rows(steps: dict, notes: dict | None = None) -> list[list[str]]:
    rows = []
    for name, by_metric in steps.items():
        a, b = name.split("->")
        step = f"{LABELS[a]} → {LABELS[b]}" + (f" ({notes[name]})" if notes else "")
        for metric, result in by_metric.items():
            per_seed = ", ".join(cell(metric, delta, signed=True) for delta in result["delta_per_seed"])
            rows.append([step, metric, cell(metric, result["delta"], True), _ci(metric, result["ci95"]), per_seed])
    return rows


def _check_line(metric: str, check: dict) -> str:
    if "ci_below_zero" in check:
        return (
            f"- `{metric}` lebih baik: CI 95% di bawah 0 = {_yes(check['ci_below_zero'])}, "
            f"turun di semua seed = {_yes(check['lower_in_every_seed'])}"
        )
    return f"- {metric} tidak turun lebih dari 0,5 poin: Δ {cell('AP', check['delta'], True)} poin = {_yes(check['passed'])}"


def _comparison_lines(title: str, payload: dict, header: tuple, names: tuple, notes: dict | None, claim: str) -> list[str]:
    rows = [
        [LABELS[exp], *(_mean_sd(name, row[name]["summary"]) for name in names)]
        for exp, row in payload["methods"].items()
    ]
    decision = payload["decision"]
    verdict = "didukung" if decision["supported"] else "tidak didukung"
    return [
        f"# {title}: {payload['stratum']}, {payload['split']} ({payload['n_images']} tile, {len(payload['seeds'])} seed)",
        "",
        "Rata-rata ± SD antar-seed. AP dan laju dalam persen; PoLiS dalam piksel; MTA dan galat θ dalam derajat.",
        "Metrik berpasangan dihitung pada GT yang ter-match oleh semua metode di tabel ini.",
        "",
        *_markdown_table(("Kode", *header), rows),
        "",
        f"## Selisih berpasangan (Δ = kanan − kiri, rata-rata seed; CI 95% bootstrap, {payload['n_rep']} replikasi)",
        "",
        "Di sini metrik berpasangan dihitung pada GT yang ter-match oleh kedua metode di baris itu.",
        "",
        *_markdown_table(("Langkah", "Metrik", "Δ", "CI 95%", "Δ per seed"), _step_rows(payload["steps"], notes)),
        "",
        f"## Keputusan ({decision['rule']})",
        "",
        *(_check_line(metric, check) for metric, check in decision["checks"].items()),
        f"- **{claim} {verdict}**",
    ]


def n1(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """B5 → B5′ → FULL-P0 in the dense stratum of test, with the decision of §8 point 1."""
    payload = _comparison(dataset, N1_METHODS, N1_STEPS, N1_METRICS, n_rep, splits_dir, theta=True)
    payload["decision"] = decide_n1(payload["steps"]["b5->full_p0"])
    header = ("PoLiS_paired ↓", "MTA_inst ↓", "galat θ median ↓", "galat θ p90 ↓", "AP")
    names = ("PoLiS_paired", "MTA_inst", "theta_err_median", "theta_err_p90", "AP")
    return payload, _comparison_lines("N1", payload, header, names, N1_STEP_NOTES, "N1")


def n3(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """FULL-P0 → FULL-P2 in the dense stratum of test, with the decision of §8 point 2."""
    payload = _comparison(dataset, N3_METHODS, N3_STEPS, N3_METRICS, n_rep, splits_dir)
    payload["decision"] = decide_n3(payload["steps"]["full_p0->full_p2"])
    header = ("merge rate ↓", "split rate ↓", "recall@0,5", "AP")
    return payload, _comparison_lines("N3", payload, header, N3_METRICS, None, "N3")


def attribution(dataset: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> tuple[dict, list[str]]:
    """FULL-P0 → FULL-P1 → FULL-P2 in the dense stratum of test, with the decision of §8 point 3."""
    payload = _comparison(dataset, ATTRIBUTION_METHODS, ATTRIBUTION_STEPS, N3_METRICS, n_rep, splits_dir)
    payload["decision"] = decide_attribution(payload["steps"])
    header = ("merge rate ↓", "split rate ↓", "recall@0,5", "AP")
    claim = 'Klaim "peta celah membantu"'
    return payload, _comparison_lines("Atribusi prior", payload, header, N3_METRICS, ATTRIBUTION_STEP_NOTES, claim)


BUILDERS = {"ablation": ablation, "full_final": full_final, "n1": n1, "n3": n3, "attribution": attribution}
REQUIRED = {
    "full_final": (FULL_CANDIDATES, "val_select"),
    "n1": (N1_METHODS, "test"),
    "n3": (N3_METHODS, "test"),
    "attribution": (ATTRIBUTION_METHODS, "test"),
}


def missing_for(dataset: str, name: str) -> list[str]:
    """The metric files table `name` needs and that do not exist ([] for the ablation table, which shows what is there)."""
    return missing_runs(dataset, *REQUIRED[name]) if name in REQUIRED else []


def write_table(dataset: str, name: str, n_rep: int = CI_REPLICATES, splits_dir: Path | None = None) -> Path:
    """Build one table; writes outputs/tables/<dataset>/<name>.json and .md and returns the path of the Markdown."""
    payload, lines = BUILDERS[name](dataset, n_rep, splits_dir)
    folder = paths.out_root() / "tables" / dataset
    record = {"dataset": dataset, "table": name, "created": timestamp(), "git": git_revision(), **payload}
    write_json(folder / f"{name}.json", _clean(record))
    write_lines(folder / f"{name}.md", lines)
    return folder / f"{name}.md"
