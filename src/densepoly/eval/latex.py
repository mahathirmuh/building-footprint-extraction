"""LaTeX tables of the paper, from the JSON records that scripts/tables.py wrote (PLAN.md Phase 10).

Every function takes the record of one table (outputs/tables/<dataset>/<name>.json) and returns the lines of a tabular
environment for paper/tables/. No number is typed by hand: scripts/paper_tables.py regenerates all of them. Numbers are
printed the English way (decimal point); AP, IoU, C-IoU and rates in percent, PoLiS in pixels, MTA in degrees.
"""

from __future__ import annotations

from densepoly.eval.tables import DECIMALS, LABELS, PERCENT

TEX_LABELS = {**LABELS, "b5p": "B5$'$", "hisup_release": "HiSup"}
STRATA = {
    "semua": "all",
    "jarang": "sparse",
    "sedang": "medium",
    "padat": "dense",
    "bersentuhan": "touching",
    "tidak_bersentuhan": "not touching",
}
DASH = "--"


def number(metric: str, value: float | None, signed: bool = False, digits: int | None = None) -> str:
    """A value of ours as the paper prints it; percent metrics are scaled, "--" stands for no value."""
    if value is None or value != value:
        return DASH
    shown = 100 * value if metric in PERCENT else value
    digits = DECIMALS.get(metric, 2) if digits is None else digits
    text = f"{shown:{'+' if signed else ''}.{digits}f}"
    return f"${text}$" if signed else text


def mean_sd(metric: str, summary: dict | None) -> str:
    if summary is None:
        return DASH
    text = number(metric, summary["mean"])
    return text if summary.get("sd") is None else f"{text}\\,{{\\scriptsize$\\pm${number(metric, summary['sd'])}}}"


def interval(metric: str, ci: list[float] | None) -> str:
    if ci is None:
        return DASH
    low, high = (number(metric, bound, signed=True).strip("$") for bound in ci)
    return f"{{\\scriptsize$[{low},\\,{high}]$}}"


def delta(metric: str, result: dict) -> str:
    """Δ with its 95 % CI, from a stats.compare record."""
    return f"{number(metric, result['delta'], signed=True)} {interval(metric, result['ci95'])}"


def tabular(spec: str, header: list[str], rows: list[list[str]], rules_after: tuple[int, ...] = ()) -> list[str]:
    """A booktabs tabular; rules_after lists the row indexes (0-based) after which a \\midrule is drawn."""
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for index, row in enumerate(rows):
        lines.append(" & ".join(row) + " \\\\")
        if index in rules_after:
            lines.append("\\midrule")
    return [*lines, "\\bottomrule", "\\end{tabular}"]


def printed(value) -> str:
    """A number of another paper, as printed there."""
    return DASH if value is None else str(value)


# --- the tables

ABLATION = ("AP", "AP50", "AP_boundary", "IoU", "PoLiS", "MTA", "CIoU", "N_ratio", "merge_rate")
ABLATION_HEADER = ["AP", "AP$_{50}$", "AP$_b$", "IoU", "PoLiS$\\downarrow$", "MTA$\\downarrow$", "C-IoU", "N-ratio", "Merge$\\downarrow$"]
ABLATION_GROUPS = (("b0",), ("b1", "b2", "b3"), ("b4", "b5", "b5p"), ("full_p0", "full_p1", "full_p2"), ("full_y", "b3m", "b3my"))


def ablation(record: dict) -> list[str]:
    rows, rules = [], []
    for group in ABLATION_GROUPS:
        for exp in group:
            if exp in record["rows"]:
                row = record["rows"][exp]
                rows.append([TEX_LABELS[exp], str(len(row["runs"])), *(mean_sd(m, row["metrics"][m]) for m in ABLATION)])
        rules.append(len(rows) - 1)
    return tabular("l c " + "c" * len(ABLATION), ["Code", "Runs", *ABLATION_HEADER], rows, tuple(rules[:-1]))


def steps(record: dict, metrics: tuple[str, ...], header: list[str], names: dict[str, str] | None = None) -> list[str]:
    """Paired differences of the comparison tables (n1, n3, attribution): one row per step."""
    rows = []
    for name, by_metric in record["steps"].items():
        a, b = name.split("->")
        label = f"{TEX_LABELS[a]} $\\rightarrow$ {TEX_LABELS[b]}"
        if names and name in names:
            label += f" ({names[name]})"
        rows.append([label, *(delta(m, by_metric[m]) for m in metrics)])
    return tabular("l " + "c" * len(metrics), ["Step", *header], rows)


def density(record: dict) -> list[str]:
    """Effect of each comparison of the n2 table per stratum, and dense minus sparse."""
    columns = [(name, metric) for name, spec in record["comparisons"].items() for metric in spec["metrics"]]
    first = next(iter(record["comparisons"].values()))
    rows = []
    for group in first["per_stratum"]:
        cells = [delta(m, record["comparisons"][name]["per_stratum"][group][m]) for name, m in columns]
        rows.append([STRATA[group], str(first["n_images"][group]), *cells])
    last = []
    for name, metric in columns:
        result = record["comparisons"][name]["dense_minus_sparse"][metric]
        last.append(f"{number(metric, result['difference'], signed=True)} {interval(metric, result['ci95'])}")
    rows.append(["dense $-$ sparse", "", *last])
    header = []
    for name, metric in columns:
        a, b = name.split("->")
        pretty = {"PoLiS_paired": "PoLiS", "MTA_inst": "MTA", "merge_rate": "merge rate"}[metric]
        header.append(f"$\\Delta$ {pretty}, {TEX_LABELS[a]}$\\rightarrow${TEX_LABELS[b]}")
    return tabular("l r " + "c" * len(columns), ["Tiles", "$n$", *header], rows, (2, len(rows) - 2))


SOTA = ("AP", "AP50", "AP_boundary", "IoU", "PoLiS", "MTA", "CIoU", "N_ratio")
SOTA_HEADER = ["AP", "AP$_{50}$", "AP$_b$", "IoU", "PoLiS$\\downarrow$", "MTA$\\downarrow$", "C-IoU", "N-ratio"]
SOTA_DELTA = {"AP": "AP", "AP_boundary": "AP_boundary", "IoU": "IoU", "PoLiS": "PoLiS", "MTA": "MTA", "CIoU": "CIoU"}


def sota(record: dict) -> list[str]:
    """HiSup and the FULL final on the clean val tiles, and the paired difference."""
    reference, final = record["reference"], record["final"]
    values, deltas = record["values"]["semua"], record["deltas"]["semua"]
    rows = [
        ["HiSup (released predictions)", "yes", *(mean_sd(m, values[reference][m]["summary"]) for m in SOTA)],
        [f"{TEX_LABELS[final]} (ours)", "no", *(mean_sd(m, values[final][m]["summary"]) for m in SOTA)],
        ["$\\Delta$ (ours $-$ HiSup)", "", *(delta(m, deltas[m]) if m in SOTA_DELTA and m in deltas else "" for m in SOTA)],
    ]
    return tabular("l c " + "c" * len(SOTA), ["Method", "Polygon labels", *SOTA_HEADER], rows, (1,))


def reported(record: dict, columns: tuple[str, ...], header: list[str]) -> list[str]:
    """Numbers reported by other papers (the paper table), with the table they were copied from."""
    rows = [
        [entry["method"].replace("eps = ", "$\\epsilon=$"), *(printed(entry["values"].get(c)) for c in columns)]
        for entry in record["rows"].values()
    ]
    return tabular("l " + "c" * len(columns), ["Method (as reported)", *header], rows)


P2P = ("IoU_p2p", "CIoU_p2p", "N_ratio_p2p", "PoLiS", "MTA")
P2P_PAPER = ("IoU", "CIoU", "N_ratio", "PoLiS", "MTA")


def whu(record: dict) -> list[str]:
    """Our WHU numbers under Pix2Poly's rules next to those reported in the Pix2Poly paper."""
    rows = []
    for key, row in record["ours"].items():
        exp, gt = key.split("/")
        truth = "Pix2Poly-style" if gt == "p2p" else "official buildings"
        shown = [mean_sd(m.removesuffix("_p2p"), row["metrics"][m]["summary"]) for m in P2P]
        rows.append([f"{TEX_LABELS[exp]} (ours)", truth, *shown])
    rules = (len(rows) - 1,)
    for entry in record["paper"].values():
        rows.append([entry["method"].replace("eps = ", "$\\epsilon=$") + "$^\\dagger$", "Pix2Poly-style", *(printed(entry["values"].get(k)) for k in P2P_PAPER)])
    return tabular("l l ccccc", ["Method", "Ground truth", "IoU", "C-IoU", "N-ratio", "PoLiS$\\downarrow$", "MTA$\\downarrow$"], rows, rules)


EXPLORATORY = ("AP", "CIoU", "merge_rate", "split_rate", "PoLiS_paired", "MTA_inst")
EXPLORATORY_HEADER = ["$\\Delta$AP", "$\\Delta$C-IoU", "$\\Delta$Merge", "$\\Delta$Split", "$\\Delta$PoLiS", "$\\Delta$MTA"]


def exploratory(record: dict, group: str = "semua") -> list[str]:
    rows = []
    for name, step in record["steps"].items():
        a, b = name.split("->")
        cells = [delta(m, step["groups"][group][m]) for m in EXPLORATORY]
        rows.append([f"{TEX_LABELS[a]} $\\rightarrow$ {TEX_LABELS[b]}", *cells])
    return tabular("l " + "c" * len(EXPLORATORY), ["Step", *EXPLORATORY_HEADER], rows)


def efficiency(record: dict) -> list[str]:
    rows = []
    for exp, row in record["rows"].items():
        detector = (row.get("detector") or {}).get("seconds_per_image")
        detector_ms = 1000 * detector if detector else 0.0
        total = row["network_ms"] + row["postproc"]["total_ms"] + detector_ms
        rows.append(
            [
                TEX_LABELS[exp],
                f"{row['parameters'] / 1e6:.2f}",
                f"{row['fps']:.1f}",
                f"{row['postproc']['total_ms']:.2f}",
                f"{detector_ms:.2f}" if detector else DASH,
                f"{1000 / total:.1f}",
            ]
        )
    header = ["Code", "Params (M)", "Network FPS", "Post-proc. (ms)", "Detector (ms)", "Pipeline FPS"]
    return tabular("l ccccc", header, rows)
