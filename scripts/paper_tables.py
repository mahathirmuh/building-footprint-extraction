"""LaTeX tables of the paper from the written result tables (PLAN.md Phase 10; densepoly.eval.latex).

    python scripts/tables.py --data crowdai && python scripts/tables.py --data whu
    python scripts/paper_tables.py                  # -> paper/tables/*.tex

Reads $DENSEPOLY_OUT/tables/<dataset>/<name>.json and writes one tabular per table into paper/tables/, each headed by a
comment naming its source record and the commit that wrote it. A table whose record does not exist is skipped with a
warning; --strict makes that an error. The paper only \\input{}s these files, so no result number is typed by hand.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from densepoly import paths
from densepoly.eval import latex
from densepoly.utils.io import read_json, write_lines

N1_NAMES = {"b5->b5p": "training $\\theta$", "b5p->full_p0": "guiding by $\\theta$", "b5->full_p0": "total"}
PRIOR_NAMES = {"full_p0->full_p1": "MBI", "full_p1->full_p2": "gap map", "full_p0->full_p2": "both priors"}
COCO = ("AP", "AP50", "AP75", "AR", "AP_boundary", "IoU", "CIoU", "PoLiS", "MTA")
COCO_HEADER = ["AP", "AP$_{50}$", "AP$_{75}$", "AR", "AP$_b$", "IoU", "C-IoU", "PoLiS$\\downarrow$", "MTA$\\downarrow$"]

# file name -> (dataset, table record, function of the record)
TABLES = {
    "ablation": ("crowdai", "ablation", latex.ablation),
    "n1": ("crowdai", "n1", lambda r: latex.steps(r, ("PoLiS_paired", "MTA_inst", "AP"),
                                                  ["$\\Delta$PoLiS (px)", "$\\Delta$MTA ($^\\circ$)", "$\\Delta$AP"], N1_NAMES)),
    "priors": ("crowdai", "attribution", lambda r: latex.steps(r, ("merge_rate", "split_rate", "AP"),
                                                               ["$\\Delta$Merge", "$\\Delta$Split", "$\\Delta$AP"], PRIOR_NAMES)),
    "density": ("crowdai", "n2", latex.density),
    "sota": ("crowdai", "sota", latex.sota),
    "reported": ("crowdai", "paper", lambda r: latex.reported(r, COCO, COCO_HEADER)),
    "exploratory": ("crowdai", "exploratory", latex.exploratory),
    "exploratory_touching": ("crowdai", "exploratory", lambda r: latex.exploratory(r, "bersentuhan")),
    "efficiency": ("crowdai", "efficiency", latex.efficiency),
    "whu": ("whu", "p2p", latex.whu),
    "whu_density": ("whu", "n2", latex.density),
}  # fmt: skip


def _load(dataset: str, table: str) -> dict | None:
    source = paths.out_root() / "tables" / dataset / f"{table}.json"
    return read_json(source) if source.exists() else None


def numbers() -> dict[str, str]:
    """{macro name: value} of every number the prose of the paper quotes, read from the same records as the tables (and,
    for the few that no table holds, from the metric, run and split files named here). A value that cannot be read is
    left out, so that LaTeX stops on the undefined macro instead of printing a stale number."""
    out: dict[str, str] = {}

    def put(name: str, value: str | None) -> None:
        if value is not None and value != latex.DASH:
            out[name] = value.replace("$", "")

    def plain(result: dict | None, metric: str) -> tuple[str | None, str | None]:
        if result is None:
            return None, None
        low, high = (latex.number(metric, bound, signed=True) for bound in result["ci95"])
        return latex.number(metric, result["delta"], signed=True), f"[{low}, {high}]"

    ablation = _load("crowdai", "ablation")
    codes = {"b0": "Bzero", "b3": "Bthree", "b3m": "Bthreem", "b3my": "BthreemY", "b4": "Bfour", "b5": "Bfive",
             "full_p0": "Full", "full_p2": "FullPtwo", "full_y": "FullY"}  # fmt: skip
    metrics = {"AP": "Ap", "AP_boundary": "Apb", "CIoU": "Ciou", "MTA": "Mta", "PoLiS": "Polis", "merge_rate": "Merge",
               "N_ratio": "Nratio", "IoU": "Iou"}  # fmt: skip
    if ablation:
        for exp, code in codes.items():
            for metric, short in metrics.items():
                summary = ablation["rows"].get(exp, {}).get("metrics", {}).get(metric)
                if summary:
                    put(f"N{short}{code}", latex.number(metric, summary["mean"]))
    strata = _load("crowdai", "strata")
    if strata:
        for exp, code in codes.items():
            row = strata["rows"].get(exp)
            if row and row["strata"]["bersentuhan"]["merge_rate"]["summary"]:
                put(f"NMergeTouching{code}", latex.number("merge_rate", row["strata"]["bersentuhan"]["merge_rate"]["summary"]["mean"]))
        put("NTilesDense", str(strata["sizes"]["padat"]["n_images"]))
        put("NTilesTouching", str(strata["sizes"]["bersentuhan"]["n_images"]))

    def step(record: dict | None, key: str, name: str, metric: str, holder: str = "steps") -> None:
        result = record[holder][key][metric] if record and key in record.get(holder, {}) else None
        value, ci = plain(result, metric)
        put(f"N{name}", value)
        put(f"N{name}Ci", ci)

    n1, n3 = _load("crowdai", "n1"), _load("crowdai", "n3")
    for key, name in (("b5->full_p0", "NoneTotal"), ("b5->b5p", "NoneTrain"), ("b5p->full_p0", "NoneGuide")):
        for metric, short in (("PoLiS_paired", "Polis"), ("MTA_inst", "Mta"), ("AP", "Ap")):
            step(n1, key, f"{name}{short}", metric)
    for metric, short in (("merge_rate", "Merge"), ("AP", "Ap")):
        step(n3, "full_p0->full_p2", f"NthreeDelta{short}", metric)
    n2 = _load("crowdai", "n2")
    if n2:
        for key, metric, name in (("b5->full_p0", "PoLiS_paired", "DensePolis"), ("b5->full_p0", "MTA_inst", "DenseMta"),
                                  ("full_p0->full_p2", "merge_rate", "DenseMerge")):  # fmt: skip
            result = n2["comparisons"][key]["dense_minus_sparse"][metric]
            value, ci = plain({"delta": result["difference"], "ci95": result["ci95"]}, metric)
            put(f"N{name}", value)
            put(f"N{name}Ci", ci)
            for group, tag in (("jarang", "Sparse"), ("sedang", "Medium"), ("padat", "Dense")):
                put(f"N{name}{tag}", latex.number(metric, n2["comparisons"][key]["per_stratum"][group][metric]["delta"], signed=True))
    sota = _load("crowdai", "sota")
    if sota:
        values, deltas = sota["values"]["semua"], sota["deltas"]["semua"]
        for metric, short in (("AP", "Ap"), ("AP50", "Apfifty"), ("AP_boundary", "Apb"), ("CIoU", "Ciou")):
            put(f"NClean{short}Hisup", latex.number(metric, values[sota["reference"]][metric]["summary"]["mean"]))
            put(f"NClean{short}Full", latex.number(metric, values[sota["final"]][metric]["summary"]["mean"]))
        for metric, short in (("AP", "Ap"), ("AP_boundary", "Apb"), ("PoLiS", "Polis")):
            value, ci = plain(deltas[metric], metric)
            put(f"NCleanDelta{short}", value)
            put(f"NCleanDelta{short}Ci", ci)
        put("NTilesClean", str(sota["n_images"]))
    leaky = paths.metrics_file("crowdai", "hisup_release", "ref", "val_full")
    if leaky.exists():
        put("NLeakyApHisup", latex.number("AP", read_json(leaky)["overall"]["AP"]))
    exploratory = _load("crowdai", "exploratory")
    if exploratory:
        names = {"b3->b3m": "RegBinary", "b3m->full_p0": "BinaryToFull", "full_p0->full_y": "SeedFull",
                 "b3m->b3my": "SeedBinary", "b3my->full_y": "SeededPair"}  # fmt: skip
        for key, name in names.items():
            for group, tag in (("semua", ""), ("bersentuhan", "Touching")):
                for metric, short in (("AP", "Ap"), ("CIoU", "Ciou"), ("merge_rate", "Merge"), ("split_rate", "Split"),
                                      ("MTA_inst", "Mta"), ("PoLiS_paired", "Polis")):  # fmt: skip
                    value, ci = plain(exploratory["steps"][key]["groups"][group][metric], metric)
                    put(f"N{name}{tag}{short}", value)
                    put(f"N{name}{tag}{short}Ci", ci)
    efficiency = _load("crowdai", "efficiency")
    if efficiency:
        for exp, code in (("b3", "Bthree"), ("full_p0", "Full"), ("full_y", "FullY")):
            row = efficiency["rows"].get(exp)
            if row:
                detector = 1000 * ((row.get("detector") or {}).get("seconds_per_image") or 0.0)
                total = row["network_ms"] + row["postproc"]["total_ms"] + detector
                put(f"NFps{code}", f"{1000 / total:.1f}")
                put(f"NParams{code}", f"{row['parameters'] / 1e6:.2f}")
    whu, whu_rows, whu_strata = _load("whu", "p2p"), _load("whu", "ablation"), _load("whu", "strata")
    if whu and "full_p0/p2p" in whu["ours"]:
        for metric, short in (("IoU_p2p", "Iou"), ("CIoU_p2p", "Ciou")):
            put(f"NWhu{short}", latex.number(metric.removesuffix("_p2p"), whu["ours"]["full_p0/p2p"]["metrics"][metric]["summary"]["mean"]))
        for key, code in (("pix2poly", "Pixtopoly"), ("hisup_by_pix2poly", "Hisup")):
            if key in whu["paper"]:
                put(f"NWhuCiou{code}", str(whu["paper"][key]["values"]["CIoU"]))
    if whu_rows and "full_p0" in whu_rows["rows"]:
        for metric, short in (("AP", "Ap"), ("CIoU", "CiouMain"), ("merge_rate", "Merge")):
            put(f"NWhu{short}", latex.number(metric, whu_rows["rows"]["full_p0"]["metrics"][metric]["mean"]))
    if whu_strata and "full_p0" in whu_strata["rows"]:
        touching = whu_strata["rows"]["full_p0"]["strata"]["bersentuhan"]["merge_rate"]["summary"]
        put("NWhuMergeTouching", latex.number("merge_rate", touching["mean"]))
        put("NWhuPatches", f"{sum(whu_strata['sizes'][g]['n_images'] or 0 for g in ('jarang', 'sedang', 'padat')):,}")

    report = paths.splits_dir() / "dedup_report.json"
    if report.exists():
        dedup = read_json(report)
        for name, value in (("TrainImages", dedup["train"]["images"]), ("TrainClusters", dedup["train"]["clusters"]),
                            ("ValImages", dedup["val"]["images"]), ("TrainDedup", dedup["splits"]["train_dedup"]),
                            ("TestTiles", dedup["splits"]["test"]), ("TuneTiles", dedup["splits"]["val_tune_sub"]),
                            ("SelectTiles", dedup["splits"]["val_select"])):  # fmt: skip
            put(f"N{name}", f"{value:,}")
    inactive = total = 0
    for exp in ("full_p0", "full_p1", "full_p2"):
        for seed in (0, 1, 2):
            log = paths.run_dir("crowdai", exp, paths.run_name(seed)) / "val_log.csv"
            if log.exists():
                rows = [line.split(",") for line in log.read_text(encoding="utf-8").splitlines()[1:]]
                total += 1
                inactive += max(float(row[3]) for row in rows) < 0.01
    if total:
        put("NContactInactive", str(inactive))
        put("NContactRuns", str(total))
    detector = paths.yolo_dir("crowdai") / "runs" / "seed0" / "results.csv"
    if detector.exists():
        lines = detector.read_text(encoding="utf-8").splitlines()
        last = dict(zip((name.strip() for name in lines[0].split(",")), lines[-1].split(","), strict=True))
        put("NDetectorMap", f"{float(last['metrics/mAP50(B)']):.3f}")
        put("NDetectorRecall", f"{float(last['metrics/recall(B)']):.3f}")
    return out


FIGURES = ("crowdai_n2.pdf", "crowdai_qualitative.pdf", "crowdai_strata.pdf")  # of scripts/figures.py, English labels


def check(paper: Path, macros: dict[str, str]) -> list[str]:
    """What main.tex uses and the paper folder does not hold: number macros, table and figure files, citation keys."""
    text = (paper / "main.tex").read_text(encoding="utf-8")
    problems = [f"undefined number \\{name}" for name in sorted(set(re.findall(r"\\(N[A-Z][A-Za-z]+)", text)) - set(macros))]
    problems += [f"missing {name}.tex" for name in re.findall(r"\\input\{([^}]+)\}", text) if not (paper / f"{name}.tex").exists()]
    figures = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    problems += [f"missing {name}.pdf" for name in figures if not (paper / f"{name}.pdf").exists()]
    keys = set(re.findall(r"@\w+\{([^,]+),", (paper / "refs.bib").read_text(encoding="utf-8")))
    cited = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", text) for key in group.split(",")}
    return problems + [f"citation {key} is not in refs.bib" for key in sorted(cited - keys)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=paths.REPO_ROOT / "paper" / "tables")
    parser.add_argument("--strict", action="store_true", help="fail when a table record does not exist")
    args = parser.parse_args(argv)
    missing = []
    for name, (dataset, table, build) in TABLES.items():
        source = paths.out_root() / "tables" / dataset / f"{table}.json"
        if not source.exists():
            missing.append(str(source))
            print(f"skip {name}: {source} does not exist", file=sys.stderr)
            continue
        record = read_json(source)
        header = f"% {name}: from tables/{dataset}/{table}.json (git {str(record.get('git'))[:12]}, {record.get('created')}); do not edit"
        write_lines(args.out / f"{name}.tex", [header, *build(record)])
        print(args.out / f"{name}.tex")
    macros = numbers()
    header = "% numbers quoted in the prose, read from the same records as the tables (scripts/paper_tables.py); do not edit"
    write_lines(args.out / "numbers.tex", [header, *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in sorted(macros.items()))])
    print(args.out / "numbers.tex", f"({len(macros)} numbers)")
    paper = args.out.parent
    for name in FIGURES:
        source = paths.out_root() / "figures" / "results" / name
        if source.exists():
            (paper / "figures").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, paper / "figures" / name)
        else:
            print(f"skip figure {name}: run scripts/figures.py first", file=sys.stderr)
    if (paper / "main.tex").exists():
        problems = check(paper, macros)
        print("main.tex:", "every number, table, figure and citation it uses exists" if not problems else "; ".join(problems))
        if problems and args.strict:
            sys.exit(1)
    if missing and args.strict:
        sys.exit(f"{len(missing)} table records do not exist")


if __name__ == "__main__":
    main()
