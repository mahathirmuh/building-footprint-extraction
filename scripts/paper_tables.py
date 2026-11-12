"""LaTeX tables of the paper from the written result tables (PLAN.md Phase 10; densepoly.eval.latex).

    python scripts/tables.py --data crowdai && python scripts/tables.py --data whu
    python scripts/paper_tables.py                  # -> paper/tables/*.tex

Reads $DENSEPOLY_OUT/tables/<dataset>/<name>.json and writes one tabular per table into paper/tables/, each headed by a
comment naming its source record and the commit that wrote it. A table whose record does not exist is skipped with a
warning; --strict makes that an error. The paper only \\input{}s these files, so no result number is typed by hand.
"""

from __future__ import annotations

import argparse
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
    if missing and args.strict:
        sys.exit(f"{len(missing)} table records do not exist")


if __name__ == "__main__":
    main()
