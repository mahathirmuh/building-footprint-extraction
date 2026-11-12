"""Rebuild every table and figure of the report with one command (PLAN.md Phase 10).

    python scripts/build_report.py                 # tables of both datasets, figures, paper tables, slide numbers
    python scripts/build_report.py --skip-tables   # figures and paper tables from the tables already written

Runs, in this order: scripts/tables.py for crowdai and whu (every table whose evaluations exist; the bootstrap of the
CrowdAI tables takes about an hour of CPU), scripts/figures.py --results and --qualitative with English and Indonesian
labels, and scripts/paper_tables.py, which writes paper/tables, paper/figures and slides/numbers.json and checks that
paper/main.tex only uses numbers, tables, figures and citations that exist. Nothing is evaluated again: everything is
read from $DENSEPOLY_OUT/metrics and $DENSEPOLY_OUT/preds.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DATASETS = ("crowdai", "whu")
QUALITATIVE = ("crowdai",)  # the qualitative figure needs the saved predictions of several models


def run(script: str, *args: str) -> None:
    print("$ python", f"scripts/{script}", *args, flush=True)
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-tables", action="store_true", help="keep the result tables that are already written")
    args = parser.parse_args(argv)
    if not args.skip_tables:
        for dataset in DATASETS:
            run("tables.py", "--data", dataset)
    for dataset in DATASETS:
        for lang in ("en", "id"):
            extra = ["--qualitative"] if dataset in QUALITATIVE else []
            run("figures.py", "--data", dataset, "--results", *extra, "--lang", lang)
    run("paper_tables.py", "--strict")


if __name__ == "__main__":
    main()
