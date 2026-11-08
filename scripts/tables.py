"""Result tables (docs/EXPERIMENTS.md §9) and the decisions of §8, from the saved evaluations (densepoly.eval.tables).

    python scripts/tables.py --data crowdai                      # every table whose runs are all there
    python scripts/tables.py --data crowdai --table full_final   # only this one; missing runs are then an error

Writes $DENSEPOLY_OUT/tables/<dataset>/<table>.md for reading and .json with the numbers, the bootstrap settings and
the metric files they came from.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from densepoly import paths
from densepoly.eval import tables


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--table", nargs="+", choices=sorted(tables.BUILDERS), help="default: all with complete runs")
    parser.add_argument("--n-rep", type=int, default=tables.CI_REPLICATES, help="bootstrap replicates")
    parser.add_argument("--splits-dir", type=Path, help="folder of the density files (default: repo splits/)")
    args = parser.parse_args(argv)
    for name in args.table or tables.BUILDERS:
        missing = tables.missing_for(args.data, name)
        if missing:
            if args.table:
                parser.error(f"{name} needs {len(missing)} evaluations that do not exist, e.g. {missing[0]}")
            print(f"skip {name}: {len(missing)} evaluations do not exist", file=sys.stderr)
            continue
        print(tables.write_table(args.data, name, args.n_rep, args.splits_dir))


if __name__ == "__main__":
    main()
