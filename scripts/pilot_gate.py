"""Gate of a pilot seed on a split that is not test (docs/EXPERIMENTS.md §11, densepoly.eval.reports.pilot_gate).

    python scripts/pilot_gate.py --data crowdai --a b3hp --b b3h --seed 0 --split val_select

Compares b with a on the saved evaluations of that seed and writes $DENSEPOLY_OUT/tables/<dataset>/gate_<b>.json. The
last line printed is GATE_PASSED or GATE_FAILED; the exit code is 0 either way: a failed gate is a result, not an error.
"""

from __future__ import annotations

import argparse
import json

from densepoly import paths
from densepoly.eval import reports, tables
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--a", required=True, help="the experiment without the change")
    parser.add_argument("--b", required=True, help="the experiment with the change")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=[s for s in paths.SPLITS if s != "test"], default="val_select")
    parser.add_argument("--n-rep", type=int, default=tables.CI_REPLICATES, help="bootstrap replicates")
    args = parser.parse_args(argv)
    record = reports.pilot_gate(args.data, args.a, args.b, paths.run_name(args.seed), args.split, args.n_rep)
    record |= {"git": git_revision(), "created": timestamp()}
    target = paths.out_root() / "tables" / args.data / f"gate_{args.b}.json"
    write_json(target, record)
    shown = {
        name: {"delta": result["delta"], "ci95": result["ci95"]} for name, result in record["comparisons"].items()
    }
    print(json.dumps({"file": str(target), "n_images": record["n_images"], **shown}))
    print("GATE_PASSED" if record["passed"] else "GATE_FAILED")


if __name__ == "__main__":
    main()
