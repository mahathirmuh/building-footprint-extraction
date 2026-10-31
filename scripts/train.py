"""Train one experiment (docs/EXPERIMENTS.md §3, docs/COLAB.md; densepoly.train).

    python scripts/train.py --data crowdai --exp full_p2 --benchmark 200                # it/s, memory, loss terms
    python scripts/train.py --data crowdai --exp b4 --seed 0 --resume --max-hours 11 train.epochs=40
    python scripts/train.py --data crowdai --exp b4 --seed 0 --run pilot_r5 train.epochs=10 labels.contact_radius_px=5

Writes outputs/runs/<data>/<exp>/<run>/ (run seed<k>, or --run for a pilot): config.yaml, run.json, train_log.csv,
val_log.csv, last.pt, best.pt. --resume continues from last.pt when it exists and starts the run otherwise, so the same
command fits every Colab session. --max-hours stops cleanly before the session limit, with last.pt saved. --benchmark N
measures N iterations and writes benchmark.json next to the runs of the experiment, without training a run. Config
overrides are key=value; E (train.epochs) has no default, because it is fixed after the benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from densepoly import paths, train
from densepoly.config import load_config
from densepoly.utils.io import write_json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--exp", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", help="run folder (default: seed<k>), e.g. pilot_r5")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt when it exists")
    parser.add_argument("--max-hours", type=float, help="stop cleanly after this many hours")
    parser.add_argument("--stop-after", type=int, metavar="STEPS", help="stop cleanly after this many optimizer steps")
    parser.add_argument("--benchmark", type=int, metavar="N", help="measure N iterations instead of training")
    parser.add_argument("--configs-dir", type=Path, help="config folder (default: repo configs/)")
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    parser.add_argument("overrides", nargs="*", help="config overrides key=value")
    args = parser.parse_args(argv)

    cfg = load_config(args.data, args.exp, [*args.overrides, f"seed={args.seed}"], root=args.configs_dir, tuned=False)
    if not cfg.train.enabled:
        parser.error(f"{args.exp} has no training: it uses the checkpoint of {cfg.model.source_exp}")
    if args.benchmark:
        report = train.benchmark(cfg, args.benchmark, args.splits_dir)
        write_json(paths.run_dir(args.data, args.exp, "benchmark").with_suffix(".json"), report)
        print(json.dumps(report))
        return
    run = args.run or paths.run_name(args.seed)
    outcome = train.fit(
        cfg, paths.run_dir(args.data, args.exp, run), args.resume, args.max_hours, args.stop_after, args.splits_dir
    )
    print(json.dumps(outcome))


if __name__ == "__main__":
    main()
