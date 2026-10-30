"""Sanity checks of the evaluation pipeline on real data (docs/EVALUATION.md §6.1, §6.2, §6.6; densepoly.eval.sanity).

    python scripts/sanity_checks.py --workers 12          # ±30–40 min on this laptop

Writes outputs/metrics/sanity_report.json and renders docs/sanity_report.md from it. The evaluations behind it are
ordinary run_eval outputs: experiments sanity_gt (GT as predictions), sanity_gt_noempty (without the empty masks),
sanity_repeat (§6.2, evaluated twice) and sanity_perturbed (§6.6), all with run ref. §6.4 reads the saved
evaluation of HiSup's release (scripts/evaluate.py --exp hisup_release --split val_full --released, ±1 h), which
this script does not rerun.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from densepoly import paths
from densepoly.eval import sanity
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp

CASES = (("crowdai", "val_clean", None), ("crowdai", "test", None), ("whu", "test", None), ("whu", "test", "p2p"))
EDGE_CASE_TESTS = (  # §6.2 on synthetic data
    "tests/test_pixel.py::test_edge_cases_of_docs_evaluation_6_2",
    "tests/test_run_eval.py::test_no_prediction_at_all",
    "tests/test_pix2poly_metrics.py::test_ciou_reset_changes_only_the_iou_of_an_image_without_gt",
    "tests/test_coco_io.py::test_each_problem_is_an_error_in_strict_mode",
    "tests/test_coco_io.py::test_lenient_mode_drops_only_rings_it_cannot_read",
    "tests/test_run_eval.py::test_evaluating_twice_gives_identical_numbers",
)


def edge_case_tests() -> list[dict]:
    """Run the §6.2 unit tests and record whether each passed."""
    results = []
    for test in EDGE_CASE_TESTS:
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test]
        code = subprocess.run(command, cwd=paths.REPO_ROOT, capture_output=True, text=True).returncode
        results.append({"test": test, "passed": code == 0})
    return results


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--case", nargs="+", help="cases of §6.1 to run, e.g. crowdai/val_clean whu/test/p2p")
    parser.add_argument("--n-rep", type=int, default=1000, help="bootstrap replicates of §6.6")
    parser.add_argument("--n-naive", type=int, default=200, help="replicates of the naive COCOeval comparison")
    parser.add_argument("--report-md", type=Path, default=paths.REPO_ROOT / "docs" / "sanity_report.md")
    args = parser.parse_args(argv)
    names = {"/".join(part for part in case if part): case for case in CASES}
    cases = [names[name] for name in args.case] if args.case else list(CASES)

    report = {"created": timestamp(), "git": git_revision(), "environment": sanity.environment()}
    report["gt_as_prediction"] = []
    for dataset, split, variant in cases:
        log(f"6.1 {dataset}/{split} {variant or ''}")
        report["gt_as_prediction"].append(sanity.gt_as_prediction(dataset, split, variant, args.workers))
    log("6.2 unit tests")
    report["edge_case_tests"] = edge_case_tests()
    log("6.2 evaluating twice")
    report["determinism"] = sanity.determinism(workers=args.workers)
    log("6.6 bootstrap AP")
    report["bootstrap_ap"] = sanity.bootstrap_ap(n_rep=args.n_rep, n_naive=args.n_naive, workers=args.workers)
    log("6.4 HiSup reproduction (from its saved evaluation)")
    report["hisup_reproduction"] = sanity.hisup_reproduction()

    write_json(paths.out_root() / "metrics" / "sanity_report.json", report)
    args.report_md.write_text(sanity.render_report(report), encoding="utf-8", newline="\n")
    checks = sanity.all_checks(report)
    tests_failed = sum(not t["passed"] for t in report["edge_case_tests"])
    log(f"done: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed, {tests_failed} unit test(s) failed")


if __name__ == "__main__":
    main()
