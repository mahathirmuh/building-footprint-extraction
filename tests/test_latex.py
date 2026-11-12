"""LaTeX tables of the paper (densepoly.eval.latex) on records whose numbers are known."""

import pytest

from densepoly.eval import latex


def test_numbers_are_scaled_like_the_report_with_a_decimal_point():
    assert latex.number("AP", 0.6324) == "63.24"
    assert latex.number("PoLiS_paired", 1.6351) == "1.635"
    assert latex.number("merge_rate", -0.0045, signed=True) == "$-0.45$"
    assert latex.number("MTA_inst", 0.94, signed=True) == "$+0.94$"
    assert latex.number("AP", None) == "--" and latex.number("AP", float("nan")) == "--"


def test_mean_sd_and_interval():
    assert latex.mean_sd("AP", {"mean": 0.6194, "sd": 0.0068}) == "61.94\\,{\\scriptsize$\\pm$0.68}"
    assert latex.mean_sd("AP", {"mean": 0.6354, "sd": None}) == "63.54"
    assert latex.mean_sd("AP", None) == "--"
    assert latex.interval("PoLiS_paired", [-0.0438, -0.0299]) == "{\\scriptsize$[-0.044,\\,-0.030]$}"
    assert latex.interval("AP", None) == "--"


def test_tabular_draws_its_rules():
    lines = latex.tabular("l c", ["A", "B"], [["x", "1"], ["y", "2"], ["z", "3"]], rules_after=(0,))
    assert lines[0] == "\\begin{tabular}{l c}" and lines[2] == "A & B \\\\"
    assert lines[4:7] == ["x & 1 \\\\", "\\midrule", "y & 2 \\\\"] and lines[-2:] == ["\\bottomrule", "\\end{tabular}"]


def compare(delta, low, high):
    return {"delta": delta, "ci95": [low, high], "delta_per_seed": [delta]}


def test_steps_table_names_each_step():
    record = {"steps": {"b5->b5p": {"PoLiS_paired": compare(-0.02, -0.027, -0.013), "AP": compare(0.004, 0.001, 0.006)}}}
    lines = latex.steps(record, ("PoLiS_paired", "AP"), ["dP", "dAP"], {"b5->b5p": "training"})
    row = lines[4]
    assert row.startswith("B5 $\\rightarrow$ B5$'$ (training) & $-0.020$ {\\scriptsize$[-0.027,\\,-0.013]$}")
    assert "$+0.40$ {\\scriptsize$[+0.10,\\,+0.60]$}" in row


def test_ablation_keeps_the_order_and_leaves_out_missing_rows():
    summary = {"mean": 0.5, "sd": 0.01}
    metrics = dict.fromkeys(latex.ABLATION, summary)
    record = {"rows": {"full_p0": {"runs": ["seed0", "seed1"], "metrics": metrics},
                       "b3": {"runs": ["seed0"], "metrics": {**metrics, "MTA": None}}}}  # fmt: skip
    lines = latex.ablation(record)
    body = [line for line in lines if line.startswith(("B", "FULL"))]
    assert [line.split(" & ")[0] for line in body] == ["B3", "FULL-P0"]
    assert body[0].split(" & ")[1] == "1" and "--" in body[0] and body[1].split(" & ")[1] == "2"
    assert lines.count("\\midrule") >= 2


def test_density_table_ends_with_dense_minus_sparse():
    step = {
        "metrics": ["merge_rate"],
        "n_images": {"jarang": 10, "sedang": 8, "padat": 7},
        "per_stratum": {g: {"merge_rate": compare(-0.001 * k, -0.002 * k, 0.0)} for k, g in enumerate(("jarang", "sedang", "padat"), 1)},
        "dense_minus_sparse": {"merge_rate": {"difference": -0.002, "ci95": [-0.004, -0.001]}},
    }
    lines = latex.density({"comparisons": {"full_p0->full_p2": step}})
    assert "sparse & 10 & $-0.10$" in lines[4] and lines[6].startswith("dense & 7 & $-0.30$")
    assert lines[-3].startswith("dense $-$ sparse &  & $-0.20$ {\\scriptsize$[-0.40,\\,-0.10]$}")


def test_reported_numbers_are_printed_as_in_the_paper():
    record = {"rows": {"hisup": {"method": "HiSup", "values": {"AP": 79.4, "PoLiS": 0.726}},
                       "ffl": {"method": "FFL, eps = 1.0", "values": {"AP": 61.3}}}}  # fmt: skip
    lines = latex.reported(record, ("AP", "PoLiS"), ["AP", "PoLiS"])
    assert "HiSup & 79.4 & 0.726 \\\\" in lines and "FFL, $\\epsilon=$1.0 & 61.3 & -- \\\\" in lines


def test_efficiency_counts_the_detector_in_the_pipeline_speed():
    base = {"parameters": 20_813_693, "fps": 125.0, "network_ms": 8.0, "postproc": {"total_ms": 12.0}}
    record = {"rows": {"full_p0": base, "full_y": {**base, "detector": {"seconds_per_image": 0.005}}}}
    lines = latex.efficiency(record)
    assert "FULL-P0 & 20.81 & 125.0 & 12.00 & -- & 50.0 \\\\" in lines
    assert "FULL+Y & 20.81 & 125.0 & 12.00 & 5.00 & 40.0 \\\\" in lines


def test_sota_and_whu_tables_name_the_ground_truth():
    value = {"summary": {"mean": 0.6, "sd": None}}
    record = {"reference": "hisup_release", "final": "full_p0",
              "values": {"semua": {"hisup_release": dict.fromkeys(latex.SOTA, value), "full_p0": dict.fromkeys(latex.SOTA, value)}},
              "deltas": {"semua": {m: compare(-0.04, -0.05, -0.03) for m in ("AP", "AP_boundary", "IoU", "CIoU", "PoLiS", "MTA")}}}  # fmt: skip
    lines = latex.sota(record)
    assert lines[4].startswith("HiSup (released predictions) & yes & 60.00") and "FULL-P0 (ours) & no" in lines[5]
    assert lines[7].startswith("$\\Delta$ (ours $-$ HiSup) &  & $-4.00$") and lines[7].count("&") == 9
    ours = {"full_p0/p2p": {"runs": ["seed0"], "metrics": {m: value for m in latex.P2P}}}
    paper = {"pix2poly": {"method": "Pix2Poly", "values": {"IoU": 89.15, "CIoU": 81.63}}}
    lines = latex.whu({"ours": ours, "paper": paper})
    assert lines[4].startswith("FULL-P0 (ours) & Pix2Poly-style & 60.00 & 60.00 & 0.60")
    assert "Pix2Poly$^\\dagger$ & Pix2Poly-style & 89.15 & 81.63 & -- & -- & -- \\\\" in lines
    with pytest.raises(KeyError):
        latex.whu({"ours": {"full_p0/p2p": {"runs": [], "metrics": {}}}, "paper": {}})
