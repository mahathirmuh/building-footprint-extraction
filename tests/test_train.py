"""Training on fake tiles with encoder_weights=None (densepoly.train, scripts/train.py; docs/COLAB.md §3)."""

import csv
import importlib.util
import json

import pytest
import torch
from torch import nn

from densepoly import paths, train
from densepoly.config import load_config
from densepoly.utils.io import read_json
from tests.fixtures.fake_tiles import write_tiles

SMALL = ["model.encoder_weights=null", "model.encoder=efficientnet-b0", "data.tile_size=120", "data.pad_to=128",
         "train.batch_size=2", "train.min_micro_batch=2", "train.num_workers=0", "train.epochs=2",
         "train.log_every_steps=1"]  # fmt: skip
# Largest gap in the loss curve of a resumed run. Measured 31 Oct 2026: an intact resume stays under 9e-4, and each
# resume broken on purpose (random states, Adam, BatchNorm, the skip of the sampler) reached 3.7e-3 or more.
RESUME_NOISE = 2e-3


@pytest.fixture
def splits(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    folder = tmp_path / "splits"
    folder.mkdir()
    write_tiles(folder, {"train_dedup": 4, "val_select": 2})
    return folder


def config(exp="b4", *overrides):
    return load_config("crowdai", exp, [*SMALL, *overrides])


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def losses(run_dir):
    return [float(row["loss"]) for row in rows(run_dir / "train_log.csv")]


def test_learning_rate_warms_up_linearly_then_follows_a_cosine():
    lrs = [train.learning_rate(step, 100, 10, 2e-4, 1e-6) for step in range(100)]
    assert lrs[0] == pytest.approx(2e-5) and lrs[9] == pytest.approx(2e-4) and lrs[10] == pytest.approx(2e-4)
    assert all(a <= b for a, b in zip(lrs[:10], lrs[1:10], strict=False))
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))
    assert lrs[55] == pytest.approx((2e-4 + 1e-6) / 2)
    assert lrs[-1] == pytest.approx(1e-6, abs=1e-7)


def test_micro_batch_follows_accumulation_and_has_a_lower_bound():
    assert train.micro_batch(load_config("crowdai", "b4")) == 16
    assert train.micro_batch(load_config("crowdai", "b4", ["train.accumulate=2"])) == 8
    with pytest.raises(ValueError, match="min_micro_batch"):
        train.micro_batch(load_config("crowdai", "b4", ["train.accumulate=4"]))
    with pytest.raises(ValueError, match="multiple"):
        train.micro_batch(load_config("crowdai", "b4", ["train.accumulate=3"]))


@pytest.mark.parametrize("exp", ["b2", "b4", "full_p2"])
def test_smoke_a_short_run_writes_everything(splits, tmp_path, exp):
    """The smoke test of PLAN.md Phase 5, for one binary, one three-class and the fullest model."""
    run_dir = tmp_path / "run"
    outcome = train.fit(config(exp), run_dir, splits_dir=splits)
    assert outcome == {"status": "finished", "epoch": 2, "step": 4, "best": outcome["best"]}
    assert {p.name for p in run_dir.iterdir()} == {"config.yaml", "run.json", "train_log.csv", "val_log.csv", "last.pt",
                                                  "best.pt"}  # fmt: skip
    log = rows(run_dir / "train_log.csv")
    assert [int(row["step"]) for row in log] == [1, 2, 3, 4]
    assert all(float(row["loss"]) > 0 for row in log)
    assert (float(log[0]["theta"]) > 0) == (exp == "full_p2")
    validation = rows(run_dir / "val_log.csv")
    assert [int(row["epoch"]) for row in validation] == [0, 1]
    assert float(validation[0]["fg_iou"]) >= 0 and (validation[0]["contact_f1"] == "") == (exp == "b2")
    best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    assert best["fg_iou"] == pytest.approx(outcome["best"]) and "optimizer" not in best
    assert read_json(run_dir / "run.json")["sessions"][0]["start_step"] == 0


@pytest.fixture(scope="module")
def uncut(tmp_path_factory):
    """Eight fake tiles (four steps per epoch) and a run of them that was never cut: (splits folder, run folder)."""
    root = tmp_path_factory.mktemp("uncut")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DENSEPOLY_DATA", str(root / "data"))
        patch.setenv("DENSEPOLY_OUT", str(root / "outputs"))
        folder = root / "splits"
        folder.mkdir()
        write_tiles(folder, {"train_dedup": 8, "val_select": 2})
        train.fit(config(), root / "straight", splits_dir=folder)
        yield folder, root / "straight"


def largest_gap(run_a, run_b):
    return max(abs(a - b) for a, b in zip(losses(run_a), losses(run_b), strict=True))


@pytest.mark.parametrize("stop_after", [1, 4, 6], ids=["inside_epoch_0", "end_of_epoch_0", "inside_epoch_1"])
def test_a_resumed_run_follows_the_run_that_was_cut(uncut, tmp_path, stop_after):
    """Same data, stochastic depth, BatchNorm and Adam: the loss curve goes on. Bit equality cannot be asked for: on CPU
    two uncut runs of one seed already differ by rounding, which Adam enlarges on parameters of almost no gradient
    (RESUME_NOISE)."""
    folder, straight = uncut
    cfg, cut = config(), tmp_path / "cut"
    outcome = train.fit(cfg, cut, stop_after=stop_after, splits_dir=folder)
    assert outcome["status"] == "stopped" and outcome["step"] == stop_after
    assert train.fit(cfg, cut, resume=True, splits_dir=folder)["status"] == "finished"
    assert largest_gap(straight, cut) < RESUME_NOISE
    assert [row["step"] for row in rows(cut / "train_log.csv")] == [str(step) for step in range(1, 9)]
    assert [row["epoch"] for row in rows(cut / "val_log.csv")] == ["0", "1"]
    assert [s["start_step"] for s in read_json(cut / "run.json")["sessions"]] == [0, stop_after]


def skip_rng(_states):
    """A resume that forgets the random states."""


def skip_optimizer(path, parts):
    """A resume that forgets the state of Adam."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    parts.model.load_state_dict(state["model"])
    parts.scaler.load_state_dict(state["scaler"])
    train.restore_rng(state["rng"])
    return train.Position(**state["position"])


@pytest.mark.parametrize(("name", "broken"), [("restore_rng", skip_rng), ("load_last", skip_optimizer)])
def test_the_loss_curve_shows_a_broken_resume(uncut, tmp_path, monkeypatch, name, broken):
    """The check above has teeth: a resume that loses part of its state leaves RESUME_NOISE behind."""
    folder, straight = uncut
    cfg, cut = config(), tmp_path / "cut"
    train.fit(cfg, cut, stop_after=2, splits_dir=folder)
    monkeypatch.setattr(train, name, broken)
    train.fit(cfg, cut, resume=True, splits_dir=folder)
    assert largest_gap(straight, cut) > RESUME_NOISE


def test_a_log_row_averages_the_steps_of_its_window(uncut, tmp_path):
    """A row at the start of an epoch has a shorter window: the steps since the epoch began."""
    folder, straight = uncut
    train.fit(config("b4", "train.log_every_steps=3"), tmp_path / "run", splits_dir=folder)
    every_step, log = losses(straight), rows(tmp_path / "run" / "train_log.csv")
    assert [int(row["step"]) for row in log] == [3, 6]
    assert float(log[0]["loss"]) == pytest.approx(sum(every_step[:3]) / 3, abs=RESUME_NOISE)
    assert float(log[1]["loss"]) == pytest.approx(sum(every_step[4:6]) / 2, abs=RESUME_NOISE)


def test_loader_workers_give_the_batches_of_the_main_process(uncut, tmp_path):
    folder, straight = uncut
    train.fit(config("b4", "train.num_workers=2"), tmp_path / "run", stop_after=2, splits_dir=folder)
    assert losses(tmp_path / "run") == pytest.approx(losses(straight)[:2], abs=RESUME_NOISE)


def test_a_cut_after_the_last_step_of_the_checkpoint_drops_later_log_rows(splits, tmp_path):
    run_dir = tmp_path / "run"
    train.fit(config(), run_dir, stop_after=1, splits_dir=splits)
    with (run_dir / "train_log.csv").open("a", encoding="utf-8", newline="") as f:
        f.write("2,0,0,9,9,9,0,0,0\n")  # a row written after last.pt, before the runtime was cut
    train.fit(config(), run_dir, resume=True, splits_dir=splits)
    assert [row["step"] for row in rows(run_dir / "train_log.csv")] == ["1", "2", "3", "4"]


def test_the_time_limit_stops_cleanly_with_a_checkpoint(splits, tmp_path):
    run_dir = tmp_path / "run"
    outcome = train.fit(config(), run_dir, max_hours=0.0, splits_dir=splits)
    assert outcome["status"] == "stopped" and outcome["step"] == 1
    assert torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)["position"]["step"] == 1


def test_a_run_is_not_overwritten_and_not_resumed_with_another_config(splits, tmp_path):
    run_dir = tmp_path / "run"
    train.fit(config(), run_dir, stop_after=1, splits_dir=splits)
    with pytest.raises(FileExistsError):
        train.fit(config(), run_dir, splits_dir=splits)
    with pytest.raises(ValueError, match="config differs"):
        train.fit(config("b4", "train.lr=0.1"), run_dir, resume=True, splits_dir=splits)


def test_resume_without_a_checkpoint_starts_the_run(splits, tmp_path):
    outcome = train.fit(config(), tmp_path / "run", resume=True, stop_after=1, splits_dir=splits)
    assert outcome["step"] == 1


def test_epochs_must_be_set(splits, tmp_path):
    with pytest.raises(ValueError, match="train.epochs"):
        train.fit(load_config("crowdai", "b4", ["model.encoder_weights=null"]), tmp_path / "run", splits_dir=splits)


def test_accumulation_makes_one_step_of_several_micro_batches(splits, tmp_path):
    cfg = config("b4", "train.batch_size=4", "train.accumulate=2")
    outcome = train.fit(cfg, tmp_path / "run", splits_dir=splits)
    assert outcome["step"] == 2  # 4 tiles, one step per epoch


def test_saving_is_atomic(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    train.save_atomically({"value": 1}, path)

    def broken_save(state, target):
        target.write_bytes(b"half a file")
        raise OSError("runtime cut")

    monkeypatch.setattr(train.torch, "save", broken_save)
    with pytest.raises(OSError):
        train.save_atomically({"value": 2}, path)
    assert torch.load(path, weights_only=True) == {"value": 1}


class Fixed(nn.Module):
    """A model whose output is given in advance."""

    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, _image):
        return self.output


def one_batch(classes):
    return [{"image": torch.zeros(1, 3, *classes.shape[1:]), "classes": classes}]


def three_class_output(classes):
    return torch.nn.functional.one_hot(classes.clamp(max=2), 3).permute(0, 3, 1, 2).float() * 20


def test_validate_scores_a_perfect_output_as_one_and_leaves_padding_out():
    classes = torch.tensor([[[0, 1, 1, 2], [0, 0, 255, 255]]])
    output = three_class_output(classes)
    output[:, :, classes[0] == 255] = torch.tensor([0.0, 20.0, 0.0])[None, :, None]  # padding predicted as building
    scores = train.validate(Fixed(output), one_batch(classes), torch.device("cpu"), False, 255)
    assert scores == {"fg_iou": 1.0, "contact_f1": 1.0, "contact_precision": 1.0, "contact_recall": 1.0}


def test_validate_counts_the_contact_class_and_the_foreground():
    classes = torch.tensor([[[0, 1, 2, 2]]])
    predicted = torch.tensor([[[0, 0, 2, 1]]])  # one building pixel lost, one contact pixel found, one missed
    scores = train.validate(Fixed(three_class_output(predicted)), one_batch(classes), torch.device("cpu"), False, 255)
    assert scores["fg_iou"] == pytest.approx(2 / 3)
    assert (scores["contact_precision"], scores["contact_recall"]) == (1.0, 0.5)
    assert scores["contact_f1"] == pytest.approx(2 / 3)


def test_validate_has_no_contact_on_binary_models():
    classes = torch.tensor([[[0, 1, 2, 255]]])
    output = torch.tensor([[[[-9.0, 9.0, -9.0, 9.0]]]])
    scores = train.validate(Fixed(output), one_batch(classes), torch.device("cpu"), False, 255)
    assert scores["fg_iou"] == pytest.approx(1 / 2) and scores["contact_f1"] is None


def load_script():
    spec = importlib.util.spec_from_file_location("train_script", paths.REPO_ROOT / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_trains_a_named_run_and_benchmarks(splits, capsys):
    script = load_script()
    base = ["--data", "crowdai", "--exp", "full_p2", "--splits-dir", str(splits)]
    script.main(["--data", "crowdai", "--exp", "b4", "--run", "pilot_r5", "--stop-after", "1",
                 "--splits-dir", str(splits), *SMALL, "labels.contact_radius_px=5"])  # fmt: skip
    run_dir = paths.run_dir("crowdai", "b4", "pilot_r5")
    assert (run_dir / "last.pt").exists() and "contact_radius_px: 5" in (run_dir / "config.yaml").read_text("utf-8")
    capsys.readouterr()
    script.main([*base, "--benchmark", "1", *SMALL, "train.batch_size=4"])
    report = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert report["steps_per_second"] > 0 and 0 <= report["data_fraction"] <= 1
    assert report["loss_terms"]["theta"] > 0 and report["micro_batch"] == 4
    assert read_json(paths.run_dir("crowdai", "full_p2", "benchmark").with_suffix(".json"))["iterations"] == 1


def test_the_script_refuses_an_experiment_without_training(splits):
    with pytest.raises(SystemExit):
        load_script().main(["--data", "crowdai", "--exp", "b5", "--splits-dir", str(splits), *SMALL])
