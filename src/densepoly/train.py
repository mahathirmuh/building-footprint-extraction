"""Training of one experiment (docs/EXPERIMENTS.md §3, docs/COLAB.md §3-§4; scripts/train.py).

AdamW with a linear warmup and a cosine decay per optimizer step, AMP float16 with the loss in float32, a fixed budget
of E epochs and no early stopping. After every epoch the foreground IoU on val_select picks best.pt.

last.pt is written atomically every epoch and every 20 minutes, with everything a run needs to continue: model,
optimizer, GradScaler, position and random states. The learning rate is a pure function of the step and the
augmentation a pure function of (seed, epoch, index), so a resumed run follows the run that was cut, step by step, up
to rounding: two uncut runs of one seed already differ in their last bits. The loaders have their own generator,
because creating an iterator draws from the global random state, which also drives the stochastic depth of EfficientNet.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from densepoly.config import save_config
from densepoly.data.dataset import EpochSampler, TileDataset
from densepoly.losses import SegmentationLoss
from densepoly.models.factory import build_model, split_output
from densepoly.utils.io import read_json, write_json
from densepoly.utils.provenance import git_revision, timestamp

TERMS = ("loss", "ce", "dice", "theta")
TRAIN_COLUMNS = ("step", "epoch", "lr", *TERMS, "data_fraction", "seconds")
VAL_COLUMNS = ("epoch", "step", "fg_iou", "contact_f1", "contact_precision", "contact_recall", "best", "seconds")
BENCHMARK_WARMUP = 10  # first iterations left out of the timing: workers start, cudnn.benchmark tries its algorithms


def micro_batch(cfg) -> int:
    """Samples per forward pass. Accumulation enlarges the batch of the optimizer only, not the batch of the BatchNorm
    statistics, so the micro-batch has a lower bound."""
    batch, accumulate = int(cfg.train.batch_size), int(cfg.train.accumulate)
    if batch % accumulate:
        raise ValueError(f"batch_size {batch} is not a multiple of accumulate {accumulate}")
    if batch // accumulate < int(cfg.train.min_micro_batch):
        raise ValueError(f"micro-batch {batch // accumulate} is below min_micro_batch {cfg.train.min_micro_batch}")
    return batch // accumulate


def learning_rate(step: int, total: int, warmup: int, lr: float, min_lr: float) -> float:
    """lr of optimizer step `step` (from 0): linear up to lr over `warmup` steps, then cosine down to min_lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + (lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_states() -> dict:
    states = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def save_atomically(state: dict, path: Path) -> None:
    """Through a temporary file, so that a runtime cut while saving leaves the previous checkpoint intact."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def worker_setup(_worker: int) -> None:
    """OpenCV threads would only compete with the other loader worker on the 2 vCPUs of Colab."""
    cv2.setNumThreads(0)


def make_loader(dataset: TileDataset, batch_size: int, cfg, **options) -> DataLoader:
    workers = int(cfg.train.num_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        worker_init_fn=worker_setup if workers > 0 else None,
        generator=torch.Generator().manual_seed(int(cfg.seed)),
        **options,
    )


@dataclass
class Parts:
    """What a training step needs."""

    device: torch.device
    amp: bool
    accumulate: int
    model: nn.Module
    criterion: SegmentationLoss
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    sampler: EpochSampler
    loader: DataLoader
    steps_per_epoch: int


def prepare(cfg, splits_dir: Path | None = None) -> Parts:
    micro = micro_batch(cfg)
    accumulate = int(cfg.train.accumulate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp = bool(cfg.train.amp) and device.type == "cuda"
    seed = int(cfg.seed)
    seed_everything(seed)
    model = build_model(cfg).to(device).train()
    dataset = TileDataset(cfg, "train_dedup", "train", seed, splits_dir)
    sampler = EpochSampler(len(dataset), seed)
    loader = make_loader(dataset, micro, cfg, sampler=sampler, drop_last=True)
    steps_per_epoch = len(dataset) // micro // accumulate
    if steps_per_epoch == 0:
        raise ValueError(f"{len(dataset)} tiles do not fill one optimizer step of {int(cfg.train.batch_size)}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    return Parts(
        device,
        amp,
        accumulate,
        model,
        SegmentationLoss(cfg).to(device),
        optimizer,
        torch.amp.GradScaler(device.type, enabled=amp),
        sampler,
        loader,
        steps_per_epoch,
    )


def optimizer_step(parts: Parts, iterator, lr: float) -> tuple[dict[str, torch.Tensor], float]:
    """One step over `accumulate` micro-batches; returns the mean loss terms and the seconds spent waiting for data."""
    parts.optimizer.zero_grad(set_to_none=True)
    terms, waited = {}, 0.0
    for _ in range(parts.accumulate):
        tic = time.perf_counter()
        batch = next(iterator)
        waited += time.perf_counter() - tic
        batch = {key: value.to(parts.device, non_blocking=True) for key, value in batch.items()}
        with torch.autocast(parts.device.type, dtype=torch.float16, enabled=parts.amp):
            output = parts.model(batch["image"])
        losses = parts.criterion(output, batch)
        parts.scaler.scale(losses["loss"] / parts.accumulate).backward()
        for key, value in losses.items():
            terms[key] = terms.get(key, 0.0) + value.detach() / parts.accumulate
    for group in parts.optimizer.param_groups:
        group["lr"] = lr
    parts.scaler.step(parts.optimizer)
    parts.scaler.update()
    return terms, waited


@torch.inference_mode()
def validate(model: nn.Module, loader, device: torch.device, amp: bool, ignore: int) -> dict:
    """Foreground IoU (1 − P_bg > 0.5 as in the post-processing, σ(logit) > 0.5 on binary models) and F1 of the contact
    class (argmax), both pooled over all pixels of the tiles; padding is left out. Contact is None on binary models."""
    model.eval()
    counts = dict.fromkeys(("inter", "union", "tp", "fp", "fn"), 0)
    binary = False
    for batch in loader:
        classes = batch["classes"].to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            output = model(batch["image"].to(device, non_blocking=True))
        logits, _ = split_output(output.float())
        keep, truth = classes != ignore, (classes > 0) & (classes != ignore)
        binary = logits.shape[1] == 1
        if binary:
            fg = torch.sigmoid(logits[:, 0]) > 0.5
        else:
            fg = 1.0 - F.softmax(logits, dim=1)[:, 0] > 0.5
            predicted, actual = (logits.argmax(1) == 2) & keep, classes == 2
            counts["tp"] += int((predicted & actual).sum())
            counts["fp"] += int((predicted & ~actual).sum())
            counts["fn"] += int((~predicted & actual).sum())
        counts["inter"] += int((fg & truth).sum())
        counts["union"] += int(((fg & keep) | truth).sum())
    model.train()
    scores = {"fg_iou": counts["inter"] / max(counts["union"], 1)}
    if binary:
        return {**scores, "contact_f1": None, "contact_precision": None, "contact_recall": None}
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **scores,
        "contact_f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "contact_precision": tp / max(tp + fp, 1),
        "contact_recall": tp / max(tp + fn, 1),
    }


@dataclass
class Position:
    """Where a run stands."""

    epoch: int = 0  # epoch in progress
    step_in_epoch: int = 0  # optimizer steps done in it
    step: int = 0  # optimizer steps done in the run
    best: float = -1.0  # best value of the selection metric
    seconds: float = 0.0  # training seconds of the earlier sessions


def save_last(path: Path, parts: Parts, position: Position, cfg) -> None:
    state = {
        "model": parts.model.state_dict(),
        "optimizer": parts.optimizer.state_dict(),
        "scaler": parts.scaler.state_dict(),
        "position": asdict(position),
        "rng": rng_states(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_atomically(state, path)


def load_last(path: Path, parts: Parts) -> Position:
    """Everything is read onto the CPU first: random states must stay there, and load_state_dict moves the rest."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    parts.model.load_state_dict(state["model"])
    parts.optimizer.load_state_dict(state["optimizer"])
    parts.scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    return Position(**state["position"])


class CsvLog:
    """Rows appended to a CSV file. A resumed run first drops the rows after its checkpoint (key above upto), so the
    curves go on without repeated steps."""

    def __init__(self, path: Path, columns: tuple[str, ...], key: str, upto: int):
        self.path, self.columns = path, columns
        rows = []
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                rows = [row for row in csv.DictReader(f) if int(row[key]) <= upto]
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, columns)
            writer.writeheader()
            writer.writerows(rows)

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8", newline="") as f:
            csv.DictWriter(f, self.columns).writerow(row)


def record_session(run_dir: Path, parts: Parts, position: Position) -> None:
    """run.json: one entry per training session, with what a result has to be traced to."""
    path = run_dir / "run.json"
    sessions = read_json(path)["sessions"] if path.exists() else []
    sessions.append(
        {
            "started": timestamp(),
            "git": git_revision(),
            "start_step": position.step,
            "device": torch.cuda.get_device_name(0) if parts.device.type == "cuda" else platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        }
    )
    write_json(path, {"sessions": sessions})


def check_same_config(run_dir: Path, cfg) -> None:
    saved = OmegaConf.load(run_dir / "config.yaml")
    if OmegaConf.to_container(saved) != OmegaConf.to_container(cfg, resolve=True):
        raise ValueError(f"the config differs from the one of the run in {run_dir}; a resumed run cannot change it")


def fit(
    cfg,
    run_dir: Path,
    resume: bool = False,
    max_hours: float | None = None,
    stop_after: int | None = None,
    splits_dir: Path | None = None,
) -> dict:
    """Train until the budget of E epochs ends, or until max_hours or stop_after optimizer steps stop the run cleanly
    with last.pt saved. Writes config.yaml, run.json, train_log.csv, val_log.csv, last.pt and best.pt in run_dir."""
    if cfg.train.epochs is None or int(cfg.train.epochs) < 1:
        raise ValueError("train.epochs is not set: E is fixed after the benchmark (docs/EXPERIMENTS.md §5)")
    last, best = run_dir / "last.pt", run_dir / "best.pt"
    if last.exists() and not resume:
        raise FileExistsError(f"{last} exists: pass resume=True to continue that run")
    parts = prepare(cfg, splits_dir)
    if last.exists():
        check_same_config(run_dir, cfg)
        position = load_last(last, parts)
    else:
        position = Position()
        save_config(cfg, run_dir)
    record_session(run_dir, parts, position)
    train_log = CsvLog(run_dir / "train_log.csv", TRAIN_COLUMNS, "step", position.step)
    val_log = CsvLog(run_dir / "val_log.csv", VAL_COLUMNS, "epoch", position.epoch - 1)

    micro = micro_batch(cfg)
    train_cfg = cfg.train
    epochs, steps = int(train_cfg.epochs), parts.steps_per_epoch
    total, warmup = epochs * steps, int(train_cfg.warmup_epochs) * steps
    select = TileDataset(cfg, str(train_cfg.select_split), "val", int(cfg.seed), splits_dir)
    select_loader = make_loader(select, micro, cfg, shuffle=False)
    select_metric = str(train_cfg.select_metric)
    log_every = int(train_cfg.log_every_steps)
    started = saved = time.monotonic()
    earlier = position.seconds

    def spent() -> float:
        return earlier + time.monotonic() - started

    def must_stop() -> bool:
        over_time = max_hours is not None and time.monotonic() - started >= max_hours * 3600
        return over_time or (stop_after is not None and position.step >= stop_after)

    def save() -> None:
        nonlocal saved
        position.seconds = spent()
        save_last(last, parts, position, cfg)
        saved = time.monotonic()

    def outcome(status: str) -> dict:
        return {"status": status, "epoch": position.epoch, "step": position.step, "best": position.best}

    while position.epoch < epochs:
        parts.sampler.set_epoch(position.epoch, position.step_in_epoch * int(train_cfg.batch_size))
        iterator = iter(parts.loader)
        running, window, waited, tic = {}, 0, 0.0, time.perf_counter()
        while position.step_in_epoch < steps:
            lr = learning_rate(position.step, total, warmup, float(train_cfg.lr), float(train_cfg.min_lr))
            terms, data_seconds = optimizer_step(parts, iterator, lr)
            position.step += 1
            position.step_in_epoch += 1
            waited += data_seconds
            window += 1
            for key, value in terms.items():
                running[key] = running.get(key, 0.0) + value
            if position.step % log_every == 0:
                elapsed = time.perf_counter() - tic
                values = {key: float(value) / window for key, value in running.items()}
                row = {"step": position.step, "epoch": position.epoch, "lr": lr, **values,
                       "data_fraction": waited / max(elapsed, 1e-9), "seconds": round(spent(), 1)}  # fmt: skip
                train_log.write(row)
                print(json.dumps(row), flush=True)
                running, window, waited, tic = {}, 0, 0.0, time.perf_counter()
            if position.step_in_epoch < steps:
                stop = must_stop()
                if stop or time.monotonic() - saved >= float(train_cfg.checkpoint_every_min) * 60:
                    save()
                if stop:
                    return outcome("stopped")

        scores = validate(parts.model, select_loader, parts.device, parts.amp, int(cfg.data.ignore_index))
        improved = scores[select_metric] > position.best
        if improved:
            position.best = scores[select_metric]
            state = {"model": parts.model.state_dict(), "epoch": position.epoch, **scores, "git": git_revision()}
            save_atomically(state, best)
        val_log.write({"epoch": position.epoch, "step": position.step, **scores, "best": int(improved),
                       "seconds": round(spent(), 1)})  # fmt: skip
        print(json.dumps({"epoch": position.epoch, **scores}), flush=True)
        position.epoch += 1
        position.step_in_epoch = 0
        save()
        if must_stop() and position.epoch < epochs:
            return outcome("stopped")
    return outcome("finished")


def endless_batches(parts: Parts):
    """Batches over consecutive epochs, for a benchmark that may outlast one."""
    epoch = 0
    while True:
        parts.sampler.set_epoch(epoch)
        yield from parts.loader
        epoch += 1


def benchmark(cfg, iterations: int, splits_dir: Path | None = None) -> dict:
    """Iterations per second, share of the time waiting for data, peak GPU memory and the mean loss terms, on the first
    iterations of a run; the terms are the material of the λ decision (PLAN.md Phase 6). Nothing is saved."""
    parts = prepare(cfg, splits_dir)
    iterator = endless_batches(parts)
    lr = float(cfg.train.lr)
    terms = {key: [] for key in TERMS}
    waited, tic = 0.0, time.perf_counter()
    for i in range(BENCHMARK_WARMUP + iterations):
        if i == BENCHMARK_WARMUP:
            waited, tic = 0.0, time.perf_counter()
            if parts.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
        losses, data_seconds = optimizer_step(parts, iterator, lr)
        waited += data_seconds
        for key, value in losses.items():
            terms[key].append(float(value))
    if parts.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - tic
    cuda = parts.device.type == "cuda"
    return {
        "exp": str(cfg.exp),
        "device": torch.cuda.get_device_name(0) if cuda else platform.processor(),
        "created": timestamp(),
        "git": git_revision(),
        "torch": torch.__version__,
        "iterations": iterations,
        "micro_batch": micro_batch(cfg),
        "accumulate": parts.accumulate,
        "num_workers": int(cfg.train.num_workers),
        "steps_per_second": iterations / seconds,
        "data_fraction": waited / seconds,
        "steps_per_epoch": parts.steps_per_epoch,
        "hours_per_epoch": parts.steps_per_epoch * seconds / iterations / 3600,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3 if cuda else None,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3 if cuda else None,
        "loss_terms": {key: float(np.mean(values[BENCHMARK_WARMUP:])) for key, values in terms.items()},
    }
