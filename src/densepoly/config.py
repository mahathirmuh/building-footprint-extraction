"""Experiment configuration (PLAN.md Phase 0).

The final config of a run is ``configs/base.yaml`` <- ``configs/data/<dataset>.yaml`` <- ``configs/exp/<exp>.yaml``
<- the frozen tuning ``configs/postproc/<dataset>/<exp>.yaml`` <- CLI overrides (``key=value``), later layers winning.
``base.yaml`` is the schema: the other layers may only change keys it defines, so a mistyped key in a YAML file or on
the command line raises instead of silently creating a new setting. Values marked ``???`` in ``base.yaml`` must be
filled by a lower layer. Only the ``postproc`` section of a frozen tuning is merged; the rest of that file records
how it was chosen (docs/EXPERIMENTS.md §4).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from densepoly import paths


def postproc_file(dataset: str, exp: str, root: Path | None = None) -> Path:
    """Frozen post-processing of an experiment on a dataset, written by scripts/tune_postproc.py."""
    return (paths.configs_dir() if root is None else root) / "postproc" / dataset / f"{exp}.yaml"


def load_config(
    dataset: str, exp: str, overrides: Sequence[str] = (), root: Path | None = None, tuned: bool = True
) -> DictConfig:
    """Merge the config layers of one run; raises ValueError on unknown keys, bad overrides or missing values.
    tuned=False leaves out the frozen tuning, as the tuning itself must."""
    root = paths.configs_dir() if root is None else root
    cfg = OmegaConf.load(root / "base.yaml")
    OmegaConf.set_struct(cfg, True)
    layers = [(root / "data" / f"{dataset}.yaml", None), (root / "exp" / f"{exp}.yaml", None)]
    if tuned and postproc_file(dataset, exp, root).exists():
        layers.append((postproc_file(dataset, exp, root), "postproc"))
    for layer, section in layers:
        try:
            content = OmegaConf.load(layer)
            cfg = OmegaConf.merge(cfg, content if section is None else {section: content[section]})
        except OmegaConfBaseException as err:
            raise ValueError(f"{layer}: {err}") from err

    malformed = [item for item in overrides if "=" not in item]
    if malformed:
        raise ValueError(f"overrides must look like key=value, got {malformed}")
    try:
        cfg.merge_with_dotlist(list(overrides))
    except OmegaConfBaseException as err:
        raise ValueError(f"override: {err}") from err

    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"missing values (??? in base.yaml) for {missing}")
    if (cfg.data.name, cfg.exp) != (dataset, exp):
        raise ValueError(f"config names {(cfg.data.name, cfg.exp)} do not match the requested {(dataset, exp)}")
    return cfg


def save_config(cfg: DictConfig, run_dir: Path) -> Path:
    """Write the final, resolved config into the run folder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    return path
