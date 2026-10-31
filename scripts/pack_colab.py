"""Pack what the training on Colab reads into one tar (docs/COLAB.md §1).

    python scripts/pack_colab.py --data crowdai
    python scripts/pack_colab.py --data crowdai --verify     # on Colab, after copying the tar from Drive

Writes outputs/colab/<data>_train_dedup.tar and a .json manifest next to it: the images, instance cache and prior cache
of train_dedup and val_tune, with paths relative to DENSEPOLY_DATA, uncompressed because the tiles already are. Test
data stays out: training never reads it, and inference runs on the laptop. --verify compares the tar with its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path

from densepoly import paths
from densepoly.priors.cache import units
from densepoly.utils.io import read_json, write_json
from densepoly.utils.provenance import git_revision, timestamp

SPLITS = ("train_dedup", "val_tune")
CHUNK = 1 << 20


def members(dataset: str, splits_dir: Path | None = None) -> list[Path]:
    """Every file of the tar, in a stable order."""
    files = []
    for split in SPLITS:
        files.append(paths.instance_dir(dataset, split) / "index.csv")
        for unit in units(dataset, split, splits_dir):
            files += [
                unit.image,
                paths.instance_cache(dataset, split, unit.key),
                paths.prior_cache(dataset, split, unit.key),
            ]
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} files are missing, the first is {missing[0]}")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def folder_counts(names: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(name.rsplit("/", 1)[0] for name in names).items()))


def pack(dataset: str, target: Path, splits_dir: Path | None = None) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    root = paths.data_root()
    files = members(dataset, splits_dir)
    names = [path.relative_to(root).as_posix() for path in files]
    tmp = target.with_name(target.name + ".tmp")
    with tarfile.open(tmp, "w") as tar:
        for path, name in zip(files, names, strict=True):
            tar.add(path, arcname=name, recursive=False)
    tmp.replace(target)
    return {
        "created": timestamp(),
        "git": git_revision(),
        "bytes": target.stat().st_size,
        "sha256": sha256(target),
        "files": len(names),
        "folders": folder_counts(names),
    }


def verify(target: Path, manifest: dict) -> list[str]:
    """What differs between a tar and its manifest; nothing when the copy is good."""
    problems = []
    if target.stat().st_size != manifest["bytes"]:
        problems.append(f"size {target.stat().st_size}, manifest says {manifest['bytes']}")
    if sha256(target) != manifest["sha256"]:
        problems.append("sha256 differs from the manifest")
    with tarfile.open(target) as tar:
        names = tar.getnames()
    if folder_counts(names) != manifest["folders"]:
        problems.append("the files per folder differ from the manifest")
    return problems


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=("crowdai",), required=True, help="WHU comes with Phase 7")
    parser.add_argument("--out", type=Path, help="tar file (default: outputs/colab/<data>_train_dedup.tar)")
    parser.add_argument("--verify", action="store_true", help="compare the tar with its manifest")
    parser.add_argument("--splits-dir", type=Path, help="folder of the split lists (default: repo splits/)")
    args = parser.parse_args(argv)

    target = args.out or paths.out_root() / "colab" / f"{args.data}_train_dedup.tar"
    manifest_file = target.with_suffix(".json")
    if args.verify:
        problems = verify(target, read_json(manifest_file))
        if problems:
            raise SystemExit(f"{target}: " + "; ".join(problems))
        print(f"{target}: same as its manifest")
        return
    manifest = pack(args.data, target, args.splits_dir)
    write_json(manifest_file, manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
