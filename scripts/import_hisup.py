"""Import HiSup's released predictions on CrowdAI val (docs/DATA.md §1) as the method hisup_release, run ref.

    python scripts/import_hisup.py

Reads data/crowdai/hisup_release/crowdai_test_hrnet48.json and writes, unchanged, outputs/preds/crowdai/
hisup_release/ref/val_full.json (all predictions) and val_clean.json (those on the 212 val_clean tiles), each with a
.meta.json recording the source file and its sha256. Evaluate them with --released (lenient checks, docs/EVALUATION.md
§6.4): python scripts/evaluate.py --data crowdai --exp hisup_release --split val_full --released
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from densepoly import paths
from densepoly.eval import coco_io
from densepoly.utils.io import write_json
from densepoly.utils.provenance import git_revision, timestamp

EXP = "hisup_release"
SOURCE = "crowdai_test_hrnet48.json"
SPLITS = ("val_full", "val_clean")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, help=f"released file (default: data/crowdai/hisup_release/{SOURCE})")
    args = parser.parse_args(argv)
    source = args.source or paths.dataset_dir("crowdai") / "hisup_release" / SOURCE
    data = source.read_bytes()
    sha256 = hashlib.sha256(data).hexdigest()
    results = json.loads(data)
    try:
        source_name = source.resolve().relative_to(paths.data_root()).as_posix()
    except ValueError:
        source_name = str(source)

    for split in SPLITS:
        image_ids = {img["id"] for img in coco_io.load_gt("crowdai", split)["images"]}
        if split == "val_full":
            unknown = {r["image_id"] for r in results} - image_ids
            if unknown:
                raise SystemExit(
                    f"{len(unknown)} predicted image ids are not CrowdAI val images, e.g. {sorted(unknown)[:3]}"
                )
        subset = coco_io.restrict_to_images(results, image_ids)
        out = paths.pred_file("crowdai", EXP, "ref", split)
        write_json(out, subset, indent=None)
        meta = {
            "source": source_name,
            "sha256": sha256,
            "results": len(subset),
            "images_with_predictions": len({r["image_id"] for r in subset}),
            "images": len(image_ids),
            "imported": timestamp(),
            "git": git_revision(),
        }
        write_json(out.with_name(f"{split}.meta.json"), meta)
        print(json.dumps({"split": split, **meta}))


if __name__ == "__main__":
    main()
