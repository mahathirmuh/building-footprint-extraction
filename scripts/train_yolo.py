"""Train the FULL+Y detector, YOLOv8n-OBB (docs/EXPERIMENTS.md §7), on the set exported by scripts/prepare_yolo.py.

    python scripts/train_yolo.py --data crowdai --weights /path/to/yolov8n-obb.pt
    python scripts/train_yolo.py --data crowdai --weights /path/to/yolov8n-obb.pt --epochs 1 --name benchmark

Frozen settings of §7: pretrained yolov8n-obb, imgsz 320 on CrowdAI (224 on WHU), 100 epochs, batch 32, seed 0, the
default augmentations of Ultralytics. Writes $DENSEPOLY_OUT/yolo/<data>/runs/<name>/ (weights/best.pt, results.csv).

Ultralytics is AGPL-3.0: it is imported by this script and scripts/predict_yolo.py only, never by the densepoly
package, and it runs from its own environment (pip install -e ".[yolo]").
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from densepoly import paths

IMAGE_SIZE = {"crowdai": 320, "whu": 224}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", choices=paths.DATASETS, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="pretrained yolov8n-obb.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="seed0", help="run folder")
    parser.add_argument("--resume", action="store_true", help="continue the run from its weights/last.pt")
    args = parser.parse_args(argv)

    from ultralytics import YOLO  # here, so that --help works without it

    root = paths.yolo_dir(args.data)
    last = root / "runs" / args.name / "weights" / "last.pt"
    tic = time.perf_counter()
    if args.resume and last.exists():
        model = YOLO(str(last))
        model.train(resume=True)
    else:
        model = YOLO(str(args.weights))
        model.train(
            data=str(root / "dataset" / "data.yaml"),
            imgsz=IMAGE_SIZE[args.data],
            epochs=args.epochs,
            batch=args.batch,
            seed=args.seed,
            workers=args.workers,
            project=str(root / "runs"),
            name=args.name,
            exist_ok=True,
            device=0,
            plots=False,
            verbose=False,
        )
    print(json.dumps({"status": "finished", "run": str(root / "runs" / args.name), "seconds": round(time.perf_counter() - tic, 1)}))


if __name__ == "__main__":
    main()
