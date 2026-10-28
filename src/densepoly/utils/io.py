"""File helpers shared by the scripts: atomic JSON and text writes, PNG I/O that works for any Windows path."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np


def _replace_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_json(path: Path, data, indent: int | None = 1) -> None:
    """Write JSON through a temporary file, so an interrupted run never leaves a truncated file behind."""
    _replace_atomically(path, json.dumps(data, indent=indent, ensure_ascii=False) + "\n")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_lines(path: Path, lines: Iterable[str]) -> None:
    _replace_atomically(path, "".join(f"{line}\n" for line in lines))


def read_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_png(path: Path, array: np.ndarray) -> None:
    """PNG through cv2.imencode, which keeps uint16 and, unlike cv2.imwrite, accepts non-ASCII Windows paths."""
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise OSError(f"could not encode {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer.tofile(str(path))


def read_png(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
