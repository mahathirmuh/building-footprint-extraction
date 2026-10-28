"""File helpers and provenance stamps."""

import numpy as np

from densepoly.utils.io import read_json, read_lines, read_png, write_json, write_lines, write_png
from densepoly.utils.provenance import git_revision, timestamp


def test_json_and_lines_round_trip_without_leftovers(tmp_path):
    data = {"a": [1, 2.5, None], "ü": "θ"}
    write_json(tmp_path / "x" / "d.json", data)
    assert read_json(tmp_path / "x" / "d.json") == data
    write_lines(tmp_path / "l.txt", ["a.jpg", "b.jpg"])
    assert (tmp_path / "l.txt").read_bytes() == b"a.jpg\nb.jpg\n"
    assert read_lines(tmp_path / "l.txt") == ["a.jpg", "b.jpg"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_png_keeps_uint16_values_and_accepts_unicode_paths(tmp_path):
    array = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 13  # values far above 255
    path = tmp_path / "θ folder" / "map.png"
    write_png(path, array)
    restored = read_png(path)
    assert restored.dtype == np.uint16 and np.array_equal(restored, array)


def test_provenance_stamps():
    revision = git_revision()
    assert revision == "unknown" or len(revision.removesuffix("-dirty")) == 40
    assert timestamp()[:4].isdigit() and "T" in timestamp()
