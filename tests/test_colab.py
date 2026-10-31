"""The tar for Colab and the training notebook (scripts/pack_colab.py, notebooks/colab_train.ipynb; docs/COLAB.md)."""

import importlib.util
import json
import tarfile

import pytest

from densepoly import paths
from densepoly.config import load_config
from densepoly.data.dataset import TileDataset
from densepoly.utils.io import read_json
from tests.fixtures.fake_tiles import write_tiles

SMALL = ["data.tile_size=120", "data.pad_to=128"]


def load_script():
    spec = importlib.util.spec_from_file_location("pack_colab", paths.REPO_ROOT / "scripts" / "pack_colab.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def packed(tmp_path, monkeypatch):
    """Fake tiles of the training splits and of test, and the tar: (root folder, splits folder, tar file)."""
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    splits = tmp_path / "splits"
    splits.mkdir()
    write_tiles(splits, {"train_dedup": 3, "val_tune": 2, "test": 1})
    load_script().main(["--data", "crowdai", "--splits-dir", str(splits)])
    return tmp_path, splits, paths.out_root() / "colab" / "crowdai_train_dedup.tar"


def test_the_tar_holds_the_training_splits_and_no_test_data(packed):
    _, _, tar_path = packed
    with tarfile.open(tar_path) as tar:
        names = tar.getnames()
    assert len(names) == 2 + 3 * (3 + 2)  # two index files, then image, instance map and prior per tile
    assert len(set(names)) == len(names) and not any("/test/" in name for name in names)
    assert sum(name.startswith("crowdai/train/images/") for name in names) == 5
    assert "instances/crowdai/val_tune/index.csv" in names


def test_a_dataset_reads_the_extracted_tar(packed, monkeypatch):
    root, splits, tar_path = packed
    target = root / "colab_data"
    with tarfile.open(tar_path) as tar:
        tar.extractall(target, filter="data")
    monkeypatch.setenv("DENSEPOLY_DATA", str(target))
    cfg = load_config("crowdai", "full_p2", SMALL)
    assert TileDataset(cfg, "train_dedup", "train", 0, splits)[(0, 2)]["image"].shape == (5, 128, 128)
    assert len(TileDataset(cfg, "val_tune", "val", 0, splits)) == 2


def test_verify_accepts_the_tar_and_refuses_a_damaged_copy(packed, capsys):
    _, _, tar_path = packed
    script = load_script()
    script.main(["--data", "crowdai", "--verify"])
    assert "same as its manifest" in capsys.readouterr().out
    manifest = read_json(tar_path.with_suffix(".json"))
    assert manifest["files"] == 17 and manifest["folders"]["priors/crowdai/train_dedup"] == 3
    with tar_path.open("r+b") as f:
        f.seek(2000)
        f.write(b"damaged")
    with pytest.raises(SystemExit):
        script.main(["--data", "crowdai", "--verify"])


def test_a_missing_file_stops_the_packing(packed):
    root, splits, _ = packed
    next((root / "data" / "priors" / "crowdai" / "train_dedup").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError):
        load_script().main(["--data", "crowdai", "--splits-dir", str(splits)])


def test_the_notebook_stays_thin_and_clean():
    notebook = json.loads((paths.REPO_ROOT / "notebooks" / "colab_train.ipynb").read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["outputs"] == [] and cell["execution_count"] is None for cell in code)  # no outputs in git
    source = "".join(line for cell in code for line in cell["source"])
    assert "scripts/train.py" in source and "--resume" in source and "--benchmark" in source
    assert "scripts/pack_colab.py" in source and "--verify" in source
    assert "token" not in source.lower() and "userdata" not in source  # the repo is public
    assert not any(word in source for word in ("def ", "class ", "import torch\n"))  # logic stays in src/
