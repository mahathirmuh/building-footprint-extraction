"""Import of HiSup's released predictions (scripts/import_hisup.py) on a fake release."""

import hashlib
import importlib.util
import json

import pytest

from densepoly import paths
from densepoly.utils.io import read_json, write_json
from tests.fixtures.coco_toy import make_gt, make_results, square


def load_script():
    spec = importlib.util.spec_from_file_location("import_hisup", paths.REPO_ROOT / "scripts" / "import_hisup.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def release(tmp_path, monkeypatch):
    """val with images 1-4, val_clean with images 2 and 4, and a release with predictions on images 1, 2 and 4."""
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("DENSEPOLY_OUT", str(tmp_path / "outputs"))
    houses = {i: [square(10, 10, 20)] for i in (1, 2, 3, 4)}
    write_json(paths.gt_file("crowdai", "val_full"), make_gt(houses))
    write_json(paths.gt_file("crowdai", "val_clean"), make_gt({i: houses[i] for i in (2, 4)}))
    results = make_results({1: [square(10, 10, 20)], 2: [square(11, 10, 20), square(40, 40, 8)], 4: [square(9, 9, 9)]})
    for r in results:  # as released: no theta_deg / kappa fields
        del r["theta_deg"], r["kappa"]
    source = paths.dataset_dir("crowdai") / "hisup_release" / "crowdai_test_hrnet48.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(results), encoding="utf-8")
    return results, source


def test_import_keeps_the_predictions_as_released(release):
    results, source = release
    load_script().main([])
    full = read_json(paths.pred_file("crowdai", "hisup_release", "ref", "val_full"))
    clean = read_json(paths.pred_file("crowdai", "hisup_release", "ref", "val_clean"))
    assert full == results and clean == [r for r in results if r["image_id"] in (2, 4)]
    meta = read_json(paths.pred_file("crowdai", "hisup_release", "ref", "val_clean").with_name("val_clean.meta.json"))
    assert meta["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (meta["results"], meta["images_with_predictions"], meta["images"]) == (3, 2, 2)
    assert meta["source"] == "crowdai/hisup_release/crowdai_test_hrnet48.json"


def test_import_rejects_images_outside_crowdai_val(release):
    results, source = release
    source.write_text(json.dumps([*results, {**results[0], "image_id": 77}]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not CrowdAI val images"):
        load_script().main([])
