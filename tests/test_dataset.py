"""Training dataset on fake tiles (densepoly.data.dataset; docs/METHOD.md §2.4, §3.5)."""

import numpy as np
import pytest
import torch
from PIL import Image

from densepoly import paths
from densepoly.config import load_config
from densepoly.data.dataset import IMAGENET_MEAN, IMAGENET_STD, EpochSampler, TileDataset
from densepoly.labels.contact import class_map
from densepoly.priors.cache import load_stats
from densepoly.utils.io import read_png
from tests.fixtures.fake_tiles import write_tiles

SIZE, PADDED = 120, 128
SMALL = [f"data.tile_size={SIZE}", f"data.pad_to={PADDED}"]


@pytest.fixture
def tiles(tmp_path, monkeypatch):
    monkeypatch.setenv("DENSEPOLY_DATA", str(tmp_path / "data"))
    splits = tmp_path / "splits"
    splits.mkdir()
    ids = write_tiles(splits, {"train_dedup": 4, "val_select": 2})
    return splits, ids


def config(exp="full_p2", *overrides):
    return load_config("crowdai", exp, [*SMALL, *overrides])


def dataset(tiles, mode, exp="full_p2", split="train_dedup", *overrides, seed=0):
    return TileDataset(config(exp, *overrides), split, mode, seed, tiles[0])


def tile(split, image_id):
    return np.asarray(Image.open(paths.image_dir("crowdai", split) / f"{image_id:012d}.png"))


@pytest.mark.parametrize(("exp", "channels"), [("full_p0", 3), ("full_p1", 4), ("full_p2", 5)])
def test_input_channels_follow_the_experiment(tiles, exp, channels):
    sample = dataset(tiles, "train", exp)[(0, 1)]
    assert sample["image"].shape == (channels, PADDED, PADDED) and sample["image"].dtype == torch.float32


def test_rgb_uses_imagenet_statistics_and_priors_the_train_statistics(tiles):
    data = dataset(tiles, "predict", "full_p2")
    sample = data[0]
    image_id = tiles[1]["train_dedup"][0]
    assert sample["image_id"] == image_id
    x = sample["image"].numpy()
    expected = (tile("train_dedup", image_id)[0, 0].astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(x[:3, 0, 0], expected, rtol=1e-5)
    stats = load_stats("crowdai")
    # write_tiles stores the ramps 0..255: MBI across the columns, the gap map across the rows
    assert x[3, 0, SIZE - 1] == pytest.approx((1.0 - stats["mean"][0]) / stats["std"][0], rel=1e-5)
    assert x[4, SIZE - 1, 0] == pytest.approx((1.0 - stats["mean"][1]) / stats["std"][1], rel=1e-5)


def test_full_p1_takes_only_the_mbi_channel(tiles):
    x = dataset(tiles, "predict", "full_p1")[0]["image"].numpy()
    stats = load_stats("crowdai")
    assert x[3, 0, SIZE - 1] == pytest.approx((1.0 - stats["mean"][0]) / stats["std"][0], rel=1e-5)


def test_padding_reflects_the_input_and_ignores_the_labels(tiles):
    sample = dataset(tiles, "train")[(0, 0)]
    classes = sample["classes"].numpy()
    assert (classes[SIZE:] == 255).all() and (classes[:, SIZE:] == 255).all() and (classes[:SIZE, :SIZE] < 3).all()
    assert (sample["weight"][SIZE:] == 0).all() and not sample["valid"][SIZE:].any()
    plain = dataset(tiles, "predict")[0]["image"]
    assert torch.equal(plain[:, SIZE, :], plain[:, SIZE - 2, :])  # numpy reflect does not repeat the edge row


def test_a_sample_is_a_pure_function_of_seed_epoch_and_index(tiles):
    data = dataset(tiles, "train")
    first = data[(3, 2)]
    data[(0, 0)]  # drawing another sample in between changes nothing
    again = dataset(tiles, "train")[(3, 2)]
    assert all(torch.equal(first[key], again[key]) for key in first)
    other_epochs = [data[(epoch, 2)] for epoch in range(4, 12)]
    assert any(not torch.equal(first["image"], other["image"]) for other in other_epochs)
    other_seeds = [dataset(tiles, "train", seed=seed)[(3, 2)] for seed in range(1, 9)]
    assert any(not torch.equal(first["image"], other["image"]) for other in other_seeds)


def test_labels_follow_the_augmented_image(tiles):
    """Without photometric changes, the targets of every augmented sample equal the label of the tile that the augmented
    RGB shows: the building pixels of the original tile move exactly as its RGB does."""
    data = dataset(tiles, "train", "full_p2", "train_dedup", "augment.p_photometric=0.0")
    image_id = tiles[1]["train_dedup"][1]
    original = np.dstack([tile("train_dedup", image_id), class_map(read_png(paths.instance_cache(
        "crowdai", "train_dedup", image_id)), 2)])  # fmt: skip
    seen = set()
    for epoch in range(30):
        sample = data[(epoch, 1)]
        rgb = sample["image"][:3].numpy().transpose(1, 2, 0)[:SIZE, :SIZE] * IMAGENET_STD + IMAGENET_MEAN
        rgb = np.rint(rgb * 255).astype(np.uint8)
        classes = sample["classes"].numpy()[:SIZE, :SIZE, None]
        moved = np.dstack([rgb, classes])
        matches = [
            k
            for k, candidate in enumerate(dihedral(original))
            if np.array_equal(candidate[..., :3], moved[..., :3]) and np.array_equal(candidate[..., 3], moved[..., 3])
        ]
        assert matches
        seen.add(matches[0])
    assert len(seen) >= 4  # the geometry really varies


def dihedral(array):
    """The 8 rotations and reflections of an (H, W, C) array."""
    views = [np.rot90(array, k, axes=(0, 1)) for k in range(4)]
    return views + [v[:, ::-1] for v in views]


def test_val_mode_gives_the_plain_tile_and_its_classes(tiles):
    data = dataset(tiles, "val", "b4", "val_select")
    sample = data[1]
    assert set(sample) == {"image", "classes"} and sample["image"].shape[0] == 3
    image_id = tiles[1]["val_select"][1]
    expected = class_map(read_png(paths.instance_cache("crowdai", "val_select", image_id)), 2)
    assert np.array_equal(sample["classes"].numpy()[:SIZE, :SIZE], expected)
    assert (sample["classes"].numpy()[SIZE:] == 255).all()


def test_val_mode_follows_the_contact_radius_of_the_config(tiles):
    wide = dataset(tiles, "val", "b4", "val_select", "labels.contact_radius_px=5")[0]["classes"]
    narrow = dataset(tiles, "val", "b4", "val_select")[0]["classes"]
    assert (wide == 2).sum() > (narrow == 2).sum()


def test_predict_mode_reads_no_instance_map(tiles):
    image_id = tiles[1]["val_select"][0]
    paths.instance_cache("crowdai", "val_select", image_id).unlink()
    assert dataset(tiles, "predict", "b4", "val_select")[0]["image_id"] == image_id


def test_unknown_mode_and_whu_are_refused(tiles):
    with pytest.raises(ValueError):
        dataset(tiles, "training")
    with pytest.raises(NotImplementedError):
        TileDataset(load_config("whu", "b4"), "train_dedup", "train", 0, tiles[0])


def test_epoch_sampler_shuffles_every_index_once_by_seed_and_epoch():
    sampler = EpochSampler(50, seed=0)
    sampler.set_epoch(3)
    first = list(sampler)
    assert len(sampler) == 50 and {e for e, _ in first} == {3} and sorted(i for _, i in first) == list(range(50))
    assert first == list(sampler)
    sampler.set_epoch(4)
    assert [i for _, i in sampler] != [i for _, i in first]
    other = EpochSampler(50, seed=1)
    other.set_epoch(3)
    assert list(other) != first


def test_epoch_sampler_skip_continues_the_same_order():
    sampler = EpochSampler(20, seed=2)
    sampler.set_epoch(1)
    whole = list(sampler)
    sampler.set_epoch(1, skip=8)
    assert list(sampler) == whole[8:] and len(sampler) == 12
