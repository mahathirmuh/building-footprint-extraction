"""Training transforms (docs/METHOD.md §3.5): targets re-derived after a geometric op equal the transformed targets,
with the sign rule of sin 4θ; photometric changes touch RGB only; padding values."""

import numpy as np
import pytest

from densepoly.config import load_config
from densepoly.data import transforms
from densepoly.labels.targets import targets_from_config
from tests.fixtures.synthetic import make_scene

OPS = [
    [("flip_h", 0)],
    [("flip_v", 0)],
    [("transpose", 0)],
    [("rot90", 1)],
    [("rot90", 2)],
    [("rot90", 3)],
    [("flip_h", 0), ("transpose", 0)],  # two sign changes: sin 4θ keeps its sign
    [("flip_h", 0), ("flip_v", 0), ("rot90", 1), ("transpose", 0)],
]


@pytest.fixture(scope="module")
def cfg():
    return load_config("crowdai", "full_p2")


@pytest.fixture(scope="module", params=range(2), ids=lambda s: f"seed{s}")
def scene(request):
    return make_scene(request.param)


@pytest.mark.parametrize("ops", OPS, ids=lambda ops: "+".join(f"{n}{k or ''}" for n, k in ops))
def test_targets_derived_after_the_op_equal_the_transformed_targets(cfg, scene, ops):
    before = targets_from_config(scene.instance_map, cfg.labels)
    after = targets_from_config(transforms.apply_geometry(scene.instance_map, ops), cfg.labels)
    moved = {name: transforms.apply_geometry(getattr(before, name), ops) for name in vars(before)}
    assert np.array_equal(after.classes, moved["classes"]) and np.array_equal(after.valid, moved["valid"])
    np.testing.assert_allclose(after.weight, moved["weight"], rtol=1e-6)
    np.testing.assert_allclose(after.cos4, moved["cos4"], atol=1e-4)
    np.testing.assert_allclose(after.sin4, transforms.sin4_sign(ops) * moved["sin4"], atol=1e-4)


def test_sin4_sign_follows_the_method_table():
    assert [transforms.sin4_sign(ops) for ops in OPS] == [-1, -1, -1, 1, 1, 1, 1, -1]


def test_random_geometry_follows_the_probabilities(cfg):
    rng = np.random.default_rng(0)
    always = cfg.augment.copy()
    always.p_flip_h = always.p_flip_v = always.p_rot90 = always.p_transpose = 1.0
    names = [n for n, _ in transforms.random_geometry(rng, always)]
    assert names == ["flip_h", "flip_v", "rot90", "transpose"]
    never = cfg.augment.copy()
    never.p_flip_h = never.p_flip_v = never.p_rot90 = never.p_transpose = 0.0
    assert transforms.random_geometry(rng, never) == []
    ks = {k for _ in range(200) for n, k in transforms.random_geometry(rng, always) if n == "rot90"}
    assert ks == {1, 2, 3}


def test_photometric_change_stays_within_its_limit():
    rng = np.random.default_rng(0)
    image = np.random.default_rng(1).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    out = transforms.photometric(image, rng, 0.1).astype(float)
    low, high = image * 0.9 - 25.5, image * 1.1 + 25.5
    assert ((out >= np.floor(np.clip(low, 0, 255))) & (out <= np.ceil(np.clip(high, 0, 255)))).all()


def test_make_sample_pads_and_never_touches_priors_photometrically(cfg, scene):
    priors = np.random.default_rng(2).integers(0, 256, (300, 300, 2), dtype=np.uint8)
    rng = np.random.default_rng(3)
    sample = transforms.make_sample(scene.image, priors, scene.instance_map, cfg, rng)
    assert sample.image.shape == (320, 320, 3) and sample.priors.shape == (320, 320, 2)
    t = sample.targets
    assert (t.classes[300:] == 255).all() and (t.classes[:, 300:] == 255).all()
    assert not t.valid[300:].any() and not t.weight[300:].any() and not t.sin4[:, 300:].any()
    assert (t.classes[:300, :300] != 255).all() and (t.weight[:300, :300] >= 1).all()
    # the priors went through the same geometry as the image, but no photometric change: same multiset of values
    assert np.array_equal(np.sort(sample.priors[:300, :300].ravel()), np.sort(priors.ravel()))


def test_make_sample_without_rng_keeps_the_tile_and_reflects_the_padding(cfg, scene):
    sample = transforms.make_sample(scene.image, None, scene.instance_map, cfg)
    assert sample.priors is None
    assert np.array_equal(sample.image[:300, :300], scene.image)
    assert np.array_equal(sample.image[300, :300], scene.image[298])  # numpy "reflect" skips the edge row
    assert np.array_equal(sample.image[:300, 301], scene.image[:, 297])
    expected = targets_from_config(scene.instance_map, cfg.labels)
    assert np.array_equal(sample.targets.classes[:300, :300], expected.classes)


def test_whu_tiles_need_no_padding():
    cfg = load_config("whu", "full_p2")
    scene = make_scene(5, size=256)
    image, instances = scene.image[:224, :224], scene.instance_map[:224, :224]
    sample = transforms.make_sample(image, None, instances, cfg)
    assert sample.image.shape == (224, 224, 3) and (sample.targets.classes != 255).all()
    with pytest.raises(ValueError, match="does not fit"):
        transforms.pad(np.zeros((300, 300)), 224)
