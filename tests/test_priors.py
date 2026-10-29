"""Morphological priors and their cache (docs/METHOD.md §2) on synthetic images with a known answer."""

import numpy as np
import pytest
from PIL import Image
from skimage.draw import disk

from densepoly.priors import cache
from densepoly.priors.gap import gap_map
from densepoly.priors.mbi import brightness, line_footprint, mbi, opening_by_reconstruction
from tests.fixtures.synthetic import make_scene

CROWDAI_SCALES = [11, 21, 31, 41, 51, 61, 71, 81]


def test_line_footprints_are_centred_lines_of_about_the_requested_length():
    horizontal = line_footprint(11, 0)
    assert horizontal.sum() == 11 and horizontal[5].all()
    assert np.array_equal(line_footprint(11, 90), horizontal.T)
    for angle in (45, 135):
        footprint = line_footprint(11, angle)
        assert np.array_equal(footprint, footprint[::-1, ::-1]) and footprint[5, 5] == 1
        assert footprint.sum() == 9  # 9 diagonal pixels: Euclidean length 8√2 ≈ 11
    assert np.array_equal(line_footprint(11, 45)[:, ::-1], line_footprint(11, 135))
    with pytest.raises(ValueError, match="odd"):
        line_footprint(30, 0)


def test_brightness_is_the_maximum_band():
    image = np.array([[[10, 200, 30], [255, 0, 0]]], np.uint8)
    np.testing.assert_allclose(brightness(image), [[200 / 255, 1.0]])


def test_opening_by_reconstruction_rebuilds_what_the_line_fits_in():
    b = np.zeros((40, 40), np.float32)
    b[10:15, 5:35] = 1.0  # a bar 5 px high and 30 px long
    assert np.array_equal(opening_by_reconstruction(b, line_footprint(11, 0)), b)  # fits along: rebuilt exactly
    assert not opening_by_reconstruction(b, line_footprint(11, 90)).any()  # does not fit across: removed


@pytest.mark.parametrize("seed", range(3))
def test_mbi_is_much_higher_on_bright_roofs_than_on_the_ground(seed):
    scene = make_scene(seed)
    value = mbi(brightness(scene.image), CROWDAI_SCALES)
    roofs, ground = scene.instance_map > 0, (scene.instance_map == 0) & ~scene.gap_mask
    assert value[roofs].mean() > 10 * value[ground].mean()
    assert value.dtype == np.float32 and np.isfinite(value).all() and value.min() >= 0


@pytest.mark.parametrize("mode", ["max", "contrast"])
@pytest.mark.parametrize("seed", range(3))
def test_gap_map_lights_up_the_dark_lines_between_roofs(seed, mode):
    scene = make_scene(seed)
    value = gap_map(brightness(scene.image), 9, mode)
    gaps, roofs = scene.gap_mask, scene.instance_map > 0
    assert value[gaps].mean() > 10 * value[roofs].mean()
    assert value[gaps].mean() > 10 * value[~gaps & ~roofs].mean()


def test_contrast_mode_keeps_thin_dark_lines_and_cancels_round_dark_spots():
    b = np.ones((60, 60), np.float32)
    b[10:50, 20] = 0.2  # thin vertical dark line: only lines across it fill it
    rows, cols = disk((45, 45), 2)
    b[rows, cols] = 0.2  # round dark spot: every direction fills it
    peak, contrast = gap_map(b, 9, "max"), gap_map(b, 9, "contrast")
    assert contrast[15:45, 20].min() == pytest.approx(0.8) and peak[15:45, 20].min() == pytest.approx(0.8)
    assert peak[45, 45] == pytest.approx(0.8) and contrast[45, 45] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="gap mode"):
        gap_map(b, 9, "mean")


def test_prior_png_keeps_mbi_in_red_and_the_gap_map_in_green(tmp_path):
    prior = np.zeros((4, 5, 2), np.uint8)
    prior[..., 0], prior[..., 1] = 200, 50
    cache.write_prior(tmp_path / "p.png", prior)
    assert np.array_equal(cache.read_prior(tmp_path / "p.png"), prior)
    with Image.open(tmp_path / "p.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    assert (rgb[..., 0] == 200).all() and (rgb[..., 1] == 50).all() and not rgb[..., 2].any()


def test_scale_and_normalisation_statistics():
    samples = np.random.default_rng(0).gamma(2.0, 0.05, (100_000, 2)).astype(np.float32)
    stats = cache.statistics(samples)
    np.testing.assert_allclose(stats["q"], np.percentile(samples, 99.5, axis=0))
    scaled = cache.quantize(samples, stats["q"])
    assert (scaled == 255).mean() == pytest.approx(0.005, abs=0.001)  # the top 0.5% saturate
    normalised = cache.normalize(scaled.reshape(1000, 100, 2), stats)
    np.testing.assert_allclose(normalised.reshape(-1, 2).mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(normalised.reshape(-1, 2).std(axis=0), 1.0, atol=1e-4)


def test_compute_priors_stacks_mbi_and_gap():
    scene = make_scene(0)
    priors = {"mbi_scales_px": CROWDAI_SCALES, "gap_se_length_px": 9, "gap_mode": "contrast"}
    raw = cache.compute_priors(scene.image, priors)
    b = brightness(scene.image)
    assert raw.shape == (300, 300, 2) and raw.dtype == np.float32
    np.testing.assert_allclose(raw[..., 0], mbi(b, CROWDAI_SCALES))
    np.testing.assert_allclose(raw[..., 1], gap_map(b, 9, "contrast"))
