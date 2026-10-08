"""Optimized sky filtering must retain the reference calculations and borders."""
import numpy as np
import pytest
from skimage.restoration import denoise_nl_means

from indi_allsky import sky_denoise


def _scene(height, width, noncontiguous=False):
    rng = np.random.default_rng(2718)
    storage = rng.normal(0.2, 0.025, (height, width * 2)).astype(np.float32)
    image = storage[:, ::2] if noncontiguous else storage[:, :width].copy()
    yy, xx = np.ogrid[:height, :width]
    # Uneven strips exercise both rounding directions when splitting the image.
    boundaries = [height // 4, height // 2, 3 * height // 4]
    for y in [1, *boundaries, height - 2]:
        for offset, amplitude in [(-2, 0.025), (0, 0.15), (2, 0.05)]:
            image += (amplitude * np.exp(
                -((yy - y - offset) ** 2 + (xx - width // 2) ** 2) / 2
            )).astype(np.float32)
    image[height // 2:, width // 3:] += 0.06
    image[:, :2] += 0.1
    image[:, -2:] -= 0.06
    return image


def _serial(image):
    return denoise_nl_means(
        image, h=0.025, patch_size=5, patch_distance=5, sigma=0,
        fast_mode=True, preserve_range=True, channel_axis=None,
    )


@pytest.mark.parametrize('shape', [(513, 137), (1053, 131)])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_parallel_nlm_matches_whole_image_at_seams_and_outer_edges(
        monkeypatch, shape, noncontiguous):
    monkeypatch.setattr(sky_denoise.os, 'cpu_count', lambda: 4)
    image = _scene(*shape, noncontiguous=noncontiguous)
    original = image.copy()
    expected = _serial(image)
    actual = sky_denoise._nlm_luminance(image, 0.025)

    assert actual.shape == image.shape and actual.dtype == image.dtype
    # Independent integral-image accumulation can differ by float32 roundoff.
    # The whole-array comparison includes faint sources, discontinuities,
    # internal strip seams and the original image's reflected outer borders.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('height,cpu_count', [(31, 4), (511, 4), (1053, 1), (513, None)])
def test_small_frames_and_single_cpu_use_serial_fallback(monkeypatch, height, cpu_count):
    monkeypatch.setattr(sky_denoise.os, 'cpu_count', lambda: cpu_count)

    def unexpected_pool(*args, **kwargs):
        pytest.fail('Serial filtering must not create a thread pool')

    monkeypatch.setattr(sky_denoise, 'ThreadPoolExecutor', unexpected_pool)
    image = _scene(height, 71)
    np.testing.assert_allclose(
        sky_denoise._nlm_luminance(image, 0.025), _serial(image), rtol=0, atol=1e-7,
    )


def test_shared_preparation_matches_independent_brightness_and_colour_scales():
    rng = np.random.default_rng(31415)
    image = rng.normal(0.2, 0.015, (517, 533, 3)).astype(np.float32)
    yy, xx = np.ogrid[:517, :533]
    image[:, :, 0] += (0.03 * np.sin(xx / 30)).astype(np.float32)
    valid = (xx - 266) ** 2 + (yy - 258) ** 2 < 240 ** 2
    evidence = sky_denoise._source_evidence(image, valid)
    original = image.copy()

    broad = sky_denoise._prepare(image, evidence, valid, fine=2.5)
    fine = sky_denoise._prepare(image, evidence, valid, fine=1.2)
    actual = sky_denoise._prepare_split(image, evidence, valid)

    # Enough valid 128-pixel tiles must reach the measured-statistics branch.
    assert np.all(broad['noise'] > 1e-6) and np.all(fine['noise'] > 1e-6)
    assert actual['noise'].dtype == broad['noise'].dtype
    for key, expected in [('lum', broad['lum']), ('trust', broad['trust']),
                          ('chroma', fine['chroma'])]:
        assert actual[key].dtype == expected.dtype
        np.testing.assert_array_equal(actual[key], expected)
    np.testing.assert_array_equal(actual['noise'], [broad['noise'][0], fine['noise'][1]])
    np.testing.assert_array_equal(image, original)


def test_flat_sky_has_no_pit_correction_or_empty_selection_failure():
    image = np.full((129, 137, 3), 0.2, np.float32)
    valid = np.ones(image.shape[:2], dtype=bool)
    evidence = sky_denoise._source_evidence(image, valid)
    original = image.copy()

    result, delta, stats = sky_denoise._refine_pits(image, evidence, valid)

    np.testing.assert_array_equal(delta, np.zeros(image.shape[:2], np.float32))
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(image, original)
    assert stats['pixels'] == 0 and stats['mean_linear_luma_added'] == 0
