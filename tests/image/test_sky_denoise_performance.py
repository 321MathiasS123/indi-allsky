"""Processing shortcuts must retain the complete reference image calculation."""
import cv2
import numpy as np
import pytest

from indi_allsky import sky_denoise


@pytest.fixture
def textured_sky():
    height, width = 385, 401
    yy, xx = np.ogrid[:height, :width]
    rng = np.random.default_rng(61016)
    image = rng.normal(0.15, 0.012, (height, width, 3)).astype(np.float32)
    image[:, :, 0] += 0.03 * np.sin(xx / 24)
    image[:, :, 1] += 0.04 * np.cos(yy / 37)
    for x, y, size in ((130, 140, 1.2), (270, 205, 2.1), (205, 270, 3.0)):
        source = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * size ** 2))
        image += (0.25 * source[:, :, None]).astype(np.float32)
    image[190:192, 190:192] -= 0.08
    valid = sky_denoise._sky_mask(image.shape, (200, 192, 175))
    return image, valid


def _reference_linear(image, valid, strength, star_protection):
    """Keep the full helper path and original nonmutating expression order."""
    evidence = sky_denoise._source_evidence(image, valid, star_protection)
    evidence['highpass'] = sky_denoise._original_highpass(evidence)
    stronger = max(0, strength - 3)
    filtered, sigma = sky_denoise._nlm_clean(image, evidence, 1 + 0.15 * stronger)
    if sigma <= 1e-8:
        return image.copy()
    result = sky_denoise._blend(image, filtered, evidence, 1)
    result, _, _ = sky_denoise._repair_compact_lows(result, valid)
    prepared = sky_denoise._prepare_split(result, evidence, valid)
    result = sky_denoise._apply(result, prepared, colour=1,
                                brightness=0.5 + 0.125 * stronger)
    candidate, _, _ = sky_denoise._refine_pits(result, evidence, valid,
                                              amount=0.85 + 0.05 * stronger)
    result, _ = sky_denoise._protect_horizon(result, candidate, valid)
    if strength < 3:
        result = image + (result - image) * (0.5 if strength == 1 else 0.75)
    return np.clip(result, 0, 1)


@pytest.mark.parametrize('strength', [1, 3, 5])
@pytest.mark.parametrize('star_protection', [0, 0.5, 1])
def test_processing_shortcuts_match_full_helper_path_exactly(
        textured_sky, strength, star_protection):
    image, valid = textured_sky
    original = image.copy()
    expected = _reference_linear(image, valid, strength, star_protection)
    actual = sky_denoise._denoise_linear(image, valid, strength, star_protection)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('dtype,mono', [
    (np.uint8, False), (np.uint16, False), (np.float32, False),
    (np.float64, False), (np.uint16, True),
])
def test_owned_output_conversion_matches_original_conversion(textured_sky, dtype, mono):
    source, _ = textured_sky
    if mono:
        source = source[:, :, 0]
    integer = np.issubdtype(dtype, np.integer)
    maximum = float(np.iinfo(dtype).max) if integer else 1.0
    image = (source * maximum).astype(dtype)
    original = image.copy()
    config = {'LENS_IMAGE_CIRCLE': 350}
    valid = sky_denoise._sky_mask(image.shape, sky_denoise._sky_geometry(image.shape, config, 1))
    linear = image.astype(np.float32) / maximum
    if mono:
        linear = np.repeat(linear[:, :, None], 3, axis=2)
    expected = _reference_linear(linear, valid, 3, 1)
    if mono:
        expected = expected[:, :, 0]
    expected = np.clip(expected * maximum, 0, maximum)
    if integer:
        expected = np.rint(expected)
    expected = expected.astype(dtype)
    actual = sky_denoise.denoise(image, config, strength=3)
    assert actual.dtype == dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(image, original)


def test_daytime_shortcut_retains_edge_guard_and_catalogue_reset(textured_sky):
    class Catalogue:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    image, valid = textured_sky
    catalogue = Catalogue()
    full = sky_denoise._source_evidence(image, valid, 0)
    fast = sky_denoise._source_evidence(image, valid, 0, catalogue=catalogue,
                                       processing_only=True)
    np.testing.assert_array_equal(fast['lum'], full['lum'])
    np.testing.assert_array_equal(fast['mask'], full['mask'])
    assert catalogue.resets == 1
    assert 'points' in full and 'noise' in full
    assert 'points' not in fast and 'noise' not in fast


def test_optional_work_buffers_and_diagnostics_preserve_helper_defaults(textured_sky):
    image, valid = textured_sky
    original = image.copy()
    evidence = sky_denoise._source_evidence(image, valid, processing_only=True)
    np.testing.assert_array_equal(evidence['blur5'], cv2.GaussianBlur(evidence['lum'], (0, 0), 5))
    prepared = sky_denoise._prepare_split(image, evidence, valid, return_distance=True)
    np.testing.assert_array_equal(prepared['distance'], cv2.distanceTransform(
        valid.astype(np.uint8), cv2.DIST_L2, 5))
    np.testing.assert_array_equal(sky_denoise._apply(image, prepared, inplace=True),
                                  sky_denoise._apply(image, prepared))
    filtered = cv2.GaussianBlur(image, (0, 0), 1)
    filtered_before = filtered.copy()
    expected = sky_denoise._blend(image, filtered, evidence, 1)
    np.testing.assert_array_equal(filtered, filtered_before)
    actual = sky_denoise._blend(image, filtered, evidence, 1, inplace=True)
    assert actual is filtered
    np.testing.assert_array_equal(actual, expected)
    for repair in (sky_denoise._repair_compact_lows, sky_denoise._refine_pits):
        args = (image, valid) if repair is sky_denoise._repair_compact_lows else (image, evidence, valid)
        full = repair(*args)
        fast = repair(*args, diagnostics=False)
        np.testing.assert_array_equal(fast[0], full[0])
        assert isinstance(full[2], dict) and fast[2] is None
        if repair is sky_denoise._refine_pits:
            assert fast[1] is None
    candidate = image + np.float32(0.01)
    candidate_before = candidate.copy()
    expected, weight = sky_denoise._protect_horizon(image, candidate, valid)
    np.testing.assert_array_equal(candidate, candidate_before)
    actual, cached_weight = sky_denoise._protect_horizon(
        image, candidate, valid, distance=prepared['distance'], inplace=True)
    assert actual is candidate
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cached_weight, weight)
    np.testing.assert_array_equal(image, original)


def test_preparation_owned_buffers_preserve_original_expressions_and_shared_cache(textured_sky):
    image, valid = textured_sky
    evidence = sky_denoise._source_evidence(image, valid)
    shared = {}
    sky_denoise._prepare(image, evidence, valid, shared=shared, channel=0)
    roi = shared['roi']
    expected_background = (sky_denoise._blur(shared['weighted_image'], 2.5)
                           / np.maximum(shared['norm'][:, :, None], 0.0001))
    np.testing.assert_array_equal(shared['background'], expected_background)
    before = {key: value.copy() for key, value in shared.items() if isinstance(value, np.ndarray)}
    small_norm = sky_denoise._blur(shared['weight'], 1.2)
    small = (sky_denoise._blur(shared['weighted_image'], 1.2)
             / np.maximum(small_norm[:, :, None], 0.0001))
    mix = np.clip((small_norm - 0.1) / 0.5, 0, 1)
    expected_background = expected_background + mix[:, :, None] * (small - expected_background)
    band = np.zeros_like(image)
    band[roi] = expected_background - sky_denoise._blur(expected_background, 7)
    expected_lum = band @ sky_denoise.WEIGHTS
    actual = sky_denoise._prepare(image, evidence, valid, fine=1.2, shared=shared, channel=1)
    np.testing.assert_array_equal(actual['lum'], expected_lum)
    np.testing.assert_array_equal(actual['chroma'], band - expected_lum[:, :, None])
    for key, value in before.items():
        np.testing.assert_array_equal(shared[key], value)


def test_apply_owned_buffers_preserve_original_expression_order(textured_sky):
    image, valid = textured_sky
    original = image.copy()
    evidence = sky_denoise._source_evidence(image, valid)
    prepared = sky_denoise._prepare_split(image, evidence, valid)
    before = {key: value.copy() for key, value in prepared.items()}
    l, c, t = prepared['lum'], prepared['chroma'], prepared['trust']
    nl, nc = prepared['noise']
    wl = np.minimum(1, (1.5 * nl) ** 2 / (sky_denoise._blur(l * l, 3) + 1e-12))
    wc = np.minimum(1, (1.5 * nc) ** 2 / (sky_denoise._blur(np.mean(c * c, axis=2), 3) + 1e-12))
    dl = sky_denoise._blur(l * wl * t, 1) * 0.6 * t
    dc = sky_denoise._blur(c * (wc * t)[:, :, None], 1) * 0.85 * t[:, :, None]
    correction = dl[:, :, None] + dc
    sky = t > 0.99
    assert np.any(sky) and np.all(prepared['noise'] > 0)
    correction -= correction[sky].mean(axis=0)[None, None, :] * t[:, :, None]
    expected = image - correction
    for inplace in (False, True):
        actual = sky_denoise._apply(image, prepared, inplace=inplace)
        np.testing.assert_array_equal(actual, expected)
    for key, value in before.items():
        np.testing.assert_array_equal(prepared[key], value)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('cpu_count', [1, 4])
@pytest.mark.parametrize('geometry', [(336, 320, 310), (135, 125, 520)])
def test_parallel_annular_filters_preserve_pixels_and_statistics_exactly(
        monkeypatch, cpu_count, geometry):
    height, width = 641, 673
    yy, xx = np.ogrid[:height, :width]
    rng = np.random.default_rng(16017)
    raw = rng.normal(0.2, 0.009, (height, width, 3)).astype(np.float32)
    raw[:, :, 0] += 0.015 * np.sin(xx / 45)
    raw[:, :, 1] += 0.01 * np.cos(yy / 37)
    image = cv2.GaussianBlur(raw, (0, 0), 1.4)
    for x, y in ((320, 310), (7, 330), (560, 375), (330, 620)):
        pit = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5).astype(np.float32)
        image -= 0.025 * pit[:, :, None]
        source = np.exp(-((xx - x - 9) ** 2 + (yy - y - 5) ** 2) / 2).astype(np.float32)
        raw += 0.15 * source[:, :, None]
        image += 0.15 * source[:, :, None]
    valid = sky_denoise._sky_mask(raw.shape, geometry)
    assert valid[sky_denoise._sky_slice(valid, 12)].size >= 512 * 512
    evidence = {'lum': raw @ sky_denoise.WEIGHTS}
    original = image.copy()
    pool_calls = []
    executor = sky_denoise.ThreadPoolExecutor

    def tracked_pool(*args, **kwargs):
        pool_calls.append(kwargs['max_workers'])
        return executor(*args, **kwargs)

    monkeypatch.setattr(sky_denoise.os, 'cpu_count', lambda: cpu_count)
    monkeypatch.setattr(sky_denoise, 'ThreadPoolExecutor', tracked_pool)
    expected = sky_denoise._refine_pits(image, evidence, valid)
    serial_pools = len(pool_calls)
    pool_calls.clear()
    actual = sky_denoise._refine_pits(image, evidence, valid, parallel=True)
    assert len(pool_calls) == serial_pools + (cpu_count > 1)
    if cpu_count == 1:
        assert not pool_calls
    assert expected[2]['pixels'] > 0
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2] == expected[2]
    np.testing.assert_array_equal(image, original)
