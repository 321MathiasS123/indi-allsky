"""Fast metering must preserve patch areas and brightness exactly."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from indi_allsky.highlight import compensate, measure, measure_rendered, shadow_boost


def reference(data, mask, bits, threshold=99.0, rendered=False):
    """Original full-frame reductions and component labelling."""
    if mask is None or mask.shape != data.shape[:2]:
        return None
    valid = mask != 0
    count = np.count_nonzero(valid)
    if not count:
        return None
    lowest = data.min(axis=2) if data.ndim == 3 else data
    highest = data.max(axis=2) if data.ndim == 3 else data

    def largest(region):
        _, _, stats, _ = cv2.connectedComponentsWithStats(
            (region & valid).astype(np.uint8), connectivity=8)
        return float(stats[1:, cv2.CC_STAT_AREA].max()) * 100 / count if len(stats) > 1 else 0.0

    if rendered:
        return largest(lowest >= 240), largest(highest >= 250), 0.0, None, None
    mono = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY) if data.ndim == 3 else data
    adu = cv2.mean(mono, mask=valid.astype(np.uint8))[0] / (1 << (bits - 8))
    cutoff = ((1 << bits) - 1) * threshold / 100.0
    return (largest(lowest >= cutoff), largest(highest >= cutoff), adu,
            largest(lowest >= cutoff / 1.1), largest(highest >= cutoff / 1.1))


@pytest.mark.parametrize('dtype,bits', [(np.uint8, 8), (np.uint16, 12),
                                        (np.uint16, 16), (np.float32, 16)])
@pytest.mark.parametrize('channels', [None, 3, 4])
@pytest.mark.parametrize('view', ['ordinary', 'strided'])
def test_meter_matches_original_with_noncontiguous_inputs_and_masks(dtype, bits, channels, view):
    rng = np.random.default_rng(791)
    shape = (43, 59) + ((channels,) if channels else ())
    data = rng.integers(0, 1 << bits, shape).astype(dtype)
    data[4:11, 7:18] = (1 << bits) - 1
    mask = rng.choice(np.array([0, 1, 128, 255], np.uint8), (43, 59))
    mask[4:11, 7:18] = 128
    if view == 'strided':
        data, mask = data[::-1, ::2], mask[::-1, ::2]
    data.flags.writeable = False
    mask.flags.writeable = False
    for threshold in (0, 87.5, 99, 100):
        np.testing.assert_array_equal(measure(data, mask, bits, threshold),
                                      reference(data, mask, bits, threshold))
    np.testing.assert_array_equal(measure_rendered(data, mask),
                                  reference(data, mask, bits, rendered=True))


@pytest.mark.parametrize('channels', [None, 3, 4])
def test_float_nan_and_infinity_keep_original_metering_semantics(channels):
    shape = (13, 17) + ((channels,) if channels else ())
    data = np.full(shape, 65535, np.float32)
    data[1, 1] = np.nan
    data[3, 2] = np.inf
    data[6, 4] = -np.inf
    data[8, 3] = -0.0
    if channels:
        data[2, 4, 0] = np.nan
        data[5, 6, 1] = np.nan
        data[7, 8, 2] = np.nan
    mask = np.ones((13, 17), np.uint8)
    np.testing.assert_array_equal(measure(data, mask, 16), reference(data, mask, 16))
    np.testing.assert_array_equal(measure_rendered(data, mask),
                                  reference(data, mask, 16, rendered=True))


def test_no_components_are_labelled_when_bright_pixels_are_only_outside_mask():
    data = np.full((31, 47, 3), 65535, np.uint16)
    data[5:17, 8:24] = 100
    mask = np.zeros(data.shape[:2], np.uint8)
    mask[5:17, 8:24] = 255
    expected = reference(data, mask, 16)
    with patch('indi_allsky.highlight.cv2.connectedComponentsWithStats',
               side_effect=AssertionError('Empty clipping masks need no labels')):
        assert measure(data, mask, 16) == expected
        assert measure_rendered(data, mask).full == 0


def test_diagonal_and_disconnected_patches_keep_eight_connected_area():
    data = np.zeros((25, 25, 3), np.uint16)
    for i in range(10):
        data[i, i] = 65535
    data[20:22, 20:22] = 65535
    mask = np.ones((25, 25), np.uint8)
    assert measure(data, mask, 16).full == 10 * 100 / 625
    assert measure(data, mask, 16) == reference(data, mask, 16)


def original_lift(data, bits, adu, target, max_boost):
    boost = shadow_boost(adu, target, max_boost)
    if boost == 1.0:
        return data
    maximum = (1 << bits) - 1
    values = np.arange(maximum + 1, dtype=np.float32) / maximum
    knee = 0.5 / boost
    distance = np.maximum(values - knee, 0)
    curve = np.where(values <= knee, values * boost,
                     0.5 + boost * distance / (1 + (2 * boost - 1 / (1 - knee)) * distance))
    scale = np.divide(curve, values, out=np.ones_like(values), where=values > 0)
    peak = data.max(axis=2) if data.ndim == 3 else data
    multiplier = scale[np.minimum(peak, maximum)]
    if data.ndim == 3:
        multiplier = multiplier[:, :, None]
    return np.clip(np.rint(data * multiplier), 0, maximum).astype(data.dtype)


@pytest.mark.parametrize('dtype,bits', [(np.uint8, 8), (np.uint16, 12),
                                        (np.uint16, 16), (np.uint32, 16)])
@pytest.mark.parametrize('channels', [None, 1, 3, 4])
@pytest.mark.parametrize('max_boost', [0, 0.1, 1, 2, 4])
def test_lift_keeps_exact_pixel_rounding_without_modifying_source(dtype, bits, channels, max_boost):
    rng = np.random.default_rng(607)
    shape = (31, 47) + ((channels,) if channels else ())
    # For 12-bit sources also exercise values above the declared sensor range.
    data = rng.integers(0, np.iinfo(dtype).max, shape).astype(dtype)[::-1, ::2]
    data[0, 0] = 0
    data[0, 1] = (1 << bits) - 1
    data.flags.writeable = False
    output = compensate(data, bits, 17.25, 80, max_boost)
    np.testing.assert_array_equal(output, original_lift(data, bits, 17.25, 80, max_boost))
    assert output.dtype == data.dtype
    if max_boost == 0:
        assert output is data
