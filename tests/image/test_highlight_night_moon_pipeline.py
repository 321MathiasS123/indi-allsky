"""Night Moon exceptions preserve sky safeguards and ordinary exposure policy."""
from copy import deepcopy
import math
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from indi_allsky import highlight, highlight_output, highlight_twilight
from tests.core.test_highlight_exposure import controller
from test_highlight_protection import highlight_processor
from test_highlight_twilight_output_pipeline import prepare, setup_moon


MOON = (75., 50., 6.5)


def night_frame(processor, monkeypatch, *, enabled=True, weight=1., outside=None):
    ref = prepare(processor, TWILIGHT_ENABLE=False, TWILIGHT_OUTPUT_ENABLE=False,
                  NIGHT_MOON_EXCLUSION_ENABLE=enabled)
    processor.night_av = [True, True]
    ref.exposure = 24.
    data = np.full((100, 100, 3), 7680, np.uint16)
    cv2.circle(data, (75, 50), 6, (65535, 65535, 65535), -1)
    if outside is not None:
        data[5:25, 5:25] = outside
    set_pixels(processor, ref, data)
    monkeypatch.setattr(highlight_output, 'night_moon_weight', lambda *args: weight)
    moon = Mock(return_value=MOON)
    monkeypatch.setattr(highlight_output, 'moon_region', moon)
    return ref, data, moon


def set_pixels(processor, ref, data):
    ref.hdulist[0].data = np.moveaxis(data[:, :, ::-1], -1, 0).copy()
    ref.opencv_data = data.copy()
    processor.image = (data >> 8).astype(np.uint8)


def calibrated(processor):
    return processor.calibrate_highlights(processor.measure_highlights())


def test_night_exception_masks_only_moon_and_preserves_calibrated_adu_and_inputs(
        highlight_processor, monkeypatch):
    p = highlight_processor
    ref, data, moon = night_frame(p, monkeypatch)
    original = (ref.hdulist[0].data.copy(), ref.opencv_data.copy(),
                p.image.copy(), p._adu_mask_dict[1].copy())
    config_before = deepcopy(p.config)
    raw_meter = Mock(wraps=highlight.measure)
    output_meter = Mock(wraps=highlight.measure_rendered)
    monkeypatch.setattr(highlight, 'measure', raw_meter)
    monkeypatch.setattr(highlight, 'measure_rendered', output_meter)

    protected = calibrated(p)
    rendered = p.measure_output_highlights()

    assert protected.full == protected.any == 0
    assert rendered.full == rendered.any == 0
    expected_adu = cv2.mean(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY),
                            mask=original[3])[0] / 256
    assert protected.adu == pytest.approx(expected_adu)
    expected_mask = original[3].copy()
    cv2.circle(expected_mask, (75, 50), 7, 0, -1)
    np.testing.assert_array_equal(raw_meter.call_args.args[1] != 0, expected_mask != 0)
    np.testing.assert_array_equal(output_meter.call_args.args[1] != 0, expected_mask != 0)
    assert moon.call_args_list[0].kwargs == {'native': True}
    np.testing.assert_array_equal(ref.hdulist[0].data, original[0])
    np.testing.assert_array_equal(ref.opencv_data, original[1])
    np.testing.assert_array_equal(p.image, original[2])
    np.testing.assert_array_equal(p._adu_mask_dict[1], original[3])
    np.testing.assert_array_equal(p._highlight_output_mask != 0, original[3] != 0)
    assert p.config == config_before


@pytest.mark.parametrize('outside', [(65535, 65535, 65535), (65535, 7680, 7680)])
def test_bright_patch_elsewhere_still_shortens_exposure(highlight_processor, monkeypatch, outside):
    p = highlight_processor
    _, _, _ = night_frame(p, monkeypatch, outside=outside)
    measured = calibrated(p)
    rendered = p.measure_output_highlights()
    assert measured.any > 4
    assert rendered.any > 4
    instance = controller('exposure_autogain_exp_prio_db_1_10', moon=True)
    utils = instance._expUtils
    utils.EXPOSURE_MIN_NIGHT = .001
    utils.EXPOSURE_MAX = 45.
    utils.EXPOSURE_NEXT = utils.EXPOSURE_CURRENT = 24.
    utils.GAIN_NEXT = utils.GAIN_CURRENT = 0.

    instance.compare_highlights(measured, 24., 0.)

    assert utils.EXPOSURE_NEXT < 24.
    assert utils.EXPOSURE_NEXT >= utils.EXPOSURE_MIN_NIGHT


def test_default_enabled_and_explicit_disable_restore_exact_old_metrics(highlight_processor, monkeypatch):
    p = highlight_processor
    ref, data, _ = night_frame(p, monkeypatch)
    enabled = calibrated(p), p.measure_output_highlights()
    p.config['HIGHLIGHT_PROTECTION'].pop('NIGHT_MOON_EXCLUSION_ENABLE')
    assert (calibrated(p), p.measure_output_highlights()) == enabled
    p.config['HIGHLIGHT_PROTECTION']['NIGHT_MOON_EXCLUSION_ENABLE'] = False
    old = calibrated(p)
    expected = highlight.measure(data, p._adu_mask_dict[1], 16)
    assert old == expected
    assert old.full > 1 and old.any > 1
    assert p.measure_output_highlights() == highlight.measure_rendered(p.image, p._adu_mask_dict[1])
    assert p._highlight_twilight_mask is None
    assert ref.exposure == 24.


def test_disabled_parent_does_no_optional_moon_work(highlight_processor, monkeypatch):
    p = highlight_processor
    _, data, _ = night_frame(p, monkeypatch)
    p.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    unexpected = Mock(side_effect=AssertionError('Moon exclusion while parent is disabled'))
    monkeypatch.setattr(highlight_output, 'night_moon_weight', unexpected)
    monkeypatch.setattr(highlight_output, 'moon_region', unexpected)

    assert calibrated(p) == highlight.measure(data, p._adu_mask_dict[1], 16)
    assert p.measure_output_highlights() == highlight.measure_rendered(p.image, p._adu_mask_dict[1])
    unexpected.assert_not_called()


@pytest.mark.parametrize('twilight', [False, True])
def test_day_and_existing_twilight_paths_are_identical_with_night_option_toggled(
        highlight_processor, monkeypatch, twilight):
    p = highlight_processor
    prepare(p)
    setup_moon(monkeypatch, weight=1. if twilight else 0.)
    monkeypatch.setattr(highlight_output, 'night_moon_weight', lambda *args: 0.)
    p.config['HIGHLIGHT_PROTECTION']['NIGHT_MOON_EXCLUSION_ENABLE'] = False
    before = calibrated(p), p.measure_output_highlights()
    p.config['HIGHLIGHT_PROTECTION']['NIGHT_MOON_EXCLUSION_ENABLE'] = True
    assert (calibrated(p), p.measure_output_highlights()) == before


@pytest.mark.parametrize('region', [None, (50., 50., 100.)])
def test_missing_or_unusable_projection_retains_all_highlight_safeguards(
        highlight_processor, monkeypatch, region):
    p = highlight_processor
    _, data, _ = night_frame(p, monkeypatch)
    monkeypatch.setattr(highlight_output, 'moon_region', lambda *args, **kwargs: region)
    before = p._adu_mask_dict[1].copy()

    assert calibrated(p) == highlight.measure(data, before, 16)
    assert p.measure_output_highlights() == highlight.measure_rendered(p.image, before)
    np.testing.assert_array_equal(p._adu_mask_dict[1], before)


def test_partial_night_fade_changes_only_the_small_moon_exception(highlight_processor, monkeypatch):
    p = highlight_processor
    _, _, _ = night_frame(p, monkeypatch)
    original_mask = p._adu_mask_dict[1].copy()
    metrics = []
    for weight in (0., .5, 1.):
        monkeypatch.setattr(highlight_output, 'night_moon_weight', lambda *args, value=weight: value)
        metrics.append((calibrated(p), p.measure_output_highlights()))
    assert metrics[0][0].full > metrics[1][0].full > metrics[2][0].full == 0
    assert metrics[0][1].full > metrics[1][1].full > metrics[2][1].full == 0
    assert len({row[0].adu for row in metrics}) == 1
    np.testing.assert_array_equal(p._adu_mask_dict[1], original_mask)


@pytest.mark.parametrize('maximum', [12., 30., 45.])
def test_exposure_priority_can_reach_configured_maximum_without_a_forced_floor(
        highlight_processor, monkeypatch, maximum):
    p = highlight_processor
    ref, _, _ = night_frame(p, monkeypatch)
    instance = controller('exposure_autogain_exp_prio_db_1_10', moon=True)
    instance.config['CCD_EXPOSURE_MAX'] = maximum
    utils = instance._expUtils
    utils.EXPOSURE_MAX = maximum
    utils.GAIN_MIN_NIGHT = utils.GAIN_MAX_NIGHT = 0.
    original_limits = utils.EXPOSURE_MIN_NIGHT, utils.EXPOSURE_MAX
    exposure = min(24., maximum * .6)
    history = []
    for _ in range(40):
        # Acknowledge the camera's microsecond storage without introducing an
        # artificial pending command through binary float truncation.
        utils.EXPOSURE_NEXT = utils.EXPOSURE_CURRENT = math.nextafter(exposure, math.inf)
        exposure = utils.EXPOSURE_CURRENT
        utils.GAIN_NEXT = utils.GAIN_CURRENT = 0.
        background = round(7680 * exposure / 24.)
        data = np.full((100, 100, 3), background, np.uint16)
        cv2.circle(data, (75, 50), 6, (65535, 65535, 65535), -1)
        set_pixels(p, ref, data)
        ref.exposure = exposure
        measurement = calibrated(p)
        assert measurement.full == measurement.any == 0
        instance.compare_highlights(measurement, exposure, 0.)
        requested = utils.EXPOSURE_NEXT
        assert exposure <= requested <= maximum
        history.append(requested)
        exposure = requested
        if exposure == pytest.approx(maximum, abs=.000001):
            break
    assert history[-1] == pytest.approx(maximum, abs=.000001)
    assert (utils.EXPOSURE_MIN_NIGHT, utils.EXPOSURE_MAX) == original_limits
    assert utils.EXPOSURE_MIN_NIGHT < min(history)
    assert instance.config['CCD_EXPOSURE_MAX'] == maximum
