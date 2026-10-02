from copy import deepcopy
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import textwrap

import numpy as np
import pytest

from indi_allsky.highlight import compensate, measure
from indi_allsky.stretch.mode2_mtf import IndiAllSky_Mode2_MTF_Stretch
from indi_allsky.stretch.mode2_mtf import IndiAllSky_Mode2_MTF_Stretch_x2
from indi_allsky.stretch.mode3_adaptive_mtf import IndiAllSky_Mode3_Adaptive_MTF_Stretch


@pytest.mark.parametrize('bits', [8, 10, 12, 14, 16])
def test_largest_patches_use_mask_area_and_include_white_in_any(bits):
    maximum = (1 << bits) - 1
    data = np.zeros((100, 100, 3), dtype=np.uint8 if bits == 8 else np.uint16)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:50] = 255  # 5,000 valid pixels, not 10,000
    data[0:10, 0:10] = maximum  # 100 white pixels
    data[10:20, 0:10, 0] = maximum  # 100 adjoining blue pixels
    data[30:35, 30:35] = maximum  # separate reflection, ignored
    data[60:] = maximum  # excluded from both numerator and denominator
    result = measure(data, mask, bits)
    assert result.full == 2
    assert result.any == 4
    assert result.adu > 0


def test_diagonal_connectivity_and_monochrome():
    data = np.eye(10, dtype=np.uint16) * 65535
    result = measure(data, np.ones((10, 10), dtype=np.uint8), 16)
    assert result.full == result.any == 10


def test_invalid_or_empty_mask_does_not_invent_a_percentage():
    data = np.zeros((5, 5, 3), dtype=np.uint16)
    for mask in [None, np.zeros((5, 5), dtype=np.uint8), np.ones((4, 5), dtype=np.uint8)]:
        assert measure(data, mask, 16) is None


def test_threshold_handles_calibration_offsets_without_jpeg_metering():
    data = np.full((10, 10, 3), 65000, dtype=np.uint16)
    mask = np.ones((10, 10), dtype=np.uint8)
    assert measure(data, mask, 16, 99).full == 100
    assert measure(data, mask, 16, 100).full == 0


@pytest.mark.parametrize('bits', [8, 12, 16])
@pytest.mark.parametrize('stops', [0, 1, 2, 4])
def test_lift_is_monotonic_bounded_and_does_not_create_a_white_plateau(bits, stops):
    maximum = (1 << bits) - 1
    data = np.arange(maximum + 1, dtype=np.uint8 if bits == 8 else np.uint16)[None, :]
    original = data.copy()
    output = compensate(data, bits, 10, 80, stops)
    np.testing.assert_array_equal(data, original)
    assert output.dtype == data.dtype
    assert output[0, 0] == 0 and output[0, -1] == maximum
    assert np.all(np.diff(output.astype(np.int32)) >= 0)
    assert np.all(output >= data)
    assert np.all(output <= np.minimum(data.astype(np.float64) * 2 ** stops + 1, maximum))
    # Quantisation can merge the last few codes, but not a broad highlight area.
    assert np.count_nonzero(output == maximum) / output.size < 0.07
    if stops == 0:
        assert output is data


def test_shadow_lift_preserves_channel_ratios_and_leaves_source_untouched():
    data = np.array([[[1000, 2000, 3000], [10000, 20000, 30000]]], dtype=np.uint16)
    output = compensate(data, 16, 20, 80, 2)
    np.testing.assert_allclose(output / output[:, :, 2:3], data / data[:, :, 2:3], atol=0.0001)
    np.testing.assert_array_equal(output[0, 0], [4000, 8000, 12000])
    assert data[0, 0, 0] == 1000


def test_compensation_rescues_shadows_before_existing_mtf_black_cutoff():
    config = {'IMAGE_STRETCH': {'MODE2_SHADOWS': 0.03, 'MODE2_MIDTONES': 0.4}}
    data = np.array([[1440, 2000, 4000, 65000]], dtype=np.uint16)
    normal = IndiAllSky_Mode2_MTF_Stretch(config).stretch(data, 16, 1)
    lifted = IndiAllSky_Mode2_MTF_Stretch(config).stretch(compensate(data, 16, 20, 80, 2), 16, 1)
    assert normal[0, 0] == 0
    assert lifted[0, 0] > 0
    assert lifted[0, 2] < lifted[0, 3]


@pytest.mark.parametrize('stretch_class', [IndiAllSky_Mode2_MTF_Stretch,
                                          IndiAllSky_Mode2_MTF_Stretch_x2,
                                          IndiAllSky_Mode3_Adaptive_MTF_Stretch])
def test_compensation_keeps_stretch_settings_and_normal_brightness_path(stretch_class):
    config = {'IMAGE_STRETCH': {'MODE2_SHADOWS': 0.03, 'MODE2_MIDTONES': 0.4,
                              'MODE3_MIDTONES': 0.25}}
    before = deepcopy(config)
    data = np.linspace(2000, 65535, 30000).reshape(100, 100, 3).astype(np.uint16)
    expected = stretch_class(config).stretch(data, 16, 1)
    actual = stretch_class(config).stretch(compensate(data, 16, 80, 80, 2), 16, 1)
    np.testing.assert_array_equal(actual, expected)
    lifted = stretch_class(config).stretch(compensate(data // 4, 16, 20, 80, 2), 16, 1)
    assert lifted.shape == data.shape and lifted.dtype == np.uint16
    assert config == before


@pytest.mark.parametrize('enabled,excluded,focus,active', [
    (False, False, False, False), (True, False, False, True),
    (True, True, False, False), (True, False, True, False),
])
def test_worker_routes_measurement_and_processing_without_touching_off_path(enabled, excluded, focus, active):
    # Execute the actual processing/control segment with camera and DB services
    # replaced by spies. This catches integration order and invalid-frame leaks.
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    segment = textwrap.dedent(source[source.index('        # Calculate ADU before stretch'):
                                    source.index('        # generate a new mask base')])
    events = []
    processor = SimpleNamespace(
        focus_mode=focus,
        calculate_8bit_adu=lambda: 20,
        measure_highlights=lambda: events.append('measure') or (1, 2, 20),
        denoise=lambda: events.append('denoise'),
        compensate_highlights=lambda adu: events.append('compensate'),
        stretch=lambda: events.append('stretch'),
        convert_16bit_to_8bit=lambda: events.append('convert'),
    )
    config = {'HIGHLIGHT_PROTECTION': {'ENABLE': enabled}}
    original_config = deepcopy(config)
    controller = SimpleNamespace(hist_adu=[], compare_highlights=Mock(return_value=(20, 20)),
                                 compare_exposure=Mock(return_value=(20, 20)))
    namespace = dict(self=SimpleNamespace(config=config, image_processor=processor, exposure_o=controller),
                     i_ref=SimpleNamespace(asi676mc_repair_result=None), exposure=0.01, gain=0,
                     asi676mc=SimpleNamespace(excluded_from_downstream_measurements=lambda result: excluded),
                     logger=logging.getLogger(__name__))
    exec(segment, namespace)
    assert events == (['measure', 'denoise', 'compensate', 'stretch', 'convert'] if active
                      else ['denoise', 'stretch', 'convert'])
    assert controller.compare_highlights.call_count == int(active)
    assert controller.compare_exposure.call_count == int(not excluded and not active)
    assert config == original_config


def test_processor_meters_current_capture_not_stack_and_preserves_raw_storage():
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('measure_highlights', 'compensate_highlights')]
    from indi_allsky import constants
    namespace = dict(__package__='indi_allsky', constants=constants)
    exec(compile(ast.Module(body=methods, type_ignores=[]), 'processing-highlight-methods', 'exec'), namespace)
    raw = np.zeros((10, 10, 3), dtype=np.uint16)
    raw[:2, :2] = 65000
    original = raw.copy()
    reference = SimpleNamespace(opencv_data=raw, binning=2, image_bitpix=16)
    processor = SimpleNamespace(getLatestImage=lambda: reference, max_bit_depth=16,
                                _adu_mask_dict={2: np.ones((10, 10), dtype=np.uint8)},
                                image=np.full_like(raw, 2000), night_av=[False, False],
                                config={'TARGET_ADU_DAY': 80, 'HIGHLIGHT_PROTECTION': {'MAX_BOOST': 2}})
    measured = namespace['measure_highlights'](processor)
    assert measured.full == measured.any == 4
    namespace['compensate_highlights'](processor, 20)
    assert np.all(processor.image == 8000)
    np.testing.assert_array_equal(raw, original)
    processor.config['HIGHLIGHT_PROTECTION']['THRESHOLD'] = 100
    assert namespace['measure_highlights'](processor).full == 0


@pytest.mark.parametrize('bits', [8, 12, 16])
def test_compensation_uses_fractional_adu_without_changing_disabled_meter(bits):
    import cv2
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_calculate_8bit_adu')
    namespace = dict(cv2=cv2, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'processing-adu', 'exec'), namespace)
    processor = SimpleNamespace(_adu_mask_dict={1: np.ones((100, 100), dtype=np.uint8)},
                                max_bit_depth=bits, config={'HIGHLIGHT_PROTECTION': {'ENABLE': True}})
    reference = SimpleNamespace(binning=1, image_bitpix=8 if bits == 8 else 16)
    outputs = []
    for level in (20.99, 21.01):
        # Fractional means exist even with an 8-bit camera.
        values = np.full(10000, int(level * 2 ** (bits - 8)), dtype=np.uint8 if bits == 8 else np.uint16)
        values[:round((level * 2 ** (bits - 8) % 1) * values.size)] += 1
        processor.image = values.reshape(100, 100)
        processor.config['HIGHLIGHT_PROTECTION']['ENABLE'] = True
        measured = namespace['_calculate_8bit_adu'](processor, reference)
        assert measured == pytest.approx(level, abs=0.0001)
        outputs.append(compensate(processor.image, bits, measured, 80, 2).mean() / 2 ** (bits - 8))
        processor.config['HIGHLIGHT_PROTECTION']['ENABLE'] = False
        assert namespace['_calculate_8bit_adu'](processor, reference) == int(level)
    assert abs(outputs[1] - outputs[0]) < 0.1


@pytest.mark.parametrize('method', ['average', 'maximum', 'minimum'])
def test_stack_exposure_transition_keeps_shadows_steady_and_sources_untouched(method):
    from indi_allsky.stack import IndiAllskyStacker
    stacker = IndiAllskyStacker({}, {1: None})
    scene = np.linspace(40 * 256, 120 * 256, 10000).reshape(100, 100)
    frames = []
    for exposure in [1, 1, 1, 1, 0.25, 0.25, 0.25, 0.25]:
        frames.append((scene * exposure).astype(np.uint16))
        source_copies = [data.copy() for data in frames[-4:]]
        stacked = getattr(stacker, method)(frames[-4:], np.uint16)
        output = compensate(stacked, 16, stacked.mean() / 256, 80, 2)
        assert output.mean() / 256 == pytest.approx(80, abs=0.01)
        for data, original in zip(frames[-4:], source_copies):
            np.testing.assert_array_equal(data, original)
