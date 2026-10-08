import copy
from copy import deepcopy
from datetime import datetime
import ast
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import textwrap

import cv2
import numpy as np
import pytest

from indi_allsky import asi676mc, constants
from indi_allsky.highlight import HighlightMeasurement, HighlightOutput, HighlightTransition, compensate, measure, measure_rendered
from indi_allsky.highlight_meter import apply_control_snapshot
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


def test_threshold_tolerates_near_full_scale_camera_values():
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


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('status', [None, 'normal', 'skipped', 'repaired', 'excluded', 'validation_failed'])
@pytest.mark.parametrize('focus', [False, True])
@pytest.mark.parametrize('fits_mode', ['off', 'pre_dark', 'post_dark'])
@pytest.mark.parametrize('meter_valid', [False, True])
@pytest.mark.parametrize('prepared', [False, True])
def test_worker_routes_measurement_and_processing_without_touching_off_path(enabled, status, focus, fits_mode, meter_valid, prepared, caplog):
    # Execute the actual processing/control segment with camera and DB services
    # replaced by spies. This catches integration order and invalid-frame leaks.
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    early = textwrap.dedent(source[source.index('        # Purple-frame handling deliberately'):
                                  source.index('        image_height, image_width = self.image_processor.image.shape')])
    late = textwrap.dedent(source[source.index('        # Calculate ADU before stretch'):
                                 source.index('        # generate a new mask base')])
    excluded = status in ('excluded', 'validation_failed')
    repaired = enabled and not focus and status == 'repaired'
    metered = enabled and not focus and not excluded and not repaired
    active = metered and meter_valid
    events = []
    reference = SimpleNamespace(asi676mc_repair_result=None, libcamera_black_level=0,
                                exp_date=datetime(2026, 10, 5, 7, 0, 14))

    def repair(ref):
        events.append('purple_check')
        ref.asi676mc_repair_result = {'status': status} if status else None

    processor = SimpleNamespace(
        focus_mode=focus,
        correct_asi676mc_frame=repair,
        calibrate=lambda **kwargs: events.append('dark'),
        fix_holes_early=lambda: events.append('holes'),
        calculateJankySqm=lambda: None,
        debayer=lambda: events.append('debayer'),
        stack=lambda: events.append('stack'),
        calculate_8bit_adu=lambda: 60,  # stack brightness differs from this capture
        measure_highlights=lambda: events.append('measure') or (HighlightMeasurement(1, 2, 200) if meter_valid else None),
        calibrate_highlights=lambda m: events.append('calibrated_adu') or m._replace(adu=20),
        denoise=lambda: events.append('denoise'),
        compensate_highlights=Mock(side_effect=lambda adu: events.append('compensate') or 1.25),
        stretch=lambda: events.append('stretch'),
        convert_16bit_to_8bit=lambda **kwargs: events.append('convert'),
    )
    config = {'HIGHLIGHT_PROTECTION': {'ENABLE': enabled}, 'IMAGE_SAVE_FITS': fits_mode != 'off',
              'IMAGE_SAVE_FITS_PRE_DARK': fits_mode == 'pre_dark'}
    original_config = deepcopy(config)
    controller = SimpleNamespace(hist_adu=[21, 23], highlight_transition=HighlightTransition(), highlight_output=HighlightOutput(),
                                 compare_highlights=Mock(side_effect=lambda *args: events.append('highlight_control') or (20, 20)),
                                 compare_exposure=Mock(side_effect=lambda *args: events.append('ordinary_control') or (60, 60)),
                                 reset_highlights=Mock())
    controller.highlight_transition.active = True
    controller.highlight_transition.reference = 70
    controller.highlight_transition.lift = .5
    controller.highlight_transition.gamma_mix = 1
    controller.highlight_transition.trusted = True
    worker = SimpleNamespace(config=config, image_processor=processor, exposure_o=controller,
                             image_count=0, capture_asi676mc_diagnostic_fits=Mock(),
                             start_image_save_pre_hook=Mock(), write_fit=lambda *args: events.append('save'))
    if prepared:
        worker._highlight_control = {
            'measurement': (1, 2, 20, 0, 0) if active else None,
            'adu_average': 20 if active else 0,
            'stable': active, 'current_adu_target': 20,
            'transition': {'active': enabled and not focus, 'trusted': active,
                           'reason': 'capture decision', 'reset': not enabled or focus},
        }
    namespace = dict(self=worker, i_ref=reference, exposure=0.01, gain=0, binning=1,
                     camera=Mock(), filename_p=Mock(), libcamera_black_level=0, asi676mc=asi676mc,
                     logger=logging.getLogger(__name__), apply_control_snapshot=apply_control_snapshot,
                     HighlightMeasurement=HighlightMeasurement)
    with caplog.at_level(logging.INFO):
        exec(early, namespace)
        assert controller.compare_highlights.call_count == int(active and not prepared)
        controller.compare_exposure.assert_not_called()
        exec(late, namespace)
    expected = ['purple_check']
    if fits_mode == 'pre_dark':
        expected.append('save')
    if metered and not prepared:
        expected.append('measure')
    expected += ['dark', 'holes']
    if fits_mode == 'post_dark':
        expected.append('save')
    expected.append('debayer')
    if active and not prepared:
        expected += ['calibrated_adu', 'highlight_control']
    expected += ['stack', 'denoise']
    if enabled and not focus and not excluded:
        expected.append('compensate')
    expected += ['stretch', 'convert']
    if not excluded and not repaired and not active and not prepared:
        expected.append('ordinary_control')
    assert events == expected
    assert controller.compare_highlights.call_count == int(active and not prepared)
    assert controller.compare_exposure.call_count == int(not excluded and not repaired and not active and not prepared)
    assert controller.reset_highlights.call_count == int(enabled and not focus and (excluded or repaired))
    assert processor.highlight_transition is controller.highlight_transition
    assert processor.highlight_transition.trusted is (active and prepared)
    if enabled and not focus:
        assert processor.highlight_transition.active
        assert processor.highlight_transition.lift == .5
    else:
        disabled = HighlightTransition()
        disabled.reset()
        assert processor.highlight_transition.__dict__ == disabled.__dict__
    if active and not prepared:
        assert 'Highlight control source: frame 2026-10-05T07:00:14; exposure 0.010000s @ gain 0.000' in caplog.text
        assert controller.compare_highlights.call_args.args[0] == HighlightMeasurement(1, 2, 20)
        assert controller.compare_highlights.call_args.kwargs == {}
        assert namespace['adu'] == namespace['adu_average'] == 20
        assert 'Highlight shadow lift applied: 1.250 stops' in caplog.text
    if enabled and not focus and not excluded:
        processor.compensate_highlights.assert_called_once_with(60)
    if repaired or excluded:
        assert namespace['adu_average'] == 22
        assert controller.hist_adu == [21, 23]
    if repaired:
        assert 'Highlight exposure/gain held: repaired ASI676MC frame' in caplog.text
    assert config == original_config


@pytest.fixture
def highlight_processor():
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    cls.body = [n for n in cls.body if
                (isinstance(n, ast.FunctionDef) and n.name in (
                    'measure_highlights', 'calibrate_highlights', 'compensate_highlights', '_generateAduMask',
                    'measure_output_highlights', 'highlight_output_trusted',
                    'rotate_90', '_rotate_90', 'rotate_angle', '_rotate_angle', 'flip_v', 'flip_h', '_flip', 'crop_image', '_crop_image',
                    'correct_asi676mc_frame', '_set_asi676mc_repair_result', '_debayer',
                    'apply_gamma_correction', '_apply_gamma_correction'))
                or (isinstance(n, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id in ('__cfa_bgr_map', '__cfa_gray_map') for t in n.targets))]
    namespace = dict(__package__='indi_allsky', constants=constants, numpy=np, cv2=cv2,
                     logger=logging.getLogger(__name__), math=math, asi676mc=asi676mc, copy=copy)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'processing-highlight-methods', 'exec'), namespace)
    processor = namespace['ImageProcessor']()
    processor.max_bit_depth = 16
    processor.night_av = [False, False]
    processor.config = {'TARGET_ADU_DAY': 80, 'HIGHLIGHT_PROTECTION': {'MAX_BOOST': 2}}
    processor.highlight_transition = HighlightTransition()
    return processor


@pytest.mark.parametrize('binning', [1, 2])
@pytest.mark.parametrize('geometry', [
    {}, {'IMAGE_ROTATE': 'ROTATE_90_CLOCKWISE'},
    {'IMAGE_FLIP_H': True, 'IMAGE_FLIP_V': True},
    {'IMAGE_ROTATE_ANGLE': -3, 'IMAGE_ROTATE_KEEP_SIZE': True},
    {'IMAGE_ROTATE_ANGLE': 30, 'IMAGE_ROTATE_KEEP_SIZE': False},
    {'IMAGE_CROP_ROI': [4, 6, 50, 52]},
    {'IMAGE_CROP_IMAGE_CIRCLE': True, 'LENS_IMAGE_CIRCLE': 40, 'LENS_OFFSET_X': 2, 'LENS_OFFSET_Y': -2},
])
def test_output_mask_follows_geometry_without_changing_image_or_source_mask(highlight_processor, geometry, binning):
    p = highlight_processor
    p.config.update(geometry)
    ref = SimpleNamespace(binning=binning)
    p.getLatestImage = lambda: ref
    mask = np.zeros((60, 80), np.uint8)
    mask[12:40, 10:36] = 255
    original = mask.copy()
    p._adu_mask_dict = {binning: mask}
    p.image = np.repeat(mask[:, :, None], 3, axis=2)
    p.rotate_90()
    p.rotate_angle()
    p.flip_v()
    p.flip_h()
    p.crop_image()
    rendered = p.image.copy()
    identity = p.image
    output = p.measure_output_highlights()
    assert output.full == output.any == 100
    assert p.image is identity
    np.testing.assert_array_equal(p.image, rendered)
    np.testing.assert_array_equal(mask, original)
    cached = p._highlight_output_mask
    assert p.measure_output_highlights() == output
    assert p._highlight_output_mask is cached
    p._adu_mask_dict[binning] = np.ones_like(mask) * 255
    assert p.measure_output_highlights().full < 100  # mask replacement invalidates cache


@pytest.mark.parametrize('mono', [False, True])
def test_output_meter_excludes_detection_text_and_unusable_masks(highlight_processor, mono):
    p = highlight_processor
    p.config['DETECT_DRAW'] = True
    p.getLatestImage = lambda: SimpleNamespace(binning=1, opencv_data=np.zeros((40, 40) if mono else (40, 40, 3), np.uint8))
    mask = np.ones((40, 40), np.uint8) * 255
    p._adu_mask_dict = {1: mask}

    def draw(data, binning):
        data[:20] = 200
        return data

    p._draw = SimpleNamespace(main=draw)
    p.image = np.zeros((40, 40, 3), np.uint8)
    p.image[:20] = 255  # rendered annotation became white through enhancement
    assert p.measure_output_highlights().full == 0
    p.image = np.zeros((20, 20, 3), np.uint8)
    assert p.measure_output_highlights() is None
    p._adu_mask_dict[1] = None
    assert p.measure_output_highlights() is None


@pytest.mark.parametrize('change,trusted', [({}, True), ({'exposure': .9}, False),
    ({'gain': 1}, False), ({'binning': 2}, False),
    ({'asi676mc_repair_result': {'status': 'repaired'}}, False),
    ({'asi676mc_repair_result': {'status': 'excluded'}, 'exposure': .9}, True)])
def test_output_feedback_waits_for_matching_trusted_stack(highlight_processor, change, trusted):
    p = highlight_processor
    ref = SimpleNamespace(exposure=1., gain=0, binning=1, asi676mc_repair_result=None)
    old = SimpleNamespace(**dict(vars(ref), **change))
    p.getLatestImage = lambda: ref
    p.image_list = [ref, old, None]
    assert p.highlight_output_trusted() is trusted


@pytest.mark.parametrize('valid_raw', [False, True])
@pytest.mark.parametrize('output_enabled', [False, True])
@pytest.mark.parametrize('trusted', [False, True])
@pytest.mark.parametrize('valid_output', [False, True])
@pytest.mark.parametrize('prepared', [False, True])
def test_late_output_feedback_only_records_and_never_commands(valid_raw, output_enabled, trusted, valid_output, prepared):
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    section = textwrap.dedent(source[source.index('        if highlights is not None and self.config.get'):
                                     source.index('        self.image_processor.realtimeKeogramUpdate()')])
    assert source.index('self.image_processor.colormap()') < source.index(section.splitlines()[0].strip())
    assert source.index(section.splitlines()[0].strip()) < source.index('self.image_processor.apply_logo_overlay(')
    state = HighlightOutput()
    controller = SimpleNamespace(highlight_output=state)  # no command methods available
    result = HighlightMeasurement(2.5, 4, 0)
    processor = SimpleNamespace(measure_output_highlights=Mock(return_value=result if valid_output else None),
                                highlight_output_trusted=Mock(return_value=trusted))
    worker = SimpleNamespace(config={'HIGHLIGHT_PROTECTION': {'OUTPUT_ENABLE': output_enabled}},
                             exposure_o=controller, image_processor=processor, night_av=[True, False],
                             send_highlight_feedback=Mock())
    if prepared:
        worker._highlight_control = {'status': 'metered'}
    capture = {'exp_time': 1234}
    exec(section, dict(self=worker, highlights=result if valid_raw else None, exposure=1., gain=0,
                       i_ref=SimpleNamespace(exp_date=datetime(2026, 10, 5)), logger=logging.getLogger(__name__),
                       i_dict=capture))
    assert processor.measure_output_highlights.call_count == int(valid_raw and output_enabled)
    assert state.active is (valid_raw and output_enabled and trusted and valid_output and not prepared)
    assert worker.send_highlight_feedback.call_count == int(valid_raw and output_enabled and prepared)
    if valid_raw and output_enabled and prepared:
        worker.send_highlight_feedback.assert_called_once_with(capture, result if valid_output else None, trusted)


@pytest.mark.parametrize('shared_color', [False, True, None])
@pytest.mark.parametrize('shape', [(2, 2), (2, 2, 3)])
def test_highlight_gamma_toggle_and_profile_transitions(highlight_processor, shared_color, shape):
    processor = highlight_processor
    processor.focus_mode = False
    processor._gamma_lut = None
    processor.config = {'GAMMA_CORRECTION': 0.5, 'GAMMA_CORRECTION_DAY': 2.0,
                        'HIGHLIGHT_PROTECTION': {'ENABLE': False, 'GAMMA': 1.0, 'GAMMA_DAY': 4.0}}
    if shared_color is not None:
        processor.config['USE_NIGHT_COLOR'] = shared_color
    # Reuse the processor and LUT through on/off, night, day and moon transitions.
    for enabled in (False, True, False, True):
        processor.config['HIGHLIGHT_PROTECTION']['ENABLE'] = enabled
        # Verify settled endpoints; the transition itself is tested separately.
        processor.highlight_transition.active = True
        processor.highlight_transition.gamma_mix = 1.0
        for night, moon in ((False, False), (True, False), (True, True), (False, False)):
            processor.night_av = [night, moon]
            processor.image = np.full(shape, 64, dtype=np.uint8)
            processor.apply_gamma_correction()
            night_profile = shared_color is not False or night
            expected = (64 if night_profile else 180) if enabled else (16 if night_profile else 127)
            np.testing.assert_array_equal(processor.image, np.full(shape, expected, dtype=np.uint8))
    assert processor.config['GAMMA_CORRECTION'] == 0.5
    assert processor.config['GAMMA_CORRECTION_DAY'] == 2.0


@pytest.mark.parametrize('settings', [None, {}, {'ENABLE': True},
                                    {'ENABLE': True, 'GAMMA': 0, 'GAMMA_DAY': 0},
                                    {'ENABLE': False, 'GAMMA': 4, 'GAMMA_DAY': 4}])
def test_highlight_gamma_missing_zero_or_disabled_preserves_normal_processing(highlight_processor, settings):
    processor = highlight_processor
    processor.focus_mode = False
    processor._gamma_lut = None
    processor.config = {'USE_NIGHT_COLOR': False, 'GAMMA_CORRECTION': 0.5, 'GAMMA_CORRECTION_DAY': 2.0}
    if settings is not None:
        processor.config['HIGHLIGHT_PROTECTION'] = settings
    for night, expected in ((False, 127), (True, 16)):
        processor.night_av[constants.NIGHT_NIGHT] = night
        processor.image = np.full((2, 2, 3), 64, dtype=np.uint8)
        processor.apply_gamma_correction()
        assert np.all(processor.image == expected)
    processor.focus_mode = True
    processor.config['HIGHLIGHT_PROTECTION'] = {'ENABLE': True, 'GAMMA': 4}
    original = processor.image.copy()
    processor.apply_gamma_correction()
    np.testing.assert_array_equal(processor.image, original)


def test_highlight_gamma_overrides_unity_standard_gamma(highlight_processor):
    processor = highlight_processor
    processor.highlight_transition.active = True
    processor.highlight_transition.gamma_mix = 1.0
    processor.focus_mode = False
    processor._gamma_lut = None
    processor.config = {'USE_NIGHT_COLOR': False, 'GAMMA_CORRECTION_DAY': 1,
                        'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'GAMMA_DAY': 2}}
    processor.image = np.array([[0, 64, 255]], dtype=np.uint8)
    processor.apply_gamma_correction()
    np.testing.assert_array_equal(processor.image, [[0, 127, 255]])


def test_enabled_but_unneeded_gamma_is_normal_and_entry_is_bounded(highlight_processor):
    processor = highlight_processor
    processor.focus_mode = False
    processor._gamma_lut = None
    processor.config = {'USE_NIGHT_COLOR': False, 'GAMMA_CORRECTION_DAY': 1.565,
                        'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'GAMMA_DAY': 1.85}}
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    processor.image = source.copy()
    processor.apply_gamma_correction()
    normal = processor.image.copy()
    assert processor._gamma_lut_gamma == 1.565
    processor.highlight_transition.observe(HighlightMeasurement(0, 0, 80), 80, 10, {}, False, False, False)
    processor.highlight_transition.observe(HighlightMeasurement(2, 4, 80), 80, 10, {}, False, False, False)
    for _ in range(40):
        processor.image = source.copy()
        processor.apply_gamma_correction()
        assert np.abs(processor.image.astype(int) - normal).max() <= 2
        normal = processor.image.astype(int)
    assert processor._gamma_lut_gamma == 1.85


@pytest.mark.parametrize('altitude', [-12, -9, -6])
@pytest.mark.parametrize('mix', [0, .5, 1])
def test_optional_twilight_blend_supplies_both_gamma_endpoints(highlight_processor, altitude, mix):
    twilight = pytest.importorskip('indi_allsky.twilight')
    processor = highlight_processor
    profile = twilight.TwilightTransition({
        'TWILIGHT_TRANSITION': {'ENABLE': True}, 'USE_NIGHT_COLOR': False,
        'GAMMA_CORRECTION_DAY': 1.565, 'GAMMA_CORRECTION': .87,
        'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'GAMMA_DAY': 1.85, 'GAMMA': 0},
    })
    profile.apply(altitude)
    processor.config = profile.config
    processor.focus_mode = False
    processor._gamma_lut = None
    processor.highlight_transition.gamma_mix = mix
    normal = processor.config['GAMMA_CORRECTION']
    protected = processor.config['HIGHLIGHT_PROTECTION']['GAMMA']
    gamma = normal if mix == 0 else protected if mix == 1 else 1 / ((1 - mix) / normal + mix / protected)
    source = np.arange(256, dtype=np.uint8).reshape(16, 16)
    expected = (((source.astype(np.float32) / 255) ** (1 / gamma)) * 255).astype(np.uint8)
    for night in (False, True):
        processor.night_av[0] = night
        processor.image = source.copy()
        processor.apply_gamma_correction()
        np.testing.assert_array_equal(processor.image, expected)


def test_processor_meters_pre_dark_capture_with_calibrated_adu_not_stack(highlight_processor):
    processor = highlight_processor
    raw = np.zeros((10, 10, 3), dtype=np.uint16)
    raw[:2, :2] = 65535
    original = raw.copy()
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=np.moveaxis(raw, -1, 0))],
                                binning=2, image_bitpix=16)
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {2: np.ones((10, 10), dtype=np.uint8)}
    processor.image = np.full_like(raw, 2000)
    measured = processor.measure_highlights()
    assert measured.full == measured.any == 4
    # A dark larger than the old 1% tolerance hides every clipped pixel.
    reference.opencv_data = np.maximum(raw.astype(np.int32) - 3000, 0).astype(np.uint16)
    assert measure(reference.opencv_data, processor._adu_mask_dict[2], 16).full == 0
    calibrated = processor.calibrate_highlights(measured)
    assert calibrated.full == calibrated.any == 4
    assert calibrated.full_next == measured.full_next and calibrated.any_next == measured.any_next
    assert calibrated.adu == pytest.approx(reference.opencv_data.mean() / 256)
    assert calibrated.adu != pytest.approx(processor.image.mean() / 256)
    processor.highlight_transition.active = True
    processor.highlight_transition.reference = 80
    assert processor.compensate_highlights(20) == 2.0
    assert np.all(processor.image == 8000)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize('configured,detected,expected_bits', [(12, 8, 12), (16, 12, 16), (0, 12, 12), (0, 16, 16)])
def test_meter_uses_configured_or_detected_output_range_not_fits_container(highlight_processor, configured, detected, expected_bits):
    processor = highlight_processor
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py').read_text(encoding='utf-8')
    start = source.index('        detected_bit_depth = i_ref.detected_bit_depth')
    end = source.index('        # read this before', start)
    processor.config['CCD_BIT_DEPTH'] = configured
    processor.max_bit_depth = 8
    data = np.full((10, 10), 100, dtype=np.uint16)
    data[:2, :2] = 2 ** expected_bits - 1
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=1,
                                image_bitpix=16, image_bayerpat=None, detected_bit_depth=detected)
    exec(textwrap.dedent(source[start:end]), {'self': processor, 'i_ref': reference,
                                            'logger': logging.getLogger(__name__)})
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {1: np.ones((10, 10), dtype=np.uint8)}
    assert processor.max_bit_depth == expected_bits
    assert processor.measure_highlights().full == 4


@pytest.mark.parametrize('bits', [8, 10, 12, 14, 16])
@pytest.mark.parametrize('layout', ['mono', 'rgb', 'RGGB', 'GRBG', 'BGGR', 'GBRG', 'override'])
def test_pre_dark_meter_supports_formats_and_colour_clipping_in_grayscale_mode(highlight_processor, bits, layout):
    processor = highlight_processor
    maximum = (1 << bits) - 1
    data = np.zeros((100, 100), dtype=np.uint8 if bits == 8 else np.uint16)
    pattern = None
    if layout == 'rgb':
        data = np.stack([data] * 3)
        data[0, 20:40, 20:40] = maximum  # red-only clipping
    elif layout == 'mono':
        data[20:40, 20:40] = maximum
    else:
        pattern = 'RGGB' if layout == 'override' else layout
        y, x = divmod(pattern.index('R'), 2)
        data[20+y:40:2, 20+x:40:2] = maximum
        if layout == 'override':
            processor.config['CFA_PATTERN'] = pattern
            pattern = None
    original = data.copy()
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=1,
                                image_bitpix=8 if bits == 8 else 16, image_bayerpat=pattern)
    processor.getLatestImage = lambda: reference
    processor.max_bit_depth = bits
    processor._adu_mask_dict = {1: np.ones((100, 100), dtype=np.uint8)}
    processor.config.update(NIGHT_GRAYSCALE=True, DAYTIME_GRAYSCALE=True)
    result = processor.measure_highlights()
    assert 3 < result.any <= 4
    assert result.full == (4 if layout == 'mono' else 0)
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('bitpix,dtype', [(-32, np.float32), (32, np.uint32)])
def test_pre_dark_meter_does_not_convert_or_modify_saved_source(highlight_processor, bitpix, dtype):
    processor = highlight_processor
    data = np.full((10, 10), 70000, dtype=dtype)
    data[0, 0] = -10 if bitpix == -32 else 0
    original = data.copy()
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=1,
                                image_bitpix=bitpix, image_bayerpat=None)
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {1: np.ones((10, 10), dtype=np.uint8)}
    assert processor.measure_highlights().full == 99
    assert reference.image_bitpix == bitpix and reference.hdulist[0].data is data
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize('roi', [[], [20, 20, 60, 60]])
def test_pre_dark_meter_generates_existing_roi_fallback_before_stack(highlight_processor, roi):
    processor = highlight_processor
    data = np.zeros((100, 100), dtype=np.uint16)
    data[26:28, 26:28] = 65535
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=2,
                                image_bitpix=16, image_bayerpat=None)
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {2: None}
    processor.config['ADU_ROI'] = roi
    result = processor.measure_highlights()
    expected_area = 21 * 21 if roi else 51 * 51
    assert np.count_nonzero(processor._adu_mask_dict[2]) == expected_area
    assert result.full == pytest.approx(4 * 100 / expected_area)


@pytest.mark.parametrize('repair_settings,camera', [
    (None, 'Generic camera'), ({'ENABLE': False}, 'ZWO ASI676MC'),
    ({'ENABLE': True}, 'Generic camera'), ({'ENABLE': True}, None),
])
@pytest.mark.parametrize('layout', ['mono', 'rgb', 'RGGB', 'GRBG', 'BGGR', 'GBRG'])
@pytest.mark.parametrize('bitpix,dtype', [(8, np.uint8), (16, np.uint16), (-32, np.float32), (32, np.uint32)])
def test_meter_calibration_and_lift_work_without_purple_repair(highlight_processor, repair_settings, camera, layout, bitpix, dtype):
    processor = highlight_processor
    if repair_settings is not None:
        processor.config['IMAGE_ASI676MC_REPAIR'] = repair_settings
    processor.max_bit_depth = 8 if bitpix == 8 else 16
    maximum = 2 ** processor.max_bit_depth - 1
    data = np.full((64, 64), maximum // 8, dtype=dtype)
    data[20:30, 20:30] = maximum
    if layout == 'rgb':
        data = np.stack([data] * 3)
    original = data.copy()
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data)], binning=1, image_bitpix=bitpix,
                                image_bayerpat=None if layout in ('mono', 'rgb') else layout,
                                detected_camera_name=camera, asi676mc_repair_result=None)
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {1: np.ones((64, 64), dtype=np.uint8)}
    assert processor.correct_asi676mc_frame(reference) is False
    assert reference.asi676mc_repair_result is None
    metrics = processor.measure_highlights()
    assert metrics.full > 1 and metrics.any >= metrics.full
    np.testing.assert_array_equal(data, original)
    # The ordinary debayer path normalizes supported FITS formats; neither it
    # nor highlight control requires repair metadata or a particular camera.
    reference.opencv_data = processor._debayer(reference)
    processor.image = reference.opencv_data.copy()
    calibrated = processor.calibrate_highlights(metrics)
    assert calibrated.full == metrics.full
    processor.highlight_transition.observe(calibrated, 80, 10, {}, False, False, False)
    assert 0 < processor.compensate_highlights(calibrated.adu) <= 2
    assert processor.image.dtype in (np.uint8, np.uint16)


@pytest.mark.parametrize('bad', [False, True])
def test_actual_purple_repair_retains_normal_metering_and_repaired_lift(highlight_processor, bad):
    processor = highlight_processor
    processor.config['IMAGE_ASI676MC_REPAIR'] = {'ENABLE': True, 'EXCLUDE_ONLY': False}
    data = np.full((64, 64), 1000, dtype=np.uint16)
    if bad:
        data[::2, ::2] = data[1::2, 1::2] = 4000
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=data, header={})], binning=1,
                                image_bitpix=16, image_bayerpat='RGGB', detected_camera_name='ZWO ASI676MC',
                                asi676mc_repair_result=None)
    processor.getLatestImage = lambda: reference
    processor._adu_mask_dict = {1: np.ones((64, 64), dtype=np.uint8)}
    assert processor.correct_asi676mc_frame(reference) is bad
    assert reference.asi676mc_repair_result['status'] == ('repaired' if bad else 'normal')
    if not bad:
        assert processor.measure_highlights().full == 0
    reference.opencv_data = processor._debayer(reference)
    processor.image = reference.opencv_data.copy()
    # A repaired frame retains established protection, without arming it itself.
    processor.highlight_transition.active = True
    processor.highlight_transition.reference = 80
    assert processor.compensate_highlights(float(processor.image.mean()) / 256) == 2
    assert np.all(processor.image >= reference.opencv_data)


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
