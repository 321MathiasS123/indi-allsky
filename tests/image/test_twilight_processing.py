"""Run the real image methods without importing Linux D-Bus web services."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import logging
import math
from multiprocessing import Array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import textwrap

import cv2
import numpy
import pytest

from indi_allsky import asi676mc, constants
from indi_allsky.denoise import IndiAllskyDenoise
from indi_allsky.scnr import IndiAllskyScnr
from indi_allsky.stretch.mode2_mtf import IndiAllSky_Mode2_MTF_Stretch
from indi_allsky.twilight import TwilightTransition, interpolate, runtime_weight


@pytest.fixture(scope='module')
def processor_class():
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    namespace = dict(__package__='indi_allsky', math=math, cv2=cv2, numpy=numpy, constants=constants, datetime=datetime,
                     timedelta=timedelta, timezone=timezone, logger=logging.getLogger('test'),
                     TwilightTransition=TwilightTransition, interpolate=interpolate, runtime_weight=runtime_weight,
                     IndiAllskyDenoise=IndiAllskyDenoise, IndiAllskyScnr=IndiAllskyScnr)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ImageProcessor']


@pytest.fixture
def processor(processor_class):
    def create(settings=None, altitude=-9, bits=8):
        config = {
            'TWILIGHT_TRANSITION': {'ENABLE': True}, 'USE_NIGHT_COLOR': False,
            'TARGET_ADU': 100, 'TARGET_ADU_DAY': 50,
            'IMAGE_STRETCH': {'CLASSNAME': 'mode2_mtf', 'DAYTIME': False},
        }
        config.update(settings or {})
        p = processor_class.__new__(processor_class)
        p.twilight = TwilightTransition(config)
        p.config = p.twilight.config
        p.twilight.apply(altitude)
        p.night_av = [1, 0]
        p.focus_mode = False
        p._max_bit_depth = bits
        p._twilight_filters = {}
        p._ia_denoise = IndiAllskyDenoise(p.config, p.night_av)
        p._ia_scnr = IndiAllskyScnr(p.config, p.night_av)
        p._gamma_lut = p._gamma_lut_gamma = None
        p._wb_mtf_night = 1
        p._wbb_mtf_lut = p._wbg_mtf_lut = p._wbr_mtf_lut = None
        p._stretch_o = IndiAllSky_Mode2_MTF_Stretch(p.config)
        p.image_list = [SimpleNamespace(binning=1)]
        rng = numpy.random.default_rng(14)
        p.image = rng.integers(0, 2 ** bits, size=(48, 48, 3), dtype=numpy.uint8 if bits == 8 else numpy.uint16)
        return p
    return create


@pytest.mark.parametrize('bits', [8, 16])
@pytest.mark.parametrize('altitude', [-6, -9, -12])
@pytest.mark.parametrize('mono', [False, True])
@pytest.mark.parametrize('split', [False, True])
def test_stretch_and_contrast_fade_same_frame(processor, bits, altitude, mono, split):
    p = processor({'DAYTIME_CONTRAST_ENHANCE': False, 'NIGHT_CONTRAST_ENHANCE': True,
                   'IMAGE_STRETCH': {'CLASSNAME': 'mode2_mtf', 'DAYTIME': False, 'SPLIT': split}}, altitude, bits)
    if mono:
        p.image = p.image[:, :, 0].copy()
    retained = p.image
    original = p.image.copy()
    full_stretch = p._stretch(p.getLatestImage())
    if split:
        full_stretch = p.splitscreen(original, full_stretch)
    w = p.twilight.weight
    p.stretch()
    numpy.testing.assert_array_equal(p.image, cv2.addWeighted(original, 1 - w, full_stretch, w, 0))
    numpy.testing.assert_array_equal(retained, original)
    p.image = original.copy()
    (p.contrast_clahe_16bit if bits == 16 else p.contrast_clahe)()
    full_contrast = p.image.copy()
    p.image = original.copy()
    retained = p.image
    assert p.contrast_transition(bit16=bits == 16)
    numpy.testing.assert_array_equal(p.image, cv2.addWeighted(original, 1 - w, full_contrast, w, 0))
    # Keeping a reference instead of copying is safe only if filters leave it intact.
    numpy.testing.assert_array_equal(retained, original)


@pytest.mark.parametrize('algorithm_day,algorithm_night', [('', 'gaussian_blur'), ('median_blur', ''),
                                                        ('median_blur', 'gaussian_blur'), ('gaussian_blur', 'gaussian_blur')])
@pytest.mark.parametrize('altitude,night', [(-5, False), (-6, False), (-8, False), (-9, False),
                                          (-9.001, True), (-10, True), (-12, True), (-13, True)])
def test_denoising_uses_nearest_endpoint_once(processor, monkeypatch, algorithm_day, algorithm_night, altitude, night):
    p = processor({'IMAGE_DENOISE_DAY': algorithm_day, 'IMAGE_DENOISE': algorithm_night,
                   'IMAGE_DENOISE_STRENGTH_DAY': 1, 'IMAGE_DENOISE_STRENGTH': 4}, altitude)
    original = p.image.copy()
    config_before = deepcopy(p.config)
    calls = []
    def record_filter(obj, image):
        calls.append(obj.config['IMAGE_DENOISE_STRENGTH'])
        return image + 1
    for algorithm in ('median_blur', 'gaussian_blur'):
        monkeypatch.setattr(IndiAllskyDenoise, algorithm, record_filter)
    p.denoise()
    active = algorithm_night if night else algorithm_day
    assert calls == ([4 if night else 1] if active else [])
    numpy.testing.assert_array_equal(p.image, original + 1 if active else original)
    assert p.config == config_before
    assert p.night_av == [1, 0]


@pytest.mark.parametrize('shared', [False, True])
def test_identical_denoise_endpoints_match_one_real_pass_through_dawn_and_dusk(processor, monkeypatch, shared):
    p = processor({'IMAGE_DENOISE_DAY': 'gaussian_blur', 'IMAGE_DENOISE': 'gaussian_blur',
                   'IMAGE_DENOISE_STRENGTH_DAY': 2, 'IMAGE_DENOISE_STRENGTH': 2, 'USE_NIGHT_COLOR': shared})
    original = p.image.copy()
    algorithm = IndiAllskyDenoise.gaussian_blur
    expected = algorithm(IndiAllskyDenoise(p.twilight.endpoint_config(True), [1, 0]), original.copy())
    calls = Mock()
    def wrapped(obj, image):
        calls()
        return algorithm(obj, image)
    monkeypatch.setattr(IndiAllskyDenoise, 'gaussian_blur', wrapped)
    altitudes = (-13, -10, -9.001, -9, -8, -5, -8, -9, -10, -8, -5)
    for altitude in altitudes:
        p.twilight.apply(altitude)
        p.image = original.copy()
        p.denoise()
        numpy.testing.assert_array_equal(p.image, expected)
    assert calls.call_count == len(altitudes)


def test_shared_night_color_does_not_select_day_denoising(processor, monkeypatch):
    p = processor({'USE_NIGHT_COLOR': True, 'IMAGE_DENOISE_DAY': 'median_blur',
                   'IMAGE_DENOISE': 'gaussian_blur'}, -6)
    night = Mock(return_value=p.image)
    day = Mock(return_value=p.image)
    monkeypatch.setattr(IndiAllskyDenoise, 'gaussian_blur', night)
    monkeypatch.setattr(IndiAllskyDenoise, 'median_blur', day)
    p.denoise()
    night.assert_called_once()
    day.assert_not_called()


@pytest.mark.parametrize('algorithm_day,algorithm_night', [
    ('', 'star_aware'), ('star_aware', ''), ('star_aware', 'star_aware'), ('wavelet', 'star_aware'),
])
@pytest.mark.parametrize('altitude', [-13, -9.001, -9, -5])
@pytest.mark.parametrize('gray_day,gray_night', [(False, False), (True, True), (True, False)])
def test_star_aware_transition_keeps_bayer_and_filter_selection_in_step(
        processor, monkeypatch, algorithm_day, algorithm_night, altitude, gray_day, gray_night):
    from indi_allsky import sky_denoise

    p = processor({'IMAGE_DENOISE_DAY': algorithm_day, 'IMAGE_DENOISE': algorithm_night,
                   'IMAGE_DENOISE_STRENGTH_DAY': 2, 'IMAGE_DENOISE_STRENGTH': 4,
                   'DAYTIME_GRAYSCALE': gray_day, 'NIGHT_GRAYSCALE': gray_night}, altitude)
    night = p.twilight.weight > 0.5
    # The operational capture flag can disagree with the colour endpoint.
    p.night_av[0] = int(not night)
    raw = numpy.full((64, 64), 10000, dtype=numpy.uint16)
    original = raw.copy()
    frame = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=16,
                            image_bayerpat='RGGB', binning=2)
    p.image_list = [frame]
    calls = []

    def repair(data, config, binning):
        calls.append(('bayer', binning))
        return data + 1

    def denoise(data, config, binning, strength, sun_altitude=None):
        calls.append(('denoise', binning, strength))
        return data

    monkeypatch.setattr(sky_denoise, 'repair_bayer', repair)
    monkeypatch.setattr(sky_denoise, 'denoise', denoise)
    monkeypatch.setattr(IndiAllskyDenoise, 'wavelet', lambda obj, data: data)
    p.image = p._debayer(frame)
    p.denoise()
    selected = algorithm_night if night else algorithm_day
    assert calls == ([('bayer', 2), ('denoise', 2, 4 if night else 2)]
                     if selected == 'star_aware' else [])
    numpy.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize('shared,focus', [(True, False), (False, True), (True, True)])
def test_star_aware_bayer_and_filter_honor_shared_night_color_and_focus(processor, monkeypatch, shared, focus):
    from indi_allsky import sky_denoise

    p = processor({'USE_NIGHT_COLOR': shared, 'IMAGE_DENOISE': 'star_aware',
                   'IMAGE_DENOISE_DAY': '', 'IMAGE_DENOISE_STRENGTH': 3}, -5)
    p.focus_mode = focus
    p.night_av[0] = 0
    raw = numpy.full((64, 64), 10000, dtype=numpy.uint16)
    frame = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=16,
                            image_bayerpat='RGGB', binning=2)
    p.image_list = [frame]
    calls = []
    monkeypatch.setattr(sky_denoise, 'repair_bayer',
                        lambda data, config, binning: calls.append('bayer') or data.copy())
    monkeypatch.setattr(sky_denoise, 'denoise',
                        lambda data, config, binning, strength, sun_altitude=None: calls.append('denoise') or data)
    p.image = p._debayer(frame)
    p.denoise()
    assert calls == (['bayer', 'denoise'] if not focus else [])


def test_twilight_star_denoising_receives_capture_altitude(processor, monkeypatch):
    p = processor({'IMAGE_DENOISE': 'star_aware', 'IMAGE_DENOISE_DAY': 'star_aware'}, -7)
    frame = p.image_list[0]
    altitude = Mock(return_value=-7.25)
    p._denoise_sun_altitude = altitude
    denoise = Mock(return_value=p.image)
    monkeypatch.setattr(IndiAllskyDenoise, 'star_aware', denoise)
    p.denoise()
    altitude.assert_called_once_with(frame)
    assert denoise.call_args.kwargs == {'binning': 1, 'sun_altitude': -7.25}


def test_green_removal_selects_one_algorithm_and_blends_midtones_without_stale_cache(processor, monkeypatch):
    p = processor({'SCNR_ALGORITHM': 'green_mtf', 'SCNR_ALGORITHM_DAY': 'green_mtf',
                   'SCNR_MTF_MIDTONES': .7, 'SCNR_MTF_MIDTONES_DAY': .55})
    original = p.image.copy()
    algorithm = IndiAllskyScnr.green_mtf
    calls = Mock()
    def wrapped(obj, image):
        calls()
        return algorithm(obj, image)
    monkeypatch.setattr(IndiAllskyScnr, 'green_mtf', wrapped)
    altitudes = (-12, -10, -9, -8, -6, -8, -9, -10, -12)
    for altitude in altitudes:
        p.twilight.apply(altitude)
        p.image = original.copy()
        expected_config = dict(p.config, USE_NIGHT_COLOR=True)
        expected = algorithm(IndiAllskyScnr(expected_config, p.night_av), original.copy())
        p.scnr()
        numpy.testing.assert_array_equal(p.image, expected)
    assert calls.call_count == len(altitudes)


@pytest.mark.parametrize('enabled_night', [False, True])
@pytest.mark.parametrize('strength', [.5, .51, .7])
@pytest.mark.parametrize('mono', [False, True])
def test_mtf_green_removal_fades_from_disabled_endpoint_without_switch_or_stale_cache(
        processor, monkeypatch, enabled_night, strength, mono):
    settings = {'TWILIGHT_TRANSITION': {'ENABLE': True, 'DAY_ALT': 0, 'NIGHT_ALT': -12},
                'SCNR_ALGORITHM': 'green_mtf' if enabled_night else '',
                'SCNR_ALGORITHM_DAY': '' if enabled_night else 'green_mtf',
                # A stored strength at the disabled end must not affect the fade.
                'SCNR_MTF_MIDTONES': strength if enabled_night else .9,
                'SCNR_MTF_MIDTONES_DAY': .9 if enabled_night else strength}
    p = processor(settings)
    source_before = deepcopy(p.twilight.source)
    original = numpy.tile(numpy.arange(256, dtype=numpy.uint8)[None, :, None], (4, 1, 3))
    if mono:
        original = original[:, :, 0].copy()
    algorithm = IndiAllskyScnr.green_mtf
    calls = []

    def wrapped(obj, image):
        calls.append(obj.config['SCNR_MTF_MIDTONES'])
        return algorithm(obj, image)

    monkeypatch.setattr(IndiAllskyScnr, 'green_mtf', wrapped)
    altitudes = [1, 0, -1, -3, -5.999, -6, -6.001, -9, -12, -13]
    for altitude in altitudes + altitudes[::-1]:
        p.twilight.apply(altitude)
        p.night_av[0] = altitude < -6
        amount = p.twilight.weight if enabled_night else 1 - p.twilight.weight
        midtones = .5 + (strength - .5) * amount
        p.image = original.copy()
        calls.clear()
        p.scnr()
        if amount == 0 or strength == .5:
            numpy.testing.assert_array_equal(p.image, original)
            assert calls == []
        else:
            expected = algorithm(IndiAllskyScnr({'USE_NIGHT_COLOR': True,
                                 'SCNR_MTF_MIDTONES': midtones}, p.night_av), original.copy())
            numpy.testing.assert_array_equal(p.image, expected)
            assert calls == pytest.approx([midtones])
    assert p.twilight.source == source_before


@pytest.mark.parametrize('shared,enabled,focus', [(True, True, False), (False, False, False), (False, True, True)])
def test_mtf_disabled_endpoint_fade_respects_existing_bypasses(processor, monkeypatch, shared, enabled, focus):
    p = processor({'TWILIGHT_TRANSITION': {'ENABLE': enabled}, 'USE_NIGHT_COLOR': shared,
                   'SCNR_ALGORITHM': '', 'SCNR_ALGORITHM_DAY': 'green_mtf'}, -10)
    p.focus_mode = focus
    original = p.image.copy()
    apply = Mock(side_effect=AssertionError('MTF must not run on this bypass path'))
    monkeypatch.setattr(IndiAllskyScnr, 'green_mtf', apply)
    p.scnr()
    apply.assert_not_called()
    numpy.testing.assert_array_equal(p.image, original)


@pytest.mark.parametrize('altitude,expected', [(-12, 'night'), (-9.001, 'night'), (-9, 'day'), (-6, 'day')])
def test_different_green_removal_algorithms_choose_nearest_endpoint(processor, monkeypatch, altitude, expected):
    p = processor({'SCNR_ALGORITHM': 'green_mtf', 'SCNR_ALGORITHM_DAY': 'maximum_neutral'}, altitude)
    calls = []
    monkeypatch.setattr(IndiAllskyScnr, 'green_mtf', lambda obj, im: calls.append('night') or im)
    monkeypatch.setattr(IndiAllskyScnr, 'maximum_neutral', lambda obj, im: calls.append('day') or im)
    p.scnr()
    assert calls == [expected]


def test_white_balance_cache_follows_effective_parameters_in_both_directions(processor):
    settings = {'WBR_MTF_MIDTONES_DAY': .7, 'WBR_MTF_MIDTONES': .3}
    p = processor(settings)
    original = p.image.copy()
    for altitude in (-6, -8, -10, -12, -8, -6):
        p.twilight.apply(altitude)
        p.image = original.copy()
        p.white_balance_mtf()
        fresh = processor(settings, altitude)
        fresh.image = original.copy()
        fresh.white_balance_mtf()
        numpy.testing.assert_array_equal(p.image, fresh.image)


@pytest.mark.parametrize('altitude', [-13, -12, -9, -6, -5])
@pytest.mark.parametrize('shared', [False, True])
def test_auto_white_balance_fades_instead_of_switching(processor, altitude, shared):
    p = processor({'AUTO_WB_DAY': False, 'AUTO_WB': True, 'USE_NIGHT_COLOR': shared}, altitude)
    p.image[:, :, 0] //= 2
    original = p.image.copy()
    p._white_balance_auto_bgr()
    balanced = p.image.copy()
    p.image = original.copy()
    retained = p.image
    p.white_balance_auto_bgr()
    amount = 1 if shared else p.twilight.weight
    numpy.testing.assert_array_equal(p.image, cv2.addWeighted(original, 1 - amount, balanced, amount, 0))
    numpy.testing.assert_array_equal(retained, original)


def test_grayscale_transition_preserves_shape_through_both_endpoints(processor):
    p = processor({'DAYTIME_GRAYSCALE': True, 'NIGHT_GRAYSCALE': False})
    raw = numpy.random.default_rng(18).integers(0, 255, size=(32, 32), dtype=numpy.uint8)
    frame = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=8, image_bayerpat='RGGB')
    for altitude in (-6, -6.001, -9, -11.999, -12, -9, -6):
        p.twilight.apply(altitude)
        result = p._debayer(frame)
        assert result.shape == (32, 32, 3)
        if altitude == -6:
            numpy.testing.assert_array_equal(result[:, :, 0], result[:, :, 1])


@pytest.mark.parametrize('shared_color', [False, True])
@pytest.mark.parametrize('gray_night', [False, True])
def test_grayscale_endpoint_cannot_be_recolored_by_white_balance_or_green_removal(processor, shared_color, gray_night):
    p = processor({'DAYTIME_GRAYSCALE': not gray_night, 'NIGHT_GRAYSCALE': gray_night,
                   'USE_NIGHT_COLOR': shared_color,
                   'WBR_FACTOR_DAY': 1.8, 'WBR_FACTOR': 1.8,
                   'WBR_MTF_MIDTONES_DAY': .8, 'WBR_MTF_MIDTONES': .8,
                   'SCNR_ALGORITHM': 'green_mtf', 'SCNR_ALGORITHM_DAY': 'green_mtf'}, -12 if gray_night else -6)
    raw = numpy.full((32, 32), 100, dtype=numpy.uint8)
    frame = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=8, image_bayerpat='RGGB')
    p.image_list = [frame]
    p.image = p._debayer(frame)
    original = p.image.copy()
    p.scnr()
    p.white_balance_mtf()
    p.white_balance_manual_bgr()
    p.colorize()
    numpy.testing.assert_array_equal(p.image[:, :, 0], p.image[:, :, 1])
    numpy.testing.assert_array_equal(p.image[:, :, 0], p.image[:, :, 2])
    numpy.testing.assert_array_equal(p.image, original)


@pytest.mark.parametrize('method', ['denoise', 'scnr', 'white_balance_auto_bgr', 'stretch', 'contrast_transition'])
def test_focus_mode_still_bypasses_optional_processing(processor, method):
    p = processor({'IMAGE_DENOISE': 'gaussian_blur', 'SCNR_ALGORITHM': 'green_mtf',
                   'AUTO_WB': True, 'NIGHT_CONTRAST_ENHANCE': True})
    p.focus_mode = True
    original = p.image.copy()
    getattr(p, method)()
    numpy.testing.assert_array_equal(p.image, original)


def test_disabled_transition_keeps_original_filter_path(processor):
    p = processor({'TWILIGHT_TRANSITION': {'ENABLE': False}, 'SCNR_ALGORITHM': 'green_mtf',
                   'SCNR_MTF_MIDTONES': .7})
    original = p.image.copy()
    expected = p._ia_scnr.green_mtf(original.copy())
    p.scnr()
    numpy.testing.assert_array_equal(p.image, expected)
    assert not p.contrast_transition()


def test_ingestion_uses_exposure_midpoint_and_keeps_camera_identity(processor):
    p = processor()
    p.position_av = [0] * 5
    p.position_av[constants.POSITION_LATITUDE] = 57
    p._detection_mask_dict = {}
    p._check_astro_darkness = Mock()
    p.stack_count = 1
    p._add = Mock(return_value=SimpleNamespace())
    when = datetime(2026, 6, 21, 23, 45, tzinfo=timezone.utc)
    frame = p.add('frame.fit', 20, 50, 1, when, 25, 'camera', detected_camera_name='device')
    expected = TwilightTransition(p.twilight.source)
    expected.update(when - timedelta(seconds=15), 57, 0, 0)
    assert p.twilight.weight == expected.weight
    assert p.getLatestImage() is frame
    assert p._add.call_args.kwargs == {'detected_camera_name': 'device'}


@pytest.mark.parametrize('elapsed', [0, 30])
def test_ingestion_midpoint_survives_clock_fallback_and_unknown_readout(processor, elapsed):
    from dateutil.tz import tzstr
    p = processor()
    p.position_av = [0] * 5
    p._detection_mask_dict = {}
    p._check_astro_darkness = Mock()
    p.stack_count = 1
    p._add = Mock(return_value=SimpleNamespace())
    p.twilight.update = Mock()
    zone = tzstr('CET-1CEST,M3.5.0/2,M10.5.0/3')
    received = datetime(2026, 10, 25, 2, 30, tzinfo=zone, fold=1)
    p.add('frame.fit', 20, 50, 1, received, elapsed, 'camera')
    expected = received.astimezone(timezone.utc) - timedelta(seconds=max(elapsed, 20) - 10)
    assert p.twilight.update.call_args.args[0] == expected


@pytest.mark.parametrize('start,end', [(0, -12), (-3, -9)])
@pytest.mark.parametrize('night', [False, True])
def test_gamma_follows_custom_interval_across_operational_mode_switch(processor, start, end, night):
    for altitude, gamma in [(start, 1.565), ((start + end) / 2, 1.2175), (end, .87)]:
        p = processor({'TWILIGHT_TRANSITION': {'ENABLE': True, 'DAY_ALT': start, 'NIGHT_ALT': end},
                       'NIGHT_SUN_ALT_DEG': -6, 'GAMMA_CORRECTION_DAY': 1.565,
                       'GAMMA_CORRECTION': .87}, altitude)
        p.night_av[0] = night
        original = p.image.copy()
        p.apply_gamma_correction()
        lut = (((numpy.arange(256, dtype=numpy.float32) / 255) ** (1 / gamma)) * 255).astype(numpy.uint8)
        numpy.testing.assert_array_equal(p.image, lut[original])


@pytest.mark.parametrize('transition_enabled', [False, True])
@pytest.mark.parametrize('highlight_enabled', [False, True])
def test_optional_highlight_rendering_uses_same_blended_targets(processor, transition_enabled, highlight_enabled):
    highlight = pytest.importorskip('indi_allsky.highlight')
    p = processor({'TWILIGHT_TRANSITION': {'ENABLE': transition_enabled},
                   'GAMMA_CORRECTION_DAY': 2.0, 'GAMMA_CORRECTION': 1.5,
                   'HIGHLIGHT_PROTECTION': {'ENABLE': highlight_enabled, 'GAMMA_DAY': 0, 'GAMMA': .8}})
    if hasattr(highlight, 'HighlightTransition'):
        # Test established protection; engagement is covered by highlight tests.
        p.highlight_transition = highlight.HighlightTransition()
        p.highlight_transition.active = True
        p.highlight_transition.reference = 75 if transition_enabled else 100
        p.highlight_transition.gamma_mix = 1
    original = p.image.copy()
    gamma = (1.4 if transition_enabled else .8) if highlight_enabled else (1.75 if transition_enabled else 1.5)
    p.apply_gamma_correction()
    lut = (((numpy.arange(256, dtype=numpy.float32) / 255) ** (1 / gamma)) * 255).astype(numpy.uint8)
    numpy.testing.assert_array_equal(p.image, lut[original])
    if highlight_enabled:
        p.image = original.copy()
        p.getLatestImage().image_bitpix = 8
        target = 75 if transition_enabled else 100
        expected = highlight.compensate(original, 8, 30, target, 2)
        p.compensate_highlights(30)
        numpy.testing.assert_array_equal(p.image, expected)


@pytest.mark.parametrize('status', [None, 'repaired', 'excluded', 'validation_failed'])
def test_combined_worker_keeps_invalid_and_repaired_frames_out_of_control(status):
    pytest.importorskip('indi_allsky.highlight')
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    segment = textwrap.dedent(source[source.index('        # A retained purple/failed frame'):
                                    source.index('        # generate a new mask base')])
    controller = SimpleNamespace(hist_adu=[20, 30], compare_exposure=Mock(), apply_transition_limits=Mock(),
                                 reset_highlights=Mock())
    worker = SimpleNamespace(config={'TWILIGHT_TRANSITION': {'ENABLE': True},
                                    'HIGHLIGHT_PROTECTION': {'ENABLE': True}}, exposure_o=controller,
                             night_av=[True, False], live_night_av=Array('i', [1, 0]))
    ref = SimpleNamespace(asi676mc_repair_result={'status': status} if status else None)
    namespace = dict(self=worker, i_ref=ref, asi676mc=asi676mc, adu=75, adu_average=75,
                     highlight_enabled=True, highlight_adu=75, highlight_adu_average=75, highlight_lift=1,
                     highlights=object() if status is None else None, highlight_repaired=status == 'repaired',
                     exposure=.1, gain=50, control=None, logger=logging.getLogger('test'))
    exec(segment, namespace)
    controller.compare_exposure.assert_not_called()
    assert controller.apply_transition_limits.call_count == int(status is None)
    assert controller.hist_adu == [20, 30]
