import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy
import pytest

from indi_allsky import constants
from indi_allsky.scnr import IndiAllskyScnr


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def processor():
    # Exercise the actual processing methods without camera/database services.
    source = ast.parse((ROOT / 'indi_allsky/processing.py').read_text(encoding='utf-8'))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ImageProcessor')
    names = {
        'convert_16bit_to_8bit', '_convert_16bit_to_8bit', 'restore_colour_precision',
        'normalize_colour_precision', 'finish_colour_precision', 'drawDetections', '_drawDetections',
        '_white_balance_mtf', '_generate_white_balance_lut', '_apply_gamma_correction',
        'saturation_adjust', '_saturation_adjust', '_sharpen', '_white_balance_auto_bgr',
    }
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(cv2=cv2, numpy=numpy, constants=constants, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'colour-precision-methods', 'exec'), namespace)
    obj = namespace['ImageProcessor']()
    obj.focus_mode = False
    obj.config = {}
    obj.night_av = [False, False]
    obj._gamma_lut = obj._gamma_lut_gamma = None
    obj._wb_mtf_night = None
    obj._wbb_mtf_lut = obj._wbg_mtf_lut = obj._wbr_mtf_lut = None
    obj._colour_precision_image = None
    obj._colour_detection_image = None
    obj._colour_precision_active = False
    obj.max_bit_depth = 16
    obj.getLatestImage = lambda: SimpleNamespace(image_bitpix=16, binning=1)
    return obj


@pytest.mark.parametrize('bits', [8, 10, 12, 14, 16])
def test_detection_stays_8bit_and_colour_restores_full_range(processor, bits):
    maximum = (1 << bits) - 1
    raw = numpy.array([[[0, maximum // 2, maximum], [maximum, 0, maximum // 4]]], dtype=numpy.uint16)
    unchanged = raw.copy()
    processor.image = raw
    processor.max_bit_depth = bits
    processor.convert_16bit_to_8bit(preserve_colour=True)
    numpy.testing.assert_array_equal(processor.image, raw >> (bits - 8))
    assert processor.image.dtype == numpy.uint8
    processor.restore_colour_precision()
    processor.normalize_colour_precision()
    numpy.testing.assert_allclose(processor.image, numpy.rint(raw.astype(float) * 65535 / maximum), atol=1)
    assert processor.image.dtype == numpy.uint16
    processor.finish_colour_precision()
    numpy.testing.assert_array_equal(processor.image, numpy.rint(raw.astype(float) * 255 / maximum))
    numpy.testing.assert_array_equal(raw, unchanged)


@pytest.mark.parametrize('case', ['focus', 'mono', '8bit', 'viewer'])
def test_legacy_paths_are_unchanged(processor, case):
    raw = numpy.arange(36, dtype=numpy.uint16).reshape(3, 4, 3) * 1700
    if case == 'mono':
        raw = raw[:, :, 0]
    if case == '8bit':
        raw = (raw >> 8).astype(numpy.uint8)
        processor.getLatestImage = lambda: SimpleNamespace(image_bitpix=8)
    processor.focus_mode = case == 'focus'
    processor.image = raw
    expected = raw if case == '8bit' else (raw >> 8).astype(numpy.uint8)
    processor.convert_16bit_to_8bit(preserve_colour=case != 'viewer')
    processor.restore_colour_precision()
    processor.normalize_colour_precision()
    processor.finish_colour_precision()
    numpy.testing.assert_array_equal(processor.image, expected)
    assert processor.image.dtype == numpy.uint8


def test_draw_overlay_survives_without_mutating_source(processor):
    raw = numpy.full((3, 4, 3), 12001, dtype=numpy.uint16)
    processor.image = raw
    processor.convert_16bit_to_8bit(preserve_colour=True, preserve_detections=True)
    # Line/star detectors annotate the 8-bit input before the draw overlay.
    processor.image[0, 1] = (0, 255, 0)

    def draw(image, binning):
        image[1, 2] = (255, 0, 128)
        return image

    processor._draw = SimpleNamespace(main=draw)
    processor.drawDetections()
    processor.restore_colour_precision()
    assert numpy.all(raw == 12001)
    assert numpy.all(processor.image[2] == 12001)
    processor.finish_colour_precision()
    numpy.testing.assert_array_equal(processor.image[1, 2], [255, 0, 128])
    numpy.testing.assert_array_equal(processor.image[0, 1], [0, 255, 0])


def test_no_detection_drawing_needs_no_second_8bit_copy(processor):
    processor.image = numpy.full((3, 4, 3), 12001, dtype=numpy.uint16)
    processor.convert_16bit_to_8bit(preserve_colour=True)
    assert processor._colour_detection_image is None


def test_lut_caches_follow_image_dtype(processor):
    scnr = IndiAllskyScnr({'SCNR_MTF_MIDTONES': .51}, [False, False])
    for dtype in (numpy.uint8, numpy.uint16, numpy.uint8):
        maximum = numpy.iinfo(dtype).max
        original = numpy.array([[[0, maximum // 2, maximum]]], dtype=dtype)
        processor.image = original.copy()
        processor._white_balance_mtf(.4, .45, .55)
        assert processor._wbb_mtf_lut.dtype == dtype
        assert processor._wbb_mtf_lut.size == maximum + 1
        processor._apply_gamma_correction(1.565)
        assert processor._gamma_lut.dtype == dtype
        assert processor._gamma_lut.size == maximum + 1
        result = scnr.green_mtf(original)
        assert result.dtype == dtype
        assert scnr._mtf_lut.size == maximum + 1


@pytest.mark.parametrize('dtype', [numpy.uint8, numpy.uint16])
def test_scnr_neutral_does_not_overflow(dtype):
    maximum = numpy.iinfo(dtype).max
    image = numpy.array([[[maximum, maximum, maximum], [maximum, maximum, 0]]], dtype=dtype)
    scnr = IndiAllskyScnr({}, [False, False])
    result = scnr.average_neutral(image)
    numpy.testing.assert_array_equal(result[0, :, 1], [maximum, maximum // 2])
    numpy.testing.assert_array_equal(scnr.maximum_neutral(image), image)


def test_auto_white_balance_preserves_16bit_range(processor):
    original = numpy.array([[[12000, 20000, 35000], [30000, 50000, 65000]]], dtype=numpy.uint16)
    means = original.mean(axis=(0, 1))
    expected = numpy.clip(numpy.rint(original.astype(float) * means.mean() / means), 0, 65535)
    processor.image = original.copy()
    processor._white_balance_auto_bgr()
    assert processor.image.dtype == numpy.uint16
    numpy.testing.assert_allclose(processor.image, expected, atol=1)


@pytest.mark.parametrize('factor', [0, 1, 1.3])
def test_16bit_saturation_endpoints(processor, factor):
    original = numpy.array([[[0, 0, 0], [65535, 65535, 65535], [45000, 18000, 9000]]], dtype=numpy.uint16)
    processor.image = original.copy()
    processor._saturation_adjust(factor)
    assert processor.image.dtype == numpy.uint16
    if factor == 0:
        expected = numpy.repeat(original.max(axis=2, keepdims=True), 3, axis=2)
        numpy.testing.assert_array_equal(processor.image, expected)
    elif factor == 1:
        numpy.testing.assert_allclose(processor.image, original, atol=1)
    else:
        numpy.testing.assert_array_equal(processor.image[0, :2], original[0, :2])
        assert processor.image[0, 2, 0] == 45000
        assert processor.image[0, 2, 2] < original[0, 2, 2]


def test_day_saturation_knee_preserves_smooth_high_saturation_gradient_and_source(processor):
    saturation = numpy.linspace(.9, 1.1, 201)
    minimum = numpy.rint(60000 * (1 - saturation / 1.3)).astype(numpy.uint16)
    row = numpy.stack((numpy.full_like(minimum, 60000), (minimum.astype(int) + 60000) // 2,
                       minimum), axis=1).astype(numpy.uint16)
    original = numpy.tile(row, (271, 1, 1))  # Cross both 128-row block boundaries.
    unchanged = original.copy()
    processor.image = original
    processor._saturation_adjust(1.3)
    legacy = processor.image.copy()
    processor.image = original
    processor._saturation_adjust(1.3, saturation_knee=.95)
    result = processor.image
    actual = (result[0].max(axis=1).astype(float) - result[0].min(axis=1)) / 60000
    scaled_input = (60000 - minimum.astype(float)) / 60000 * 1.3
    numpy.testing.assert_array_equal(result[:, scaled_input < .95], legacy[:, scaled_input < .95])
    assert numpy.all(numpy.diff(actual) > 0)  # No flat clipped plateau above the knee.
    assert numpy.diff(actual).max() < .0011  # No discontinuity at the knee.
    assert actual[-1] < 1
    # At requested saturation 1.05 the gentle shoulder is approximately .99323.
    assert actual[150] == pytest.approx(.99323, abs=4e-5)
    numpy.testing.assert_array_equal(result, numpy.broadcast_to(result[0], result.shape))
    numpy.testing.assert_array_equal(original, unchanged)


@pytest.mark.parametrize('factor,knee', [(1.3, 1.0), (0, .95), (.7, .95), (1, .95)])
def test_saturation_knee_keeps_legacy_full_night_and_nonboosted_values(processor, factor, knee):
    original = numpy.array([[[60000, 24000, 1200], [12345, 32768, 65535], [21000, 21000, 21000]]],
                           dtype=numpy.uint16)
    hsv = cv2.cvtColor(original.astype(numpy.float32) * (1 / 65535), cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] *= factor
    numpy.minimum(hsv[:, :, 1], 1, out=hsv[:, :, 1])
    expected = numpy.rint(numpy.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 65535, 0, 65535))
    processor.image = original
    processor._saturation_adjust(factor, saturation_knee=knee)
    numpy.testing.assert_array_equal(processor.image, expected.astype(numpy.uint16))


@pytest.mark.parametrize('night,shared,enabled,weight,expected', [
    (False, False, False, None, .95), (True, False, False, None, 1),
    (False, True, False, None, 1), (False, True, True, 0, 1),
    (True, False, True, -.2, .95), (True, False, True, .4, .97),
    (False, False, True, 1.2, 1), (False, False, False, .8, .95),
    (True, False, True, None, 1),
])
def test_saturation_knee_follows_colour_mode_and_enabled_twilight_weight(
        processor, night, shared, enabled, weight, expected):
    processor.config = dict(USE_NIGHT_COLOR=shared, SATURATION_FACTOR=1.3,
                            SATURATION_FACTOR_DAY=1.4, TWILIGHT_TRANSITION={'ENABLE': enabled})
    if weight is not None:
        processor.config['_TWILIGHT_WEIGHT'] = weight
    processor.night_av[constants.NIGHT_NIGHT] = night
    processor.image = numpy.full((2, 2, 3), 10000, numpy.uint16)
    calls = []
    processor._saturation_adjust = lambda factor, saturation_knee=1: calls.append((factor, saturation_knee))
    assert processor.saturation_adjust() is True
    assert len(calls) == 1
    assert calls[0] == pytest.approx((1.3 if night or shared else 1.4, expected))


@pytest.mark.parametrize('case', ['focus', 'mono', 'factor_one'])
def test_saturation_wrapper_keeps_existing_noop_gates(processor, case):
    processor.config = dict(USE_NIGHT_COLOR=False, SATURATION_FACTOR_DAY=1 if case == 'factor_one' else 1.3)
    original = numpy.full((2, 2) if case == 'mono' else (2, 2, 3), 10000, numpy.uint16)
    processor.image = original
    processor.focus_mode = case == 'focus'
    calls = []
    processor._saturation_adjust = lambda *args, **kwargs: calls.append(args)
    assert processor.saturation_adjust() is None
    assert processor.image is original and not calls


def test_8bit_colour_keeps_legacy_rounding(processor):
    image = numpy.arange(256, dtype=numpy.uint8).reshape(16, 16)
    processor.image = image.copy()
    processor._apply_gamma_correction(1.565)
    expected = (((image.astype(numpy.float32) / 255) ** (1 / 1.565)) * 255).astype(numpy.uint8)
    numpy.testing.assert_array_equal(processor.image, expected)
    image = numpy.dstack([image, numpy.flip(image, 0), numpy.flip(image, 1)])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)
    processor.image = image.copy()
    processor._saturation_adjust(1.3, saturation_knee=.95)
    numpy.testing.assert_array_equal(processor.image, cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))


@pytest.mark.parametrize('gamma,saturation,sharpen', [(1.85, 1.3, .75), (.87, 1.1, .2)])
def test_smooth_colour_matches_continuous_reference(processor, gamma, saturation, sharpen):
    # A smooth sky ramp crosses HSV sector and row-block boundaries.
    x = numpy.linspace(.02, .6, 287, dtype=numpy.float32)
    image = numpy.tile(numpy.stack([x + .25, x + .1, x], axis=1), (271, 1, 1))
    image = numpy.rint(image * 65535).astype(numpy.uint16)
    reference = image.astype(numpy.float32) / 65535
    hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = numpy.minimum(hsv[:, :, 1] * saturation, 1)
    reference = numpy.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), 0, 1) ** (1 / gamma)
    reference = cv2.addWeighted(reference, 1 + sharpen, cv2.GaussianBlur(reference, (0, 0), 2), -sharpen, 0)
    reference = numpy.clip(numpy.rint(reference * 255), 0, 255).astype(numpy.uint8)
    processor.image = image
    processor._saturation_adjust(saturation)
    processor._apply_gamma_correction(gamma)
    processor.image = processor._sharpen(sharpen)
    processor._colour_precision_active = True
    processor.finish_colour_precision()
    assert numpy.abs(processor.image.astype(int) - reference.astype(int)).max() <= 1


def test_worker_cycle_timer_includes_context_teardown():
    from indi_allsky.capture_period import CaptureSequenceTracker, read_inflight, set_inflight
    source = ast.parse((ROOT / 'indi_allsky/image.py').read_text(encoding='utf-8'))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ImageWorker')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'saferun')
    events = []

    class Context:
        def __enter__(self):
            events.append('enter')

        def __exit__(self, *args):
            events.append('exit')

    tasks = iter([dict(exp_time=123, exposure=.1), dict(stop=True)])
    ticks = iter([10.0, 31.0])
    logger = SimpleNamespace(info=lambda *args: events.append(args), warning=lambda *args: None)
    namespace = dict(time=SimpleNamespace(monotonic=lambda: next(ticks)), logger=logger,
                     app=SimpleNamespace(app_context=Context), read_inflight=read_inflight,
                     set_inflight=set_inflight)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'worker-cycle-method', 'exec'), namespace)
    worker = SimpleNamespace(image_q=SimpleNamespace(get=lambda **kwargs: next(tasks)), _shutdown=False,
                             backlog_state=None,
                             period_inflight=None, capture_sequence=CaptureSequenceTracker(),
                             _checkCaptureSequence=lambda *args: None,
                             processImage=lambda task: events.append('process'),
                             image_processor=SimpleNamespace(realtimeKeogramDataSave=lambda: None))
    namespace['saferun'](worker)
    assert events[:5] == ['enter', 'exit', 'enter', 'process', 'exit']
    assert events[5][1:] == (21.0, 123, .1, False)
