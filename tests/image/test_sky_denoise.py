"""Single-frame denoising, original-data safety and normal pipeline dispatch."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from indi_allsky import constants, sky_denoise
from indi_allsky.denoise import IndiAllskyDenoise


@pytest.fixture
def scene():
    rng = np.random.default_rng(123)
    image = rng.normal(0.15, 0.015, (384, 384, 3)).astype(np.float32)
    yy, xx = np.ogrid[:384, :384]
    for x, y, amplitude in [(192, 192, 0.4), (230, 145, 0.10)]:
        image += (amplitude * np.exp(-((xx-x)**2 + (yy-y)**2) / 3))[:, :, None]
    return image


def test_reduces_background_noise_and_retains_bright_and_faint_cores(scene):
    before = scene.copy()
    result = sky_denoise.denoise(scene, {}, strength=3)
    assert result[130:160, 130:160].std() < scene[130:160, 130:160].std() * 0.65
    for x, y in [(192, 192), (230, 145)]:
        assert np.max(np.abs(result[y, x] - scene[y, x])) < 0.01
    np.testing.assert_array_equal(scene, before)


def test_strengths_are_distinct_and_gentle_settings_blend_toward_original(scene):
    results = [sky_denoise.denoise(scene, {}, strength=s) for s in range(1, 6)]
    np.testing.assert_allclose(results[0], scene + (results[2] - scene) * 0.5, atol=1e-7)
    np.testing.assert_allclose(results[1], scene + (results[2] - scene) * 0.75, atol=1e-7)
    for left, right in zip(results, results[1:]):
        assert np.isfinite(right).all()
        assert not np.array_equal(left, right)


@pytest.mark.parametrize('dtype,maximum', [(np.uint8, 255), (np.uint16, 65535), (np.float32, 1)])
@pytest.mark.parametrize('mono', [True, False])
def test_preserves_dtype_shape_and_range(scene, dtype, maximum, mono):
    image = (scene * maximum).astype(dtype)
    if mono:
        image = image[:, :, 0]
    result = sky_denoise.denoise(image, {}, strength=3)
    assert result.shape == image.shape and result.dtype == image.dtype
    assert np.isfinite(result).all() and result.min() >= 0 and result.max() <= maximum


@pytest.mark.parametrize('shape,value', [((16, 16), 0), ((100, 100), 0), ((100, 100, 3), 0.4)])
def test_small_black_and_flat_frames_stay_finite_and_unchanged(shape, value):
    image = np.full(shape, value, np.float32)
    np.testing.assert_array_equal(sky_denoise.denoise(image, {}), image)


def test_empty_sky_and_disabled_strength_are_noops(scene):
    np.testing.assert_array_equal(sky_denoise.denoise(scene, {'LENS_OFFSET_X': 10000}), scene)
    np.testing.assert_array_equal(sky_denoise.denoise(scene, {}, strength=0), scene)


def test_low_bit_depth_data_in_uint16_container_is_denoised():
    rng = np.random.default_rng(321)
    image = rng.normal(100, 10, (128, 128, 3)).astype(np.uint16)
    result = sky_denoise.denoise(image, {})
    assert result[50:70, 50:70].std() < image[50:70, 50:70].std() * 0.65


def test_cold_bayer_repair_is_local_and_does_not_mutate_capture():
    image = np.full((128, 128), 10000, np.uint16)
    image[60, 60] = 100
    image[70, 70] = image[70, 72] = 100  # Connected feature, not an isolated outlier.
    image[1, 1] = 100  # Incomplete neighbourhood.
    original = image.copy()
    result = sky_denoise.repair_bayer(image, {})
    assert result[60, 60] == 10000
    assert np.count_nonzero(result != image) == 1
    np.testing.assert_array_equal(image, original)


def test_geometry_undoes_orientation_and_binning():
    cfg = dict(LENS_IMAGE_CIRCLE=800, LENS_OFFSET_X=40, LENS_OFFSET_Y=20,
               IMAGE_FLIP_H=True, IMAGE_ROTATE='ROTATE_90_CLOCKWISE')
    assert sky_denoise._sky_geometry((500, 600), cfg, 2) == (290, 270, 180)
    cfg.update(IMAGE_ROTATE='', IMAGE_ROTATE_ANGLE=90)
    np.testing.assert_allclose(sky_denoise._sky_geometry((500, 600), cfg, 2), (310, 230, 180))


def test_extra_dark_patch_repair_does_not_cross_horizon_guard():
    baseline = np.full((240, 240, 3), 0.1, np.float32)
    candidate = baseline + 0.01
    yy, xx = np.ogrid[:240, :240]
    valid = (xx-120)**2 + (yy-120)**2 < 110**2
    result, weight = sky_denoise._protect_horizon(baseline, candidate, valid)
    np.testing.assert_array_equal(result[weight == 0], baseline[weight == 0])
    np.testing.assert_array_equal(result[weight == 1], candidate[weight == 1])


@pytest.fixture
def processor_class():
    # Exercise the real pipeline methods without Linux camera/DB services.
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'_debayer', 'denoise', '_denoise', 'getLatestImage'}]
    ns = dict(__name__='indi_allsky.processing', __package__='indi_allsky', cv2=cv2,
              numpy=np, constants=constants, logger=logging.getLogger('test'))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    return ns['ImageProcessor']


@pytest.mark.parametrize('night,use_night,method,day_method,enabled', [
    (True, False, 'star_aware', '', True), (False, False, 'star_aware', '', False),
    (False, True, 'star_aware', '', True), (True, False, 'wavelet', '', False),
    (True, True, '', '', False), (False, False, '', 'star_aware', True),
])
@pytest.mark.parametrize('focus', [False, True])
def test_bayer_hook_and_denoise_agree_on_day_night_selection(
        processor_class, monkeypatch, night, use_night, method, day_method, enabled, focus):
    obj = processor_class()
    obj.config = dict(IMAGE_DENOISE=method, IMAGE_DENOISE_DAY=day_method, USE_NIGHT_COLOR=use_night)
    obj.focus_mode = focus
    obj.night_av = [night] + [False] * 7
    raw = np.full((64, 64), 10000, np.uint16)
    ref = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=16,
                          image_bayerpat='RGGB', binning=2)
    obj.image_list = [ref]
    obj._ImageProcessor__cfa_bgr_map = {'RGGB': cv2.COLOR_BAYER_BG2BGR}
    calls = []

    def repair(data, config, binning):
        calls.append(('bayer', binning))
        return data.copy()

    def denoise(data, config, binning, strength):
        calls.append(('denoise', binning, strength))
        return data

    monkeypatch.setattr(sky_denoise, 'repair_bayer', repair)
    monkeypatch.setattr(sky_denoise, 'denoise', denoise)
    obj._ia_denoise = IndiAllskyDenoise(obj.config, obj.night_av)
    obj._ia_denoise.wavelet = lambda data: data  # Existing method is dispatched normally.
    obj.image = obj._debayer(ref)
    obj.denoise()
    assert calls == ([('bayer', 2), ('denoise', 2, 3)] if enabled and not focus else [])
    assert ref.hdulist[0].data is raw and np.all(raw == 10000)


def test_separate_day_strength_is_used(monkeypatch, scene):
    denoiser = IndiAllskyDenoise(dict(USE_NIGHT_COLOR=False,
        IMAGE_DENOISE_STRENGTH=2, IMAGE_DENOISE_STRENGTH_DAY=4), [False] * 8)
    received = []
    monkeypatch.setattr(sky_denoise, 'denoise',
        lambda image, config, binning, strength: received.append(strength) or image)
    denoiser.star_aware(scene)
    assert received == [4]


def test_disabled_bayer_repair_does_not_require_filter_state(processor_class):
    obj = processor_class()
    obj.config = {}
    obj._ImageProcessor__cfa_bgr_map = {'RGGB': cv2.COLOR_BAYER_BG2BGR}
    raw = np.full((64, 64), 10000, np.uint16)
    ref = SimpleNamespace(hdulist=[SimpleNamespace(data=raw)], image_bitpix=16,
                          image_bayerpat='RGGB', binning=1)
    np.testing.assert_array_equal(obj._debayer(ref), cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2BGR))


@pytest.mark.parametrize('shape,tile', [((17, 19), 64), ((256, 256), 64),
                                        ((271, 517), 128), ((515, 523), 128)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_noise_grid_matches_independent_tile_mads_without_mutating_source(monkeypatch, shape, tile, dtype):
    monkeypatch.setattr(sky_denoise.os, 'cpu_count', lambda: 4)
    rng = np.random.default_rng(742)
    storage = rng.normal(0, 0.05, (shape[0] * 2, shape[1] * 2)).astype(dtype)
    data = storage[::2, ::2]  # Channel/Bayer views need not be contiguous.
    data[0, 0], data[-1, -1] = -1, 1
    original = storage.copy()
    grid = np.empty(((shape[0] + tile - 1) // tile, (shape[1] + tile - 1) // tile), np.float32)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            values = data[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile]
            grid[y, x] = max(float(np.median(np.abs(values - np.median(values)))) * 1.4826, 1e-6)
    expected = cv2.resize(grid, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(sky_denoise._noise_grid(data, tile), expected)
    np.testing.assert_array_equal(storage, original)


def test_noise_grid_preserves_floor_and_nan_tiles(monkeypatch):
    monkeypatch.setattr(sky_denoise.os, 'cpu_count', lambda: 4)
    data = np.zeros((512, 512), np.float32)
    np.testing.assert_array_equal(sky_denoise._noise_grid(data), np.full(data.shape, 1e-6, np.float32))
    data[:128, :128] = np.nan
    grid = np.full((4, 4), 1e-6, np.float32)
    grid[0, 0] = np.nan
    expected = cv2.resize(grid, (512, 512), interpolation=cv2.INTER_LINEAR)
    np.testing.assert_array_equal(sky_denoise._noise_grid(data), expected)
