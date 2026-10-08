"""Cropping local pit repair must preserve complete sky components and edges."""
import warnings

import numpy as np
import pytest

from indi_allsky import sky_denoise


@pytest.mark.parametrize('geometry', [
    (105, 96, 62, 62),       # centred sky
    (32, 28, 67, 49),        # original top and left reflection boundaries
    (197, 182, 51, 64),      # original bottom and right reflection boundaries
    (107, 94, 30, 81),       # eccentric sky region
    (-35, 98, 41, 65),       # narrow visible sliver
    (105, 96, 1000, 1000),   # no crop
    (-1000, -1000, 1, 1),   # no valid pixels
])
@pytest.mark.parametrize('radius,inner', [(3, 1), (12, 4)])
def test_cropped_pit_repair_matches_full_frame(monkeypatch, geometry, radius, inner):
    height, width = 193, 211
    yy, xx = np.ogrid[:height, :width]
    x, y, rx, ry = geometry
    valid = ((xx-x)/rx)**2 + ((yy-y)/ry)**2 < 1
    rng = np.random.default_rng(8192)
    raw = rng.normal(0.2, 0.004, (height, width, 3)).astype(np.float32)
    image = np.full_like(raw, 0.2)
    # Compact lows and positive cores occur both inside the region and beside
    # its boundaries; connected-component decisions must not change on cropping.
    for px, py in [(105, 96), (32, 28), (197, 182), (3, 95), (103, 38)]:
        pit = np.exp(-((xx-px)**2 + (yy-py)**2)/3).astype(np.float32)
        image -= 0.022 * pit[:, :, None]
        star = np.exp(-((xx-px-7)**2 + (yy-py-4)**2)/2).astype(np.float32)
        raw += 0.12 * star[:, :, None]
        image += 0.12 * star[:, :, None]
    evidence = {'lum': raw @ sky_denoise.WEIGHTS}
    before = image.copy()
    raw_before = evidence['lum'].copy()
    valid_before = valid.copy()
    with warnings.catch_warnings():
        # The existing empty-sky statistics contain NaNs; preserve that behavior.
        warnings.simplefilter('ignore', RuntimeWarning)
        actual = sky_denoise._refine_pits(image, evidence, valid, radius=radius, inner=inner)
        with monkeypatch.context() as patch:
            patch.setattr(sky_denoise, '_sky_slice', lambda valid, padding: np.s_[:, :])
            expected = sky_denoise._refine_pits(image, evidence, valid, radius=radius, inner=inner)

    assert actual[0].shape == image.shape and actual[0].dtype == image.dtype
    assert actual[1].shape == valid.shape and actual[1].dtype == np.float32
    # OpenCV may choose different FFT blocks for the smaller convolution input.
    np.testing.assert_allclose(actual[0], expected[0], rtol=0, atol=2e-7)
    np.testing.assert_allclose(actual[1], expected[1], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(actual[0][~valid], image[~valid])
    np.testing.assert_array_equal(actual[1][~valid], np.zeros(np.count_nonzero(~valid), np.float32))
    for key in ['components', 'pixels', 'sky_percentage']:
        np.testing.assert_equal(actual[2][key], expected[2][key])
    for key in ['mean_linear_luma_added', 'max_linear_luma_added']:
        np.testing.assert_allclose(actual[2][key], expected[2][key], rtol=0, atol=2e-7, equal_nan=True)
    if valid[96, 105] and radius == 12:
        assert expected[1][96, 105] > 0.001
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(evidence['lum'], raw_before)
    np.testing.assert_array_equal(valid, valid_before)


@pytest.mark.parametrize('padding', [1, 3, 12, 40])
def test_sky_crop_retains_requested_neighbours(padding):
    valid = np.zeros((151, 173), bool)
    valid[45:71, 61:82] = True
    roi = sky_denoise._sky_slice(valid, padding)
    assert roi == (slice(45-padding, 71+padding), slice(61-padding, 82+padding))
