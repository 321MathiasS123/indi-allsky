"""Circle-bounded smoothing must keep the original noise grid and edge support."""
import numpy as np
import pytest

from indi_allsky import sky_denoise


@pytest.mark.parametrize('geometry', [
    (321, 257, 175),       # enough sky for measured 128-pixel noise tiles
    (72, 81, 190),         # original top and left reflection boundaries
    (602, 472, 180),       # original bottom and right reflection boundaries
    (-110, 257, 140),      # a narrow sliver without qualifying noise tiles
    (321, 257, 1),         # no clear-background statistics
    (321, 257, 1000),      # full frame, no available crop
    (-1000, -1000, 1),     # no valid pixels
])
@pytest.mark.parametrize('split', [False, True])
def test_cropped_preparation_preserves_full_frame_statistics(monkeypatch, geometry, split):
    height, width = 513, 641
    yy, xx = np.ogrid[:height, :width]
    rng = np.random.default_rng(1836)
    image = rng.normal(0.15, 0.02, (height, width, 3)).astype(np.float32)
    # Broad gradients and chromatic texture make shifted tiles observably wrong.
    image[:, :, 0] += 0.04 * np.sin(xx / 19)
    image[:, :, 1] += 0.03 * np.cos(yy / 31)
    valid = sky_denoise._sky_mask(image.shape, geometry)
    mask = np.clip((13 - np.hypot(xx - 315, yy - 251)) / 4, 0, 1).astype(np.float32)
    mask[188:225, 279:343] = 1  # a protected extended feature
    evidence = {'lum': image @ sky_denoise.WEIGHTS, 'mask': mask}
    before = image.copy()
    valid_before = valid.copy()
    evidence_before = {key: value.copy() for key, value in evidence.items()}

    def prepare():
        if split:
            return sky_denoise._prepare_split(image, evidence, valid)
        return sky_denoise._prepare(image, evidence, valid, fine=1.2)

    actual = prepare()
    with monkeypatch.context() as patch:
        patch.setattr(sky_denoise, '_sky_slice', lambda valid, padding: np.s_[:, :])
        expected = prepare()

    for key in expected:
        # Changed crop widths can move OpenCV's SIMD/scalar boundary. Bound
        # that float32 roundoff, including its propagation into noise estimates.
        if key == 'noise':
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-6, atol=1e-12, err_msg=key)
        else:
            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=2e-7, err_msg=key)
        assert actual[key].dtype == expected[key].dtype
    if geometry == (321, 257, 175):
        assert np.all(expected['noise'] > 1e-6)
    np.testing.assert_array_equal(image, before)
    np.testing.assert_array_equal(valid, valid_before)
    for key in evidence:
        np.testing.assert_array_equal(evidence[key], evidence_before[key])
