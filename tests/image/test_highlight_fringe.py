import cv2
import numpy
import pytest

from indi_allsky.highlight_fringe import reduce_highlight_fringes


LUMA = numpy.array([0.114, 0.587, 0.299])


def scene(colour=(0.70, 0.55, 0.30), dtype=numpy.uint16):
    floating = numpy.broadcast_to(numpy.array(colour), (128, 192, 3)).copy()
    floating[:, 105:140] = 0.99
    return numpy.rint(floating * numpy.iinfo(dtype).max).astype(dtype)


@pytest.mark.parametrize('colour', [(0.45, 0.45, 0.45), (0.8, 0.5, 0.25),
                                    (0.3, 0.6, 0.85), (0.8, 0.5, 0.8)])
@pytest.mark.parametrize('dtype', [numpy.uint8, numpy.uint16])
def test_broad_colours_beside_white_remain_identical(colour, dtype):
    image = scene(colour, dtype)
    unchanged = image.copy()
    result = reduce_highlight_fringes(image)
    numpy.testing.assert_array_equal(result, unchanged)
    numpy.testing.assert_array_equal(image, unchanged)


def test_broad_pink_rectangle_corners_are_protected():
    image = scene()
    image[:] = numpy.rint(numpy.array([0.7, 0.55, 0.3]) * 65535)
    image[22:106, 55:105] = numpy.rint(numpy.array([0.8, 0.5, 0.8]) * 65535)
    image[16:112, 105:130] = round(0.99 * 65535)
    numpy.testing.assert_array_equal(reduce_highlight_fringes(image), image)


@pytest.mark.parametrize('colour', [(0.8, 0.45, 0.8), (0.85, 0.45, 0.25)])
@pytest.mark.parametrize('dtype', [numpy.uint8, numpy.uint16])
def test_narrow_fringe_reduced_without_changing_brightness_or_input(colour, dtype):
    image = scene(dtype=dtype)
    image[:, 101:105] = numpy.rint(numpy.array(colour) * numpy.iinfo(dtype).max)
    unchanged = image.copy()
    result = reduce_highlight_fringes(image)
    delta = result.astype(numpy.int32) - image
    assert numpy.any(delta[:, 101:105])
    assert numpy.all(delta[:, 101:105, 0] <= 0)
    assert numpy.all(delta[:, 101:105, 1] >= 0)
    assert numpy.max(numpy.abs(delta @ LUMA)) <= 1.0
    assert result.shape == image.shape and result.dtype == dtype
    numpy.testing.assert_array_equal(image, unchanged)
    numpy.testing.assert_array_equal(result[:, :80], image[:, :80])
    numpy.testing.assert_array_equal(result[:, 105:140], image[:, 105:140])


def test_uint8_matches_quantized_uint16_path():
    image = scene(dtype=numpy.uint8)
    image[10:60, 101:105] = (220, 115, 200)
    image[65:115, 101:105] = (225, 110, 60)
    expanded = image.astype(numpy.uint16) * 257
    expected = cv2.convertScaleAbs(reduce_highlight_fringes(expanded), alpha=1.0 / 257)
    numpy.testing.assert_array_equal(reduce_highlight_fringes(image), expected)


@pytest.mark.parametrize('image', [numpy.zeros((16, 16), numpy.uint16),
                                 numpy.zeros((16, 16, 4), numpy.uint16),
                                 numpy.ones((16, 16, 3), numpy.float32),
                                 numpy.zeros((0, 16, 3), numpy.uint16)])
def test_unsupported_images_are_noops(image):
    assert reduce_highlight_fringes(image) is image


@pytest.mark.parametrize('value', [20000, 65535])
def test_no_highlight_or_no_contrast_returns_input(value):
    image = numpy.full((32, 40, 3), value, dtype=numpy.uint16)
    assert reduce_highlight_fringes(image) is image


def test_fringe_far_from_highlights_is_unchanged():
    image = scene()
    image[:, 20:24] = (60000, 20000, 50000)
    numpy.testing.assert_array_equal(reduce_highlight_fringes(image), image)


def test_extreme_colours_stay_in_gamut_and_preserve_luma():
    image = scene((0.1, 0.1, 0.1))
    image[15:60, 101:105] = (65535, 0, 65535)
    image[65:110, 101:105] = (65535, 0, 0)
    result = reduce_highlight_fringes(image)
    assert numpy.any(result != image)
    assert numpy.max(numpy.abs((result.astype(float) - image) @ LUMA)) <= 1.0
    assert numpy.all(result[:, 101:105, 0] <= image[:, 101:105, 0])
    assert numpy.all(result[:, 101:105, 1] >= image[:, 101:105, 1])


def test_extra_distant_white_pixel_does_not_change_local_context():
    image = numpy.full((192, 256, 3), 20000, dtype=numpy.uint16)
    image[80:100, 140:150] = 65000
    image[65:112, 120:140] = (50000, 35000, 48000)
    image[76:105, 136:140] = (61000, 22000, 57000)
    reference = reduce_highlight_fringes(image)
    expanded = image.copy()
    expanded[10, 10] = 65000
    result = reduce_highlight_fringes(expanded)
    numpy.testing.assert_array_equal(result[50:135, 90:180], reference[50:135, 90:180])
