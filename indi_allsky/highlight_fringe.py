"""Reduce narrow purple and blue excess beside bright, contrasting edges.

This operates on displayed BGR before sharpening and preserves BT.601 display
luma apart from integer rounding. Daytime eligibility belongs to the caller:
small blue details beside bright features cannot always be distinguished from
fringing using one image.
"""
import cv2
import numpy


_LUMA = numpy.array([0.114, 0.587, 0.299], dtype=numpy.float32)
_NEAR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_CONTRAST_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_COLOUR_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_BROAD_KERNEL = numpy.ones((9, 9), dtype=numpy.uint8)


def _thin_excess(colour):
    broad = cv2.morphologyEx(colour, cv2.MORPH_OPEN, _COLOUR_KERNEL)
    # Include corners and borders of broad colour, not just its eroded core.
    broad = cv2.dilate(broad, _BROAD_KERNEL)
    return numpy.maximum(colour - broad, 0)


def reduce_highlight_fringes(image):
    """Return corrected uint8/uint16 BGR, leaving the input unchanged."""
    if image.ndim != 3 or image.shape[2] != 3 or not image.size:
        return image
    if image.dtype not in (numpy.uint8, numpy.uint16):
        return image

    maximum = numpy.iinfo(image.dtype).max
    minimum = numpy.minimum(image[:, :, 0], image[:, :, 1])
    numpy.minimum(minimum, image[:, :, 2], out=minimum)
    x, y, width, height = cv2.boundingRect((minimum > int(0.94 * maximum)).astype(numpy.uint8))
    if not width or not height:
        return image

    # Include the full dependencies of the highlight and colour neighbourhoods.
    padding = 24
    x1, x2 = max(0, x - padding), min(image.shape[1], x + width + padding)
    y1, y2 = max(0, y - padding), min(image.shape[0], y + height + padding)
    original = image[y1:y2, x1:x2]
    if image.dtype == numpy.uint8:
        original = original.astype(numpy.uint16) * 257
    source = original.astype(numpy.float32) / 65535
    white = minimum[y1:y2, x1:x2].astype(numpy.float32)
    if image.dtype == numpy.uint8:
        white *= 257
    white = numpy.clip((white / 65535 - 0.94) / 0.04, 0, 1)
    near = cv2.dilate(white, _NEAR_KERNEL)
    near = cv2.GaussianBlur(near, (0, 0), 0.8)
    luma = source @ _LUMA
    contrast = cv2.dilate(luma, _CONTRAST_KERNEL) - cv2.erode(luma, _CONTRAST_KERNEL)
    contrast = numpy.clip((contrast - 0.03) / 0.10, 0, 1)
    if not numpy.any((near > 0) & (contrast > 0)):
        return image

    purple = numpy.maximum(numpy.minimum(source[:, :, 0], source[:, :, 2]) - source[:, :, 1], 0)
    if numpy.any(purple):
        correction = 0.9 * near * contrast * _thin_excess(purple)
        source[:, :, 0] -= 0.587 * correction
        source[:, :, 1] += 0.413 * correction
        source[:, :, 2] -= 0.587 * correction
        # Preserve the reviewed intermediate rounding before measuring blue.
        corrected = numpy.rint(source * 65535).astype(numpy.uint16)
        source = corrected.astype(numpy.float32) / 65535

    blue = numpy.maximum(source[:, :, 0] - numpy.maximum(source[:, :, 1], source[:, :, 2]), 0)
    if numpy.any(blue):
        correction = numpy.minimum(blue, 0.85 * near * contrast * _thin_excess(blue))
        source[:, :, 0] -= 0.886 * correction
        source[:, :, 1] += 0.114 * correction
        source[:, :, 2] += 0.114 * correction
    corrected = numpy.rint(source * 65535).astype(numpy.uint16)
    if image.dtype == numpy.uint8:
        corrected = cv2.convertScaleAbs(corrected, alpha=255.0 / 65535.0)
    if numpy.array_equal(corrected, image[y1:y2, x1:x2]):
        return image
    result = image.copy()
    result[y1:y2, x1:x2] = corrected
    return result
