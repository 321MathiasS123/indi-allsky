"""Raw horizon headroom, excluding angularly compact moon and star light."""
import math

import numpy


def capture_twilight(exp_date, exposure, position):
    """Solar envelope and midpoint for a capture timestamped at receipt/end."""
    from datetime import datetime, timezone
    import ephem

    try:
        midpoint = exp_date.timestamp() - float(exposure) / 2.0
        latitude, longitude, elevation = map(float, position[:3])
        if (not all(math.isfinite(value) for value in (midpoint, latitude, longitude, elevation))
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            return 0.0, 0.0
        observer = ephem.Observer()
        observer.lat, observer.lon = math.radians(latitude), math.radians(longitude)
        observer.elevation, observer.pressure = elevation, 0
        observer.date = datetime.fromtimestamp(midpoint, timezone.utc)
        return twilight_weight(math.degrees(ephem.Sun(observer).alt)), midpoint
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return 0.0, 0.0


def twilight_weight(sun_alt):
    """Smoothly enter and leave protection over -14 to -2 degrees."""
    try:
        altitude = float(sun_alt)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(altitude) or not -14.0 < altitude < -2.0:
        return 0.0
    position = min((altitude + 14.0) / 2.0, (-2.0 - altitude) / 2.0, 1.0)
    return position * position * (3.0 - 2.0 * position)


def measure_ring(data, mask, bit_depth, config, binning=1):
    """Return normalized, broad any-channel headroom in the outer 30% radius.

    Pixels and a supplied detection mask must already share the final rotated,
    flipped and cropped geometry, before display scaling or borders. Match the
    displayed image-circle centre, including the sign of the vertical offset.
    A 20-degree angular opening removes isolated stars and the moon with a
    bounded halo, while retaining even radially thin, broad horizon clipping.
    Missing mask sectors break angular support rather than being interpolated.
    Invalid geometry or an empty usable ring disables this optional meter.
    """
    if (not isinstance(data, numpy.ndarray) or not data.size
            or data.dtype not in (numpy.uint8, numpy.uint16)
            or data.ndim not in (2, 3)
            or (data.ndim == 3 and data.shape[2] != 3)):
        return 0.0
    if (mask is not None and (not isinstance(mask, numpy.ndarray)
                            or mask.shape != data.shape[:2]
                            or mask.dtype.kind not in 'buif'
                            or not numpy.isfinite(mask).all())):
        return 0.0
    try:
        depth = int(bit_depth)
        if depth != bit_depth or not 1 <= depth <= numpy.iinfo(data.dtype).bits:
            return 0.0
        scale = float(binning)
        circle = config.get('IMAGE_CIRCLE_MASK', {})
        diameter = float(circle.get('DIAMETER', 0)) if circle.get('ENABLE') else 0.0
        if diameter <= 0:
            diameter = float(config['LENS_IMAGE_CIRCLE'])
        offset_x = float(config.get('LENS_OFFSET_X', 0))
        offset_y = float(config.get('LENS_OFFSET_Y', 0))
        if (not all(math.isfinite(value) for value in (scale, diameter, offset_x, offset_y))
                or scale <= 0 or diameter <= 0):
            return 0.0
        height, width = data.shape[:2]
        center_x = int(width / 2) + int(offset_x / scale)
        center_y = int(height / 2) - int(offset_y / scale)
        radius = int(diameter / (2.0 * scale))
        if radius <= 0:
            return 0.0
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return 0.0

    x = numpy.arange(width, dtype=numpy.float32) - center_x
    y = numpy.arange(height, dtype=numpy.float32) - center_y
    distance = numpy.hypot(y[:, None], x)
    valid = (distance > 0.70 * radius) & (distance <= radius)
    if mask is not None:
        valid &= mask != 0
    if not numpy.any(valid):
        return 0.0

    # A sector spans at least three inner-ring pixels so even a one-pixel
    # horizon arc has angular support at small resolutions. Normal camera
    # sizes use half-degree sectors.
    sectors = min(720, max(36, int(2.0 * math.pi * 0.70 * radius / 3.0)))
    rows, columns = numpy.nonzero(valid)
    angles = numpy.arctan2(rows - center_y, columns - center_x)
    bins = numpy.floor((angles + math.pi) * sectors / (2.0 * math.pi)).astype(numpy.intp)
    numpy.minimum(bins, sectors - 1, out=bins)
    values = data[rows, columns]
    if data.ndim == 3:
        values = values.max(axis=1)
    values = numpy.minimum(values.astype(numpy.float32) / ((1 << depth) - 1), 1.0)
    feather = numpy.clip((distance[valid] / radius - 0.70) / 0.05, 0.0, 1.0)
    values *= feather * feather * (3.0 - 2.0 * feather)
    # Keep modest radial coherence: a dense star field at unrelated radii
    # must not assemble a continuous bright angular profile. Overlapping
    # 7.5%-radius bands still admit a thin horizon with gradual radial drift.
    radial_bins = numpy.minimum(((distance[valid] / radius - 0.70) * 40).astype(numpy.intp), 11)
    profile = numpy.zeros((12, sectors), dtype=numpy.float32)
    numpy.maximum.at(profile, (radial_bins, bins), values)
    profile = numpy.maximum(numpy.maximum(profile[:-2], profile[1:-1]), profile[2:])

    # Erode then dilate the circular profile. Zero for an unknown sector is a
    # hard support boundary; a star on either side cannot bridge a mask hole.
    support = int(math.ceil(sectors * 20.0 / 360.0))
    support += 1 - support % 2
    padding = support // 2
    windows = numpy.lib.stride_tricks.sliding_window_view(
        numpy.pad(profile, ((0, 0), (padding, padding)), mode='wrap'), support, axis=1)
    eroded = windows.min(axis=2)
    windows = numpy.lib.stride_tricks.sliding_window_view(
        numpy.pad(eroded, ((0, 0), (padding, padding)), mode='wrap'), support, axis=1)
    return float(windows.max(axis=2).max())
