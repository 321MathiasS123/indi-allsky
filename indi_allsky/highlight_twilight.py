"""Raw horizon headroom with a phase-dependent allowance for night sources."""
import math

import numpy
import cv2


def measure_channels(data, mask, bit_depth, config):
    """Normalized BGR tail levels for forecasting, not a clipping-area policy.

    Omit the total area allowed by the existing any-channel upper limit. Tiny
    sources below that area cannot dominate this statistic. The caller rejects
    raw-censored levels before replacing eligible values with calibrated ones;
    a clipped plateau cannot certify that the scene stopped brightening.
    """
    unavailable = (0.0, 0.0, 0.0)
    if (not isinstance(data, numpy.ndarray) or not data.size
            or data.dtype not in (numpy.uint8, numpy.uint16)
            or data.ndim != 3 or data.shape[2] != 3
            or not isinstance(mask, numpy.ndarray) or mask.shape != data.shape[:2]
            or mask.dtype.kind not in 'buif' or not numpy.isfinite(mask).all()):
        return unavailable
    try:
        depth = int(bit_depth)
        settings = config.get('HIGHLIGHT_PROTECTION', {})
        fraction = (float(settings.get('ANY_TARGET', 2.0))
                    + float(settings.get('ANY_DEV', 0.4))) / 100.0
        if (depth != bit_depth or not 1 <= depth <= numpy.iinfo(data.dtype).bits
                or not math.isfinite(fraction) or not 0 < fraction < 1):
            return unavailable
    except (AttributeError, TypeError, ValueError, OverflowError):
        return unavailable
    valid = numpy.asarray(mask != 0, dtype=numpy.uint8)
    count = cv2.countNonZero(valid)
    if not count:
        return unavailable
    maximum = (1 << depth) - 1
    rank = count - 1 - int(count * fraction)
    result = []
    for channel in range(3):
        histogram = cv2.calcHist([data], [channel], valid, [maximum + 1], [0, maximum + 1]).ravel()
        # Out-of-range samples cannot provide a trustworthy source trend.
        if int(histogram.sum()) != count:
            result.append(0.0)
            continue
        value = int(numpy.searchsorted(numpy.cumsum(histogram), rank + 1))
        result.append(value / maximum)
    return tuple(result)


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
    """Keep two-degree fades at -14..-12 and -2..0 degrees."""
    try:
        altitude = float(sun_alt)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(altitude) or not -14.0 < altitude < 0.0:
        return 0.0
    position = min((altitude + 14.0) / 2.0, -altitude / 2.0, 1.0)
    return position * position * (3.0 - 2.0 * position)


def capture_moon(exp_date, exposure, position):
    """Use capture-time Moon phase and altitude, independently of render lag."""
    from datetime import datetime, timezone
    import ephem

    try:
        midpoint = exp_date.timestamp() - float(exposure) / 2.0
        latitude, longitude, elevation = map(float, position[:3])
        if (not all(math.isfinite(value) for value in (midpoint, latitude, longitude, elevation))
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180):
            return 0.0, -90.0
        observer = ephem.Observer()
        observer.lat, observer.lon = math.radians(latitude), math.radians(longitude)
        observer.elevation, observer.pressure = elevation, 0
        observer.date = datetime.fromtimestamp(midpoint, timezone.utc)
        moon = ephem.Moon(observer)
        return float(moon.moon_phase) * 100.0, math.degrees(moon.alt)
    except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
        return 0.0, -90.0


def clipping_budget(moon_phase, moon_alt, config):
    """Fit a monotone phase curve through new, half and full Moon allowances.

    Settings are percentages of the usable ring, including the small transient
    margin. Below the horizon only the stars allowance applies. The Moon's
    contribution enters smoothly from -1 to 0 degrees geometric altitude,
    covering the visible upper limb and ordinary atmospheric refraction.
    """
    try:
        settings = config.get('HIGHLIGHT_PROTECTION', {})
        stars = float(settings.get('TWILIGHT_STAR_ALLOWANCE', 0.1))
        half = float(settings.get('TWILIGHT_MOON_ALLOWANCE_50', 1.8))
        full = float(settings.get('TWILIGHT_MOON_ALLOWANCE_FULL', 24.0))
        if (not all(math.isfinite(value) for value in (stars, half, full))
                or not 0 <= stars <= half <= full <= 50):
            return 0.0
        phase, altitude = float(moon_phase), float(moon_alt)
        if not math.isfinite(phase) or not math.isfinite(altitude):
            return stars / 100.0
        phase = min(1.0, max(0.0, phase / 100.0))
        def moonlight(illuminated):
            # Krisciunas & Schaefer (1991), equation 9, with the smooth
            # opposition enhancement described there for phase angles <7deg.
            angle = math.degrees(math.acos(2.0 * illuminated - 1.0))
            opposition = max(0.0, 1.0 - angle / 7.0)
            opposition *= opposition * (3.0 - 2.0 * opposition)
            return 10 ** (-0.4 * (0.026 * angle + 4e-9 * angle ** 4)) * (1.0 + 0.35 * opposition)

        if stars < half < full:
            dark, bright = moonlight(0.0), moonlight(1.0)
            half_light = (moonlight(0.5) - dark) / (bright - dark)
            light = max(0.0, (moonlight(phase) - dark) / (bright - dark))
            power = math.log((half - stars) / (full - stars)) / math.log(half_light)
            allowance = stars + (full - stars) * light ** power
        else:
            position = min(1.0, phase * 2) if phase <= 0.5 else (phase - 0.5) * 2
            weight = position * position * (3.0 - 2.0 * position)
            allowance = stars + (half - stars) * weight if phase <= 0.5 else half + (full - half) * weight
        visible = min(1.0, max(0.0, altitude + 1.0))
        visible *= visible * (3.0 - 2.0 * visible)
        return (stars + (allowance - stars) * visible) / 100.0
    except (AttributeError, TypeError, ValueError, OverflowError):
        return 0.0


def measure_ring(data, mask, bit_depth, config, binning=1, clipping_fraction=0.0, invalid=0.0):
    """Return any-channel headroom after allowing a bounded source-pixel area.

    Pixels and a supplied detection mask must already share the final rotated,
    flipped and cropped geometry, before display scaling or borders. Match the
    displayed image-circle centre, including the sign of the vertical offset.
    The brightest allowed fraction is omitted irrespective of shape, so compact
    Moon, stars, reflections and a small meteor need not trigger a correction.
    Narrow horizon clipping beyond that allowance remains protected.
    Invalid geometry or an empty usable ring disables this optional meter.
    """
    if (not isinstance(data, numpy.ndarray) or not data.size
            or data.dtype not in (numpy.uint8, numpy.uint16)
            or data.ndim not in (2, 3)
            or (data.ndim == 3 and data.shape[2] != 3)):
        return invalid
    if (mask is not None and (not isinstance(mask, numpy.ndarray)
                            or mask.shape != data.shape[:2]
                            or mask.dtype.kind not in 'buif'
                            or not numpy.isfinite(mask).all())):
        return invalid
    try:
        depth = int(bit_depth)
        if depth != bit_depth or not 1 <= depth <= numpy.iinfo(data.dtype).bits:
            return invalid
        scale = float(binning)
        allowed = float(clipping_fraction)
        circle = config.get('IMAGE_CIRCLE_MASK', {})
        diameter = float(circle.get('DIAMETER', 0)) if circle.get('ENABLE') else 0.0
        if diameter <= 0:
            diameter = float(config['LENS_IMAGE_CIRCLE'])
        offset_x = float(config.get('LENS_OFFSET_X', 0))
        offset_y = float(config.get('LENS_OFFSET_Y', 0))
        if (not all(math.isfinite(value) for value in (scale, diameter, offset_x, offset_y, allowed))
                or scale <= 0 or diameter <= 0 or not 0 <= allowed <= 0.5):
            return invalid
        height, width = data.shape[:2]
        center_x = int(width / 2) + int(offset_x / scale)
        center_y = int(height / 2) - int(offset_y / scale)
        radius = int(diameter / (2.0 * scale))
        if radius <= 0:
            return invalid
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return invalid

    x = numpy.arange(width, dtype=numpy.float32) - center_x
    y = numpy.arange(height, dtype=numpy.float32) - center_y
    distance = numpy.hypot(y[:, None], x)
    valid = (distance > 0.70 * radius) & (distance <= radius)
    if mask is not None:
        valid &= mask != 0
    if not numpy.any(valid):
        return invalid

    values = data[valid]
    if data.ndim == 3:
        values = values.max(axis=1)
    values = numpy.minimum(values.astype(numpy.float32) / ((1 << depth) - 1), 1.0)
    feather = numpy.clip((distance[valid] / radius - 0.70) / 0.05, 0.0, 1.0)
    values *= feather * feather * (3.0 - 2.0 * feather)
    index = values.size - 1 - int(values.size * allowed)
    return float(numpy.partition(values, index)[index])
