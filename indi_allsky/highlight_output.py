"""Small display-tail meter and bounded Moon exceptions for highlight control.

The sparse geometry follows processing/VirtualSky without importing the lens
solver's catalogue or eager Astropy initialization in the capture hot path.
"""
import math
from datetime import datetime, timezone

import cv2
import numpy as np


def _geometry(config, shape, binning):
    """Return sensor->delivered and sensor->pre-border transforms and sizes."""
    h, w = map(int, shape[:2])
    if min(h, w) <= 0 or max(h, w) > 30000 or binning < 1:
        raise ValueError('Invalid sensor size')
    transform = np.eye(3)
    rotation = config.get('IMAGE_ROTATE')
    if rotation == 'ROTATE_90_CLOCKWISE':
        transform = np.array([[0., -1., h-1], [1., 0., 0.], [0., 0., 1.]])
        w, h = h, w
    elif rotation == 'ROTATE_90_COUNTERCLOCKWISE':
        transform = np.array([[0., 1., 0.], [-1., 0., w-1], [0., 0., 1.]])
        w, h = h, w
    elif rotation == 'ROTATE_180':
        transform = np.array([[-1., 0., w-1], [0., -1., h-1], [0., 0., 1.]])
    elif rotation:
        raise ValueError('Unknown rotation')
    angle = float(config.get('IMAGE_ROTATE_ANGLE', 0))
    if not math.isfinite(angle):
        raise ValueError('Invalid angle')
    if angle:
        c, s = math.cos(math.radians(int(angle))), math.sin(math.radians(int(angle)))
        cx, cy = w//2, h//2
        nw, nh = ((w, h) if config.get('IMAGE_ROTATE_KEEP_SIZE', False)
                  else (int(h*abs(s)+w*abs(c)), int(h*abs(c)+w*abs(s))))
        transform = np.array([[c, s, (1-c)*cx-s*cy+nw/2-cx],
                              [-s, c, s*cx+(1-c)*cy+nh/2-cy], [0., 0., 1.]]) @ transform
        w, h = nw-nw % 2, nh-nh % 2
    for key, axis, size in (('IMAGE_FLIP_V', 1, h), ('IMAGE_FLIP_H', 0, w)):
        if config.get(key, False):
            flip = np.eye(3)
            flip[axis, axis], flip[axis, 2] = -1, size-1
            transform = flip @ transform
    if config.get('IMAGE_CROP_IMAGE_CIRCLE', False):
        radius = int(float(config.get('LENS_IMAGE_CIRCLE', 0))/2)
        ox, oy = (int(config.get(key, 0)) for key in ('LENS_OFFSET_X', 'LENS_OFFSET_Y'))
        if radius <= 0:
            raise ValueError('Invalid image circle')
        x1, x2 = max(0, w//2-radius+min(0, 2*ox)), min(w, w//2+radius+max(0, 2*ox))
        y1, y2 = max(0, h//2-radius-max(0, 2*oy)), min(h, h//2+radius-min(0, 2*oy))
    elif config.get('IMAGE_CROP_ROI'):
        x1, y1, x2, y2 = (int(float(value)/binning) for value in config['IMAGE_CROP_ROI'])
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    x1, x2, _ = slice(x1, x2).indices(w)
    y1, y2, _ = slice(y1, y2).indices(h)
    transform = np.array([[1., 0., -x1], [0., 1., -y1], [0., 0., 1.]]) @ transform
    w, h = x2-x1, y2-y1
    if min(w, h) <= 0:
        raise ValueError('Empty crop')
    preborder, pre_shape = transform.copy(), (h, w)
    scale = float(config.get('IMAGE_SCALE', 100))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid scale')
    if scale != 100:
        nw, nh = int(w*scale/100), int(h*scale/100)
        nw, nh = nw-nw % 2, nh-nh % 2
        if min(nw, nh) <= 0:
            raise ValueError('Empty scaled image')
        sx, sy = nw/w, nh/h
        transform = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]]) @ transform
        w, h = nw, nh
    border = config.get('IMAGE_BORDER', {})
    left, top, right, bottom = (float(border.get(key, 0)) for key in ('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    if any(not math.isfinite(value) or value < 0 or value != int(value)
           for value in (left, top, right, bottom)):
        raise ValueError('Invalid border')
    transform = np.array([[1., 0., left], [0., 1., top], [0., 0., 1.]]) @ transform
    return transform, (int(h+top+bottom), int(w+left+right)), preborder, pre_shape


def night_moon_weight(exp_date, exposure, position):
    """Fade the night Moon exception from Sun -16 to -18 degrees.

    Use the geometric solar centre at the capture midpoint, independently of
    rendering delay. Unusable capture context retains the existing safeguard.
    """
    import ephem

    try:
        latitude, longitude, elevation = map(float, position[:3])
        exposure = float(exposure)
        midpoint = float(exp_date.timestamp()) - exposure / 2
        if (not all(math.isfinite(value) for value in (latitude, longitude, elevation, exposure, midpoint))
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
                or exposure < 0 or not 0 <= midpoint <= 4102444800):
            return 0.0
        observer = ephem.Observer()
        observer.lat, observer.lon = math.radians(latitude), math.radians(longitude)
        observer.elevation, observer.pressure = elevation, 0
        observer.date = datetime.fromtimestamp(midpoint, timezone.utc)
        altitude = math.degrees(float(ephem.Sun(observer).alt))
        if not math.isfinite(altitude):
            return 0.0
        strength = min(1.0, max(0.0, (-altitude - 16.0) / 2.0))
        return strength * strength * (3.0 - 2.0 * strength)
    except (AttributeError, IndexError, OSError, TypeError, ValueError, OverflowError):
        return 0.0


def moon_mask_weight(exp_date, exposure, position):
    """Keep the physical Moon exception through night and twilight.

    This does not control the night exposure preference or twilight controller.
    Only the existing daylight-side fade, Sun -2 to 0 degrees, changes its area.
    """
    import ephem

    try:
        latitude, longitude, elevation = map(float, position[:3])
        exposure = float(exposure)
        midpoint = float(exp_date.timestamp()) - exposure / 2
        if (not all(math.isfinite(value) for value in (latitude, longitude, elevation, exposure, midpoint))
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
                or exposure < 0 or not 0 <= midpoint <= 4102444800):
            return 0.0
        observer = ephem.Observer()
        observer.lat, observer.lon = math.radians(latitude), math.radians(longitude)
        observer.elevation, observer.pressure = elevation, 0
        observer.date = datetime.fromtimestamp(midpoint, timezone.utc)
        altitude = math.degrees(float(ephem.Sun(observer).alt))
        if not math.isfinite(altitude):
            return 0.0
        strength = min(1.0, max(0.0, -altitude / 2.0))
        return strength * strength * (3.0 - 2.0 * strength)
    except (AttributeError, IndexError, OSError, TypeError, ValueError, OverflowError):
        return 0.0


def moon_region(config, native_shape, binning, exp_date, exposure, position,
                camera_uuid='', *, native=False):
    """Moon (x,y,radius) in sensor or rotated/flipped/cropped pre-border pixels.

    Use the capture midpoint, saved base VirtualSky alignment and its radial
    model. The bounded phase-dependent halo is at most 13% of the image-circle
    radius; it cannot exempt an unrelated horizon or an unlimited cloud bank.
    Missing/invalid alignment, below-horizon Moon and enabled residual models
    unsupported by this standalone feature return None, retaining raw safeguards.
    """
    import ephem

    try:
        latitude, longitude, elevation = map(float, position[:3])
        midpoint = float(exp_date.timestamp())-float(exposure)/2
        binning = float(binning)
        if (not all(math.isfinite(value) for value in (latitude, longitude, elevation, midpoint, binning))
                or not -90 <= latitude <= 90 or not -180 <= longitude <= 180
                or not 0 <= midpoint <= 4102444800 or binning < 1 or binning != int(binning)
                or not math.isfinite(float(exposure)) or float(exposure) < 0):
            return None
        vs = config['VIRTUALSKY']
        if any(type(vs.get(key, False)) is not bool for key in ('FLIP_H', 'FLIP_V', 'CALIBRATION_ENABLED')):
            return None
        if vs.get('CALIBRATION_ENABLED', False) and vs.get('CALIBRATION') is not None:
            return None
        geometry = [float(value) for value in (
            config.get('LENS_AZIMUTH', 0), vs.get('LATITUDE_OFFSET', 0), vs.get('LONGITUDE_OFFSET', 0),
            vs['IMAGE_CIRCLE_DIAMETER'], vs.get('OFFSET_X', 0), vs.get('OFFSET_Y', 0),
            config.get('LENS_ALTITUDE', 90), vs.get('POINTING_AZIMUTH', 0), vs.get('RADIAL_DISTORTION', 0))]
        if (not all(math.isfinite(value) for value in geometry) or not 0 < geometry[3] <= 30000
                or not 0 <= geometry[6] <= 90 or not -.5 <= geometry[8] <= 1):
            return None
        if config.get('PRIVACY_MODE', False):
            latitude, longitude = round(latitude), round(longitude)
        latitude, longitude = latitude+geometry[1], longitude+geometry[2]
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            return None
        delivered, output_shape, preborder, pre_shape = _geometry(config, native_shape, binning)
        observer = ephem.Observer()
        observer.lat, observer.lon = math.radians(latitude), math.radians(longitude)
        observer.elevation, observer.pressure = elevation, 0
        observer.date = datetime.fromtimestamp(midpoint, timezone.utc)
        moon = ephem.Moon(observer)
        alt, az = float(moon.alt), float(moon.az)
        if alt+float(moon.radius) <= 0:
            return None
        roll, _, _, diameter, ox, oy, altitude, heading, radial = geometry
        tilt, heading = math.radians(90-altitude), math.radians(heading)
        across, along, up = math.cos(alt)*math.sin(az-heading), math.cos(alt)*math.cos(az-heading), math.sin(alt)
        forward, axis = math.cos(tilt)*along-math.sin(tilt)*up, math.sin(tilt)*along+math.cos(tilt)*up
        alt, az = math.atan2(axis, math.hypot(across, forward)), heading+math.atan2(across, forward)
        if alt <= 0:
            return None
        radius = diameter/2
        r = radius*math.sin((math.pi/2-alt)/2)/.70710678
        r *= max(2-(r/radius)**2, 1e-12)**(-radial)
        psi = az-math.radians(roll)
        point = np.array([output_shape[1]/2+ox+(1 if vs.get('FLIP_H', False) else -1)*r*math.sin(psi),
                          output_shape[0]/2-oy+(1 if vs.get('FLIP_V', False) else -1)*r*math.cos(psi), 1.])
        destination = np.eye(3) if native else preborder
        mapping = destination @ np.linalg.inv(delivered)
        point = mapping @ point
        bounds = native_shape[:2] if native else pre_shape
        if not (0 <= point[0] < bounds[1] and 0 <= point[1] < bounds[0]):
            return None
        circle = config.get('IMAGE_CIRCLE_MASK', {})
        circle_diameter = float(circle.get('DIAMETER', 0) if circle.get('ENABLE') else config.get('LENS_IMAGE_CIRCLE', 0))
        if not math.isfinite(circle_diameter) or circle_diameter <= 0:
            return None
        # Full Moon bloom can be wider than the lunar disk under clouds. The
        # exception stays local and shrinks toward new Moon; stars use a separate
        # tiny tail allowance. Binning changes geometry, not its sky footprint.
        halo = circle_diameter/(2*binning)*(.025+.105*math.sqrt(max(0., min(1., float(moon.moon_phase)))))
        return float(point[0]), float(point[1]), float(halo)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return None


def output_tail(data, mask=None, star_allowance=.001, target=252):
    """Return normalized maxRGB tail and fraction above target after tiny sources.

    One 256-bin histogram reuses the existing rendered pixels. Equal areas in
    adjacent different colours remain protected because maxRGB is per pixel.
    An invalid/empty view returns (0,0), retaining the caller's raw fallback.
    """
    if (not isinstance(data, np.ndarray) or data.dtype != np.uint8 or not data.size
            or data.ndim not in (2, 3) or (data.ndim == 3 and data.shape[2] != 3)):
        return 0., 0.
    if mask is not None and (not isinstance(mask, np.ndarray) or mask.shape != data.shape[:2]
                             or mask.dtype.kind not in 'buif' or not np.isfinite(mask).all()):
        return 0., 0.
    try:
        allowance = float(star_allowance)
        ceiling = int(target)
        if (not math.isfinite(allowance) or not 0 <= allowance <= .01
                or ceiling != target or not 0 <= ceiling < 255):
            return 0., 0.
    except (TypeError, ValueError, OverflowError):
        return 0., 0.
    valid = None if mask is None else np.asarray(mask != 0, np.uint8)
    count = data.shape[0]*data.shape[1] if valid is None else cv2.countNonZero(valid)
    if not count:
        return 0., 0.
    values = data if data.ndim == 2 else cv2.max(cv2.max(data[:, :, 0], data[:, :, 1]), data[:, :, 2])
    histogram = cv2.calcHist([values], [0], valid, [256], [0, 256]).ravel()
    omitted = int(count*allowance)
    rank = count-1-omitted
    peak = float(np.searchsorted(np.cumsum(histogram), rank+1))/255
    over = max(0., float(histogram[ceiling+1:].sum())-omitted)/count
    return peak, over
