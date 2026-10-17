"""Optional catalogue positions in the denoiser's unrotated sensor coordinates.

This is a confidence aid for observed sources, not a plate solver or a star mask.
Keep the small projection/calibration formulas compatible with lens_solver and
VirtualSky, without importing the solver's eager Astropy initialization on a Pi.
"""
import functools
import hashlib
import json
import math
from pathlib import Path

import numpy as np


PIPELINE_KEYS = (
    'IMAGE_ROTATE', 'IMAGE_ROTATE_ANGLE', 'IMAGE_ROTATE_KEEP_SIZE',
    'IMAGE_FLIP_H', 'IMAGE_FLIP_V', 'IMAGE_CROP_ROI',
    'IMAGE_CROP_IMAGE_CIRCLE', 'IMAGE_SCALE', 'IMAGE_BORDER',
    'LENS_IMAGE_CIRCLE', 'LENS_OFFSET_X', 'LENS_OFFSET_Y',
)
_EMPTY = np.empty((0, 3), dtype=float)
_EMPTY.setflags(write=False)


def _number(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Boolean coordinate')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Non-finite coordinate')
    return value


def _boolean(config, key, default=False):
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError('Invalid boolean')
    return value


def _pipeline_signature(config):
    return hashlib.sha256(json.dumps([config.get(k) for k in PIPELINE_KEYS],
                                    sort_keys=True).encode()).hexdigest()


def _sensor_to_output(config, shape, binning):
    """Sparse equivalent of rotate, flip, crop, resize and add_border."""
    h, w = shape[:2]
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

    angle = _number(config.get('IMAGE_ROTATE_ANGLE', 0))
    if angle:
        angle = math.radians(int(angle))
        c, s = math.cos(angle), math.sin(angle)
        cx, cy = w//2, h//2
        keep = _boolean(config, 'IMAGE_ROTATE_KEEP_SIZE')
        nw, nh = (w, h) if keep else (int(h*abs(s)+w*abs(c)), int(h*abs(c)+w*abs(s)))
        rotation = np.array([[c, s, (1-c)*cx-s*cy+nw/2-cx],
                             [-s, c, s*cx+(1-c)*cy+nh/2-cy], [0., 0., 1.]])
        transform = rotation @ transform
        w, h = nw-nw % 2, nh-nh % 2

    for key, axis, size in (('IMAGE_FLIP_V', 1, h), ('IMAGE_FLIP_H', 0, w)):
        if _boolean(config, key):
            flip = np.eye(3)
            flip[axis, axis], flip[axis, 2] = -1, size-1
            transform = flip @ transform

    if _boolean(config, 'IMAGE_CROP_IMAGE_CIRCLE'):
        # Match processing.py: circle crop is not divided by binning, unlike ROI.
        radius = int(_number(config.get('LENS_IMAGE_CIRCLE', 3000))/2)
        ox = _number(config.get('LENS_OFFSET_X', 0))
        oy = _number(config.get('LENS_OFFSET_Y', 0))
        if ox != int(ox) or oy != int(oy) or radius <= 0:
            raise ValueError('Invalid circle crop')
        x1, x2 = max(0, w//2-radius+min(0, 2*int(ox))), min(w, w//2+radius+max(0, 2*int(ox)))
        y1, y2 = max(0, h//2-radius-max(0, 2*int(oy))), min(h, h//2+radius-min(0, 2*int(oy)))
    elif config.get('IMAGE_CROP_ROI'):
        roi = config['IMAGE_CROP_ROI']
        if len(roi) != 4:
            raise ValueError('Invalid crop')
        x1, y1, x2, y2 = (int(_number(v)/binning) for v in roi)
    else:
        x1, y1, x2, y2 = 0, 0, w, h
    x1, x2, _ = slice(x1, x2).indices(w)
    y1, y2, _ = slice(y1, y2).indices(h)
    transform = np.array([[1., 0., -x1], [0., 1., -y1], [0., 0., 1.]]) @ transform
    w, h = x2-x1, y2-y1
    if min(w, h) <= 0:
        raise ValueError('Empty crop')

    scale = _number(config.get('IMAGE_SCALE', 100))
    if scale != 100:
        nw, nh = int(w*scale/100), int(h*scale/100)
        nw, nh = nw-nw % 2, nh-nh % 2
        if min(nw, nh) <= 0:
            raise ValueError('Empty scaled image')
        sx, sy = nw/w, nh/h
        transform = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]]) @ transform
        w, h = nw, nh

    border = config.get('IMAGE_BORDER', {})
    left, top, right, bottom = (_number(border.get(k, 0)) for k in ('LEFT', 'TOP', 'RIGHT', 'BOTTOM'))
    if any(v < 0 or v != int(v) for v in (left, top, right, bottom)):
        raise ValueError('Invalid border')
    transform = np.array([[1., 0., left], [0., 1., top], [0., 0., 1.]]) @ transform
    return transform, (int(h+top+bottom), int(w+left+right)), (left, top, left+w, top+h)


def _displacement(xy, model):
    bounds = np.asarray(model['bounds'])
    outside = np.maximum(bounds[:2]-xy, xy-bounds[2:])
    t = np.clip(outside/.15, 0, 1)
    weight = np.prod(1-t*t*(3-2*t), axis=1)
    t = np.clip((np.linalg.norm(xy, axis=1)-1)/.15, 0, 1)
    weight *= 1-t*t*(3-2*t)
    u, v = xy.T
    basis = np.column_stack([np.ones(len(u)), u, v, u*u, u*v, v*v,
                             u*u*u, u*u*v, u*v*v, v*v*v])
    return (basis @ np.asarray(model['coefficients']))*weight[:, None]


def _valid_model(model):
    if not isinstance(model, dict) or type(model.get('version')) is not int or model['version'] not in (1, 2):
        return False
    keys = {'version', 'coefficients', 'bounds', 'geometry', 'image_size',
            'context', 'pipeline', 'camera_uuid', 'summary'}
    if set(model)-{'orientation'} != keys:
        return False
    for key, shape in (('coefficients', (10, 2)), ('bounds', (4,)),
                       ('geometry', (10 if model['version'] == 2 else 8,)),
                       ('image_size', (2,)), ('context', (3,))):
        a = np.asarray(model[key])
        if (a.shape != shape or a.dtype.kind not in 'ifu' or not np.isfinite(a).all()
                or any(isinstance(v, bool) for v in np.asarray(model[key], dtype=object).flat)):
            return False
    if model['version'] == 2 and (not -.5 <= model['geometry'][8] <= 1 or model['geometry'][9] not in (0, 1)):
        return False
    orientation = model.get('orientation', [False, False])
    if not isinstance(orientation, list) or len(orientation) != 2 or any(type(v) is not bool for v in orientation):
        return False
    bounds = np.asarray(model['bounds'])
    if (np.max(np.abs(model['coefficients'])) > 1 or np.max(np.abs(bounds)) > 3
            or np.any(bounds[2:]-bounds[:2] < .15)
            or not all(1 <= v <= 30000 for v in model['image_size'])):
        return False
    if any(not isinstance(model[k], str) or len(model[k]) > limit
           for k, limit in (('pipeline', 64), ('camera_uuid', 64), ('summary', 500))):
        return False
    if len(model['pipeline']) != 64:
        return False
    # Same bounded displacement/Jacobian checks as saved solver calibrations.
    xy = np.array(np.meshgrid(np.linspace(bounds[0]-.15, bounds[2]+.15, 49),
                             np.linspace(bounds[1]-.15, bounds[3]+.15, 49))).reshape(2, -1).T
    delta = _displacement(xy, model)
    jac = np.stack([(_displacement(xy+step, model)-_displacement(xy-step, model))/2e-5
                    for step in ([1e-5, 0], [0, 1e-5])], axis=-1)
    return np.max(np.linalg.norm(delta, axis=1)) <= .06 and np.max(np.linalg.norm(jac, axis=(1, 2))) < .45


@functools.lru_cache(maxsize=1)
def _catalogue():
    with Path(__file__).with_name('data').joinpath('lens_solver_stars.json').open() as f:
        rows = json.load(f)['stars']
    # This is a positional reference, not a fit; the bundled catalogue is
    # reasonably complete only to 5.5. Observed matches establish confidence.
    result = np.array([[ra, dec, mag] for _, mag, ra, dec in rows if mag <= 5.5], dtype=float)
    result = result[np.argsort(result[:, 2])]
    result.setflags(write=False)
    return result


@functools.lru_cache(maxsize=8)
def _equatorial(epoch_day):
    catalog = _catalogue()
    ra, dec = np.radians(catalog[:, :2].T)
    if epoch_day is not None:
        # Day midpoint bounds the precession approximation to half a day.
        t = (epoch_day+.5+2440587.5-2451545.)/36525
        zeta, z, theta = np.radians(np.array([
            2306.2181*t+.30188*t*t+.017998*t*t*t,
            2306.2181*t+1.09468*t*t+.018203*t*t*t,
            2004.3109*t-.42665*t*t-.041833*t*t*t])/3600)
        a = np.cos(dec)*np.sin(ra+zeta)
        b = np.cos(theta)*np.cos(dec)*np.cos(ra+zeta)-np.sin(theta)*np.sin(dec)
        c = np.sin(theta)*np.cos(dec)*np.cos(ra+zeta)+np.cos(theta)*np.sin(dec)
        ra, dec = np.arctan2(a, b)+z, np.arctan2(c, np.hypot(a, b))
    return ra, np.sin(dec), np.cos(dec)


def _gmst(timestamp):
    # Mean sidereal time using UTC rather than UT1. The sub-second UT1 offset
    # is immaterial at all-sky pixel scales; tested against the Astropy solver.
    days = timestamp/86400+2440587.5-2451545.
    centuries = days/36525
    return math.radians((280.46061837+360.98564736629*days
                         +.000387933*centuries**2-centuries**3/38710000) % 360)


class VirtualSkyReference:
    """Cached geometry; predictions are optional Nx3 sensor x/y/magnitude rows.

    Recreate when camera, shape, binning, location, privacy, VirtualSky or image
    geometry settings change. Disabled/invalid residual correction does not
    enable unsafe coordinates: an enabled stale model rejects the reference.
    With correction disabled or absent, base alignment remains usable and must
    still pass the caller's observed-star matching checks.
    """

    def __init__(self, config, shape, binning, camera_uuid, latitude, longitude):
        self.valid = False
        self.binning = 1
        self._model = None
        self._validated_time = None
        try:
            self.binning = _number(binning)
            if self.binning < 1 or self.binning != int(self.binning):
                return
            self._shape = tuple(shape[:2])
            if len(self._shape) != 2 or any(type(v) not in (int, np.int64, np.int32) or not 1 <= v <= 30000 for v in self._shape):
                return
            vs = config['VIRTUALSKY']
            self._geometry = np.array([_number(v) for v in (
                config.get('LENS_AZIMUTH', 0), vs.get('LATITUDE_OFFSET', 0),
                vs.get('LONGITUDE_OFFSET', 0), vs['IMAGE_CIRCLE_DIAMETER'],
                vs.get('OFFSET_X', 0), vs.get('OFFSET_Y', 0),
                config.get('LENS_ALTITUDE', 90), vs.get('POINTING_AZIMUTH', 0),
                vs.get('RADIAL_DISTORTION', 0))])
            if not (0 < self._geometry[3] <= 30000 and 0 <= self._geometry[6] <= 90
                    and -.5 <= self._geometry[8] <= 1):
                return
            self._precession = _boolean(vs, 'PRECESSION')
            self._orientation = [_boolean(vs, 'FLIP_H'), _boolean(vs, 'FLIP_V')]
            latitude, longitude = _number(latitude), _number(longitude)
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                return
            if _boolean(config, 'PRIVACY_MODE'):
                latitude, longitude = round(latitude), round(longitude)
            self._latitude = math.radians(latitude+self._geometry[1])
            self._longitude = math.radians(longitude+self._geometry[2])
            transform, self._output_shape, self._bounds = _sensor_to_output(config, shape, self.binning)
            if max(self._output_shape) > 30000:
                return
            self._inverse = np.linalg.inv(transform)
            self._transform = transform
            model = vs.get('CALIBRATION')
            if _boolean(vs, 'CALIBRATION_ENABLED') and model is not None:
                if not _valid_model(model):
                    return
                geometry = list(self._geometry)+[int(self._precession)]
                saved = model['geometry']+([0, 0] if model['version'] == 1 else [])
                if saved != geometry or model.get('orientation', [False, False]) != self._orientation:
                    return
                if (model['pipeline'] != _pipeline_signature(config) or model['camera_uuid'] != camera_uuid
                        or model['image_size'] != list(self._output_shape[::-1])
                        or model['context'][:2] != [latitude, longitude]):
                    return
                # context[2] bridges UI/server time zones, not UTC capture time.
                self._model = {k: np.array(model[k], copy=True) for k in ('bounds', 'coefficients')}
            self.valid = True
        except (KeyError, TypeError, ValueError, OverflowError, AttributeError, np.linalg.LinAlgError):
            pass

    def mark_validated(self, capture_time):
        """Remember an alignment verified against distributed measured stars."""
        try:
            timestamp = _number(capture_time)
            if not self.valid or not 0 <= timestamp <= 4102444800:
                return
            self._validated_time = timestamp
        except (TypeError, ValueError, OverflowError):
            pass

    def motion_available(self, capture_time):
        """Clouds may hide anchors, but a recent observed alignment stays useful."""
        try:
            timestamp = _number(capture_time)
            return (self.valid and self._validated_time is not None
                    and 0 <= timestamp <= 4102444800
                    and 0 <= timestamp-self._validated_time <= 6*3600)
        except (TypeError, ValueError, OverflowError):
            return False

    def displacements(self, points, elapsed_seconds, *, reference_time):
        """Sparse hypothetical sky motion at measured sensor positions.

        Only an observed, recently validated alignment authorizes this fallback.
        Unsupported positions (near/below the horizon or outside the image)
        return zero motion, so they cannot acquire new sensor-defect proof.
        Elapsed time is the actual midpoint interval; the separate reference
        timestamp is only the clock used to validate the saved overlay.
        """
        if not self.motion_available(reference_time):
            return None
        try:
            points = np.asarray(points, dtype=float)
            elapsed = np.broadcast_to(np.asarray(elapsed_seconds, dtype=float), (len(points),))
            if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
                    or not np.isfinite(elapsed).all() or np.any(elapsed < 0)
                    or np.any(elapsed > 6*3600)):
                return None
            if not len(points):
                return np.empty((0, 2))
            roll, _, _, diameter, ox, oy, altitude, heading, radial = self._geometry
            radius = diameter/2
            center = np.array([self._output_shape[1]/2+ox, self._output_shape[0]/2-oy])
            output = points @ self._transform[:2, :2].T+self._transform[:2, 2]
            x1, y1, x2, y2 = self._bounds
            supported = ((points[:, 0] >= 0) & (points[:, 0] < self._shape[1])
                         & (points[:, 1] >= 0) & (points[:, 1] < self._shape[0])
                         & (output[:, 0] >= x1) & (output[:, 0] < x2)
                         & (output[:, 1] >= y1) & (output[:, 1] < y2))
            xy = (output-center)/radius
            if self._model is not None:
                target = xy.copy()
                # Saved corrections have Jacobian norm <.45. Bounded fixed-point
                # inversion avoids a solver/dependency and verifies convergence.
                for _ in range(16):
                    xy = target-_displacement(xy, self._model)
                supported &= np.linalg.norm(xy+_displacement(xy, self._model)-target, axis=1) < 1e-6
            signs = np.array([1 if flag else -1 for flag in self._orientation])
            xy *= signs
            observed_r = np.linalg.norm(xy, axis=1)
            r = np.clip(observed_r, 0, 1)
            if radial:
                # Invert the monotone radial mapping on the visible hemisphere.
                for _ in range(7):
                    base = 2-r*r
                    value = r*base**(-radial)
                    derivative = base**(-radial)*(1+2*radial*r*r/base)
                    r = np.clip(r-(value-observed_r)/np.maximum(derivative, 1e-8), 0, 1)
            supported &= observed_r <= 1
            alt = np.pi/2-2*np.arcsin(np.clip(r*.70710678, 0, 1))
            az = np.arctan2(xy[:, 0], xy[:, 1])+math.radians(roll)
            tilt, heading = math.radians(90-altitude), math.radians(heading)
            across, forward, axis = np.cos(alt)*np.sin(az-heading), np.cos(alt)*np.cos(az-heading), np.sin(alt)
            along = math.cos(tilt)*forward+math.sin(tilt)*axis
            up = -math.sin(tilt)*forward+math.cos(tilt)*axis
            vector = np.column_stack((across*math.cos(heading)+along*math.sin(heading),
                                      along*math.cos(heading)-across*math.sin(heading), up))
            supported &= up > math.sin(math.radians(10))
            pole = np.array([0., math.cos(self._latitude), math.sin(self._latitude)])
            angle = -elapsed*(2*np.pi/86164.0905)
            sine, cosine = np.sin(angle)[:, None], np.cos(angle)[:, None]
            vector = (vector*cosine+np.cross(pole, vector)*sine
                      +np.outer(vector @ pole, pole)*(1-cosine))
            supported &= vector[:, 2] > math.sin(math.radians(10))
            across = vector[:, 0]*math.cos(heading)-vector[:, 1]*math.sin(heading)
            along = vector[:, 1]*math.cos(heading)+vector[:, 0]*math.sin(heading)
            forward = math.cos(tilt)*along-math.sin(tilt)*vector[:, 2]
            axis = math.sin(tilt)*along+math.cos(tilt)*vector[:, 2]
            alt, az = np.arctan2(axis, np.hypot(across, forward)), heading+np.arctan2(across, forward)
            supported &= alt > 0
            r = np.sin((np.pi/2-alt)/2)/.70710678
            if radial:
                r *= np.maximum(2-r*r, 1e-12)**(-radial)
            psi = az-math.radians(roll)
            xy = np.column_stack((r*np.sin(psi), r*np.cos(psi)))*signs
            if self._model is not None:
                xy += _displacement(xy, self._model)
            xy = xy*radius+center
            xy = xy @ self._inverse[:2, :2].T+self._inverse[:2, 2]
            delta = xy-points
            if not np.isfinite(delta).all():
                return None
            delta[~supported] = 0
            return delta
        except (TypeError, ValueError, OverflowError, AttributeError):
            return None

    def predictions(self, capture_time):
        """Project at the UTC image timestamp used by the saved alignment.

        This can differ from the exposure midpoint used for temporal tracking.
        """
        if not self.valid:
            return _EMPTY
        try:
            timestamp = _number(capture_time)
            if not 0 <= timestamp <= 4102444800:  # bounded supported UTC epochs, 1970-2100
                return _EMPTY
            ra, sin_dec, cos_dec = _equatorial(math.floor(timestamp/86400) if self._precession else None)
            ha = _gmst(timestamp)+self._longitude-ra
            sl, cl = math.sin(self._latitude), math.cos(self._latitude)
            alt = np.arcsin(np.clip(sin_dec*sl+cos_dec*cl*np.cos(ha), -1, 1))
            az = np.arctan2(-cos_dec*np.sin(ha), sin_dec*cl-cos_dec*sl*np.cos(ha))
            visible = alt > math.radians(10)  # avoid large unmodelled horizon refraction
            roll, _, _, diameter, ox, oy, altitude, heading, radial = self._geometry
            if altitude != 90:
                tilt, heading = math.radians(90-altitude), math.radians(heading)
                across, along, up = np.cos(alt)*np.sin(az-heading), np.cos(alt)*np.cos(az-heading), np.sin(alt)
                forward = math.cos(tilt)*along-math.sin(tilt)*up
                axis = math.sin(tilt)*along+math.cos(tilt)*up
                alt, az = np.arctan2(axis, np.hypot(across, forward)), heading+np.arctan2(across, forward)
            visible &= alt > 0
            radius = diameter/2
            r = radius*np.sin((np.pi/2-alt)/2)/.70710678
            if radial:
                r *= np.maximum(2-(r/radius)**2, 1e-12)**(-radial)
            psi = az-math.radians(roll)
            center = np.array([self._output_shape[1]/2+ox, self._output_shape[0]/2-oy])
            xy = np.column_stack([(1 if self._orientation[0] else -1)*r*np.sin(psi),
                                   (1 if self._orientation[1] else -1)*r*np.cos(psi)])+center
            if self._model is not None:
                xy += radius*_displacement((xy-center)/radius, self._model)
            x1, y1, x2, y2 = self._bounds
            visible &= (xy[:, 0] >= x1) & (xy[:, 0] < x2) & (xy[:, 1] >= y1) & (xy[:, 1] < y2)
            xy = xy @ self._inverse[:2, :2].T+self._inverse[:2, 2]
            visible &= (xy[:, 0] >= 0) & (xy[:, 0] < self._shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < self._shape[0])
            return np.column_stack([xy[visible], _catalogue()[visible, 2]])
        except (OSError, KeyError, TypeError, ValueError, OverflowError):
            return _EMPTY
