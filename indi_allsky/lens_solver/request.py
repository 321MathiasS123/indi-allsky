import math
from .projection import RADIAL_MIN, RADIAL_MAX

from .calibration import validateCalibration


# (key, cast, min, max) -- solver input limits, not manual save limits.
SOLVER_REQUEST_FIELDS = (
    ('AZIMUTH_ANGLE', float, 0.0, 360.0),
    ('LATITUDE_OFFSET', float, -30.0, 30.0),
    ('LONGITUDE_OFFSET', float, -30.0, 30.0),
    ('IMAGE_CIRCLE_DIAMETER', int, 100, 20000),
    ('OFFSET_X', int, -10000, 10000),
    ('OFFSET_Y', int, -10000, 10000),
)


def parseSolverRequestValues(data, for_save=False):
    """Validate geometry and optional lens calibration from request JSON.
    Returns (values, None) or (None, error); unknown keys are discarded.
    """
    values = {}
    for key, cast, vmin, vmax in SOLVER_REQUEST_FIELDS + (
            ('POINTING_AZIMUTH', float, 0.0, 360.0), ('LENS_ALTITUDE', float, 0.0, 90.0),
            ('RADIAL_DISTORTION', float, RADIAL_MIN, RADIAL_MAX)):
        if key not in data:
            if key in ('POINTING_AZIMUTH', 'LENS_ALTITUDE', 'RADIAL_DISTORTION'):
                continue  # optional for clients that predate camera pointing
            return None, 'Missing field: {0:s}'.format(key)
        try:
            if isinstance(data[key], bool):
                raise ValueError  # JSON booleans are not calibration numbers
            # json accepts literal Infinity/NaN; int(inf) raises OverflowError
            v = cast(float(data[key]))
        except (TypeError, ValueError, OverflowError):
            return None, 'Invalid value for {0:s}'.format(key)
        # Config accepts arbitrary finite latitude/longitude offsets.
        manual_offset = for_save and key in ('LATITUDE_OFFSET', 'LONGITUDE_OFFSET')
        if not math.isfinite(v) or (not manual_offset and not vmin <= v <= vmax):
            return None, '{0:s} out of range'.format(key)
        values[key] = v

    if 'PRECESSION' in data:
        if not isinstance(data['PRECESSION'], bool):
            return None, 'PRECESSION must be a boolean'
        values['PRECESSION'] = data['PRECESSION']
    if 'CALIBRATION_ENABLED' in data:
        if not isinstance(data['CALIBRATION_ENABLED'], bool):
            return None, 'CALIBRATION_ENABLED must be a boolean'
        values['CALIBRATION_ENABLED'] = data['CALIBRATION_ENABLED']
        if for_save:
            model = data.get('CALIBRATION')
            if model is not None:
                if not validateCalibration(model):
                    return None, 'Invalid lens calibration; solve again'
                geometry = [values.get(k) for k in ('AZIMUTH_ANGLE', 'LATITUDE_OFFSET',
                    'LONGITUDE_OFFSET', 'IMAGE_CIRCLE_DIAMETER', 'OFFSET_X', 'OFFSET_Y',
                    'LENS_ALTITUDE', 'POINTING_AZIMUTH')]
                geometry += [values.get('RADIAL_DISTORTION', 0), int(values.get('PRECESSION', False))]
                # Older corrections belong to the original lens and catalogue convention.
                saved_geometry = model['geometry'] + ([0, 0] if model['version'] == 1 else [])
                if saved_geometry != geometry:
                    return None, 'Alignment changed since calibration; solve again'
            # A successful solve may need no extra correction. Keep the opt-in
            # preference without blocking Save or applying an absent model.
            values['CALIBRATION'] = model
    return values, None


def applySolvedValuesToConfig(config, values):
    """Write overlay calibration and optional camera pointing, in place.
    The LENS_IMAGE_CIRCLE family drives unrelated behavior and stays unchanged.
    """
    config['LENS_AZIMUTH'] = values['AZIMUTH_ANGLE']
    if 'LENS_ALTITUDE' in values:
        config['LENS_ALTITUDE'] = values['LENS_ALTITUDE']

    if 'VIRTUALSKY' not in config:
        config['VIRTUALSKY'] = {}

    virtualsky = config['VIRTUALSKY']
    virtualsky['LATITUDE_OFFSET'] = values['LATITUDE_OFFSET']
    virtualsky['LONGITUDE_OFFSET'] = values['LONGITUDE_OFFSET']
    virtualsky['IMAGE_CIRCLE_DIAMETER'] = values['IMAGE_CIRCLE_DIAMETER']
    virtualsky['OFFSET_X'] = values['OFFSET_X']
    virtualsky['OFFSET_Y'] = values['OFFSET_Y']
    if 'POINTING_AZIMUTH' in values:
        virtualsky['POINTING_AZIMUTH'] = values['POINTING_AZIMUTH']
    if 'CALIBRATION_ENABLED' in values:
        virtualsky['CALIBRATION_ENABLED'] = values['CALIBRATION_ENABLED']
        virtualsky['CALIBRATION'] = values['CALIBRATION']
    if 'PRECESSION' in values:
        virtualsky['PRECESSION'] = values['PRECESSION']
    if 'RADIAL_DISTORTION' in values:
        virtualsky['RADIAL_DISTORTION'] = values['RADIAL_DISTORTION']

    return config
