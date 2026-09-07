import math


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
    """Validate and coerce the six solver form values from request JSON.
    Returns (values, None) or (None, error); only the six known keys are
    ever passed through, plus optional camera pointing angles.
    """
    values = {}
    for key, cast, vmin, vmax in SOLVER_REQUEST_FIELDS:
        if key not in data:
            return None, 'Missing field: {0:s}'.format(key)
        try:
            # json accepts literal Infinity/NaN; int(inf) raises OverflowError
            v = cast(float(data[key]))
        except (TypeError, ValueError, OverflowError):
            return None, 'Invalid value for {0:s}'.format(key)
        # Config accepts arbitrary finite latitude/longitude offsets.
        manual_offset = for_save and key in ('LATITUDE_OFFSET', 'LONGITUDE_OFFSET')
        if not math.isfinite(v) or (not manual_offset and not vmin <= v <= vmax):
            return None, '{0:s} out of range'.format(key)
        values[key] = v

    for key, vmin, vmax in (('POINTING_AZIMUTH', 0.0, 360.0), ('LENS_ALTITUDE', 0.0, 90.0)):
        if key not in data:
            continue
        try:
            angle = float(data[key])
        except (TypeError, ValueError, OverflowError):
            return None, 'Invalid value for {0:s}'.format(key)
        if not vmin <= angle <= vmax:
            return None, '{0:s} out of range'.format(key)
        values[key] = angle

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

    return config
