"""Optional solar-elevation interpolation of image settings.

The runtime settings are a private copy.  Saved configuration and the operational
day/night flag (capture policy, files, cooling, etc.) are never changed here.
"""
import math
from datetime import timezone

import ephem


COLOR_DEFAULTS = {
    'WBR_FACTOR': 1.0, 'WBG_FACTOR': 1.0, 'WBB_FACTOR': 1.0,
    'WBR_MTF_MIDTONES': 0.5, 'WBG_MTF_MIDTONES': 0.5, 'WBB_MTF_MIDTONES': 0.5,
    'SATURATION_FACTOR': 1.0, 'GAMMA_CORRECTION': 1.0, 'SHARPEN_AMOUNT': 0.0,
    'SCNR_MTF_MIDTONES': 0.55,
}


def night_weight(altitude, start=-6.0, end=-12.0):
    if not all(math.isfinite(v) for v in (altitude, start, end)) or end >= start:
        raise ValueError('Full night elevation must be below the day/night elevation')
    u = max(0.0, min(1.0, (start - altitude) / (start - end)))
    return u * u * (3.0 - 2.0 * u)


def interpolate(day, night, weight, logarithmic=False):
    # Preserve exact endpoints, including camera minimum values.
    if weight <= 0 or day == night:
        return day
    if weight >= 1:
        return night
    if logarithmic and day > 0 and night > 0:
        return math.exp((1.0 - weight) * math.log(day) + weight * math.log(night))
    return day + weight * (night - day)


def runtime_weight(config):
    if config.get('TWILIGHT_TRANSITION', {}).get('ENABLE', False):
        return config.get('_TWILIGHT_WEIGHT')
    return None


def exposure_minimum(config, exposure_utils, night, weight=None):
    if weight is None:
        weight = runtime_weight(config)
    if weight is None:
        return exposure_utils.EXPOSURE_MIN_NIGHT if night else exposure_utils.EXPOSURE_MIN_DAY
    minimum = interpolate(exposure_utils.EXPOSURE_MIN_DAY, exposure_utils.EXPOSURE_MIN_NIGHT,
                          weight, logarithmic=True)
    # Shared exposure values have microsecond precision. Round the lower limit
    # up, without turning floating-point noise into an extra microsecond.
    minimum = math.ceil(round(minimum * 1000000, 6)) / 1000000
    return min(minimum, exposure_utils.EXPOSURE_MAX)


def capture_period(config, altitude):
    weight = night_weight(altitude, config.get('NIGHT_SUN_ALT_DEG', -6.0),
                          config.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
    return interpolate(config['EXPOSURE_PERIOD_DAY'], config['EXPOSURE_PERIOD'], weight)


def observer_at(when, latitude, longitude, elevation):
    obs = ephem.Observer()
    obs.lat, obs.lon = math.radians(latitude), math.radians(longitude)
    obs.elevation, obs.pressure = elevation, 0
    # A naive frame datetime uses the capture host's local timezone.
    obs.date = when.astimezone(timezone.utc)
    return obs


class TwilightTransition:
    def __init__(self, config):
        self.source = config
        self.enabled = bool(config.get('TWILIGHT_TRANSITION', {}).get('ENABLE', False))
        self.config = dict(config) if self.enabled else config
        self.weight = None
        self.altitude = None

    def update(self, when, latitude, longitude, elevation):
        if not self.enabled:
            return
        obs = observer_at(when, latitude, longitude, elevation)
        sun = ephem.Sun(obs)
        self.apply(math.degrees(sun.alt))

    def apply(self, altitude):
        if not self.enabled:
            return
        self.altitude = altitude
        self.weight = night_weight(altitude, self.source.get('NIGHT_SUN_ALT_DEG', -6.0),
                                   self.source.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
        self.config['_TWILIGHT_WEIGHT'] = self.weight
        target = self.value('TARGET_ADU', 'TARGET_ADU_DAY', 75)
        self.config['TARGET_ADU'] = self.config['TARGET_ADU_DAY'] = target

        if not self.source.get('USE_NIGHT_COLOR', True):
            for key, default in COLOR_DEFAULTS.items():
                self.config[key] = self.config[key + '_DAY'] = self.value(key, key + '_DAY', default)

        # Optional compatibility with highlight protection.  No highlight module
        # or settings are required by the independent transition feature.
        highlight = self.source.get('HIGHLIGHT_PROTECTION', {})
        if highlight.get('ENABLE') and not self.source.get('USE_NIGHT_COLOR', True):
            gamma_day = highlight.get('GAMMA_DAY', 0.0) or self.source.get('GAMMA_CORRECTION_DAY', 1.0)
            gamma_night = highlight.get('GAMMA', 0.0) or self.source.get('GAMMA_CORRECTION', 1.0)
            gamma = interpolate(gamma_day, gamma_night, self.weight)
            self.config['HIGHLIGHT_PROTECTION'] = dict(highlight, GAMMA=gamma, GAMMA_DAY=gamma)

    def value(self, night_key, day_key, default=False, color=False):
        night = self.source.get(night_key, default)
        day = night if color and self.source.get('USE_NIGHT_COLOR', True) else self.source.get(day_key, default)
        return float(interpolate(day, night, self.weight))

    def endpoint_config(self, night):
        """Fixed filter endpoints; neither config nor shared mode is mutated."""
        config = dict(self.source)
        if not night and not config.get('USE_NIGHT_COLOR', True):
            for key in ('IMAGE_DENOISE', 'IMAGE_DENOISE_STRENGTH', 'BILATERAL_SIGMA_COLOR',
                        'BILATERAL_SIGMA_SPACE', 'SCNR_ALGORITHM', 'SCNR_MTF_MIDTONES'):
                if key + '_DAY' in config:
                    config[key] = config[key + '_DAY']
        config['USE_NIGHT_COLOR'] = True
        return config


def transition_forecast(config, when, latitude, longitude, elevation=0):
    """Bound predictions to the solar cycle containing now, including polar days."""
    start = float(config.get('NIGHT_SUN_ALT_DEG', -6.0))
    end = float(config.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
    night_weight(start, start, end)  # validate imported configurations too
    obs = observer_at(when, latitude, longitude, elevation)
    sun = ephem.Sun()
    noon = obs.previous_transit(sun)
    midnight = obs.next_antitransit(sun, start=noon)
    next_noon = obs.next_transit(sun, start=midnight)

    def crossing(altitude, rising):
        obs.horizon = str(altitude)  # strings are degrees in PyEphem
        try:
            t = (obs.next_rising if rising else obs.next_setting)(
                sun, start=midnight if rising else noon, use_center=True)
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            return None
        return t if noon <= t <= next_noon else None

    dusk_start, dusk_end = crossing(start, False), crossing(end, False)
    dawn_start, dawn_end = crossing(end, True), crossing(start, True)
    altitudes = []
    for t in (noon, midnight, next_noon):
        obs.date = t
        sun.compute(obs)
        altitudes.append(math.degrees(sun.alt))
    maximum = night_weight(min(altitudes), start, end)
    minimum = night_weight(max(altitudes), start, end)
    return {
        'dusk_minutes': (dusk_end - dusk_start) * 1440 if dusk_start is not None and dusk_end is not None else None,
        'dawn_minutes': (dawn_end - dawn_start) * 1440 if dawn_start is not None and dawn_end is not None else None,
        'maximum': maximum, 'minimum': minimum,
        'lowest_altitude': min(altitudes),
        # Near the poles the Sun can move in one direction throughout the
        # cycle; a solar antitransit is then not a twilight reversal.
        'reversal_utc': (midnight.datetime().replace(tzinfo=timezone.utc)
                         if altitudes[1] < min(altitudes[0], altitudes[2]) else None),
    }
