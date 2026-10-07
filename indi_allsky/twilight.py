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
    """Return the night fraction, clamped to the configured solar interval."""
    if not all(math.isfinite(v) for v in (altitude, start, end)) or end >= start:
        raise ValueError('Full night elevation must be below the full day elevation')
    u = max(0.0, min(1.0, (start - altitude) / (start - end)))
    # Smoothstep gives zero slope at both endpoints. Using elevation rather
    # than elapsed time also lets partial summer nights reverse naturally.
    return u * u * (3.0 - 2.0 * u)


def day_altitude(config):
    """Preserve the mode-threshold endpoint until a separate day endpoint is saved."""
    altitude = config.get('TWILIGHT_TRANSITION', {}).get('DAY_ALT')
    return config.get('NIGHT_SUN_ALT_DEG', -6.0) if altitude is None else altitude


def interpolate(day, night, weight, logarithmic=False):
    """Blend scalar values; positive exposure limits can use equal ratios."""
    # Preserve exact endpoints, including camera minimum values.
    if weight <= 0 or day == night:
        return day
    if weight >= 1:
        return night
    if logarithmic and day > 0 and night > 0:
        return math.exp((1.0 - weight) * math.log(day) + weight * math.log(night))
    return day + weight * (night - day)


def runtime_weight(config):
    """None means ordinary day/night selection; zero is an active day endpoint."""
    if config.get('TWILIGHT_TRANSITION', {}).get('ENABLE', False):
        return config.get('_TWILIGHT_WEIGHT')
    return None


def exposure_minimum(config, exposure_utils, night, weight=None):
    """Use camera-resolved limits, with the original mode fallback when inactive."""
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
    """Use the same solar curve for scheduling as for image processing."""
    weight = night_weight(altitude, day_altitude(config),
                          config.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
    return interpolate(config['EXPOSURE_PERIOD_DAY'], config['EXPOSURE_PERIOD'], weight)


def observer_at(when, latitude, longitude, elevation):
    """Use geometric solar elevation, consistent with the capture mode switch."""
    obs = ephem.Observer()
    obs.lat, obs.lon = math.radians(latitude), math.radians(longitude)
    obs.elevation, obs.pressure = elevation, 0
    # A naive frame datetime uses the capture host's local timezone.
    obs.date = when.astimezone(timezone.utc)
    return obs


class TwilightTransition:
    """Keep saved endpoints separate from the per-frame settings consumers share."""

    def __init__(self, config):
        self.source = config
        self.enabled = bool(config.get('TWILIGHT_TRANSITION', {}).get('ENABLE', False))
        self.config = dict(config) if self.enabled else config
        self.weight = None
        self.altitude = None

    def update(self, when, latitude, longitude, elevation):
        """Resolve the Sun at capture time, including when replaying queued frames."""
        if not self.enabled:
            return
        obs = observer_at(when, latitude, longitude, elevation)
        sun = ephem.Sun(obs)
        self.apply(math.degrees(sun.alt))

    def apply(self, altitude):
        """Rebuild effective values from saved endpoints so repeated blends cannot drift."""
        if not self.enabled:
            return
        self.altitude = altitude
        self.weight = night_weight(altitude, day_altitude(self.source),
                                   self.source.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
        self.config['_TWILIGHT_WEIGHT'] = self.weight
        target = self.value('TARGET_ADU', 'TARGET_ADU_DAY', 75)
        # Both slots must agree: consumers still select by the operational
        # day/night flag, whose threshold is independent of these endpoints.
        self.config['TARGET_ADU'] = self.config['TARGET_ADU_DAY'] = target

        if self.source.get('USE_NIGHT_COLOR', True):
            return  # Sharing night colors does not disable the exposure transition.

        for key, default in COLOR_DEFAULTS.items():
            self.config[key] = self.config[key + '_DAY'] = self.value(key, key + '_DAY', default)

        # Optional compatibility with highlight protection.  No highlight module
        # or settings are required by the independent transition feature.
        highlight = self.source.get('HIGHLIGHT_PROTECTION', {})
        if highlight.get('ENABLE'):
            # Resolve "inherit" against each saved endpoint before blending.
            gamma_day = highlight.get('GAMMA_DAY', 0.0) or self.source.get('GAMMA_CORRECTION_DAY', 1.0)
            gamma_night = highlight.get('GAMMA', 0.0) or self.source.get('GAMMA_CORRECTION', 1.0)
            gamma = interpolate(gamma_day, gamma_night, self.weight)
            self.config['HIGHLIGHT_PROTECTION'] = dict(highlight, GAMMA=gamma, GAMMA_DAY=gamma)

    def value(self, night_key, day_key, default=False, color=False):
        """Blend saved scalars or on/off effects, honoring shared night colors."""
        night = self.source.get(night_key, default)
        day = night if color and self.source.get('USE_NIGHT_COLOR', True) else self.source.get(day_key, default)
        return float(interpolate(day, night, self.weight))

    def endpoint_config(self, night):
        """Copy one discrete profile without changing the shared camera mode."""
        config = dict(self.source)
        if not night and not config.get('USE_NIGHT_COLOR', True):
            for key in ('IMAGE_DENOISE', 'IMAGE_DENOISE_STRENGTH', 'BILATERAL_SIGMA_COLOR',
                        'BILATERAL_SIGMA_SPACE', 'SCNR_ALGORITHM', 'SCNR_MTF_MIDTONES'):
                if key + '_DAY' in config:
                    config[key] = config[key + '_DAY']
        # Existing filters now read the selected profile from their night keys.
        config['USE_NIGHT_COLOR'] = True
        return config


def transition_forecast(config, when, latitude, longitude, elevation=0):
    """Bound predictions to the solar cycle containing now, including polar days."""
    start = float(day_altitude(config))
    end = float(config.get('TWILIGHT_TRANSITION', {}).get('NIGHT_ALT', -12.0))
    night_weight(start, start, end)  # validate imported configurations too
    obs = observer_at(when, latitude, longitude, elevation)
    sun = ephem.Sun()
    # A noon-to-noon cycle contains one dusk/dawn pair. Reject crossings from
    # later cycles rather than presenting a months-long polar twilight duration.
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
    # Include both noons: seasonal drift can make either the higher endpoint.
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
