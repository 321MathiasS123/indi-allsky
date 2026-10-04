"""Gain precision shared by camera commands and exposure bookkeeping."""
import math


def gain_quantum(driver):
    # These SDK/driver paths take integer gain: ASI/SVB long, Player One
    # intValue, and ToupBase's integer ExpoAGain. INDI's numeric `step` is
    # often only a GUI increment (ASI uses range/10), not hardware resolution.
    # Fractional/unknown interfaces retain their precision; never infer a
    # hardware step from that GUI hint. DSLR ISO lists are handled separately.
    integer_drivers = {
        'indi_asi_ccd', 'indi_asi_single_ccd',
        'indi_playerone_ccd', 'indi_playerone_single_ccd',
        'indi_svbony_ccd', 'indi_svbonycam_ccd', 'indi_sv305_ccd',
        'indi_toupcam_ccd', 'indi_altair_ccd', 'indi_altaircam_ccd',
        'indi_nncam_ccd', 'indi_tscam_ccd', 'indi_ogmacam_ccd', 'indi_omegonprocam_ccd',
    }
    return 1.0 if driver in integer_drivers else 0.0


def quantize_gain(value, quantum=0.0, values=()):
    """Nearest supported command, with deterministic half-step rounding."""
    if values:
        return min(values, key=lambda gain: (abs(gain - value), gain))
    if quantum:
        return math.floor(value / quantum + 0.5) * quantum
    return value


def gain_limits(minimum, maximum, quantum=0.0, values=()):
    """Representable limits inside a configured range, including fixed gain."""
    if values:
        allowed = [gain for gain in values if minimum <= gain <= maximum]
        if allowed:
            return min(allowed), max(allowed)
    elif quantum:
        lower = math.ceil(minimum / quantum) * quantum
        upper = math.floor(maximum / quantum) * quantum
        if lower <= upper:
            return lower, upper
    else:
        return minimum, maximum
    # A fixed/interpolated value (or a range narrower than one hardware step)
    # cannot meet both bounds. Use its nearest supported value consistently.
    fixed = quantize_gain((minimum + maximum) / 2, quantum, values)
    return fixed, fixed
