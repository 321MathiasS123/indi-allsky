"""In-memory repair for the ASI676MC RAW16 purple-frame failure."""

from functools import lru_cache
import re
import time

import numpy


DEFAULT_SETTINGS = {
    'PURPLE_RATIO_THRESHOLD': 1.5,
    'RED_SIDE_RATIO_THRESHOLD': 1.25,
    'BLUE_SIDE_RATIO_THRESHOLD': 1.5,
    'SAMPLE_STEP': 32,
    'SOURCE_SATURATION_THRESHOLD': 65000,
    'GAIN_R': 0.91004,
    'GAIN_G1': 1.68652,
    'GAIN_G2': 1.09238,
    'GAIN_B': 0.59537,
    'CHUNK_ROWS': 128,
}


DIAGNOSTIC_METADATA_KEY = 'asi676mc_diagnostic'
DIAGNOSTIC_BAD_STATUSES = ('repaired', 'validation_failed')
# Maintenance and removal guide: docs/asi676mc-frame-repair.md


_CAMERA_NAME_RE = re.compile(r'(?<![A-Z0-9])ASI[\s_-]*676MC(?![A-Z0-9])', re.IGNORECASE)


def camera_name_matches(camera_name):
    """Return whether a detected camera name identifies an ASI676MC."""
    return bool(_CAMERA_NAME_RE.search(str(camera_name or '')))


def camera_record_matches(camera):
    """Return whether any persistent name for a camera identifies an ASI676MC."""
    return any(
        camera_name_matches(getattr(camera, attr, None))
        for attr in ('name', 'name_alt1', 'name_alt2', 'friendlyName')
    )


def diagnostic_capture_plan(pending_capture_id, status, new_capture_id=None):
    """Plan diagnostic roles for this frame and the next pending capture."""
    roles = []
    if pending_capture_id:
        roles.append({
            'capture_id': str(pending_capture_id),
            'role': 'following',
        })

    next_capture_id = None
    if status in DIAGNOSTIC_BAD_STATUSES:
        if not new_capture_id:
            raise ValueError('new_capture_id is required for a bad diagnostic frame')

        next_capture_id = str(new_capture_id)
        roles.append({
            'capture_id': next_capture_id,
            'role': 'bad',
        })

    return roles, next_capture_id


def normalize_settings(settings=None):
    """Merge and validate configured repair settings."""
    config = dict(DEFAULT_SETTINGS)
    if settings:
        config.update(settings)

    normalized = {
        'PURPLE_RATIO_THRESHOLD': float(config['PURPLE_RATIO_THRESHOLD']),
        'RED_SIDE_RATIO_THRESHOLD': float(config['RED_SIDE_RATIO_THRESHOLD']),
        'BLUE_SIDE_RATIO_THRESHOLD': float(config['BLUE_SIDE_RATIO_THRESHOLD']),
        'SAMPLE_STEP': int(config['SAMPLE_STEP']),
        'SOURCE_SATURATION_THRESHOLD': int(config['SOURCE_SATURATION_THRESHOLD']),
        'GAIN_R': float(config['GAIN_R']),
        'GAIN_G1': float(config['GAIN_G1']),
        'GAIN_G2': float(config['GAIN_G2']),
        'GAIN_B': float(config['GAIN_B']),
        'CHUNK_ROWS': int(config['CHUNK_ROWS']),
    }

    for key in (
        'PURPLE_RATIO_THRESHOLD',
        'RED_SIDE_RATIO_THRESHOLD',
        'BLUE_SIDE_RATIO_THRESHOLD',
    ):
        if normalized[key] <= 0:
            raise ValueError('{0:s} must be greater than zero'.format(key))

    for key in ('GAIN_R', 'GAIN_G1', 'GAIN_G2', 'GAIN_B'):
        if normalized[key] <= 0:
            raise ValueError('{0:s} must be greater than zero'.format(key))

    if normalized['SAMPLE_STEP'] < 2 or normalized['SAMPLE_STEP'] % 2:
        raise ValueError('SAMPLE_STEP must be an even number of at least two')

    saturation_threshold = normalized['SOURCE_SATURATION_THRESHOLD']
    if saturation_threshold < 1 or saturation_threshold > 65535:
        raise ValueError('SOURCE_SATURATION_THRESHOLD must be between 1 and 65535')

    if normalized['CHUNK_ROWS'] < 2 or normalized['CHUNK_ROWS'] % 2:
        raise ValueError('CHUNK_ROWS must be an even number of at least two')

    return normalized


def audit_metadata(
    status,
    reason=None,
    signature_before=None,
    signature_after=None,
    timing=None,
):
    """Build the JSON-safe repair audit record stored with a saved image."""
    metadata = {
        'status' : str(status),
    }

    if reason:
        metadata['reason'] = str(reason)

    for key, signature in (
        ('signature_before', signature_before),
        ('signature_after', signature_after),
    ):
        if not signature:
            continue

        metadata[key] = {
            'purple_ratio'    : float(signature['purple_ratio']),
            'red_side_ratio'  : float(signature['red_side_ratio']),
            'blue_side_ratio' : float(signature['blue_side_ratio']),
        }

    if timing:
        metadata['timing'] = {
            'detection_s' : float(timing.get('detection_s', 0.0)),
            'repair_s'    : float(timing.get('repair_s', 0.0)),
            'total_s'     : float(timing.get('total_s', 0.0)),
        }

    return metadata


def validate_raw_mosaic(data):
    """Reject inputs that cannot safely use the RGGB parity repair."""
    if not isinstance(data, numpy.ndarray):
        raise ValueError('repair requires a NumPy array')
    if data.ndim != 2:
        raise ValueError('repair requires a two-dimensional RAW16 frame')
    if data.dtype.kind != 'u' or data.dtype.itemsize != 2:
        raise ValueError('repair requires unsigned 16-bit RAW data')
    if not data.flags.writeable:
        raise ValueError('repair requires a writable RAW frame')

    height, width = data.shape
    if height < 4 or width < 4:
        raise ValueError('repair requires at least four rows and four columns')
    if height % 2 or width % 2:
        raise ValueError('repair requires even RAW frame dimensions')


def frame_signature(data, settings=None):
    """Measure the four Bayer parities and classify the purple-frame fault."""
    validate_raw_mosaic(data)
    config = normalize_settings(settings)
    return _frame_signature(data, config)


def _frame_signature(data, config):
    """Measure a validated frame using already-normalized settings."""
    height, width = data.shape
    sample_step = config['SAMPLE_STEP']

    y_start = (height // 4) & ~1
    y_stop = (3 * height // 4) & ~1
    x_start = (width // 4) & ~1
    x_stop = (3 * width // 4) & ~1

    medians = [
        float(numpy.median(data[
            y_start + row:y_stop:sample_step,
            x_start + column:x_stop:sample_step,
        ]))
        for row in range(2)
        for column in range(2)
    ]

    green_sum = medians[1] + medians[2]
    purple_ratio = (
        (medians[0] + medians[3]) / green_sum
        if green_sum > 0
        else float('inf')
    )
    red_side_ratio = medians[0] / medians[1] if medians[1] > 0 else float('inf')
    blue_side_ratio = medians[3] / medians[2] if medians[2] > 0 else float('inf')

    is_bad = (
        purple_ratio >= config['PURPLE_RATIO_THRESHOLD']
        and red_side_ratio >= config['RED_SIDE_RATIO_THRESHOLD']
        and blue_side_ratio >= config['BLUE_SIDE_RATIO_THRESHOLD']
    )

    return {
        'parity_medians': medians,
        'purple_ratio': purple_ratio,
        'red_side_ratio': red_side_ratio,
        'blue_side_ratio': blue_side_ratio,
        'is_bad': is_bad,
    }


@lru_cache(maxsize=1)
def _build_lookup_tables(gains):
    values = numpy.arange(65536, dtype=numpy.float64)
    return tuple(
        numpy.rint(numpy.clip(values / gain, 0, 65535)).astype(numpy.uint16)
        for gain in gains
    )


def _pack_clipped_green_mask(data, saturation_threshold, chunk_rows):
    """Record clipped G1 samples compactly before applying the gain tables."""
    green1_clipped_packed, _both_green_clipped_packed = (
        _pack_clipped_green_masks(
            data,
            saturation_threshold,
            chunk_rows,
        )
    )
    return green1_clipped_packed


def _pack_clipped_green_masks(data, saturation_threshold, chunk_rows):
    """Record clipped G1 and jointly clipped green samples compactly."""
    green1 = data[0::2, 1::2]
    green2 = data[1::2, 0::2]
    plane_height, plane_width = green1.shape
    plane_chunk_rows = max(1, chunk_rows // 2)
    packed_width = (plane_width + 7) // 8
    green1_clipped_packed = numpy.empty(
        (plane_height, packed_width),
        dtype=numpy.uint8,
    )
    both_green_clipped_packed = numpy.empty(
        (plane_height, packed_width),
        dtype=numpy.uint8,
    )

    for row_start in range(0, plane_height, plane_chunk_rows):
        row_stop = min(row_start + plane_chunk_rows, plane_height)
        green1_clipped = green1[row_start:row_stop] >= saturation_threshold
        green2_clipped = green2[row_start:row_stop] >= saturation_threshold
        green1_clipped_packed[row_start:row_stop] = numpy.packbits(
            green1_clipped,
            axis=1,
        )
        numpy.logical_and(
            green1_clipped,
            green2_clipped,
            out=green2_clipped,
        )
        both_green_clipped_packed[row_start:row_stop] = numpy.packbits(
            green2_clipped,
            axis=1,
        )

    return green1_clipped_packed, both_green_clipped_packed


def _reconstruct_clipped_green(
    data,
    green1_clipped_packed,
    both_green_clipped_packed,
    chunk_rows,
):
    red = data[0::2, 0::2]
    green1 = data[0::2, 1::2]
    green2 = data[1::2, 0::2]
    blue = data[1::2, 1::2]
    plane_height, plane_width = green1.shape
    plane_chunk_rows = max(1, chunk_rows // 2)

    for row_start in range(0, plane_height, plane_chunk_rows):
        row_stop = min(row_start + plane_chunk_rows, plane_height)
        rows = numpy.arange(row_start, row_stop)

        upper = green2[numpy.maximum(rows - 1, 0)].astype(numpy.uint32)
        lower = green2[rows].astype(numpy.uint32)
        estimate = numpy.empty((row_stop - row_start, plane_width), dtype=numpy.uint16)

        estimate[:, :-1] = numpy.rint(
            (
                upper[:, :-1]
                + upper[:, 1:]
                + lower[:, :-1]
                + lower[:, 1:]
            ) / 4.0
        ).astype(numpy.uint16)
        estimate[:, -1] = numpy.rint(
            (upper[:, -1] + lower[:, -1]) / 2.0
        ).astype(numpy.uint16)

        mask = numpy.unpackbits(
            green1_clipped_packed[row_start:row_stop],
            axis=1,
            count=plane_width,
        ).view(numpy.bool_)
        target = green1[row_start:row_stop]
        replace = mask & (estimate > target)
        target[replace] = estimate[replace]

        both_green_clipped = numpy.unpackbits(
            both_green_clipped_packed[row_start:row_stop],
            axis=1,
            count=plane_width,
        ).view(numpy.bool_)
        if not numpy.any(both_green_clipped):
            continue

        # Both green values have lost their highlight information.  The
        # corrected red/blue pair provides the remaining local lower bound;
        # raising both greens to that level prevents a false magenta plateau.
        # Reuse the interpolation buffer to keep peak memory bounded.
        numpy.maximum(
            red[row_start:row_stop],
            blue[row_start:row_stop],
            out=estimate,
        )
        numpy.maximum(
            target,
            estimate,
            out=target,
            where=both_green_clipped,
        )
        green2_target = green2[row_start:row_stop]
        numpy.maximum(
            green2_target,
            estimate,
            out=green2_target,
            where=both_green_clipped,
        )


def repair_in_place(data, settings=None):
    """Restore row phase, Bayer gains, and prematurely clipped green samples."""
    validate_raw_mosaic(data)
    config = normalize_settings(settings)
    return _repair_in_place(data, config)


def _repair_in_place(data, config):
    """Repair a validated frame using already-normalized settings."""
    chunk_rows = config['CHUNK_ROWS']
    height = data.shape[0]
    gains = (
        config['GAIN_R'],
        config['GAIN_G1'],
        config['GAIN_G2'],
        config['GAIN_B'],
    )
    lookup_tables = _build_lookup_tables(gains)

    for row_start in range(0, height - 1, chunk_rows):
        row_stop = min(row_start + chunk_rows, height - 1)
        data[row_start:row_stop] = data[row_start + 1:row_stop + 1]
    data[-1] = data[-3]

    green1_clipped_packed, both_green_clipped_packed = (
        _pack_clipped_green_masks(
            data,
            config['SOURCE_SATURATION_THRESHOLD'],
            chunk_rows,
        )
    )

    for row_start in range(0, height, chunk_rows):
        row_stop = min(row_start + chunk_rows, height)
        for row_parity in range(2):
            for column_parity in range(2):
                plane = data[
                    row_start + row_parity:row_stop:2,
                    column_parity::2,
                ]
                lookup = lookup_tables[row_parity * 2 + column_parity]
                plane[:] = lookup[plane]

    _reconstruct_clipped_green(
        data,
        green1_clipped_packed,
        both_green_clipped_packed,
        chunk_rows,
    )
    return data


def repair_if_needed(data, settings=None):
    """Repair a bad frame and report before/after diagnostic signatures."""
    total_start = time.perf_counter()
    config = normalize_settings(settings)
    validate_raw_mosaic(data)

    detection_start = time.perf_counter()
    signature_before = _frame_signature(data, config)
    detection_elapsed_s = time.perf_counter() - detection_start

    if not signature_before['is_bad']:
        total_elapsed_s = time.perf_counter() - total_start
        return {
            'repaired': False,
            'validation_failed': False,
            'signature_before': signature_before,
            'signature_after': None,
            'timing': {
                'detection_s': detection_elapsed_s,
                'repair_s': 0.0,
                'total_s': total_elapsed_s,
            },
        }

    # The offline tool can leave its source FITS untouched when validation
    # fails.  The live pipeline needs the same safety property in memory, so a
    # detected bad frame is repaired on a temporary array and committed only
    # after its signature no longer matches the failure.
    repair_start = time.perf_counter()
    repaired_data = data.copy()
    repair_in_place(repaired_data, config)
    signature_after = _frame_signature(repaired_data, config)
    repair_elapsed_s = time.perf_counter() - repair_start
    total_elapsed_s = time.perf_counter() - total_start

    timing = {
        'detection_s': detection_elapsed_s,
        'repair_s': repair_elapsed_s,
        'total_s': total_elapsed_s,
    }

    if signature_after['is_bad']:
        return {
            'repaired': False,
            'validation_failed': True,
            'signature_before': signature_before,
            'signature_after': signature_after,
            'timing': timing,
        }

    data[:] = repaired_data
    return {
        'repaired': True,
        'validation_failed': False,
        'signature_before': signature_before,
        'signature_after': signature_after,
        'timing': timing,
    }
