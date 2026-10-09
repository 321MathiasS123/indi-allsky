"""Sky denoising before stretching, always using only current-frame pixels.

Measured point sources and resolved edges protect detail while luminance noise,
colour mottling and compact negative outliers are treated separately. No temporal
stacking, trained model, calibration subtraction or generated detail is involved.
An optional bounded coordinate catalogue confirms faint point protection.
Strength 3 is the reference tuning; pixel measurements use the original frame.
The one-sided dark-patch repair changes local background and aperture flux:
this filter is intended for display images, not photometric measurements.
"""
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import platform
import time

import cv2
import numpy as np
from scipy.ndimage import maximum_filter
from skimage.restoration import denoise_nl_means

WEIGHTS = np.array([0.114, 0.587, 0.299], dtype=np.float32)
STAR_PROTECTION_DAY_ALT = -6.0
STAR_PROTECTION_NIGHT_ALT = -8.0
logger = logging.getLogger('indi_allsky')


def _star_protection(sun_altitude):
    """Fade point protection as stars become visible; unknown dates stay safe."""
    try:
        altitude = float(sun_altitude)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(altitude):
        return 1.0
    x = np.clip((STAR_PROTECTION_DAY_ALT - altitude)
                / (STAR_PROTECTION_DAY_ALT - STAR_PROTECTION_NIGHT_ALT), 0, 1)
    return float(x * x * (3 - 2 * x))


def _sky_geometry(shape, config, binning):
    """Map the configured lens circle back to the unrotated sensor image.

    Denoising precedes rotation and flips. Lens offsets describe the oriented
    image; undo that orientation before measuring the sky. A small inset avoids
    sampling the black fisheye edge. No camera-specific coordinates are used.
    """
    height, width = shape[:2]
    binning = max(1, int(binning))
    offset = np.array([config.get('LENS_OFFSET_X', 0),
                       -config.get('LENS_OFFSET_Y', 0)], dtype=float) / binning
    if config.get('IMAGE_FLIP_H'):
        offset[0] *= -1
    if config.get('IMAGE_FLIP_V'):
        offset[1] *= -1
    angle = int(config.get('IMAGE_ROTATE_ANGLE', 0) or 0)
    rotation = cv2.getRotationMatrix2D((0, 0), -angle, 1)[:, :2]
    offset = rotation @ offset
    orientation = config.get('IMAGE_ROTATE')
    if orientation == 'ROTATE_90_CLOCKWISE':
        offset = np.array([offset[1], -offset[0]])
    elif orientation == 'ROTATE_90_COUNTERCLOCKWISE':
        offset = np.array([-offset[1], offset[0]])
    elif orientation == 'ROTATE_180':
        offset = -offset
    circle = config.get('IMAGE_CIRCLE_MASK', {})
    diameter = config.get('LENS_IMAGE_CIRCLE', min(height, width) * binning)
    if circle.get('ENABLE'):
        diameter = min(diameter, circle.get('DIAMETER', diameter))
    radius = max(0, float(diameter) / (2 * binning) - 20)
    return width / 2 + offset[0], height / 2 + offset[1], radius


def _sky_mask(shape, geometry):
    x, y, radius = geometry
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 < radius ** 2


def repair_bayer(raw, config, binning=1):
    """Repair only isolated, extreme negative same-colour Bayer samples.

    Work on a copy so that the calibrated FITS and capture data are preserved.
    Requiring separation from every neighbour rejects broad dark structures.
    This is not another dark-frame subtraction or a general hot-pixel filter.
    """
    if raw.ndim != 2 or min(raw.shape) < 16:
        return raw
    valid = _sky_mask(raw.shape, _sky_geometry(raw.shape, config, binning))
    result = raw.copy()
    kernel = np.ones((3, 3), np.uint8)
    kernel[1, 1] = 0
    for y in range(2):
        for x in range(2):
            plane = raw[y::2, x::2].astype(np.float32)
            median = cv2.medianBlur(plane, 3)
            sigma = _noise_grid(plane - median, 64)
            neighbour_min = cv2.erode(plane, kernel)
            bad = ((plane < median - 6 * sigma)
                   & (plane < neighbour_min - 2 * sigma)
                   & valid[y::2, x::2])
            bad[:2] = bad[-2:] = False
            bad[:, :2] = bad[:, -2:] = False
            target = result[y::2, x::2]
            target[bad] = median[bad].astype(raw.dtype)
    return result


def denoise(image, config, binning=1, strength=3, sun_altitude=None,
            catalogue=None, capture_context=None):
    """Filter BGR or monochrome data, retaining its layout and numeric range."""
    if strength <= 0 or min(image.shape[:2]) < 32:
        return image
    geometry = _sky_geometry(image.shape, config, binning)
    height, width = image.shape[:2]
    x, y, radius = geometry
    # Retain a full neighbourhood outside the lens circle. Tile-aligned,
    # symmetric padding keeps statistics stable and avoids filtering black
    # corners of large sensors unnecessarily.
    space = min(x - radius, y - radius, width - x - radius, height - y - radius)
    margin = max(0, int((space - 128) // 128) * 128)
    if min(height, width) - 2 * margin < 32:
        return image
    roi = np.s_[margin:height - margin, margin:width - margin]
    source = image[roi]
    valid = _sky_mask(source.shape, (x - margin, y - margin, radius))
    if not np.any(valid):
        return image
    integer = np.issubdtype(image.dtype, np.integer)
    maximum = float(np.iinfo(image.dtype).max) if integer else 1.0
    linear = source.astype(np.float32) / maximum
    mono = linear.ndim == 2
    if mono:
        linear = np.repeat(linear[:, :, None], 3, axis=2)
    temporal = None
    if catalogue is not None and capture_context is not None:
        temporal = dict(capture_context)
        temporal['geometry_key'] = (temporal.get('geometry_key'), image.shape,
                                    int(binning), geometry, margin)
        temporal['sensor_shape'] = image.shape[:2]
    filtered = _denoise_linear(linear, valid, strength, _star_protection(sun_altitude),
                               catalogue, temporal, margin)
    if mono:
        filtered = filtered[:, :, 0]
    filtered = np.clip(filtered * maximum, 0, maximum)
    if integer:
        filtered = np.rint(filtered)
    result = image.copy()
    result[roi] = filtered.astype(image.dtype)
    return result


def _denoise_linear(image, valid, strength=3, star_protection=1.0,
                    catalogue=None, capture_context=None, margin=0):
    """Reference tuning at 3; gentler blends at 1/2, stronger smoothing at 4/5."""
    strength = max(1, min(int(strength), 5))
    evidence = _source_evidence(image, valid, star_protection,
                                catalogue, capture_context, margin)
    evidence['highpass'] = _original_highpass(evidence)
    stronger = max(0, strength - 3)
    filtered, sigma = _nlm_clean(image, evidence, h_multiplier=1 + 0.15 * stronger)
    if sigma <= 1e-8:
        return image.copy()
    result = _blend(image, filtered, evidence, 1)
    result, _, _ = _repair_compact_lows(result, valid)
    prepared = _prepare_split(result, evidence, valid)
    result = _apply(result, prepared, colour=1, brightness=0.5 + 0.125 * stronger)
    candidate, _, _ = _refine_pits(result, evidence, valid, amount=0.85 + 0.05 * stronger)
    result, _ = _protect_horizon(result, candidate, valid)
    _repair_sensor_pixels(result, evidence.get('sensor_defects'), evidence.get('sensor_clearance'))
    if strength < 3:
        result = image + (result - image) * (0.5 if strength == 1 else 0.75)
    return np.clip(result, 0, 1)


def _noise_grid(data, tile=128):
    """Robust local scatter of an already high-pass-filtered image."""
    h, w = data.shape
    ny = (h + tile - 1) // tile
    nx = (w + tile - 1) // tile
    grid = np.empty((ny, nx), np.float32)
    full_width = w // tile * tile

    def row(y):
        start, end = y * tile, min(h, (y + 1) * tile)
        if full_width:
            # Own the tile samples before partitioning them in place. Batching
            # one tile row bounds scratch memory and retains the original grid.
            samples = data[start:end, :full_width].reshape(end - start, -1, tile)
            samples = samples.transpose(1, 0, 2).reshape(-1, (end - start) * tile).copy()
            centers = np.median(samples, axis=1, overwrite_input=True)
            samples -= centers[:, None]
            np.abs(samples, out=samples)
            # Match the original Python-float scaling before rounding to float32.
            scatter = np.median(samples, axis=1, overwrite_input=True).astype(np.float64) * 1.4826
            grid[y, :w // tile] = np.maximum(scatter, 1e-06)
        if full_width < w:
            a = data[start:end, full_width:]
            grid[y, -1] = max(float(np.median(np.abs(a - np.median(a)))) * 1.4826, 1e-06)

    workers = min(4, os.cpu_count() or 1, ny)
    if workers > 1 and data.size >= 512 * 512:
        # NumPy's partitions release the GIL; rows write disjoint grid cells.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(row, range(ny)))
    else:
        for y in range(ny):
            row(y)
    return cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)

def _source_evidence(image, valid=None, star_protection=1.0,
                     catalogue=None, capture_context=None, margin=0):
    """Find multi-scale, multi-colour point sources and strong resolved edges."""
    lum = image @ WEIGHTS
    fine = cv2.GaussianBlur(lum, (0, 0), 1.0)
    dog = fine - cv2.GaussianBlur(lum, (0, 0), 3.5)
    broad = cv2.GaussianBlur(lum, (0, 0), 1.8) - cv2.GaussianBlur(lum, (0, 0), 5.0)
    noise = _noise_grid(dog)
    broad_noise = _noise_grid(broad)
    score = np.minimum(dog / noise, broad / broad_noise)
    colour_dog = cv2.GaussianBlur(image, (0, 0), 1) - cv2.GaussianBlur(image, (0, 0), 3.5)
    colour_scores = [
        colour_dog[:, :, c] / _noise_grid(colour_dog[:, :, c]) for c in range(3)
    ]
    # Median of three without stacking and partitioning a full RGB image.
    # fmin preserves partition's ordering when one of the scores is NaN.
    a, b, c = colour_scores
    support = np.fmin(np.maximum(a, b), np.maximum(np.fmin(a, b), c))
    temporal = catalogue is not None and capture_context is not None and star_protection > 0
    peaks = ((dog == maximum_filter(dog, size=5))
             & (score > (2.0 if temporal else 2.7)) & (support > 1.2))
    if valid is not None:
        peaks &= valid
    # Incomplete neighbourhoods at crop boundaries are not reliable sources.
    peaks[:12] = False
    peaks[-12:] = False
    peaks[:, :12] = False
    peaks[:, -12:] = False
    if temporal:
        support_y, support_x = np.nonzero(peaks)
        support_points = np.column_stack((support_x + margin, support_y + margin,
                                          score[support_y, support_x]))
        peaks &= score > 2.7
    yy, xx = np.nonzero(peaks)
    strengths = score[yy, xx]
    distance = cv2.distanceTransform((~peaks).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Retain four-pixel RGB cores; smoothly taper their wings to 6.5 pixels.
    mask = np.clip((6.5 - distance) / 2.5, 0, 1)
    # Fade only point candidates: daytime noise highs can resemble faint stars.
    # Keep the full-night arithmetic and independent resolved-edge guard intact.
    if star_protection < 1:
        mask *= star_protection
    coherent = fine - cv2.GaussianBlur(lum, (0, 0), 4)
    edge = np.clip((np.abs(coherent) / _noise_grid(coherent) - 7) / 5, 0, 1)
    edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8)
    mask = np.maximum(mask, edge)
    points = np.column_stack((xx, yy, strengths))
    evidence = dict(lum=lum, mask=mask, points=points, noise=noise)
    if temporal:
        start = time.monotonic()
        sensor_points = points.copy()
        sensor_points[:, :2] += margin
        # A sparse outer ring rejects broad sources and trails from the fixed
        # sensor-pixel hypothesis. Ambiguous morphology stays protected.
        outer = np.maximum.reduce([dog[yy + dy, xx + dx] for dy, dx in
            ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (-3, 3), (3, -3), (3, 3))])
        compact = outer < 0.3 * dog[yy, xx]
        defects = _single_colour_points(colour_dog, colour_scores, support, distance, valid)
        sensor_defects = defects.copy()
        sensor_defects[:, :2] += margin
        weights, diagnostics = catalogue.weights(sensor_points, support_points,
                                                  compact=compact, sensor_points=sensor_defects,
                                                  **capture_context)
        evidence['catalogue'] = diagnostics
        defect_weights = diagnostics.get('sensor_weights', ())
        if len(defect_weights) == len(defects):
            proven = np.asarray(diagnostics.get('sensor_stationary', np.zeros(len(defects), bool)))
            selected = proven & (np.asarray(defect_weights) < 1)
            evidence['sensor_defects'] = np.column_stack((defects[selected, :2],
                                                          1 - np.asarray(defect_weights)[selected]))
            # Keep only tiny patch masks, not another full-frame distance map.
            evidence['sensor_clearance'] = [
                distance[int(y)-4:int(y)+5, int(x)-4:int(x)+5] > 6.5
                for x, y in defects[selected, :2]]
        if np.any(weights < 1):
            # A coherent current-frame trail must not wait for persistence.
            # Reuse measured fields at sparse points instead of introducing
            # another image-wide line detector or protecting every broad peak.
            line = _line_sources(dog, noise, xx, yy)
            weights[line] = 1
            # Keep the original exclusion mask for every noise/background
            # statistic. Only current-source restoration changes with history.
            evidence['restore_mask'] = _weighted_points(
                edge, points, weights * star_protection, diagnostics.get('stationary_mask'))
            stationary = np.asarray(diagnostics.get('stationary_mask', np.zeros(len(points), bool)))
            selected = np.flatnonzero(stationary & (weights < 1))
            if len(selected):
                # Noise in a second channel can move the same proven defect
                # between source tables. Both paths need the final RGB repair.
                normal_defects = np.column_stack((points[selected, :2], 1 - weights[selected]))
                evidence['sensor_defects'] = np.concatenate((
                    evidence.get('sensor_defects', np.empty((0, 3))), normal_defects))
                clearance = evidence.setdefault('sensor_clearance', [])
                gy, gx = np.mgrid[-4:5, -4:5]
                for index in selected:
                    x, y = xx[index], yy[index]
                    # Exclude only this defect's own source core. Other source
                    # wings remain untouched, using just a tiny local mask.
                    nearby = (np.abs(xx - x) < 11) & (np.abs(yy - y) < 11)
                    nearby[index] = False
                    clear = np.ones((9, 9), bool)
                    for nx, ny in zip(xx[nearby], yy[nearby]):
                        clear &= (gx + x - nx)**2 + (gy + y - ny)**2 > 6.5**2
                    clearance.append(clear)
        logger.info('Sky source catalogue status=%s points=%d confirmed=%d suppressed=%d stationary=%d sensor_pixels=%d time=%.3fs',
                    diagnostics.get('status', 'unknown'), len(points),
                    diagnostics.get('confirmed', 0), diagnostics.get('suppressed', 0),
                    diagnostics.get('stationary', 0), len(evidence.get('sensor_defects', ())),
                    time.monotonic() - start)
    elif catalogue is not None and star_protection == 0:
        catalogue.reset()
    return evidence


def _single_colour_points(colour_dog, scores, support, distance, valid):
    """Compact single-channel evidence for the sensor table, never star votes.

    Reuse the existing channel measurements. Only sparse high-significance
    candidates need neighbourhood checks; no further image-wide filters.
    """
    candidates = ((scores[0] > 6) | (scores[1] > 6) | (scores[2] > 6))
    candidates &= (support <= 1.2) & (distance > 6.5)
    if valid is not None:
        candidates &= valid
    candidates[:12] = candidates[-12:] = False
    candidates[:, :12] = candidates[:, -12:] = False
    # An unexpected large coloured structure must not create unbounded work.
    if np.count_nonzero(candidates) > 20000:
        return np.empty((0, 3))
    y, x = np.nonzero(candidates)
    values = np.column_stack([channel[y, x] for channel in scores])
    channel = np.argmax(values, axis=1)
    peak = colour_dog[y, x, channel]
    keep = np.ones(len(x), bool)
    for dy, dx in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)):
        neighbour = colour_dog[y + dy, x + dx, channel]
        # Pick one deterministic centroid on a flat maximum.
        keep &= peak > neighbour if (dy, dx) < (0, 0) else peak >= neighbour
    for dy, dx in ((-4,0), (4,0), (0,-4), (0,4), (-3,-3), (-3,3), (3,-3), (3,3)):
        keep &= colour_dog[y + dy, x + dx, channel] < .3 * peak
    return np.column_stack((x[keep], y[keep], values[np.arange(len(x)), channel][keep]))


def _repair_sensor_pixels(image, defects, clearance=None):
    """Fade proven tiny sensor residuals toward the current local background.

    NLM can retain a strong positive outlier even without a protection mask.
    Its luminance also spreads into the other channels, so repair the small RGB
    core after filtering. Symmetric interpolation avoids retaining only the
    negative noise excursions in a repaired core. Use only current-frame pixels.
    The caller owns this working image; originals and noise statistics stay put.
    """
    if defects is None or not len(defects):
        return
    y, x = np.mgrid[-7:8, -7:8]
    radius = np.hypot(x, y)
    ring = (radius >= 5) & (radius <= 7)
    taper = np.clip((4 - radius[3:12, 3:12]) / 2, 0, 1)[:, :, None]
    for index, (cx, cy, amount) in enumerate(defects):
        cx, cy = int(cx), int(cy)
        background = np.median(image[cy-7:cy+8, cx-7:cx+8][ring], axis=0)
        patch = image[cy-4:cy+5, cx-4:cx+5]
        # A passing star can overlap the repair footprint despite its centre
        # being farther away. Preserve every pixel in its protected wings.
        weight = taper if clearance is None else taper * clearance[index][:, :, None]
        patch += amount * weight * (background - patch)


def _line_sources(dog, noise, x, y):
    """Positive signal at three separated samples supports a real short trail."""
    line = np.zeros(len(x), bool)
    for angle in np.arange(32) * (np.pi / 16):
        supported = np.ones(len(x), bool)
        for radius in (4, 8, 12):
            dx, dy = int(round(radius * np.cos(angle))), int(round(radius * np.sin(angle)))
            supported &= dog[y + dy, x + dx] > 2 * noise[y + dy, x + dx]
        line |= supported
    return line


def _weighted_points(edge, points, weights, stationary=None):
    """Splat the existing 4-pixel core/6.5-pixel taper without another image pass.

    Each offset addresses unique pixels, so vectorized maxima also handle
    overlapping sources correctly. Resolved edges keep their original guard.
    """
    result = edge.copy()
    x, y = points[:, :2].astype(np.intp).T
    if stationary is not None and np.any(stationary):
        # A proven compact sensor residual can also trigger the strong-edge
        # guard. Fade only its tiny core; the statistics mask remains intact.
        gy, gx = np.mgrid[-4:5, -4:5]
        core = np.clip((4 - np.hypot(gx, gy)) / 2, 0, 1).astype(np.float32)
        for index in np.flatnonzero(stationary):
            result[y[index]-4:y[index]+5, x[index]-4:x[index]+5] *= 1 - (1 - weights[index]) * core
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            taper = min(1.0, max(0.0, (6.5 - (dx * dx + dy * dy) ** 0.5) / 2.5))
            if taper:
                # Sources are at least 12 pixels from every image boundary.
                yy, xx = y + dy, x + dx
                result[yy, xx] = np.maximum(result[yy, xx], weights * taper)
    return result

def _nlm_clean(image, evidence, h_multiplier=1.0):
    """Use measured sky scatter, with a small local patch-search window."""
    lum = evidence['lum']
    hp = _original_highpass(evidence)
    sky = (evidence['mask'] < 0.01) & (lum > 0.005)
    sample = hp[sky]
    if sample.size == 0:
        # Low-bit-depth sensors may never reach the reference sky threshold
        # within their uint16 container. Use their positive background instead.
        sample = hp[(evidence['mask'] < 0.01) & (lum > 0)]
    if sample.size == 0:
        return image.copy(), 0.0
    sigma = float(np.median(np.abs(sample - np.median(sample)))) * 1.4826
    if sigma <= 1e-8:
        return image.copy(), sigma
    clean_lum = _nlm_luminance(lum, h_multiplier * sigma)
    # Colour speckle is smoothed independently; source RGB is blended back later.
    chroma = image - lum[:, :, None]
    clean_chroma = cv2.GaussianBlur(chroma, (0, 0), 1.4)
    return (clean_lum[:, :, None] + clean_chroma, sigma)


def _original_highpass(evidence):
    """Reuse the same original-frame noise samples across the three stages."""
    if 'highpass' in evidence:
        return evidence['highpass']
    lum = evidence['lum']
    return lum - cv2.GaussianBlur(lum, (0, 0), 2)


def _nlm_luminance(lum, h):
    """Run the same local filter in bounded, overlapping regions."""
    options = dict(h=h, patch_size=5, patch_distance=5, sigma=0,
                   fast_mode=True, preserve_range=True, channel_axis=None)
    workers = min(4, os.cpu_count() or 1, max(1, len(lum) // 256))
    if workers == 1:
        return denoise_nl_means(lum, **options)
    height, width = lum.shape
    if platform.machine().lower().startswith(('arm', 'aarch64')):
        # Small tiles reduce repeated main-memory traffic on the Pi. Desktop
        # CPUs benefit more from wider strips and less reflected padding.
        rows = [*range(0, height, 64), height]
        columns = [*range(0, width, 64), width]
    else:
        rows = np.linspace(0, height, workers + 1, dtype=int)
        columns = [0, width]
    result = np.empty_like(lum)
    # Include both the search distance and patch radius at every internal edge.
    # Keep the original outer edges so scikit-image's reflection stays unchanged.
    halo = options['patch_distance'] + options['patch_size'] // 2

    def run(position):
        y, bottom, x, right = position
        first, last = max(0, y - halo), min(height, bottom + halo)
        left, edge = max(0, x - halo), min(width, right + halo)
        cleaned = denoise_nl_means(lum[first:last, left:edge], **options)
        result[y:bottom, x:right] = cleaned[y - first:bottom - first, x - left:right - left]

    # NLM releases the GIL. Each worker writes only its non-overlapping interior.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        positions = ((y, bottom, x, right)
                     for y, bottom in zip(rows, rows[1:])
                     for x, right in zip(columns, columns[1:]))
        list(pool.map(run, positions))
    return result


def _blend(original, filtered, evidence, amount):
    weight = (amount * (1 - evidence.get('restore_mask', evidence['mask'])))[:, :, None]
    return original + (filtered - original) * weight

def _repair_compact_lows(image, valid, threshold=3.0, max_area=16):
    """Lift small, significant negative excursions, leaving brighter channels."""
    lum = image @ WEIGHTS
    median = cv2.medianBlur(lum, 5)
    residual = lum - median
    sigma = _noise_grid(residual)
    low = (residual < -1.5 * sigma) & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    strong = (residual < -threshold * sigma) & valid
    seeds = np.bincount(labels[strong], minlength=n) > 0
    keep = seeds & (stats[:, cv2.CC_STAT_AREA] <= max_area)
    keep[0] = False
    selected = keep[labels]
    target = np.maximum(image, cv2.medianBlur(image, 5))
    result = image.copy()
    result[selected] = target[selected]
    return result, selected, dict(
        components=int(keep.sum()), pixels=int(selected.sum()),
        sky_percentage=float(selected[valid].mean() * 100),
    )

def _blur(a, s):
    return cv2.GaussianBlur(a, (0, 0), s)

def _prepare(image, evidence, valid, fine=2.5, shared=None, channel=None):
    """Measure background texture without borrowing signal from protected stars."""
    # channel selects the statistics retained by the split: 0=luma, 1=chroma.
    if shared is None:
        shared = {}
    if not shared:
        # Weighted samples are zero outside valid. Their Gaussian support
        # extends 10 pixels at sigma=2.5, then 28 at sigma=7; retain 40.
        roi = _sky_slice(valid, 40)
        shared['roi'] = roi
        weight = (1 - evidence['mask'][roi]) * valid[roi]
        norm = _blur(weight, 2.5)
        weighted_image = image[roi] * weight[:, :, None]
        background = _blur(weighted_image, 2.5) / np.maximum(norm[:, :, None], 0.0001)
        shared.update(weight=weight, norm=norm, weighted_image=weighted_image, background=background)
        shared['norm_full'] = np.zeros(valid.shape, np.float32)
        shared['norm_full'][roi] = norm
    else:
        weight = shared['weight']
        norm = shared['norm']
        weighted_image = shared['weighted_image']
        background = shared['background']
    roi = shared['roi']
    if fine < 2.5:
        small_norm = _blur(weight, fine)
        small = _blur(weighted_image, fine) / np.maximum(small_norm[:, :, None], 0.0001)
        mix = np.clip((small_norm - 0.1) / 0.5, 0, 1)
        background = background + mix[:, :, None] * (small - background)
    norm = shared['norm_full']
    # Do not infer a background where large objects leave no usable samples.
    if 'trust' not in shared:
        trust = (np.clip((norm - 0.05) / 0.2, 0, 1)
                 * np.clip(cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5) / 20, 0, 1))
        shared['trust'] = _blur(trust, 3) * valid
    trust = shared['trust']
    # Restore the original frame and tile coordinates for noise measurements.
    band = np.zeros_like(image)
    band[roi] = background - _blur(background, 7)
    lum = band @ WEIGHTS
    chroma = band - lum[:, :, None] if channel != 0 else None
    # Quieter tiles keep clouds from inflating the estimated noise floor.
    sigmas = []
    for y in range(0, image.shape[0] - 128, 128):
        for x in range(0, image.shape[1] - 128, 128):
            sl = np.s_[y:y + 128, x:x + 128]
            sel = (norm[sl] > 0.7) & valid[sl]
            if sel.sum() < 8192:
                continue
            a = band[sl][sel]
            l = a @ WEIGHTS
            c = a - l[:, None] if channel != 0 else None
            # Keep the reference scalar precision when a statistic is skipped.
            sigmas.append([
                np.median(np.abs(l - np.median(l))) * 1.4826 if channel != 1 else np.float32(0),
                np.sqrt(np.mean(np.median(np.abs(c - np.median(c, axis=0)), axis=0) ** 2)) * 1.4826 if channel != 0 else np.float32(0),
            ])
    # Without enough clear background, skip this optional scale correction.
    noise = np.maximum(np.percentile(sigmas, 25, axis=0), 1e-6) if sigmas else np.zeros(2)
    if 'original_sigma' not in shared:
        hp = _original_highpass(evidence)
        sample = hp[(evidence['mask'] < 0.01) & valid]
        shared['original_sigma'] = np.median(np.abs(sample - np.median(sample))) * 1.4826 if sample.size else 0.0
    original_sigma = shared['original_sigma']
    # Cap daytime luminance smoothing using the original fine-scale noise.
    noise[0] = min(noise[0], 0.35 * original_sigma)
    return dict(lum=lum, chroma=chroma, trust=trust, noise=noise)

def _apply(image, prepared, colour=0.85, brightness=0.6):
    """Suppress low-power mottling while retaining coherent cloud/edge texture."""
    l, c, t = (prepared['lum'], prepared['chroma'], prepared['trust'])
    nl, nc = prepared['noise']
    wl = np.minimum(1, (1.5 * nl) ** 2 / (_blur(l * l, 3) + 1e-12))
    wc = np.minimum(1, (1.5 * nc) ** 2 / (_blur(np.mean(c * c, axis=2), 3) + 1e-12))
    dl = _blur(l * wl * t, 1) * brightness * t
    dc = _blur(c * (wc * t)[:, :, None], 1) * colour * t[:, :, None]
    correction = dl[:, :, None] + dc
    # Keep this two-sided scale correction neutral in mean sky brightness.
    sky = t > 0.99
    if np.any(sky):
        correction -= correction[sky].mean(axis=0)[None, None, :] * t[:, :, None]
    return image - correction

def _prepare_split(image, evidence, valid):
    """Treat finer colour blotches separately from stellar luminance detail."""
    # Reuse only within this frame; skip statistics discarded by the split.
    shared = {}
    broad = _prepare(image, evidence, valid, fine=2.5, shared=shared, channel=0)
    fine = _prepare(image, evidence, valid, fine=1.2, shared=shared, channel=1)
    broad['chroma'] = fine['chroma']
    broad['noise'][1] = fine['noise'][1]
    return broad

def _sky_slice(valid, padding):
    """Keep every valid pixel and a requested halo, preserving image borders."""
    rows = np.flatnonzero(valid.any(axis=1))
    if not rows.size:
        return np.s_[:, :]
    columns = np.flatnonzero(valid.any(axis=0))
    return np.s_[max(0, rows[0] - padding):min(valid.shape[0], rows[-1] + padding + 1),
                 max(0, columns[0] - padding):min(valid.shape[1], columns[-1] + padding + 1)]


def _refine_pits(image, evidence, valid, amount=0.85, radius=12, inner=4,
                 sector_rank=1, colour_amount=1.0):
    """Bounded luminance and chroma repair of the accepted mottling baseline.

    Positive compact cores, not entire circular source neighborhoods, are kept.
    By default, three of four sectors must support a higher local background. Broad
    coherent structure is checked at a larger scale using original noise.
    """
    raw = evidence['lum']
    lum = image @ WEIGHTS
    dog = cv2.GaussianBlur(raw, (0, 0), 1.25) - cv2.GaussianBlur(raw, (0, 0), 5)
    dog_sigma = _noise_grid(dog)
    peaks = (dog == cv2.dilate(dog, np.ones((7, 7), np.uint8))) & (dog > 2.7 * dog_sigma) & valid
    distance = cv2.distanceTransform((~peaks).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    weight = (np.clip((distance - 3) / 2, 0, 1) * valid).astype(np.float32)
    original_sigma = _noise_grid(_original_highpass(evidence))
    broad = cv2.GaussianBlur(raw, (0, 0), 6) - cv2.GaussianBlur(raw, (0, 0), 18)
    structure_trust = np.clip((1.5 - np.abs(broad) / (original_sigma + 1e-08)) / 0.8, 0, 1)
    # Preserve the full-frame measurements above. The remaining convolutions
    # and connected components only contribute inside valid, with a local halo.
    full_image = image
    # The annular kernels need radius pixels; the final sigma=0.6 blur needs 3.
    roi = _sky_slice(valid, max(radius, 3))
    image = image[roi]
    raw = raw[roi]
    lum = lum[roi]
    distance = distance[roi]
    weight = weight[roi]
    original_sigma = original_sigma[roi]
    structure_trust = structure_trust[roi]
    valid = valid[roi]
    gy, gx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    rr = gx * gx + gy * gy
    ring = (rr <= radius * radius) & (rr >= inner * inner)
    quadrants = [ring & (gx >= np.abs(gy)), ring & (-gx >= np.abs(gy)),
                 ring & (gy >= np.abs(gx)), ring & (-gy >= np.abs(gx))]
    backgrounds = []
    supports = []
    average_rgb = np.zeros_like(image)
    weighted_image = image * weight[:, :, None]
    for q in quadrants:
        kernel = q.astype(np.float32)
        kernel /= kernel.sum()
        norm = cv2.filter2D(weight, -1, kernel)
        background_rgb = (cv2.filter2D(weighted_image, -1, kernel)
                          / np.maximum(norm[:, :, None], 1e-05))
        backgrounds.append(background_rgb @ WEIGHTS)
        average_rgb += background_rgb * 0.25
        supports.append(norm)
    rim = np.stack(backgrounds)
    support = np.min(supports, axis=0)
    average = rim.mean(axis=0)
    if sector_rank == 1:
        # Second-lowest of four sectors, without sorting every pixel's values.
        lower_a = np.minimum(rim[0], rim[1])
        upper_a = np.maximum(rim[0], rim[1])
        lower_b = np.minimum(rim[2], rim[3])
        upper_b = np.maximum(rim[2], rim[3])
        lowest = np.minimum(np.maximum(lower_a, lower_b), np.minimum(upper_a, upper_b))
        del lower_a, upper_a, lower_b, upper_b
    else:
        lowest = np.partition(rim, sector_rank, axis=0)[sector_rank]
    # Protect actual positive source samples, not negative outliers that merely
    # happen to fall within a circular star-protection neighbourhood.
    guard = np.where(distance < 6.5,
                     np.clip((average - raw) / (0.35 * original_sigma + 1e-08), 0, 1), 1)
    guard *= valid
    depression = lowest - cv2.GaussianBlur(lum, (0, 0), 0.6)
    enclosed = np.clip((depression - 0.5 * original_sigma) / (0.8 * original_sigma + 1e-08), 0, 1)
    low = (enclosed > 0.05) & (average - lum > 0.5 * original_sigma) & valid
    n, labels, stats, _ = cv2.connectedComponentsWithStats(low.astype(np.uint8), 8)
    area = stats[:, cv2.CC_STAT_AREA]
    w = stats[:, cv2.CC_STAT_WIDTH]
    h = stats[:, cv2.CC_STAT_HEIGHT]
    keep = ((area <= 196) & (w <= 24) & (h <= 24)
            & (np.maximum(w, h) <= 4 * np.maximum(1, np.minimum(w, h))))
    keep[0] = False
    stable = np.clip((support - 0.2) / 0.4, 0, 1)
    delta = (amount * np.maximum(average - lum, 0) * enclosed * guard
             * stable * structure_trust * keep[labels])
    delta = np.minimum(delta, 3.0 * original_sigma)
    brightest = np.maximum(np.maximum(image[:, :, 0], image[:, :, 1]), image[:, :, 2])
    delta = np.minimum(delta, np.maximum(1 - brightest, 0)).astype(np.float32)
    result = image + delta[:, :, None]
    # Grey increments alone leave coloured halos. Repair chroma only where a
    # luminance deficit was filled, using the source-excluded annular background.
    # Most pixels have no deficit; avoid building full RGB temporaries for them.
    selected = delta > 0
    selected_result = result[selected]
    selected_image = image[selected]
    selected_lum = lum[selected]
    selected_average = average[selected]
    fraction = np.clip(delta[selected] / np.maximum(selected_average - selected_lum, 1e-08), 0, 1) * colour_amount
    colour = (average_rgb[selected] - selected_average[:, None]
              - (selected_image - selected_lum[:, None])) * fraction[:, None]
    colour -= (colour @ WEIGHTS)[:, None]
    # Stay in range without changing the zero-luminance colour direction.
    allowed = np.where(
        colour > 0, np.maximum(1 - selected_result, 0) / np.maximum(colour, 1e-08),
        np.where(colour < 0, np.maximum(selected_result, 0) / np.maximum(-colour, 1e-08), 1),
    )
    colour_scale = np.minimum(1, allowed.min(axis=1))
    result[selected] += colour * colour_scale[:, None]
    statistics = dict(
        components=int(keep.sum()), pixels=int((delta > 0).sum()),
        sky_percentage=float(np.mean(delta[valid] > 0) * 100),
        mean_linear_luma_added=float(delta[valid].mean()),
        max_linear_luma_added=float(delta.max()),
    )
    if result.shape != full_image.shape:
        full_result = full_image.copy()
        full_result[roi] = result
        full_delta = np.zeros(full_image.shape[:2], dtype=delta.dtype)
        full_delta[roi] = delta
        result, delta = full_result, full_delta
    return result, delta, statistics

def _protect_horizon(baseline, candidate, valid):
    """Exclude the outer 50 pixels from extra pit repair; taper the next 30."""
    distance = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
    weight = np.clip((distance - 50) / 30, 0, 1)
    result = candidate.copy()
    transition = (weight > 0) & (weight < 1)
    np.copyto(result, baseline, where=(weight == 0)[:, :, None])
    result[transition] = baseline[transition] + (candidate[transition] - baseline[transition]) * weight[transition, None]
    return (result, weight)
