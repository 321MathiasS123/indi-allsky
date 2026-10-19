import math

import numpy
import pytest

from indi_allsky.highlight_twilight import clipping_budget, measure_ring, twilight_weight


CONFIG = {'LENS_IMAGE_CIRCLE': 400, 'LENS_OFFSET_X': 0, 'LENS_OFFSET_Y': 0}


def scene(shape=(440, 440), center=(220, 220), radius=200, baseline=0.1,
          dtype=numpy.uint16, depth=16):
    image = numpy.full(shape, round(baseline * ((1 << depth) - 1)), dtype=dtype)
    rows, columns = numpy.indices(shape[:2])
    distance = numpy.hypot(columns - center[0], rows - center[1]) / radius
    angle = numpy.degrees(numpy.arctan2(rows - center[1], columns - center[0]))
    return image, distance, angle


def patch(image, distance, angle, *, radius=(0.85, 0.95), azimuth=0, width=80,
          value=65535):
    arc = numpy.abs((angle - azimuth + 180) % 360 - 180) < width / 2
    image[arc & (distance >= radius[0]) & (distance <= radius[1])] = value


@pytest.mark.parametrize('altitude', [None, float('nan'), float('inf'),
                                    -float('inf'), 'bad', -90, -14, 0, 50])
def test_inactive_or_invalid_sun_altitudes_have_no_weight(altitude):
    assert twilight_weight(altitude) == 0


@pytest.mark.parametrize('altitude,expected', [(-13.5, 0.15625), (-13, 0.5),
                                             (-12, 1), (-8, 1), (-2, 1),
                                             (-1, 0.5), (-0.5, 0.15625)])
def test_weight_is_smooth_and_symmetric(altitude, expected):
    assert twilight_weight(altitude) == pytest.approx(expected)
    assert twilight_weight(-14 - altitude) == pytest.approx(expected)


def test_envelope_has_zero_slope_at_boundaries_and_plateau():
    epsilon = 0.0001
    for altitude in (-14, -12, -2, 0):
        assert abs(twilight_weight(altitude + epsilon)
                   - twilight_weight(altitude - epsilon)) / epsilon < 0.001


PHASE_CONFIG = {'HIGHLIGHT_PROTECTION': {'TWILIGHT_STAR_ALLOWANCE': .02,
                                        'TWILIGHT_MOON_ALLOWANCE_50': .7,
                                        'TWILIGHT_MOON_ALLOWANCE_FULL': 2.4}}


@pytest.mark.parametrize('phase,expected', [(0, .0002), (50, .007), (100, .024)])
def test_phase_reference_allowances_are_total_percentages_of_ring(phase, expected):
    assert clipping_budget(phase, 10, PHASE_CONFIG) == pytest.approx(expected)


def test_phase_allowance_is_monotonic_and_continuous_through_half_moon():
    budgets = [clipping_budget(phase, 10, PHASE_CONFIG) for phase in numpy.linspace(0, 100, 201)]
    assert all(before <= after for before, after in zip(budgets, budgets[1:]))
    epsilon = .001
    middle = clipping_budget(50, 10, PHASE_CONFIG)
    left = (middle - clipping_budget(50 - epsilon, 10, PHASE_CONFIG)) / epsilon
    right = (clipping_budget(50 + epsilon, 10, PHASE_CONFIG) - middle) / epsilon
    assert left > 0 and right == pytest.approx(left, rel=.001)
    for phase, direction in ((0, 1), (100, -1)):
        endpoint = clipping_budget(phase, 10, PHASE_CONFIG)
        assert math.isfinite(endpoint)
        distances = [abs(clipping_budget(phase + direction * delta, 10, PHASE_CONFIG) - endpoint)
                     for delta in (.01, .0001, .000001, .00000001)]
        assert all(after <= before for before, after in zip(distances, distances[1:]))
        assert distances[-1] < distances[0] / 5


def test_default_physical_phase_curve_keeps_measured_sixty_percent_moon_allowance():
    assert clipping_budget(60.4, 10, {}) * 100 == pytest.approx(2.59986, abs=.00001)


@pytest.mark.parametrize('phase', [0, 25, 50, 75, 100])
@pytest.mark.parametrize('altitude', [-90, -2, -1])
def test_moon_below_visibility_envelope_only_has_star_allowance(phase, altitude):
    assert clipping_budget(phase, altitude, PHASE_CONFIG) == pytest.approx(.0002)


def test_moon_visibility_fades_smoothly_from_minus_one_to_zero_degrees():
    below = clipping_budget(100, -1, PHASE_CONFIG)
    above = clipping_budget(100, 0, PHASE_CONFIG)
    assert clipping_budget(100, -.5, PHASE_CONFIG) == pytest.approx((below + above) / 2)
    budgets = [clipping_budget(100, altitude, PHASE_CONFIG) for altitude in numpy.linspace(-1, 0, 101)]
    assert all(before <= after for before, after in zip(budgets, budgets[1:]))
    epsilon = .0001
    for altitude in (-1, 0):
        derivative = (clipping_budget(100, altitude + epsilon, PHASE_CONFIG)
                      - clipping_budget(100, altitude - epsilon, PHASE_CONFIG)) / epsilon
        assert abs(derivative) < .00001


@pytest.mark.parametrize('phase,altitude', [(float('nan'), 10), (50, float('inf')),
                                         (None, 10), ('bad', 10), (50, None)])
def test_invalid_moon_context_produces_a_finite_bounded_allowance(phase, altitude):
    budget = clipping_budget(phase, altitude, PHASE_CONFIG)
    assert math.isfinite(budget) and 0 <= budget <= .0002


@pytest.mark.parametrize('settings', [None, {'TWILIGHT_STAR_ALLOWANCE': -1},
                                    {'TWILIGHT_MOON_ALLOWANCE_FULL': 51},
                                    {'TWILIGHT_MOON_ALLOWANCE_50': 25},
                                    {'TWILIGHT_STAR_ALLOWANCE': float('nan')},
                                    {'TWILIGHT_MOON_ALLOWANCE_FULL': float('inf')}])
def test_invalid_allowances_disable_area_omission(settings):
    assert clipping_budget(100, 10, {'HIGHLIGHT_PROTECTION': settings}) == 0


def test_out_of_range_moon_phase_is_bounded_to_reference_allowances():
    assert clipping_budget(-100, 10, PHASE_CONFIG) == pytest.approx(.0002)
    assert clipping_budget(200, 10, PHASE_CONFIG) == pytest.approx(.024)


@pytest.mark.parametrize('azimuth', [0, 90, 179, -179])
@pytest.mark.parametrize('radial_range', [(0.85, 0.95), (0.98, 0.99)])
def test_broad_horizon_is_retained_even_when_thin_and_wrapping(azimuth, radial_range):
    image, distance, angle = scene()
    patch(image, distance, angle, azimuth=azimuth, radius=radial_range)
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(1)


def test_any_colour_channel_is_protected():
    image, distance, angle = scene(shape=(440, 440, 3))
    patch(image[:, :, 2], distance, angle, value=60000)
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(60000 / 65535)


@pytest.mark.parametrize('bright_pixels', [1, 4, 100])
def test_source_allowance_omits_exactly_bounded_pixel_area(bright_pixels):
    image, distance, angle = scene()
    candidates = numpy.column_stack(numpy.nonzero((distance > .8) & (distance < .99)))[:4096]
    mask = numpy.zeros(image.shape, numpy.uint8)
    mask[candidates[:, 0], candidates[:, 1]] = 255
    image[candidates[:bright_pixels, 0], candidates[:bright_pixels, 1]] = 65535
    assert measure_ring(image, mask, 16, CONFIG, clipping_fraction=bright_pixels / 4096) == pytest.approx(6554 / 65535)
    assert measure_ring(image, mask, 16, CONFIG, clipping_fraction=(bright_pixels - 1) / 4096) == 1
    assert measure_ring(image, mask, 16, CONFIG) == 1


def test_scattered_stars_and_tiny_meteor_fit_within_the_same_area_margin():
    image, distance, angle = scene()
    generator = numpy.random.default_rng(8)
    candidates = numpy.column_stack(numpy.nonzero((distance > 0.75) & (distance < 1)))
    stars = candidates[generator.choice(len(candidates), 10, replace=False)]
    image[stars[:, 0], stars[:, 1]] = 65535
    image[220, 390:408] = 65535  # A thin meteor adds eighteen pixels.
    count = numpy.count_nonzero((distance > .7) & (distance <= 1))
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=32.5 / count) == pytest.approx(6554 / 65535)
    assert measure_ring(image, None, 16, CONFIG) == 1


@pytest.mark.parametrize('azimuth', [0, 90, 179])
def test_narrow_arc_is_detected_regardless_of_angular_width_when_over_budget(azimuth):
    image, distance, angle = scene()
    patch(image, distance, angle, azimuth=azimuth, width=.5)
    count = numpy.count_nonzero((distance > .7) & (distance <= 1))
    bright = numpy.count_nonzero(image == 65535)
    assert bright > 0
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=(bright - .5) / count) == 1
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=(bright + .5) / count) == pytest.approx(6554 / 65535)


@pytest.mark.parametrize('radial_center', [.8, .9, .95])
def test_moon_core_fits_allowance_without_excluding_its_entire_halo(radial_center):
    image, distance, _ = scene()
    rows, columns = numpy.indices(image.shape)
    moon_distance = numpy.hypot(columns - (220 + 200 * radial_center), rows - 220)
    image[moon_distance <= 20] = 45000
    image[moon_distance <= 6] = 65535
    valid = (distance > .7) & (distance <= 1)
    bright = numpy.count_nonzero((image == 65535) & valid)
    allowance = (bright + .5) / numpy.count_nonzero(valid)
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=allowance) == pytest.approx(45000 / 65535)
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=allowance) < .9


def test_moon_does_not_hide_broad_twilight_underneath():
    image, distance, angle = scene()
    patch(image, distance, angle, width=70, value=52000)
    patch(image, distance, angle, width=12, value=65535)
    valid = (distance > .7) & (distance <= 1)
    allowance = (numpy.count_nonzero((image == 65535) & valid) + .5) / numpy.count_nonzero(valid)
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=allowance) == pytest.approx(52000 / 65535)


def test_thin_horizon_beside_low_moon_and_halo_is_still_protected():
    image, distance, angle = scene()
    patch(image, distance, angle, radius=(0.98, 0.99), width=75, value=52000)
    rows, columns = numpy.indices(image.shape)
    moon_distance = numpy.hypot(columns - 410, rows - 220)
    image[moon_distance <= 20] = 58000
    image[moon_distance <= 5] = 65535
    valid = (distance > .7) & (distance <= 1)
    allowance = (numpy.count_nonzero((image == 65535) & valid) + .5) / numpy.count_nonzero(valid)
    # The unexcluded halo stays below the .9 raw protection target.
    assert measure_ring(image, None, 16, CONFIG, clipping_fraction=allowance) == pytest.approx(58000 / 65535)


def test_radially_drifting_thin_horizon_keeps_local_angular_support():
    image, distance, angle = scene()
    horizon_radius = 0.88 + angle * 0.001
    image[(numpy.abs(angle) < 60) & (numpy.abs(distance - horizon_radius) < 0.006)] = 52000
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(52000 / 65535)


def test_inner_boundary_feathers_without_including_centre_or_outside_circle():
    image, distance, angle = scene(baseline=0)
    image[(distance < 0.70) | (distance > 1)] = 65535
    assert measure_ring(image, None, 16, CONFIG) == 0
    patch(image, distance, angle, radius=(0.723, 0.727))
    assert 0.43 < measure_ring(image, None, 16, CONFIG) < 0.57


def test_corrected_geometry_uses_offset_sign_and_binning_after_crop():
    config = dict(CONFIG, LENS_IMAGE_CIRCLE=600, LENS_OFFSET_X=80, LENS_OFFSET_Y=-90)
    image, distance, angle = scene(shape=(400, 460), center=(270, 245), radius=150,
                                    baseline=0)
    patch(image, distance, angle, azimuth=70, radius=(0.98, 0.995))
    assert measure_ring(image, None, 16, config, binning=2) == pytest.approx(1)
    assert measure_ring(image, None, 16, dict(config, LENS_OFFSET_Y=90), binning=2) == 0


def test_fractional_binned_radius_matches_the_display_mask_rounding():
    image, distance, _ = scene(shape=(200, 200), center=(100, 100), radius=66, baseline=0)
    image[(distance > 1) & (distance < 67/66)] = 65535
    assert measure_ring(image, None, 16, dict(CONFIG, LENS_IMAGE_CIRCLE=401), binning=3) == 0


def test_enabled_image_circle_diameter_overrides_lens_diameter():
    image, distance, angle = scene()
    patch(image, distance, angle)
    config = dict(CONFIG, LENS_IMAGE_CIRCLE=150,
                  IMAGE_CIRCLE_MASK={'ENABLE': True, 'DIAMETER': 400})
    assert measure_ring(image, None, 16, config) == pytest.approx(1)
    config['IMAGE_CIRCLE_MASK']['ENABLE'] = False
    assert measure_ring(image, None, 16, config) == pytest.approx(6554 / 65535)


@pytest.mark.parametrize('diameter', [0, -4])
def test_nonpositive_mask_diameter_falls_back_to_lens(diameter):
    image, distance, angle = scene()
    patch(image, distance, angle)
    config = dict(CONFIG, IMAGE_CIRCLE_MASK={'ENABLE': True, 'DIAMETER': diameter})
    assert measure_ring(image, None, 16, config) == pytest.approx(1)


def test_custom_mask_excludes_pixels_without_restricting_to_brightness_roi():
    image, distance, angle = scene()
    patch(image, distance, angle, azimuth=0)
    patch(image, distance, angle, azimuth=120, value=50000)
    mask = numpy.ones(image.shape, dtype=numpy.uint8)
    mask[numpy.abs(angle) < 45] = 0
    unchanged, original_mask = image.copy(), mask.copy()
    assert measure_ring(image, mask, 16, CONFIG) == pytest.approx(50000 / 65535)
    numpy.testing.assert_array_equal(image, unchanged)
    numpy.testing.assert_array_equal(mask, original_mask)


def test_mask_holes_do_not_hide_over_budget_valid_pixels():
    image, distance, angle = scene(baseline=0)
    patch(image, distance, angle, width=34)
    mask = numpy.ones(image.shape, dtype=numpy.uint8)
    mask[numpy.abs(angle) < 4] = 0
    assert measure_ring(image, mask, 16, CONFIG) == 1


def test_mask_hole_keeps_broad_valid_horizon_on_either_side():
    image, distance, angle = scene()
    patch(image, distance, angle, width=90)
    mask = numpy.ones(image.shape, dtype=numpy.uint8)
    mask[numpy.abs(angle) < 5] = 0
    assert measure_ring(image, mask, 16, CONFIG) == pytest.approx(1)


@pytest.mark.parametrize('depth,dtype', [(8, numpy.uint8), (10, numpy.uint16),
                                       (12, numpy.uint16), (14, numpy.uint16),
                                       (16, numpy.uint16)])
def test_normalization_uses_sensor_depth(depth, dtype):
    image, distance, angle = scene(depth=depth, dtype=dtype)
    value = round(0.93 * ((1 << depth) - 1))
    patch(image, distance, angle, value=value)
    assert measure_ring(image, None, depth, CONFIG) == pytest.approx(value / ((1 << depth) - 1))


@pytest.mark.parametrize('changes', [{'LENS_IMAGE_CIRCLE': 0}, {'LENS_IMAGE_CIRCLE': -2},
                                   {'LENS_IMAGE_CIRCLE': float('nan')},
                                   {'LENS_IMAGE_CIRCLE': float('inf')},
                                   {'LENS_OFFSET_X': float('inf')},
                                   {'LENS_OFFSET_Y': 'bad'},
                                   {'IMAGE_CIRCLE_MASK': None}])
def test_invalid_geometry_disables_optional_meter(changes):
    assert measure_ring(scene()[0], None, 16, dict(CONFIG, **changes)) == 0


@pytest.mark.parametrize('binning', [0, -1, None, float('nan'), float('inf')])
def test_invalid_binning_disables_optional_meter(binning):
    assert measure_ring(scene()[0], None, 16, CONFIG, binning) == 0


@pytest.mark.parametrize('data,mask,depth,config', [
    (numpy.zeros((0, 10), numpy.uint16), None, 16, CONFIG),
    (numpy.zeros((10, 10, 4), numpy.uint16), None, 16, CONFIG),
    (numpy.zeros((10, 10), numpy.float32), None, 16, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), numpy.ones((2, 2)), 16, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), numpy.zeros((440, 440)), 16, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), numpy.full((440, 440), numpy.nan), 16, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), None, 17, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), None, 0, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), None, 12.5, CONFIG),
    (numpy.zeros((440, 440), numpy.uint16), None, 16, {}),
    (numpy.zeros((440, 440), numpy.uint16), None, 16,
     dict(CONFIG, LENS_OFFSET_X=10000)),
])
def test_unusable_inputs_disable_optional_meter(data, mask, depth, config):
    assert measure_ring(data, mask, depth, config) == 0


@pytest.mark.parametrize('fraction', [None, -.001, .5001, float('nan'), float('inf'), 'bad'])
def test_invalid_source_area_fraction_disables_optional_meter(fraction):
    assert measure_ring(scene()[0], None, 16, CONFIG, clipping_fraction=fraction) == 0
