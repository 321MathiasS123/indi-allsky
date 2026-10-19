import math

import numpy
import pytest

from indi_allsky.highlight_twilight import measure_ring, twilight_weight


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
                                    -float('inf'), 'bad', -90, -14, -2, 50])
def test_inactive_or_invalid_sun_altitudes_have_no_weight(altitude):
    assert twilight_weight(altitude) == 0


@pytest.mark.parametrize('altitude,expected', [(-13.5, 0.15625), (-13, 0.5),
                                             (-12, 1), (-8, 1), (-4, 1),
                                             (-3, 0.5), (-2.5, 0.15625)])
def test_weight_is_smooth_and_symmetric(altitude, expected):
    assert twilight_weight(altitude) == pytest.approx(expected)
    assert twilight_weight(-16 - altitude) == pytest.approx(expected)


def test_envelope_has_zero_slope_at_boundaries_and_plateau():
    epsilon = 0.0001
    for altitude in (-14, -12, -4, -2):
        assert abs(twilight_weight(altitude + epsilon)
                   - twilight_weight(altitude - epsilon)) / epsilon < 0.001


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


@pytest.mark.parametrize('azimuth', [0, 75, 179])
@pytest.mark.parametrize('radial_center', [0.8, 0.9, 0.95])
def test_compact_moon_and_bounded_halo_do_not_drive_meter(azimuth, radial_center):
    image, distance, angle = scene()
    moon_x = 220 + 200 * radial_center * math.cos(math.radians(azimuth))
    moon_y = 220 + 200 * radial_center * math.sin(math.radians(azimuth))
    rows, columns = numpy.indices(image.shape)
    moon_distance = numpy.hypot(columns - moon_x, rows - moon_y)
    image[moon_distance <= 20] = 45000  # halo, about 15 degrees across
    image[moon_distance <= 6] = 65535
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(6554 / 65535)


def test_scattered_saturated_stars_do_not_drive_meter():
    image, distance, angle = scene()
    generator = numpy.random.default_rng(8)
    candidates = numpy.column_stack(numpy.nonzero((distance > 0.75) & (distance < 1)))
    stars = candidates[generator.choice(len(candidates), 100, replace=False)]
    image[stars[:, 0], stars[:, 1]] = 65535
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(6554 / 65535)


def test_dense_stars_at_unrelated_radii_cannot_assemble_a_bright_horizon():
    image = numpy.full((2100, 2100), 6000, dtype=numpy.uint16)
    generator = numpy.random.default_rng(36)
    # Every half-degree sector has a star, but at independently scattered
    # radii. Each source already has a 3x3 footprint, as in raw CFA metering.
    angles = numpy.linspace(-math.pi, math.pi, 1000, endpoint=False)
    radii = generator.uniform(0.76, 0.99, len(angles)) * 1000
    rows = numpy.rint(1050 + numpy.sin(angles) * radii).astype(int)
    columns = numpy.rint(1050 + numpy.cos(angles) * radii).astype(int)
    for row, column in zip(rows, columns):
        image[row - 1:row + 2, column - 1:column + 2] = 65535
    config = dict(CONFIG, LENS_IMAGE_CIRCLE=2000)
    assert measure_ring(image, None, 16, config) == pytest.approx(6000 / 65535)


def test_moon_does_not_hide_broad_twilight_underneath():
    image, distance, angle = scene()
    patch(image, distance, angle, width=70, value=52000)
    patch(image, distance, angle, width=12, value=65535)
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(52000 / 65535)


def test_thin_horizon_beside_low_moon_and_halo_is_still_protected():
    image, distance, angle = scene()
    patch(image, distance, angle, radius=(0.98, 0.99), width=75, value=52000)
    rows, columns = numpy.indices(image.shape)
    moon_distance = numpy.hypot(columns - 410, rows - 220)
    image[moon_distance <= 20] = 58000
    image[moon_distance <= 5] = 65535
    assert measure_ring(image, None, 16, CONFIG) == pytest.approx(52000 / 65535)


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


def test_unknown_mask_sectors_do_not_join_two_compact_sources():
    image, distance, angle = scene(baseline=0)
    patch(image, distance, angle, width=34)
    mask = numpy.ones(image.shape, dtype=numpy.uint8)
    mask[numpy.abs(angle) < 4] = 0
    assert measure_ring(image, mask, 16, CONFIG) == 0


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
