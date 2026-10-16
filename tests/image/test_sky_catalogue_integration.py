"""Temporal confidence changes restoration, not the measured noise mask."""
import ast
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from indi_allsky import asi676mc, constants, sky_denoise


class Catalogue:
    def __init__(self, value):
        self.value = value
        self.received = None
        self.resets = 0

    def weights(self, points, support, **context):
        self.received = points.copy(), support.copy(), context
        return np.full(len(points), self.value, np.float32), {}

    def reset(self):
        self.resets += 1


@pytest.fixture
def scene():
    image = np.random.default_rng(8271).normal(.15, .008, (256, 256, 3)).astype(np.float32)
    y, x = np.ogrid[:256, :256]
    for cx, cy, amplitude in [(128, 128, .12), (190, 145, .035)]:
        image += (amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / 3))[:, :, None]
    image[:, :55] += .2
    return image


def test_weighted_cores_match_independent_overlapping_radial_envelopes():
    edge = np.zeros((64, 64), np.float32)
    edge[:, 20] = .7
    points = np.array([[25, 25, 4], [30, 25, 8], [37, 30, 5]])
    weights = np.array([.5, 1, 0], np.float32)
    y, x = np.indices(edge.shape)
    expected = edge.copy()
    for (cx, cy, _), confidence in zip(points, weights):
        taper = np.clip((6.5 - np.hypot(x - cx, y - cy)) / 2.5, 0, 1)
        expected = np.maximum(expected, confidence * taper)
    np.testing.assert_allclose(sky_denoise._weighted_points(edge, points, weights), expected, atol=1e-7)
    assert edge[25, 25] == 0


def test_proven_compact_sensor_residual_fades_only_its_edge_core():
    edge = np.ones((64, 64), np.float32)
    points = np.array([[25, 25, 12], [45, 45, 15]])
    weights = np.array([.5, 1], np.float32)
    result = sky_denoise._weighted_points(edge, points, weights, np.array([True, False]))
    assert result[25, 25] == .5
    assert result[45, 45] == 1
    assert result[25, 29] == 1
    assert result[25, 28] == .75
    assert np.count_nonzero(result != edge) < 49


def test_trail_support_requires_separated_current_positive_samples():
    dog = np.zeros((64, 64), np.float32)
    noise = np.ones_like(dog)
    dog[32, 28:45] = 3
    dog[20, 20] = 4  # Isolated source has no extended evidence.
    assert sky_denoise._line_sources(dog, noise, np.array([28, 20]), np.array([32, 20])).tolist() == [True, False]
    dog[32, 40] = -3  # One bright neighbour cannot bridge a missing segment.
    assert not sky_denoise._line_sources(dog, noise, np.array([28]), np.array([32]))[0]


def test_history_leaves_noise_statistics_and_edges_unchanged(scene):
    valid = np.ones(scene.shape[:2], bool)
    reference = sky_denoise._source_evidence(scene, valid)
    catalogue = Catalogue(.5)
    measured = sky_denoise._source_evidence(scene, valid, catalogue=catalogue,
                                           capture_context={'capture_time': 100}, margin=128)
    for key in ('mask', 'noise', 'lum', 'points'):
        np.testing.assert_array_equal(measured[key], reference[key])
    assert measured['restore_mask'][145, 190] < measured['mask'][145, 190]
    np.testing.assert_array_equal(measured['restore_mask'][80:180, 53:57],
                                  measured['mask'][80:180, 53:57])
    points, support, context = catalogue.received
    np.testing.assert_array_equal(points[:, :2], reference['points'][:, :2] + 128)
    assert len(support) >= len(points)
    assert np.all(support[:, 2] > 2)
    assert context['compact'].shape == (len(points),)


def test_confirmed_identity_weights_preserve_entire_filter_exactly(scene):
    original = scene.copy()
    baseline = sky_denoise.denoise(scene, {})
    catalogue = Catalogue(1)
    result = sky_denoise.denoise(scene, {}, catalogue=catalogue,
                                 capture_context={'capture_time': 100, 'geometry_key': 'camera'})
    np.testing.assert_array_equal(result, baseline)
    np.testing.assert_array_equal(scene, original)
    assert catalogue.received[2]['sensor_shape'] == scene.shape[:2]


def test_daylight_and_missing_context_keep_existing_filter(scene):
    catalogue = Catalogue(0)
    baseline = sky_denoise.denoise(scene, {}, sun_altitude=5)
    result = sky_denoise.denoise(scene, {}, sun_altitude=5, catalogue=catalogue,
                                 capture_context={'capture_time': 100, 'geometry_key': 'camera'})
    np.testing.assert_array_equal(result, baseline)
    assert catalogue.received is None and catalogue.resets == 1
    baseline = sky_denoise.denoise(scene, {})
    np.testing.assert_array_equal(sky_denoise.denoise(scene, {}, catalogue=catalogue), baseline)


@pytest.fixture
def processor():
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ImageProcessor')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in {'_denoise_temporal_kwargs', '_denoise_star_reference'}]
    ns = dict(__name__='indi_allsky.processing', __package__='indi_allsky',
              constants=constants, json=json, math=math, timedelta=timedelta, asi676mc=asi676mc)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
    obj = ns['ImageProcessor']()
    obj.config = dict(EXPOSURE_PERIOD=20, EXPOSURE_PERIOD_DAY=25)
    obj.night_av = [False] * 8
    ref = SimpleNamespace(exp_elapsed=30.9, exposure=30,
                          exp_date_utc=datetime(2026, 10, 8, 19, 0, tzinfo=timezone.utc),
                          binning=1, camera_id=1, image_bayerpat='RGGB', capture_night=True)
    obj.image_list = [ref]
    return obj, ref


def test_context_uses_capture_midpoint_and_reuses_one_catalogue(processor):
    obj, ref = processor
    first = obj._denoise_temporal_kwargs(ref)
    second = obj._denoise_temporal_kwargs(ref)
    assert first['catalogue'] is second['catalogue']
    assert set(first) == {'catalogue', 'capture_context'}
    context = first['capture_context']
    assert context['capture_time'] == (ref.exp_date_utc - timedelta(seconds=15.9)).timestamp()
    assert context['capture_interval'] == 30.9
    assert context['geometry_key'] == (1, 'RGGB')
    ref.gain = 200
    ref.exposure = 20
    assert obj._denoise_temporal_kwargs(ref)['catalogue'] is first['catalogue']


@pytest.mark.parametrize('invalid', ['preview', 'stack', 'bad_frame', 'nan'])
def test_nonindependent_frames_clear_history_without_using_it(processor, invalid):
    obj, ref = processor
    obj._sky_source_catalogue = Catalogue(0)
    if invalid == 'preview':
        ref.exp_elapsed = 0
    elif invalid == 'stack':
        obj.image_list.append(ref)
    elif invalid == 'bad_frame':
        ref.asi676mc_repair_result = {'status': 'excluded'}
    else:
        ref.exposure = float('nan')
    assert obj._denoise_temporal_kwargs(ref) == {}
    assert obj._sky_source_catalogue.resets == 1


@pytest.fixture
def star_reference(processor, monkeypatch):
    instances = []

    class Reference:
        def __init__(self, *args):
            self.args = args
            instances.append(self)

    monkeypatch.setitem(sys.modules, 'indi_allsky.sky_reference',
                        SimpleNamespace(VirtualSkyReference=Reference))
    obj, ref = processor
    obj.config.update(IMAGE_DENOISE_CATALOGUE=True, VIRTUALSKY={'OFFSET_X': 0})
    obj.image = np.zeros((80, 100, 3), np.uint8)
    obj.position_av = [0.] * 8
    obj.position_av[constants.POSITION_LATITUDE] = 53.24
    obj.position_av[constants.POSITION_LONGITUDE] = 11.38
    ref.camera_uuid = 'test-camera'
    return obj, ref, instances


def test_reference_reuses_mapping_and_capture_time_without_extra_ephemeris(star_reference):
    obj, ref, instances = star_reference
    first = obj._denoise_temporal_kwargs(ref)
    assert first['capture_context']['reference_time'] == int(ref.exp_date_utc.timestamp())
    assert first['capture_context']['capture_time'] == (ref.exp_date_utc - timedelta(seconds=15.9)).timestamp()
    ref.exp_date_utc += timedelta(seconds=31)
    second = obj._denoise_temporal_kwargs(ref)
    assert first['star_reference'] is second['star_reference'] is instances[0]
    assert len(instances) == 1
    assert second['capture_context']['capture_time'] - first['capture_context']['capture_time'] == 31
    assert second['capture_context']['reference_time'] - first['capture_context']['reference_time'] == 31
    assert instances[0].args[1:] == ((80, 100, 3), 1, 'test-camera', 53.24, 11.38)
    obj.config['IMAGE_DENOISE_STRENGTH'] = 5
    assert obj._denoise_temporal_kwargs(ref)['star_reference'] is instances[0]
    obj.config['IMAGE_DENOISE_CATALOGUE'] = False
    disabled = obj._denoise_temporal_kwargs(ref)
    assert set(disabled) == {'catalogue', 'capture_context'}
    assert 'reference_time' not in disabled['capture_context']


@pytest.mark.parametrize('change', ['nested_geometry', 'rotation', 'border', 'shape',
                                  'binning', 'camera', 'position', 'privacy'])
def test_reference_mapping_invalidates_only_when_its_context_changes(star_reference, change):
    obj, ref, instances = star_reference
    first = obj._denoise_temporal_kwargs(ref)['star_reference']
    if change == 'nested_geometry':
        obj.config['VIRTUALSKY']['OFFSET_X'] = 2
    elif change == 'rotation':
        obj.config['IMAGE_ROTATE_ANGLE'] = 15
    elif change == 'border':
        obj.config['IMAGE_BORDER'] = {'TOP': 10}
    elif change == 'shape':
        obj.image = np.zeros((60, 90), np.uint16)
    elif change == 'binning':
        ref.binning = 2
    elif change == 'camera':
        ref.camera_uuid = 'other-camera'
    elif change == 'position':
        obj.position_av[constants.POSITION_LONGITUDE] += .1
    else:
        obj.config['PRIVACY_MODE'] = True
    second = obj._denoise_temporal_kwargs(ref)['star_reference']
    assert second is not first and len(instances) == 2
    assert obj._denoise_temporal_kwargs(ref)['star_reference'] is second
    if change == 'privacy':
        assert second.args[-2:] == (53., 11.)


@pytest.mark.parametrize('invalid', ['preview', 'stack', 'bad_frame', 'missing_camera', 'bad_geometry'])
def test_reference_falls_back_without_preventing_normal_processing(star_reference, invalid):
    obj, ref, instances = star_reference
    if invalid == 'preview':
        ref.exp_elapsed = 0
    elif invalid == 'stack':
        obj.image_list.append(ref)
    elif invalid == 'bad_frame':
        ref.asi676mc_repair_result = {'status': 'excluded'}
    elif invalid == 'missing_camera':
        del ref.camera_uuid
    else:
        obj.config['VIRTUALSKY'] = object()  # Malformed config cannot be cached/projected.
    result = obj._denoise_temporal_kwargs(ref)
    assert 'star_reference' not in result
    assert len(instances) == 0
    if invalid in {'missing_camera', 'bad_geometry'}:
        assert set(result) == {'catalogue', 'capture_context'}
    else:
        assert result == {}
