"""Actual BPM selection and FITS loading without camera or Flask services."""
import ast
from datetime import datetime, timedelta, timezone
import logging
import math
from pathlib import Path
from types import SimpleNamespace

from astropy.io import fits
import numpy as np
import pytest
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql.expression import true as sa_true

from indi_allsky import asi676mc, constants


@pytest.fixture
def environment(tmp_path, monkeypatch):
    base = declarative_base()

    class Map(base):
        __tablename__ = 'maps'
        id = Column(Integer, primary_key=True)
        camera_id = Column(Integer)
        active = Column(Boolean)
        bitdepth = Column(Integer)
        binmode = Column(Integer)
        width = Column(Integer)
        height = Column(Integer)
        gain = Column(Float)
        exposure = Column(Float)
        temp = Column(Float)
        createDate = Column(DateTime)
        filename = Column(String)

        def getFilesystemPath(self):
            return self.filename

    engine = create_engine('sqlite://')
    base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    Map.query = session.query(Map)
    calls = dict(query=0, read=0)

    @event.listens_for(engine, 'before_cursor_execute')
    def count_query(conn, cursor, statement, parameters, context, executemany):
        if statement.lstrip().upper().startswith('SELECT'):
            calls['query'] += 1

    original_open = fits.open

    def open_fits(*args, **kwargs):
        calls['read'] += 1
        return original_open(*args, **kwargs)

    monkeypatch.setattr(fits, 'open', open_fits)
    clock = SimpleNamespace(now=1000.)
    clock.monotonic = lambda: clock.now
    source = Path(__file__).resolve().parents[2] / 'indi_allsky/processing.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ImageProcessor')
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {
        '_denoise_temporal_kwargs', '_denoise_bad_pixel_map', '_denoise_load_bad_pixel_map'}]
    namespace = dict(__name__='indi_allsky.processing', __package__='indi_allsky',
                     numpy=np, time=clock, math=math, Path=Path, timedelta=timedelta,
                     asi676mc=asi676mc, constants=constants, sa_true=sa_true,
                     IndiAllSkyDbBadPixelMapTable=Map, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    processor = namespace['ImageProcessor']()
    processor.config = {'IMAGE_DENOISE_BPM': True}
    processor.image = np.zeros((32, 32, 3), np.uint16)
    processor.night_av = [True] * 8
    raw = SimpleNamespace(data=np.zeros((32, 32), np.uint16), header={})
    ref = SimpleNamespace(hdulist=[raw], camera_id=1, image_bitpix=16, binning=1,
                          image_bayerpat='RGGB', gain=10, exposure=30., exp_elapsed=30.9,
                          exp_date_utc=datetime(2026, 10, 17, 20, tzinfo=timezone.utc))
    processor.image_list = [ref]
    serial = 0

    def add_map(data=None, header=None, **changes):
        nonlocal serial
        serial += 1
        data = np.zeros((32, 32), np.uint16) if data is None else data
        if not np.any(data):
            data = data.copy()
            data[5, 8] = 5000
        path = tmp_path / f'map-{serial}.fits'
        hdu = fits.PrimaryHDU(data)
        hdu.header['BAYERPAT'] = 'RGGB'
        hdu.header['XBINNING'] = hdu.header['YBINNING'] = 1
        for key, value in (header or {}).items():
            if value is None:
                hdu.header.pop(key, None)
            else:
                hdu.header[key] = value
        hdu.writeto(path)
        defaults = dict(camera_id=1, active=True, bitdepth=hdu.header['BITPIX'], binmode=1,
                        width=data.shape[-1], height=data.shape[-2], gain=300,
                        exposure=30, temp=20, createDate=datetime(2026, 10, 17), filename=str(path))
        defaults.update(changes)
        row = Map(**defaults)
        session.add(row)
        session.commit()
        return row

    yield SimpleNamespace(processor=processor, ref=ref, raw=raw, clock=clock, calls=calls,
                          add_map=add_map, session=session, Map=Map)
    session.close()
    engine.dispose()


def test_disabled_and_missing_capture_never_query_or_load(environment):
    env = environment
    env.add_map()
    env.processor.config['IMAGE_DENOISE_BPM'] = False
    assert 'bad_pixel_map' not in env.processor._denoise_temporal_kwargs(env.ref)
    env.processor.config['IMAGE_DENOISE_BPM'] = True
    env.ref.exp_elapsed = 0
    assert env.processor._denoise_temporal_kwargs(env.ref) == {}
    assert env.calls == dict(query=0, read=0)


def test_no_map_is_cached_and_new_active_map_discovered(environment):
    env = environment
    assert 'bad_pixel_map' not in env.processor._denoise_temporal_kwargs(env.ref)
    assert env.processor._denoise_bad_pixel_map(env.ref) is None
    assert env.calls == dict(query=1, read=0)
    env.add_map()
    env.clock.now += 301
    points = env.processor._denoise_temporal_kwargs(env.ref)['bad_pixel_map']
    np.testing.assert_array_equal(points.data, [[8, 5]])
    assert not points.data.flags.writeable


def test_map_and_spatial_index_are_cached_across_frames_gain_and_refresh(environment):
    env = environment
    env.add_map()
    first = env.processor._denoise_bad_pixel_map(env.ref)
    for gain in (0, 50, 200, 300):
        env.ref.gain = gain
        assert env.processor._denoise_bad_pixel_map(env.ref) is first
    assert env.calls == dict(query=1, read=1)
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is first
    assert env.calls == dict(query=2, read=1)


def test_only_matching_camera_bitdepth_bin_dimensions_and_active_entries(environment):
    env = environment
    env.add_map(gain=250)
    for changes in ({'camera_id': 2}, {'bitdepth': 8}, {'binmode': 2},
                    {'width': 31}, {'height': 31}, {'active': False}):
        env.add_map(gain=400, **changes)
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    assert env.calls == dict(query=1, read=1)


def test_selection_prefers_gain_then_exposure_temperature_and_newest(environment):
    env = environment
    env.add_map(gain=100, exposure=180, temp=50)
    env.add_map(gain=300, exposure=20, temp=50)
    env.add_map(gain=300, exposure=30, temp=10)
    env.add_map(gain=300, exposure=30, temp=20)
    data = np.zeros((32, 32), np.uint16)
    data[10, 11] = 1234
    env.add_map(data, gain=300, exposure=30, temp=20, createDate=datetime(2026, 10, 18))
    points = env.processor._denoise_bad_pixel_map(env.ref)
    np.testing.assert_array_equal(points.data, [[11, 10]])
    assert env.calls['read'] == 1


@pytest.mark.parametrize('header', [
    {'BAYERPAT': 'BGGR'}, {'BAYERPAT': None}, {'XORGSUBF': 2}, {'YORGSUBF': 1},
    {'XBAYROFF': 1}, {'YBAYROFF': 1}, {'XBINNING': 2}, {'YBINNING': 2}])
def test_incompatible_fits_metadata_falls_back_to_compatible_map(environment, header):
    env = environment
    env.add_map(gain=100)
    env.add_map(header=header, gain=300)
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    assert env.calls['read'] == 2
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    assert env.calls['read'] == 2  # Rejected file versions are cached too.


def test_actual_fits_dimensions_override_database_metadata(environment):
    env = environment
    env.add_map(np.ones((20, 20), np.uint16), width=32, height=32)
    assert env.processor._denoise_bad_pixel_map(env.ref) is None


def test_malformed_deleted_or_disabled_map_never_reuses_previous_coordinates(environment):
    env = environment
    row = env.add_map()
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    row.active = False
    env.session.commit()
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is None
    row.active = True
    env.session.commit()
    Path(row.filename).write_bytes(b'not a FITS image')
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is None
    Path(row.filename).unlink()
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is None


def test_changed_file_version_is_reloaded(environment):
    env = environment
    row = env.add_map()
    before = env.processor._denoise_bad_pixel_map(env.ref)
    replacement = np.zeros((32, 32), np.uint16)
    replacement[20, 21] = 3000
    fits.writeto(row.filename, replacement,
                 header=fits.Header({'BAYERPAT': 'RGGB'}), overwrite=True)
    env.clock.now += 301
    after = env.processor._denoise_bad_pixel_map(env.ref)
    assert after is not before
    np.testing.assert_array_equal(after.data, [[21, 20]])
    assert env.calls['read'] == 2


@pytest.mark.parametrize('change', ['camera', 'binning', 'origin', 'bayer', 'processed_shape'])
def test_capture_geometry_changes_immediately_invalidate_cached_map(environment, change):
    env = environment
    env.add_map()
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    if change == 'camera':
        env.ref.camera_id = 2
    elif change == 'binning':
        env.ref.binning = 2
    elif change == 'origin':
        env.raw.header['XORGSUBF'] = 2
    elif change == 'bayer':
        env.processor.config['CFA_PATTERN'] = 'BGGR'
    else:
        env.processor.image = np.zeros((30, 30, 3), np.uint16)
    assert env.processor._denoise_bad_pixel_map(env.ref) is None


def test_only_finite_positive_map_samples_are_candidates(environment):
    env = environment
    data = np.zeros((32, 32), np.float32)
    data[1, :4] = [np.nan, np.inf, -1, 100]
    env.ref.image_bitpix = -32
    env.raw.data = np.zeros((32, 32), np.float32)
    env.add_map(data)
    np.testing.assert_array_equal(env.processor._denoise_bad_pixel_map(env.ref).data, [[3, 1]])


@pytest.mark.parametrize('size', [32, 160])
def test_dense_map_is_rejected_instead_of_allocating_unbounded_coordinate_table(environment, size):
    env = environment
    env.raw.data = np.zeros((size, size), np.uint16)
    env.processor.image = np.zeros((size, size, 3), np.uint16)
    env.add_map(np.full((size, size), 50, np.uint16))
    assert env.processor._denoise_bad_pixel_map(env.ref) is None


def test_rgb_map_combines_channels_without_bayer_coordinate_shift(environment):
    env = environment
    data = np.zeros((3, 32, 32), np.uint16)
    data[0, 4, 7] = 50
    data[2, 10, 12] = 40
    env.raw.data = np.zeros_like(data)
    env.ref.image_bayerpat = None
    env.add_map(data, header={'BAYERPAT': None})
    np.testing.assert_array_equal(env.processor._denoise_bad_pixel_map(env.ref).data, [[7, 4], [12, 10]])


def test_database_error_drops_old_map_and_uses_bounded_retry(environment):
    env = environment
    env.add_map()
    assert env.processor._denoise_bad_pixel_map(env.ref) is not None
    env.Map.query = None
    env.clock.now += 301
    assert env.processor._denoise_bad_pixel_map(env.ref) is None
    assert env.processor._denoise_bad_pixel_map(env.ref) is None
