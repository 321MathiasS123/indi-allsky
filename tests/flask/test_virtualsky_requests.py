import ast
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import threading
import math

import flask
import pytest

from indi_allsky.lens_solver.request import parseSolverRequestValues, applySolvedValuesToConfig


VALUES = dict(AZIMUTH_ANGLE=200, POINTING_AZIMUTH=123, LATITUDE_OFFSET=43.49,
              LONGITUDE_OFFSET=0, IMAGE_CIRCLE_DIAMETER=2951, OFFSET_X=7, OFFSET_Y=-135)


@pytest.fixture
def endpoint():
    # Execute the real view with injected infrastructure, avoiding the camera,
    # system D-Bus and database services that the full application imports.
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/views.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AjaxLensSolverView')
    namespace = dict(BaseView=object, login_required=lambda f: f, threading=threading,
                     app=flask.current_app, current_user=SimpleNamespace(is_admin=True, username='test'),
                     request=flask.request, jsonify=flask.jsonify, datetime=datetime, timedelta=timedelta,
                     parseSolverRequestValues=parseSolverRequestValues,
                     applySolvedValuesToConfig=applySolvedValuesToConfig, ConfigSaveException=RuntimeError)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    view = namespace['AjaxLensSolverView']()
    view.indi_allsky_config = {'LENS_ALTITUDE': 54, 'VIRTUALSKY': {'MAGNITUDE': 6, 'POINTING_AZIMUTH': 88}}
    saved = []
    view._indi_allsky_config_obj = SimpleNamespace(save=lambda *args: saved.append(deepcopy(view.indi_allsky_config)))
    app = flask.Flask(__name__)
    app.config['LOGIN_DISABLED'] = False
    return app, view, namespace, saved


def test_manual_save_without_solve(endpoint):
    app, view, _, saved = endpoint
    with app.test_request_context(json=dict(VALUES, action='save')):
        assert view.dispatch_request().get_json()['success']
    assert saved[0]['VIRTUALSKY']['LATITUDE_OFFSET'] == 43.49
    assert saved[0]['VIRTUALSKY']['POINTING_AZIMUTH'] == 123
    assert saved[0]['VIRTUALSKY']['MAGNITUDE'] == 6
    assert saved[0]['LENS_ALTITUDE'] == 54


def test_old_save_payload_preserves_pointing(endpoint):
    app, view, _, saved = endpoint
    values = dict(VALUES, action='save')
    del values['POINTING_AZIMUTH']
    with app.test_request_context(json=values):
        assert view.dispatch_request().get_json()['success']
    assert saved[0]['VIRTUALSKY']['POINTING_AZIMUTH'] == 88
    assert saved[0]['LENS_ALTITUDE'] == 54


@pytest.mark.parametrize('altitude', [0, 54, 90])
def test_save_recovered_pointing(endpoint, altitude):
    app, view, _, saved = endpoint
    with app.test_request_context(json=dict(VALUES, action='save', LENS_ALTITUDE=altitude,
                                           POINTING_AZIMUTH=0, LATITUDE_OFFSET=0)):
        assert view.dispatch_request().get_json()['success']
    assert saved[0]['LENS_ALTITUDE'] == altitude
    assert saved[0]['VIRTUALSKY']['POINTING_AZIMUTH'] == 0
    assert saved[0]['VIRTUALSKY']['LATITUDE_OFFSET'] == 0


@pytest.mark.parametrize('altitude', [-1, 91, None, 'bad', float('nan'), float('inf')])
def test_invalid_altitude_never_saves(endpoint, altitude):
    app, view, _, saved = endpoint
    with app.test_request_context(json=dict(VALUES, action='save', LENS_ALTITUDE=altitude)):
        assert view.dispatch_request()[1] == 400
    assert saved == []


@pytest.mark.parametrize('login_disabled', [False, True])
def test_save_keeps_admin_authorization(endpoint, login_disabled):
    app, view, namespace, saved = endpoint
    app.config['LOGIN_DISABLED'] = login_disabled
    namespace['current_user'].is_admin = False
    with app.test_request_context(json=dict(VALUES, action='save')):
        response = view.dispatch_request()
    if login_disabled:
        assert response.get_json()['success']
        assert len(saved) == 1
    else:
        assert response[1] == 403
        assert saved == []


@pytest.mark.parametrize('heading', [-1, 361, 'bad', None])
def test_invalid_pointing_never_saves(endpoint, heading):
    app, view, _, saved = endpoint
    with app.test_request_context(json=dict(VALUES, action='save', POINTING_AZIMUTH=heading)):
        assert view.dispatch_request()[1] == 400
    assert saved == []


@pytest.mark.parametrize('explicit_heading', [None, 0, 123])
@pytest.mark.parametrize('explicit_altitude', [None, 0, 54])
def test_solve_uses_selected_camera_orientation(endpoint, tmp_path, explicit_heading, explicit_altitude):
    app, view, namespace, _ = endpoint
    camera = SimpleNamespace(id=7, alt=20, data={'vs_pointing_azimuth': 250})
    image_file = tmp_path / 'sky.png'
    image_file.touch()
    image = SimpleNamespace(getFilesystemPath=lambda: image_file)

    class Query:
        def __init__(self, row):
            self.row = row

        def filter(self, *args):
            return self

        def first(self):
            return self.row

    namespace['IndiAllSkyDbCameraTable'] = SimpleNamespace(id=7, query=Query(camera))
    namespace['IndiAllSkyDbImageTable'] = SimpleNamespace(
        camera_id=7, createDate=datetime.fromtimestamp(1770000000), query=Query(image))
    view.cameraSetup = lambda **kwargs: None
    view.getCameraPrivacyLatLong = lambda selected: (46.51, 8)
    view.camera_time_offset = 0
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return {'success': True}

    namespace['IndiAllSkyLensSolver'] = lambda config: SimpleNamespace(solve=solve)
    values = dict(VALUES, action='solve', camera_id=7, timestamp=1770000000, LATITUDE_OFFSET=0)
    if explicit_heading is None:
        del values['POINTING_AZIMUTH']
    else:
        values['POINTING_AZIMUTH'] = explicit_heading
    if explicit_altitude is not None:
        values['LENS_ALTITUDE'] = explicit_altitude
    with app.test_request_context(json=values):
        assert view.dispatch_request().get_json()['success']
    expected_heading = 250 if explicit_heading is None else explicit_heading
    expected_altitude = 20 if explicit_altitude is None else explicit_altitude
    assert calls[0][1] == {'lens_altitude': expected_altitude, 'pointing_azimuth': expected_heading}


@pytest.mark.parametrize('altitude,expected', [(None, 90), (90, 90), (0, 0), (54, 54)])
def test_page_uses_camera_altitude_with_legacy_fallback(altitude, expected):
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/views.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VirtualSkyView')

    class TemplateView:
        def get_context(self):
            return {}

    namespace = dict(TemplateView=TemplateView, math=math, datetime=datetime, request=flask.request,
                     IndiAllskyVirtualSkyHelperForm=lambda data: data)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    view = namespace['VirtualSkyView']()
    view.camera = SimpleNamespace(alt=altitude, az=200, data={}, utc_offset=0)
    view.indi_allsky_config = {}
    view.getCameraPrivacyLatLong = lambda camera: (46.51, 8)
    with flask.Flask(__name__).test_request_context():
        context = view.get_context()
    assert context['camera_altitude'] == expected
    assert context['form_virtualsky']['POINTING_AZIMUTH'] == 0
    assert context['form_virtualsky']['AZIMUTH_ANGLE'] == 200
