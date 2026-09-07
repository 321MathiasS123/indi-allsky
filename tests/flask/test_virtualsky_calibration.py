import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from flask_sqlalchemy import SQLAlchemy

from indi_allsky.lens_solver.calibration import displacement
from tests.flask.test_virtualsky import run_node
from tests.lens_solver.test_calibration import saved_model
from tests.flask.test_virtualsky_requests import endpoint, VALUES


@pytest.fixture(scope='module')
def camera_models():
    # Load the real schema with an isolated ORM registry. Importing the Flask
    # package normally also starts imports of camera/D-Bus infrastructure.
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/models.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None)]
    namespace = {'__name__': 'virtualsky_test_models', 'db': SQLAlchemy()}
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['IndiAllSkyDbCameraTable'], namespace['IndiAllSkyDbImageTable']


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('binmode', [None, 1, 2, 4])
@pytest.mark.parametrize('dimensions', [(3000, 2000), (None, None)])
def test_solve_endpoint_uses_real_image_binning(endpoint, camera_models, tmp_path,
                                              enabled, binmode, dimensions):
    app, view, namespace, saved = endpoint
    Camera, Image = camera_models
    camera = Camera(id=7, uuid='test-camera', alt=54, width=dimensions[0], height=dimensions[1],
                    data={'vs_pointing_azimuth': 123})
    image_file = tmp_path / 'sky.png'
    image_file.touch()
    image = Image(camera_id=7, binmode=binmode, filename=str(image_file))
    assert not hasattr(image, 'binning')  # the capture database calls this binmode
    image.getFilesystemPath = lambda: image_file

    class Query:
        def __init__(self, row):
            self.row = row

        def filter(self, *args):
            return self

        def first(self):
            return self.row

    namespace['IndiAllSkyDbCameraTable'] = SimpleNamespace(id=Camera.id, query=Query(camera))
    namespace['IndiAllSkyDbImageTable'] = SimpleNamespace(
        camera_id=Image.camera_id, createDate=Image.createDate, query=Query(image))
    view.cameraSetup = lambda **kwargs: None
    view.getCameraPrivacyLatLong = lambda selected: (53, 11)
    view.camera_time_offset = 30
    calls = []

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return {'success': True, 'calibration': saved_model() if enabled else None}

    namespace['IndiAllSkyLensSolver'] = lambda config: SimpleNamespace(solve=solve)
    with app.test_request_context(json=dict(VALUES, action='solve', camera_id=7,
            timestamp=1770000000, LATITUDE_OFFSET=0, CALIBRATION_ENABLED=enabled)):
        response = app.make_response(view.dispatch_request())
    assert response.status_code == 200, response.get_json()
    result = response.get_json()
    assert result['success']
    assert calls[0][0][3] == 1770000000-30
    hints = {'lens_altitude': 54, 'pointing_azimuth': 123}
    if enabled and dimensions[0]:
        binning = binmode or 1
        hints.update(binning=binning, sensor_shape=(dimensions[1]//binning, dimensions[0]//binning))
    assert calls[0][1] == hints
    if enabled:
        assert result['calibration']['camera_uuid'] == 'test-camera'
        assert result['calibration']['context'][2] == 30
    assert saved == []


def test_calibration_javascript():
    run_node('--test', 'tests/flask/virtualsky_calibration.test.cjs')


def test_python_and_browser_apply_identical_correction():
    model = saved_model()
    xy = np.random.default_rng(4).uniform(-1.4, 1.4, (1000, 2))
    actual = json.loads(run_node('-e', '''
const api = require('./indi_allsky/flask/static/js/virtualsky-calibration.js');
const input = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
console.log(JSON.stringify(input.xy.map(([u,v]) => api.delta(input.model,u,v))));
''', input=json.dumps(dict(model=model, xy=xy.tolist()))))
    np.testing.assert_allclose(actual, displacement(xy, model), atol=1e-14, rtol=0)


def test_endpoint_saves_validated_calibration_and_disabled_preference(endpoint):
    app, view, _, saved = endpoint
    model = saved_model()
    for enabled in (True, False):
        with app.test_request_context(json=dict(VALUES, action='save', LENS_ALTITUDE=90,
                CALIBRATION_ENABLED=enabled, CALIBRATION=model)):
            assert view.dispatch_request().get_json()['success']
        assert saved[-1]['VIRTUALSKY']['CALIBRATION_ENABLED'] is enabled
        assert saved[-1]['VIRTUALSKY']['CALIBRATION'] == model
    with app.test_request_context(json=dict(VALUES, action='save', LENS_ALTITUDE=90,
            CALIBRATION_ENABLED=True, CALIBRATION={'version': 1})):
        assert view.dispatch_request()[1] == 400
    assert len(saved) == 2
