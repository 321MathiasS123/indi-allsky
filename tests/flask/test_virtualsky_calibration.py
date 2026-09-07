import json

import numpy as np

from indi_allsky.lens_solver.calibration import displacement
from tests.flask.test_virtualsky import run_node
from tests.lens_solver.test_calibration import saved_model
from tests.flask.test_virtualsky_requests import endpoint, VALUES


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
