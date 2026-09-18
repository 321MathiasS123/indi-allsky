"""Exercise map config persistence without camera or database services."""

import ast
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from wtforms import Form, FloatField, StringField
from wtforms.validators import ValidationError

from indi_allsky.allsky_map import send_allsky_map_ping


@pytest.fixture(scope='module')
def coordinate_form():
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/forms.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'LOCATION_LATITUDE', 'LOCATION_LONGITUDE',
             'ALLSKYMAP__MAP_LATITUDE', 'ALLSKYMAP__MAP_LONGITUDE'}
    validator_names = {name + '_validator' for name in names}
    validators = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name in validator_names]
    form = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == 'IndiAllskyConfigForm')
    fields = [node for node in form.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id in names
                      for target in node.targets)]
    namespace = dict(FloatField=FloatField, StringField=StringField, ValidationError=ValidationError)
    exec(compile(ast.Module(body=validators + fields, type_ignores=[]), str(path), 'exec'), namespace)
    return type('CoordinateForm', (Form,), {name: namespace[name] for name in names})


@pytest.mark.parametrize('axis,camera_value', [
    ('LATITUDE', 52), ('LONGITUDE', 11), ('LATITUDE', 0), ('LONGITUDE', 0),
])
@pytest.mark.parametrize('offset', [-1.0001, -1, 0, 1, 1.0001])
def test_override_must_be_within_one_degree(coordinate_form, axis, camera_value, offset):
    data = dict(LOCATION_LATITUDE=52, LOCATION_LONGITUDE=11,
                ALLSKYMAP__MAP_LATITUDE='', ALLSKYMAP__MAP_LONGITUDE='')
    field_name = 'ALLSKYMAP__MAP_' + axis
    data['LOCATION_' + axis] = camera_value
    data[field_name] = str(camera_value + offset)
    form = coordinate_form(data=data)
    assert form.validate() is (abs(offset) <= 1)
    if abs(offset) > 1:
        assert 'within 1 degree' in form.errors[field_name][0]


@pytest.mark.parametrize('camera_longitude,map_longitude,valid', [
    (179.5, -179.5, True), (179.5, -179.4, False),
    (-179.5, 179.5, True), (-179.5, 179.4, False),
])
def test_longitude_distance_wraps_at_dateline(coordinate_form, camera_longitude, map_longitude, valid):
    form = coordinate_form(data=dict(
        LOCATION_LATITUDE=52, LOCATION_LONGITUDE=camera_longitude,
        ALLSKYMAP__MAP_LATITUDE='', ALLSKYMAP__MAP_LONGITUDE=str(map_longitude),
    ))
    assert form.validate() is valid
    if not valid:
        assert 'within 1 degree' in form.errors['ALLSKYMAP__MAP_LONGITUDE'][0]


def test_blank_overrides_remain_valid(coordinate_form):
    form = coordinate_form(data=dict(
        LOCATION_LATITUDE=52, LOCATION_LONGITUDE=11,
        ALLSKYMAP__MAP_LATITUDE='', ALLSKYMAP__MAP_LONGITUDE='',
    ))
    assert form.validate()


@pytest.fixture(scope='module')
def map_config_code():
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/views.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    config_view = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'ConfigView')
    form_data = next(node.value for node in ast.walk(config_view)
                     if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'form_data'
                             for target in node.targets))
    entries = [(key, value) for key, value in zip(form_data.keys, form_data.values)
               if isinstance(key, ast.Constant) and key.value.startswith('ALLSKYMAP__')]
    load = ast.Expression(ast.Dict(keys=[key for key, _ in entries],
                                   values=[value for _, value in entries]))
    save_view = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'AjaxConfigView')
    assignments = [node for node in ast.walk(save_view)
                   if isinstance(node, ast.Assign)
                   and any(ast.unparse(target).startswith("self.indi_allsky_config['ALLSKYMAP']")
                           for target in node.targets)]
    save = ast.Module(body=assignments, type_ignores=[])
    return (compile(save, str(path), 'exec'),
            compile(ast.fix_missing_locations(load), str(path), 'eval'))


@pytest.mark.parametrize('latitude,longitude,expected', [
    ('50.12678', '8.07654', (50.12678, 8.07654)),
    ('0', '0', (0.0, 0.0)),
    ('', '', (50.12, 8.12)),
    ('', '8.07654', (50.12, 8.07654)),
    ('50.12678', '', (50.12678, 8.12)),
])
def test_coordinates_survive_save_reload_and_ping(map_config_code, latitude, longitude, expected):
    save, load = map_config_code
    config = {
        'LOCATION_LATITUDE': 50.1234567,
        'LOCATION_LONGITUDE': 8.1234567,
        'ALLSKYMAP': {
            'ENABLE': True, 'API_URL': 'https://map.invalid', 'API_KEY': 'test-key',
            'CAMERA_NAME': 'Test', 'CAMERA_OWNER': '', 'WEBSITE_URL': '',
            'UPLOAD_IMAGE': False, 'INTERVAL': 10,
            'MAP_LATITUDE': '50.2', 'MAP_LONGITUDE': '8.2',
        },
    }
    submitted = {'ALLSKYMAP__' + key: value for key, value in config['ALLSKYMAP'].items()}
    submitted.update(ALLSKYMAP__MAP_LATITUDE=latitude, ALLSKYMAP__MAP_LONGITUDE=longitude)
    context = {'self': SimpleNamespace(indi_allsky_config=config),
               'request': SimpleNamespace(json=submitted)}
    exec(save, context)
    assert config['ALLSKYMAP']['MAP_LATITUDE'] == latitude
    assert config['ALLSKYMAP']['MAP_LONGITUDE'] == longitude
    assert config['LOCATION_LATITUDE'] == 50.1234567
    assert config['LOCATION_LONGITUDE'] == 8.1234567

    restored = json.loads(json.dumps(config))
    form_data = eval(load, {'self': SimpleNamespace(indi_allsky_config=restored)})
    assert form_data['ALLSKYMAP__MAP_LATITUDE'] == latitude
    assert form_data['ALLSKYMAP__MAP_LONGITUDE'] == longitude

    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"ok": true}'
    with patch('urllib.request.urlopen', return_value=response) as send:
        assert send_allsky_map_ping(restored, None)[0]
    payload = json.loads(send.call_args.args[0].data)
    assert (payload['lat'], payload['lng']) == expected


@pytest.mark.parametrize('config', [{}, {'ALLSKYMAP': {}}])
def test_older_config_loads_blank_overrides(map_config_code, config):
    _, load = map_config_code
    form_data = eval(load, {'self': SimpleNamespace(indi_allsky_config=config)})
    assert form_data['ALLSKYMAP__MAP_LATITUDE'] == ''
    assert form_data['ALLSKYMAP__MAP_LONGITUDE'] == ''


def test_browser_submits_map_coordinates():
    node = shutil.which('node')
    assert node is not None, 'Node.js is required to run the map config tests'
    result = subprocess.run(
        [node, '--test', str(Path(__file__).with_name('allsky_map_config.test.cjs'))],
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
