"""Exercise the production form/load/save expressions without Linux D-Bus."""
import ast
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from flask import Flask
from flask_wtf import FlaskForm
from jinja2 import Environment
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import BooleanField, FloatField
from wtforms.validators import ValidationError
from wtforms.widgets import NumberInput

from indi_allsky.twilight import day_altitude
from indi_allsky.exceptions import ConfigSaveException


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky/flask'


@pytest.fixture
def form_class():
    tree = ast.parse((ROOT / 'forms.py').read_text(encoding='utf-8'))
    validators = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name in ('NIGHT_SUN_ALT_DEG_validator', 'TWILIGHT_DAY_ALT_validator', 'TWILIGHT_NIGHT_ALT_validator')]
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndiAllskyConfigForm')
    fields = [n for n in original.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and (n.targets[0].id.startswith('TWILIGHT_TRANSITION__') or n.targets[0].id == 'NIGHT_SUN_ALT_DEG')]
    cls = ast.ClassDef(name='TwilightForm', bases=[ast.Name(id='FlaskForm', ctx=ast.Load())],
                       keywords=[], body=fields, decorator_list=[])
    namespace = dict(math=math, FlaskForm=FlaskForm, BooleanField=BooleanField, FloatField=FloatField,
                     NumberInput=NumberInput, ValidationError=ValidationError)
    exec(compile(ast.fix_missing_locations(ast.Module(body=validators + [cls], type_ignores=[])),
                 str(ROOT / 'forms.py'), 'exec'), namespace)
    return namespace['TwilightForm']


@pytest.mark.parametrize('start,end,enabled,valid', [
    ('-6', '-12', True, True), ('-6', '-6', True, False), ('-6', '-5', True, False),
    ('-20', '-12', False, True), ('-20', '-25', True, True),
    ('nan', '-12', True, False), ('-6', 'nan', True, False),
    ('-6', '-inf', True, False), ('-6', '-91', True, False), ('-6', '', True, False),
    ('0', '-12', True, True), ('-3', '-9', True, True), ('4', '-18', True, True),
    ('90', '-90', True, True), ('91', '-12', True, False), ('-91', '-12', False, False),
    ('inf', '-12', True, False), ('', '-12', True, False), ('0', '0', True, False),
])
def test_form_rejects_invalid_transition_interval(form_class, start, end, enabled, valid):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        fields = {'NIGHT_SUN_ALT_DEG': '-6', 'TWILIGHT_TRANSITION__DAY_ALT': start,
                  'TWILIGHT_TRANSITION__NIGHT_ALT': end}
        if enabled:
            fields['TWILIGHT_TRANSITION__ENABLE'] = 'y'
        form = form_class(formdata=MultiDict(fields))
        assert form.validate() == valid, form.errors


def test_defaults_and_controls_use_existing_save_registry(form_class):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(data={'NIGHT_SUN_ALT_DEG': -6})
        assert form.validate(), form.errors
        assert not form.TWILIGHT_TRANSITION__ENABLE.data
        assert form.TWILIGHT_TRANSITION__DAY_ALT.data == -6
        assert form.TWILIGHT_TRANSITION__NIGHT_ALT.data == -12
        template = (ROOT / 'templates/config/location.html').read_text(encoding='utf-8')
        start = template.index('        <!-- Smooth Day/Night Transition Card -->')
        end = template.index('        <!-- VirtualSky Configuration Card -->', start)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form, twilight_forecast=None)
        assert 'Smooth Day/Night Transition</span>' in html
        assert 'id="NIGHT_SUN_ALT_DEG"' not in html
        assert 'TWILIGHT_TRANSITION__' not in template[:start]
        registry = (ROOT / 'templates/config.html').read_text(encoding='utf-8')
        for name in ('TWILIGHT_TRANSITION__ENABLE', 'TWILIGHT_TRANSITION__DAY_ALT', 'TWILIGHT_TRANSITION__NIGHT_ALT'):
            assert f'id="{name}"' in html
            assert f'id="{name}-error"' in html
            assert registry.count("'" + name + "'") == 1


def test_config_round_trip_and_older_clients_preserve_existing_values():
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    load_dict = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                     and any(isinstance(k, ast.Constant) and k.value == 'TWILIGHT_TRANSITION__ENABLE' for k in n.keys))
    selected = [(k, v) for k, v in zip(load_dict.keys, load_dict.values)
                if isinstance(k, ast.Constant) and str(k.value).startswith('TWILIGHT_TRANSITION__')]
    load = compile(ast.fix_missing_locations(ast.Expression(body=ast.Dict(keys=[k for k, v in selected],
                    values=[v for k, v in selected]))), 'load-twilight-fields', 'eval')
    save = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'update'
                and 'TWILIGHT_TRANSITION' in ast.unparse(n))
    save_code = compile(ast.fix_missing_locations(ast.Module(body=[save], type_ignores=[])), 'save-twilight-fields', 'exec')
    config = {'NIGHT_SUN_ALT_DEG': -6, 'TARGET_ADU': 75}
    view = SimpleNamespace(indi_allsky_config=config)
    payload = eval(load, {'self': view, 'day_altitude': day_altitude})
    assert payload == {'TWILIGHT_TRANSITION__ENABLE': False, 'TWILIGHT_TRANSITION__DAY_ALT': -6,
                       'TWILIGHT_TRANSITION__NIGHT_ALT': -12}
    config['NIGHT_SUN_ALT_DEG'] = -4
    config['TWILIGHT_TRANSITION'] = {'DAY_ALT': None}
    assert eval(load, {'self': view, 'day_altitude': day_altitude})['TWILIGHT_TRANSITION__DAY_ALT'] == -4
    payload.update(TWILIGHT_TRANSITION__ENABLE=True, TWILIGHT_TRANSITION__DAY_ALT=0, TWILIGHT_TRANSITION__NIGHT_ALT=-15)
    exec(save_code, {'self': view, 'request': SimpleNamespace(json=payload)})
    assert config == {'NIGHT_SUN_ALT_DEG': -4, 'TARGET_ADU': 75,
                      'TWILIGHT_TRANSITION': {'ENABLE': True, 'DAY_ALT': 0, 'NIGHT_ALT': -15}}
    assert eval(load, {'self': view, 'day_altitude': day_altitude})['TWILIGHT_TRANSITION__DAY_ALT'] == 0
    exec(save_code, {'self': view, 'request': SimpleNamespace(json={})})
    assert config['TWILIGHT_TRANSITION'] == {'ENABLE': True, 'DAY_ALT': 0, 'NIGHT_ALT': -15}


@pytest.mark.parametrize('saved_day,valid', [(None, False), (0, True), (-3, True)])
def test_older_clients_validate_the_retained_transition_interval(form_class, saved_day, valid):
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AjaxConfigView')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch_request')
    setup = compile(ast.Module(body=method.body[:3], type_ignores=[]), 'ajax-form-setup', 'exec')
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        namespace = dict(self=SimpleNamespace(indi_allsky_config={
            'TWILIGHT_TRANSITION': {'ENABLE': True, 'DAY_ALT': saved_day, 'NIGHT_ALT': -15}}),
            request=SimpleNamespace(json={'NIGHT_SUN_ALT_DEG': -20}), IndiAllskyConfigForm=form_class)
        exec(setup, namespace)
        assert namespace['form_config'].validate() == valid
        assert namespace['form_config'].TWILIGHT_TRANSITION__ENABLE.data
        assert namespace['form_config'].TWILIGHT_TRANSITION__NIGHT_ALT.data == -15


@pytest.fixture
def config_store():
    path = ROOT.parent / 'config.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in ('IndiAllSkyConfigBase', 'IndiAllSkyConfig')]
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    scope = dict(OrderedDict=OrderedDict, Path=Path, app=app, datetime=datetime,
                 timezone=timezone, ConfigSaveException=ConfigSaveException,
                 IndiAllSkyDbUserTable=MagicMock())
    exec(compile(tree, str(path), 'exec'), scope)
    cls = scope['IndiAllSkyConfig']
    saved = [deepcopy(cls._base_config)]
    # Exercise real validation, password handling and reload. Replace only the
    # database boundary, retaining its JSON serialization of configuration data.
    cls._getConfigEntry = lambda self: SimpleNamespace(
        data=deepcopy(saved[-1]), id=len(saved), level='test', createDate=datetime.now())

    def store(self, config, user, note, encrypted):
        saved.append(json.loads(json.dumps(config)))
        return SimpleNamespace(id=len(saved))

    cls._setConfigEntry = store
    return app, cls, saved


@pytest.mark.parametrize('entered', ['-3', '-3.0', '0', '0.0'])
def test_day_endpoint_survives_form_save_validation_and_reload(form_class, config_store, entered):
    app, cls, saved = config_store
    obj = cls()
    with app.test_request_context():
        form = form_class(formdata=MultiDict({
            'NIGHT_SUN_ALT_DEG': '-6', 'TWILIGHT_TRANSITION__ENABLE': 'y',
            'TWILIGHT_TRANSITION__DAY_ALT': entered, 'TWILIGHT_TRANSITION__NIGHT_ALT': '-12'}))
        assert form.validate(), form.errors
        payload = {name: field.data for name, field in form._fields.items()}
        tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
        save = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'update'
                    and 'TWILIGHT_TRANSITION' in ast.unparse(n))
        exec(compile(ast.Module(body=[save], type_ignores=[]), 'save-twilight-fields', 'exec'),
             dict(self=SimpleNamespace(indi_allsky_config=obj.config), request=SimpleNamespace(json=payload)))
        obj.save('test', 'twilight endpoint')
        reloaded = cls()
        assert reloaded.config['TWILIGHT_TRANSITION'] == {
            'ENABLE': True, 'DAY_ALT': float(entered), 'NIGHT_ALT': -12.0}
        assert day_altitude(reloaded.config) == float(entered)
        assert reloaded.config['NIGHT_SUN_ALT_DEG'] == -6
        assert len(saved) == 2


@pytest.mark.parametrize('value', [None, -3, -3.0, 0, 0.0])
def test_imported_or_inherited_day_endpoint_survives_config_save(config_store, value):
    _, cls, _ = config_store
    obj = cls()
    obj.config['NIGHT_SUN_ALT_DEG'] = -4.0
    obj.config['TWILIGHT_TRANSITION']['DAY_ALT'] = value
    obj.save('test', 'existing configuration')
    assert day_altitude(cls().config) == (-4.0 if value is None else value)


@pytest.mark.parametrize('key,value', [('DAY_ALT', '-3'), ('DAY_ALT', []), ('DAY_ALT', {}),
                                     ('NIGHT_ALT', None), ('NIGHT_ALT', '-12')])
def test_endpoint_save_type_exception_stays_local(config_store, key, value):
    _, cls, saved = config_store
    obj = cls()
    obj.config['TWILIGHT_TRANSITION'][key] = value
    with pytest.raises(ConfigSaveException, match='wrong type'):
        obj.save('test', 'invalid type')
    assert len(saved) == 1


@pytest.mark.parametrize('class_name', ['Fits2JpegView', 'JsonImageProcessingView'])
def test_manual_fits_views_do_not_blend_away_selected_settings(class_name):
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch_request')
    start = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'p_config')
    assignments = [method.body[start]]
    if 'TWILIGHT_TRANSITION' in ast.unparse(method.body[start + 1]):
        assignments.append(method.body[start + 1])
    setup = compile(ast.Module(body=assignments, type_ignores=[]), 'fits-preview-config', 'exec')
    saved = {'TWILIGHT_TRANSITION': {'ENABLE': True, 'NIGHT_ALT': -12}}
    namespace = {'self': SimpleNamespace(indi_allsky_config=saved)}
    exec(setup, namespace)
    assert not namespace['p_config']['TWILIGHT_TRANSITION']['ENABLE']
    assert saved['TWILIGHT_TRANSITION']['ENABLE']


def test_near_complete_polar_forecast_is_still_labeled_partial(form_class):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(data={'NIGHT_SUN_ALT_DEG': -6})
        template = (ROOT / 'templates/config/location.html').read_text(encoding='utf-8')
        start = template.index('                {% if twilight_forecast %}')
        end = template.index('        <!-- VirtualSky Configuration Card -->', start)
        forecast = dict(minimum=.99, maximum=.99999, lowest_altitude=-11.99, reversal_utc=None)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form, twilight_forecast=forecast)
        assert 'partial transition' in html
        assert 'over 99.9%' in html
        assert '100.0%' not in html and 'UTC' not in html
