"""Exercise the production form/load/save expressions without Linux D-Bus."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from flask_wtf import FlaskForm
from jinja2 import Environment
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import BooleanField, FloatField
from wtforms.validators import ValidationError
from wtforms.widgets import NumberInput


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky/flask'


@pytest.fixture
def form_class():
    tree = ast.parse((ROOT / 'forms.py').read_text(encoding='utf-8'))
    validators = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name in ('NIGHT_SUN_ALT_DEG_validator', 'TWILIGHT_NIGHT_ALT_validator')]
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
])
def test_form_rejects_invalid_transition_interval(form_class, start, end, enabled, valid):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        fields = {'NIGHT_SUN_ALT_DEG': start, 'TWILIGHT_TRANSITION__NIGHT_ALT': end}
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
        assert form.TWILIGHT_TRANSITION__NIGHT_ALT.data == -12
        template = (ROOT / 'templates/config/location.html').read_text(encoding='utf-8')
        start = template.index('                <div class="tw:flex tw:items-center tw:justify-between tw:gap-4">')
        end = template.index('                <div class="tw:grid tw:grid-cols-1 md:tw:grid-cols-3', start)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form, twilight_forecast=None)
        registry = (ROOT / 'templates/config.html').read_text(encoding='utf-8')
        for name in ('TWILIGHT_TRANSITION__ENABLE', 'TWILIGHT_TRANSITION__NIGHT_ALT'):
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
    payload = eval(load, {'self': view})
    assert payload == {'TWILIGHT_TRANSITION__ENABLE': False, 'TWILIGHT_TRANSITION__NIGHT_ALT': -12}
    payload.update(TWILIGHT_TRANSITION__ENABLE=True, TWILIGHT_TRANSITION__NIGHT_ALT=-15)
    exec(save_code, {'self': view, 'request': SimpleNamespace(json=payload)})
    assert config == {'NIGHT_SUN_ALT_DEG': -6, 'TARGET_ADU': 75,
                      'TWILIGHT_TRANSITION': {'ENABLE': True, 'NIGHT_ALT': -15}}
    exec(save_code, {'self': view, 'request': SimpleNamespace(json={})})
    assert config['TWILIGHT_TRANSITION'] == {'ENABLE': True, 'NIGHT_ALT': -15}


def test_older_clients_validate_the_retained_transition_interval(form_class):
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AjaxConfigView')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch_request')
    setup = compile(ast.Module(body=method.body[:3], type_ignores=[]), 'ajax-form-setup', 'exec')
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        namespace = dict(self=SimpleNamespace(indi_allsky_config={
            'TWILIGHT_TRANSITION': {'ENABLE': True, 'NIGHT_ALT': -15}}),
            request=SimpleNamespace(json={'NIGHT_SUN_ALT_DEG': -20}), IndiAllskyConfigForm=form_class)
        exec(setup, namespace)
        assert not namespace['form_config'].validate()
        assert namespace['form_config'].TWILIGHT_TRANSITION__ENABLE.data
        assert namespace['form_config'].TWILIGHT_TRANSITION__NIGHT_ALT.data == -15


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
        end = template.index('                <div class="tw:grid tw:grid-cols-1 md:tw:grid-cols-3', start)
        forecast = dict(minimum=.99, maximum=.99999, lowest_altitude=-11.99, reversal_utc=None)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form, twilight_forecast=forecast)
        assert 'partial transition' in html
        assert 'over 99.9%' in html
        assert '100.0%' not in html and 'UTC' not in html
