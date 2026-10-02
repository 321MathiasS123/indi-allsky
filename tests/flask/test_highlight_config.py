"""Exercise the real fields and validators without Linux D-Bus services."""
import ast
import math
from pathlib import Path

from flask import Flask
from flask_wtf import FlaskForm
from jinja2 import Environment
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import BooleanField, FloatField
from wtforms.validators import NumberRange, ValidationError
from wtforms.widgets import NumberInput


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky/flask'


@pytest.fixture
def form_class():
    tree = ast.parse((ROOT / 'forms.py').read_text(encoding='utf-8'))
    validators = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('HIGHLIGHT_')]
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndiAllskyConfigForm')
    fields = [n for n in original.body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('HIGHLIGHT_PROTECTION__')]
    cls = ast.ClassDef(name='HighlightForm', bases=[ast.Name(id='FlaskForm', ctx=ast.Load())],
                       keywords=[], body=fields, decorator_list=[])
    namespace = dict(math=math, FlaskForm=FlaskForm, BooleanField=BooleanField, FloatField=FloatField,
                     NumberRange=NumberRange, NumberInput=NumberInput, ValidationError=ValidationError)
    exec(compile(ast.fix_missing_locations(ast.Module(body=validators + [cls], type_ignores=[])),
                 str(ROOT / 'forms.py'), 'exec'), namespace)
    return namespace['HighlightForm']


@pytest.mark.parametrize('field,value', [
    ('FULL_TARGET', 'nan'), ('ANY_TARGET', 'inf'), ('FULL_DEV', '-0.1'),
    ('FULL_DEV', '0.8'), ('ANY_DEV', '2.1'), ('THRESHOLD', '89'),
    ('THRESHOLD', '101'), ('MAX_BOOST', '4.1'), ('MAX_BOOST', '-1'),
    ('MAX_BOOST', ''), ('FULL_TARGET', '0'),
    ('GAMMA', '-0.1'), ('GAMMA_DAY', '-1'), ('GAMMA', 'nan'),
    ('GAMMA_DAY', 'inf'), ('GAMMA_DAY', ''),
])
def test_invalid_settings_rejected(form_class, field, value):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(formdata=MultiDict({'HIGHLIGHT_PROTECTION__' + field: value}))
        assert not form.validate()
        assert form['HIGHLIGHT_PROTECTION__' + field].errors


@pytest.mark.parametrize('field', ['GAMMA', 'GAMMA_DAY'])
@pytest.mark.parametrize('value', ['0', '1', '1.85'])
def test_highlight_gamma_accepts_inherit_unity_and_override(form_class, field, value):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(formdata=MultiDict({'HIGHLIGHT_PROTECTION__' + field: value}))
        assert form.validate(), form.errors


def test_defaults_off_controls_render_and_use_existing_save_registry(form_class):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class()
        assert form.validate(), form.errors
        assert not form.HIGHLIGHT_PROTECTION__ENABLE.data
        assert form.HIGHLIGHT_PROTECTION__GAMMA.data == 0
        assert form.HIGHLIGHT_PROTECTION__GAMMA_DAY.data == 0
        template = (ROOT / 'templates/config/image.html').read_text(encoding='utf-8')
        start = template.index('                <div class="tw:flex tw:items-center tw:justify-between tw:gap-4">')
        end = template.index('                <!-- ADU FOV Div & ROI Coordinates -->', start)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form)
        registry = (ROOT / 'templates/config.html').read_text(encoding='utf-8')
        for field in form:
            assert f'id="{field.id}"' in html
            assert f'id="{field.id}-error"' in html
            assert registry.count("'" + field.id + "'") == 1


def test_settings_round_trip_through_real_config_view_assignments(form_class):
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    # Run the exact new load/save expressions, preserving surrounding settings.
    load_dict = next(n for n in ast.walk(tree) if isinstance(n, ast.Dict)
                     and any(isinstance(k, ast.Constant) and k.value == 'HIGHLIGHT_PROTECTION__ENABLE' for k in n.keys))
    selected = [(k, v) for k, v in zip(load_dict.keys, load_dict.values)
                if isinstance(k, ast.Constant) and str(k.value).startswith('HIGHLIGHT_PROTECTION__')]
    load = compile(ast.fix_missing_locations(ast.Expression(body=ast.Dict(keys=[k for k, v in selected], values=[v for k, v in selected]))),
                   'load-highlight-fields', 'eval')
    save = next(n for n in ast.walk(tree) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'update'
                and 'HIGHLIGHT_PROTECTION' in ast.unparse(n))
    from types import SimpleNamespace
    config = {'TARGET_ADU': 70, 'IMAGE_STRETCH': {'MODE2_MIDTONES': 0.4},
              'GAMMA_CORRECTION': 0.87, 'GAMMA_CORRECTION_DAY': 1.565}
    view = SimpleNamespace(indi_allsky_config=config)
    payload = eval(load, {'self': view})
    payload.update(HIGHLIGHT_PROTECTION__ENABLE=True, HIGHLIGHT_PROTECTION__FULL_TARGET=0.9,
                   HIGHLIGHT_PROTECTION__GAMMA=0.95, HIGHLIGHT_PROTECTION__GAMMA_DAY=1.85)
    exec(compile(ast.Module(body=[save], type_ignores=[]), 'save-highlight-fields', 'exec'),
         {'self': view, 'request': SimpleNamespace(json=payload)})
    assert config['TARGET_ADU'] == 70
    assert config['GAMMA_CORRECTION'] == 0.87
    assert config['GAMMA_CORRECTION_DAY'] == 1.565
    assert config['IMAGE_STRETCH'] == {'MODE2_MIDTONES': 0.4}
    assert eval(load, {'self': view}) == payload
    # A configuration tab opened before the upgrade does not supply new fields.
    # Saving it must neither fail nor switch off an already-enabled feature.
    exec(compile(ast.Module(body=[save], type_ignores=[]), 'save-old-form', 'exec'),
         {'self': view, 'request': SimpleNamespace(json={})})
    assert eval(load, {'self': view}) == payload
