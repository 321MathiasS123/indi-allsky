"""Night Moon exclusion keeps its setting across real configuration load/save."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from flask_wtf import FlaskForm
from jinja2 import Environment
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import BooleanField


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky'
KEY = 'NIGHT_MOON_EXCLUSION_ENABLE'
FIELD = 'HIGHLIGHT_PROTECTION__' + KEY


@pytest.fixture
def config_expressions():
    tree = ast.parse((ROOT / 'flask/views.py').read_text(encoding='utf-8'))
    load_dict = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                     and any(isinstance(key, ast.Constant) and key.value == FIELD
                             for key in node.keys))
    value = next(value for key, value in zip(load_dict.keys, load_dict.values)
                 if isinstance(key, ast.Constant) and key.value == FIELD)
    load = compile(ast.Expression(body=value), 'load-night-moon-exclusion', 'eval')
    save = next(node for node in ast.walk(tree)
                if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'update'
                and 'HIGHLIGHT_PROTECTION' in ast.unparse(node))
    save = compile(ast.Module(body=[save], type_ignores=[]),
                   'save-night-moon-exclusion', 'exec')
    return load, save


@pytest.fixture
def form_class():
    tree = ast.parse((ROOT / 'flask/forms.py').read_text(encoding='utf-8'))
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'IndiAllskyConfigForm')
    fields = [node for node in cls.body if isinstance(node, ast.Assign)
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in (FIELD, 'HIGHLIGHT_PROTECTION__ENABLE')]
    cls = ast.ClassDef(name='MoonForm', bases=[ast.Name(id='FlaskForm', ctx=ast.Load())],
                       keywords=[], body=fields, decorator_list=[])
    namespace = dict(FlaskForm=FlaskForm, BooleanField=BooleanField)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 'night-moon-fields', 'exec'), namespace)
    return namespace['MoonForm']


def test_night_moon_defaults_enabled_inside_disabled_highlight_parent(config_expressions):
    tree = ast.parse((ROOT / 'config.py').read_text(encoding='utf-8'))
    defaults = next(value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and key.value == 'HIGHLIGHT_PROTECTION')
    values = ast.literal_eval(defaults)
    assert values['ENABLE'] is False
    assert values[KEY] is True
    load, _ = config_expressions
    assert eval(load, {'self': SimpleNamespace(indi_allsky_config={})}) is True


def test_round_trip_and_older_open_pages_preserve_moon_choice(config_expressions):
    load, save = config_expressions
    config = {'CCD_EXPOSURE_MAX': 47.5, 'EXPOSURE_PERIOD': 20.,
              'EXPOSURE_PERIOD_DAY': 7., 'TARGET_ADU': 75,
              'HIGHLIGHT_PROTECTION': {'ENABLE': False, 'TWILIGHT_ENABLE': False}}
    unchanged = copy.deepcopy({key: value for key, value in config.items()
                               if key != 'HIGHLIGHT_PROTECTION'})
    view = SimpleNamespace(indi_allsky_config=config)
    for payload, expected in [
        ({FIELD: False}, False),
        ({'HIGHLIGHT_PROTECTION__ENABLE': True}, False),
        ({FIELD: True}, True),
        ({'HIGHLIGHT_PROTECTION__TWILIGHT_ENABLE': False}, True),
    ]:
        exec(save, {'self': view, 'request': SimpleNamespace(json=payload)})
        view.indi_allsky_config = json.loads(json.dumps(view.indi_allsky_config))
        assert eval(load, {'self': view}) is expected
        assert {key: value for key, value in view.indi_allsky_config.items()
                if key != 'HIGHLIGHT_PROTECTION'} == unchanged


@pytest.mark.parametrize('checked', [False, True])
def test_real_checkbox_posts_both_enable_and_disable(form_class, checked):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(formdata=MultiDict({FIELD: 'y'} if checked else {}))
        assert form.validate(), form.errors
        assert form[FIELD].data is checked


def test_control_renders_checked_and_is_gated_by_highlight_parent(form_class):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class()
        assert form[FIELD].data is True
        assert form.HIGHLIGHT_PROTECTION__ENABLE.data is False
        template = (ROOT / 'flask/templates/config/image.html').read_text(encoding='utf-8')
        start = template.index('                    <label for="' + FIELD + '"')
        end = template.index('                <div class="tw:flex tw:items-center', start)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form)
        assert 'id="' + FIELD + '"' in html
        assert 'checked' in html
        assert 'configured Maximum Exposure' in html
        assert 'Independent of the twilight settings' in html
        assert 'capture intervals stay unchanged' in html
        registry = (ROOT / 'flask/templates/config.html').read_text(encoding='utf-8')
        assert registry.count("'" + FIELD + "'") == 1
        gates = [line for line in registry.splitlines()
                 if ('group_on_ready(' in line or 'group_on_change(' in line)
                 and 'form_config.' + FIELD + '.id' in line]
        assert len(gates) == 2
        assert all('({{ form_config.HIGHLIGHT_PROTECTION__ENABLE.id' in line
                   for line in gates)
