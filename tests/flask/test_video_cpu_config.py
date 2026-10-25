"""Background CPU control is reversible and survives older open forms."""
import ast
from pathlib import Path
from types import SimpleNamespace

from flask import Flask
from flask_wtf import FlaskForm
from jinja2 import Environment
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import IntegerField
from wtforms.validators import NumberRange
from wtforms.widgets import NumberInput


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky'


@pytest.fixture
def cpu_form():
    tree = ast.parse((ROOT / 'flask/forms.py').read_text(encoding='utf-8'))
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == 'IndiAllskyConfigForm')
    field = next(node for node in original.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'VIDEO_CPU_LIMIT')
    selected = ast.ClassDef(name='CpuForm', bases=[ast.Name(id='FlaskForm', ctx=ast.Load())],
                            keywords=[], body=[field], decorator_list=[])
    namespace = dict(FlaskForm=FlaskForm, IntegerField=IntegerField,
                     NumberRange=NumberRange, NumberInput=NumberInput)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[])),
                 'video-cpu-form', 'exec'), namespace)
    return namespace['CpuForm']


@pytest.mark.parametrize('value', [0, 1, 2, 64])
def test_cpu_limit_accepts_unrestricted_and_bounded_core_counts(cpu_form, value):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = cpu_form(formdata=MultiDict({'VIDEO_CPU_LIMIT': str(value)}))
        assert form.validate(), form.errors
        assert form.VIDEO_CPU_LIMIT.data == value


@pytest.mark.parametrize('value', ['-1', '65', '1.5', 'nan', 'inf', ''])
def test_cpu_limit_rejects_invalid_core_counts(cpu_form, value):
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = cpu_form(formdata=MultiDict({'VIDEO_CPU_LIMIT': value}))
        assert not form.validate()
        assert form.VIDEO_CPU_LIMIT.errors


def test_default_renders_with_save_registry(cpu_form):
    tree = ast.parse((ROOT / 'config.py').read_text(encoding='utf-8'))
    default = next(value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                   for key, value in zip(node.keys, node.values)
                   if isinstance(key, ast.Constant) and key.value == 'VIDEO_CPU_LIMIT')
    assert ast.literal_eval(default) == 1
    app = Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = cpu_form()
        assert form.validate(), form.errors
        assert form.VIDEO_CPU_LIMIT.data == 1
        template = (ROOT / 'flask/templates/config/timelapse.html').read_text(encoding='utf-8')
        start = template.index('<!-- Background Video CPU Limit -->')
        end = template.index('<!-- Pre-Processor Selection -->', start)
        html = Environment(autoescape=True).from_string(template[start:end]).render(form_config=form)
        assert 'id="VIDEO_CPU_LIMIT"' in html
        assert 'id="VIDEO_CPU_LIMIT-error"' in html
        assert 'min="0"' in html and 'max="64"' in html
        registry = (ROOT / 'flask/templates/config.html').read_text(encoding='utf-8')
        assert registry.count("'VIDEO_CPU_LIMIT'") == 1


def test_cpu_limit_round_trip_and_old_form_preservation():
    tree = ast.parse((ROOT / 'flask/views.py').read_text(encoding='utf-8'))
    load_dict = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                     and any(isinstance(key, ast.Constant) and key.value == 'VIDEO_CPU_LIMIT'
                             for key in node.keys))
    value = next(value for key, value in zip(load_dict.keys, load_dict.values)
                 if isinstance(key, ast.Constant) and key.value == 'VIDEO_CPU_LIMIT')
    load = compile(ast.Expression(body=value), 'load-video-cpu', 'eval')
    save = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "'VIDEO_CPU_LIMIT' in request.json")
    save = compile(ast.Module(body=[save], type_ignores=[]), 'save-video-cpu', 'exec')
    view = SimpleNamespace(indi_allsky_config={'FFMPEG_FRAMERATE': 25})
    assert eval(load, {'self': view}) == 1
    for payload, expected in (({'VIDEO_CPU_LIMIT': 2}, 2), ({}, 2), ({'VIDEO_CPU_LIMIT': 0}, 0)):
        exec(save, {'self': view, 'request': SimpleNamespace(json=payload)})
        assert eval(load, {'self': view}) == expected
        assert view.indi_allsky_config['FFMPEG_FRAMERATE'] == 25
