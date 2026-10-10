"""Real denoise form fields and config/preview serialization, without services."""
import ast
from datetime import datetime
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import flask
from flask_wtf import FlaskForm
import pytest
from werkzeug.datastructures import MultiDict
from wtforms import SelectField, IntegerField
from wtforms.validators import ValidationError
from wtforms.widgets import NumberInput


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky/flask'
FIELDS = {'IMAGE_DENOISE', 'IMAGE_DENOISE_DAY',
          'IMAGE_DENOISE_STRENGTH', 'IMAGE_DENOISE_STRENGTH_DAY'}


@pytest.mark.parametrize('recorded_date', [datetime(2026, 9, 8, 22, 15), None])
def test_processed_fits_replay_passes_archive_date_to_processor(tmp_path, recorded_date):
    # Execute the endpoint's date resolution and ingestion statements without
    # unrelated camera/colour services. This catches using wall time at add().
    path = ROOT / 'views.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'JsonImageProcessingView')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'dispatch_request')
    assignments = [n for n in ast.walk(method) if isinstance(n, ast.Assign)]
    date = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'image_date' for t in n.targets))
    ingestion = next(n for n in assignments if any(isinstance(t, ast.Name) and t.id == 'i_ref' for t in n.targets))
    filename = tmp_path / 'archive.fit'
    filename.touch()
    os.utime(filename, (1600000000, 1600000000))
    processor = Mock()
    ns = dict(datetime=datetime, filename_p=filename, fits_entry=SimpleNamespace(createDate=recorded_date, camera=object()),
              image_processor=processor, exposure=20, gain=100, binning=1)
    exec(compile(ast.Module(body=[date, ingestion], type_ignores=[]), str(path), 'exec'), ns)
    expected = recorded_date or datetime.fromtimestamp(filename.stat().st_mtime)
    assert processor.add.call_args.args[4] == expected
    assert processor.add.call_args.args[5] == 0.0  # Archive readout duration is unknown.


def load_forms():
    tree = ast.parse((ROOT / 'forms.py').read_text(encoding='utf-8'))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
                'IMAGE_DENOISE_validator', 'IMAGE_DENOISE_STRENGTH_validator'}:
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in {
                'IndiAllskyConfigForm', 'IndiAllskyImageProcessingForm'}:
            node.body = [n for n in node.body if isinstance(n, ast.Assign)
                         and isinstance(n.targets[0], ast.Name)
                         and n.targets[0].id in FIELDS | {'IMAGE_DENOISE_choices'}]
            nodes.append(node)
    ns = dict(FlaskForm=FlaskForm, SelectField=SelectField, IntegerField=IntegerField,
              ValidationError=ValidationError, NumberInput=NumberInput)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'actual-denoise-fields', 'exec'), ns)
    return ns['IndiAllskyConfigForm'], ns['IndiAllskyImageProcessingForm']


def save_fields(data, preview=False):
    """Execute the application's existing assignments, not a mock serializer."""
    tree = ast.parse((ROOT / 'views.py').read_text(encoding='utf-8'))
    destination = 'p_config' if preview else 'self.indi_allsky_config'
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if (isinstance(target, ast.Subscript) and ast.unparse(target.value) == destination
                and isinstance(target.slice, ast.Constant) and target.slice.value in FIELDS):
            assignments.append(node)
    assert len(assignments) == (2 if preview else 4)
    saved = {}
    ns = dict(self=SimpleNamespace(indi_allsky_config=saved), p_config=saved,
              request=SimpleNamespace(json=data))
    exec(compile(ast.Module(body=assignments, type_ignores=[]), 'actual-denoise-save', 'exec'), ns)
    return saved


@pytest.mark.parametrize('preview', [False, True])
@pytest.mark.parametrize('strength', [1, 3, 5])
def test_selection_validates_saves_and_reloads(preview, strength):
    config_form, preview_form = load_forms()
    form_class = preview_form if preview else config_form
    fields = FIELDS if not preview else FIELDS - {'IMAGE_DENOISE_DAY', 'IMAGE_DENOISE_STRENGTH_DAY'}
    data = {key: str(strength) if 'STRENGTH' in key else 'star_aware' for key in fields}
    app = flask.Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = form_class(MultiDict(data))
        assert form.validate(), form.errors
        saved = save_fields(form.data, preview)
        reloaded = form_class(data=saved)
        assert reloaded.IMAGE_DENOISE.data == 'star_aware'
        assert reloaded.IMAGE_DENOISE_STRENGTH.data == strength
        html = str(reloaded.IMAGE_DENOISE())
        assert 'selected value="star_aware"' in html
        for method in ['', 'gaussian_blur', 'median_blur', 'bilateral', 'wavelet']:
            assert f'value="{method}"' in html


@pytest.mark.parametrize('method,strength', [('unknown', '3'), ('star_aware', '0'), ('star_aware', '6')])
def test_invalid_method_and_strength_are_rejected(method, strength):
    _, preview_form = load_forms()
    app = flask.Flask(__name__)
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_request_context():
        form = preview_form(MultiDict({'IMAGE_DENOISE': method, 'IMAGE_DENOISE_STRENGTH': strength}))
        assert not form.validate()
