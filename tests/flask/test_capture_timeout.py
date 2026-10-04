import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from wtforms.validators import ValidationError


@pytest.fixture
def validate_timeout():
    path = Path(__file__).resolve().parents[2] / 'indi_allsky/flask/forms.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'CCD_EXPOSURE_TIMEOUT_validator')
    namespace = {'ValidationError': ValidationError}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CCD_EXPOSURE_TIMEOUT_validator']


@pytest.mark.parametrize('seconds', [0, 1, 60, 80, 120, 150, 330, 1000])
def test_timeout_allows_automatic_or_adjustable_seconds(validate_timeout, seconds):
    validate_timeout(None, SimpleNamespace(data=seconds))


@pytest.mark.parametrize('seconds', [-1, None, '80', 80.5])
def test_timeout_rejects_negative_or_noninteger_values(validate_timeout, seconds):
    with pytest.raises(ValidationError):
        validate_timeout(None, SimpleNamespace(data=seconds))
