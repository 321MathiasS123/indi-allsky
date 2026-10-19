"""Twilight opt-in survives both new and already-open configuration pages."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky'


def test_twilight_defaults_off():
    tree = ast.parse((ROOT / 'config.py').read_text(encoding='utf-8'))
    defaults = next(value for node in ast.walk(tree) if isinstance(node, ast.Dict)
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and key.value == 'HIGHLIGHT_PROTECTION')
    values = ast.literal_eval(defaults)
    assert values['TWILIGHT_ENABLE'] is False
    allowances = [values[key] for key in ('TWILIGHT_STAR_ALLOWANCE', 'TWILIGHT_MOON_ALLOWANCE_50',
                                        'TWILIGHT_MOON_ALLOWANCE_FULL')]
    assert all(math.isfinite(value) for value in allowances)
    assert 0 <= allowances[0] <= allowances[1] <= allowances[2] <= 50


def test_twilight_config_round_trip_and_older_page_preserves_setting():
    tree = ast.parse((ROOT / 'flask/views.py').read_text(encoding='utf-8'))
    load_dict = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                     and any(isinstance(key, ast.Constant) and key.value == 'HIGHLIGHT_PROTECTION__ENABLE'
                             for key in node.keys))
    load_value = next(value for key, value in zip(load_dict.keys, load_dict.values)
                      if isinstance(key, ast.Constant) and key.value == 'HIGHLIGHT_PROTECTION__TWILIGHT_ENABLE')
    load = compile(ast.Expression(body=load_value), 'load-twilight-setting', 'eval')
    save = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'update'
                and 'HIGHLIGHT_PROTECTION' in ast.unparse(node))
    save_code = compile(ast.Module(body=[save], type_ignores=[]), 'save-twilight-setting', 'exec')
    view = SimpleNamespace(indi_allsky_config={'TARGET_ADU': 75})
    assert eval(load, {'self': view}) is False

    for payload, expected in (({'HIGHLIGHT_PROTECTION__TWILIGHT_ENABLE': True}, True),
                              ({'HIGHLIGHT_PROTECTION__ENABLE': True}, True),
                              ({'HIGHLIGHT_PROTECTION__TWILIGHT_ENABLE': False}, False)):
        exec(save_code, {'self': view, 'request': SimpleNamespace(json=payload)})
        assert eval(load, {'self': view}) is expected
        assert view.indi_allsky_config['TARGET_ADU'] == 75


def test_all_twilight_allowances_round_trip_and_survive_older_open_page():
    tree = ast.parse((ROOT / 'flask/views.py').read_text(encoding='utf-8'))
    load_dict = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                     and any(isinstance(key, ast.Constant) and key.value == 'HIGHLIGHT_PROTECTION__ENABLE'
                             for key in node.keys))
    selected = [(key, value) for key, value in zip(load_dict.keys, load_dict.values)
                if isinstance(key, ast.Constant) and key.value.startswith('HIGHLIGHT_PROTECTION__TWILIGHT_')]
    load = compile(ast.fix_missing_locations(ast.Expression(body=ast.Dict(
        keys=[key for key, _ in selected], values=[value for _, value in selected]))), 'load-twilight-fields', 'eval')
    save = next(node for node in ast.walk(tree) if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'update'
                and 'HIGHLIGHT_PROTECTION' in ast.unparse(node))
    save_code = compile(ast.Module(body=[save], type_ignores=[]), 'save-twilight-fields', 'exec')
    view = SimpleNamespace(indi_allsky_config={'TARGET_ADU': 75})
    payload = {'HIGHLIGHT_PROTECTION__TWILIGHT_ENABLE': True,
               'HIGHLIGHT_PROTECTION__TWILIGHT_STAR_ALLOWANCE': .02,
               'HIGHLIGHT_PROTECTION__TWILIGHT_MOON_ALLOWANCE_50': .7,
               'HIGHLIGHT_PROTECTION__TWILIGHT_MOON_ALLOWANCE_FULL': 2.4}
    exec(save_code, {'self': view, 'request': SimpleNamespace(json=payload)})
    assert eval(load, {'self': view}) == payload
    # An older page sends neither the opt-in nor the new phase fields.
    exec(save_code, {'self': view, 'request': SimpleNamespace(json={'HIGHLIGHT_PROTECTION__ENABLE': True})})
    assert eval(load, {'self': view}) == payload
    assert view.indi_allsky_config['TARGET_ADU'] == 75
