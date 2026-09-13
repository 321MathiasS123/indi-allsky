"""Exercise the shipped HA policy templates and quiet NAS probe without HA."""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib import request

from jinja2 import StrictUndefined
from jinja2.nativetypes import NativeEnvironment
import pytest
import yaml


@pytest.fixture
def package():
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('!secret', lambda loader, node: 'private-token-placeholder')
    path = Path(__file__).resolve().parents[2] / 'examples/homeassistant/indi_allsky.yaml'
    return yaml.load(path.read_text(), Loader=Loader)


def templates(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from templates(child)
    elif isinstance(value, list):
        for child in value:
            yield from templates(child)
    elif isinstance(value, str) and ('{{' in value or '{%' in value):
        yield value


def environment():
    env = NativeEnvironment(undefined=StrictUndefined)
    env.filters['to_json'] = lambda value: value
    def timestamp(value, default=None):
        if isinstance(value, datetime):
            return value.timestamp()
        if isinstance(value, (int, float)):
            return value
        try:
            return datetime.fromisoformat(value).timestamp()
        except (ValueError, TypeError):
            return default
    env.globals.update(as_timestamp=timestamp, now=lambda: datetime.fromtimestamp(100000, timezone.utc))
    return env


def test_all_shipped_templates_compile(package):
    env = environment()
    for source in templates(package):
        env.from_string(source)


@pytest.mark.parametrize('state,reason,age,expected', [
    ('idle', None, 0, True), ('interrupted', 'maintenance', 0, True),
    ('complete', None, 21599, False), ('complete', None, 21600, True),
    ('failed', 'connection', 1799, False), ('failed', 'connection', 1800, True),
    ('failed', 'authentication', 100000, False), ('cancelled', 'manual_cancel', 100000, False),
])
def test_sync_cadence_and_terminal_states(package, state, reason, age, expected):
    source = next(value for value in templates(package['automation'][0]) if 'state in [' in value)
    result = environment().from_string(source).render(sync={
        'state': state, 'reason': reason, 'finished': 100000 - age})
    # HA trims rendered template strings before interpreting condition booleans.
    assert str(result).strip() == str(expected)


def test_recovery_requires_new_capture_and_new_boot(package):
    source = next(value for value in templates(package['automation'][1])
                  if "get('boot_id') != original_boot" in value)
    template = environment().from_string(source)
    variables = dict(repeat=SimpleNamespace(index=1), original_boot='old', before=100)
    for boot, capture, expected in [('old', 101, False), ('new', 100, False), ('new', 101, True)]:
        result = template.render(**variables, status_reply=dict(status=200,
            content=dict(boot_id=boot, capture=dict(last_capture=capture))))
        assert result is expected
    assert template.render(**variables, status_reply={}) is False


@pytest.mark.parametrize('online', [True, False])
def test_nas_probe_treats_offline_as_normal_data(package, monkeypatch, capsys, online):
    command = package['command_line'][0]['binary_sensor']['command']
    source = command.split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    def urlopen(*args, **kwargs):
        if not online:
            raise OSError('NAS is off')
        return Response()
    monkeypatch.setattr(request, 'urlopen', urlopen)
    exec(compile(source, '<NAS probe>', 'exec'), {})
    output = capsys.readouterr()
    assert output.out == ('ON\n' if online else 'OFF\n')
    assert output.err == ''
