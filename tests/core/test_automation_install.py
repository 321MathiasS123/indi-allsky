import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture(params=['flask.json', 'custom-flask.json'])
def installer(tmp_path, monkeypatch, request):
    source = Path(__file__).resolve().parents[2] / 'misc/install_automation.py'
    spec = importlib.util.spec_from_file_location('automation_installer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / 'installation'
    runtime = root / 'virtualenv/indi-allsky/bin/python3'
    runtime.parent.mkdir(parents=True)
    runtime.touch()
    service = root / 'service/indi-allsky-automation.service'
    service.parent.mkdir()
    service.write_text((source.parents[1] / 'service/indi-allsky-automation.service').read_text())
    monkeypatch.setattr(module, '__file__', str(root / 'misc/install_automation.py'))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path / 'user')
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda args, **kw: calls.append(args))
    config = tmp_path / request.param
    config.write_text(json.dumps({'SECRET_KEY': 'preserve-me',
        'AUTOMATION_STATE_DIR': str(tmp_path / 'control'), 'ALLSKY_SERVICE_NAME': 'camera.service'}))
    return module, config, calls


def test_installer_preserves_configuration_and_never_restarts_capture(installer, tmp_path):
    module, config_path, calls = installer
    module.install(config_path, '192.168.1.20', enable_recovery=True)
    config = json.loads(config_path.read_text())
    assert config['SECRET_KEY'] == 'preserve-me'
    assert len(config['AUTOMATION_TOKEN']) >= 32
    assert config['AUTOMATION_ALLOWED_NETWORKS'] == ['192.168.1.20/32']
    assert config['AUTOMATION_RECOVERY_ENABLE'] is True
    assert config['AUTOMATION_RESTART_INDISERVER'] is False
    assert calls == [['systemctl', '--user', 'daemon-reload']]
    dropin = tmp_path / 'user/.config/systemd/user/camera.service.d/automation-shutdown.conf'
    assert 'ExecStop=\n' in dropin.read_text()
    helper = dropin.parents[1] / 'indi-allsky-automation.service'
    assert str(config_path) in helper.read_text()
    assert '%ALLSKY_' not in helper.read_text()
    assert list(tmp_path.glob(config_path.name + '.automation-*'))


def test_invalid_ha_address_does_not_modify_config(installer):
    module, config, calls = installer
    before = config.read_bytes()
    with pytest.raises(ValueError):
        module.install(config, '192.168.1.0/24')
    assert config.read_bytes() == before and calls == []
