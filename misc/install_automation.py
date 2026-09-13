#!/usr/bin/env python3
"""Install control units without restarting capture or rebooting the Pi."""
import argparse
from datetime import datetime
import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys


def install(config_path, ha_address, enable_recovery=False, restart_indiserver=False):
    config_path = Path(config_path).resolve()
    address = ipaddress.ip_address(ha_address)
    config = json.loads(config_path.read_text())
    source = Path(__file__).resolve().parents[1]
    unit = config.get('ALLSKY_SERVICE_NAME', 'indi-allsky.service')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', unit):
        raise ValueError('Invalid ALLSKY_SERVICE_NAME')
    interpreter = source / 'virtualenv/indi-allsky/bin/python3'
    if not interpreter.is_file():
        raise ValueError('Run this installer from the Pi installation with its existing virtualenv')
    # Keep a private backup and leave other configuration keys untouched.
    backup = config_path.with_name(config_path.name + '.automation-' + datetime.now().strftime('%Y%m%d%H%M%S'))
    shutil.copy2(config_path, backup)
    backup.chmod(0o600)
    if len(config.get('AUTOMATION_TOKEN', '')) < 32:
        config['AUTOMATION_TOKEN'] = secrets.token_urlsafe(32)
    config['AUTOMATION_ALLOWED_NETWORKS'] = [str(address) + ('/32' if address.version == 4 else '/128')]
    config['AUTOMATION_RECOVERY_ENABLE'] = enable_recovery
    config['AUTOMATION_RESTART_INDISERVER'] = restart_indiserver
    config.setdefault('AUTOMATION_STATE_DIR', '/var/lib/indi-allsky/automation')
    state_dir = Path(config['AUTOMATION_STATE_DIR'])
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    secret_file = state_dir / 'homeassistant-secret.yaml'
    secret_file.touch(mode=0o600, exist_ok=True)
    secret_file.chmod(0o600)
    secret_file.write_text('allsky_automation_authorization: "Bearer ' + config['AUTOMATION_TOKEN'] + '"\n')
    # Open the existing file to preserve its owner and permissions.
    with config_path.open('w') as stream:
        json.dump(config, stream, indent=4)
        stream.write('\n')
    user_units = Path.home() / '.config/systemd/user'
    user_units.mkdir(parents=True, exist_ok=True)
    helper = (source / 'service/indi-allsky-automation.service').read_text()
    # A non-default flask.json must be identical in Gunicorn and this helper.
    helper = helper.replace('%ALLSKY_ETC%/flask.json', str(config_path))
    helper = helper.replace('%ALLSKY_DIRECTORY%', str(source)).replace('%ALLSKY_ETC%', str(config_path.parent))
    (user_units / 'indi-allsky-automation.service').write_text(helper)
    dropin = user_units / (unit + '.d')
    dropin.mkdir(exist_ok=True)
    (dropin / 'automation-shutdown.conf').write_text(
        '[Service]\nExecStop=\nKillSignal=SIGINT\nKillMode=mixed\n'
        'Environment=INDI_ALLSKY_SHUTDOWN_GRACE=180\nTimeoutStopSec=240\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True, timeout=30)
    print('Installed. Capture was not restarted.')
    print('Private HA secret:', secret_file)
    print('Configuration backup:', backup)
    print('Restart only the Gunicorn web service to load the token; then verify /automation/status.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ha-address', required=True, help='IP address of the HA host as seen by the Pi')
    parser.add_argument('--config', default=os.environ.get('INDI_ALLSKY_FLASK_CONFIG', '/etc/indi-allsky/flask.json'))
    parser.add_argument('--enable-recovery', action='store_true')
    parser.add_argument('--restart-indiserver', action='store_true', help='Also restart a local INDI driver during capture recovery')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Installation requires Linux/systemd; unit tests can run on other platforms')
    install(args.config, args.ha_address, args.enable_recovery, args.restart_indiserver)
