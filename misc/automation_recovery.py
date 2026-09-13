#!/usr/bin/env python3
"""Entry point for the fixed systemd recovery job (no Flask/camera imports)."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indi_allsky.automation import ControlStore, SystemServices, boot_id, execute_recovery


if __name__ == '__main__':
    path = os.environ.get('INDI_ALLSKY_FLASK_CONFIG', '/etc/indi-allsky/flask.json')
    with open(path) as stream:
        config = json.load(stream)
    execute_recovery(ControlStore(config), SystemServices(config), boot_id())
