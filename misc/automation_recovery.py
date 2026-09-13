#!/usr/bin/env python3
"""Entry point for the fixed systemd recovery job (no Flask/camera imports)."""
import json
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indi_allsky.automation import ControlStore, SystemServices, boot_id, execute_recovery


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=os.environ.get('INDI_ALLSKY_FLASK_CONFIG', '/etc/indi-allsky/flask.json'))
    args = parser.parse_args()
    with open(args.config) as stream:
        config = json.load(stream)
    execute_recovery(ControlStore(config), SystemServices(config), boot_id())
