"""Run the on-demand panel regressions through the shared pytest suite."""
from pathlib import Path
import shutil
import subprocess


def test_syncapi_on_demand_panel():
    node = shutil.which('node')
    assert node, 'Node.js is required for the synchronization panel tests'
    result = subprocess.run([node, '--test', str(Path(__file__).with_name('syncapi_on_demand.test.cjs'))],
                            capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
