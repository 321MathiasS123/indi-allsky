"""Run the camera boundary without native INDI or a connected camera."""
import ast
from copy import deepcopy
from datetime import timedelta
import logging
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest
from indi_allsky.gain import gain_quantum, quantize_gain


ROOT = Path(__file__).resolve().parents[2] / 'indi_allsky'


@pytest.fixture
def client_class():
    # Exercise real driver methods without importing native INDI or starting
    # database services. Device calls below remain observable test doubles.
    tree = ast.parse((ROOT / 'camera/indi.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'IndiClient')
    cls.bases = []
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name in ('getCcdGain', 'setCcdGain', 'getCcdInfo', 'normalize_gain', 'setCcdExposure')]
    namespace = {'logger': logging.getLogger(__name__), 'PyIndi': SimpleNamespace(IP_RO=0),
                 'TimeOutException': TimeoutError, 'gain_quantum': gain_quantum, 'quantize_gain': quantize_gain,
                 'time': SimpleNamespace(time=lambda: 0)}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'camera-control', 'exec'), namespace)
    return namespace['IndiClient']


@pytest.mark.parametrize('driver', ['indi_gphoto_ccd', 'indi_canon_ccd', 'indi_nikon_ccd',
                                   'indi_pentax_ccd', 'indi_sony_ccd'])
@pytest.mark.parametrize('requested,actual', [(360, 400), (401.2, 400), (200, 200), (900, 800)])
def test_discrete_iso_command_and_metadata_agree(client_class, driver, requested, actual):
    client = client_class()
    client.ccd_device = SimpleNamespace(getDriverExec=lambda: driver)
    client._IndiClient__canon_gain_to_iso = {}
    client._IndiClient__canon_iso_to_gain = {}
    switches = [SimpleNamespace(getLabel=lambda g=g: str(g), getName=lambda g=g: 'iso' + str(g))
                for g in [100, 200, 400, 800]]
    switches.append(SimpleNamespace(getLabel=lambda: 'Auto'))
    client.get_control = lambda *a: switches
    info = client.getCcdGain()
    assert info['values'] == [100, 200, 400, 800]
    commands = []
    client.configureDevice = lambda device, settings, **kw: commands.append(settings)
    client._expUtils = SimpleNamespace()
    client.setCcdGain(requested)
    assert commands == [{'SWITCHES': {'CCD_ISO': {'on': ['iso' + str(actual)]}}}]
    assert client.gain == client._expUtils.GAIN_CURRENT == actual


@pytest.mark.parametrize('permission,expected', [(0, False), (1, True), (2, True)])
def test_indi_exposure_permission_is_reported(client_class, permission, expected):
    class Vector(list):
        def getPermission(self):
            return permission
    client = client_class()
    client.ccd_device = object()
    client.get_control = lambda *a, **kw: Vector()
    client.getCcdGain = lambda: {'min': -1, 'max': -1}
    client.getCcdBinning = lambda: {}
    client.getCcdSerialNumber = lambda: {}
    assert client.getCcdInfo()['EXPOSURE_CONTROL'] is expected


@pytest.mark.parametrize('camera_file,class_name', [('indi_passive.py', 'IndiClientPassive'),
                                                  ('pycurl_camera.py', 'IndiClientPycurl'),
                                                  ('test_cameras.py', 'IndiClientTestCameraBase')])
def test_non_controlling_interfaces_explicitly_disable_control(camera_file, class_name):
    tree = ast.parse((ROOT / 'camera' / camera_file).read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    capability = next(n for n in cls.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'exposure_control' for t in n.targets))
    assert ast.literal_eval(capability.value) is False


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('interface_control,driver_control', [(True, True), (True, False), (False, True)])
def test_unsupported_camera_warns_and_disables_all_highlight_processing_only_at_runtime(enabled, interface_control, driver_control):
    saved = {'HIGHLIGHT_PROTECTION': {'ENABLE': enabled, 'GAMMA_DAY': 1.85}, 'TARGET_ADU': 70}
    notifications = []
    worker = SimpleNamespace(config=deepcopy(saved),
                             indiclient=SimpleNamespace(exposure_control=interface_control),
                             _miscDb=SimpleNamespace(addNotification=lambda *a, **kw: notifications.append(a)),
                             exposure_o=SimpleNamespace())
    namespace = dict(self=worker, ccd_info={'EXPOSURE_CONTROL': driver_control},
                     logger=logging.getLogger(__name__), timedelta=timedelta,
                     NotificationCategory=SimpleNamespace(GENERAL='general'))
    source = (ROOT / 'capture.py').read_text(encoding='utf-8')
    start = source.index('        exposure_control = (')
    end = source.index("        if self.config.get('CFA_PATTERN'):", start)
    exec(textwrap.dedent(source[start:end]), namespace)
    control = namespace['exposure_control']
    assert control is (interface_control and driver_control)
    assert bool(notifications) is (enabled and not control)
    camera = SimpleNamespace(data={'exposure_control': control, 'gain_values': [100, 200, 400], 'gain_quantum': 1})
    source = (ROOT / 'image.py').read_text(encoding='utf-8')
    start = source.index('        camera_data = camera.data or {}')
    end = source.index('        ### Special function:', start)
    namespace['camera'] = camera
    exec(textwrap.dedent(source[start:end]), namespace)
    start = source.index('        # Apply the capability guard')
    end = source.index('        # Purple-frame handling deliberately', start)
    guard = textwrap.dedent(source[start:end])
    # Per-frame settings providers may replace the highlight settings dictionary.
    for _ in range(2):
        worker.config['HIGHLIGHT_PROTECTION'] = dict(saved['HIGHLIGHT_PROTECTION'])
        exec(guard, namespace)
        assert worker.config['HIGHLIGHT_PROTECTION']['ENABLE'] is (enabled and control)
    assert worker.config['HIGHLIGHT_PROTECTION']['GAMMA_DAY'] == 1.85
    assert worker.exposure_o.gain_values == [100, 200, 400]
    assert worker.exposure_o.gain_quantum == 1
    assert saved['HIGHLIGHT_PROTECTION']['ENABLE'] is enabled


@pytest.mark.parametrize('driver', ['indi_asi_ccd', 'indi_asi_single_ccd', 'indi_playerone_ccd',
                                   'indi_playerone_single_ccd', 'indi_toupcam_ccd', 'indi_altair_ccd',
                                   'indi_altaircam_ccd', 'indi_nncam_ccd', 'indi_tscam_ccd',
                                   'indi_ogmacam_ccd', 'indi_omegonprocam_ccd', 'indi_svbony_ccd',
                                   'indi_svbonycam_ccd', 'indi_sv305_ccd', 'indi_qhy_ccd', 'indi_libcamera_ccd'])
def test_driver_precision_overrides_gui_step_and_agrees_with_command_metadata(client_class, driver):
    client = client_class()
    client.ccd_device = SimpleNamespace(getDriverExec=lambda: driver)
    client._IndiClient__canon_gain_to_iso = {}
    # ASI's range/10 GUI increment must not become a 60-unit hardware step.
    widget = SimpleNamespace(getValue=lambda: 85.912, min=0, max=600, step=60, format='%g')
    client.get_control = lambda *a, **kw: [widget]
    client._IndiClient__map_indexes = lambda ctl, names: {names[0]: 0}
    info = client.getCcdGain()
    expected = 85.912 if driver in ('indi_qhy_ccd', 'indi_libcamera_ccd') else 86
    assert info['quantum'] == (0 if expected == 85.912 else 1)
    commands = []
    client.configureDevice = lambda device, settings, **kw: commands.append(settings)
    client._expUtils = SimpleNamespace()
    client.setCcdGain(85.912)
    property_values = next(iter(commands[0]['PROPERTIES'].values()))
    assert next(iter(property_values.values())) == expected
    assert client.gain == client._expUtils.GAIN_CURRENT == expected


def test_equivalent_integer_gain_requests_do_not_resend_camera_commands(client_class):
    client = client_class()
    client.ccd_device = SimpleNamespace(getDriverExec=lambda: 'indi_asi_ccd')
    client._IndiClient__canon_gain_to_iso = {}
    client.timeout, client.gain, client.binning = 60, -1, 1
    client._expUtils = SimpleNamespace()
    commands = []
    client.configureDevice = lambda device, settings, **kw: commands.append(settings)
    client.set_number = lambda *a, **kw: None
    client.setCcdExposure(30, 85.912, 1)
    client.setCcdExposure(30, 86.1, 1)
    assert len(commands) == 1 and client.gain == 86
    assert client._expUtils.GAIN_CURRENT == 86 and client._expUtils.EXPOSURE_CURRENT == 30


@pytest.mark.parametrize('info,requested,expected', [
    ({'quantum': 1}, 85.912, 86), ({'quantum': 0}, 1.875, 1.875),
    ({'values': [100, 200, 400, 800]}, 385.9, 400), ({}, 85.912, 85.912),
])
def test_capture_startup_seeds_representable_pending_and_current_gain(info, requested, expected):
    source = (ROOT / 'capture.py').read_text(encoding='utf-8')
    start = source.index('        # Seed pending/current with the same command')
    end = source.index('        self._expUtils.GAIN_MAX_NIGHT', start)
    worker = SimpleNamespace(_expUtils=SimpleNamespace())
    namespace = dict(self=worker, ccd_gain_default=requested, ccd_info={'GAIN_INFO': info}, quantize_gain=quantize_gain)
    exec(textwrap.dedent(source[start:end]), namespace)
    assert worker._expUtils.GAIN_CURRENT == worker._expUtils.GAIN_NEXT == expected
