from contextlib import nullcontext
from multiprocessing import Array
import queue
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indi_allsky.exposure import exposure_basic
from indi_allsky.highlight import HighlightMeasurement, HighlightTransition
from indi_allsky.highlight_meter import (
    HighlightMeterWorker, OutputFeedbackGate, apply_control_snapshot, capture_cadence,
)


def controller():
    config = {'TARGET_ADU': 70, 'TARGET_ADU_DAY': 80, 'TARGET_ADU_DEV': 10,
              'TARGET_ADU_DEV_DAY': 10, 'EXPOSURE_PERIOD': 20, 'EXPOSURE_PERIOD_DAY': 20,
              'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'OUTPUT_ENABLE': True}}
    result = exposure_basic(config, Array('i', 7), Array('i', 10), Array('i', 6), [1, 0])
    utils = result._expUtils
    utils.EXPOSURE_MIN_DAY = utils.EXPOSURE_MIN_NIGHT = .001
    utils.EXPOSURE_MAX = 30
    utils.GAIN_MIN_NIGHT = utils.GAIN_MAX_NIGHT = 0
    utils.BINNING_NIGHT = 1
    return result


def job(**changes):
    return dict({'filename': 'original.fit', 'exp_time': 100., 'exp_elapsed': 1.,
                 'exposure': 1., 'gain': 0, 'binning': 1, 'camera_id': 1,
                 'capture_mode': (1, 0), 'capture_period': 20}, **changes)


def feedback(**changes):
    return dict(job(exp_time=80.), measurement=(3., 5., 0., 0., 0.), trusted=True, **changes)


@pytest.mark.parametrize('period,exposure,expected', [(5, 1, 5), (20, 1, 20), (20, 30, 30), (60, 10, 60)])
def test_feedback_age_tracks_configured_cadence_and_long_exposure(period, exposure, expected):
    control = controller()
    capture = job(capture_period=period, exposure=exposure)
    assert capture_cadence(capture, control.config) == expected
    gate = OutputFeedbackGate()
    output = feedback()
    output.update(exp_time=100 - expected, exposure=exposure)
    assert gate.update(control, capture, output)
    assert control.highlight_output.active
    capture['exp_time'] += .1
    assert not gate.update(control, capture, None)
    assert not control.highlight_output.active


@pytest.mark.parametrize('change', [
    {'exp_time': 79.}, {'exp_time': 101.}, {'camera_id': 2}, {'binning': 2},
    {'exposure': 1.1}, {'gain': 1}, {'capture_mode': (0, 0)},
    {'trusted': False}, {'measurement': None},
])
def test_stale_or_incompatible_output_cannot_hold_raw_control(change):
    control = controller()
    control.highlight_output.observe(HighlightMeasurement(3, 5, 0), 1, 0, (1, 0), {})
    output = feedback()
    output.update(change)
    assert not OutputFeedbackGate().update(control, job(), output)
    assert not control.highlight_output.active
    assert control.highlight_output.measurement is None


def test_reengagement_requires_two_distinct_fresh_outputs_not_two_reads():
    control = controller()
    gate = OutputFeedbackGate()
    assert not gate.update(control, job(), None)
    assert not gate.update(control, job(), feedback())
    assert not gate.update(control, job(), None)
    capture = job(exp_time=120.)
    output = feedback()
    output['exp_time'] = 100.
    assert gate.update(control, capture, output)
    assert control.highlight_output.active
    assert not gate.degraded


def test_metering_allowance_and_output_toggle_never_rewind_latest_feedback():
    control = controller()
    gate = OutputFeedbackGate()
    assert gate.update(control, job(exp_time=102.), feedback(), allowance=2)
    older = feedback()
    older.update(exp_time=70., measurement=(0, 0, 0, 0, 0))
    assert gate.update(control, job(exp_time=102.), older, allowance=2)
    assert control.highlight_output.active
    control.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = False
    assert not gate.update(control, job(exp_time=102.), None, allowance=2)
    assert not control.highlight_output.active


def snapshot(control, **state):
    transition = control.highlight_transition
    return {'stable': control.target_adu_found, 'current_adu_target': control.current_adu_target,
            'transition': dict(active=transition.active, trusted=transition.trusted,
                               reason=transition.reason, **state)}


def test_capture_decisions_preserve_renderer_lift_and_gamma_progress():
    control = controller()
    render = controller()
    control.highlight_transition.active = True
    control.highlight_transition.trusted = True
    render.highlight_transition.active = True
    render.highlight_transition.lift = 1.25
    render.highlight_transition.gamma_mix = .65
    render.highlight_transition.reference = 40.
    apply_control_snapshot(render, snapshot(control))
    assert render.highlight_transition.lift == 1.25
    assert render.highlight_transition.gamma_mix == .65
    assert render.highlight_transition.reference == 40.
    control.highlight_transition.active = False
    apply_control_snapshot(render, snapshot(control))
    assert not render.highlight_transition.active
    assert render.highlight_transition.lift == 1.25  # ordinary gradual release owns this
    assert render.highlight_transition.gamma_mix == .65


def test_startup_seed_is_applied_once_and_disabled_state_resets_rendering():
    control = controller()
    render = controller()
    control.highlight_transition.active = True
    apply_control_snapshot(render, snapshot(control, seed=True, target=70.))
    assert render.highlight_transition.reference == 70
    assert render.highlight_transition.gamma_mix == 1.
    render.highlight_transition.gamma_mix = .7
    apply_control_snapshot(render, snapshot(control, seed=False))
    assert render.highlight_transition.gamma_mix == .7
    apply_control_snapshot(render, snapshot(control, reset=True))
    expected = HighlightTransition()
    expected.reset()
    assert render.highlight_transition.__dict__ == expected.__dict__


def worker():
    result = HighlightMeterWorker.__new__(HighlightMeterWorker)
    result.controller = controller()
    result.config = result.controller.config
    result.frame_mode = [1, 0]
    result.night_av = [1, 0]
    result.image_q = queue.Queue()
    return result


@pytest.mark.parametrize('failure', [False, True])
def test_forward_retains_original_and_suppresses_later_control_even_on_failure(tmp_path, monkeypatch, failure):
    import indi_allsky.highlight_meter as module
    monkeypatch.setattr(module.time, 'time', lambda: 101.)
    original = tmp_path / 'original.fit'
    original.write_bytes(b'original camera data')
    meter = worker()
    if failure:
        meter.meter = Mock(side_effect=RuntimeError('calibration unavailable'))
    else:
        meter.meter = Mock(return_value=(HighlightMeasurement(0, 0, 30), 'metered', False, False))
    capture = job(filename=str(original))
    meter.forward(capture)
    forwarded = meter.image_q.get_nowait()
    assert forwarded is capture
    assert original.read_bytes() == b'original camera data'
    assert forwarded['highlight_control']['status'] == ('metering failed' if failure else 'metered')
    assert forwarded['highlight_control']['measurement'] == (None if failure else (0, 0, 30, 0, 0))
    assert meter.image_q.empty()


@pytest.mark.parametrize('change,now,status', [
    ({'capture_mode': (0, 0)}, 101, 'old capture mode'),
    ({}, 121, 'old capture'),
])
def test_stale_capture_is_forwarded_without_any_exposure_request(monkeypatch, change, now, status):
    import indi_allsky.highlight_meter as module
    monkeypatch.setattr(module.time, 'time', lambda: now)
    meter = worker()
    meter.meter = Mock()
    meter.forward(job(**change))
    meter.meter.assert_not_called()
    assert meter.image_q.get_nowait()['highlight_control']['status'] == status


def test_raw_pending_guard_survives_output_bypass_and_recovers_on_actual_capture():
    control = controller()
    control._expUtils.EXPOSURE_NEXT = 1.889
    gate = OutputFeedbackGate()
    assert not gate.update(control, job(exposure=1.7), feedback())
    control.compare_highlights(HighlightMeasurement(0, 0, 3.86), 1.7, 0)
    assert control._expUtils.EXPOSURE_NEXT == pytest.approx(1.889, abs=1e-6)
    control.compare_highlights(HighlightMeasurement(0, 0, 3.86), 1.889, 0)
    assert control._expUtils.EXPOSURE_NEXT == pytest.approx(2.0779, abs=1e-6)


@pytest.mark.parametrize('mode_changes,finishes_at', [(False, 101), (True, 101), (False, 121)])
def test_meter_orders_raw_clipping_before_calibration_and_commands_only_fresh_mode(monkeypatch, mode_changes, finishes_at):
    import indi_allsky.highlight_meter as module
    monkeypatch.setattr(module.time, 'time', lambda: finishes_at)
    meter = worker()
    meter.config['CAMERA_INTERFACE'] = 'indi'
    meter.feedback_gate = OutputFeedbackGate()
    meter.output_feedback_q = queue.Queue()
    meter.controller.compare_highlights = Mock(wraps=meter.controller.compare_highlights)
    meter.controller.apply_transition_limits = Mock()
    events = []
    reference = SimpleNamespace(asi676mc_repair_result=None, libcamera_black_level=0)
    measurement = HighlightMeasurement(1, 2, 200)

    def raw_meter():
        events.append('raw')
        if mode_changes:
            meter.night_av[:] = [0, 0]
        return measurement

    meter.processor = SimpleNamespace(
        focus_mode=False, update_astrometric_data=lambda *_: events.append('astro'),
        add=lambda *args, **kwargs: events.append('decode') or reference,
        correct_asi676mc_frame=lambda *_: events.append('repair'),
        measure_highlights=raw_meter,
        calibrate=lambda **kwargs: events.append('calibrate'),
        fix_holes_early=lambda: events.append('holes'),
        debayer=lambda: events.append('debayer'),
        calibrate_highlights=lambda m: events.append('calibrated ADU') or m._replace(adu=20),
    )
    camera = SimpleNamespace(data={'gain_values': [], 'gain_quantum': 0})
    table = SimpleNamespace(id=1, query=SimpleNamespace(filter=lambda *_: SimpleNamespace(one=lambda: camera)))
    monkeypatch.setitem(sys.modules, 'indi_allsky.flask.models', SimpleNamespace(IndiAllSkyDbCameraTable=table))
    result, status, _, _ = meter.meter(job(), 0)
    assert events == ['astro', 'decode', 'repair', 'raw', 'calibrate', 'holes', 'debayer', 'calibrated ADU']
    assert result == HighlightMeasurement(1, 2, 20)
    fresh = not mode_changes and finishes_at <= 120
    assert meter.controller.compare_highlights.call_count == int(fresh)
    assert meter.controller.apply_transition_limits.call_count == int(fresh)
    assert status == ('old capture mode' if mode_changes else 'metered' if fresh else 'old capture')


def test_period_barriers_and_sqm_keep_fifo_order_and_stop_is_not_forwarded():
    meter = worker()
    meter.input_q = queue.Queue()
    first = job()
    barrier = {'period_end': {'period_id': 'night', 'tasks': [{'task_id': 1}]}}
    sqm = job(sqm_exposure=True, exp_time=110.)
    second = job(exp_time=120.)
    for item in (first, barrier, sqm, second, {'stop': True}):
        meter.input_q.put(item)
    meter.forward = Mock(side_effect=meter.image_q.put)
    meter.saferun(SimpleNamespace(app_context=nullcontext))
    assert [meter.image_q.get_nowait() for _ in range(4)] == [first, barrier, sqm, second]
    assert meter.image_q.empty()
    assert meter.forward.call_count == 2


@pytest.mark.parametrize('stat_fails', [False, True])
def test_storage_observation_precedes_meter_and_never_consumes_frame(monkeypatch, stat_fails):
    import indi_allsky.highlight_meter as module
    monkeypatch.setattr(module.time, 'time', lambda: 101.)
    meter = worker()
    events = []

    def observe(filename):
        events.append(('stat', filename))
        if stat_fails:
            raise OSError('spool stat unavailable')

    meter.backlog_state = SimpleNamespace(observe_file=observe)
    meter.meter = Mock(side_effect=lambda *_: events.append(('meter', None)) or (
        HighlightMeasurement(0, 0, 30), 'metered', False, False))
    hdus = SimpleNamespace(close=Mock())
    meter.processor = SimpleNamespace(image_list=[SimpleNamespace(hdulist=hdus)])
    meter.frame_temperature = [0, 1]
    meter.sensors_temp_av = [15, 3]
    capture = job(capture_temperature=12.5)
    meter.forward(capture)
    assert events == [('stat', 'original.fit'), ('meter', None)]
    assert meter.frame_temperature == [12.5, 3]
    assert meter.image_q.get_nowait() is capture
    hdus.close.assert_called_once_with()
    assert meter.processor.image_list == []


def test_meter_child_resets_inherited_force_termination_handlers(monkeypatch):
    import indi_allsky.highlight_meter as module
    application = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'indi_allsky.flask', SimpleNamespace(create_app=lambda: application))
    signals = Mock()
    monkeypatch.setattr(module.signal, 'signal', signals)
    meter = worker()
    meter.initialize = Mock()
    meter.saferun = Mock()
    meter.run()
    assert (module.signal.SIGTERM, module.signal.SIG_DFL) in [call.args for call in signals.call_args_list]
    assert (module.signal.SIGINT, module.signal.SIG_DFL) in [call.args for call in signals.call_args_list]
    if hasattr(module.signal, 'SIGHUP'):
        assert (module.signal.SIGHUP, module.signal.SIG_IGN) in [call.args for call in signals.call_args_list]
    meter.initialize.assert_called_once_with()
    meter.saferun.assert_called_once_with(application)
