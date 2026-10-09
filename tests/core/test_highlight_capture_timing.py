"""Dispatch real exposure requests on a clock independent of render completion."""
import ast
import heapq
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from indi_allsky.highlight import HighlightMeasurement
from indi_allsky.highlight_meter import OutputFeedbackGate
from test_highlight_exposure import controller


def capture_method():
    # Execute the real dispatch boundary without importing camera/DBus drivers.
    source = Path(__file__).parents[2] / 'indi_allsky' / 'capture.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'CaptureWorker')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'shoot')
    namespace = {'logger': logging.getLogger(__name__)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['shoot']


def run_capture_clock(initial, render_seconds):
    control = controller('exposure_basic', night=False)
    control.config.update(EXPOSURE_PERIOD=20, EXPOSURE_PERIOD_DAY=20)
    control.config['HIGHLIGHT_PROTECTION']['OUTPUT_ENABLE'] = True
    state = control._expUtils
    state.EXPOSURE_NEXT, state.GAIN_NEXT, state.BINNING_NEXT = initial, 0, 1
    gate = OutputFeedbackGate()
    events = [(1000., 'capture', 0)]
    jobs, commands, decisions, displayed = {}, [], [], []
    render_free = meter_free = 0.
    feedback = None
    last_decision = None
    now = 0.

    def camera_command(exposure, gain, binning, **kwargs):
        # The driver records CURRENT when issuing the hardware command.
        state.EXPOSURE_CURRENT, state.GAIN_CURRENT = exposure, gain
        commands.append((now, exposure, gain, last_decision))

    capture = SimpleNamespace(
        indiclient=SimpleNamespace(setCcdExposure=camera_command),
        config=control.config, focus_mode=False, night=False, add_period_delay=0,
        frame_deadline=Mock(), night_av=control.night_av, _period_queue=Mock(),
        camera_id=1, _dateCalcs=Mock(),
    )
    shoot = capture_method()
    while events:
        now, kind, index = heapq.heappop(events)
        if kind == 'capture':
            shoot(capture, state.EXPOSURE_NEXT, state.GAIN_NEXT, state.BINNING_NEXT, sync=False)
            exposure = commands[-1][1]
            arrival = now + exposure + .9
            jobs[index] = dict(exp_time=arrival, exposure=exposure, exp_elapsed=exposure + .9,
                               gain=0, binning=1, camera_id=1, capture_mode=(0, 0), capture_period=20)
            heapq.heappush(events, (arrival, 'raw', index))
            if index < 39:
                # Production starts when both the period and camera readiness
                # permit it; it does not wait for either meter or renderer.
                heapq.heappush(events, (max(now + 20, arrival + .02), 'capture', index + 1))
        elif kind == 'raw':
            meter_free = max(now, meter_free) + .85
            heapq.heappush(events, (meter_free, 'meter', index))
        elif kind == 'meter':
            job = jobs[index]
            gate.update(control, job, feedback, allowance=.85)
            scene = 1.06 ** min(index, 20) * .94 ** max(index - 20, 0)
            measurement = HighlightMeasurement(0, 0, 110 * scene * job['exposure'] / initial, 0, 0)
            control.compare_highlights(measurement, job['exposure'], job['gain'])
            last_decision = index
            decisions.append((now, index, state.EXPOSURE_NEXT, state.GAIN_NEXT))
            render_free = max(now, render_free) + render_seconds
            heapq.heappush(events, (render_free, 'render', index))
        else:
            feedback = dict(jobs[index], measurement=(0, 0, 80, 0, 0), trusted=True)
            displayed.append((now, index))
    return commands, decisions, displayed


@pytest.mark.parametrize('initial', [1., 17.5, 18.5, 30.])
def test_renderer_crossing_the_capture_period_does_not_add_a_reaction_frame(initial):
    fast, slow = [run_capture_clock(initial, duration) for duration in (16, 21)]
    assert slow[0] == fast[0]  # Actual camera dispatches, not just queued NEXT.
    assert slow[1] == fast[1]  # Raw decisions finish at exactly the same times.
    assert slow[2][-1][0] > fast[2][-1][0] + 20  # Only display accumulates lag.
    for commands, decisions, _ in (fast, slow):
        for start, exposure, gain, decision_index in commands[1:]:
            ready = [decision for decision in decisions if decision[0] <= start]
            if ready:
                _, expected_index, expected_exposure, expected_gain = ready[-1]
                assert (exposure, gain, decision_index) == (expected_exposure, expected_gain, expected_index)
        if initial > 18:
            # Metering completes after the next long exposure has started.
            assert commands[1][1:] == (initial, 0, None)
            assert commands[2][1] == pytest.approx(initial * .9, abs=1e-6)
        else:
            assert commands[1][1] == pytest.approx(initial * .9, abs=1e-6)
            assert commands[1][3] == 0
