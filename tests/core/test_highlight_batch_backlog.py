"""Batch rendering must not delay the independent raw exposure controller."""
from multiprocessing import Array

import pytest

from indi_allsky.exposure import exposure_basic
from indi_allsky.highlight import HighlightMeasurement
from indi_allsky.highlight_meter import OutputFeedbackGate


def make_controller(period):
    config = {'TARGET_ADU': 70, 'TARGET_ADU_DAY': 80, 'TARGET_ADU_DEV': 10,
              'TARGET_ADU_DEV_DAY': 10, 'EXPOSURE_PERIOD': period,
              'EXPOSURE_PERIOD_DAY': period,
              'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'OUTPUT_ENABLE': True}}
    control = exposure_basic(config, Array('i', 7), Array('i', 10), Array('i', 6), [1, 0])
    state = control._expUtils
    state.EXPOSURE_MIN_DAY = state.EXPOSURE_MIN_NIGHT = .001
    state.EXPOSURE_MAX = 30
    state.GAIN_MIN_NIGHT = state.GAIN_MAX_NIGHT = 0
    state.BINNING_NIGHT = 1
    state.EXPOSURE_CURRENT = state.EXPOSURE_NEXT = 1
    state.GAIN_CURRENT = state.GAIN_NEXT = 0
    return control


@pytest.mark.parametrize('period', [10, 20, 40])
def test_batch_load_keeps_raw_control_current_and_feedback_recovers(period):
    control, raw_only = make_controller(period), make_controller(period)
    gate = OutputFeedbackGate()
    finished, render_end = [], 0.
    next_result = 0
    accepted, exposures, backlog = [], [], []
    # At the real 20s cadence these are 30 minutes of 21s renders followed by
    # 16s renders. Other cadences exercise the same automatic freshness rules.
    for index in range(150):
        exposure = control._expUtils.EXPOSURE_NEXT
        capture_end = 1000 + index * period + exposure + .9
        metered_at = capture_end + .85
        capture = dict(exp_time=capture_end, exposure=exposure, exp_elapsed=exposure+.9,
                       gain=0, binning=1, camera_id=1, capture_mode=(1, 0),
                       capture_period=period)
        latest = None
        while next_result < len(finished) and finished[next_result][0] <= metered_at:
            latest = finished[next_result][1]
            next_result += 1
        accepted.append(gate.update(control, capture, latest, allowance=.85))
        if index < 90:
            assert not accepted[-1]
        backlog.append(len(finished) - next_result)
        # Dawn brightens, then a cloud darkens the sky during the batch job.
        # Rendered clipping from older frames must not block that recovery.
        scene = 1.035 ** min(index, 30)
        if index > 30:
            scene *= .97 ** min(index-30, 40)
        measurement = HighlightMeasurement(0, 0, 70 * scene * exposure, 0, 0)
        for instance in (control, raw_only):
            instance._expUtils.EXPOSURE_CURRENT = exposure
            instance.compare_highlights(measurement, exposure, 0)
        assert control._expUtils.EXPOSURE_NEXT == pytest.approx(raw_only._expUtils.EXPOSURE_NEXT)
        exposures.append(control._expUtils.EXPOSURE_NEXT)
        output = dict(capture, measurement=(3, 5, 0, 0, 0) if index < 90 else (0, 0, 70, 0, 0),
                      trusted=True)
        render_end = max(render_end, metered_at) + period * (1.05 if index < 90 else .8)
        finished.append((render_end, output))
    assert max(backlog) <= 6
    assert backlog[-1] == 0
    assert exposures[30] < exposures[0]
    assert exposures[75] > exposures[30]
    assert all(accepted[-10:])
    assert not gate.degraded
