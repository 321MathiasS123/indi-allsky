"""Rendered feedback cannot override raw safety or chase artificial clipping."""
import numpy as np
import pytest

from indi_allsky.highlight import HighlightMeasurement, HighlightOutput, compensate, measure, measure_rendered
from test_highlight_exposure import MODE_NAMES, controller


SETTINGS = {'OUTPUT_ENABLE': True}
MODE = (1, 0)


def output_state(full=5, any_channel=8):
    state = HighlightOutput()
    state.observe(HighlightMeasurement(full, any_channel, 0), 1., 0, MODE, SETTINGS)
    return state


def request(state, scale=1.1, adu=60, exposure=1., gain=0, mode=MODE, settings=SETTINGS):
    return state.constrain(scale, adu, 80, 10, exposure, gain, mode, settings)


def test_output_has_separate_thresholds_mask_and_connected_areas():
    image = np.zeros((100, 100, 3), np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[:50] = 255
    image[:10, :10] = 240
    image[10:20, :10] = [250, 240, 240]
    image[30:35, 30:35] = 255  # smaller disconnected reflection
    image[60:] = 255  # outside mask
    result = measure_rendered(image, mask)
    assert result.full == 4  # 200 / 5000, near-white can exceed near-clip
    assert result.any == 2
    assert result.full_next is None and result.any_next is None
    assert measure_rendered(image, None) is None
    assert measure_rendered(image, mask[:40]) is None
    assert measure_rendered(image, mask * 0) is None


def test_mono_keeps_distinct_output_brightness_thresholds():
    image = np.full((10, 10), 245, np.uint8)
    assert measure_rendered(image, np.ones_like(image)).full == 100
    assert measure_rendered(image, np.ones_like(image)).any == 0


def test_output_hysteresis_does_not_aim_for_clipping():
    state = output_state(0, 0)
    assert request(state) == (1.1, None)  # untouched ordinary recovery
    state.observe(HighlightMeasurement(1.5, 2.5, 0), 1., 0, MODE, SETTINGS)
    assert not state.active  # in-band entry does not turn on extra processing
    state.observe(HighlightMeasurement(2.4, 2.5, 0), 1., 0, MODE, SETTINGS)
    assert request(state)[0] == pytest.approx(.98)
    state.observe(HighlightMeasurement(1.5, 2.5, 0), 1., 0, MODE, SETTINGS)
    assert state.active and request(state)[0] == 1
    state.observe(HighlightMeasurement(.9, 2.5, 0), 1., 0, MODE, SETTINGS)
    assert state.active  # both lower limits required
    state.observe(HighlightMeasurement(0, 0, 0), 1., 0, MODE, SETTINGS)
    assert not state.active and request(state)[0] == 1.1
    assert request(state, adu=75) == (1.1, None)


def test_output_tapers_and_never_weakens_raw_or_floor_corrections():
    state = output_state(2.1, 2)
    assert request(state)[0] == pytest.approx(.995)
    assert request(state, scale=.85)[0] == .85
    assert request(state, adu=20)[0] == 1.1
    assert request(state, adu=10) == (1.1, None)
    assert request(state, adu=20.5)[0] >= 20 / 20.5


@pytest.mark.parametrize('adu', [19.38, 19.81, 19.95, 40, 60])
@pytest.mark.parametrize('exposure', [1., 1.01])
def test_clear_output_releases_recovery_even_below_normal_adu(adu, exposure):
    state = output_state()
    state.observe(HighlightMeasurement(0, 1.442, 0), 1., 0, MODE, SETTINGS)
    # The 18:30 sunset was already safely below both output lower limits.
    # Its old recovery latch still held or allowed only 1%, until ADU < 19.6.
    assert request(state, adu=adu, exposure=exposure) == (1.1, None)


@pytest.mark.parametrize('feedback', ['matching', 'stale', 'untrusted'])
@pytest.mark.parametrize('max_boost', [0, .05, .5, 2, 4])
def test_output_floor_recovery_is_continuous_and_preserves_raw_limits(feedback, max_boost):
    state = output_state()
    if feedback == 'untrusted':
        state.observe(None, 1, 0, MODE, SETTINGS)
    exposure = .98 if feedback == 'stale' else 1.
    settings = dict(SETTINGS, MAX_BOOST=max_boost)
    floor = 80 / 2 ** max_boost
    adus = np.linspace(floor * .9, floor * 1.2, 301)
    scales = [request(state, adu=adu, exposure=exposure, settings=settings)[0] for adu in adus]
    # A tiny ADU change must not alternate holding and 10% rescue requests.
    assert max(abs(np.diff(scales))) < .002
    assert np.all(np.diff(scales) <= 1e-12)
    for adu in adus[::30]:
        assert request(state, scale=.85, adu=adu, exposure=exposure, settings=settings)[0] == .85
    if feedback != 'matching':
        assert min(scales) >= 1  # invalid/stale output never adds a cut


@pytest.mark.parametrize('delay,fade', [(0, .94), (1, .98), (3, .99)])
@pytest.mark.parametrize('name', MODE_NAMES)
def test_fading_sky_keeps_lift_reserve_despite_persistent_output_clipping(name, delay, fade):
    instance = controller(name)
    instance.config['HIGHLIGHT_PROTECTION'].update(SETTINGS)
    gain = instance.gain_max if name == 'exposure_basic' else instance.gain_min
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = .1, gain
    pending = [(.1, gain)] * (delay + 1)
    rendered = []
    for frame in range(80):
        exposure, gain = pending.pop(0)
        adu = 35 * fade ** frame * exposure / .1
        instance.compare_highlights(HighlightMeasurement(0, 0, adu), exposure, gain)
        pending.append((instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT))
        state = instance.highlight_transition
        state.render_target(adu, 70, 2)
        rendered.append(adu * 2 ** state.lift)
        # Persistent red-channel pressure, as in the observed 19:05 sunset.
        instance.highlight_output.observe(HighlightMeasurement(0, 6.2, 0), exposure, gain, MODE, SETTINGS)
    assert min(rendered[-30:]) >= 70 * .995
    assert max(abs(np.diff(rendered[-30:]))) < .01


def test_stale_settings_hold_recovery_without_compounding_a_cut():
    state = output_state()
    for kwargs in ({'exposure': .98}, {'gain': 1}):
        assert request(state, **kwargs)[0] == 1
        assert request(state, scale=.9, **kwargs)[0] == .9
    state.observe(None, 1, 0, MODE, SETTINGS)  # mixed/repaired stack
    assert state.active
    assert request(state)[0] == 1
    assert request(state, scale=.9)[0] == .9
    assert request(state, adu=10)[0] == 1.1


@pytest.mark.parametrize('kwargs', [{'settings': {}}, {'mode': (0, 0)}])
def test_disabled_or_new_mode_drops_old_feedback(kwargs):
    state = output_state()
    assert request(state, **kwargs) == (1.1, None)
    assert not state.active and state.measurement is None


@pytest.mark.parametrize('name', MODE_NAMES)
def test_output_uses_every_existing_camera_mode_and_preserves_stronger_pending_cut(name):
    instance = controller(name)
    instance.config['HIGHLIGHT_PROTECTION'].update(SETTINGS)
    gain = instance.gain_max if name == 'exposure_basic' else instance.gain_min
    instance._expUtils.EXPOSURE_NEXT = 1
    instance._expUtils.GAIN_NEXT = gain
    instance.highlight_output.observe(HighlightMeasurement(5, 8, 0), 1, gain, MODE, SETTINGS)
    instance.compare_highlights(HighlightMeasurement(0, 0, 60), 1, gain)
    assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(.98, abs=1e-6)
    assert instance.highlight_transition.active
    # Newer raw feedback already requested a stronger reduction.
    instance._expUtils.EXPOSURE_NEXT = .9
    instance.compare_highlights(HighlightMeasurement(0, 0, 60), 1, gain)
    assert instance._expUtils.EXPOSURE_NEXT == .9


@pytest.mark.parametrize('name', MODE_NAMES)
def test_output_at_hardware_floor_does_not_accumulate_debt(name):
    instance = controller(name, night=False)
    instance.config['HIGHLIGHT_PROTECTION'].update(SETTINGS)
    exposure, gain = instance.exposure_min, instance.gain_min
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = exposure, gain
    for _ in range(10):
        instance.highlight_output.observe(HighlightMeasurement(20, 30, 0), exposure, gain, (0, 0), SETTINGS)
        instance.compare_highlights(HighlightMeasurement(0, 0, 80), exposure, gain)
        assert instance._expUtils.EXPOSURE_NEXT == pytest.approx(exposure, abs=1e-6)
        assert instance._expUtils.GAIN_NEXT == gain
    instance.highlight_output.observe(HighlightMeasurement(0, 0, 0), exposure, gain, (0, 0), SETTINGS)
    instance.compare_highlights(HighlightMeasurement(0, 0, 80), exposure, gain)
    assert not instance.highlight_output.active


@pytest.mark.parametrize('delay', [0, 1, 3])
@pytest.mark.parametrize('name', MODE_NAMES)
def test_rendered_cloud_then_clear_scene_recovers_without_forcing_clipping(name, delay):
    instance = controller(name)
    instance.config['HIGHLIGHT_PROTECTION'].update(SETTINGS)
    gain = instance.gain_max if name == 'exposure_basic' else instance.gain_min
    instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT = 1, gain
    pending = [(1., gain)] * (delay + 1)
    scene = np.full((80, 80, 3), .20)
    y, x = np.mgrid[:80, :80]
    mask = np.ones((80, 80), np.uint8)
    commanded, patches = [], []
    for frame in range(900):
        e, g = pending.pop(0)
        scene[:] = .20
        if frame < 350:
            scene += .75 * np.exp(-((x - 40) ** 2 + (y - 40) ** 2) / 180)[..., None]
        raw = np.minimum(scene * e * 65535, 65535).astype(np.uint16)
        m = measure(raw, mask, 16)
        instance.compare_highlights(m, e, g)
        next_settings = instance._expUtils.EXPOSURE_NEXT, instance._expUtils.GAIN_NEXT
        pending.append(next_settings)
        commanded.append(next_settings[0])
        state = instance.highlight_transition
        target = state.render_target(m.adu, 70, 2)
        rendered = compensate(raw, 16, m.adu, target, 2) / 65535
        # White balance creates output clipping while raw patches stay small.
        rendered[:, :, 2] *= 1.4
        rendered = (np.minimum(rendered, 1) ** (1 / state.gamma(1.5, 1.85)) * 255).astype(np.uint8)
        output = measure_rendered(rendered, mask)
        instance.highlight_output.observe(output, e, g, MODE, SETTINGS)
        patches.append(output.any)
        assert instance.exposure_min <= next_settings[0] <= instance.exposure_max
        if frame > 860:
            assert output.full == output.any == 0
            assert state.phase == 'normal'
    assert max(patches[:350]) > 3
    assert max(patches[300:350]) <= 3.1
    assert max(abs(np.diff(commanded[300:350]))) < .005
    assert commanded[-1] > min(commanded)
