"""Display continuity across renderer replacement without reviving capture control."""
from copy import deepcopy
import ast
import ctypes
import hashlib
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
import textwrap

import pytest

from indi_allsky.highlight import HighlightOutput, HighlightRenderHistory, HighlightTransition
from indi_allsky.highlight_meter import apply_control_snapshot


MODE = (False, False)
SHAPE = (3552, 3552)


def config():
    return {'TARGET_ADU_DAY': 80, 'TARGET_ADU': 70,
            'GAMMA_CORRECTION_DAY': 1.565, 'EXPOSURE_PERIOD_DAY': 20,
            'HIGHLIGHT_PROTECTION': {'ENABLE': True, 'MAX_BOOST': 2, 'GAMMA_DAY': 1.85},
            'IMAGE_STRETCH': {'MODE2_SHADOWS': .03},
            'TWILIGHT_TRANSITION': {'ENABLE': True, 'DAY_ALT': -3, 'NIGHT_ALT': -9}}


def job(**changes):
    return dict({'camera_id': 1, 'binning': 1, 'exposure': .000286,
                 'gain': 0, 'exp_time': 1000., 'capture_period': 20}, **changes)


def protected():
    transition = HighlightTransition()
    transition.active = transition.trusted = True
    transition.reference = 80.
    transition.lift = 1.643
    transition.gamma_mix = .85
    return transition


def display(transition):
    return transition.active, transition.reference, transition.lift, transition.gamma_mix


def shared_history():
    shared = multiprocessing.RawArray('c', 2048)
    HighlightRenderHistory(config(), shared).remember(protected(), job(), MODE, SHAPE)
    return shared


def snapshot(trusted=False, seed=False):
    return {'stable': False, 'current_adu_target': 21,
            'transition': {'active': trusted, 'trusted': trusted, 'reason': 'raw decision',
                           'seed': seed, 'target': 80.}}


def remember_in_worker(shared):
    HighlightRenderHistory(config(), shared).remember(protected(), job(), MODE, SHAPE)


def test_actual_shared_buffer_survives_renderer_process_exit():
    context = multiprocessing.get_context('spawn')
    shared = context.RawArray('c', 2048)
    worker = context.Process(target=remember_in_worker, args=(shared,))
    worker.start()
    worker.join(20)
    if worker.is_alive():
        worker.terminate()
        worker.join(5)
        pytest.fail('Renderer handoff process did not complete')
    assert worker.exitcode == 0
    restored = HighlightTransition()
    assert HighlightRenderHistory(config(), shared).restore(restored, job(exp_time=1049), MODE, SHAPE)
    assert display(restored) == display(protected())
    assert not restored.trusted


def test_first_repaired_snapshot_retains_display_without_restoring_capture_commands():
    render = SimpleNamespace(highlight_transition=HighlightTransition(),
                             exposure_next=.000286, gain_next=0, hist_adu=[])
    assert HighlightRenderHistory(config(), shared_history()).restore(
        render.highlight_transition, job(exp_time=1049), MODE, SHAPE)
    apply_control_snapshot(render, snapshot())
    assert display(render.highlight_transition) == display(protected())
    assert not render.highlight_transition.trusted
    assert (render.exposure_next, render.gain_next, render.hist_adu) == (.000286, 0, [])
    assert set(vars(render)) == {'highlight_transition', 'exposure_next', 'gain_next',
                                'hist_adu', '_target_adu_found', '_current_adu_target'}
    # The next healthy startup decision cannot snap the restored gamma to 1.
    apply_control_snapshot(render, snapshot(trusted=True, seed=True))
    assert display(render.highlight_transition) == display(protected())
    assert render.highlight_transition.trusted


@pytest.mark.parametrize('shared', [None, b'', b'interrupted write'])
def test_no_history_allows_trusted_startup_seed_after_initial_repaired_frame(shared):
    buffer = None if shared is None else SimpleNamespace(value=shared)
    render = SimpleNamespace(highlight_transition=HighlightTransition())
    assert not HighlightRenderHistory(config(), buffer).restore(
        render.highlight_transition, job(), MODE, SHAPE)
    apply_control_snapshot(render, snapshot())
    assert render.highlight_transition._startup
    assert display(render.highlight_transition) == (False, None, 0, 0)
    apply_control_snapshot(render, snapshot(trusted=True, seed=True))
    assert render.highlight_transition.reference == 80
    assert render.highlight_transition.gamma_mix == 1
    assert not render.highlight_transition._startup


@pytest.mark.parametrize('changes,mode,shape', [
    ({'exp_time': 1120.01}, MODE, SHAPE),
    ({'exp_time': 999.99}, MODE, SHAPE),
    ({'camera_id': 2}, MODE, SHAPE),
    ({'binning': 2}, MODE, SHAPE),
    ({}, (True, False), SHAPE),
    ({}, (False, True), SHAPE),
    ({}, MODE, (1776, 1776)),
    ({}, MODE, (3, 3552, 3552)),
])
def test_stale_future_or_incompatible_frame_cannot_reuse_display(changes, mode, shape):
    restored = HighlightTransition()
    assert not HighlightRenderHistory(config(), shared_history()).restore(
        restored, job(**changes), mode, shape)
    assert vars(restored) == vars(HighlightTransition())


@pytest.mark.parametrize('period,exposure,age,accepted', [
    (20, .000286, 120, True), (60, 30, 180, True), (60, 30, 181, False),
    (20, 60, 180, True), (120, 30, 300, True), (120, 30, 301, False),
])
def test_freshness_scales_with_cadence_but_is_bounded(period, exposure, age, accepted):
    assert HighlightRenderHistory(config(), shared_history()).restore(
        HighlightTransition(), job(exp_time=1000 + age, capture_period=period, exposure=exposure),
        MODE, SHAPE) is accepted


@pytest.mark.parametrize('key,value', [
    ('TARGET_ADU_DAY', 90), ('GAMMA_CORRECTION_DAY', 1.8), ('CCD_BIT_DEPTH', 12),
    ('IMAGE_STRETCH', {'MODE2_SHADOWS': .04}), ('IMAGE_CALIBRATE_DARK', True),
    ('TWILIGHT_TRANSITION', {'ENABLE': False}),
    ('HIGHLIGHT_PROTECTION', {'ENABLE': True, 'MAX_BOOST': 3, 'GAMMA_DAY': 1.85}),
])
def test_brightness_or_calibration_config_change_rejects_old_display(key, value):
    settings = config()
    settings[key] = value
    assert not HighlightRenderHistory(settings, shared_history()).restore(
        HighlightTransition(), job(exp_time=1020), MODE, SHAPE)


@pytest.mark.parametrize('change', [{'HIGHLIGHT_PROTECTION': {'ENABLE': False}}, {'FOCUS_MODE': True}])
def test_disabling_highlights_or_enabling_focus_clears_shared_history(change):
    shared = shared_history()
    settings = config()
    settings.update(change)
    HighlightRenderHistory(settings, shared)
    assert shared.value == b''
    assert not HighlightRenderHistory(config(), shared).restore(
        HighlightTransition(), job(exp_time=1020), MODE, SHAPE)


def test_enabling_display_only_fringe_option_retains_recent_brightness():
    settings = config()
    settings['HIGHLIGHT_PROTECTION']['FRINGE_REDUCTION'] = True
    assert HighlightRenderHistory(settings, shared_history()).restore(
        HighlightTransition(), job(exp_time=1020), MODE, SHAPE)


def test_untrusted_frames_never_refresh_retention_window():
    shared = shared_history()
    original = shared.value
    history = HighlightRenderHistory(config(), shared)
    restored = HighlightTransition()
    assert history.restore(restored, job(exp_time=1049), MODE, SHAPE)
    for timestamp in (1050, 1070, 1090, 1110):
        restored.render_target(27, 80, 2)
        restored.gamma(1.565, 1.85)
        history.remember(restored, job(exp_time=timestamp), MODE, SHAPE)
    assert shared.value == original
    assert not HighlightRenderHistory(config(), shared).restore(
        HighlightTransition(), job(exp_time=1121), MODE, SHAPE)


@pytest.mark.parametrize('payload', [
    b'not json', b'[]', b'{}', b'null', b'\xff',
])
def test_malformed_checksummed_payload_is_ignored(payload):
    shared = SimpleNamespace(value=hashlib.sha256(payload).hexdigest().encode() + b'\n' + payload)
    assert not HighlightRenderHistory(config(), shared).restore(HighlightTransition(), job(), MODE, SHAPE)


@pytest.mark.parametrize('field,value', [
    ('time', float('nan')), ('time', float('inf')),
    ('display', [1, 80, 1, 1]), ('display', [True, -1, 1, 1]),
    ('display', [True, float('nan'), 1, 1]), ('display', [True, 80, float('inf'), 1]),
    ('display', [True, 80, -1, 1]), ('display', [True, 80, 1, 1.1]),
    ('display', [True, 80, 1, float('nan')]), ('display', [True, 80, 1]),
])
def test_invalid_display_values_are_ignored(field, value):
    shared = shared_history()
    saved = json.loads(shared.value.split(b'\n', 1)[1])
    saved[field] = value
    payload = json.dumps(saved).encode()
    shared.value = hashlib.sha256(payload).hexdigest().encode() + b'\n' + payload
    assert not HighlightRenderHistory(config(), shared).restore(HighlightTransition(), job(), MODE, SHAPE)


def test_checksum_rejects_interrupted_or_corrupted_write():
    shared = shared_history()
    shared.value = shared.value.replace(b'1.643', b'1.644')
    assert not HighlightRenderHistory(config(), shared).restore(HighlightTransition(), job(), MODE, SHAPE)


def test_config_identity_uses_original_endpoints_before_optional_twilight_interpolation():
    settings = config()
    original = deepcopy(settings)
    shared = multiprocessing.RawArray('c', 2048)
    history = HighlightRenderHistory(settings, shared)
    settings['TARGET_ADU_DAY'] = 75
    settings['GAMMA_CORRECTION_DAY'] = 1.3
    settings['HIGHLIGHT_PROTECTION']['GAMMA_DAY'] = 1.6
    history.remember(protected(), job(), MODE, SHAPE)
    assert HighlightRenderHistory(original, shared).restore(
        HighlightTransition(), job(exp_time=1020), MODE, SHAPE)


def test_worker_restores_before_capture_snapshot_and_remembers_after_gamma():
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    start = source.index('        # Meter clipping before dark/black-level')
    prepare = textwrap.dedent(source[start:source.index('        highlight_repaired =', start)])
    start = source.index('        # gamma correction')
    finish = textwrap.dedent(source[start:source.index('        # sharpening', start)])
    shared = shared_history()
    history = HighlightRenderHistory(config(), shared)
    controller = SimpleNamespace(highlight_transition=HighlightTransition())
    processor = SimpleNamespace(focus_mode=False)
    worker = SimpleNamespace(config=config(), exposure_o=controller, image_processor=processor,
                             night_av=MODE, highlight_render_history=history, _highlight_control=snapshot())
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=SimpleNamespace(shape=SHAPE))])
    namespace = dict(self=worker, i_dict=job(exp_time=1049), i_ref=reference,
                     apply_control_snapshot=apply_control_snapshot)
    exec(prepare, namespace)
    assert display(controller.highlight_transition) == display(protected())
    assert not controller.highlight_transition.trusted
    worker._highlight_control = snapshot(trusted=True)
    exec(prepare, namespace)
    def gamma():
        controller.highlight_transition.gamma_mix = .95
    processor.apply_gamma_correction = gamma
    exec(finish, namespace)
    restored = HighlightTransition()
    assert HighlightRenderHistory(config(), shared).restore(restored, job(exp_time=1069), MODE, SHAPE)
    assert restored.gamma_mix == .95


@pytest.mark.parametrize('disabled', ['highlight', 'focus', 'capture_reset'])
def test_worker_clears_retained_display_on_runtime_disable_or_capture_reset(disabled):
    source = (Path(__file__).resolve().parents[2] / 'indi_allsky/image.py').read_text(encoding='utf-8')
    start = source.index('        # Meter clipping before dark/black-level')
    prepare = textwrap.dedent(source[start:source.index('        highlight_repaired =', start)])
    shared = shared_history()
    settings = config()
    history = HighlightRenderHistory(settings, shared)
    controller = SimpleNamespace(highlight_transition=protected(), highlight_output=HighlightOutput())
    worker = SimpleNamespace(config=settings, exposure_o=controller, night_av=MODE,
                             image_processor=SimpleNamespace(focus_mode=disabled == 'focus'),
                             highlight_render_history=history)
    if disabled == 'highlight':
        settings['HIGHLIGHT_PROTECTION']['ENABLE'] = False
    if disabled == 'capture_reset':
        worker._highlight_control = snapshot()
        worker._highlight_control['transition']['reset'] = True
    reference = SimpleNamespace(hdulist=[SimpleNamespace(data=SimpleNamespace(shape=SHAPE))])
    exec(prepare, dict(self=worker, i_dict=job(exp_time=1049), i_ref=reference,
                       apply_control_snapshot=apply_control_snapshot))
    assert shared.value == b''
    assert display(controller.highlight_transition) == (False, None, 0, 0)
    assert not HighlightRenderHistory(config(), shared).restore(
        HighlightTransition(), job(exp_time=1069), MODE, SHAPE)


def test_supervisor_passes_same_display_buffer_to_replacement_workers():
    root = Path(__file__).resolve().parents[2] / 'indi_allsky'
    parent = ast.parse((root / 'allsky.py').read_text(encoding='utf-8'))
    parent_class = next(node for node in parent.body if isinstance(node, ast.ClassDef) and node.name == 'IndiAllSky')
    allocate = next(node for node in ast.walk(parent_class) if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Attribute) and target.attr == 'highlight_render_state'
                            for target in node.targets))
    launch = next(node for node in parent_class.body if isinstance(node, ast.FunctionDef) and node.name == '_startImageWorker')
    construct = next(node for node in launch.body if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == 'ImageWorker')
    worker_tree = ast.parse((root / 'image.py').read_text(encoding='utf-8'))
    worker_class = next(node for node in worker_tree.body if isinstance(node, ast.ClassDef) and node.name == 'ImageWorker')
    initialize = next(node for node in worker_class.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    # Run the actual constructor through history creation, before queues,
    # camera services and DB-backed image processing need external fixtures.
    history_index = next(index for index, node in enumerate(initialize.body)
                         if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Attribute) and target.attr == 'highlight_render_history'
                                 for target in node.targets))
    initialize.body = initialize.body[:history_index + 1]
    worker_class.bases, worker_class.body = [], [initialize]
    namespace = dict(HighlightRenderHistory=HighlightRenderHistory)
    exec(compile(ast.Module(body=[worker_class], type_ignores=[]), '<renderer constructor>', 'exec'), namespace)
    supervisor = SimpleNamespace(**{name: None for name in (
        'image_error_q', 'image_q', 'upload_q', 'position_av', 'exposure_av', 'gain_av', 'binning_av',
        'sensors_temp_av', 'sensors_user_av', 'night_av', 'astro_av', 'video_q', 'period_inflight',
        'period_sequence', 'highlight_feedback_q', 'render_backlog', 'processing_allowance')})
    supervisor.config, supervisor.image_worker_idx = config(), 1
    supervisor._highlightMeterEnabled = lambda: True
    namespace.update(self=supervisor, Array=multiprocessing.Array, ctypes=ctypes)
    exec(compile(ast.Module(body=[allocate], type_ignores=[]), '<supervisor state>', 'exec'), namespace)
    exec(compile(ast.Module(body=[construct], type_ignores=[]), '<start renderer>', 'exec'), namespace)
    supervisor.image_worker.highlight_render_history.remember(protected(), job(), MODE, SHAPE)
    supervisor.image_worker_idx += 1
    exec(compile(ast.Module(body=[construct], type_ignores=[]), '<replace renderer>', 'exec'), namespace)
    restored = HighlightTransition()
    assert supervisor.image_worker.highlight_render_history.shared is supervisor.highlight_render_state
    assert supervisor.image_worker.highlight_render_history.restore(restored, job(exp_time=1049), MODE, SHAPE)
    assert display(restored) == display(protected())
