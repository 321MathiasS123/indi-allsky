"""Fresh capture control ahead of the ordered, potentially slower renderer."""
import copy
from datetime import datetime
import logging
import math
from multiprocessing import Process
from pathlib import Path
import queue
import signal
import time
import traceback

from . import constants
from .highlight import HighlightMeasurement


logger = logging.getLogger('indi_allsky')


def capture_cadence(job, config):
    mode = job.get('capture_mode', (False, False))
    period = job.get('capture_period', config.get(
        'EXPOSURE_PERIOD' if mode[constants.NIGHT_NIGHT] else 'EXPOSURE_PERIOD_DAY', 15))
    return max(float(period), float(job['exposure']), 0.001)


def apply_control_snapshot(controller, snapshot):
    """Apply capture decisions without rewinding the renderer's lift/gamma."""
    transition = controller.highlight_transition
    state = snapshot['transition']
    if state.get('reset'):
        transition.reset()
    else:
        if state['active'] and not transition.active:
            transition.reference = None
        transition.active = state['active']
        transition.trusted = state['trusted']
        transition.reason = state['reason']
        if state.get('seed'):
            transition.reference = state['target']
            transition.gamma_mix = 1.0
        transition._startup = False
    controller._target_adu_found = snapshot['stable']
    controller._current_adu_target = snapshot['current_adu_target']


class OutputFeedbackGate:
    """Expire rendered feedback by capture cadence; require two fresh returns.

    A renderer behind the capture stream must not hold recovery indefinitely.
    Matching settings alone are insufficient: those settings can persist for
    minutes. Two successive usable outputs prevent flapping at the age limit.
    """

    def __init__(self):
        self.latest = None
        self.degraded = False
        self.fresh_count = 0
        self.last_accepted = None

    def update(self, controller, job, feedback, allowance=0.0):
        if feedback is not None and (
                self.latest is None or feedback['exp_time'] > self.latest['exp_time']):
            self.latest = feedback
        settings = controller.config.get('HIGHLIGHT_PROTECTION', {})
        output = controller.highlight_output
        candidate = self.latest
        mode = tuple(job['capture_mode'])
        window = capture_cadence(job, controller.config) + max(0.0, allowance)
        usable = (
            settings.get('OUTPUT_ENABLE', False)
            and candidate is not None and candidate.get('trusted', False)
            and candidate.get('measurement') is not None
            and candidate['camera_id'] == job['camera_id']
            and candidate['binning'] == job['binning']
            and tuple(candidate['capture_mode']) == mode
            and 0 <= job['exp_time'] - candidate['exp_time'] <= window
            and math.isclose(job['exposure'], candidate['exposure'], rel_tol=0, abs_tol=0.0000005)
            and math.isclose(job['gain'], candidate['gain'], rel_tol=0, abs_tol=0.0005)
        )
        if not usable:
            output.reset()
            self.degraded = True
            self.fresh_count = 0
            return False
        if candidate['exp_time'] != self.last_accepted:
            self.last_accepted = candidate['exp_time']
            self.fresh_count += 1
        if self.degraded and self.fresh_count < 2:
            output.reset()
            return False
        self.degraded = False
        output.observe(HighlightMeasurement(*candidate['measurement']),
                       candidate['exposure'], candidate['gain'], mode, settings)
        return True


class HighlightMeterWorker(Process):
    """Own capture commands; forward every original file for ordinary rendering.

    Calibration here is a disposable measurement view. The renderer retains
    ownership of original/diagnostic FITS saving, hooks and its ordered stack.
    It receives an explicit result even when metering fails, so an old render
    cannot issue a second or conflicting exposure request.
    """

    def __init__(self, idx, config, error_q, input_q, image_q, output_feedback_q,
                 position_av, exposure_av, gain_av, binning_av, sensors_temp_av,
                 sensors_user_av, night_av, astro_av, backlog_state=None):
        super().__init__()
        self.name = 'HighlightMeter-{0:d}'.format(idx)
        self.config = copy.deepcopy(config)
        self.error_q = error_q
        self.input_q = input_q
        self.image_q = image_q
        self.output_feedback_q = output_feedback_q
        self.position_av = position_av
        self.exposure_av = exposure_av
        self.gain_av = gain_av
        self.binning_av = binning_av
        self.sensors_temp_av = sensors_temp_av
        self.sensors_user_av = sensors_user_av
        self.night_av = night_av
        self.astro_av = astro_av
        self.backlog_state = backlog_state

    def initialize(self):
        from . import exposure as exposure_module
        from .processing import ImageProcessor

        # Capture mode is local to this frame; shared camera state can advance
        # while the renderer still processes an earlier twilight/night frame.
        self.frame_mode = list(self.night_av)
        self.frame_temperature = list(self.sensors_temp_av)
        self.processor = ImageProcessor(
            self.config, self.position_av, self.exposure_av, self.gain_av,
            self.binning_av, self.frame_temperature, self.sensors_user_av,
            self.frame_mode, self.astro_av)
        self.processor.stack_count = 1
        # On installations with a per-frame settings blend, add() updates this
        # same configuration using the exposure midpoint before metering.
        self.config = self.processor.config
        name = self.config.get('CCD_CONFIG', {}).get('EXPOSURE_CLASSNAME', 'exposure_basic')
        exposure_class = getattr(exposure_module, name, exposure_module.exposure_basic)
        self.controller = exposure_class(self.config, self.exposure_av, self.gain_av,
                                         self.binning_av, self.frame_mode)
        self.feedback_gate = OutputFeedbackGate()

    def run(self):
        from .flask import create_app

        # Fork inherits the supervisor's handlers. Forced shutdown must really
        # terminate this child; normal reload drains via its queued stop marker.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            self.initialize()
            application = create_app()
            self.saferun(application)
        except Exception as exc:
            self.error_q.put((str(exc), traceback.format_exc()))
            raise

    def saferun(self, application):
        while True:
            job = self.input_q.get()
            if job.get('stop'):
                # The supervisor stops the renderer after this worker has
                # forwarded its complete prefix of captures and barriers.
                return
            if job.get('period_end') or job.get('sqm_exposure'):
                self.image_q.put(job)
                continue
            with application.app_context():
                self.forward(job)

    def forward(self, job):
        """No failure in this optional control path may consume the original."""
        started = time.monotonic()
        transition = self.controller.highlight_transition
        transition.trusted = False
        seed = False
        reset = False
        measurement = None
        status = 'held'
        if getattr(self, 'backlog_state', None) is not None:
            try:
                self.backlog_state.observe_file(job['filename'])
            except OSError:
                logger.exception('Unable to inspect retained frame storage')
        mode = tuple(job.get('capture_mode', tuple(self.night_av)))
        job['capture_mode'] = mode
        self.frame_mode[:] = mode
        if hasattr(self, 'frame_temperature'):
            self.frame_temperature[:] = self.sensors_temp_av
            if 'capture_temperature' in job:
                self.frame_temperature[constants.SENSOR_TEMP_CCD_TEMP] = job['capture_temperature']
        try:
            if mode != tuple(self.night_av):
                status = 'old capture mode'
            elif time.time() - job['exp_time'] > capture_cadence(job, self.config):
                status = 'old capture'
            else:
                measurement, status, seed, reset = self.meter(job, started)
        except Exception:
            logger.exception('Highlight metering failed; retaining frame and holding capture settings')
            self.controller.reset_highlights()
            self.controller.highlight_output.reset()
            transition.trusted = False
            status = 'metering failed'
        finally:
            # No pixel arrays travel through IPC. Release the disposable raw
            # and debayered measurement view while the renderer owns its copy.
            processor = getattr(self, 'processor', None)
            for ref in getattr(processor, 'image_list', ()):
                if ref is not None:
                    try:
                        ref.hdulist.close()
                    except Exception:
                        logger.exception('Unable to close disposable highlight measurement')
            if processor is not None:
                processor.image_list.clear()
        snapshot = {
            'status': status,
            'measurement': tuple(measurement) if measurement is not None else None,
            'adu_average': measurement.adu if measurement is not None else 0.0,
            'stable': self.controller.target_adu_found,
            'current_adu_target': self.controller.current_adu_target,
            'transition': {
                'active': transition.active, 'trusted': transition.trusted,
                'reason': transition.reason, 'seed': seed, 'reset': reset,
                'target': self.config['TARGET_ADU' if mode[constants.NIGHT_NIGHT] else 'TARGET_ADU_DAY'],
            },
        }
        job['highlight_control'] = snapshot
        self.image_q.put(job)
        logger.info('Fresh highlight control: %s; frame %s; metering %.4fs',
                    status, datetime.fromtimestamp(job['exp_time']).isoformat(), time.monotonic() - started)

    def meter(self, job, started):
        from . import asi676mc
        from .flask.models import IndiAllSkyDbCameraTable

        camera = IndiAllSkyDbCameraTable.query.filter(
            IndiAllSkyDbCameraTable.id == job['camera_id']).one()
        camera_data = camera.data or {}
        settings = self.config.get('HIGHLIGHT_PROTECTION', {})
        if (not settings.get('ENABLE', False) or self.processor.focus_mode
                or camera_data.get('exposure_control') is False):
            self.controller.highlight_transition.reset()
            self.controller.highlight_output.reset()
            return None, 'inactive', False, True
        self.controller.gain_values = camera_data.get('gain_values', [])
        self.controller.gain_quantum = camera_data.get('gain_quantum', 0.0)
        filename = Path(job['filename'])
        if self.config['CAMERA_INTERFACE'].startswith(('libcamera_', 'mqtt_')):
            self.processor.libcamera_raw = filename.suffix == '.dng'
        self.processor.update_astrometric_data(datetime.fromtimestamp(job['exp_time']))
        ref = self.processor.add(
            filename, job['exposure'], job['gain'], job['binning'],
            datetime.fromtimestamp(job['exp_time']), job['exp_elapsed'], camera,
            detected_camera_name=job.get('camera_name'))
        self.processor.correct_asi676mc_frame(ref)
        if ((ref.asi676mc_repair_result or {}).get('status') == 'repaired'
                or asi676mc.excluded_from_downstream_measurements(ref.asi676mc_repair_result)):
            self.controller.reset_highlights()
            self.controller.highlight_output.reset()
            return None, 'untrusted capture', False, False
        measurement = self.processor.measure_highlights()
        if measurement is None:
            self.controller.highlight_output.reset()
            return None, 'unavailable mask', False, False
        black_level = ref.libcamera_black_level or job.get('libcamera_black_level', 0)
        self.processor.calibrate(libcamera_black_level=black_level)
        self.processor.fix_holes_early()
        self.processor.debayer()
        measurement = self.processor.calibrate_highlights(measurement)
        if not all(math.isfinite(value) for value in measurement):
            raise ValueError('Nonfinite highlight measurement')
        latest = None
        while self.output_feedback_q is not None:
            try:
                feedback = self.output_feedback_q.get_nowait()
                if latest is None or feedback['exp_time'] > latest['exp_time']:
                    latest = feedback
            except queue.Empty:
                break
        # Mode can change during metering. Never apply an older mode's request
        # to a camera that has already switched its limits and capture settings.
        if tuple(job['capture_mode']) != tuple(self.night_av):
            self.controller.highlight_output.reset()
            return measurement, 'old capture mode', False, False
        if time.time() - job['exp_time'] > capture_cadence(job, self.config):
            self.controller.highlight_output.reset()
            return measurement, 'old capture', False, False
        self.feedback_gate.update(self.controller, job, latest, time.monotonic() - started)
        transition = self.controller.highlight_transition
        startup = transition._startup
        self.controller.compare_highlights(measurement, job['exposure'], job['gain'])
        # Optional independently maintained twilight feature clamps the moving
        # limits here, never later from an obsolete rendered frame.
        apply_limits = getattr(self.controller, 'apply_transition_limits', None)
        if apply_limits is not None:
            apply_limits()
        return measurement, 'metered', startup and transition.active, False
