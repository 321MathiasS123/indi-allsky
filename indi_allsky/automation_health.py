"""Expose the Latest page's capture health with additional recovery safeguards."""
import math
from pathlib import Path

from .automation import ControlError


def capture_health(config, maintenance=False):
    import ephem
    from .flask import db, models
    try:
        from .capture_health import local_capture_health
    except ImportError as exc:
        raise ControlError('capture_health_not_installed', 501) from exc

    def state(key, default=None):
        row = db.session.get(models.IndiAllSkyDbStateTable, key, populate_existing=True)
        return row.value if row else default

    camera_id = int(state('DB_CAMERA_ID', 0))
    camera = db.session.get(models.IndiAllSkyDbCameraTable, camera_id)
    if not camera:
        return dict(expected=False, stale=False, last_capture=None, health=None)
    observer = ephem.Observer()
    observer.lat = math.radians(camera.latitude or 0)
    observer.lon = math.radians(camera.longitude or 0)
    night = float(ephem.Sun(observer).alt) < math.radians(
        camera.nightSunAlt if camera.nightSunAlt is not None else -6)
    health = local_capture_health(config, camera, night)
    # Dark-library capture intentionally suspends normal sky images. This query
    # is harmless when that optional feature is not installed.
    dark_task = models.IndiAllSkyDbTaskQueueTable.query.filter(
        models.IndiAllSkyDbTaskQueueTable.state.in_((models.TaskQueueState.MANUAL,
            models.TaskQueueState.QUEUED, models.TaskQueueState.RUNNING)),
        models.IndiAllSkyDbTaskQueueTable.data['action'].as_string() == 'dark_automation',
    ).first()
    expected = health['capture_expected'] and not maintenance and dark_task is None
    recovery_allowed = bool(camera.local and not camera.capture_pause and
                            (night or camera.daytime_capture) and dark_task is None)
    # The page warns promptly; automatic recovery retains HA's eight-minute
    # grace and extends it when the shared health calculation allows long frames.
    threshold = max(480, health['stale_seconds'])
    uptime = float(Path('/proc/uptime').read_text().split()[0])
    age = health['last_success_age_s']
    stale = bool(expected and health['status'] == 'error' and uptime > threshold
                 and age is not None and age > threshold)
    return dict(expected=bool(expected), stale=stale,
                recovery_allowed=recovery_allowed,
                last_capture=health['last_success_timestamp'],
                age_seconds=age, stale_seconds=threshold, camera_id=camera_id,
                health=health, dark_capture_active=dark_task is not None)
