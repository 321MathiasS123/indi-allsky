"""Describe the local outputs of one automatic day/night generation batch."""

from datetime import datetime, timezone
from pathlib import Path


def record_media(task, outputs):
    """Save exact output paths; a missing optional entry means it was skipped."""
    media = {}
    for name, (entry, path) in outputs.items():
        media[name] = {
            'status': ('success' if entry.success else 'failed') if entry else 'skipped',
            'path': str(path) if entry else None,
        }
    task.data = dict(task.data, local_media=media)


def completion_payload(task, dependencies):
    """Never infer success from the final encoder or task state alone."""
    outputs = {}
    failed_tasks = []
    for task_id, dependency in dependencies.items():
        if dependency is None or dependency.state.value != 'Success':
            failed_tasks.append(task_id)
        if dependency is None:
            continue
        media = dependency.data.get('local_media', {})
        if not media and task_id not in failed_tasks:
            failed_tasks.append(task_id)
        for name, item in media.items():
            status = item['status']
            if status == 'success':
                try:
                    if not Path(item['path']).is_file() or Path(item['path']).stat().st_size == 0:
                        status = 'failed'
                except OSError:
                    status = 'failed'
            outputs[name] = status

    kwargs = task.data['kwargs']
    success = not failed_tasks and bool(outputs) and all(
        status in ('success', 'skipped') for status in outputs.values()
    )
    return {
        'event': 'timelapse_complete',
        'event_id': 'timelapse-{0}'.format(task.id),
        'camera_id': kwargs['camera_id'],
        'date': datetime.strptime(kwargs['timespec'], '%Y%m%d').date().isoformat(),
        'period': 'night' if kwargs['night'] else 'day',
        'status': 'success' if success else 'failed',
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'outputs': outputs,
        'failed_task_ids': failed_tasks,
    }
