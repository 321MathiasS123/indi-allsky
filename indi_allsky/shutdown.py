"""Bound shutdown waits without changing ordinary worker reload behavior."""
import logging
import time

logger = logging.getLogger('indi_allsky')


def join_worker(worker, deadline=None):
    if deadline is None:
        worker.join()
        return True
    worker.join(timeout=max(0, deadline - time.monotonic()))
    if not worker.is_alive():
        return getattr(worker, 'exitcode', 0) in (None, 0)
    logger.warning('Shutdown grace expired for %s', worker.name)
    # Processes can be terminated; a network thread cannot safely be killed.
    # Leave that thread to the service manager's final cgroup deadline.
    if hasattr(worker, 'terminate'):
        worker.terminate()
        worker.join(timeout=2)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
    return False
