from threading import Event, Lock, Thread, current_thread


class ManagedStageMove:
    def __init__(self, blocked=True):
        self._lock = Lock()
        self._cancel = Event()
        self._thread = None
        self._stage = None
        self._blocked = blocked

    @property
    def thread(self):
        with self._lock:
            return self._thread

    def start(self, stage, target, on_success=None, teardown_requested=None):
        if teardown_requested is None:
            teardown_requested = lambda: False

        with self._lock:
            if self._blocked or stage is None or teardown_requested():
                return False
            if self._thread is not None and self._thread.is_alive():
                return False

            cancel = Event()
            self._cancel = cancel
            self._stage = stage

            def move():
                try:
                    moved = stage.move_abs(target, 'mm', wait_until_idle=True)
                    if moved and not cancel.is_set() \
                            and not teardown_requested() \
                            and on_success is not None:
                        on_success()
                finally:
                    with self._lock:
                        if self._thread is current_thread():
                            self._stage = None

            self._thread = Thread(
                target=move, daemon=True, name='GoToMovement'
            )
            self._thread.start()
            return True

    def request_stop(self, block_new=False):
        with self._lock:
            if block_new:
                self._blocked = True
            self._cancel.set()
            thread = self._thread
            stage = self._stage
        if thread is not None and thread.is_alive() and stage is not None:
            stage.emergency_stop()

    def wait(self, timeout=None):
        with self._lock:
            thread = self._thread
        if thread is None or thread is current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def is_active(self):
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def allow(self):
        with self._lock:
            self._blocked = False


def controller_velocity(value, fast_speed, slow_speed):
    velocity = fast_speed * value / 32767
    if abs(velocity) < slow_speed * 0.01:
        return None
    return velocity


def append_new_focus_values(focus_log, graph_x, graph_y, lock):
    with lock:
        start = len(graph_y)
        end = len(focus_log)
        if start >= end:
            return 0
        graph_x.extend(range(start, end))
        graph_y.extend(focus_log[start:end])
        return end - start
