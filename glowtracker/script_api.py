"""User-plugin API: load a participant's Python file and call it once per camera frame.

The plugin receives a read-only :class:`WormState` snapshot and a :class:`Scope` handle that can
set the DAQ voltage, fetch the latest frame, log decisions and (when the tracker is idle) move
the stage. The app side injects small callables into :class:`PluginHost`, so this module has no
Kivy or hardware imports and can be unit tested on its own.

Plugin file contract (either form works)::

    class Controller:
        def setup(self, scope): ...            # optional, once when started
        def update(self, state, scope): ...    # required, once per frame
        def teardown(self, scope): ...         # optional, once when stopped

    # or simply
    def update(state, scope): ...
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True)
class BrightnessStats:
    """Live-analysis image statistics, as shown in the app's live analysis label.

    Computed on the same image the tracker sees (main side in dual-colour mode), cropped to the
    tracking region when the Live analysis "region mode" is set to Tracking. Median, skewness and
    the percentiles are computed on a 4x subsampled image.
    """
    min: float
    max: float
    mean: float
    median: float
    skewness: float
    percentile_5: float
    percentile_95: float


@dataclass(frozen=True)
class WormState:
    """Snapshot of what the app knows about the animal when a frame arrived.

    All positions are stage coordinates in millimetres.
    """
    frame: int                          # frame counter of the current acquisition
    time_s: float                       # seconds since the acquisition (live view / recording) started
    wall_time: float                    # time.time() when the snapshot was built
    stage_xy: tuple[float, float]       # stage position (mm)
    worm_xy: tuple[float, float]        # stage position + in-frame centroid offset (mm); == stage_xy when not tracking
    cms_offset_px: tuple[float, float]  # centroid offset from image centre in pixels (x right, y up); (0, 0) when not tracking
    trail: np.ndarray                   # N x 2 history of stage positions (mm), oldest first; empty when not tracking
    velocity: tuple[float, float]       # last trail step (mm per tracking step); (0, 0) if fewer than 2 points
    is_reversing: bool                  # verdict of the app's reversal detector (tuned in the DAQ > Reversal tab)
    is_tracking: bool
    is_recording: bool
    voltage: float                      # DAQ voltage currently applied (or requested, in a dry run)
    image_shape: tuple                  # shape of the latest frame
    fps: float = 0.0                    # measured camera frame rate (frames/s) over the last ~30 frames; 0 until known
    analysis: BrightnessStats | None = None  # live-analysis stats for this frame, or None when the app is not computing them

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    @property
    def frame_period_s(self) -> float:
        """Seconds per frame (0 until the frame rate is known)."""
        return 1.0 / self.fps if self.fps > 0 else 0.0

    def frames_for(self, seconds: float) -> int:
        """Number of frames that span ``seconds`` at the current frame rate (at least 1)."""
        return max(1, int(round(seconds * self.fps))) if self.fps > 0 else 1


def worm_position_mm(
        stage_xy: Sequence[float],
        cms_offset_px: Sequence[float] | None,
        image_to_stage_mat: np.ndarray,
        unit_to_mm: float) -> tuple[float, float]:
    """Convert the tracker's pixel centroid offset into an absolute stage position.

    Mirrors the conversion in the tracking loop: the matrix maps ``[offset_y, offset_x]`` (pixels,
    y up) to ``[dy, dx]`` in the calibration step unit, which ``unit_to_mm`` scales to mm.
    """
    x, y = float(stage_xy[0]), float(stage_xy[1])
    if cms_offset_px is None or cms_offset_px[0] is None or cms_offset_px[1] is None:
        return (x, y)
    offset = np.array([float(cms_offset_px[1]), float(cms_offset_px[0])], dtype=float)
    dy, dx = np.matmul(np.asarray(image_to_stage_mat, dtype=float), offset)
    return (x + float(dx) * unit_to_mm, y + float(dy) * unit_to_mm)


def trail_velocity(trail: np.ndarray) -> tuple[float, float]:
    if trail is None or len(trail) < 2:
        return (0.0, 0.0)
    step = trail[-1] - trail[-2]
    return (float(step[0]), float(step[1]))


class Scope:
    """Control handle handed to the plugin. Safe to call from the plugin's own thread only."""

    def __init__(self, host: 'PluginHost'):
        self._host = host

    # --- DAQ -------------------------------------------------------------------------------
    def set_voltage(self, volts: float, channel: int | None = None) -> float:
        """Drive the DAC outputs to ``volts`` (clamped to the DAQ range). Returns the applied value.

        ``channel`` None (default) drives DAC0 and DAC1 together; 0 or 1 drives one output alone,
        for a second light source or a trigger line.
        """
        daq = self._host._daq_getter()
        if daq is None:
            return 0.0
        return float(daq.set_voltage(float(volts), channel))

    def light_off(self, channel: int | None = None) -> float:
        return self.set_voltage(0.0, channel)

    @property
    def voltage(self) -> float:
        """The larger of the two output voltages (what the recording logs as daqVol)."""
        daq = self._host._daq_getter()
        return float(getattr(daq, 'currentVoltage', 0.0)) if daq is not None else 0.0

    @property
    def voltages(self) -> tuple[float, float]:
        """(DAC0, DAC1) voltages currently applied."""
        daq = self._host._daq_getter()
        values = getattr(daq, 'channelVoltages', None) if daq is not None else None
        return (float(values[0]), float(values[1])) if values else (0.0, 0.0)

    @property
    def daq_connected(self) -> bool:
        daq = self._host._daq_getter()
        return bool(daq is not None and daq.isConnected())

    # --- Camera ----------------------------------------------------------------------------
    def get_frame(self, copy: bool = True) -> np.ndarray | None:
        """Latest camera frame as a numpy array (None before the first frame)."""
        frame = self._host._frame_provider()
        if frame is None:
            return None
        return np.array(frame, copy=True) if copy else frame

    # --- Stage -----------------------------------------------------------------------------
    def get_position(self) -> tuple[float, float, float] | None:
        stage = self._host._stage_getter()
        if stage is None:
            return None
        position = stage.get_cached_position('mm')
        return None if position is None else tuple(float(v) for v in position)

    def move_rel(self, dx: float, dy: float, dz: float = 0.0) -> bool:
        """Relative stage move in mm. Refused while the tracker, a plate run or Go-To owns the stage."""
        stage = self._blocked_or_stage('move_rel')
        if stage is None:
            return False
        return bool(stage.move_rel((dx, dy, dz), 'mm', wait_until_idle=True))

    def move_abs(self, x: float, y: float, z: float | None = None) -> bool:
        """Absolute stage move in mm. ``z`` defaults to the current Z. Same refusal rules as move_rel."""
        stage = self._blocked_or_stage('move_abs')
        if stage is None:
            return False
        if z is None:
            position = stage.get_cached_position('mm')
            if position is None:
                self._host._warn('move_abs refused: stage position unknown')
                return False
            z = position[2]
        return bool(stage.move_abs((x, y, z), 'mm', wait_until_idle=True))

    def _blocked_or_stage(self, what: str):
        reason = self._host._moves_blocked_reason()
        if reason:
            self._host._warn(f'{what} refused: {reason}')
            return None
        stage = self._host._stage_getter()
        if stage is None:
            self._host._warn(f'{what} refused: stage not connected')
            return None
        return stage

    # --- Recording -------------------------------------------------------------------------
    def start_recording(self) -> None:
        self._host._recording_control(True)

    def stop_recording(self) -> None:
        self._host._recording_control(False)

    @property
    def is_recording(self) -> bool:
        return bool(self._host._recording_state())

    def wait_for_recording(self, recording: bool = True, timeout: float | None = None) -> bool:
        """Block until recording is on (``recording=True``) or off (``False``).

        Returns True when the condition was met, False on timeout or when the plugin is being
        stopped. Blocks the plugin thread only; frames that arrive meanwhile are skipped.
        Typical use: ``scope.wait_for_recording()`` in ``setup`` so the light logic starts with
        the recording.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._host._stop_event.is_set():
            if self.is_recording == recording:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.02)
        return False

    # --- Diagnostics -----------------------------------------------------------------------
    def log(self, **fields) -> None:
        """Append one JSON line to the plugin log in the recording directory."""
        self._host._log(fields)

    def print(self, *args) -> None:
        """Show a message in the Plugin tab status line (also printed to the console)."""
        self._host.message = ' '.join(str(a) for a in args)
        print('[plugin]', self._host.message)

    @property
    def is_stopping(self) -> bool:
        return self._host._stop_event.is_set()

    @property
    def fps(self) -> float:
        """Measured camera frame rate (frames/s), 0 until at least two frames arrived."""
        return float(self._host.fps)


class PluginHost:
    """Loads a plugin file and runs its ``update`` on a dedicated thread, once per new frame."""

    def __init__(
            self, *,
            state_provider: Callable[[], WormState | None],
            frame_provider: Callable[[], np.ndarray | None],
            daq_getter: Callable[[], object | None],
            stage_getter: Callable[[], object | None] = lambda: None,
            moves_blocked_reason: Callable[[], str | None] = lambda: None,
            recording_control: Callable[[bool], None] = lambda start: None,
            recording_state: Callable[[], bool] = lambda: False,
            log_dir_getter: Callable[[], str | None] = lambda: None,
            update_budget_s: float = 0.1,
    ):
        self._state_provider = state_provider
        self._frame_provider = frame_provider
        self._daq_getter = daq_getter
        self._stage_getter = stage_getter
        self._moves_blocked_reason = moves_blocked_reason
        self._recording_control = recording_control
        self._recording_state = recording_state
        self._log_dir_getter = log_dir_getter
        self.update_budget_s = update_budget_s

        self.path: str | None = None
        self.module = None
        self.controller = None
        self.status: str = 'idle'          # idle | loaded | running | stopped | error
        self.last_error: str = ''
        self.message: str = ''
        self.frames_processed: int = 0
        self.last_update_ms: float = 0.0
        self._budget_warned = False
        self.fps: float = 0.0
        self._frame_times: deque[float] = deque(maxlen=30)

        self._thread: threading.Thread | None = None
        self._frame_event = threading.Event()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._log_file = None
        self._log_path: str | None = None

    # --- lifecycle -------------------------------------------------------------------------
    def load(self, path: str) -> None:
        """Import ``path`` as a fresh module and pick up its Controller class or update function."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        name = f'glowtracker_plugin_{uuid.uuid4().hex}'
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'Cannot import plugin {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise

        controller_cls = getattr(module, 'Controller', None)
        if controller_cls is not None:
            controller = controller_cls()
        elif callable(getattr(module, 'update', None)):
            controller = module
        else:
            sys.modules.pop(name, None)
            raise ValueError('Plugin must define a Controller class with update(state, scope) '
                             'or a module-level update(state, scope) function')
        if not callable(getattr(controller, 'update', None)):
            raise ValueError('Controller has no callable update(state, scope)')

        with self._lock:
            self.path = path
            self.module = module
            self.controller = controller
            self.status = 'loaded'
            self.last_error = ''
            self.message = ''

    def start(self) -> bool:
        with self._lock:
            if self.is_running():
                return False
            if self.controller is None:
                if self.path is None:
                    raise RuntimeError('No plugin loaded')
        if self.controller is None:
            self.load(self.path)

        self._stop_event.clear()
        self._frame_event.clear()
        self.frames_processed = 0
        self.last_update_ms = 0.0
        self.last_error = ''
        self._budget_warned = False
        self.status = 'running'
        self._thread = threading.Thread(target=self._run, name='ScriptPlugin', daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout: float | None = 3.0) -> bool:
        self._stop_event.set()
        self._frame_event.set()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def reload(self) -> None:
        """Stop, re-import the same file, and restart if it was running."""
        if self.path is None:
            raise RuntimeError('No plugin loaded')
        was_running = self.is_running()
        self.stop()
        self.load(self.path)
        if was_running:
            self.start()

    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def notify_frame(self) -> None:
        """Called from the camera thread after each frame. Cheap: a timestamp and an event."""
        now = time.perf_counter()
        times = self._frame_times
        if times and now - times[-1] > 2.0:
            times.clear()               # acquisition was paused; restart the measurement
        times.append(now)
        if len(times) >= 2:
            self.fps = (len(times) - 1) / (times[-1] - times[0])
        if self.is_running():
            self._frame_event.set()

    # --- worker ----------------------------------------------------------------------------
    def _run(self) -> None:
        scope = Scope(self)
        controller = self.controller
        failed = False
        try:
            setup = getattr(controller, 'setup', None)
            if callable(setup):
                setup(scope)

            while not self._stop_event.is_set():
                if not self._frame_event.wait(0.5):
                    continue
                self._frame_event.clear()
                if self._stop_event.is_set():
                    break
                state = self._state_provider()
                if state is None:
                    continue
                state = replace(state, fps=self.fps)
                t0 = time.perf_counter()
                controller.update(state, scope)
                elapsed = time.perf_counter() - t0
                self.last_update_ms = elapsed * 1000.0
                self.frames_processed += 1
                if elapsed > self.update_budget_s and not self._budget_warned:
                    self._budget_warned = True
                    self._warn(f'update() took {elapsed * 1000:.0f} ms; frames are being skipped')
        except Exception:
            failed = True
            self.last_error = traceback.format_exc()
            self.status = 'error'
            print(f'[plugin] stopped with error:\n{self.last_error}')
        finally:
            teardown = getattr(controller, 'teardown', None)
            if callable(teardown):
                try:
                    teardown(scope)
                except Exception:
                    if not failed:
                        failed = True
                        self.last_error = traceback.format_exc()
                        self.status = 'error'
                    print(f'[plugin] teardown failed:\n{traceback.format_exc()}')
            self._light_off()
            self._close_log()
            if not failed:
                self.status = 'stopped'

    def _light_off(self) -> None:
        try:
            daq = self._daq_getter()
            if daq is not None:
                daq.safe_off()
        except Exception as e:
            print(f'[plugin] switching the light off failed: {e}')

    def _warn(self, text: str) -> None:
        self.message = text
        print(f'[plugin] {text}')

    # --- logging ---------------------------------------------------------------------------
    def _log(self, fields: dict) -> None:
        with self._lock:
            if self._log_file is None:
                directory = self._log_dir_getter() or os.getcwd()
                os.makedirs(directory, exist_ok=True)
                stamp = time.strftime('%Y%m%d_%H%M%S')
                self._log_path = os.path.join(directory, f'plugin_log_{stamp}.jsonl')
                self._log_file = open(self._log_path, 'a', encoding='utf-8')
            record = {'wall_time': time.time(), **fields}
            self._log_file.write(json.dumps(record, default=_json_default) + '\n')
            self._log_file.flush()

    def _close_log(self) -> None:
        with self._lock:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                finally:
                    self._log_file = None

    @property
    def log_path(self) -> str | None:
        return self._log_path


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
