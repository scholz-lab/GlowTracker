"""Optional row sweeps with continuous capture and stationary confirmation."""

from collections import deque
from itertools import groupby
import time

import numpy as np
from kivy.app import App
from kivy.clock import Clock
from pypylon import pylon

import Microscope_macros as macro
from plate_plan import SearchReport


class ScanInterrupted(Exception):
    """The scan was stopped or hardware is shutting down."""


def scan_rows(tiles):
    """Preserve the existing serpentine ordering and use each row's endpoints."""
    return [(row[0], row[-1]) for _, points in groupby(tiles, key=lambda p: p[1])
            if (row := list(points))]


class ContinuousScanMixin:
    def _continuous_check(self):
        if self._stop_scan or self._stop_all or self._teardown_requested:
            raise ScanInterrupted()

    def _continuous_position(self, stage):
        position = stage.get_position(unit='mm', isAsync=False)
        if position is None:
            raise RuntimeError('Continuous scan lost the stage position')
        return tuple(position)

    def _continuous_move(self, stage, target):
        self._continuous_check()
        if not stage.move_abs(target, 'mm', wait_until_idle=True):
            raise RuntimeError('Continuous scan move failed')
        self._continuous_check()
        position = self._continuous_position(stage)
        if any(abs(a - b) > 0.05 for a, b in zip(position, target)):
            raise RuntimeError('Continuous scan did not reach its target')
        if self._wait_or_stop(self.scan_settle):
            raise ScanInterrupted()

    def _continuous_stop_stage(self, stage):
        if not stage.emergency_stop():
            raise RuntimeError('Could not stop the stage during continuous scan')
        stop_deadline = time.monotonic() + 5
        while stage.is_busy():
            if time.monotonic() >= stop_deadline:
                raise RuntimeError('Stage did not stop during continuous scan')
            time.sleep(0.02)

    def _continuous_frame(self, camera):
        # Short polls keep Stop responsive during long exposures.
        frame_deadline = time.monotonic() + max(2.0, self.scan_exposure / 1e6 * 2 + 1)
        while True:
            self._continuous_check()
            ok, image, _, _ = camera.retrieveGrabbingResult(timeout_ms=50)
            self._continuous_check()
            if ok:
                self._continuous_frames += 1
                app = App.get_running_app()
                Clock.schedule_once(lambda dt, im=image: setattr(app, 'image', im))
                return image
            if not camera.IsGrabbing() or time.monotonic() >= frame_deadline:
                raise RuntimeError('No frames received during continuous scan')

    def _continuous_snapshot(self, camera):
        # Restart after settling: a buffered moving frame must never confirm a find.
        self._continuous_check()
        camera.StopGrabbing()
        try:
            camera.StartGrabbingMax(1)
            return self._continuous_frame(camera)
        finally:
            camera.StopGrabbing()

    def _continuous_confirm(self, app):
        """Only stationary images may drive recentering or start tracking."""
        if self._wait_or_stop(self.scan_settle):
            raise ScanInterrupted()
        for attempt in range(int(self.scan_recenter_iters) + 1):
            image = self._continuous_snapshot(app.camera)
            present, offset = macro.detect_worm(image, self.scan_threshold, self.scan_min_pixels)
            if not present:
                return False
            if (max(abs(offset[0]), abs(offset[1])) <= self.scan_center_tol
                    or attempt == int(self.scan_recenter_iters)):
                self._continuous_check()
                return True
            self._continuous_check()
            dy, dx = macro.getStageDistances(
                np.array([-offset[1], offset[0]]), app.imageToStageMat)
            units = app.config.get('Calibration', 'step_units')
            if not app.stage.move_rel((dx, dy, 0), unit=units, wait_until_idle=True):
                raise RuntimeError('Continuous scan recentering move failed')
            if self._wait_or_stop(self.scan_settle):
                raise ScanInterrupted()
        return False

    def _continuous_sweep(self, app, start, end, z, scan_motion, precise_motion):
        stage, camera = app.stage, app.camera
        stage.set_motion(*precise_motion)
        self._continuous_move(stage, (*start, z))
        if self._continuous_confirm(app):
            return True
        if start == end:
            return False

        resume = (*start, z)
        if self._continuous_position(stage) != resume:
            self._continuous_move(stage, resume)
        while True:
            self._continuous_check()
            stage.set_motion(*scan_motion)
            # X-only sweeps stay at the already validated Y and Z. Unlike move_abs,
            # move_x supports nonblocking motion, so this thread can read frames.
            camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            history = deque([resume], maxlen=3)
            try:
                self._continuous_check()
                if not stage.move_x(end[0] - resume[0], unit='mm', wait_until_idle=False):
                    raise RuntimeError('Continuous row movement failed')
                while True:
                    image = self._continuous_frame(camera)
                    position = self._continuous_position(stage)
                    history.append(position)
                    present, _ = macro.detect_worm(image, self.scan_threshold, self.scan_min_pixels)
                    if present:
                        break
                    if not stage.is_busy():
                        # The move can finish after the earlier position sample.
                        # Validate a fresh position taken after observing idle.
                        position = self._continuous_position(stage)
                        self._continuous_check()
                        if abs(position[0] - end[0]) > 0.05:
                            raise RuntimeError('Continuous row stopped before its endpoint: '
                                               f'actual X={position[0]:.3f} mm, target X={end[0]:.3f} mm')
                        break
            finally:
                # Stop motion even when acquisition, detection, or position reads fail.
                try:
                    self._continuous_stop_stage(stage)
                finally:
                    camera.StopGrabbing()

            self._continuous_check()
            resume = self._continuous_position(stage)
            stage.set_motion(*precise_motion)
            if self._continuous_confirm(app):
                return True
            if present:
                # Host samples are approximate, not synchronized exposure positions.
                # Recheck the recent path to recover a candidate passed before stopping.
                oldest = history[0]
                midpoint = tuple((a + b) / 2 for a, b in zip(oldest, resume))
                for target in (midpoint, oldest):
                    self._continuous_move(stage, target)
                    if self._continuous_confirm(app):
                        return True
                self._continuous_move(stage, resume)
            if abs(resume[0] - end[0]) <= 0.05:
                return False

    def _scan_continuous(self, z=None):
        app = App.get_running_app()
        if app.camera is None or app.stage is None or app.plateCenter is None or app.plateRadius is None:
            raise RuntimeError('Connect the camera and stage and define a plate before scanning')
        fov = app.get_fov_mm()
        if fov is None:
            raise RuntimeError('Calibrate the field of view before scanning')
        z = self.scan_z if z is None else z
        tiles = macro.generate_scan_tiles(
            app.plateCenter, app.plateRadius, *fov,
            overlap_w=self.scan_overlap_w / 100, overlap_h=self.scan_overlap_h / 100)
        limits = [float(v) for v in app.config.get('Stage', 'stage_limits').split(',')]
        tiles = [(x, y) for x, y in tiles if 0 <= x <= limits[0] and 0 <= y <= limits[1]
                 and 0 <= z <= limits[2] and app.stage.is_safe(x, y, z)]
        rows = scan_rows(tiles)
        if not rows:
            raise RuntimeError('No safe rows to scan at this Z')

        units = (app.config.get('Stage', 'speed_unit'), app.config.get('Stage', 'acceleration_unit'))
        scan_motion = (float(app.config.get('Stage', 'scan_speed')),
                       float(app.config.get('Stage', 'scan_acceleration')), *units)
        precise_motion = (float(app.config.get('Stage', 'precise_speed')),
                          float(app.config.get('Stage', 'precise_acceleration')), *units)
        started = time.monotonic()
        self._continuous_frames = 0
        found = False
        completed_rows = 0
        try:
            for scan_pass in range(int(self.search_passes)):
                for index, (start, end) in enumerate(rows):
                    self._continuous_check()
                    status = (f'{self._active_plate_name or "Plate"} • Continuous search • '
                              f'pass {scan_pass + 1}, row {index + 1}/{len(rows)}')
                    Clock.schedule_once(lambda dt, text=status: setattr(self, 'run_status', text))
                    Clock.schedule_once(lambda dt, v=index / len(rows): setattr(self, 'scan_progress', v))
                    found = self._continuous_sweep(app, start, end, z, scan_motion, precise_motion)
                    completed_rows += 1
                    Clock.schedule_once(lambda dt, v=(index + 1) / len(rows): setattr(self, 'scan_progress', v))
                    if found:
                        break
                if found:
                    break
        except ScanInterrupted:
            found = False
        finally:
            try:
                self._continuous_stop_stage(app.stage)
            finally:
                try:
                    app.camera.StopGrabbing()
                finally:
                    app.stage.set_motion(*precise_motion)
        if found:
            try:
                # Explicit cancellation must still prevent the handoff after cleanup.
                self._continuous_check()
                app.update_coordinates(isAsync=False)
                self._continuous_check()
            except ScanInterrupted:
                found = False
        elapsed = time.monotonic() - started
        self._search_report = SearchReport(completed_rows, len(rows) * int(self.search_passes),
                                           'rows')
        print(f'continuous scan: {elapsed:.2f}s | {self._continuous_frames} frames | '
              f'{completed_rows} rows visited | found={found}'
              f'{"" if found else " | " + self._search_report.summary}')
        return found
