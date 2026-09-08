"""UI coordination and sequential, cancellable multi-plate visits."""

import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from threading import Event, Thread, current_thread
import time
from uuid import uuid4

import numpy as np
from kivy.app import App
from kivy.clock import Clock

from plate_plan import FIELDS, brightness_steps, create_run_directory, create_visit_directory, validate_plate, visits


class RunCancelled(Exception):
    pass


class PlateRunController:
    def commit_fields(self):
        valid = [field.commit() for field in getattr(self, '_fields', [])]
        return all(valid)

    def _profile(self):
        return {key: getattr(self, key) for key in FIELDS}

    def _draft(self):
        app = App.get_running_app()
        return validate_plate({
            'id': uuid4().hex[:8], 'name': self.ids.scenarioname.text.strip(),
            'center': list(app.plateCenter) if app.plateCenter is not None else [],
            'radius': app.plateRadius or 0, 'points': deepcopy(list(self.points)),
            'settings': self._profile(), 'enabled': True, 'status': 'Ready',
        })

    def store_plate(self):
        if self.running or not self.commit_fields():
            return False
        try:
            plate = self._draft()
        except ValueError as error:
            self.run_status = str(error)
            return False
        plates = deepcopy(list(self.plates))
        if 0 <= self.selected_plate < len(plates):
            old = plates[self.selected_plate]
            plate.update(id=old['id'], enabled=old['enabled'])
            plates[self.selected_plate] = plate
        else:
            plates.append(plate)
            self.selected_plate = len(plates) - 1
        self.plates = plates
        self._update_cycle_summary()
        self.run_status = f'{plate["name"]} ready'
        return True

    def _save_selected(self):
        return self.store_plate() if self.selected_plate >= 0 else self.commit_fields()

    def new_plate(self):
        if self.running:
            return
        self.selected_plate = -1
        self.reset()
        for key, spec in FIELDS.items():
            setattr(self, key, spec[1])
        self.ids.scenarioname.text = f'Plate {len(self.plates) + 1}'
        self.run_status = 'Capture three points around the new plate’s rim'

    def select_plate(self, index):
        if self.running:
            return
        self.selected_plate = index
        self._load_plate(self.plates[index])
        self.run_status = f'Editing {self.plates[index]["name"]}'

    def _load_plate(self, plate):
        self.points = deepcopy(plate.get('points', []))
        self.ids.scenarioname.text = plate['name']
        for key, spec in FIELDS.items():
            setattr(self, key, plate['settings'].get(key, spec[1]))
        self._set_plate(plate['center'], plate['radius'])
        self.ids.resultlabel.text = 'Diameter: {:.2f} mm  Center: ({:.2f}, {:.2f})'.format(
            2 * plate['radius'], *plate['center'])

    def remove_plate(self):
        if self.running or self.selected_plate < 0:
            return
        plates = list(self.plates)
        del plates[self.selected_plate]
        self.selected_plate = -1
        self.plates = plates
        self.reset()
        self._update_cycle_summary()

    def enable_plate(self, index, enabled):
        if self.running:
            return
        plates = deepcopy(list(self.plates))
        plates[index]['enabled'] = enabled
        self.plates = plates
        self._update_cycle_summary()

    def use_current_z(self):
        stage = App.get_running_app().stage
        position = stage.get_cached_position(max_age=1) if stage is not None else None
        if position is None:
            self.run_status = 'Connect the stage to use its current Z'
        else:
            self.scan_z = position[2]

    def _update_cycle_summary(self):
        enabled = [p for p in self.plates if p.get('enabled', True)]
        track = sum(p['settings']['track_interval'] for p in enabled)
        self.cycle_summary = (f'{len(enabled)} enabled • {track / 60:g} min tracking per cycle\n'
                              'Search ends on a find or after all passes')

    def _ui(self, callback, cleanup=False):
        """Wait for UI-owned state changes; abandoned callbacks cannot start work."""
        done = Event()
        result = []
        generation = self._run_generation
        abandoned = Event()

        def apply(dt):
            try:
                if abandoned.is_set() or self._teardown_requested or (
                        not cleanup and (self._stop_all or generation != self._run_generation)):
                    raise RunCancelled()
                result.append(callback())
            except Exception as error:
                result.append(error)
            finally:
                done.set()
        event = Clock.schedule_once(apply)
        deadline = time.monotonic() + 30
        while not done.wait(0.05):
            if self._teardown_requested or (self._stop_all and not cleanup):
                abandoned.set()
                event.cancel()
                raise RunCancelled()
            if time.monotonic() > deadline:
                abandoned.set()
                event.cancel()
                raise RuntimeError('The interface did not respond; run stopped')
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]

    def _status(self, text, plate_id=None, plate_status=None):
        def apply():
            self.run_status = text
            if plate_id is not None:
                plates = deepcopy(list(self.plates))
                for plate in plates:
                    if plate['id'] == plate_id:
                        plate['status'] = plate_status or text
                self.plates = plates
        self._ui(apply)

    def start_run(self, *args):
        if self.running or any(t is not None and t.is_alive()
                               for t in (self._scan_thread, self._plates_thread)):
            return
        if not self.plates:
            self.run_status = 'Add a plate before starting the run'
            return
        app = App.get_running_app()
        try:
            plan = [validate_plate(p) for p in self.plates if p.get('enabled', True)]
            if not plan:
                raise ValueError('Enable at least one plate')
            if app.camera is None or app.stage is None or app.stage.connection is None:
                raise ValueError('Connect the camera and stage before starting')
            if getattr(app, '_hardware_teardown', False):
                raise ValueError('Hardware is disconnecting')
            if any(getattr(app.stage.state, f'isMoving_{axis}', False) for axis in 'xyz'):
                raise ValueError('Stop manual stage movement before starting the run')
            go_to = getattr(app.root.ids.leftcolumn.ids, 'gotocontrols', None)
            move_thread = getattr(go_to, '_moveThread', None)
            if move_thread is not None and move_thread.is_alive():
                raise ValueError('Wait for the current Go To movement to finish')
            rc = app.root.ids.middlecolumn.ids.runtimecontrols
            mgr = rc.ids.imageacquisitionmanager
            if rc.isTracking or rc.livefocuscheckbox.state == 'down' or mgr.recordbutton.state == 'down':
                raise ValueError('Stop current tracking, autofocus and recording before starting a plate run')
            for name in ('trackthread', 'liveFocusThread'):
                thread = getattr(rc, name, None)
                if thread is not None and thread.is_alive():
                    raise ValueError('Wait for tracking and autofocus to finish stopping')
            if app.get_fov_mm() is None:
                raise ValueError('Calibrate the camera field of view before scanning')
            limits = [float(v) for v in app.config.get('Stage', 'stage_limits').split(',')]
            for plate in plan:
                x, y = plate['center']
                z = plate['settings']['scan_z']
                half = plate['settings']['scan_z_range'] / 2
                if not (0 <= x <= limits[0] and 0 <= y <= limits[1]
                        and 0 <= z - half <= z + half <= limits[2]):
                    raise ValueError(f'{plate["name"]}: center or focus range is outside stage limits')
                if not app.stage.is_safe(x, y, z + half):
                    raise ValueError(f'{plate["name"]}: focus range intersects the stage keep-out area')
                for key, feature in (('scan_gain', 'Gain'), ('track_gain', 'Gain'),
                                     ('scan_exposure', 'ExposureTime'), ('track_exposure', 'ExposureTime'),
                                     ('track_framerate', 'AcquisitionFrameRate')):
                    node = getattr(app.camera, feature)
                    value = plate['settings'][key]
                    if not node.Min <= value <= node.Max:
                        raise ValueError(f'{plate["name"]}: {FIELDS[key][0]} must be {node.Min:g} to {node.Max:g}')
            run_dir = create_run_directory(self.record_directory, self.record_name) if self.record_enabled else None
            if run_dir:
                (run_dir / 'plan.json').write_text(json.dumps(plan, indent=2))
        except (ValueError, OSError) as error:
            self.run_status = str(error)
            return
        self._stop_all = self._stop_scan = self._teardown_requested = False
        self._run_generation += 1
        self._resume_run = Event()
        self._resume_run.set()
        self.pause_requested = self.paused = False
        self.running = True
        self._recording_owned = False
        app._plate_run_active = True
        self.run_status = 'Starting plate run…'
        app.unbind_keys()
        self._plates_thread = Thread(target=self._execute_plan, args=(plan, run_dir, self.repeat_run), daemon=True,
                                     name='PlateRun')
        self._plates_thread.start()

    def _execute_plan(self, plan, run_dir, repeat):
        asyncio.set_event_loop(asyncio.new_event_loop())
        app = App.get_running_app()
        original_profile = self._profile()
        learned_z = {}
        tracked = empty = incomplete = 0
        last_incomplete = ''
        final_status = 'Run complete'
        try:
            for cycle, plate in visits(plan, repeat):
                if self._stop_all:
                    raise RunCancelled()
                if plate['id'] in learned_z:
                    plate['settings']['scan_z'] = learned_z[plate['id']]
                self._ui(lambda p=plate: self._load_plate(p))
                self._status(f'{plate["name"]} • visit {cycle} • Finding focus…', plate['id'], 'Focusing')
                camera_prepared = False
                found = False
                self._search_report = None
                try:
                    camera_prepared = self._begin_scan_camera()
                    if not camera_prepared:
                        raise RuntimeError('Could not prepare the scan camera')
                    z = self._find_scan_z()
                    if z is None:
                        if self._stop_all:
                            raise RunCancelled()
                        raise RuntimeError(f'{plate["name"]}: focus search failed')
                    learned_z[plate['id']] = z
                    self._active_plate_name = plate['name']
                    self._status(f'{plate["name"]} • Searching…', plate['id'], 'Searching')
                    found = self._scan(z)
                finally:
                    if camera_prepared:
                        self._end_scan_camera(found)
                if self._stop_all:
                    raise RunCancelled()
                if self._stop_scan:
                    raise RuntimeError('Stage movement failed; run stopped')
                if found:
                    self._status(f'{plate["name"]} • Worm found — starting tracking and autofocus…',
                                 plate['id'], 'Starting tracking')
                    folder = create_visit_directory(run_dir, plate, cycle) if run_dir else None
                    outcome = self._track_visit(plate, folder)
                    tracked += 1
                else:
                    report = self._search_report
                    outcome = report.summary if report is not None else 'No worm found'
                    if report is not None and report.incomplete:
                        incomplete += 1
                        last_incomplete = f'{plate["name"]}: {outcome}'
                    else:
                        empty += 1
                self._status(f'{plate["name"]} • {outcome}', plate['id'], outcome)
                if self.pause_requested and (repeat or plate['id'] != plan[-1]['id']):
                    self._resume_run.clear()
                    self._ui(lambda: setattr(self, 'paused', True))
                    self._status('Paused between plates — press Resume run')
                    while not self._resume_run.wait(0.1):
                        if self._stop_all:
                            raise RunCancelled()
            if incomplete:
                final_status = (f'Run incomplete — {tracked} tracked, {empty} no worm found, '
                                f'{incomplete} incomplete searches. {last_incomplete}')
            elif empty:
                final_status = f'Run finished — {tracked} tracked, {empty} no worm found'
        except RunCancelled:
            final_status = 'Run stopped'
        except Exception as error:
            final_status = f'Run stopped: {error}'
            if app.stage is not None:
                app.stage.emergency_stop()
        finally:
            try:
                if not self._teardown_requested:
                    self._quiesce_visit()
            except Exception as error:
                final_status = f'Run stopped; cleanup needs attention: {error}'
            def finish(dt):
                status = final_status
                try:
                    for key, value in original_profile.items():
                        setattr(self, key, value)
                    if 0 <= self.selected_plate < len(self.plates):
                        self._load_plate(self.plates[self.selected_plate])
                except Exception as error:
                    status = f'{status}; could not restore plate display: {error}'
                finally:
                    self.running = self.paused = self.pause_requested = False
                    app._plate_run_active = False
                    self.run_status = status
                    if not self._teardown_requested:
                        app.bind_keys()
            Clock.schedule_once(finish)
            asyncio.get_event_loop().close()

    def _check_cancelled(self):
        if self._stop_all or self._stop_scan or self._teardown_requested:
            raise RunCancelled()

    def _wait_for_preview(self, after):
        app = App.get_running_app()
        mgr = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager
        ready_by = time.monotonic() + 10
        while True:
            self._check_cancelled()
            if app.camera is None:
                raise RuntimeError('Camera disconnected during tracking preparation')
            error = getattr(mgr.liveviewbutton, 'acquisitionError', None)
            if error:
                raise RuntimeError(f'Camera preview failed: {error}')
            if app.camera.IsGrabbing() and app.image is not None \
                    and mgr.imageRetrieveTimeStamp > after:
                return
            if time.monotonic() >= ready_by:
                raise RuntimeError('Camera preview did not deliver a fresh frame')
            time.sleep(0.05)

    def _wait_tracking_focus(self, duration):
        """Keep tracking/focus active and allow multiple fresh focus batches."""
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        target_batches = rc.focus_batches + 3  # new reference, Z step, response to that step
        deadline = time.monotonic() + duration
        focus_rate = min(app.config.getfloat('Autofocus', 'focusfps'), app.camera.ResultingFrameRate())
        timeout = deadline + max(10, 6 * app.config.getint('Autofocus', 'buffer_n') / max(0.1, focus_rate))
        while True:
            self._check_cancelled()
            if not rc.isTracking or rc.track_done.is_set():
                raise RuntimeError('Tracking stopped during the exposure ramp')
            thread = getattr(rc, 'liveFocusThread', None)
            if rc.livefocuscheckbox.state != 'down' or thread is None or not thread.is_alive():
                raise RuntimeError('Autofocus stopped during the exposure ramp')
            if app.camera is None or not app.camera.IsGrabbing():
                raise RuntimeError('Camera stopped during the exposure ramp')
            if time.monotonic() >= deadline and rc.focus_batches >= target_batches \
                    and rc._focus_applied_epoch == rc._focus_brightness_epoch:
                return
            if time.monotonic() >= timeout:
                raise RuntimeError('Autofocus did not receive enough fresh frames; exposure ramp stopped')
            time.sleep(0.05)

    def _prepare_tracking(self, plate):
        """Follow the worm at scan brightness, then ramp while continuously focusing."""
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        mgr = rc.ids.imageacquisitionmanager
        after = time.perf_counter()
        self._ui(lambda: setattr(mgr.liveviewbutton, 'state', 'down'))
        self._wait_for_preview(after)
        def begin():
            rc.track_done.clear()
            rc.isShowTrackingDialogueFirstTime = False
            rc.trackingcheckbox.state = 'down'
            h, w = app.image.shape[:2]
            cx = w / 2
            if app.config.getboolean('DualColor', 'dualcolormode'):
                cx = w * (0.75 if app.config.get('DualColor', 'mainside') == 'Right' else 0.25)
            rc.startTracking(np.array([cx, h / 2]))
            if not rc.isTracking:
                raise RuntimeError('Tracking did not start')
        self._ui(begin)
        # Let the first cropped frame arrive before autofocus uses the tracking ROI.
        self._wait_for_preview(time.perf_counter() + app.camera.ExposureTime.Value / 1e6)
        self._ui(lambda: setattr(rc.livefocuscheckbox, 'state', 'down'))
        settings = plate['settings']
        self._status(f'{plate["name"]} • Tracking and focusing at scan exposure…', plate['id'], 'Focusing')
        self._wait_tracking_focus(settings['focus_settle_seconds'])
        steps = list(brightness_steps(app.camera.ExposureTime.Value, app.camera.Gain.Value,
                                      settings['track_exposure'], settings['track_gain']))
        for index, (exposure, gain) in enumerate(steps, 1):
            self._check_cancelled()
            self._status(f'{plate["name"]} • Exposure step {index}/{len(steps)} • '
                         f'Focusing at {exposure:g} us', plate['id'], 'Adjusting exposure')
            self._ui(lambda e=exposure, g=gain: rc.set_tracking_brightness(e, g))
            self._wait_tracking_focus(settings['exposure_ramp_seconds'] / len(steps))
        def configure_fps():
            app.camera.AcquisitionFrameRateEnable.Value = True
            app.camera.AcquisitionFrameRate.Value = float(settings['track_framerate'])
        self._ui(configure_fps)
        self._status(f'{plate["name"]} • Tracking and focusing at target exposure…', plate['id'], 'Focusing')
        self._wait_tracking_focus(settings['focus_settle_seconds'])
        self._check_cancelled()

    def _track_visit(self, plate, folder):
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        mgr = rc.ids.imageacquisitionmanager
        left = app.root.ids.leftcolumn
        saved = None
        try:
            self._prepare_tracking(plate)
            self._status(f'{plate["name"]} • Tracking', plate['id'], 'Recording' if folder else 'Tracking')
            if folder:
                self._recording_owned = True
                saved = self._ui(lambda: (left.savefile, {k: app.config.get('Experiment', k)
                                      for k in ('iscontinuous', 'extension', 'duration', 'nframes')}))
                def record():
                    left.savefile = str(folder)
                    left.ids.camprops.framerate = plate['settings']['track_framerate']
                    app.config.set('Experiment', 'iscontinuous', '1')
                    app.config.set('Experiment', 'extension', self.record_format)
                    app.config.set('Experiment', 'duration', str(plate['settings']['track_interval']))
                    app.config.set('Experiment', 'nframes', str(int(plate['settings']['track_interval'] * plate['settings']['track_framerate'])))
                    mgr.recordbutton.state = 'down'
                    if mgr.recordbutton.state != 'down':
                        raise RuntimeError('Recording did not start')
                self._ui(record)
            deadline = time.monotonic() + plate['settings']['track_interval']
            last_second = None
            while time.monotonic() < deadline:
                if self._stop_all:
                    raise RunCancelled()
                if folder:
                    error = getattr(mgr.recordbutton, 'saveHandoffError', None) or getattr(mgr.recordbutton, 'acquisitionError', None)
                    if error or mgr.recordbutton.state != 'down':
                        raise RuntimeError(f'Recording stopped unexpectedly: {error or "camera stopped"}')
                if rc.track_done.is_set():
                    raise RuntimeError('Tracking ended before the visit finished')
                remaining = max(0, int(deadline - time.monotonic()))
                if remaining != last_second:
                    self._status(f'{plate["name"]} • {"Recording" if folder else "Tracking"} • {remaining}s remaining')
                    last_second = remaining
                time.sleep(0.1)
            return 'Visit complete'
        finally:
            try:
                self._quiesce_visit()
            finally:
                if saved is not None and not self._teardown_requested:
                    def restore():
                        left.savefile = saved[0]
                        for key, value in saved[1].items():
                            app.config.set('Experiment', key, value)
                        app.config.write()
                    self._ui(restore, cleanup=True)

    def _quiesce_visit(self):
        if self._teardown_requested:
            return
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        mgr = rc.ids.imageacquisitionmanager
        def stop():
            rc.livefocuscheckbox.state = 'normal'
            rc.isTracking = False
            rc.track_done.set()
        self._ui(stop, cleanup=True)
        self._join_workers([getattr(rc, key, None) for key in ('trackthread', 'liveFocusThread')])
        if app.stage is not None:
            app.stage.emergency_stop()
        def stop_camera():
            if mgr.recordbutton.state == 'down':
                # The run owns the next camera transition; suppress auto-preview.
                mgr.recordbutton.prevLiveViewButtonState = 'normal'
                mgr.recordbutton.state = 'normal'
            rc.trackingcheckbox.state = 'normal'
            mgr.liveviewbutton.state = 'normal'
        self._ui(stop_camera, cleanup=True)
        self._join_workers([getattr(button, 'imageAcquisitionThread', None)
                            for button in (mgr.recordbutton, mgr.liveviewbutton)])
        if getattr(self, '_recording_owned', False) and getattr(mgr.recordbutton, 'saveHandoffError', None):
            raise RuntimeError(str(mgr.recordbutton.saveHandoffError))
        self._recording_owned = False

    @staticmethod
    def _join_workers(threads):
        deadline = time.monotonic() + 10
        for thread in threads:
            if thread is not None and thread is not current_thread() and thread.is_alive():
                thread.join(max(0, deadline - time.monotonic()))
                if thread.is_alive():
                    raise RuntimeError('A camera or motion worker did not stop; the next plate was not started')

    def toggle_pause(self):
        if self.paused:
            self.paused = self.pause_requested = False
            self._resume_run.set()
        else:
            self.pause_requested = not self.pause_requested
