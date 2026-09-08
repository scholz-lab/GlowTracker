from __future__ import annotations

from kivy.uix.boxlayout import BoxLayout
from kivy.properties import NumericProperty, ListProperty, BooleanProperty, StringProperty
import os
from threading import Event, Thread, current_thread
import asyncio
import json
import time
from kivy.clock import Clock
from kivy.app import App

import Microscope_macros as macro
import numpy as np
from plate_run import PlateRunController, RunCancelled
from plate_plan import FIELDS, validate_plate
from platformdirs import user_config_dir
from continuous_scan import ContinuousScanMixin

class CenterRadiusFromThreePoints(ContinuousScanMixin, PlateRunController, BoxLayout):
    points = ListProperty([])
    _stop_scan = False
    _stop_all = False
    _preview_saved = None
    scan_progress = NumericProperty(0)
    scan_mode = StringProperty('Sequential')
    saved_scenarios = ListProperty([])
    scan_z = NumericProperty(133)
    scan_exposure = NumericProperty(100000)
    scan_gain = NumericProperty(30)
    scan_settle = NumericProperty(0.01)
    scan_threshold = NumericProperty(150)
    scan_min_pixels = NumericProperty(50)
    scan_overlap_w = NumericProperty(10)
    scan_overlap_h = NumericProperty(10)
    scan_recenter_iters = NumericProperty(3)
    scan_center_tol = NumericProperty(15)
    scan_z_range = NumericProperty(14.0)
    scan_z_frames = NumericProperty(100)
    track_exposure = NumericProperty(5000)
    track_gain = NumericProperty(22)
    track_framerate = NumericProperty(30)
    track_interval = NumericProperty(120)
    focus_settle_seconds = NumericProperty(3)
    exposure_ramp_seconds = NumericProperty(20)
    search_seconds = NumericProperty(60)
    search_passes = NumericProperty(1)
    plates = ListProperty([])
    selected_plate = NumericProperty(-1)
    running = BooleanProperty(False)
    paused = BooleanProperty(False)
    pause_requested = BooleanProperty(False)
    repeat_run = BooleanProperty(False)
    record_enabled = BooleanProperty(False)
    record_directory = StringProperty('')
    record_name = StringProperty('plate_run')
    record_format = StringProperty('tiff')
    run_status = StringProperty('Define a plate or load a saved plate to begin')
    cycle_summary = StringProperty('No plates enabled')
    _found = False

    def __init__(self, **kwargs):
        self._scan_thread = None
        self._plates_thread = None
        self._teardown_requested = False
        self._run_generation = 0
        self._fields = []
        self._active_plate_name = ''
        super().__init__(**kwargs)

    def on_kv_post(self, *args):
        from scan_ui import build_scan_ui
        build_scan_ui(self)
        app = App.get_running_app()
        self.record_directory = app.root.ids.leftcolumn.savefile if app.root is not None else ''
        self.refresh_scenarios()

    def _scenario_path(self):
        return os.path.join(user_config_dir('GlowTracker', 'Monika Scholz'), 'scan_scenarios.json')

    def _read_scenarios(self):
        try:
            path = self._scenario_path()
            if not os.path.exists(path):
                path = os.path.join(os.path.dirname(__file__), 'settings', 'scan_scenarios.json')
            with open(path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def refresh_scenarios(self):
        self.saved_scenarios = sorted(self._read_scenarios().keys())

    def save_scenario(self, name):
        if not self.commit_fields():
            return
        name = name.strip()
        if not name:
            print('enter a scenario name')
            return
        app = App.get_running_app()
        data = self._read_scenarios()
        entry = {'points': [list(p) for p in self.points], 'settings': self._profile()}
        if app.plateCenter is not None:
            entry['center'] = list(app.plateCenter)
            entry['radius'] = app.plateRadius
        data[name] = entry
        path = self._scenario_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.refresh_scenarios()
        self.run_status = f'Saved plate preset: {name}'

    def load_scenario(self, name):
        entry = self._read_scenarios().get(name)
        if entry is None:
            return
        self.selected_plate = -1
        self.ids.scenarioname.text = name
        for key, spec in FIELDS.items():
            setattr(self, key, entry.get('settings', {}).get(key, spec[1]))
        self.points = [list(p) for p in entry.get('points', [])]
        if len(self.points) >= 3:
            self.calculate()
        elif 'center' in entry and 'radius' in entry:
            app = App.get_running_app()
            app.plateCenter = tuple(entry['center'])
            app.plateRadius = entry['radius']
            self.ids.resultlabel.text = 'Diameter: {:.2f} mm    Center: ({:.2f}, {:.2f})'.format(2 * entry['radius'], *entry['center'])
        self.run_status = f'Loaded {name}; add it to the run'

    def set_points_from_text(self, text):
        if not text.strip():
            return
        pts = []
        for pair in text.replace('\n', ';').split(';'):
            pair = pair.strip()
            if not pair:
                continue
            try:
                x, y = (float(v) for v in pair.split(','))
            except ValueError:
                print(f'bad point: {pair}')
                return
            pts.append([x, y])
        self.points = pts
        self.calculate()

    def capture_points(self):
        stage = App.get_running_app().stage
        coords = stage.get_cached_position(max_age=1) if stage is not None else None
        if coords is None:
            self.run_status = 'Connect the stage before capturing a rim point'
            return
        self.points.append(list(coords[:2]))
        if len(self.points) >= 3:
            self.calculate()

    def compute_circle(self):
        if len(self.points) < 3:
            print('not enough points, add at least 3 points')
            return None

        pts = np.array(self.points, dtype=float)
        A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        B = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)

        X, _, rank, _ = np.linalg.lstsq(A, B, rcond=None)
        if rank < 3:
            return None

        xc, yc = -X[0] / 2, -X[1] / 2
        underRoot = xc ** 2 + yc ** 2 - X[2]
        if underRoot <= 0:
            return None

        radius = np.sqrt(underRoot)
        return (float(xc), float(yc)), float(radius)

    def calculate(self):
        app = App.get_running_app()
        result = self.compute_circle()
        if result is None:
            app.plateCenter = None
            app.plateRadius = None
            self.ids.resultlabel.text = 'Diameter: -    Center: -'
            self.run_status = 'Capture three distinct points around the plate rim'
            return
        (xc, yc), radius = result
        app.plateCenter = (xc, yc)
        app.plateRadius = radius
        self.ids.resultlabel.text = \
            'Diameter: {:.2f} mm    Center: ({:.2f}, {:.2f})'.format(2 * radius, xc, yc)

    def reset(self):
        app = App.get_running_app()
        self.points = []
        app.plateCenter = None
        app.plateRadius = None
        self.ids.resultlabel.text = 'Diameter: -    Center: -'

    def scan_area(self):
        if self._scan_thread is not None and self._scan_thread.is_alive():
            return
        if self._plates_thread is not None and self._plates_thread.is_alive():
            return
        app = App.get_running_app()
        if app.camera is None or app.stage is None or getattr(app, '_hardware_teardown', False):
            print('camera or stage not connected')
            return
        self._stop_scan = False
        self._stop_all = False
        self._teardown_requested = False
        self._run_generation += 1
        self.running = True
        app._plate_run_active = True
        app.unbind_keys()

        def worker():
            asyncio.set_event_loop(asyncio.new_event_loop())
            found = False
            cameraPrepared = False
            final_status = 'Scan complete'
            try:
                cameraPrepared = self._begin_scan_camera()
                if cameraPrepared and not self._stop_scan:
                    peakZ = self._find_scan_z()
                    if peakZ is not None and not self._stop_scan:
                        found = self._scan(z= peakZ)
                if cameraPrepared:
                    self._end_scan_camera(found)
                if found:
                    self._track_visit(self._ui(self._draft), None)
            except RunCancelled:
                final_status = 'Run stopped'
            except Exception as e:
                final_status = f'Scan stopped: {e}'
                print(f'scan failed: {e}')
                if cameraPrepared:
                    try:
                        self._end_scan_camera(False)
                    except Exception as restoreError:
                        print(f'restoring camera after scan failed: {restoreError}')
            finally:
                def finish(dt):
                    self.running = False
                    app._plate_run_active = False
                    self.run_status = 'Run stopped' if self._stop_all else final_status
                    if not self._teardown_requested:
                        app.bind_keys()
                Clock.schedule_once(finish)
                asyncio.get_event_loop().close()
        self._scan_thread = Thread(target= worker, daemon= True)
        self._scan_thread.start()

    def _scan(self, z: float = None) -> bool:
        if self.scan_mode == 'Continuous':
            return self._scan_continuous(z)
        app = App.get_running_app()
        if app.stage is None:
            print('connect the stage first')
            return False
        if app.plateCenter is None or app.plateRadius is None:
            print('calculate plate region first')
            return False
        if app.camera is None:
            print('connect the camera first')
            return False
        fov = app.get_fov_mm()
        if fov is None:
            print('no fov returned')
            return False

        threshold = self.scan_threshold
        min_pixels = self.scan_min_pixels
        settle = self.scan_settle
        z = self.scan_z if z is None else z

        tiles = macro.generate_scan_tiles(app.plateCenter, app.plateRadius, *fov,
                                          overlap_w= self.scan_overlap_w / 100.0,
                                          overlap_h= self.scan_overlap_h / 100.0)
        limits = [float(v) for v in app.config.get('Stage', 'stage_limits').split(',')]
        tiles = [(x, y) for (x, y) in tiles if 0 <= x <= limits[0]
                 and 0 <= y <= limits[1] and 0 <= z <= limits[2] and app.stage.is_safe(x, y, z)]
        if not tiles:
            print('no safe tiles to scan at this Z')
            return False

        speed_unit = app.config.get('Stage', 'speed_unit')
        accel_unit = app.config.get('Stage', 'acceleration_unit')
        precise_speed = float(app.config.get('Stage', 'precise_speed'))
        precise_accel = float(app.config.get('Stage', 'precise_acceleration'))
        scan_speed = float(app.config.get('Stage', 'scan_speed'))
        scan_accel = float(app.config.get('Stage', 'scan_acceleration'))

        found = False
        deadline = time.monotonic() + self.search_seconds
        try:
            scan_pass = 0
            while (not self._stop_scan and not self._stop_all
                   and scan_pass < self.search_passes and time.monotonic() < deadline):
                scan_pass += 1
                Clock.schedule_once(lambda dt: setattr(self, 'scan_progress', 0))
                t_move = t_settle = t_grab = t_detect = t_disp = 0.0
                n_tiles = 0
                pass_start = time.perf_counter()
                print(f'scan pass {scan_pass}')
                app.stage.set_motion(precise_speed, precise_accel, speed_unit, accel_unit)
                for i, (x, y) in enumerate(tiles):
                    if self._stop_scan or self._stop_all or time.monotonic() >= deadline:
                        break
                    frac = (i + 1) / len(tiles)
                    Clock.schedule_once(lambda dt, v=frac: setattr(self, 'scan_progress', v))
                    status = f'{self._active_plate_name or "Plate"} • Searching • pass {scan_pass}, tile {i + 1}/{len(tiles)}'
                    Clock.schedule_once(lambda dt, text=status: setattr(self, 'run_status', text))

                    t0 = time.perf_counter()
                    moved = app.stage.move_abs((x, y, z), 'mm', wait_until_idle= True)
                    t1 = time.perf_counter()
                    pos = app.stage.get_position(unit= 'mm', isAsync= False)
                    if (not moved) or pos is None \
                            or abs(pos[0] - x) > 1.0 or abs(pos[1] - y) > 1.0:
                        print(f'scan aborted: move did not reach target ({x:.2f}, {y:.2f}), got {pos}')
                        self._stop_scan = True
                        break
                    if i == 0:
                        app.stage.set_motion(scan_speed, scan_accel, speed_unit, accel_unit)
                    if self._wait_or_stop(settle):
                        break
                    t2 = time.perf_counter()
                    ok, img = app.camera.singleTake()
                    t3 = time.perf_counter()
                    if not ok:
                        print('failed to capture image, skipping tile')
                        continue
                    Clock.schedule_once(lambda dt, im=img: setattr(app, 'image', im))
                    t4 = time.perf_counter()
                    present, offset = macro.detect_worm(img, threshold, min_pixels)
                    t5 = time.perf_counter()

                    t_move += t1 - t0
                    t_settle += t2 - t1
                    t_grab += t3 - t2
                    t_disp += t4 - t3
                    t_detect += t5 - t4
                    n_tiles += 1
                    if present:
                        print('Found a worm !!')
                        units = app.config.get('Calibration', 'step_units')
                        for _ in range(int(self.scan_recenter_iters)):
                            if self._stop_scan or self._stop_all or time.monotonic() >= deadline:
                                break
                            dy, dx = macro.getStageDistances(
                                np.array([-offset[1], offset[0]]), app.imageToStageMat)
                            if not app.stage.move_rel(
                                    (dx, dy, 0), unit= units, wait_until_idle= True):
                                self._stop_scan = True
                                break
                            if self._wait_or_stop(settle):
                                break
                            ok2, img2 = app.camera.singleTake()
                            if not ok2:
                                break
                            Clock.schedule_once(lambda dt, im=img2: setattr(app, 'image', im))
                            present2, offset2 = macro.detect_worm(img2, threshold, min_pixels)
                            if not present2:
                                break
                            offset = offset2
                            if abs(offset[0]) <= self.scan_center_tol and abs(offset[1]) <= self.scan_center_tol:
                                break
                        if self._stop_scan or self._stop_all:
                            break
                        app.update_coordinates(isAsync= False)
                        found = True
                        break

                if n_tiles > 0:
                    pass_elapsed = time.perf_counter() - pass_start
                    per = lambda s: s / n_tiles * 1000.0
                    print(
                        f'pass {scan_pass}: move {per(t_move):.0f}ms | '
                        f'settle {per(t_settle):.0f}ms | grab {per(t_grab):.0f}ms | '
                        f'detect {per(t_detect):.0f}ms | disp {per(t_disp):.0f}ms | '
                        f'total {pass_elapsed / n_tiles * 1000.0:.0f}ms/tile | '
                        f'{n_tiles / pass_elapsed:.1f} tiles/s  ({n_tiles} tiles)'
                    )
                if found:
                    break
        finally:
            if app.stage is not None:
                app.stage.set_motion(precise_speed, precise_accel, speed_unit, accel_unit)
        return found

    def _wait_or_stop(self, duration):
        deadline = time.monotonic() + max(0.0, duration)
        while time.monotonic() < deadline:
            if self._stop_scan or self._stop_all:
                return True
            time.sleep(max(0.0, min(0.02, deadline - time.monotonic())))
        return self._stop_scan or self._stop_all

    def _begin_scan_camera(self):
        app = App.get_running_app()
        if app.camera is None or self._teardown_requested \
                or getattr(app, '_hardware_teardown', False):
            return False
        mrg = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager
        self._cam_saved = {
            'live' : mrg.liveviewbutton.state,
            'exposure' : app.camera.ExposureTime(),
            'gain' : app.camera.Gain(),
            'fr_enable' : app.camera.AcquisitionFrameRateEnable(),
            'fr' : app.camera.AcquisitionFrameRate()
        }
        Clock.schedule_once(lambda dt: setattr(mrg.liveviewbutton, 'state', 'normal'))
        t0 = time.perf_counter()
        while app.camera.IsGrabbing() and time.perf_counter() - t0 < 2.0:
            if self._stop_scan or self._stop_all or self._teardown_requested:
                if not self._teardown_requested \
                        and not getattr(app, '_hardware_teardown', False):
                    Clock.schedule_once(
                        lambda dt: setattr(
                            mrg.liveviewbutton, 'state', self._cam_saved['live']
                        )
                    )
                return False
            time.sleep(0.02)
        if app.camera.IsGrabbing():
            raise RuntimeError('camera did not stop before scan configuration')
        thread = getattr(mrg.liveviewbutton, 'imageAcquisitionThread', None)
        if thread is not None and thread is not current_thread() and thread.is_alive():
            thread.join(2)
            if thread.is_alive():
                raise RuntimeError('preview acquisition did not finish before scanning')
        if self._stop_scan or self._stop_all or self._teardown_requested:
            return False
        app.camera.AcquisitionFrameRateEnable.Value = False
        app.camera.ExposureTime.Value = float(self.scan_exposure)
        app.camera.Gain.Value = float(self.scan_gain)
        return True

    def _end_scan_camera(self, found):
        app = App.get_running_app()
        if app.camera is None or self._teardown_requested \
                or getattr(app, '_hardware_teardown', False):
            return
        mrg = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager
        if not found:
            app.camera.AcquisitionFrameRate.Value = self._cam_saved['fr']
            app.camera.AcquisitionFrameRateEnable.Value = self._cam_saved['fr_enable']
            app.camera.ExposureTime.Value = self._cam_saved['exposure']
            app.camera.Gain.Value = self._cam_saved['gain']
            Clock.schedule_once(lambda dt: setattr(mrg.liveviewbutton, 'state', self._cam_saved['live']))

    def stop_scan(self):
        # The scan and its tracking handoff are one cancellable operation.
        self.stop_plates()

    def _find_scan_z(self, searchDistance= None, numImages= None) -> float | None:
        app = App.get_running_app()
        stage = app.stage
        if app.camera is None or app.stage is None:
            print('camera or stage not connected')
            return None

        if searchDistance is None:
            searchDistance = self.scan_z_range
        if numImages is None:
            numImages = int(self.scan_z_frames)

        if app.plateCenter is None:
            print('no stage center found')
            return None
        if self._stop_scan or self._stop_all:
            return None
        if not stage.move_abs(
                (app.plateCenter[0], app.plateCenter[1], self.scan_z),
                'mm', wait_until_idle= True):
            print('failed to move to the Z-sweep position')
            return None
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        mainSide = app.config.get('DualColor', 'mainside')

        zStart = self.scan_z - searchDistance / 2
        zEnd = self.scan_z + searchDistance / 2

        sweeper = macro.IntensitySweeper()
        try:
            sweeper.sweep(
                app.camera, app.stage, zStart, zEnd, numImages,
                dualColorMode, mainSide,
                stopRequested=lambda: self._stop_scan or self._stop_all
            )
            scanZ = sweeper.findScanZ()
        except Exception as e:
            print(f'z-sweep failed: {e}')
            return None

        scanZ = float(scanZ)
        Clock.schedule_once(lambda dt, v=scanZ: setattr(self, 'scan_z', v))
        print(f'z-sweep picked scan_z = {scanZ:.4f} mm')
        return scanZ

    def run_plates(self, plates=None, record_duration=None):
        """Compatibility entry point; the visible run controls use start_run()."""
        if plates is not None:
            from uuid import uuid4
            profile = self._profile()
            if record_duration is not None:
                profile['track_interval'] = record_duration
            self.plates = [validate_plate({
                'id': uuid4().hex[:8], 'name': f'Plate {index + 1}',
                'center': list(center), 'radius': radius,
                'settings': dict(profile), 'enabled': True,
            }) for index, (center, radius) in enumerate(plates)]
            self.selected_plate = -1
        self.start_run()

    def _set_plate(self, center, radius):
        app = App.get_running_app()
        app.plateCenter = tuple(float(value) for value in center)
        app.plateRadius = float(radius)

    def stop_plates(self):
        if not self.running and not any(t is not None and t.is_alive()
                                        for t in (self._scan_thread, self._plates_thread)):
            return
        self._run_generation += 1
        self._stop_all = self._stop_scan = True
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        rc.isTracking = False
        rc.livefocuscheckbox.state = 'normal'
        rc.track_done.set()
        self.run_status = 'Stopping run…'
        if hasattr(self, '_resume_run'):
            self._resume_run.set()
        if app.stage is not None:
            app.stage.emergency_stop()

    def request_shutdown(self):
        self._run_generation += 1
        self._teardown_requested = True
        active = any(
            thread is not None and thread.is_alive()
            for thread in (self._scan_thread, self._plates_thread)
        )
        self._stop_all = True
        self._stop_scan = True
        app = App.get_running_app()
        rc = app.root.ids.middlecolumn.ids.runtimecontrols
        rc.isTracking = False
        rc.livefocuscheckbox.state = 'normal'
        rc.track_done.set()
        if active and app.stage is not None:
            app.stage.emergency_stop()

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        for thread in (self._scan_thread, self._plates_thread):
            if thread is None or thread is current_thread() or not thread.is_alive():
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        return all(
            thread is None or thread is current_thread() or not thread.is_alive()
            for thread in (self._scan_thread, self._plates_thread)
        )

    def toggle_preview(self):
        if self.running or not self.commit_fields():
            return
        app = App.get_running_app()
        if app.camera is None:
            return
        btn = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager.liveviewbutton
        if btn.state == 'down':
            btn.state = 'normal'
            if self._preview_saved is not None:
                exp, gain, fr_en, fr = self._preview_saved
                app.camera.ExposureTime.Value = exp
                app.camera.Gain.Value = gain
                app.camera.AcquisitionFrameRate.Value = fr
                app.camera.AcquisitionFrameRateEnable.Value = fr_en
                self._preview_saved = None
        else:
            self._preview_saved = (
                app.camera.ExposureTime(), app.camera.Gain(),
                app.camera.AcquisitionFrameRateEnable(), app.camera.AcquisitionFrameRate())
            app.camera.AcquisitionFrameRateEnable.Value = False
            app.camera.ExposureTime.Value = float(self.scan_exposure)
            app.camera.Gain.Value = float(self.scan_gain)
            btn.state = 'down'
