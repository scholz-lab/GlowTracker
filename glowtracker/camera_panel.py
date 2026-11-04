"""Camera settings without the Basler app: a panel with the imaging settings and the user's presets
(camera_presets), to switch e.g. between brightfield and fluorescence with one click. It floats
next to the left column without blocking the window, so the live image shows what a change does.
The look is in layout.kv.
"""
from __future__ import annotations

import math

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

import camera_presets
import run_records

REFRESH_S = 0.5
SEND_S = 0.1                # a dragged slider updates the camera at most this often
BRIGHT_WORM = 'Bright on dark'
DARK_WORM = 'Dark on bright'

# (camera setting, label, on a log scale): exposure and frame rate span several decades
ROWS = (('ExposureTime', 'Exposure (µs)', True), ('AcquisitionFrameRate', 'Frame rate (fps)', True),
        ('Gain', 'Gain (dB)', False), ('BlackLevel', 'Black level', False), ('Gamma', 'Gamma', False))

TIPS = {
    'ExposureTime': 'How long the sensor collects light for each frame. Longer gives a brighter image, but a '
                    'moving animal blurs and the frame rate can drop. Usually the first thing to change.',
    'AcquisitionFrameRate': 'Frames per second the camera aims for. Higher follows fast movement better but '
                            'gives more data to save. It cannot go above what the exposure allows.',
    'Gain': 'Amplifies the signal after it is collected, so it brightens the noise as well. Prefer more light '
            'or a longer exposure, and use gain when those are not possible. 12-bit allows less gain than 8-bit.',
    'BlackLevel': 'Adds a constant to every pixel, which lifts the darkest parts of the image. Keep it low: it '
                  'adds no signal and reduces contrast.',
    'Gamma': 'Bends brightness to bring out dark or bright parts. 1 keeps brightness proportional to light; keep '
             '1 for anything you measure.',
}


def data_root(app) -> str:
    return run_records.data_root(app.config.get('Experiment', 'dataroot', fallback=''))


def tracking_values(app) -> dict:
    return {'dark_bg': app.config.getboolean('Tracking', 'dark_bg'),
            'threshold': app.config.getint('Tracking', 'threshold')}


def busy_reason(app) -> str:
    """Why imaging cannot be changed now, or ''."""
    try:
        runtime = app.root.ids.middlecolumn.runtimecontrols
        if runtime.imageacquisitionmanager.recordbutton.state == 'down':
            return 'Locked while recording, so the whole recording has the same settings.'
        if runtime.isTracking:
            return 'Stop tracking first: it reads its settings when it starts.'
    except (AttributeError, KeyError):
        pass
    return ''


def change_while_idle(app, pixel_format: str | None = None, binning: int | None = None, mode: str | None = None) -> None:
    """What the camera only takes while idle. Live view pauses, and pixel distances follow."""
    camera, left = app.camera, app.root.ids.leftcolumn

    def change():
        if pixel_format is not None:
            camera.setPixelFormat(pixel_format)
        if binning is not None:
            camera.setBinning(binning, mode)
        left.noteImageGeometry()

    left.whileLiveStopped(change)


def apply_preset(app, preset: dict) -> None:
    """Set the camera and tracking as the preset says."""
    camera = app.camera
    values = preset['camera']
    now = camera.readSettings()
    wanted = {key: values[key] for key in ('PixelFormat', 'Binning', 'BinningMode')
              if key in values and key in now and values[key] != now[key]}
    if wanted:
        binning = values.get('Binning', now.get('Binning')) if {'Binning', 'BinningMode'} & set(wanted) else None
        change_while_idle(app, wanted.get('PixelFormat'), binning, values.get('BinningMode'))
    problems = camera.applySettings(values)
    for key, value in (preset.get('tracking') or {}).items():
        app.config.set('Tracking', key, int(value))
    app.config.set('Camera', 'preset', preset['name'])
    app.config.write()
    app.root.ids.leftcolumn.update_settings_display()
    app.log('hw', f'camera preset: {preset["name"]}')
    for problem in problems:
        app.log('hw', problem, 'warn')


class SettingRow(BoxLayout):
    """One camera setting: a slider and the exact value, confirmed with Enter."""
    title = StringProperty('')
    low = NumericProperty(0)
    high = NumericProperty(1)

    def __init__(self, panel, name: str, title: str, log: bool, limits: tuple, **kwargs):
        super().__init__(**kwargs)
        self.panel, self.name, self.log = panel, name, log
        self.title = title
        low = max(limits[0], 1e-3) if log else limits[0]
        self.low, self.high = self._toSlider(low), self._toSlider(limits[1])
        self.dragging = False
        self._sendSoon = Clock.create_trigger(lambda dt: self._send(self._fromSlider(self.ids.slider.value)), SEND_S)

    def _toSlider(self, value: float) -> float:
        return math.log10(max(value, 1e-3)) if self.log else value

    def _fromSlider(self, value: float) -> float:
        return 10 ** value if self.log else value

    def on_touch_down(self, touch):
        self.dragging = self.ids.slider.collide_point(*touch.pos) and not self.disabled
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        handled = super().on_touch_up(touch)
        if self.dragging:
            self.dragging = False
            self._send(self._fromSlider(self.ids.slider.value))
        return handled

    def sliderMoved(self) -> None:
        if self.dragging:
            self.ids.field.text = self._format(self._fromSlider(self.ids.slider.value))
            self._sendSoon()

    def typed(self) -> None:
        try:
            self._send(float(self.ids.field.text.replace(',', '.')))
        except ValueError:
            pass                    # the next refresh shows the real value again

    def _send(self, value: float) -> None:
        self.panel.setValues({self.name: value})

    @staticmethod
    def _format(value: float) -> str:
        return f'{value:.0f}' if abs(value) >= 100 else f'{value:.2f}'

    def show(self, value: float) -> None:
        if not self.dragging and not self.ids.field.focus:
            self.ids.slider.value = self._toSlider(value)
            self.ids.field.text = self._format(value)


class CameraSettingsPanel(BoxLayout):
    keepsTooltips = True
    hint = StringProperty('')
    locked = BooleanProperty(False)

    def __init__(self, left, **kwargs):
        super().__init__(**kwargs)
        self.left = left
        self.app = App.get_running_app()
        self._camera = None         # the camera and pixel format the rows were built for
        self._format = None
        self.rows: dict[str, SettingRow] = {}
        self._active = None
        self._refreshEvent = Clock.schedule_interval(self.refresh, REFRESH_S)
        self.bind(size=self.place)
        Window.bind(size=self.place)
        self.showPresets()
        self.refresh()

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            super().on_touch_down(touch)
            return True             # not through to the image underneath
        return False

    def place(self, *args) -> None:
        self.x = self.left.right + dp(8)
        self.top = Window.height - dp(8)

    def close(self) -> None:
        self._refreshEvent.cancel()
        Window.unbind(size=self.place)
        Window.remove_widget(self)
        self.left.cameraPanel = None

    # --- the settings ------------------------------------------------------------------------
    def _buildRows(self, camera) -> None:
        box = self.ids.rows
        box.clear_widgets()
        self.rows = {}
        self._camera = camera
        self._format = None
        if camera is None:
            return
        self._format = camera.readSettings().get('PixelFormat')
        for name, title, log in ROWS:
            limits = camera.settingRange(name)
            if limits is not None:
                self.rows[name] = SettingRow(self, name, title, log, limits, tooltip=TIPS[name])
                box.add_widget(self.rows[name])
        self.ids.pixelformat.values = camera.monoFormats()
        self.ids.binning.values = [self.binningName(factor) for factor in camera.binningFactors()]

    @staticmethod
    def binningName(factor: int) -> str:
        return 'Off' if factor == 1 else f'{factor} x {factor}'

    def refresh(self, *args) -> None:
        camera = self.app.camera
        # the limits follow the pixel format, e.g. less gain in 12 bit
        if camera is not self._camera or (camera is not None and camera.readSettings().get('PixelFormat') != self._format):
            self._buildRows(camera)
        reason = busy_reason(self.app)
        self.locked = camera is None or bool(reason)
        if camera is None:
            self.hint = 'Connect the camera to change its settings.'
            return
        settings = camera.readSettings()
        for name, row in self.rows.items():
            if name in settings:
                row.show(settings[name])
        self.ids.pixelformat.text = settings.get('PixelFormat', '')
        self.ids.binning.text = self.binningName(int(settings.get('Binning', 1)))
        self.ids.binningmode.text = settings.get('BinningMode', '')
        self.ids.binningmode.disabled = int(settings.get('Binning', 1)) == 1
        tracking = tracking_values(self.app)
        self.ids.polarity.text = BRIGHT_WORM if tracking['dark_bg'] else DARK_WORM
        if not self.ids.threshold.focus:
            self.ids.threshold.text = str(tracking['threshold'])
        hint = reason
        wanted = settings.get('AcquisitionFrameRate', 0)
        actual = camera.ResultingFrameRate()
        if not hint and actual < wanted * 0.99:
            hint = f'The exposure limits the frame rate to {actual:.1f} fps.'
        self.hint = hint
        active = next((p['name'] for p in self.presets if camera_presets.matches(p, settings, tracking)), None)
        if active != self._active:
            self._active = active
            self.showPresets()

    def setValues(self, values: dict) -> None:
        if self.locked or self.app.camera is None:
            return
        try:
            problems = self.app.camera.applySettings(values)
        except Exception as e:
            problems = [f'Could not change the camera: {e}']
        for problem in problems:
            self.app.log('hw', problem, 'warn')
        self.left.update_settings_display()
        self.refresh()

    def setPixelFormat(self, name: str) -> None:
        camera = self.app.camera
        if self.locked or camera is None or not name or name == camera.readSettings().get('PixelFormat'):
            return
        try:
            change_while_idle(self.app, pixel_format=name)
            self.app.log('hw', f'pixel format: {name}')
        except Exception as e:
            self.app.log('hw', f'could not set the pixel format: {e}', 'warn')
        self.refresh()

    def setBinning(self, name: str, mode: str = '') -> None:
        camera = self.app.camera
        if self.locked or camera is None or not name:
            return
        settings = camera.readSettings()
        factor = 1 if name == 'Off' else int(name[0])
        mode = mode or settings.get('BinningMode')
        if factor == int(settings.get('Binning', 1)) and mode == settings.get('BinningMode'):
            return
        try:
            change_while_idle(self.app, binning=factor, mode=mode)
            self.app.log('hw', f'binning: {name} ({mode}), image {camera.Width()} x {camera.Height()}')
        except Exception as e:
            self.app.log('hw', f'could not set the binning: {e}', 'warn')
        self.refresh()

    def setPolarity(self, text: str) -> None:
        if not self.locked and text in (BRIGHT_WORM, DARK_WORM):
            self.app.config.set('Tracking', 'dark_bg', int(text == BRIGHT_WORM))
            self.app.config.write()

    def setThreshold(self) -> None:
        try:
            value = int(float(self.ids.threshold.text))
        except ValueError:
            return
        if not self.locked:
            self.app.config.set('Tracking', 'threshold', value)
            self.app.config.write()

    # --- presets -----------------------------------------------------------------------------
    def showPresets(self) -> None:
        self.presets = camera_presets.list_presets(data_root(self.app))
        box = self.ids.presets
        box.clear_widgets()
        if not self.presets:
            box.add_widget(Factory.CameraPresetHint())
        for preset in self.presets:
            row = Factory.CameraPresetRow()
            row.ids.use.text = preset['name']
            if preset['name'] == self._active:
                row.ids.use.tint = (0.13, 0.55, 0.38, 1)
            row.ids.use.bind(on_release=lambda b, p=preset: self.usePreset(p))
            row.ids.delete.bind(on_release=lambda b, p=preset: self.deletePreset(b, p))
            box.add_widget(row)

    def usePreset(self, preset: dict) -> None:
        if self.locked:
            return
        try:
            apply_preset(self.app, preset)
        except Exception as e:
            self.app.log('hw', f'could not apply {preset["name"]}: {e}', 'error')
        self.refresh()

    def savePreset(self) -> None:
        name = self.ids.presetname.text.strip()
        button = self.ids.savebutton
        camera = self.app.camera
        if not name or camera is None:
            return
        root = data_root(self.app)
        if camera_presets.exists(root, name) and button.text != 'Replace?':
            button.text = 'Replace?'
            return
        try:
            camera_presets.save(root, name, camera.readSettings(), tracking_values(self.app))
        except OSError as e:
            self.app.log('hw', f'could not save the preset: {e}', 'error')
            return
        self.app.config.set('Camera', 'preset', name)
        self.app.config.write()
        self.app.log('hw', f'camera preset saved: {name}')
        self.ids.presetname.text = ''
        self._active = None
        self.showPresets()
        self.refresh()

    def deletePreset(self, button, preset: dict) -> None:
        if button.text != 'Sure?':
            button.text = 'Sure?'
            return
        try:
            camera_presets.delete(data_root(self.app), preset['name'])
        except OSError as e:
            self.app.log('hw', f'could not delete the preset: {e}', 'error')
        self._active = None
        self.showPresets()
