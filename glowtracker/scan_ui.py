"""The plate scan window: plate cards and run controls on the left, the stage map with the live
camera on the right; plus the plate editor and the recording dialog.

Built in Python on the scan panel (scan.py); the shared looks (FlatButton, DarkInput, SectionLabel,
CloseKey, DaqSpinner, the switch) come from layout.kv, the plate card's from the rule below.
"""

import os
from pathlib import Path

from kivy.app import App
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.switch import Switch
from kivy.uix.widget import Widget

from file_browser import FileBrowser, style_file_popup
from plate_map import CameraView, DishPreview, PlateMap, status_colour
from plate_plan import FIELDS, parse_setting

BLUE = (0.04, 0.52, 1, 1)
GREEN = (0.13, 0.55, 0.38, 1)
RED = (0.72, 0.18, 0.2, 1)
MUTED = (0.6, 0.6, 0.65, 1)
TEXT = (0.92, 0.92, 0.94, 1)

Builder.load_string('''
<PlateCard>:
    size_hint_y: None
    height: dp(64)
    padding: dp(10), dp(8), dp(12), dp(8)
    spacing: dp(12)
    canvas.before:
        Color:
            rgba: (0.17, 0.18, 0.21, 1) if self.selected else C_INPUT
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]
        Color:
            rgba: C_ACCENT if self.selected else (1, 1, 1, 0.05)
        Line:
            rounded_rectangle: self.x + dp(0.5), self.y + dp(0.5), self.width - dp(1), self.height - dp(1), dp(10)
            width: dp(1.3) if self.selected else 1

    Switch:
        size_hint_x: None
        width: dp(54)
        active: root.enabled
        disabled: root.locked
        on_active: root.toggled(self.active)

    # a small dish in the status colour
    Widget:
        size_hint_x: None
        width: dp(26)
        canvas:
            Color:
                rgba: 0.19, 0.2, 0.235, 1
            Ellipse:
                pos: self.center_x - dp(11), self.center_y - dp(11)
                size: dp(22), dp(22)
            Color:
                rgba: root.colour
            Line:
                circle: self.center_x, self.center_y, dp(11)
                width: dp(1.6)

    BoxLayout:
        orientation: 'vertical'
        Label:
            text: root.name
            bold: True
            font_size: '15sp'
            color: C_TEXT
            halign: 'left'
            valign: 'bottom'
            text_size: self.size
            shorten: True
        Label:
            text: root.detail
            font_size: '12.5sp'
            color: C_MUTED
            halign: 'left'
            valign: 'top'
            text_size: self.size
            shorten: True

    # the status as a tinted pill
    Label:
        text: root.status
        size_hint: None, None
        size: self.texture_size[0] + dp(20), dp(24)
        pos_hint: {'center_y': .5}
        font_size: '12sp'
        bold: True
        color: root.colour
        canvas.before:
            Color:
                rgba: root.colour[:3] + [0.16]
            RoundedRectangle:
                pos: self.pos
                size: self.size
                radius: [dp(12)]


<ViewPane>:
    orientation: 'vertical'
    padding: (dp(10), dp(4), dp(10), dp(10)) if self.inset else (0, 0, 0, 0)
    spacing: dp(4)
    canvas.before:
        Color:
            rgba: (0.075, 0.08, 0.09, 1) if self.inset else (0, 0, 0, 0)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(10)]
        Color:
            rgba: (1, 1, 1, 0.1) if self.inset else (0, 0, 0, 0)
        Line:
            rounded_rectangle: self.x, self.y, self.width, self.height, dp(10)
            width: 1
    BoxLayout:
        size_hint_y: None
        height: dp(34) if root.inset else 0
        opacity: 1 if root.inset else 0
        Label:
            text: root.title
            font_size: '12sp'
            color: C_MUTED
            halign: 'left'
            valign: 'middle'
            text_size: self.size
        SwapKey:
            id: swap
            pos_hint: {'center_y': .5}
            disabled: not root.inset
            tooltip: 'Swap the large view and this one'
    BoxLayout:
        id: body
        orientation: 'vertical'
        spacing: dp(4)

# two arrows, one each way
<SwapKey@ButtonBehavior+Widget>:
    size_hint: None, None
    size: dp(32), dp(32)
    canvas:
        Color:
            rgba: C_PRESSED if self.state == 'down' else C_BUTTON
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [dp(8)]
        Color:
            rgba: C_TEXT
        Line:
            points: self.x + dp(9), self.center_y + dp(4), self.right - dp(9), self.center_y + dp(4)
            width: dp(1.2)
        Line:
            points: self.right - dp(13), self.center_y + dp(8), self.right - dp(9), self.center_y + dp(4), self.right - dp(13), self.center_y
            width: dp(1.2)
        Line:
            points: self.right - dp(9), self.center_y - dp(4), self.x + dp(9), self.center_y - dp(4)
            width: dp(1.2)
        Line:
            points: self.x + dp(13), self.center_y, self.x + dp(9), self.center_y - dp(4), self.x + dp(13), self.center_y - dp(8)
            width: dp(1.2)
''')


class ViewPane(BoxLayout):
    """A view with a title. As an `inset` it is framed, sits over the large view, and shows the
    key that swaps the two."""
    title = StringProperty('')
    inset = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.body = self.ids.body
        self.swap_key = self.ids.swap


class PlateCard(ButtonBehavior, BoxLayout):
    """One plate in the run list: on/off switch, status dish, name, details and status pill."""
    name = StringProperty('')
    detail = StringProperty('')
    status = StringProperty('Ready')
    colour = ListProperty([0.42, 0.43, 0.48, 1])
    selected = BooleanProperty(False)
    enabled = BooleanProperty(True)
    locked = BooleanProperty(False)
    index = NumericProperty(-1)
    panel = ObjectProperty(None, allownone=True)

    def toggled(self, active):
        if self.panel is not None and active != self.enabled and not self.locked:
            self.panel.enable_plate(self.index, active)

    def on_release(self):
        if self.panel is not None and not self.locked:
            self.panel.select_plate(self.index)


def label(text='', size=13.5, colour=TEXT, height=None, wrap=True, **kwargs):
    widget = Label(text=text, font_size=sp(size), color=colour, halign='left', valign='middle', **kwargs)
    if height is not None:
        widget.size_hint_y = None
        widget.height = dp(height)
    if wrap:
        widget.bind(size=lambda w, s: setattr(w, 'text_size', (s[0], None if height is None else s[1])))
    return widget


def button(text, callback, tint=None, width=None, height=38):
    widget = Factory.FlatButton(text=text)
    widget.height = dp(height)
    if tint is not None:
        widget.tint = tint
    if width is not None:
        widget.size_hint_x = None
        widget.width = dp(width)
    widget.bind(on_release=lambda *_: callback())
    return widget


def row(*widgets, height=38, spacing=8):
    box = BoxLayout(size_hint_y=None, height=dp(height), spacing=dp(spacing))
    for widget in widgets:
        box.add_widget(widget)
    return box


def text_input(text='', hint=''):
    widget = Factory.DarkInput(text=text, hint_text=hint)
    widget.bind(focus=lambda _, focused: App.get_running_app().toggle_key_binding(focused))
    return widget


def switch_row(text, active, on_change, height=38):
    toggle = Switch(active=active, size_hint_x=None, width=dp(60))
    toggle.bind(active=lambda _, value: on_change(value))
    return row(label(text), toggle, height=height), toggle


def section(text):
    widget = Factory.SectionLabel(text=text)
    widget.height = dp(30)
    return widget


def column(spacing=8):
    box = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(spacing))
    box.bind(minimum_height=box.setter('height'))
    return box


class ScanFieldRow(BoxLayout):
    """A setting: its name, a right-aligned number field, and a short error under it."""

    def __init__(self, panel, key, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=dp(40), **kwargs)
        self.panel, self.key = panel, key
        self.input = text_input(str(getattr(panel, key)))
        self.input.halign = 'right'
        self.input.size_hint_x = None
        self.input.width = dp(130)
        self.error = Label(text='', font_size=sp(12), color=(1, 0.62, 0.3, 1), halign='right',
                           size_hint_y=None, height=0)
        self.error.bind(size=lambda w, s: setattr(w, 'text_size', s), text=self._fit)
        self.add_widget(row(label(FIELDS[key][0], 14), self.input, height=38, spacing=12))
        self.add_widget(self.error)
        self.input.bind(focus=lambda _, focused: self.commit() if not focused else None,
                        on_text_validate=lambda *_: self.commit())
        panel.bind(**{key: self.refresh})

    def _fit(self, *args):
        self.error.height = dp(18) if self.error.text else 0
        self.height = dp(40) + self.error.height

    def refresh(self, *args):
        if not self.input.focus:
            self.input.text = f'{getattr(self.panel, self.key):g}'
            self.error.text = ''

    def commit(self):
        try:
            value = parse_setting(self.key, self.input.text)
        except ValueError as error:
            self.error.text = str(error).split(': ', 1)[-1]
            self.panel.run_status = str(error)
            return False
        setattr(self.panel, self.key, value)
        self.error.text = ''
        return True


ScanNumberField = ScanFieldRow      # the old name


def _short(status: str) -> str:
    """A status pill's text: the part before any explanation."""
    return (status or 'Ready').split(' — ')[0].split(' • ')[0].rstrip('…. ')[:22]


def _duration(seconds: float) -> str:
    return f'{seconds / 60:g} min' if seconds >= 60 else f'{seconds:g} s'


def build_scan_ui(panel):
    panel.orientation = 'horizontal'
    panel.spacing = dp(22)
    panel.padding = 0
    app = App.get_running_app()

    # ---- left: plates and the run ------------------------------------------------------------
    left = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=dp(10))
    panel.add_widget(left)

    # saved runs: the whole plate list at once
    left.add_widget(section('Saved runs'))
    runs = Factory.DaqSpinner(text='Choose a saved run…', values=panel.run_presets)
    runs.size_hint_x = 1
    panel.bind(run_presets=lambda _, names: setattr(runs, 'values', names))
    load_run = button('Load', lambda: panel.load_run_preset(runs.text), width=80)
    save_run = button('Save as…', lambda: open_save_run(panel, runs), width=100)

    def update_runs(*args):
        load_run.disabled = panel.running or runs.text not in panel.run_presets
        save_run.disabled = panel.running or not panel.plates
        runs.disabled = panel.running
    runs.bind(text=update_runs)
    panel.bind(run_presets=update_runs, running=update_runs, plates=update_runs)
    update_runs()
    left.add_widget(row(runs, load_run, save_run))
    left.add_widget(Widget(size_hint_y=None, height=dp(4)))

    add = button('+  Add plate', lambda: open_plate_settings(panel), width=130, height=32)
    left.add_widget(row(section('Plates'), Widget(), add, height=34))
    left.add_widget(label('Visited from top to bottom. Tap a plate to select it.', 12.5, MUTED, height=18))

    plate_list = column(spacing=8)
    plate_scroll = ScrollView(do_scroll_x=False, bar_width=dp(4))
    plate_scroll.add_widget(plate_list)
    left.add_widget(plate_scroll)

    def refresh_plates(*args):
        plate_list.clear_widgets()
        if not panel.plates:
            empty = label('No plates yet.\nAdd one and capture three points on its rim.', 14, MUTED, height=70)
            empty.halign = 'center'
            plate_list.add_widget(empty)
        for index, plate in enumerate(panel.plates):
            settings = plate['settings']
            card = PlateCard(
                name=plate['name'], index=index, panel=panel,
                detail=f'{_duration(settings["track_interval"])} per visit  ·  {settings.get("scan_mode", "Sequential")}'
                       f'  ·  Ø {2 * plate["radius"]:.0f} mm',
                status=_short(plate.get('status', 'Ready')),
                colour=list(status_colour(plate.get('status', ''), plate.get('enabled', True))),
                selected=index == panel.selected_plate, enabled=plate.get('enabled', True),
                locked=panel.running)
            plate_list.add_widget(card)
    panel.bind(plates=refresh_plates, selected_plate=refresh_plates, running=refresh_plates)
    refresh_plates()

    edit = button('Edit', lambda: open_plate_settings(panel, panel.selected_plate))
    remove = button('Remove', panel.remove_plate)
    remove.color = (1, 0.45, 0.45, 1)

    def update_actions(*args):
        add.disabled = panel.running
        edit.disabled = remove.disabled = panel.running or panel.selected_plate < 0
    panel.bind(running=update_actions, selected_plate=update_actions)
    update_actions()
    left.add_widget(row(edit, remove))

    panel._plate_editor = build_plate_editor(panel)

    left.add_widget(Widget(size_hint_y=None, height=dp(6)))
    left.add_widget(section('Run'))
    repeat_row, repeat = switch_row('Repeat visits until stopped', panel.repeat_run,
                                    lambda value: setattr(panel, 'repeat_run', value))
    panel.bind(running=lambda _, value: setattr(repeat, 'disabled', value),
               repeat_run=lambda _, value: setattr(repeat, 'active', value))
    left.add_widget(repeat_row)
    estimate = label(panel.cycle_summary, 12.5, MUTED, height=36)
    panel.bind(cycle_summary=lambda _, text: setattr(estimate, 'text', text))
    left.add_widget(estimate)

    preview = button('Live preview', panel.toggle_preview)
    record = button('Recording: off', lambda: open_record_settings(panel))
    panel.bind(record_enabled=lambda _, active: setattr(record, 'text', 'Recording: on' if active else 'Recording: off'))
    left.add_widget(row(preview, record))

    # Start, or Pause / Stop while a run is going
    controls = BoxLayout(size_hint_y=None, height=dp(44), spacing=dp(8))
    start = button('Start run', panel.start_run, tint=GREEN, height=44)
    start.bold = True
    pause = button('Pause after plate', panel.toggle_pause, height=44)
    stop = button('Stop run', panel.stop_plates, tint=RED, height=44)
    stop.bold = True

    def update_run(*args):
        controls.clear_widgets()
        for widget in ((pause, stop) if panel.running else (start,)):
            controls.add_widget(widget)
        preview.disabled = record.disabled = panel.running
    panel.bind(running=update_run)
    panel.bind(paused=lambda _, value: setattr(pause, 'text', 'Resume run' if value else 'Pause after plate'),
               pause_requested=lambda _, value: setattr(pause, 'text', 'Cancel pause' if value else 'Pause after plate'))
    update_run()
    left.add_widget(controls)

    # ---- right: status, then the camera and the stage map ------------------------------------
    right = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=dp(10))
    panel.add_widget(right)

    dot = Widget(size_hint_x=None, width=dp(14))

    def draw_dot(*args):
        from kivy.graphics import Color, Ellipse
        dot.canvas.clear()
        colour = status_colour(panel.run_status) if panel.running else (0.42, 0.43, 0.48, 1)
        with dot.canvas:
            Color(*colour)
            Ellipse(pos=(dot.center_x - dp(5), dot.center_y - dp(5)), size=(dp(10), dp(10)))
    dot.bind(pos=draw_dot, size=draw_dot)
    panel.bind(run_status=draw_dot, running=draw_dot)
    status = label(panel.run_status, 14)
    status.shorten = True
    panel.bind(run_status=lambda _, text: setattr(status, 'text', text))
    close = Factory.CloseKey()
    close.bind(on_release=lambda *_: panel._popup.dismiss())
    right.add_widget(row(dot, status, close, height=38, spacing=10))

    # a slim progress line for the current search
    progress = Widget(size_hint_y=None, height=dp(4))

    def draw_progress(*args):
        from kivy.graphics import Color, RoundedRectangle
        progress.canvas.clear()
        with progress.canvas:
            Color(1, 1, 1, 0.07)
            RoundedRectangle(pos=progress.pos, size=progress.size, radius=[dp(2)])
            if panel.running and panel.scan_progress > 0:
                Color(*BLUE)
                RoundedRectangle(pos=progress.pos, size=(progress.width * min(1, panel.scan_progress), progress.height),
                                 radius=[dp(2)])
    progress.bind(pos=draw_progress, size=draw_progress)
    panel.bind(scan_progress=draw_progress, running=draw_progress)
    right.add_widget(progress)

    # one view fills the area, the other is a framed inset in a corner; the swap key on the
    # inset exchanges them
    plate_map = PlateMap()
    plate_map.panel = panel
    panel.ids.minimap = plate_map
    camera = CameraView(texture=app.texture)
    app.bind(texture=lambda _, texture: setattr(camera, 'texture', texture))

    map_pane = ViewPane(title='Stage')
    map_pane.body.add_widget(plate_map)
    camera_pane = ViewPane(title='Camera')
    camera_pane.body.add_widget(camera)

    stage = FloatLayout()
    right.add_widget(stage)
    panel._large_view = 'map'

    def arrange(*args):
        stage.clear_widgets()
        large, small = (map_pane, camera_pane) if panel._large_view == 'map' else (camera_pane, map_pane)
        large.size_hint, large.pos_hint, large.inset = (1, 1), {'x': 0, 'y': 0}, False
        small.size_hint, small.inset = (0.42, 0.42), True
        stage.add_widget(large)
        stage.add_widget(small)
        place_inset()

    def place_inset(*args):
        """Put the inset in the corner that hides the least: of the plates when the map is large,
        away from the scale bar and FOV when the camera is."""
        small = camera_pane if panel._large_view == 'map' else map_pane
        if panel._large_view != 'map':
            small.pos_hint = {'right': 0.985, 'top': 0.985}
            return
        corners = ({'right': 0.985, 'top': 0.985}, {'right': 0.985, 'y': 0.06},
                   {'x': 0.015, 'top': 0.985}, {'x': 0.015, 'y': 0.06})
        w, h = stage.width * small.size_hint_x, stage.height * small.size_hint_y
        scale = plate_map._fit()[0]
        dishes = []
        for index, plate in enumerate(panel.plates):
            cx, cy = plate_map.to_px(*plate['center'])
            r = plate['radius'] * scale + dp(20)       # with the name under it
            weight = 3 if index == panel.selected_plate else 1   # hide the selected plate least
            dishes.append((cx - r, cy - r, cx + r, cy + r, weight))
        legend = plate_map.legend_box()
        if legend is not None:
            dishes.append((*legend, 2))

        def hidden(corner):
            x = stage.x + (stage.width * corner['x'] if 'x' in corner else stage.width * corner['right'] - w)
            y = stage.y + (stage.height * corner['y'] if 'y' in corner else stage.height * corner['top'] - h)
            return sum(weight * max(0, min(x + w, x1) - max(x, x0)) * max(0, min(y + h, y1) - max(y, y0))
                       for x0, y0, x1, y1, weight in dishes)
        best = min(corners, key=hidden)
        if small.pos_hint != best:
            small.pos_hint = best
    plate_map.bind(size=place_inset, pos=place_inset)       # once the map has its real size
    panel.bind(plates=place_inset, selected_plate=place_inset)

    def swap():
        panel._large_view = 'camera' if panel._large_view == 'map' else 'map'
        arrange()
    for pane in (map_pane, camera_pane):
        pane.swap_key.bind(on_release=lambda *_: swap())
    arrange()


def build_plate_editor(panel):
    """The Add plate / Plate settings dialog: the plate and its rim on the left, the scan and
    tracking settings on the right."""
    content = BoxLayout(orientation='vertical', spacing=dp(14))
    body = BoxLayout(spacing=dp(28))
    content.add_widget(body)

    # ---- left: name, rim points and their preview, presets -----------------------------------
    left = BoxLayout(orientation='vertical', size_hint_x=0.44, spacing=dp(10))
    body.add_widget(left)
    name = text_input('Plate 1', 'Plate name')
    panel.ids.scenarioname = name
    left.add_widget(section('Plate'))
    left.add_widget(name)

    left.add_widget(section('Rim'))
    left.add_widget(label('Move the stage to three points on the plate’s rim and capture each one.',
                          12.5, MUTED, height=34))
    preview = DishPreview()
    preview.panel = panel
    left.add_widget(preview)

    count = label('0 of 3 points', 13.5, height=24)
    panel.ids.countlabel = count
    panel.bind(points=lambda _, points: setattr(
        count, 'text', f'{len(points)} of 3 points' if len(points) < 3 else f'{len(points)} points  ·  circle fitted'))
    result = label('Diameter: -    Center: -', 12.5, MUTED, height=20)
    panel.ids.resultlabel = result
    left.add_widget(count)
    left.add_widget(result)
    capture = button('Capture rim point', panel.capture_points, tint=BLUE)
    left.add_widget(row(capture, button('Reset', panel.reset, width=100)))
    left.add_widget(row(button('Use current Z', panel.use_current_z),
                        button('Live preview', panel.toggle_preview)))

    presets = Factory.DaqSpinner(text='Saved plate…', values=panel.saved_scenarios)
    presets.size_hint_x = 1
    panel.ids.scenariospinner = presets
    panel.bind(saved_scenarios=lambda _, names: setattr(presets, 'values', names))

    def load_preset():
        selected = panel.selected_plate
        panel.load_scenario(presets.text)
        panel.selected_plate = selected
    left.add_widget(section('Presets'))
    left.add_widget(row(presets, button('Load', load_preset, width=80),
                        button('Save', lambda: panel.save_scenario(name.text), width=80)))

    # ---- right: settings ---------------------------------------------------------------------
    scroll = ScrollView(do_scroll_x=False, size_hint_x=0.56, bar_width=dp(4))
    setup = column(spacing=6)
    setup.padding = (0, 0, dp(14), 0)      # room for the scroll bar
    scroll.add_widget(setup)
    body.add_widget(scroll)

    panel._fields = []
    mode = Factory.DaqSpinner(text=panel.scan_mode, values=('Sequential', 'Continuous'))
    mode.width = dp(150)
    mode.bind(text=lambda _, value: setattr(panel, 'scan_mode', value))
    panel.bind(scan_mode=lambda _, value: setattr(mode, 'text', value))
    setup.add_widget(section('Scan'))
    setup.add_widget(row(label('Scan mode', 14), mode, spacing=12))
    mode_help = label('', 12.5, MUTED, height=36)

    def update_mode_help(*args):
        mode_help.text = ('Captures while sweeping each row; speed is Stage settings → Scan speed.'
                          if panel.scan_mode == 'Continuous' else 'Stops and captures at each tile.')
    panel.bind(scan_mode=update_mode_help)
    update_mode_help()
    setup.add_widget(mode_help)
    for title, keys in (
        ('Find an animal', ('scan_z', 'scan_exposure', 'scan_gain', 'search_passes')),
        ('Track each visit', ('track_interval', 'track_exposure', 'track_gain', 'track_framerate',
                             'focus_settle_seconds', 'exposure_ramp_seconds')),
    ):
        setup.add_widget(section(title))
        for key in keys:
            field = ScanFieldRow(panel, key)
            panel._fields.append(field)
            setup.add_widget(field)
    setup.add_widget(label('The exposure ramps in four equal steps with a focus pause after each. '
                           'Tracking stays on; recording starts once it has settled.', 12.5, MUTED, height=40))

    advanced = column(spacing=6)
    for key in ('scan_settle', 'scan_threshold', 'scan_min_pixels', 'scan_overlap_w',
                'scan_overlap_h', 'scan_z_range', 'scan_z_frames'):
        field = ScanFieldRow(panel, key)
        panel._fields.append(field)
        advanced.add_widget(field)
    manual_points = text_input(hint='x,y; x,y; x,y')
    advanced.add_widget(label('Rim points typed in (mm)', 14, height=26))
    advanced.add_widget(row(manual_points, button('Set points', lambda: panel.set_points_from_text(manual_points.text),
                                                  width=110)))

    toggle = Factory.FlatButton(text='Show advanced settings')
    toggle.tint = (0, 0, 0, 0)
    toggle.pressed_tint = (1, 1, 1, 0.05)
    toggle.color = BLUE
    toggle.halign = 'left'
    toggle.bind(size=lambda w, s: setattr(w, 'text_size', s))
    toggle.valign = 'middle'

    def toggle_advanced(*args):
        if advanced.parent:
            setup.remove_widget(advanced)
            toggle.text = 'Show advanced settings'
        else:
            setup.add_widget(advanced)
            toggle.text = 'Hide advanced settings'
    toggle.bind(on_release=toggle_advanced)
    setup.add_widget(toggle)

    # ---- footer ------------------------------------------------------------------------------
    message = label('', 13, MUTED)
    message.shorten = True
    panel.bind(run_status=lambda _, text: setattr(message, 'text', text))
    content.add_widget(row(message, button('Cancel', lambda: panel._plate_popup.dismiss(), width=120),
                           button('Save plate', lambda: save_plate_editor(panel), tint=BLUE, width=140)))
    return content


def open_plate_settings(panel, index=None):
    if panel.running or (index is not None and not 0 <= index < len(panel.plates)):
        return
    app = App.get_running_app()
    from copy import deepcopy
    panel._editor_snapshot = {
        'selected': panel.selected_plate, 'profile': panel._profile(),
        'points': deepcopy(list(panel.points)),
        'center': list(app.plateCenter) if app.plateCenter is not None else None,
        'radius': app.plateRadius, 'name': panel.ids.scenarioname.text,
        'result': panel.ids.resultlabel.text, 'status': panel.run_status,
    }
    panel._editor_saved = False
    if index is None:
        panel.new_plate()
    else:
        panel.select_plate(index)
    for field in panel._fields:
        field.refresh()
    if panel._plate_editor.parent is not None:
        panel._plate_editor.parent.remove_widget(panel._plate_editor)
    popup = Popup(title='Add plate' if index is None else 'Plate settings',
                  content=panel._plate_editor, size_hint=(0.8, 0.9), auto_dismiss=False)
    style_file_popup(popup)
    panel._plate_popup = popup

    def dismissed(*args):
        for widget in panel._plate_editor.walk():
            if hasattr(widget, 'focus'):
                widget.focus = False
        if not panel._editor_saved:
            snapshot = panel._editor_snapshot
            panel.selected_plate = snapshot['selected']
            for key, value in snapshot['profile'].items():
                setattr(panel, key, value)
            panel.points = snapshot['points']
            app.plateCenter, app.plateRadius = snapshot['center'], snapshot['radius']
            panel.ids.scenarioname.text = snapshot['name']
            panel.ids.resultlabel.text = snapshot['result']
            panel.run_status = snapshot['status']
        for field in panel._fields:
            field.refresh()
    popup.bind(on_dismiss=dismissed)
    popup.open()


def save_plate_editor(panel):
    if panel.store_plate():
        panel._editor_saved = True
        panel._plate_popup.dismiss()


def open_record_settings(panel):
    if not panel.commit_fields():
        return
    content = BoxLayout(orientation='vertical', spacing=dp(12))
    enabled_row, enabled = switch_row('Record each tracking visit', panel.record_enabled, lambda value: None)
    content.add_widget(enabled_row)
    run_name = text_input(panel.record_name, 'Run name')
    extension = Factory.DaqSpinner(text=panel.record_format, values=('tiff', 'png'))
    extension.width = dp(120)
    content.add_widget(row(label('Run name', 14, size_hint_x=None, width=dp(90)), run_name,
                           label('Format', 14, size_hint_x=None, width=dp(60)), extension, spacing=12))

    content.add_widget(Factory.SectionLabel(text='Folder'))
    browser = FileBrowser(dirselect=True)
    browser.path = panel.record_directory or str(Path.home())
    content.add_widget(browser)
    destination = text_input(panel.record_directory, 'Recording folder')
    browser.bind(path=lambda _, path: setattr(destination, 'text', path),
                 selection=lambda _, selection: setattr(destination, 'text', selection[0]) if selection else None)
    error = label('', 12.5, (1, 0.45, 0.45, 1), height=0)
    content.add_widget(row(label('Save in', 13, MUTED, size_hint_x=None, width=dp(60)), destination, spacing=10))
    content.add_widget(error)
    content.add_widget(label('Each visit records for its plate’s tracking time. Files: run / plate / visit; '
                             'the framerate comes from the plate.', 12.5, MUTED, height=34))
    popup = Popup(title='Recording', content=content, size_hint=(0.7, 0.85), auto_dismiss=False)
    style_file_popup(popup)

    def apply(start=False):
        panel.record_enabled = enabled.active
        panel.record_directory = destination.text.strip()
        panel.record_name = run_name.text.strip() or 'plate_run'
        panel.record_format = extension.text
        if panel.record_enabled and not os.path.isdir(panel.record_directory):
            error.text = 'Choose an existing folder to record into'
            error.height = dp(18)
            return
        popup.dismiss()
        if start:
            panel.start_run()
    content.add_widget(row(Widget(), button('Cancel', popup.dismiss, width=110),
                           button('Save', apply, tint=BLUE, width=110),
                           button('Save and start', lambda: apply(True), tint=GREEN, width=150)))
    popup.open()


def open_save_run(panel, spinner):
    """Ask for a name and save the plate list and run options under it."""
    content = BoxLayout(orientation='vertical', spacing=dp(12))
    current = spinner.text if spinner.text in panel.run_presets else ''
    name = text_input(current, 'Name of the run, e.g. Chrimson 4 plates')
    note = label('', 12.5, MUTED, height=20)

    def update_note(*args):
        note.text = (f'Replaces the saved run “{name.text.strip()}”.' if name.text.strip() in panel.run_presets
                     else f'Saves {len(panel.plates)} plate{"s" if len(panel.plates) != 1 else ""} and the run options.')
    name.bind(text=update_note)
    update_note()
    content.add_widget(name)
    content.add_widget(note)
    content.add_widget(Widget())
    popup = Popup(title='Save run', content=content, size_hint=(None, None), size=(dp(460), dp(230)),
                  auto_dismiss=False)
    style_file_popup(popup)

    def save():
        if panel.save_run_preset(name.text):
            spinner.text = name.text.strip()
            popup.dismiss()
        else:
            note.text = panel.run_status
    name.bind(on_text_validate=lambda *_: save())
    content.add_widget(row(Widget(), button('Cancel', popup.dismiss, width=110),
                           button('Save', save, tint=BLUE, width=110)))
    popup.open()
    name.focus = True
