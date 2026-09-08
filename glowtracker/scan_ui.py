"""Plate setup, visit controls, and a separate recording dialog."""

import os
from pathlib import Path
from kivy.app import App
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.checkbox import CheckBox
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput

from plate_plan import FIELDS, parse_setting
from Scan_map import ScanMinimap


def label(text, height=28, **kwargs):
    return Label(text=text, size_hint_y=None, height=dp(height),
                 font_size=sp(13), **kwargs)


def button(text, callback):
    widget = Button(text=text, size_hint_y=None, height=dp(34),
                    font_size=sp(13), background_normal='',
                    background_color=(0.25, 0.25, 0.28, 1))
    widget.bind(on_release=lambda *_: callback())
    return widget


def row(*widgets):
    box = BoxLayout(size_hint_y=None, height=dp(36), spacing=dp(6))
    for widget in widgets:
        box.add_widget(widget)
    return box


def text_input(text='', hint=''):
    widget = TextInput(text=text, hint_text=hint, multiline=False,
                       size_hint_y=None, height=dp(34), font_size=sp(13))
    widget.bind(focus=lambda _, focused: App.get_running_app().toggle_key_binding(focused))
    return widget


class ScanNumberField(BoxLayout):
    def __init__(self, panel, key, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, height=dp(56), **kwargs)
        self.panel, self.key = panel, key
        self.input = text_input(str(getattr(panel, key)))
        self.error = label('', 18, color=(1, 0.65, 0.3, 1))
        self.add_widget(row(label(FIELDS[key][0], 34), self.input))
        self.add_widget(self.error)
        self.input.bind(focus=lambda _, focused: self.commit() if not focused else None,
                        on_text_validate=lambda *_: self.commit())
        panel.bind(**{key: self.refresh})

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


def build_scan_ui(panel):
    panel.orientation = 'horizontal'
    panel.spacing = dp(12)
    panel.padding = dp(10)
    left = BoxLayout(orientation='vertical', size_hint_x=0.46, spacing=dp(6))
    panel.add_widget(left)
    left.add_widget(label('Plates — visited from top to bottom'))
    plate_list = GridLayout(cols=1, size_hint_y=None, spacing=dp(4))
    plate_list.bind(minimum_height=plate_list.setter('height'))
    plate_scroll = ScrollView(do_scroll_x=False)
    plate_scroll.add_widget(plate_list)
    left.add_widget(plate_scroll)

    def refresh_plates(*args):
        plate_list.clear_widgets()
        if not panel.plates:
            plate_list.add_widget(label('Click Add plate to define your first plate.'))
        for index, plate in enumerate(panel.plates):
            check = CheckBox(active=plate.get('enabled', True), size_hint_x=None, width=dp(32))
            check.disabled = panel.running
            check.bind(active=lambda _, active, i=index: panel.enable_plate(i, active))
            name = f'{plate["name"]}  |  {plate["settings"]["track_interval"]:g}s  |  {plate.get("status", "Ready")}'
            select = button(name, lambda i=index: panel.select_plate(i))
            if index != panel.selected_plate:
                select.background_color = (0.15, 0.15, 0.17, 1)
            select.disabled = panel.running
            plate_list.add_widget(row(check, select))
    panel.bind(plates=refresh_plates, selected_plate=refresh_plates, running=refresh_plates)
    refresh_plates()
    add = button('Add plate', lambda: open_plate_settings(panel))
    edit = button('Edit selected', lambda: open_plate_settings(panel, panel.selected_plate))
    remove = button('Remove', panel.remove_plate)
    def update_actions(*args):
        add.disabled = panel.running
        edit.disabled = remove.disabled = panel.running or panel.selected_plate < 0
    panel.bind(running=update_actions, selected_plate=update_actions)
    update_actions()
    left.add_widget(row(add, edit, remove))

    panel._plate_editor = build_plate_editor(panel)

    repeat = CheckBox(active=panel.repeat_run, size_hint_x=None, width=dp(32))
    repeat.bind(active=lambda _, value: setattr(panel, 'repeat_run', value))
    panel.bind(running=lambda _, value: setattr(repeat, 'disabled', value))
    left.add_widget(row(repeat, label('Repeat visits until stopped', 34)))
    estimate = label(panel.cycle_summary, 40)
    estimate.bind(size=lambda instance, size: setattr(instance, 'text_size', size))
    panel.bind(cycle_summary=lambda _, text: setattr(estimate, 'text', text))
    left.add_widget(estimate)
    start = button('Start run', panel.start_run)
    preview = button('Live preview', panel.toggle_preview)
    record = button('Record…', lambda: open_record_settings(panel))
    panel.bind(record_enabled=lambda _, active: setattr(record, 'text', 'Record: ON…' if active else 'Record…'))
    panel.bind(running=lambda _, value: [setattr(w, 'disabled', value) for w in (start, preview, record)])
    left.add_widget(row(preview, record, start))
    pause = button('Pause after plate', panel.toggle_pause)
    panel.bind(paused=lambda _, value: setattr(pause, 'text', 'Resume run' if value else 'Pause after plate'),
               pause_requested=lambda _, value: setattr(pause, 'text', 'Cancel pause' if value else 'Pause after plate'),
               running=lambda _, value: setattr(pause, 'disabled', not value))
    pause.disabled = True
    stop = button('Stop run', panel.stop_plates)
    close = button('Close', lambda: panel._popup.dismiss())
    stop.disabled = True
    panel.bind(running=lambda _, active: setattr(stop, 'disabled', not active))
    left.add_widget(row(pause, stop, close))

    right = BoxLayout(orientation='vertical', size_hint_x=0.54, spacing=dp(6))
    status = label(panel.run_status, 52)
    status.bind(size=lambda instance, size: setattr(instance, 'text_size', size))
    panel.bind(run_status=lambda _, text: setattr(status, 'text', text))
    right.add_widget(status)
    preview_image = Image(fit_mode='contain', texture=App.get_running_app().texture)
    App.get_running_app().bind(texture=lambda _, texture: setattr(preview_image, 'texture', texture))
    right.add_widget(preview_image)
    minimap = ScanMinimap(size_hint_y=0.28, stage_min=[0, 0], stage_max=[152.4, 152.4])
    panel.ids.minimap = minimap
    right.add_widget(minimap)
    progress = ProgressBar(max=1, value=panel.scan_progress, size_hint_y=None, height=dp(18))
    panel.bind(scan_progress=lambda _, value: setattr(progress, 'value', value))
    right.add_widget(progress)
    panel.add_widget(right)


def build_plate_editor(panel):
    content = BoxLayout(orientation='vertical', spacing=dp(8), padding=dp(8))
    scroll = ScrollView(do_scroll_x=False)
    setup = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(5))
    setup.bind(minimum_height=setup.setter('height'))
    scroll.add_widget(setup)
    content.add_widget(scroll)
    name = text_input('Plate 1', 'Plate name')
    panel.ids.scenarioname = name
    setup.add_widget(row(label('Plate name', 34), name))
    setup.add_widget(label('Move to three points around the rim and capture each.', 32))
    count = label('Points: 0', 34)
    panel.ids.countlabel = count
    panel.bind(points=lambda _, points: setattr(count, 'text', f'Points: {len(points)}'))
    setup.add_widget(row(button('Capture rim point', panel.capture_points), count,
                         button('Reset points', panel.reset)))
    result = label('Diameter: -    Center: -', 36)
    result.font_size = sp(12)
    panel.ids.resultlabel = result
    setup.add_widget(result)
    setup.add_widget(row(button('Use current Z', panel.use_current_z),
                         button('Live preview', panel.toggle_preview)))
    presets = Spinner(text='Saved plate…', values=panel.saved_scenarios,
                      size_hint_y=None, height=dp(34))
    panel.ids.scenariospinner = presets
    panel.bind(saved_scenarios=lambda _, names: setattr(presets, 'values', names))
    def load_preset():
        selected = panel.selected_plate
        panel.load_scenario(presets.text)
        panel.selected_plate = selected
    setup.add_widget(row(presets, button('Load', load_preset),
                         button('Save preset', lambda: panel.save_scenario(name.text))))

    panel._fields = []
    mode = Spinner(text=panel.scan_mode, values=('Sequential', 'Continuous'),
                   size_hint_y=None, height=dp(34))
    mode.bind(text=lambda _, value: setattr(panel, 'scan_mode', value))
    panel.bind(scan_mode=lambda _, value: setattr(mode, 'text', value))
    setup.add_widget(row(label('Scan mode', 34), mode))
    mode_help = label('', 48)
    mode_help.font_size = sp(12)
    mode_help.bind(size=lambda widget, size: setattr(widget, 'text_size', size))
    def update_mode_help(*args):
        mode_help.text = ('Continuous: capture while sweeping each row.\n'
                          'Speed: Stage settings → Scan speed.'
                          if panel.scan_mode == 'Continuous' else
                          'Sequential: stop and capture at each tile.')
    panel.bind(scan_mode=update_mode_help)
    update_mode_help()
    setup.add_widget(mode_help)
    for title, keys in (
        ('Find an animal', ('scan_z', 'scan_exposure', 'scan_gain', 'search_seconds', 'search_passes')),
        ('Track each visit', ('track_interval', 'track_exposure', 'track_gain', 'track_framerate',
                             'focus_settle_seconds')),
    ):
        setup.add_widget(label(title))
        for key in keys:
            field = ScanNumberField(panel, key)
            panel._fields.append(field)
            setup.add_widget(field)
    advanced = BoxLayout(orientation='vertical', size_hint_y=None)
    advanced.bind(minimum_height=advanced.setter('height'))
    for key in ('scan_settle', 'scan_threshold', 'scan_min_pixels', 'scan_overlap_w',
                'scan_overlap_h', 'scan_z_range', 'scan_z_frames'):
        field = ScanNumberField(panel, key)
        panel._fields.append(field)
        advanced.add_widget(field)
    manual_points = text_input(hint='x,y; x,y; x,y')
    advanced.add_widget(row(manual_points, button('Set points', lambda: panel.set_points_from_text(manual_points.text))))
    advanced_toggle = button('+ Advanced', lambda: toggle_advanced())
    def toggle_advanced():
        if advanced.parent:
            setup.remove_widget(advanced)
            advanced_toggle.text = '+ Advanced'
        else:
            setup.add_widget(advanced)
            advanced_toggle.text = '- Advanced'
    setup.add_widget(advanced_toggle)

    message = label('', 38)
    message.bind(size=lambda widget, size: setattr(widget, 'text_size', size))
    panel.bind(run_status=lambda _, text: setattr(message, 'text', text))
    content.add_widget(message)
    content.add_widget(row(button('Cancel', lambda: panel._plate_popup.dismiss()),
                           button('Save plate', lambda: save_plate_editor(panel))))
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
                  content=panel._plate_editor, size_hint=(0.7, 0.9), auto_dismiss=False)
    panel._plate_popup = popup
    def dismissed(*args):
        for widget in panel._plate_editor.walk():
            if isinstance(widget, TextInput):
                widget.focus = False
        for field in panel._fields:
            field.input.focus = False
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
    content = BoxLayout(orientation='vertical', spacing=dp(8), padding=dp(12))
    enabled = CheckBox(active=panel.record_enabled, size_hint_x=None, width=dp(36))
    content.add_widget(row(enabled, label('Record each tracking visit', 34)))
    destination = text_input(panel.record_directory, 'Recording folder')
    run_name = text_input(panel.record_name, 'Run name')
    extension = Spinner(text=panel.record_format, values=('tiff', 'png'),
                        size_hint_y=None, height=dp(34))
    content.add_widget(row(label('Run name', 34), run_name))
    content.add_widget(row(label('Format', 34), extension))
    content.add_widget(destination)
    chooser = FileChooserListView(path=panel.record_directory or str(Path.home()),
                                  dirselect=True)
    chooser.bind(path=lambda _, path: setattr(destination, 'text', path))
    content.add_widget(chooser)
    content.add_widget(label('Each visit records for its plate’s tracking duration.\nFiles: run / plate / visit. Framerate comes from the plate.', 52))
    popup = Popup(title='Recording settings', content=content, size_hint=(0.65, 0.8), auto_dismiss=False)

    def apply(start=False):
        panel.record_enabled = enabled.active
        panel.record_directory = destination.text.strip()
        panel.record_name = run_name.text.strip() or 'plate_run'
        panel.record_format = extension.text
        if panel.record_enabled and not os.path.isdir(panel.record_directory):
            destination.hint_text = 'Choose an existing folder'
            destination.background_color = (1, 0.8, 0.55, 1)
            return
        popup.dismiss()
        if start:
            panel.start_run()
    content.add_widget(row(button('Cancel', popup.dismiss), button('Save settings', apply),
                           button('Start run', lambda: apply(True))))
    popup.open()
