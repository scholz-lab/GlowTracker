"""What the right-click menus of the live view and the minimap offer, and what their items do."""
from __future__ import annotations

import math
import os

import numpy as np
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.metrics import dp, sp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget

import image_scale
import Microscope_macros as macro
import run_records
import stage_markers
from Advanced_settings import style_card
from context_menu import SEPARATOR, ContextMenu

LONG_MOVE_MM = 10.0         # a minimap move this far asks first: a misclick must not cross the stage
MARKER_PICK_PX = 10


def data_root(app) -> str:
    return run_records.data_root(app.config.get('Experiment', 'dataroot', fallback=''))


def move_blocked_reason(app) -> str:
    """Why the stage must not be moved by hand now, or ''."""
    if app.stage is None or app.stage.connection is None:
        return 'the stage is not connected'
    if getattr(app, '_plate_run_active', False):
        return 'a plate run is going'
    runtime = app.root.ids.middlecolumn.runtimecontrols
    if runtime.imageacquisitionmanager.recordbutton.state == 'down':
        return 'a recording is running'
    if runtime.isTracking:
        return 'tracking is running'
    return ''


def stage_xy(app) -> tuple | None:
    position = app.stage.get_cached_position(unit='mm', max_age=1.0) if app.stage is not None else None
    return (position[0], position[1]) if position else None


def spot_to_stage_mm(app, spot) -> tuple:
    """How far the stage must move, in mm, to bring this image spot to the centre of the view."""
    image = app.image
    dual = app.config.getboolean('DualColor', 'dualcolormode')
    offset = image_scale.offset_from_center(spot, image.shape[1], image.shape[0], dual,
                                            app.config.get('DualColor', 'mainside'))
    ystep, xstep = macro.getStageDistances(np.array([offset[1], offset[0]]), app.imageToStageMat)
    to_mm = 0.001 if app.config.get('Calibration', 'step_units') == 'um' else 1.0
    return float(xstep) * to_mm, float(ystep) * to_mm


def go_to_xy(app, x: float, y: float) -> None:
    """Move there the way the Go button does, with the same safety checks. Z stays."""
    reason = move_blocked_reason(app)
    if reason:
        app.log('stage', f'not moved: {reason}', 'warn')
        return
    app.root.ids.leftcolumn.ids.gotocontrols.move_xy(x, y)


def go_to_xy_checked(app, x: float, y: float) -> None:
    """A move picked on the map: a long one asks first."""
    here = stage_xy(app)
    if here is None:
        app.log('stage', 'position unavailable', 'warn')
    elif math.hypot(x - here[0], y - here[1]) < LONG_MOVE_MM:
        go_to_xy(app, x, y)
    else:
        distance = math.hypot(x - here[0], y - here[1])
        _ask(app, 'Long stage move', f'Move {distance:.1f} mm to X {x:.2f}, Y {y:.2f} mm?', 'Move',
             lambda: go_to_xy(app, x, y))


def _popup(app, title: str, content: BoxLayout, height: float) -> Popup:
    popup = Popup(title=title, content=content, size_hint=(None, None), size=(dp(460), dp(height)), auto_dismiss=True)
    style_card(popup, title_size=sp(18), padding=dp(20))
    app.unbind_keys()
    popup.bind(on_dismiss=lambda *a: app.bind_keys())
    popup.open()
    return popup


def _buttons(popup_box: BoxLayout, ok_text: str) -> tuple:
    row = BoxLayout(size_hint_y=None, height=dp(40), spacing=dp(8))
    cancel, ok = Factory.FlatButton(text='Cancel'), Factory.FlatButton(text=ok_text)
    for widget in (Widget(), cancel, ok):
        row.add_widget(widget)
    popup_box.add_widget(row)
    return cancel, ok


def _ask(app, title: str, text: str, ok_text: str, action) -> None:
    box = BoxLayout(orientation='vertical', spacing=dp(12))
    label = Label(text=text, font_size='14sp', halign='left', valign='middle')
    label.bind(size=lambda w, *a: setattr(w, 'text_size', w.size))
    box.add_widget(label)
    cancel, ok = _buttons(box, ok_text)
    popup = _popup(app, title, box, 190)
    cancel.bind(on_release=lambda *a: popup.dismiss())
    ok.bind(on_release=lambda *a: (popup.dismiss(), action()))


# --- markers --------------------------------------------------------------------------------

def show_markers(app) -> None:
    app.root.ids.middlecolumn.ids.stage_minimap.markers = stage_markers.load(data_root(app))


def mark(app, x: float, y: float) -> None:
    """Ask for a name and keep this position."""
    root = data_root(app)
    box = BoxLayout(orientation='vertical', spacing=dp(12))
    name = Factory.DarkInput(text=stage_markers.next_name(stage_markers.load(root)), multiline=False,
                             size_hint_y=None, height=dp(40))
    box.add_widget(name)
    box.add_widget(Label(text=f'X {x:.2f}, Y {y:.2f} mm', font_size='13sp', color=(0.7, 0.7, 0.74, 1),
                         size_hint_y=None, height=dp(20)))
    cancel, ok = _buttons(box, 'Mark')
    popup = _popup(app, 'Mark this position', box, 230)

    def keep(*args):
        text = name.text.strip()
        if not text:
            return
        popup.dismiss()
        try:
            stage_markers.add(root, text, x, y)
        except OSError as e:
            app.log('stage', f'could not save the marker: {e}', 'error')
            return
        show_markers(app)
        app.log('stage', f'marked {text}: X {x:.2f}, Y {y:.2f} mm')

    ok.bind(on_release=keep)
    name.bind(on_text_validate=keep)
    cancel.bind(on_release=lambda *a: popup.dismiss())
    Clock.schedule_once(lambda dt: (setattr(name, 'focus', True), name.select_all()), 0.1)


def delete_marker(app, marker: dict) -> None:
    stage_markers.remove(data_root(app), marker)
    show_markers(app)
    app.log('stage', f'marker {marker["name"]} deleted')


# --- the menus ------------------------------------------------------------------------------

def open_image_menu(app, preview, pos: tuple) -> None:
    runtime = app.root.ids.middlecolumn.runtimecontrols
    spot = preview.spotUnderMouse()
    blocked = move_blocked_reason(app)
    here = stage_xy(app)
    onImage = spot is not None
    grabbing = app.camera is not None and app.camera.IsGrabbing()

    def move_here():
        dx, dy = spot_to_stage_mm(app, spot)
        go_to_xy(app, here[0] + dx, here[1] + dy)

    def track():
        runtime.isShowTrackingDialogueFirstTime = False     # "click on an animal": this click did
        runtime.trackingcheckbox.state = 'down'
        preview.startTrackingAt(spot)

    def mark_here():
        dx, dy = spot_to_stage_mm(app, spot)
        mark(app, here[0] + dx, here[1] + dy)

    items = [
        ('Move here', move_here, onImage and not blocked and here is not None),
        ('Track this animal', track, onImage and not blocked and grabbing),
        SEPARATOR,
        ('Measure (click two points)', preview.startMeasuring, onImage),
    ]
    if preview.measure is not None:
        items.append(('Clear measurement', preview.clearMeasure))
    items += [
        ('Save snapshot', runtime.imageacquisitionmanager.snap, app.camera is not None),
        SEPARATOR,
        ('Mark this position', mark_here, onImage and here is not None),
    ]
    ContextMenu.show(pos, items)


def open_minimap_menu(app, minimap, pos: tuple) -> None:
    x, y = minimap.px_to_mm(*pos)
    onStage = minimap.reaches(x, y)
    blocked = move_blocked_reason(app)
    marker = stage_markers.nearest(minimap.markers, x, y, MARKER_PICK_PX / max(minimap.scale_px_per_mm(), 1e-6))
    items = []
    if marker is not None:
        items += [
            (f'Go to {marker["name"]}', lambda: go_to_xy_checked(app, marker['x'], marker['y']), not blocked),
            (f'Delete {marker["name"]}', lambda: delete_marker(app, marker)),
            SEPARATOR,
        ]
    items += [
        ('Go here', lambda: go_to_xy_checked(app, x, y), onStage and not blocked),
        ('Mark this position', lambda: mark(app, x, y), onStage),
        SEPARATOR,
        ('Show this area', lambda: minimap.show_area(x, y), onStage),
    ]
    if minimap.pinned:
        items.append(('Follow the stage again', minimap.follow_stage))
    items.append(('Clear trail', minimap.clear_trail))
    ContextMenu.show(pos, items)
