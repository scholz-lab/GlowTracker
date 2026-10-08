"""Drawn views for the plate scan window.

PlateMap: the stage seen from above as a plot in mm, every plate a disc with a status-coloured
rim (the selected one highlighted), the scan tiles of the plate being searched filling in as the
scan moves, the rim points, a crosshair at the stage position, and a legend of the rim colours.
CameraView: the live image with a calibrated scale bar and the field of view.

DishPreview: the plate editor's view of the captured rim points and the circle fitted to them.

Both share the main window's minimap orientation: stage X runs down the view, stage Y across.
"""
from __future__ import annotations

import math

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.properties import ListProperty, NumericProperty, ObjectProperty
from kivy.uix.image import Image
from kivy.uix.widget import Widget

import Microscope_macros as macro

TEXT = (0.92, 0.92, 0.94, 1)
MUTED = (0.6, 0.6, 0.65, 1)
ACCENT = (0.87, 0.38, 0.58, 1)
BLUE = (0.04, 0.52, 1, 1)
GREEN = (0.19, 0.82, 0.35, 1)
ORANGE = (1, 0.62, 0.2, 1)
RED = (1, 0.36, 0.36, 1)
IDLE = (0.42, 0.43, 0.48, 1)


def status_colour(status: str, enabled: bool = True):
    """A plate's status as a colour: searching blue, tracking green, problems orange or red."""
    text = (status or '').lower()
    if not enabled:
        return (0.3, 0.31, 0.34, 1)
    if any(word in text for word in ('fail', 'error', 'stopped')):
        return RED
    if any(word in text for word in ('no worm', 'incomplete')):
        return ORANGE
    if any(word in text for word in ('search', 'focus')):
        return BLUE
    if any(word in text for word in ('track', 'record')):
        return GREEN
    return IDLE


_TEXTURES: dict = {}


def _text(text: str, size: float, colour=TEXT, bold: bool = False, italic: bool = False):
    """A text texture, cached: the map redraws often with the same labels."""
    key = (text, size, tuple(colour), bold, italic)
    if key not in _TEXTURES:
        if len(_TEXTURES) > 500:
            _TEXTURES.clear()
        label = CoreLabel(text=text, font_size=size, bold=bold, italic=italic, color=colour)
        label.refresh()
        _TEXTURES[key] = label.texture
    return _TEXTURES[key]


class PlateMap(Widget):
    """The stage from above. Set `panel` (the scan panel); it redraws when the plates, the run
    or the stage position change."""
    panel = ObjectProperty(None, allownone=True)
    stage_size = ListProperty([152.4, 152.4])

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trigger = Clock.create_trigger(self.redraw, 0)
        self.bind(pos=self._trigger, size=self._trigger, panel=self._watch, stage_size=self._trigger)

    def _watch(self, *args):
        panel = self.panel
        if panel is None:
            return
        panel.bind(plates=self._trigger, selected_plate=self._trigger, running=self._trigger,
                   scan_progress=self._trigger, points=self._trigger)
        app = App.get_running_app()
        if app is not None:
            app.bind(coords=self._trigger, plateCenter=self._trigger, plateRadius=self._trigger)
            try:
                limits = [float(v) for v in app.config.get('Stage', 'stage_limits').split(',')]
                self.stage_size = limits[:2]
            except Exception:
                pass
        self._trigger()

    # stage mm -> widget px (X down, Y across, as in the main window's minimap); room left and
    # below for the axes
    PAD = (46, 26, 14, 36)      # left, top, right, bottom (dp)

    def _fit(self):
        sx, sy = self.stage_size
        left, top, right, bottom = (dp(v) for v in self.PAD)
        if self.width > self.height * 1.1:      # wide enough: keep a column for the legend
            right = dp(130)
        width, height = self.width - left - right, self.height - top - bottom
        scale = max(1e-6, min(width / sy, height / sx))
        ox = self.x + left + (width - sy * scale) / 2
        oy = self.y + bottom + (height - sx * scale) / 2
        return scale, ox, oy

    def to_px(self, x: float, y: float):
        scale, ox, oy = self._fit()
        return ox + y * scale, oy + (self.stage_size[0] - x) * scale

    def redraw(self, *args):
        self.canvas.clear()
        panel, app = self.panel, App.get_running_app()
        if panel is None or app is None or self.width < 10:
            return
        scale, ox, oy = self._fit()
        with self.canvas:
            self._axes(scale, ox, oy)
            active = self._activeIndex(app)
            for index, plate in enumerate(panel.plates):
                self._dish(plate, index == panel.selected_plate, index == active, scale)

            # the plate being defined, not stored yet: its fitted circle (dashes need width 1)
            if panel.selected_plate < 0 and app.plateCenter is not None and app.plateRadius:
                cx, cy = self.to_px(*app.plateCenter)
                Color(*ACCENT[:3], 0.9)
                Line(circle=(cx, cy, app.plateRadius * scale), width=1, dash_length=6, dash_offset=4)
            self._tiles(app, panel, scale)
            Color(1, 1, 1, 0.95)
            for point in panel.points:
                px, py = self.to_px(*point[:2])
                Ellipse(pos=(px - dp(2.5), py - dp(2.5)), size=(dp(5), dp(5)))

            self._crosshair(app, scale)
            self._legend(scale, ox, oy)
            if not panel.plates and app.plateCenter is None:
                texture = _text('Add a plate to see it on the stage', sp(13), MUTED)
                Color(1, 1, 1, 1)
                Rectangle(texture=texture, size=texture.size,
                          pos=(self.center_x - texture.width / 2, self.center_y - texture.height / 2))

    def _axes(self, scale, ox, oy):
        """The stage travel area as a plot frame: x down the left side, y along the bottom, in mm,
        with inward major (50 mm, labelled) and minor (10 mm) ticks and a faint major grid."""
        sx, sy = self.stage_size
        w, h = sy * scale, sx * scale
        Color(0.095, 0.1, 0.115, 1)
        Rectangle(pos=(ox, oy), size=(w, h))
        Color(1, 1, 1, 0.045)
        for mm in range(50, int(sy), 50):
            Line(points=[ox + mm * scale, oy, ox + mm * scale, oy + h], width=1)
        for mm in range(50, int(sx), 50):
            Line(points=[ox, oy + h - mm * scale, ox + w, oy + h - mm * scale], width=1)
        Color(1, 1, 1, 0.35)
        Line(rectangle=(ox, oy, w, h), width=1)
        for mm in range(0, int(sy) + 1, 10):            # y: bottom and top edges
            px, length = ox + mm * scale, dp(6) if mm % 50 == 0 else dp(3)
            Color(1, 1, 1, 0.35)
            Line(points=[px, oy, px, oy + length], width=1)
            Line(points=[px, oy + h, px, oy + h - length], width=1)
            if mm % 50 == 0:
                texture = _text(str(mm), sp(11), MUTED)
                Color(1, 1, 1, 1)
                Rectangle(texture=texture, size=texture.size, pos=(px - texture.width / 2, oy - texture.height - dp(3)))
        for mm in range(0, int(sx) + 1, 10):            # x: left and right edges, 0 at the top
            py, length = oy + h - mm * scale, dp(6) if mm % 50 == 0 else dp(3)
            Color(1, 1, 1, 0.35)
            Line(points=[ox, py, ox + length, py], width=1)
            Line(points=[ox + w, py, ox + w - length, py], width=1)
            if mm % 50 == 0:
                texture = _text(str(mm), sp(11), MUTED)
                Color(1, 1, 1, 1)
                Rectangle(texture=texture, size=texture.size, pos=(ox - texture.width - dp(5), py - texture.height / 2))
        for text, pos in (('y (mm)', lambda t: (ox + w - t.width, oy - t.height * 2 - dp(4))),
                          ('x (mm)', lambda t: (ox + dp(2), oy + h + dp(6)))):
            texture = _text(text, sp(11.5), TEXT, italic=True)
            Color(1, 1, 1, 1)
            Rectangle(texture=texture, size=texture.size, pos=pos(texture))

    def _activeIndex(self, app):
        if not self.panel.running or app.plateCenter is None:
            return -1
        for index, plate in enumerate(self.panel.plates):
            if all(abs(a - b) < 1e-6 for a, b in zip(plate['center'], app.plateCenter)):
                return index
        return -1

    def _dish(self, plate, selected: bool, active: bool, scale: float):
        """A plate: a flat disc, its rim in the status colour, a centre mark and its name."""
        cx, cy = self.to_px(*plate['center'])
        r = plate['radius'] * scale
        enabled = plate.get('enabled', True)
        rim = status_colour(plate.get('status', ''), enabled)
        Color(0.16, 0.17, 0.2, 1 if enabled else 0.5)
        Ellipse(pos=(cx - r, cy - r), size=(2 * r, 2 * r))
        if active:
            Color(*rim[:3], 0.35)
            Line(circle=(cx, cy, r + dp(4)), width=dp(1.5))
        Color(*(ACCENT if selected else rim))
        Line(circle=(cx, cy, r), width=dp(1.6) if selected else dp(1.1))
        Color(*MUTED[:3], 0.8)
        Line(points=[cx - dp(4), cy, cx + dp(4), cy], width=1)
        Line(points=[cx, cy - dp(4), cx, cy + dp(4)], width=1)
        if scale >= dp(2.5):        # names only when the map is large enough to read them
            texture = _text(plate['name'], sp(12), TEXT if enabled else MUTED, bold=selected)
            Color(1, 1, 1, 1)
            Rectangle(texture=texture, size=texture.size, pos=(cx - texture.width / 2, cy - r - texture.height - dp(4)))

    def _tiles(self, app, panel, scale):
        """The search tiles of the plate in the view (selected, or being searched), visited ones
        filled in order as the scan progresses."""
        if app.plateCenter is None or not app.plateRadius:
            return
        try:
            fov = app.get_fov_mm()
        except Exception:
            fov = None
        if not fov:
            return
        tiles = macro.generate_scan_tiles(app.plateCenter, app.plateRadius, *fov,
                                          overlap_w=panel.scan_overlap_w / 100.0,
                                          overlap_h=panel.scan_overlap_h / 100.0)
        if not tiles or len(tiles) > 4000:
            return
        searching = panel.running and 0 < panel.scan_progress < 1
        done = int(round(panel.scan_progress * len(tiles))) if searching else 0
        w, h = fov[1] * scale, fov[0] * scale      # X runs down the view
        for index, (x, y) in enumerate(tiles):
            px, py = self.to_px(x, y)
            if index < done:
                Color(*BLUE[:3], 0.6 if index == done - 1 else 0.25)
                Rectangle(pos=(px - w / 2, py - h / 2), size=(w, h))
            else:
                Color(1, 1, 1, 0.07)
                Line(rectangle=(px - w / 2, py - h / 2, w, h), width=1)

    LEGEND = (('Ready', IDLE), ('Searching', BLUE), ('Tracking', GREEN), ('No worm found', ORANGE),
              ('Failed', RED), ('Selected', ACCENT))

    def legend_box(self):
        """Where the legend is drawn (x0, y0, x1, y1), or None when there is no room for it."""
        scale, ox, oy = self._fit()
        x = ox + self.stage_size[1] * scale + dp(18)
        if self.right - x < dp(110):
            return None
        top = oy + self.stage_size[0] * scale + dp(4)
        return x, top - len(self.LEGEND) * dp(20), x + dp(120), top

    def _legend(self, scale, ox, oy):
        """The rim colours, as a column to the right of the plot when there is room."""
        x = ox + self.stage_size[1] * scale + dp(18)
        if self.right - x < dp(110):
            return
        y = oy + self.stage_size[0] * scale - dp(8)
        for name, colour in self.LEGEND:
            Color(*colour)
            Line(circle=(x + dp(5), y, dp(4.5)), width=dp(1.3))
            texture = _text(name, sp(11.5), MUTED)
            Color(1, 1, 1, 1)
            Rectangle(texture=texture, size=texture.size, pos=(x + dp(15), y - texture.height / 2))
            y -= dp(20)

    def _crosshair(self, app, scale):
        coords = list(app.coords or [])
        if len(coords) < 2:
            return
        px, py = self.to_px(coords[0], coords[1])
        if not (self.x <= px <= self.right and self.y <= py <= self.top):
            return
        Color(1, 1, 1, 0.9)
        Line(circle=(px, py, dp(5)), width=1)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            Line(points=[px + dx * dp(8), py + dy * dp(8), px + dx * dp(13), py + dy * dp(13)], width=1)
        if scale >= dp(2.5):        # no room for the label in a small map
            texture = _text(f'({coords[0]:.2f}, {coords[1]:.2f}) mm', sp(11), MUTED)
            Rectangle(texture=texture, size=texture.size, pos=(px + dp(10), py + dp(6)))


class CameraView(Image):
    """The live camera image with a calibrated scale bar and the field of view, or a placeholder."""

    def __init__(self, **kwargs):
        super().__init__(fit_mode='contain', **kwargs)
        self.bind(texture=self._redraw, pos=self._redraw, size=self._redraw)
        self._redraw()

    def _redraw(self, *args):
        self.color = (1, 1, 1, 1) if self.texture else (0, 0, 0, 0)
        self.canvas.after.clear()
        with self.canvas.after:
            if not self.texture:
                texture = _text('No camera image', sp(13), MUTED)
                Color(1, 1, 1, 1)
                Rectangle(texture=texture, size=texture.size,
                          pos=(self.center_x - texture.width / 2, self.center_y - texture.height / 2))
                return
            app = App.get_running_app()
            try:
                pixel = app.config.getfloat('Camera', 'pixelsize')
                pixel_um = pixel * (1000.0 if app.config.get('Calibration', 'step_units') == 'mm' else 1.0)
            except Exception:
                return
            shown_w, shown_h = self.norm_image_size
            if pixel_um <= 0 or shown_w < 20:
                return
            um_per_px = pixel_um * self.texture.width / shown_w
            target = 0.2 * shown_w * um_per_px
            magnitude = 10 ** math.floor(math.log10(target))
            length_um = max(m * magnitude for m in (1, 2, 5) if m * magnitude <= target)
            length = length_um / um_per_px
            left = self.center_x - shown_w / 2 + dp(14)
            bottom = self.center_y - shown_h / 2 + dp(14)
            label = f'{length_um / 1000:g} mm' if length_um >= 1000 else f'{length_um:g} µm'
            texture = _text(label, sp(12), TEXT, bold=True)
            Color(0, 0, 0, 0.45)
            RoundedRectangle(pos=(left - dp(8), bottom - dp(8)),
                             size=(max(length, texture.width) + dp(16), texture.height + dp(22)), radius=[dp(6)])
            Color(1, 1, 1, 0.95)
            Rectangle(pos=(left, bottom), size=(length, dp(3)))
            Rectangle(texture=texture, size=texture.size, pos=(left, bottom + dp(6)))
            fov_w, fov_h = pixel_um * self.texture.width / 1000, pixel_um * self.texture.height / 1000
            texture = _text(f'FOV {fov_w:.2f} × {fov_h:.2f} mm', sp(11.5), TEXT)
            Color(0, 0, 0, 0.45)
            right = self.center_x + shown_w / 2 - dp(14)
            RoundedRectangle(pos=(right - texture.width - dp(8), bottom - dp(8)),
                             size=(texture.width + dp(16), texture.height + dp(10)), radius=[dp(6)])
            Color(1, 1, 1, 1)
            Rectangle(texture=texture, size=texture.size, pos=(right - texture.width, bottom - dp(3)))


class DishPreview(Widget):
    """The rim points captured so far and the fitted circle, scaled to fill the view."""
    panel = ObjectProperty(None, allownone=True)
    needed = NumericProperty(3)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trigger = Clock.create_trigger(self.redraw, 0)
        self.bind(pos=self._trigger, size=self._trigger, panel=self._watch)

    def _watch(self, *args):
        if self.panel is None:
            return
        self.panel.bind(points=self._trigger)
        app = App.get_running_app()
        if app is not None:
            app.bind(plateCenter=self._trigger, plateRadius=self._trigger)
        self._trigger()

    def redraw(self, *args):
        self.canvas.clear()
        if self.panel is None:
            return
        app = App.get_running_app()
        points = [p[:2] for p in self.panel.points]
        center = app.plateCenter if app is not None else None
        radius = app.plateRadius if app is not None else None
        size = min(self.width, self.height)
        cx0, cy0 = self.center
        with self.canvas:
            Color(0.105, 0.11, 0.125, 1)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
            if center is not None and radius:
                span, mid = radius * 1.25, center
            elif points:
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                span = max(max(xs) - min(xs), max(ys) - min(ys), 10) * 0.8
                mid = ((max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2)
            else:
                span, mid = None, None
            scale = (size / 2 - dp(16)) / span if span else 1

            def px(point):
                # X down, Y across, as on the stage map
                return cx0 + (point[1] - mid[1]) * scale, cy0 - (point[0] - mid[0]) * scale

            if center is not None and radius:
                x, y = px(center)
                r = radius * scale
                Color(0.15, 0.16, 0.19, 1)
                Ellipse(pos=(x - r, y - r), size=(2 * r, 2 * r))
                Color(0.19, 0.2, 0.235, 1)
                Ellipse(pos=(x - r * 0.9, y - r * 0.9), size=(1.8 * r, 1.8 * r))
                Color(*ACCENT)
                Line(circle=(x, y, r), width=dp(1.6))
                Color(*ACCENT[:3], 0.8)
                Line(points=[x - dp(5), y, x + dp(5), y], width=dp(1))
                Line(points=[x, y - dp(5), x, y + dp(5)], width=dp(1))
            else:
                # a dashed placeholder dish
                Color(1, 1, 1, 0.12)
                Line(circle=(cx0, cy0, size / 2 - dp(22)), width=1, dash_length=6, dash_offset=6)
            for index, point in enumerate(points):
                x, y = px(point) if mid is not None else (cx0, cy0)
                Color(1, 1, 1, 1)
                Ellipse(pos=(x - dp(5), y - dp(5)), size=(dp(10), dp(10)))
                texture = _text(str(index + 1), sp(11), MUTED, bold=True)
                Rectangle(texture=texture, size=texture.size, pos=(x + dp(7), y + dp(4)))
            if not points:
                texture = _text('No rim points yet', sp(13), MUTED)
                Color(1, 1, 1, 1)
                Rectangle(texture=texture, size=texture.size,
                          pos=(cx0 - texture.width / 2, cy0 - texture.height / 2))
