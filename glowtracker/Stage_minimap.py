"""Live stage position for the main window, independent of plate scanning."""

from math import ceil, floor, isfinite, log10

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.metrics import dp
from kivy.properties import BoundedNumericProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget


class StageMinimapOverlay(BoxLayout):
    """Keep clicks and scrolling on the minimap out of the live image."""

    def on_touch_down(self, touch):
        # Let the map handle zoom/reset, then stop propagation to the preview.
        handled = super().on_touch_down(touch)
        return self.collide_point(*touch.pos) or handled

    def on_touch_move(self, touch):
        return self.collide_point(*touch.pos) or super().on_touch_move(touch)

    def on_touch_up(self, touch):
        return self.collide_point(*touch.pos) or super().on_touch_up(touch)


class StageMinimap(Widget):
    # Physical travel of the X-LSM150A axes; this does not change motion limits.
    travel_mm = ListProperty([152.4, 152.4])
    position_mm = ListProperty([])
    status_text = StringProperty('Stage disconnected')
    zoom_level = BoundedNumericProperty(0, min=0, max=4)

    def __init__(self, **kwargs):
        self._update_event = None
        self._last_position_mm = None
        super().__init__(**kwargs)
        self._redraw_trigger = Clock.create_trigger(self.redraw)
        self.bind(pos=self._redraw_trigger, size=self._redraw_trigger,
                  travel_mm=self._redraw_trigger,
                  position_mm=self._redraw_trigger,
                  zoom_level=self._redraw_trigger)
        self._redraw_trigger()

    def on_position_mm(self, instance, position):
        if position:
            self._last_position_mm = list(position)

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        if touch.is_mouse_scrolling:
            if touch.button == 'scrollup':
                self.zoom_level = min(4, self.zoom_level + 1)
            elif touch.button == 'scrolldown':
                self.zoom_level = max(0, self.zoom_level - 1)
        elif touch.is_double_tap:
            self.zoom_level = 0
        return True

    def on_parent(self, instance, parent):
        if self._update_event is not None:
            self._update_event.cancel()
            self._update_event = None
        if parent is not None:
            self._update_event = Clock.schedule_interval(self.update_position, 0.2)
            self.update_position()

    def update_position(self, dt=0):
        app = App.get_running_app()
        stage = getattr(app, 'stage', None)
        if stage is None or stage.connection is None:
            self.position_mm = []
            self.status_text = 'Stage disconnected'
            return

        # Read the background poller's cache only; never query serial hardware
        # from the UI thread. Hide stale positions instead of showing a live dot.
        position = stage.get_cached_position(unit='mm', max_age=1.0)
        if position is None or len(position) < 2 or not all(
                isfinite(value) for value in position[:2]):
            self.position_mm = []
            self.status_text = 'Position unavailable'
            return

        x, y = position[:2]
        self.status_text = f'X {x:.2f} mm   Y {y:.2f} mm'
        if not (0 <= x <= self.travel_mm[0] and 0 <= y <= self.travel_mm[1]):
            self.position_mm = []
            self.status_text = 'Position outside map'
            return
        self.position_mm = [x, y]

    def view_bounds_mm(self):
        """Return xmin, ymin, xmax, ymax; zoomed views follow the stage."""
        if self.zoom_level == 0:
            return 0, 0, self.travel_mm[0], self.travel_mm[1]
        center = self.position_mm or self._last_position_mm
        if center is None:
            center = [value / 2 for value in self.travel_mm]
        half_x, half_y = (value / (2 * 2 ** self.zoom_level)
                          for value in self.travel_mm)
        # Allow the view to extend beyond travel limits so the dot stays centered.
        # Those unreachable areas are shaded separately in redraw().
        return (center[0] - half_x, center[1] - half_y,
                center[0] + half_x, center[1] + half_y)

    def _fit(self):
        xmin, ymin, xmax, ymax = self.view_bounds_mm()
        span_x, span_y = xmax - xmin, ymax - ymin
        width = max(0, self.width - dp(70))
        height = max(0, self.height - dp(52))
        scale = min(width / span_y, height / span_x)
        ox = self.center_x - span_y * scale / 2
        oy = self.y + dp(28) + (height - span_x * scale) / 2
        return scale, ox, oy

    def mm_to_px(self, x, y):
        """Rotate clockwise: +Y points right and +X points down."""
        scale, ox, oy = self._fit()
        xmin, ymin, xmax, ymax = self.view_bounds_mm()
        return ox + (y - ymin) * scale, oy + (xmax - x) * scale

    @staticmethod
    def _nice_step(target):
        """A readable 1/2/5 interval no larger than the requested distance."""
        magnitude = 10 ** floor(log10(target))
        return max(value * magnitude for value in (1, 2, 5)
                   if value * magnitude <= target)

    @staticmethod
    def _label(text, cx, cy):
        label = CoreLabel(text=text, font_size=dp(11))
        label.refresh()
        w, h = label.texture.size
        Rectangle(texture=label.texture, pos=(cx - w / 2, cy - h / 2), size=(w, h))

    def redraw(self, *args):
        self.canvas.clear()
        scale, ox, oy = self._fit()
        if scale <= 0:
            return
        xmin, ymin, xmax, ymax = self.view_bounds_mm()
        width = (ymax - ymin) * scale
        height = (xmax - xmin) * scale
        step = self._nice_step(min(xmax - xmin, ymax - ymin) / 4)
        valid_xmin, valid_ymin = max(0, xmin), max(0, ymin)
        valid_xmax = min(self.travel_mm[0], xmax)
        valid_ymax = min(self.travel_mm[1], ymax)
        with self.canvas:
            Color(0.06, 0.05, 0.055, 1)
            Rectangle(pos=(ox, oy), size=(width, height))
            if valid_xmin < valid_xmax and valid_ymin < valid_ymax:
                vx, vy = self.mm_to_px(valid_xmax, valid_ymin)
                vw = (valid_ymax - valid_ymin) * scale
                vh = (valid_xmax - valid_xmin) * scale
                Color(41 / 255, 35 / 255, 38 / 255, 1)
                Rectangle(pos=(vx, vy), size=(vw, vh))
                Color(64 / 255, 55 / 255, 58 / 255, 1)
                # Anchor grid lines to stage coordinates so they move while following.
                for index in range(ceil(valid_xmin / step), floor(valid_xmax / step) + 1):
                    _, py = self.mm_to_px(index * step, valid_ymin)
                    Line(points=[vx, py, vx + vw, py])
                for index in range(ceil(valid_ymin / step), floor(valid_ymax / step) + 1):
                    px, _ = self.mm_to_px(valid_xmax, index * step)
                    Line(points=[px, vy, px, vy + vh])
                Color(142 / 255, 0, 69 / 255, 1)
                Line(rectangle=(vx, vy, vw, vh), width=1.2)
            Color(142 / 255, 0, 69 / 255, 1)
            Line(rectangle=(ox, oy, width, height), width=1.2)
            Color(0.90, 0.86, 0.88, 1)
            self._label(f'{ymax:.1f}', ox + width, oy + height + dp(12))
            self._label('Y', ox + width / 2, oy + height + dp(12))
            self._label(f'{xmax:.1f}', ox - dp(22), oy)
            self._label('X', ox - dp(12), oy + height / 2)
            if self.zoom_level:
                self._label(f'{ymin:.1f}', ox, oy + height + dp(12))
                self._label(f'{xmin:.1f}', ox - dp(22), oy + height - dp(4))

            # One physical grid interval, with the distance shown in millimetres.
            bar_width = step * scale
            bar_x, bar_y = ox + width / 2 - bar_width, oy - dp(14)
            Line(points=[bar_x, bar_y, bar_x + bar_width, bar_y], width=1)
            for px in (bar_x, bar_x + bar_width):
                Line(points=[px, bar_y - dp(3), px, bar_y + dp(3)], width=1)
            self._label(f'{step:g} mm', ox + width / 2 + dp(22), bar_y)
            if self.position_mm:
                px, py = self.mm_to_px(*self.position_mm)
                radius = dp(4)
                Color(185 / 255, 76 / 255, 137 / 255, 1)
                Ellipse(pos=(px - radius, py - radius),
                        size=(radius * 2, radius * 2))
                Color(0.96, 0.90, 0.93, 1)
                Line(circle=(px, py, radius), width=1)
