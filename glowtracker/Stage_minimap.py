"""Live stage position for the main window, independent of plate scanning."""

from math import isfinite

from kivy.app import App
from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Ellipse, Line, Rectangle
from kivy.metrics import dp
from kivy.properties import ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.widget import Widget


class StageMinimapOverlay(BoxLayout):
    """Keep clicks and scrolling on the minimap out of the live image."""

    def on_touch_down(self, touch):
        return self.collide_point(*touch.pos) or super().on_touch_down(touch)

    def on_touch_move(self, touch):
        return self.collide_point(*touch.pos) or super().on_touch_move(touch)

    def on_touch_up(self, touch):
        return self.collide_point(*touch.pos) or super().on_touch_up(touch)


class StageMinimap(Widget):
    # Physical travel of the X-LSM150A axes; this does not change motion limits.
    travel_mm = ListProperty([152.4, 152.4])
    position_mm = ListProperty([])
    status_text = StringProperty('Stage disconnected')

    def __init__(self, **kwargs):
        self._update_event = None
        super().__init__(**kwargs)
        self._redraw_trigger = Clock.create_trigger(self.redraw)
        self.bind(pos=self._redraw_trigger, size=self._redraw_trigger,
                  travel_mm=self._redraw_trigger,
                  position_mm=self._redraw_trigger)
        self._redraw_trigger()

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

    def _fit(self):
        width = max(0, self.width - dp(70))
        height = max(0, self.height - dp(34))
        scale = min(width / self.travel_mm[1], height / self.travel_mm[0])
        ox = self.center_x - self.travel_mm[1] * scale / 2
        oy = self.y + dp(10) + (height - self.travel_mm[0] * scale) / 2
        return scale, ox, oy

    def mm_to_px(self, x, y):
        """Rotate clockwise: +Y points right and +X points down."""
        scale, ox, oy = self._fit()
        return ox + y * scale, oy + (self.travel_mm[0] - x) * scale

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
        width = self.travel_mm[1] * scale
        height = self.travel_mm[0] * scale
        with self.canvas:
            Color(41 / 255, 35 / 255, 38 / 255, 1)
            Rectangle(pos=(ox, oy), size=(width, height))
            Color(64 / 255, 55 / 255, 58 / 255, 1)
            for fraction in (0.25, 0.5, 0.75):
                Line(points=[ox + width * fraction, oy,
                             ox + width * fraction, oy + height])
                Line(points=[ox, oy + height * fraction,
                             ox + width, oy + height * fraction])
            Color(142 / 255, 0, 69 / 255, 1)
            Line(rectangle=(ox, oy, width, height), width=1.2)
            Color(0.90, 0.86, 0.88, 1)
            self._label(f'{self.travel_mm[1]:g}', ox + width, oy + height + dp(12))
            self._label('Y', ox + width / 2, oy + height + dp(12))
            self._label(f'{self.travel_mm[0]:g}', ox - dp(20), oy)
            self._label('X', ox - dp(12), oy + height / 2)
            if self.position_mm:
                px, py = self.mm_to_px(*self.position_mm)
                radius = dp(4)
                Color(185 / 255, 76 / 255, 137 / 255, 1)
                Ellipse(pos=(px - radius, py - radius),
                        size=(radius * 2, radius * 2))
                Color(0.96, 0.90, 0.93, 1)
                Line(circle=(px, py, radius), width=1)
