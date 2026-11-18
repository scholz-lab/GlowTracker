from kivy.uix.widget import Widget
from kivy.properties import ListProperty, BooleanProperty
from kivy.graphics import Color, Line
from kivy.app import App

class ScanMinimap(Widget):
    stage_min = ListProperty([0, 72])
    stage_max = ListProperty([136, 145])

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(pos=self.redraw, size=self.redraw,
                  stage_min=self.redraw, stage_max=self.redraw)

    def on_kv_post(self, *args):
        app = App.get_running_app()
        if app is not None:
            app.bind(plateCenter=self.redraw, plateRadius=self.redraw)
        self.redraw()

    def _fit(self):
        sw = self.stage_max[0] - self.stage_min[0]
        sh = self.stage_max[1] - self.stage_min[1]
        bw, bh = sh, sw
        scale = min(self.width / bw, self.height / bh) * 0.9
        ox = self.center_x - bw * scale / 2
        oy = self.center_y - bh * scale / 2
        return scale, ox, oy

    def mm_to_px(self, x, y):
        scale, ox, oy = self._fit()
        dx = x - self.stage_min[0]
        dy = y - self.stage_min[1]
        sw = self.stage_max[0] - self.stage_min[0]
        px = ox + dy * scale
        py = oy + (sw - dx) * scale
        return px, py

    def redraw(self, *a):
        self.canvas.clear()
        app = App.get_running_app()
        scale, _, _ = self._fit()
        with self.canvas:
            Color(0.4, 0.7, 1, 1)
            x0, y0 = self.mm_to_px(*self.stage_min)
            x1, y1 = self.mm_to_px(*self.stage_max)
            Line(rectangle=(min(x0, x1), min(y0, y1),
                            abs(x1 - x0), abs(y1 - y0)), width=1.2)

            if app is not None and app.plateCenter is not None and app.plateRadius is not None:
                Color(1, 0.6, 0.2, 1)
                cx, cy = self.mm_to_px(app.plateCenter[0], app.plateCenter[1])
                Line(circle=(cx, cy, app.plateRadius * scale), width=1.2)
