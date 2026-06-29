from kivy.uix.widget import Widget
from kivy.properties import ListProperty
from kivy.graphics import Color, Line

class ScanMinimap(Widget):
    stage_min = ListProperty([0, 72])     
    stage_max = ListProperty([136, 145])  

    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(pos=self.redraw, size=self.redraw,
                  stage_min=self.redraw, stage_max=self.redraw)

    def _fit(self):
        sw = self.stage_max[0] - self.stage_min[0]   
        sh = self.stage_max[1] - self.stage_min[1]   
        scale = min(self.width / sw, self.height / sh) * 0.9 
        ox = self.center_x - sw * scale / 2          
        oy = self.center_y - sh * scale / 2
        return scale, ox, oy

    def mm_to_px(self, x, y):
        scale, ox, oy = self._fit()
        px = ox + (x - self.stage_min[0]) * scale
        py = oy + (y - self.stage_min[1]) * scale    
        return px, py

    def redraw(self, *a):
        self.canvas.clear()
        with self.canvas:
            Color(0.4, 0.7, 1, 1)
            x0, y0 = self.mm_to_px(*self.stage_min)
            x1, y1 = self.mm_to_px(*self.stage_max)
            Line(rectangle=(x0, y0, x1 - x0, y1 - y0), width=1.2)
