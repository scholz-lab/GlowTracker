import kivy
kivy.require('2.0.0')
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter
#from kivy.uix.scatterplane import ScatterPlane
from kivy.core.window import Window
from kivy.graphics.transformation import Matrix
from kivy.lang import Builder

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scrollview import ScrollView


kv = '''
#:kivy 1.11.1
<MyScatter>:
    do_translation_y: False
    do_rotation: False
    do_scale: True
    canvas:
        Color:
            hsv: .1, 1, .5
        Rectangle:
            size: 100, 100

<MyScrollView>:
    size_hint: None, None
    size: 640, 480
    pos_hint: {'center_x': .5, 'center_y': .5}
    scroll_type: ['bars']
    bar_width: 10
    bar_inactive_color: self.bar_color
    canvas:
        Color:
            rgb: 1, 0, 0
        Rectangle:
            pos: self.pos
            size: self.size

<MyScatterLayout>:
    size_hint: None, None
    size: 1280 * self.scale, 720 * self.scale  # this is the correction
    do_translation: False
    do_rotation: False
    pos_hint: {'center_x': 0.5, 'center_y': 0.5}
    canvas:
        Color:
            rgb: 0, 0, 1
        Rectangle:
            pos: 0, 0
            size: self.size
'''

Builder.load_string(kv)


class MyScatter(Scatter):
    pass


class MyScatterLayout(ScatterLayout):
    pass


class MyScrollView(ScrollView):
    pass


class MyApp(App):
    def build(self):
        layout = MyScatterLayout()
        layout.add_widget(MyScatter())
        root = MyScrollView()
        root.add_widget(layout)
        return root


if __name__ == '__main__':
    MyApp().run()
