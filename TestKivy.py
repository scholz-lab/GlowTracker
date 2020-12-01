from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.properties import StringProperty


class ChooseProgScreen(Screen):
    pass


class SwitchScreen(Screen):
    pass


class ScreenManagement(ScreenManager):
    MY_GLOBAL = StringProperty('test')


class MainApp(App):

    def build(self):
        return Builder.load_file("kivy.kv")


if __name__ == "__main__":
    MainApp().run()