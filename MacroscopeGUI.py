import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty, StringProperty
from kivy.clock import Clock
import os
import Zaber_control as stg
import Macroscope_macros as macro
import Basler_control as cam

import globals
# TODO: separate stage from camera in the programs
# TODO: Structure the GUI such that it shares variables across GUI components

    
# main GUI
class MainWindow(GridLayout):
    expPath = StringProperty()
    def __init__(self,  **kwargs):
        super(MainWindow, self).__init__(**kwargs)
        # experimental path
        self.expPath = globals.SAVEPATH
        Clock.schedule_once(self._do_setup)
    # Note: Work around: IDs are not available at init, 
    # so we schedule an update for the path at one frame later (when app is already running)
    def _do_setup(self, *l):
        # display save location
        self.ids.leftcolumn.ids.saveloc.text = self.expPath


class LeftColumn(BoxLayout):
    # file saving
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    cameraprops = ObjectProperty(None)

    #
    def __init__(self,  **kwargs):
        super(LeftColumn, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)
    
    def _do_setup(self, *l):
        #pass
        self.savefile = str(self.ids.saveloc.text)
        self.loadfile = str(self.ids.saveloc.text)
    
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_load(self):
        content = LoadCameraProperties(load=self.load, cancel=self.dismiss_popup)
        content.ids.filechooser2.path = self.loadfile
        self._popup = Popup(title="Load camera file", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def show_save(self):
        content = SaveExperiment(save=self.save, cancel=self.dismiss_popup)
        content.ids.filechooser.path = self.savefile
        self._popup = Popup(title="Select save location", content=content,
                            size_hint=(0.9, 0.9))
        self._popup.open()

    def load(self, path, filename):
        self.loadfile = os.path.join(path, filename[0])
        #with open(os.path.join(path, filename[0])) as stream:
            ### TODO edit to load the psf file and alter camera settings
            #self.cameraprops.text = stream.read()
        self.dismiss_popup()
    
    def save(self, path, filename):
        self.savefile = os.path.join(path, filename)
        self.ids.saveloc.text = (self.savefile)
        self.dismiss_popup()
    

class MiddleColumn(BoxLayout):
    pass

class RightColumn(BoxLayout):
    pass
# Stage controls
class XControls(Widget):
    pass

class YControls(Widget):
    pass 
    
class ZControls(Widget):
    pass

# load camera settings
class LoadCameraProperties(BoxLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    

# save location for images and meta data
class SaveExperiment(GridLayout):
    save = ObjectProperty(None)
    #savepath = ObjectProperty(None)
    #filename = StringProperty(None)
    cancel = ObjectProperty(None)


# camera properties
class CameraProperties(GridLayout):
    pass

#record and live view buttons
class RecordButtons(BoxLayout):
    pass
    
    
# image preview
class PreviewImage(Image):
    pass
    
class RuntimeControls(BoxLayout):
    pass
    

# display if hardware is connected
class Connections(BoxLayout):
    pass

class ExitApp(BoxLayout):
   stop = ObjectProperty(None)
   cancel = ObjectProperty(None)
# set window size at startup
Window.size = (1280, 800)
# load the layout
class MacroscopeApp(App):
    def build(self):
        layout = Builder.load_file('layout.kv')
        # connect x-close button to action
        Window.bind(on_request_close=self.on_request_close)
        return layout
    # populate default
    def on_start(self):
        pass
    #    # TODO: load default paths

        #App.get_running_app().root.LeftColumn.saveloc.text = 'TEST'
    # ask for confirmation of closing
    def on_request_close(self, *args):
        content = ExitApp(stop=self.graceful_exit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Exit Macroscope GUI", content=content,
                            size_hint=(0.5, 0.2))
        self._popup.open()
        return True
        
    def dismiss_popup(self):
        self._popup.dismiss()
    
    def graceful_exit(self):
        #TODO add connection closing etc. here to make a nice exit
        self.stop()
            

            
       

def reset():
    # Cleaner for the events in memory
    import kivy.core.window as window
    from kivy.base import EventLoop
    if not EventLoop.event_listeners:
        from kivy.cache import Cache
        window.Window = window.core_select_lib('window', window.window_impl, True)
        Cache.print_usage()
        for cat in Cache._categories:
            Cache._objects[cat] = {}
     
        
if __name__ == '__main__':
    reset()
    # TODO: connection to the decive and error handling is done here
    App = MacroscopeApp()
    App.run()  # This runs the App in an endless loop until it closes. At this point it will execute the code below
    # TODO: disconnect here from the active devices on closure of the GUI
