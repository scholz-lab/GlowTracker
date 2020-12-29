import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.factory import Factory
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty, StringProperty, BoundedNumericProperty, NumericProperty
from kivy.clock import Clock
from threading import Thread
from functools import partial
import os
import Zaber_control as stg
import Macroscope_macros as macro
import Basler_control as basler

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
    # file saving and loading
    loadfile = ObjectProperty(None)
    savefile = ObjectProperty(None)
    cameraprops = ObjectProperty(None)
    #
    def __init__(self,  **kwargs):
        super(LeftColumn, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)
    
    def _do_setup(self, *l):
        self.savefile = str(self.ids.saveloc.text)
        self.loadfile = globals.DEFAULT_BASLER
        self.apply_cam_settings()
    
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
        self.apply_cam_settings()
        self.dismiss_popup()
    
    def save(self, path, filename):
        self.savefile = os.path.join(path, filename)
        self.ids.saveloc.text = (self.savefile)
        self.dismiss_popup()
        
    def apply_cam_settings(self):
        camera = App.get_running_app().camera
        ### TODO this reference is quite ugly, check if more elegant solution is possible
        if self.parent.ids.rightcolumn.ids.connections.ids.cam_connection.state == 'down':
            print('Updating camera settings')
            basler.update_props(camera, propfile=self.loadfile)
            self.update_settings_display()

    # when file is loaded - update slider values which updates the camera
    def update_settings_display(self):
        # update slider value using ids
        camera = App.get_running_app().camera
        self.ids.camprops.ids.exposure.value = camera.ExposureTime()
        self.ids.camprops.ids.gain.value = camera.Gain()
        self.ids.camprops.ids.framerate.value = camera.ResultingFrameRate()
    # stage motion functions


class MiddleColumn(BoxLayout):
    runtimecontrols = ObjectProperty(None)
    previewimage = ObjectProperty(None)

class RightColumn(BoxLayout):
    # store settings from popups
    nframes = ObjectProperty(None)
    fileformat = ObjectProperty(None)
    #
    def __init__(self,  **kwargs):
        super(RightColumn, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)
        self.nframes= str(globals.DEFAULT_FRAMES)

    def _do_setup(self, *l):
        self.nframes= str(globals.DEFAULT_FRAMES)
        self.fileformat = globals.IMG_FORMAT
    
    def dismiss_popup(self):
        self._popup.dismiss()

    def show_recording_settings(self):
        fps = App.get_running_app().root.ids.leftcolumn.ids.camprops.framerate.value
        content = RecordingSettings(update=self.update, cancel=self.dismiss_popup, \
                            frames = self.nframes, framerate = str(fps), fileformat = self.fileformat)
        self._popup = Popup(title="Recording Settings", content=content,
                            size_hint=(0.5, 0.25))
        self._popup.open()
    
    def update(self, frames):
        if frames is not None:
            self.nframes = frames
        self.dismiss_popup()

# Stage controls
class XControls(BoxLayout):
    def __init__(self,  **kwargs):
        super(XControls, self).__init__(**kwargs)
    
    

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
    cancel = ObjectProperty(None)

class RecordingSettings(BoxLayout):
    cancel = ObjectProperty(None)
    update = ObjectProperty(None)
    frames = ObjectProperty(None)
    fileformat = ObjectProperty(None)
    framerate = ObjectProperty(None)
    

# camera properties
class CameraProperties(GridLayout):
    # camera properties
    gain = ObjectProperty(None)
    exposure = ObjectProperty(None)
    framerate = ObjectProperty(None)

     # update camera params when text or slider is changed
    def change_gain(self):
        camera = App.get_running_app().camera
        if camera is not None:
            camera.Gain= float(self.gain.value)
            self.gain.value = camera.Gain()
        else:
            self.gain.value = 0

    def change_exposure(self):
        camera = App.get_running_app().camera
        if camera is not None:
            camera.ExposureTime = float(self.exposure.value)
            self.exposure.value = camera.ExposureTime()
        else:
            self.exposure.value = 0

    def change_framerate(self):
        """update framerate"""
        camera = App.get_running_app().camera
        if camera is not None:
            camera.AcquisitionFrameRateEnable = True
            camera.AcquisitionFrameRate = float(self.framerate.value)
            self.framerate.value = camera.ResultingFrameRate()
        else:
            self.framerate.value = 0
### TODO check if event unscheduling succesful
import threading
#record and live view buttons
class RecordButtons(BoxLayout):
    recordbutton = ObjectProperty(None)

    def __init__(self,  **kwargs):
        super(RecordButtons, self).__init__(**kwargs)
        
    def startPreview(self):
        camera = App.get_running_app().camera
        if camera is not None:
            basler.start_grabbing(camera)
            # update the image
            fps = App.get_running_app().root.ids.leftcolumn.ids.camprops.framerate.value
            print("Display Framerate:", fps)
            self.event = Clock.schedule_interval(self.update, 1.0 / fps/2)
            
    def stopPreview(self):
        camera = App.get_running_app().camera
        if camera is not None:
            basler.stop_grabbing(camera)
        Clock.unschedule(self.event)
        self.parent.framecounter.value = 0

    def startRecording(self):
        camera = App.get_running_app().camera
        if camera is not None:
            nframes = int(App.get_running_app().root.ids.rightcolumn.nframes)
            basler.start_grabbing(camera, numberOfImagesToGrab = nframes, record = True)
            # update the image
            fps = App.get_running_app().root.ids.leftcolumn.ids.camprops.framerate.value
            print("Display Framerate:", fps)
            self.event = Clock.schedule_interval(self.record, 1.0 / fps)

    def stopRecording(self):
        camera = App.get_running_app().camera
        if camera is not None:
            basler.stop_grabbing(camera)
        Clock.unschedule(self.event)
        self.parent.framecounter.value = 0

    def update(self, dt, save = False):
        camera = App.get_running_app().camera
        if camera is not None and camera.IsGrabbing():
            ret, img = basler.retrieve_result(camera)
            if ret:
                buf = img.tobytes()
                image_texture = Texture.create(
                    size=(img.shape[1], img.shape[0]), colorfmt="luminance"
                )
                image_texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
                App.get_running_app().root.ids.middlecolumn.previewimage.texture = image_texture
                self.parent.framecounter.value += 1

    def record(self, dt):
        camera = App.get_running_app().camera
        nframes = int(App.get_running_app().root.ids.rightcolumn.nframes)
        if camera is not None:
            if camera.IsGrabbing() and self.parent.framecounter.value < nframes:
                ret, img = basler.retrieve_result(camera)
                if ret:
                    # show on GUI
                    buf = img.tobytes()
                    image_texture = Texture.create(
                        size=(img.shape[1], img.shape[0]), colorfmt="luminance"
                    )
                    image_texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
                    App.get_running_app().root.ids.middlecolumn.ids.previewimage.texture = image_texture
                    #thread1 = threading.Thread(target=self.save, args=(img, ))
                    #thread1.start()
                    self.save(img)
                    self.parent.framecounter.value +=  1
                    print(self.parent.framecounter.value)

            else:
                self.recordbutton.state = 'normal'

    def save(self, img):
        # save image to file
        path =  App.get_running_app().root.ids.leftcolumn.savefile
        ext = App.get_running_app().root.ids.rightcolumn.fileformat
        fname = f"basler_{self.parent.framecounter.value}{ext}"
        basler.save_image(img,path,fname)

    
# image preview
class PreviewImage(Image):
    previewimage = ObjectProperty(None)
    def __init__(self,  **kwargs):
        super(PreviewImage, self).__init__(**kwargs)
        
class RuntimeControls(BoxLayout):
    framecounter = ObjectProperty(rebind = True)
    def on_framecounter(self, instance, value):
        self.text = str(value)
    
# display if hardware is connected
class Connections(BoxLayout):
    cam_connection = ObjectProperty(None)
    stage_connection = ObjectProperty(None)
    
    def __init__(self,  **kwargs):
        super(Connections, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)

    def _do_setup(self, *l):
        self.stage_connection.state ='down'
        self.cam_connection.state ='down'
    
    def connectCamera(self):
        print('connecting Camera')
        # connect camera
        App.get_running_app().camera = basler.camera_init()
        #
        if App.get_running_app().camera is None:
            self.cam_connection.state = 'normal'
        else:
            # load and apply default pfs
            self.parent.parent.ids.leftcolumn.apply_cam_settings()

    def disconnectCamera(self):
        camera = App.get_running_app().camera
        if App.get_running_app().camera is not None:
            print('disconnecting')
            camera.Close()
        
    def connectStage(self):
        print('connecting Stage')
        stage = stg.Stage(port='/dev/ttyUSB0')
        if stage is not None:
            App.get_running_app().stage = stage
        if App.get_running_app().stage is None:
            self.stage_connection.state = 'normal'
        else:
            # home stage - do this in a trhead it is slow
            t = Thread(target=App.get_running_app().stage.on_connect, args = (False, ))
            # set daemon to true so the thread dies when app is closed
            t.daemon = True
            # start the thread
            t.start()
            
        
    def disconnectStage(self):
        print('disconnecting Stage')
        if App.get_running_app().stage is None:
            self.stage_connection.state = 'normal'
        else:
            App.get_running_app().stage.disconnect()
        

class MyCounter(Label):
    value = NumericProperty(0)

class ExitApp(BoxLayout):
   stop = ObjectProperty(None)
   cancel = ObjectProperty(None)
# set window size at startup
Window.size = (1280, 800)
# load the layout
class MacroscopeApp(App):
    def __init__(self,  **kwargs):
        super(MacroscopeApp, self).__init__(**kwargs)
        # bind key presses to stage motion
        ### TODO implement key press fxns
        # Window.bind(on_key_up=self._keyup)
        # Window.bind(on_key_down=self._keydown)
        # Window.bind(on_key_left=self._keyleft)
        # Window.bind(on_key_right=self._keyright)
        self.camera = None
        self.stage = None
    
    def build(self):
        layout = Builder.load_file('layout.kv')
        # connect x-close button to action
        Window.bind(on_request_close=self.on_request_close)
        return layout

    
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
