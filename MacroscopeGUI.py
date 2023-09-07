import os
# Suppress kivy normal initialization logs in the beginning
# for easier debugging
os.environ["KCFG_KIVY_LOG_LEVEL"] = "warning"

import kivy
kivy.require('2.0.0')

from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config
from kivy.cache import Cache
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stencilview import StencilView
from kivy.graphics.transformation import Matrix
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar
from kivy.factory import Factory
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty, StringProperty, BoundedNumericProperty, NumericProperty, ConfigParserProperty, ListProperty
from kivy.clock import Clock
from threading import Thread
import _thread as thread
from functools import partial
from pathlib import Path
import numpy as np
import datetime
import os
import time
from Zaber_control import Stage, AxisEnum
import Macroscope_macros as macro
import Basler_control as basler
from pypylon import pylon
from skimage.io import imsave
#import cv2
#from pypylon import pylon
#from pypylon import genicam
from typing import Tuple
from concurrent.futures import ThreadPoolExecutor
import math

# get the free clock (more accurate timing)
#Config.set('graphics', 'KIVY_CLOCK', 'free')
Config.set('modules', 'monitor', '')
# helper functions
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    """This creates a timestamped filename so we don't overwrite our good work."""
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def im_to_texture(image):
    """helper function to create kivy textures from image arrays."""

    buf = image.tobytes()
    w, h = image.shape
    image_texture = Texture.create(
        size=(h, w), colorfmt="luminance"
    )
    image_texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
    return image_texture


class WarningPopup(Popup):
    ok_text = StringProperty('OK')
    text = StringProperty('Label')

    def __init__(self, text = 'warning', **kwargs):
        super(WarningPopup, self).__init__(**kwargs)
        self.text = text
         # call dismiss_popup in 2 seconds
        Clock.schedule_once(self.dismiss, 2)

    def ok(self):
        self.dismiss()


# main GUI
class MainWindow(GridLayout):
    pass


class LeftColumn(BoxLayout):
    # file saving and loading
    loadfile = ObjectProperty(None)
    savefile = StringProperty("")
    cameraprops = ObjectProperty(None)
    saveloc = ObjectProperty(None)
    #

    def __init__(self,  **kwargs):
        super(LeftColumn, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)

    def _do_setup(self, *l):
        self.savefile = App.get_running_app().config.get("Experiment", "exppath")
        self.path_validate()
        self.loadfile = App.get_running_app().config.get('Camera', 'default_settings')
        self.apply_cam_settings()


    def path_validate(self):
        p = Path(self.saveloc.text)
        app = App.get_running_app()
        if p.exists() and p.is_dir():
            app.config.set("Experiment", "exppath", self.saveloc.text)
            app.config.write()
        # check if the parent dir exists, then create the folder
        elif p.parent.exists():
            p.mkdir(mode=0o777, parents=False, exist_ok=True)
        else:
            self.saveloc.text = self.savefile
        app.config.set("Experiment", "exppath", self.saveloc.text)
        app.config.write()
        self.savefile = App.get_running_app().config.get("Experiment", "exppath")
        # reset the stage keys
        app.bind_keys()
        print('saving path changed')


    def dismiss_popup(self):
        App.get_running_app().bind_keys()
        self._popup.dismiss()


    # popup camera file selector
    def show_load(self):
        content = LoadCameraProperties(load=self.load, cancel=self.dismiss_popup)
        content.ids.filechooser2.path = self.loadfile
        self._popup = Popup(title="Load camera file", content=content,
                            size_hint=(0.9, 0.9))
         #unbind keyboard events
        App.get_running_app().unbind_keys()
        self._popup.open()


    # popup experiment dialog selector
    def show_save(self):
        content = SaveExperiment(save=self.save, cancel=self.dismiss_popup)
        content.ids.filechooser.path = self.savefile
        self._popup = Popup(title="Select save location", content=content,
                            size_hint=(0.9, 0.9))
        #unbind keyboard events
        App.get_running_app().unbind_keys()
        self._popup.open()


    def load(self, path, filename):
        self.loadfile = os.path.join(path, filename[0])
        self.apply_cam_settings()
        self.dismiss_popup()


    def save(self, path, filename):
        self.savefile = os.path.join(path, filename)
        self.saveloc.text = (self.savefile)
        self.path_validate()
        self.dismiss_popup()


    def apply_cam_settings(self):
        camera = App.get_running_app().camera
        if camera is not None:
            print('Updating camera settings')
            basler.update_props(camera, propfile=self.loadfile)
            self.update_settings_display()


    # when file is loaded - update slider values which updates the camera
    def update_settings_display(self):
        # update slider value using ids
        camera = App.get_running_app().camera
        self.ids.camprops.exposure = camera.ExposureTime()
        self.ids.camprops.gain = camera.Gain()
        self.ids.camprops.framerate = camera.ResultingFrameRate()


    #autofocus popup
    def show_autofocus(self):
        app = App.get_running_app()
        camera = app.camera
        stage = app.stage
        if camera is not None and stage is not None:
            content = AutoFocus(run_autofocus = self.run_autofocus, cancel=self.dismiss_popup)
            self._popup = Popup(title="Focus the camera", content=content,
                                size_hint=(0.9, 0.9))
            #unbind keyboard events
            App.get_running_app().unbind_keys()
            self._popup.open()
        else:
            self._popup = WarningPopup(title="Autofocus", text='Autofocus requires a stage and a camera!',
                            size_hint=(0.5, 0.25))
            self._popup.open()


    # run autofocussing once on current location
    def run_autofocus(self):
        app = App.get_running_app()
        camera = app.camera
        stage = app.stage

        if camera is not None and stage is not None:
            # stop grabbing
            app.root.ids.middlecolumn.ids.runtimecontrols.ids.recordbuttons.ids.liveviewbutton.state = 'normal'
            # get config values
            stepsize = self._popup.content.stepsize#app.config.getfloat('Autofocus', 'step_size')
            stepunits = self._popup.content.stepunits#app.config.get('Autofocus', 'step_units')
            nsteps = self._popup.content.nsteps#app.config.getint('Autofocus', 'nsteps')
            # run the autofocus
            _, imstack, _, focal_plane = macro.zFocus(stage, camera, stepsize, stepunits, nsteps)
            # update the images shown - delete old ones if rerunning
            self._popup.content.delete_images()
            self._popup.content.add_images(imstack, nsteps, focal_plane)


class MiddleColumn(BoxLayout, StencilView):
    runtimecontrols = ObjectProperty(None)
    previewimage = ObjectProperty(None)


class RightColumn(BoxLayout):
    #
    #
    def __init__(self,  **kwargs):
        super(RightColumn, self).__init__(**kwargs)


    def dismiss_popup(self):
        #rebind keyboard events
        App.get_running_app().bind_keys()
        self._popup.dismiss()


    def show_recording_settings(self):
        """change recording settings."""
        content = RecordingSettings(ok=self.dismiss_popup)
        self._popup = Popup(title="Recording Settings", content=content,
                            size_hint=(0.5, 0.35))
        #unbind keyboard events
        App.get_running_app().unbind_keys()
        self._popup.open()


    def show_calibration(self):
        app = App.get_running_app()
        camera = app.camera
        stage = app.stage
        if camera is not None and stage is not None:
            content = AutoCalibration(calibrate=self.calibrate, cancel=self.dismiss_popup, \
                                )
            self._popup = Popup(title="Autocalibration", content=content,
                                size_hint=(0.9, 0.75))
            self._popup.open()
        else:
            self._popup = WarningPopup(title="Calibration", text='Autocalibration requires a stage and a camera. Connect a stage or use a calibration slide.',
                            size_hint=(0.5, 0.25))
            self._popup.open()


    def calibrate(self):
        app = App.get_running_app()
        camera = app.camera
        stage = app.stage

        if camera is not None and stage is not None:
            # stop camera if already running
            app.root.ids.middlecolumn.ids.runtimecontrols.ids.recordbuttons.ids.liveviewbutton.state = 'normal'
            
            # get config values
            stepsize = app.config.getfloat('Calibration', 'step_size')
            stepunits = app.config.get('Calibration', 'step_units')

            # run the calibration
            img1, img2 = macro.take_calibration_images(stage, camera, stepsize, stepunits)
            self._popup.content.ids.image_one.texture = im_to_texture(img1)
            self._popup.content.ids.image_two.texture = im_to_texture(img2)
            
            # calculate calibration from shift
            pxsize, rotation = macro.getCalibrationMatrix(img1, img2, stepsize)
            app.config.set('Camera', 'pixelsize', pxsize)
            app.config.set('Camera', 'rotation', rotation)
            
            # update calibration matrix
            app.calibration_matrix = macro.genCalibrationMatrix(pxsize, rotation)

            # update labels shown
            self._popup.content.ids.pxsize.text = f"Pixelsize ({app.config.get('Calibration', 'step_units')}/px)  {app.config.getfloat('Camera', 'pixelsize'):.2f}"
            self._popup.content.ids.rotation.text = f"Rotation (rad)  {app.config.getfloat('Camera', 'rotation'):.3f}"

            # save configs
            app.config.write()


class AutoCalibration(BoxLayout):
    calibrate = ObjectProperty(None)
    cancel = ObjectProperty(None)


# Stage controls
class XControls(BoxLayout):
    def __init__(self,  **kwargs):
        super(XControls, self).__init__(**kwargs)

    def disable_all(self):
        for id in self.ids:
            self.ids[id].disabled = True
    
    def enable_all(self):
        for id in self.ids:
            self.ids[id].disabled = False


class YControls(Widget):
    def __init__(self,  **kwargs):
        super(YControls, self).__init__(**kwargs)

    def disable_all(self):
        for id in self.ids:
            self.ids[id].disabled = True
    
    def enable_all(self):
        for id in self.ids:
            self.ids[id].disabled = False


class ZControls(Widget):
    def __init__(self,  **kwargs):
        super(ZControls, self).__init__(**kwargs)

    def disable_all(self):
        for id in self.ids:
            self.ids[id].disabled = True

    def enable_all(self):
        for id in self.ids:
            self.ids[id].disabled = False


# load camera settings
class LoadCameraProperties(BoxLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


# save location for images and meta data
class SaveExperiment(GridLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)


# save location for images and meta data
class AutoFocus(BoxLayout):
    run_autofocus = ObjectProperty(None)
    cancel = ObjectProperty(None)
    # make config editable
    nsteps = ConfigParserProperty(5, 'Autofocus', 'nsteps', 'app', val_type=int, )
    stepsize = ConfigParserProperty(50, 'Autofocus', 'step_size', 'app', val_type=float)
    stepunits = ConfigParserProperty('um', 'Autofocus', 'step_units', 'app', val_type=str)

    def __init__(self,  **kwargs):
        super(AutoFocus, self).__init__(**kwargs)
        self.imagewidgets = []


    def add_images(self, imstack, n, focal_plane):
         # build as many labelled images as we will need
        for i in range(n):
            tmp = Factory.LabelImage()
            tmp.label.text = f'Image {i}'
            if i == focal_plane:
                tmp.label.color = 'red'
                tmp.label.text = f'Focus: Image {i}'
            tmp.image.texture = im_to_texture(imstack[i])
            self.imagewidgets.append(tmp)
            self.ids.multipleimages.add_widget(tmp)


    def delete_images(self):
        for wg in self.imagewidgets:
            self.ids.multipleimages.remove_widget(wg)



class LabelImage():
    def __init__(self,  **kwargs):
        super(LabelImage, self).__init__(**kwargs)
        self.text = ''
        self.texture = ''


class MultipleImages(GridLayout):
    pass


class RecordingSettings(BoxLayout):
    ok = ObjectProperty(None)
    # store recording settings from popups
    nframes = ConfigParserProperty(5, 'Experiment', 'nframes', 'app', val_type=int)
    fileformat = ConfigParserProperty('jpg', 'Experiment', 'extension', 'app', val_type=str)
    framerate = ConfigParserProperty(25, 'Experiment', 'framerate', 'app', val_type=float)
    duration = ConfigParserProperty(5, 'Experiment', 'duration', 'app', val_type=float)
    buffersize = ConfigParserProperty(1000, 'Experiment', 'buffersize', 'app', val_type=int)

    def __init__(self,  **kwargs):
        super(RecordingSettings, self).__init__(**kwargs)
        self.duration = self.nframes/self.framerate


# camera properties
class CameraProperties(GridLayout):
    # camera properties
    gain = NumericProperty(0)
    exposure = NumericProperty(0)
    framerate = NumericProperty(0)

     # update camera params when text or slider is changed
    def change_gain(self):
        camera = App.get_running_app().camera
        if camera is not None:
            camera.Gain = float(self.gain)
            self.gain = camera.Gain()
        else:
            self.gain = 0


    def change_exposure(self):
        camera = App.get_running_app().camera
        if camera is not None:
            camera.ExposureTime = float(self.exposure)
            self.exposure = camera.ExposureTime()
        else:
            self.exposure = 0


    def change_framerate(self):
        """update framerate"""
        camera = App.get_running_app().camera
        if camera is not None:
            camera.AcquisitionFrameRateEnable = True
            camera.AcquisitionFrameRate = float(self.framerate)
            self.framerate = camera.ResultingFrameRate()
        else:
            self.framerate = 0


#record and live view buttons
class RecordButtons(BoxLayout):
    recordbutton = ObjectProperty(None, rebind = True)
    liveviewbutton = ObjectProperty(None, rebind = True)
    snapbutton = ObjectProperty(None, rebind = True)

    def __init__(self,  **kwargs):
        super(RecordButtons, self).__init__(**kwargs)

    def snap(self):
        """save a single image to experiment location."""
        app = App.get_running_app()
        ext = app.config.get('Experiment', 'extension')
        path = app.root.ids.leftcolumn.savefile
        snap_filename = timeStamped("snap."+f"{ext}")
        
        if app.camera is None:
            return
        
        # Get an image appropriately acoording to current viewing mode
        if self.liveviewbutton.state == 'normal':
            # Call capture an image
            isSuccess, img = basler.single_take(app.camera)
            if isSuccess:
                basler.save_image(img,path,snap_filename)
                
        elif self.liveviewbutton.state == 'down':
            # If currently in live view mode
            #   then save the current image
            img = app.lastframe
            basler.save_image(img,path,snap_filename)
                


    def startPreview(self):
        camera = App.get_running_app().camera
        if camera is not None:
            # create a texture
            #App.get_running_app().create_texture(*basler.get_shape(camera))
            basler.start_grabbing(camera)
            # update the image
            fps = camera.ResultingFrameRate()
            print("Grabbing Framerate:", fps)
            self.updatethread = Thread(target=self.update, daemon = True)
            self.updatethread.start()
        else:
            self.liveviewbutton.state = 'normal'


    def stopPreview(self):
        app = App.get_running_app()
        camera = app.camera
        print('stopping preview')
        if camera is not None:
            Clock.unschedule(self.event)
            if self.updatethread.is_alive():
                self.updatethread.join()
            basler.stop_grabbing(camera)
        # reset displayed framecounter
        self.parent.framecounter.value = 0
        # reset scale of image
        app.root.ids.middlecolumn.ids.scalableimage.reset()


    def update(self):
        app = App.get_running_app()
        camera = app.camera
        
        # schedule a display update
        fps = app.config.getfloat('Camera', 'display_fps')
        self.event = Clock.schedule_interval(self.display, 1.0 /fps)
        print(f'Displaying at {fps} fps')
            
        while camera is not None and camera.IsGrabbing() and self.liveviewbutton.state == 'down':
            isSuccess, img, timestamp, retrieveTimestamp = basler.retrieve_grabbing_result(camera)
            if isSuccess:
                #print('dt: ', timestamp-app.timestamp)
                cropY, cropX = self.parent.cropY, self.parent.cropX
                
                app.lastframe = img[cropY:img.shape[0]-cropY, cropX:img.shape[1]-cropX]
                app.timestamp = timestamp
                app.retrieveTimestamp = retrieveTimestamp
                self.parent.framecounter.value += 1
        return


    def display(self, dt):
        if App.get_running_app().lastframe is not None:
            App.get_running_app().image = App.get_running_app().lastframe
    

    def update_buffer(self, dt):
        # update buffer display
        camera = App.get_running_app().camera
        self.parent.buffer.value = camera.MaxNumBuffer()- camera.NumQueuedBuffers()
        

    def startRecording(self):
        app = App.get_running_app()
        camera = app.camera

        self.parent.framecounter.value = 0
        self.path = app.root.ids.leftcolumn.savefile

        if camera is not None:
            # stop camera if already running 
            self.liveviewbutton.state = 'normal'

            # open coordinate file
            self.open_coord_file()

            # schedule buffer update
            self.buffertrigger = Clock.create_trigger(self.update_buffer)
            
            # Create thread pool for image saving
            self.savingthreadpool = ThreadPoolExecutor(thread_name_prefix='RecordingThreadPool')

            # start thread for grabbing and saving images
            record_args = self.init_recording()
            self.recordthread = Thread(target=self.record, args = record_args, daemon = True)
            self.recordthread.start()

        else:
            self.recordbutton.state = 'normal'


    def stopRecording(self):
        camera = App.get_running_app().camera

        if camera is None:
            return
        
        # Unschedule self.display() event
        Clock.unschedule(self.event)

        # Schedule closing coordinate file a bit later
        Clock.schedule_once(lambda dt: self.coordinate_file.close(), 0.5)
        
        # Close recording threads 
        if self.recordthread.is_alive():
            self.recordthread.join()
        
        # Close the saving thread pool
        self.savingthreadpool.shutdown(wait= True)
            
        # Tell camera to stop grabbing mode
        basler.stop_grabbing(camera)
        # Reset frame counter
        self.parent.framecounter.value = 0
        # Call update_buffer() once
        self.buffertrigger()
        # Reset scale of image
        App.get_running_app().root.ids.middlecolumn.ids.scalableimage.reset()
        # Set button state back to normal
        self.recordbutton.state = 'normal'
        
        print("Finished recording")


    def init_recording(self) -> Tuple[ int, int, int, int ]:
        """Initialize recording parameters

        Returns:
            nframes (int): number of frames to grab
            buffersize (int): number of buffers to allocate, used for grabbing
            cropX (int): starting cropping position in X
            cropY (int): starting cropping position in Y
        """
        app = App.get_running_app()
        # set up grabbing with recording settings here
        fps = app.root.ids.leftcolumn.ids.camprops.framerate
        
        nframes = app.config.getint('Experiment', 'nframes')
        buffersize = app.config.getint('Experiment', 'buffersize')
        # get cropping
        cropY, cropX = self.parent.cropY, self.parent.cropX

        print("Desired recording Framerate:", fps)
        # set recording framerate - returns
        fps = basler.set_framerate(app.camera, fps)
        print('Actual recording fps: ' + str(fps))
        app.root.ids.leftcolumn.update_settings_display()

        # precalculate the filename
        ext = app.config.get('Experiment', 'extension')
        self.image_filename = timeStamped("basler_{}_{}."+f"{ext}")
        return nframes, buffersize, cropX, cropY


    def record(self, nframes: int, buffersize: int, cropX: int, cropY: int) -> None:
        """Camera recording loop. Start the camera grabbing mode and a loop to repeatedly call retrieving result.

        Args:
            nframes (int): maximum number of frames to record
            buffersize (int): number of camera's internal frame buffers
            cropX (int): cropping position in X
            cropY (int): cropping position in Y
        """
        app = App.get_running_app()
        camera = app.camera
        stage: Stage = app.stage
        basler.start_grabbing(app.camera, numberOfImagesToGrab=nframes, record=True, buffersize=buffersize)

        # schedule a display update
        fps = app.config.getfloat('Camera', 'display_fps')
        self.event = Clock.schedule_interval(self.display, 1.0 /fps)
        print(f'Displaying at {fps} fps')
        
        # NEW
        rel_positions = [[0,26,0],
                         [26,0,0],
                         [26,0,0],
                         [26,0,0],
                         [0,-26,0],
                         [0,-26,0],
                         [-26,0,0],
                         [0,26,0],
                         [-26,0,0],
                         [0,-26,0],
                         [-26,0,0],
                         [0,26,0],
                         ]
        """
        rel_positions = [[0,26,0],
                         [0,26,0],
                         [26,0,0],
                         [0,-26,0],
                         [0,-26,0],
                         [-26,0,0]]"""

        target_pos_index = 0
        camera_spf = 1 / camera.ResultingFrameRate()
        well_record_time = (60*5)-5 #in sec

        
        # grab and write images
        counter = 0
        well_counter = 0
        while camera is not None and counter <nframes and self.recordbutton.state == 'down':

            # NEW
            # Moving to target position
            if well_counter == 0:
                target_pos = rel_positions[target_pos_index]
                print(f'Moving to new relative pos {target_pos}')
                #stage.move_abs(target_pos, 'mm', wait_until_idle= True)
                stage.move_rel(target_pos, 'mm', wait_until_idle= True)
                
                # Wait for new position image
                time.sleep(camera_spf*5)
                well_start = time.perf_counter()

            # get image
            isSuccess, img, timestamp, retrieveTimestamp = basler.retrieve_grabbing_result(camera)
            # trigger a buffer update
            self.buffertrigger()
            if isSuccess:
                # print('dt: ', timestamp-app.timestamp)
                # print('(x,y,z): ', app.coords)
                
                # Crop the retrieved image and set as the latest frame
                app.lastframe = img[cropY:img.shape[0]-cropY, cropX:img.shape[1]-cropX]
                # write coordinate into file
                self.coordinate_file.write(f"{self.parent.framecounter.value} {target_pos_index} {timestamp} {app.coords[0]} {app.coords[1]} {app.coords[2]} \n")

                # Apply an image saving job into the pool if the pool is not closed yet
                if not self.savingthreadpool._shutdown_lock.locked():
                    self.savingthreadpool.submit(
                        basler.save_image, 
                        app.lastframe, 
                        self.path, 
                        self.image_filename.format(target_pos_index, self.parent.framecounter.value)
                    )
                
                # update time and frame counter
                app.timestamp = timestamp
                app.retrieveTimestamp = retrieveTimestamp
                self.parent.framecounter.value += 1
                counter += 1
                # NEW
                well_counter += 1
                
                # NEW
                if time.perf_counter() >= well_start + well_record_time: # time passed
                    target_pos_index += 1
                    target_pos_index = target_pos_index % len(rel_positions)
                    well_counter = 0
        
        print(f'Finished recordings {counter} frames.')
        self.buffertrigger()
        self.recordbutton.state = 'normal'
        
        return


    def open_coord_file(self, *args):
        """open coordinate file."""
        # open a file for the coordinates
        self.coordinate_file = open(os.path.join(self.path, timeStamped("coords.txt")), 'a')
        self.coordinate_file.write(f"Frame Time X Y Z \n")


class ScalableImage(ScatterLayout):
    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if touch.is_mouse_scrolling:
                if touch.button == 'scrollup':
                    mat = Matrix().scale(.9, .9, .9)
                    self.apply_transform(mat, anchor=touch.pos)
                elif touch.button == 'scrolldown':
                    mat = Matrix().scale(1.1, 1.1, 1.1)
                    self.apply_transform(mat, anchor=touch.pos)
        return super().on_touch_up(touch)

    # reset tranformation
    def reset(self):
        self.scale = 1
        self.center_x = self.parent.center_x
        self.center_y = self.parent.center_y


# image preview
class PreviewImage(Image):
    #previewimage = ObjectProperty(None)
    circle= ListProperty([0, 0, 0])
    offset = ListProperty([0, 0])

    def __init__(self,  **kwargs):
        super(PreviewImage, self).__init__(**kwargs)
        Window.bind(mouse_pos=self.mouse_pos)

    def mouse_pos(self, window, pos):
        pos = self.to_widget(pos[0], pos[1])
        # read mouse hover events and get image value
        if self.collide_point(*pos):
            #print(*pos, self.center_x, self.center_y, self.norm_image_size)
            # by default the touch coordinates are relative to GUI window
            #wx, wy = self.to_widget(pos[0], pos[1], relative = True)
            wx, wy = pos[0], pos[1]
            image = App.get_running_app().image
            # get the image we last took
            if image is not None:
                texture_w, texture_h = self.norm_image_size
                #offset if the image is not fitting inside the widget
                cx, cy = self.center_x, self.center_y  #, relative = True)
                ox, oy = cx - texture_w / 2., cy - texture_h/ 2
                h, w = image.shape

                imy, imx = int((wy-oy)*h/texture_h), int((wx-ox)*w/texture_w)
                if 0 <= imy < h and 0 <= imx < w:
                    val = image[imy, imx]
                    App.get_running_app().root.ids.middlecolumn.ids.pixelvalue.text = f'({imx},{imy},{val})'
                    #self.parent.parent.parent.ids.


    def captureCircle(self, pos):
        """define the capture circle and draw it."""
        wx, wy = pos#self.to_widget(pos[0], pos[1])#, relative = True)
        image = App.get_running_app().image
        h, w = image.shape
        # paint a circle and make the coordinates available
        radius = App.get_running_app().config.getfloat('Tracking', 'capture_radius')
        # make the circle into pixel units
        r = radius/w*self.norm_image_size[0]#, radius/h*self.norm_image_size[1]
        self.circle = (*pos, r)
        # calculate in image units where the click was relative to image center and return that
        #offset if the image is not fitting inside the widget
        texture_w, texture_h = self.norm_image_size
        #offset if the image is not fitting inside the widget
        cx, cy = self.center_x, self.center_y
        ox, oy = cx - texture_w / 2., cy - texture_h/ 2
        imy, imx = int((wy-oy)*h/texture_h), int((wx-ox)*w/texture_w)
        # offset of click from center of image - origin is left lower corner
        self.offset = (imy-h//2, imx-w//2)


    def clearcircle(self):
        self.circle = (0, 0, 0)


    # # for reading mouse clicks
    def on_touch_down(self, touch):
        rtc = App.get_running_app().root.ids.middlecolumn.runtimecontrols
        # transform to local because of scatter
        #pos = self.to_widget(touch.pos[0], touch.pos[1])
        # if a click happens in this widget
        if self.collide_point(*touch.pos):
            #if tracking is active and not yet scheduled:
            if rtc.trackingcheckbox.state == 'down' and not rtc.trackingevent:
                # Draw a red circle
                self.captureCircle(touch.pos)
                # Start tracking procedure
                Clock.schedule_once(lambda dt: rtc.startTracking(), 0)
                # remove the circle 
                # Clock.schedule_once((lambda dt: self.circle = (0, 0, 0)), 0.5)
                Clock.schedule_once(lambda dt: self.clearcircle(), 0.5)


class RuntimeControls(BoxLayout):
    framecounter = ObjectProperty(rebind=True)
    autofocuscheckbox = ObjectProperty(rebind=True)
    trackingcheckbox = ObjectProperty(rebind=True)
    cropX = ObjectProperty(0, rebind=True)
    cropY = ObjectProperty(0, rebind=True)
    

    def __init__(self,  **kwargs):
        super(RuntimeControls, self).__init__(**kwargs)
        self.focus_history = []
        self.focusevent = None
        self.focus_motion = 0
        self.trackingevent = False
        self.coord_updateevent = None


    def on_framecounter(self, instance, value):
        self.text = str(value)


    def startFocus(self):
        # schedule a focus routine
        camera = App.get_running_app().camera
        stage = App.get_running_app().stage
        if camera is not None and stage is not None and camera.IsGrabbing():
             # get config values
            focus_fps = App.get_running_app().config.getfloat('Livefocus', 'focus_fps')
            print("Focus Framerate:", focus_fps)
            z_step = App.get_running_app().config.getfloat('Livefocus', 'min_step')
            unit = App.get_running_app().config.get('Livefocus', 'step_units')
            factor = App.get_running_app().config.getfloat('Livefocus', 'factor')

            self.focusevent = Clock.schedule_interval(partial(self.focus,  z_step, unit, factor), 1.0 / focus_fps)
        else:
            self._popup = WarningPopup(title="Autofocus", text='Focus requires: \n - a stage \n - a camera \n - camera needs to be grabbing.',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            self.autofocuscheckbox.state = 'normal'


    def stopFocus(self):
        # unschedule a focus routine
        if self.focusevent:
            Clock.unschedule(self.focusevent)


    def focus(self, z_step, unit, focus_step_factor, *args):
        # run the actual focus routine - calculate the focus values and correct accordinly.
        start = time.time()
        app = App.get_running_app()
        if not app.camera.IsGrabbing():
            self.autofocuscheckbox.state = 'normal'
            return
        # calculate current value of focus
        self.focus_history.append(macro.calculate_focus(app.image))

        # calculate control variables if we have enough history
        if len(self.focus_history)>1 and self.focus_motion != 0:
            self.focus_motion = macro.calculate_focus_move(self.focus_motion, self.focus_history, z_step, focus_step_factor)
        else:
            self.focus_motion = z_step
        print('Move (z)',self.focus_motion,unit)
        app.stage.move_z(self.focus_motion, unit)
        app.coords[2] += self.focus_motion/1000
        # throw away stuff
        self.focus_history = self.focus_history[-1:]

        #print('Saving time: ',time.time() - start)
        return


    def trackingButtonCallback(self):
        """Display tracking instruction popup
        """
        app = App.get_running_app()
        # schedule a tracking routine
        camera = app.camera
        stage = app.stage

        if camera is not None and stage is not None and camera.IsGrabbing():
             # get config values
            # find an animal and center it once by moving the stage
            self._popup = WarningPopup(title="Click on animal", text = 'Click on an animal to start tracking it.',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            # make a capture circle - all of this happens in Image Widget, and record offset from center, then dispatch the centering routine
            #schedule a tracking loop
        else:
            self._popup = WarningPopup(title="Tracking", text='Tracking requires a stage, a camera and the camera needs to be grabbing.',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            self.trackingcheckbox.state = 'normal'


    def stopTracking(self):
        self.trackingevent = False
        # unschedule a tracking routine
        #if self.trackthread.is_alive():
        if self.coord_updateevent is not None:
            Clock.unschedule(self.coord_updateevent)
            self.coord_updateevent = None
        # reset camera params
        self.reset_ROI()
        self.cropX = 0
        self.cropY = 0


    def startTracking(self) -> None:
        """Start the tracking procedure by gathering variables, setting up the camera, and then spawn a tracking loop
        """
        app = App.get_running_app()
        stage = app.stage
        
        # 
        # Move stage based on user input - happens here.
        # 
        ystep, xstep = macro.getStageDistances(app.root.ids.middlecolumn.previewimage.offset, app.calibration_matrix)
        units = app.config.get('Calibration', 'step_units')
        minstep = app.config.getfloat('Tracking', 'min_step')
        print('Centering image',xstep, ystep, 'um')
        
        if xstep > minstep:
            stage.move_x(xstep, unit=units, wait_until_idle=True)
        if ystep > minstep:
            stage.move_y(ystep, unit=units, wait_until_idle=True)

        app.coords =  app.stage.get_position()
        print('updated coords')
        
        # 
        # Set smaller FOV for the worm
        # 
        roiX, roiY  = app.config.getint('Tracking', 'roi_x'), app.config.getint('Tracking', 'roi_y')
        self.set_ROI(roiX, roiY)
        
        # 
        # Start the tracking
        # 
        capture_radius = app.config.getint('Tracking', 'capture_radius')
        binning = app.config.getint('Tracking', 'binning')
        dark_bg = app.config.getboolean('Tracking', 'dark_bg')
        mode =  app.config.get('Tracking', 'mode')
        area = app.config.getint('Tracking', 'area')
        threshold = app.config.getfloat('Tracking', 'threshold')

        # make a tracking thread 
        track_args = minstep, units, capture_radius, binning, dark_bg, area, threshold, mode
        self.trackthread = Thread(target=self.tracking, args = track_args, daemon = True)
        self.trackthread.start()
        print('started tracking thread')
        # schedule occasional position check of the stage
        self.coord_updateevent = Clock.schedule_interval(lambda dt: stage.get_position(), 10)


    def tracking(self, minstep: int, units: str, capture_radius: int, binning: int, dark_bg: bool, area: int, threshold: int, mode: str) -> None:
        """Tracking function to be running inside a thread
        """
        app = App.get_running_app()
        stage: Stage = app.stage
        camera: pylon.InstantCamera = app.camera

        # Compute second per frame to determine the lower bound waiting time
        camera_spf = 1 / camera.ResultingFrameRate()
        
        self.trackingevent = True
        app.prevframe = None
        scale = 1.0

        estimated_next_timestamp: float | None = None

        while camera is not None and camera.IsGrabbing() and self.trackingcheckbox.state == 'down':

            # Handling image cycle synchronization.
            # Because the recording and tracking thread are asynchronous
            # and doesn't have the same priority, it could be the case that
            # one thread get executed more than the other and the estimated time
            # became inaccurate.
            wait_time = 0
            if estimated_next_timestamp is not None:
                
                img2_retrieveTimestamp = app.retrieveTimestamp
                diff_estimated_time = estimated_next_timestamp - img2_retrieveTimestamp

                # If the estimated time is approximately close to the image timestamp
                # then it's ok to use the current image. The epsilon in this case is 10% of the camera_spf
                if abs(diff_estimated_time)/camera_spf < 0.1:
                    pass
                else:
                    # If the estimated time is less than the current time
                    # then it is also ok to use the current image
                    if estimated_next_timestamp < img2_retrieveTimestamp:
                        pass
                    # If the estimated time is more than the current image timestamp
                    # then compute the estimated next cycle time and wait
                    else:
                        current_time = time.perf_counter()

                        diff_time_factor = (current_time - img2_retrieveTimestamp) / camera_spf
                        fractional_part, integer_part = math.modf(diff_time_factor)

                        wait_time = camera_spf * ( 1.0 - fractional_part )

                        time.sleep(wait_time)
            else:
                img2_retrieveTimestamp = app.retrieveTimestamp
                estimated_next_timestamp = app.retrieveTimestamp

            # Get the latest image
            img2 = app.lastframe
            img2_retrieveTimestamp = app.retrieveTimestamp

            print(f'{app.timestamp}, {app.retrieveTimestamp}')

            tracking_frame_start_time = time.perf_counter()

            # If prev frame is empty then use the same as current
            if app.prevframe is None:
                app.prevframe = img2
                
            # Extract worm position
            if mode=='Diff':
                ystep, xstep = macro.extractWormsDiff(app.prevframe, img2, capture_radius, binning, area, threshold, dark_bg)
            elif mode=='Min/Max':
                ystep, xstep = macro.extractWorms(img2, capture_radius = capture_radius,  bin_factor=binning, dark_bg = dark_bg, display = False)
            else:
                ystep, xstep = macro.extractWormsCMS(img2, capture_radius = capture_radius,  bin_factor=binning, dark_bg = dark_bg, display = False)
            
            # Compute relative distancec in each axis
            ystep, xstep = macro.getStageDistances([ystep, xstep], app.calibration_matrix)
            ystep *= scale
            xstep *= scale

            # getting stage coord is slow so we will interpolate from movements
            if abs(xstep) > minstep:
                stage.move_x(xstep, unit=units, wait_until_idle =False)
                app.coords[0] += xstep/1000.
                app.prevframe = img2
            if abs(ystep) > minstep:
                stage.move_y(ystep, unit=units, wait_until_idle = False)
                app.coords[1] += ystep/1000.
                app.prevframe = img2

            tracking_frame_end_time = time.perf_counter()

            #   Wait for stage movement to finish to not get motion blur.
            #   This could be done by checking with stage.is_busy().
            #   However, that function call is very costly (~3 secs) 
            #   and is not good for loop checking.
            #   So we are going to just estimate it here.

            #   Delay from receing the image in recording and tracking it
            delay_receive_image_and_tracking_time = tracking_frame_start_time - img2_retrieveTimestamp

            #   Time take to compute tracking
            computation_time = tracking_frame_end_time - tracking_frame_start_time

            #   Communication delay from host to stage is 20 ms
            communication_delay = 20e-3 

            #   Travel time
            #       Because x and y axis travel independently, the speed that we have to wait 
            #       is the maximum between the two.
            max_travel_dist = max(abs(xstep), abs(ystep))       # in micro meter : 1e-6
            stage_travel_time = stage.estimateTravelTime(max_travel_dist * 1e-3)

            #   Sums up all the waiting time ingredient
            tracking_process_time = delay_receive_image_and_tracking_time + computation_time + communication_delay + stage_travel_time 

            #   Compute the waiting time to reach the next receive image
            fractional_part, integer_part = math.modf(tracking_process_time / camera_spf )
            time_to_next_receive_image = (1.0 - fractional_part) * camera_spf

            #   Sums up the total time we need to wait, which are:
            #       communication delay
            #       + stage travelling time
            #       + time to receiving the last blurry image
            total_waiting_time = communication_delay + stage_travel_time + time_to_next_receive_image

            estimated_next_timestamp = tracking_frame_end_time + total_waiting_time

            # Wait
            time.sleep(total_waiting_time)

        # When the camera is not grabbing or is None and exit the loop, make sure to change the state button back to normal
        self.trackingcheckbox.state = 'normal'


    def cont_tracking(self, minstep, units, area, binning):
        """thread for tracking. Uses continual motion of the x,y axes"""
        app = App.get_running_app()
        stage = app.stage
        camera = app.camera
        while camera is not None and camera.IsGrabbing() and self.trackingcheckbox.state == 'down':
            # threshold and find objects
            coords = macro.extractWorms(app.lastframe, area, bin_factor=binning, li_init=10)
            print('tracking', coords)
            # if we find stuff move
            if len(coords) > 0:
                print(len(coords))
                offset = macro.getDistanceToCenter(coords, app.lastframe.shape)
                ystep, xstep = macro.getStageDistances(offset, app.calibration_matrix)
                # getting stage coord is slow so we will interpolate from movements
                if xstep > minstep:
                    stage.move_x(xstep, unit=units, wait_until_idle = True)
                    app.coords[0] += xstep/1000.
                if ystep > minstep:
                    stage.move_y(ystep, unit=units, wait_until_idle = True)
                    app.coords[1] += ystep/1000.
                print("Move stage (x,y)", xstep, ystep)
        # reset camera params
        camera = App.get_running_app().camera
        basler.cam_resetROI(camera)
        self.cropX = 0
        self.cropY = 0


    def set_ROI(self, roiX, roiY):
        app = App.get_running_app()
        camera = app.camera
        rec = app.root.ids.middlecolumn.ids.runtimecontrols.ids.recordbuttons.ids.recordbutton.state
        disp = app.root.ids.middlecolumn.ids.runtimecontrols.ids.recordbuttons.ids.liveviewbutton.state
       
        if rec == 'down':
            #basler.stop_grabbing(camera)
            rec = 'normal'
            # reset camera field of view to smaller size around center
            hc, wc = basler.cam_setROI(camera, roiX, roiY, center = True)
            rec = 'down'
        elif disp == 'down':
            #basler.stop_grabbing(camera)
            disp= 'normal'
            # reset camera field of view to smaller size around center
            hc, wc = basler.cam_setROI(camera, roiX, roiY, center = True)
            disp = 'down'
            # 
        print(hc, wc, roiX, roiY)
        # if desired FOV is smaller than allowed by camera, crop in GUI
        if wc > roiX:
            self.cropX = int((wc-roiX)//2)
        if hc > roiY:
            self.cropY = int((hc-roiY)//2)


    def reset_ROI(self):
        app = App.get_running_app()
        camera = app.camera
        basler.cam_resetROI(camera)

# display if hardware is connected
class Connections(BoxLayout):
    cam_connection = ObjectProperty(None)
    stage_connection = ObjectProperty(None)

    def __init__(self,  **kwargs):
        super(Connections, self).__init__(**kwargs)
        Clock.schedule_once(self._do_setup)


    def _do_setup(self, *l):
        self.stage_connection.state = 'down'
        self.cam_connection.state = 'down'


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
        if camera is not None:
            print('disconnecting')
            camera.Close()


    def connectStage(self):
        print('connecting Stage')
        app = App.get_running_app()
        port = app.config.get('Stage', 'port')
        maxspeed = float( app.config.get('Stage', 'maxspeed') )
        maxspeed_unit = app.config.get('Stage', 'maxspeed_unit')
        accel = float( app.config.get('Stage', 'acceleration') )
        accel_unit = app.config.get('Stage', 'acceleration_unit')
        stage = Stage(port, maxspeed, maxspeed_unit, accel, accel_unit)
        
        if stage.connection is None:
            self.stage_connection.state = 'normal'
            App.get_running_app().stage = None

        else:
            app.stage: Stage = stage
            
            homing = app.config.getboolean('Stage', 'homing')
            move_start = app.config.getboolean('Stage', 'move_start')
            startloc = [float(x) for x in app.config.get('Stage', 'start_loc').split(',')]
            limits = [float(x) for x in app.config.get('Stage', 'stage_limits').split(',')]
            
            def connect_async():

                # home stage - do this in a thread, it is slow, ~2 sec
                app.stage.on_connect(homing,  move_start, startloc, limits)
                
                # Call update_coordinates once.
                #   We have to specify not to run 'update_coordinates' in async mode because it's going to
                #   be run inside a thread.
                app.update_coordinates(isAsync= False)  

            
            thread_connect_async = Thread(target= connect_async)
            thread_connect_async.daemon = True
            thread_connect_async.start()
            
            app.root.ids.leftcolumn.ids.xcontrols.enable_all()
            app.root.ids.leftcolumn.ids.ycontrols.enable_all()
            app.root.ids.leftcolumn.ids.zcontrols.enable_all()
            

    def disconnectStage(self):
        print('disconnecting Stage')
        app = App.get_running_app()
        if app.stage is None:
            self.stage_connection.state = 'normal'
        else:
            app.stage.disconnect()
            app.stage = None
        # disable buttons
        app.root.ids.leftcolumn.ids.xcontrols.disable_all()
        app.root.ids.leftcolumn.ids.ycontrols.disable_all()
        app.root.ids.leftcolumn.ids.zcontrols.disable_all()


class MyCounter():
    value = NumericProperty(0)


class ExitApp(BoxLayout):
    stop = ObjectProperty(None)
    cancel = ObjectProperty(None)
# set window size at startup



# load the layout
class MacroscopeApp(App):
    # stage configuration properties - these will update when changed in config menu
    vhigh = ConfigParserProperty(20,
                    'Stage', 'vhigh', 'app', val_type=float)
    vlow = ConfigParserProperty(20,
                    'Stage', 'vlow', 'app', val_type=float)
    unit = ConfigParserProperty('mm/s',
                    'Stage', 'speed_unit', 'app', val_type=str)
    # stage coordinates and current image
    texture = ObjectProperty(None, force_dispatch=True, rebind=True)
    lastframe = ObjectProperty(None, force_dispatch=True, rebind=True)
    image = ObjectProperty(None, force_dispatch=True, rebind=True)
    coords = ListProperty([0, 0, 0])
    timestamp = NumericProperty(0, force_dispatch=True, rebind=True)
    frameBuffer = list()
    #

    def __init__(self,  **kwargs):
        super(MacroscopeApp, self).__init__(**kwargs)
        # define settings menu style
        self.settings_cls = SettingsWithSidebar
        # bind key presses to stage motion - right now also happens in settings!
        self.bind_keys()
        # hardware
        self.camera = None
        self.stage: Stage = Stage(None)

    def build(self):
        layout = Builder.load_file('layout.kv')
        # connect x-close button to action
        Window.bind(on_request_close=self.on_request_close)

        # manage xbox input
        Window.bind(on_joy_axis= self.on_controller_input)
        self.stopevent = Clock.create_trigger(lambda dt: self.stage.stop(), 0.1)
        # load some stuff
        # other useful features
        pixelsize = self.config.getfloat('Camera', 'pixelsize')
        rotation = self.config.getfloat('Camera', 'rotation')
        self.calibration_matrix = macro.genCalibrationMatrix(pixelsize, rotation)
        return layout

    #
    def build_config(self, config):
        """
        Set the default values for the configs sections.
        """
        config.read('macroscope.ini')
        #config.setdefaults('Stage', {'speed': 50, 'speed_unit': 'um/s', 'stage_limit_x':155})
        #config.setdefaults('Experiment', {'exppath':155})


    # use custom settings for our GUI
    def build_settings(self, settings):
        """build the settings window"""
        settings.add_json_panel('Macroscope GUI', self.config, 'settings/gui_settings.json')
        settings.add_json_panel('Experiment', self.config, 'settings/experiment_settings.json')


    def create_settings(self):
        '''Create the settings panel. This method will normally
        be called only one time per
        application life-time and the result is cached internally,
        but it may be called again if the cached panel is removed
        by :meth:`destroy_settings`.

        By default, it will build a settings panel according to
        :attr:`settings_cls`, call :meth:`build_settings`, add a Kivy panel if
        :attr:`use_kivy_settings` is True, and bind to
        on_close/on_config_change.

        If you want to plug your own way of doing settings, without the Kivy
        panel or close/config change events, this is the method you want to
        overload.

        .. versionadded:: 1.8.0
        '''

        self.config.read('macroscope.ini')
        s = self.settings_cls()
        self.build_settings(s)
        self.unbind_keys()
        #if self.use_kivy_settings:
        #    s.add_kivy_panel()
        s.bind(on_close=self.close__destroy_settings,
               on_config_change=self._on_config_change)
        return s


    def close__destroy_settings(self, *largs):
        '''Close the previously opened settings panel.

        :return:
            True if the settings has been closed.
        '''
        self.close_settings()
        self.bind_keys()
        self.destroy_settings()


    def stage_stop(self):
        """stop all axes and report coordinates."""
        self.stage.stop()
        self.coords = self.stage.get_position()
        self.stopevent = None
        print('stopped')


    def on_controller_input(self, win, stickid, axisid, value) -> None:
        """Handle controller input from Kivi App"""

        print(win, stickid, axisid, value)

        if self.stage is None or self.stage.is_busy():
            return

        if self.stopevent is not None:
            Clock.unschedule(self.stopevent)
            
        #scale velocity
        v = self.vhigh*value/32767
        if v < self.vlow*0.01:
            self.stage_stop()
        else:
            direction = {
                0: (v,0,0),
                1: (0,v,0),
                4: (0,0,v)
            }
            if axisid in [0,1,4]:
                self.stopevent = Clock.schedule_once(lambda dt: self.stage_stop(), 0.1)
                self.stage.start_move(direction[axisid], self.unit)
                    
            

    
    def _keydown(self, instance, key, scancode, codepoint, modifier) -> None:
        """Manage keyboard input for stage and focus"""
        
        if self.stage is None:
            return

        print(key, scancode, codepoint, modifier)

        # use arrow key codes here. This might be OS dependent.
        if 'shift' in modifier:
            v = self.vlow
        else:
            v = self.vhigh
        
        # TODO: Move this direction keymap into a setting file
        direction = {
            273: (0,-v,0),
            274: (0,v,0),
            275: (-v,0,0),
            276: (v,0,0),
            280: (0,0,-v),
            281: (0,0,v)
        }
        
        if key not in direction.keys():
            return
        
        self.stage.start_move(direction[key], self.unit)


    def _keyup(self, instance, key, scancode) -> None:
        """Handle keyup callbacks. This is usually only for stopping axis movement"""
        if self.stage is None:
            return
        
        # Movement key up
        #   Call the coresponding axis to stop and update te stage position
        if key == 275 or key == 276:
            self.stage.stop(stopAxis= AxisEnum.X)
            self.coords = self.stage.get_position()

        elif key == 273 or key == 274:
            self.stage.stop(stopAxis= AxisEnum.Y)
            self.coords = self.stage.get_position()

        elif key == 280 or key == 281:
            self.stage.stop(stopAxis= AxisEnum.Z)
            self.coords = self.stage.get_position()

        print(f'Stage position: {self.coords}')


    def unbind_keys(self):
        #unbind keyboard events
        Window.unbind(on_key_up=self._keyup)
        Window.unbind(on_key_down=self._keydown)


    def bind_keys(self):
        Window.bind(on_key_up=self._keyup)
        Window.bind(on_key_down=self._keydown)


    def toggle_key_binding(self, focus):
        if focus:
            self.unbind_keys()
        else:
            self.bind_keys()


    def on_config_change(self, config, section, key, value):
        """if config changes, update certain things."""
        if config is self.config:
            print('changed config!', section, key)
            token = (section, key)
            if token == ('Camera', 'pixelsize') or token == ('Camera', 'rotation'):
                print('updated calibration matrix')
                pixelsize = self.config.getfloat('Camera', 'pixelsize')
                rotation= self.config.getfloat('Camera', 'rotation')
                self.calibration_matrix = macro.genCalibrationMatrix(pixelsize, rotation)
            if token == ('Experiment', 'exppath'):
                self.root.ids.leftcolumn.ids.saveloc.text = value


    def on_image(self, instance, value):
        """update GUI texture when image changes."""
        if self.texture is None:
            self.create_texture(*self.image.shape)
        elif self.image.shape[::-1] != self.texture.size:
            self.create_texture(*self.image.shape)
        self.im_to_texture()


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
        # disconnect hardware
        # stop remaining stage motion
        if self.stage is not None:
            self.stage.stop()
            self.stage.disconnect()
        if self.camera is not None:
            print('disconnecting')
            self.camera.Close()
        # stop the app
        self.stop()
        # close the window
        self.root_window.close()


    def create_texture(self, w, h):
        """create the initial texture."""
        self.texture = Texture.create(
            size=(h, w), colorfmt="luminance"
        )


    def im_to_texture(self):
        """helper function to create kivy textures from image arrays."""
        buf = self.image.tobytes()
        self.texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
    
    
    def update_coordinates(self, dt= None, isAsync= True) -> None:
        """get the current stage position."""
        if self.stage is not None:
            self.coords = self.stage.get_position(isAsync= isAsync)



def reset():
    # Cleaner for the events in memory
    if not EventLoop.event_listeners:
        
        Window = Window.core_select_lib('window', Window.window_impl, True)
        Cache.print_usage()
        for cat in Cache._categories:
            Cache._objects[cat] = {}


if __name__ == '__main__':
    reset()
    Window.size = (1280, 800)
    Config.set('graphics', 'position', 'custom')
    Config.set('graphics', 'top', '0') 
    Config.set('graphics', 'left', '0') 
    App = MacroscopeApp()
    App.run()  # This runs the App in an endless loop until it closes. At this point it will execute the code below
