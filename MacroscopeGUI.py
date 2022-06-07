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
import Zaber_control as stg
import Macroscope_macros as macro
import Basler_control as basler
from skimage.io import imsave
#import cv2
#from pypylon import pylon
#from pypylon import genicam

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
            app.config.write()
            app.calibration_matrix = macro.genCalibrationMatrix(pxsize, rotation)
            #update labels shown
            self._popup.content.ids.pxsize.text = f"Pixelsize ({app.config.get('Calibration', 'step_units')}/px)  {app.config.getfloat('Camera', 'pixelsize'):.2f}"
            self._popup.content.ids.rotation.text = f"Rotation (rad)  {app.config.getfloat('Camera', 'rotation'):.3f}"


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
    recordbutton = ObjectProperty(None)
    liveviewbutton = ObjectProperty(None)
    snapbutton = ObjectProperty(None)

    def __init__(self,  **kwargs):
        super(RecordButtons, self).__init__(**kwargs)

    def snap(self):
        """single image saved to experiment location."""
        app = App.get_running_app()
        ext = app.config.get('Experiment', 'extension')
        path = app.root.ids.leftcolumn.savefile
        snap_filename = timeStamped("snap."+f"{ext}")
        # if we have a camera this will save a single take
        if app.camera is not None:
            self.liveviewbutton.state = 'normal'
            # get image
            ret, im = basler.single_take(app.camera)
            if ret:
                basler.save_image(im,path,snap_filename)


    def startPreview(self):
        camera = App.get_running_app().camera
        if camera is not None:
            # create a texture
            App.get_running_app().create_texture(*basler.get_shape(camera))
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
        
        if camera is not None:
            Clock.unschedule(self.event)
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
            ret, img, timestamp = basler.retrieve_result(camera)
            if ret:
                print('dt: ', timestamp-app.timestamp)
                cropY, cropX = self.parent.cropY, self.parent.cropX
                
                app.lastframe = img[cropY:img.shape[0]-cropY, cropX:img.shape[1]-cropX]
                app.timestamp = timestamp
                self.parent.framecounter.value += 1
        return


    def display(self, dt, record = False):
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
            # schedule immediately
            Clock.schedule_once(self.open_file, 0)
            # schedule buffer update
            self.buffertrigger = Clock.create_trigger(self.update_buffer)
            # start thread for grabbing and saving images
            record_args = self.init_recording()
            self.recordthread = Thread(target=self.record, args = record_args, daemon = True)
            self.recordthread.start()
        else:
            self.recordbutton.state = 'normal'


    def stopRecording(self):
        camera = App.get_running_app().camera
        if camera is not None:
           
            Clock.unschedule(self.event)
            # close file a bit later
            Clock.schedule_once(lambda dt: self.coordinate_file.close(), 0.5)
            if self.recordthread.is_alive():
                self.recordthread.join()
            basler.stop_grabbing(camera)
        self.parent.framecounter.value = 0
        self.buffertrigger()
        print("Finished recording")
        # reset scale of image
        App.get_running_app().root.ids.middlecolumn.ids.scalableimage.reset()
        self.recordbutton.state = 'normal'


    def init_recording(self):
        app = App.get_running_app()
        camera = app.camera
        # create a texture
        App.get_running_app().create_texture(*basler.get_shape(camera))
        # set up grabbing with recording settings here
        fps = app.config.getfloat('Experiment', 'framerate')
        nframes = app.config.getint('Experiment', 'nframes')
        buffersize = app.config.getint('Experiment', 'buffersize')
        # get cropping
        cropY, cropX = self.parent.cropY, self.parent.cropX

        print("Desired recording Framerate:", fps)
        # set recording framerate - returns
        fps = basler.set_framerate(app.camera, fps)
        print('Actual recording fps: ' + str(fps))
        app.root.ids.leftcolumn.update_settings_display()

        # set filename dummy
        # precalculate the filename
        ext = app.config.get('Experiment', 'extension')
        self.image_filename = timeStamped("basler_{}."+f"{ext}")
        return nframes, buffersize, cropX, cropY


    def record(self, nframes, buffersize, cropX, cropY):
        app = App.get_running_app()
        camera = app.camera
        basler.start_grabbing(app.camera, numberOfImagesToGrab=nframes, record=True, buffersize=buffersize)

        # schedule a display update
        fps = app.config.getfloat('Camera', 'display_fps')
        self.event = Clock.schedule_interval(self.display, 1.0 /fps)
        counter = 0
        # grab and write images
        while camera is not None and counter <nframes and self.recordbutton.state == 'down':# and camera.GetGrabResultWaitObject().Wait(0):
            # get image
            ret, img, timestamp = basler.retrieve_result(camera)
            # trigger a buffer update
            self.buffertrigger()
            if ret:
                print('dt: ', timestamp-app.timestamp)
                print('(x,y,z): ', app.coords)
                
                app.lastframe = img[cropY:img.shape[0]-cropY, cropX:img.shape[1]-cropX]
                # write coordinates
                self.coordinate_file.write(f"{self.parent.framecounter.value} {timestamp} {app.coords[0]} {app.coords[1]} {app.coords[2]} \n")
                # write image in thread
                t = Thread(target=self.save,args=(app.lastframe, self.path, self.image_filename.format(self.parent.framecounter.value)), daemon = True).start()
                # update time and frame counter
                app.timestamp = timestamp
                self.parent.framecounter.value += 1
                counter += 1
        print(f'Finished recordings {counter} frames.')
        self.buffertrigger()
        self.recordbutton.state = 'normal'
        
        return 

    def save(self, img, path, file_name):
        # save image to file
        basler.save_image(img, path, file_name)
        

    def open_file(self, *args):
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
        # offset of click from center of image
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
            if rtc.trackingcheckbox.state == 'down' and rtc.trackingevent is None:
                #
                self.captureCircle(touch.pos)
                Clock.schedule_once(lambda dt: rtc.center_image(), 0.5)
                # remove the circle 
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
        self.trackingevent = None


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


    def startTracking(self):
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
        # unschedule a tracking routine
        if self.trackingevent:
            Clock.unschedule(self.trackingevent)
            self.trackingevent = None
            # reset camera params
            camera = App.get_running_app().camera
            basler.cam_resetROI(camera)
            self.cropX = None
            self.cropY = None


    def center_image(self):
        app = App.get_running_app()
        # schedule a tracking routine
        camera = app.camera
        stage = app.stage
        # smaller FOV for the worm
        roiX, roiY  = app.config.getint('Tracking', 'roi_x'), app.config.getint('Tracking', 'roi_y')
        # move stage based on user input - happens here.
        #print(self.parent.parent.ids.previewimage.offset)
        ystep, xstep = macro.getStageDistances(app.root.ids.middlecolumn.previewimage.offset, app.calibration_matrix)
        #print("Move stage (x,y)", xstep, ystep)
        stage.move_x(xstep, unit='um', wait_until_idle=False)
        stage.move_y(ystep, unit='um', wait_until_idle=False)
        # reset camera field of view to smaller size around center
        hc, wc = basler.cam_setROI(camera, roiX, roiY, center = True)
        # if desired FOV is smaller than allowed by camera, crop in GUI
        if wc > roiX:
            self.cropX = int((wc-roiX)//2)
        if hc > roiY:
            self.cropY = int((hc-roiY)//2)
        # schedule the tracker
        focus_fps = app.config.getfloat('Livefocus', 'focus_fps')
        app.coords =  app.stage.get_position()
        self.trackingevent = Clock.schedule_interval(self.tracking, 1.0 / focus_fps)


    def tracking(self,*args):
        # execute actual tracking code
        app = App.get_running_app()
        stage = app.stage
        img = app.image
        # threshold and find objects
        coords = macro.extractWorms(img, area=50, bin_factor=2, li_init=10)
        # if we find stuff move
        if len(coords) > 0:
            print(len(coords))
            offset = macro.getDistanceToCenter(coords, img.shape)
            ystep, xstep = macro.getStageDistances(offset, app.calibration_matrix)
            stage.move_x(xstep, unit='um', wait_until_idle = False)
            stage.move_y(ystep, unit='um', wait_until_idle = False)
            print("Move stage (x,y)", xstep, ystep)
            # getting stage coord is slow so we will interpolate from movements
            app.coords[0] += xstep/1000.
            app.coords[1] += ystep/1000.
            #print(app.coords)


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
        stage = stg.Stage(port)
        if stage.connection is None:
            self.stage_connection.state = 'normal'
            App.get_running_app().stage = None
        else:
            app.stage = stage
            homing = app.config.getboolean('Stage', 'homing')
            startloc = [float(x) for x in app.config.get('Stage', 'start_loc').split(',')]
            limits = [float(x) for x in app.config.get('Stage', 'stage_limits').split(',')]
            # home stage - do this in a thread, it is slow
            t = Thread(target=app.stage.on_connect, args = (homing,  startloc, limits, app.coords))
            # set daemon to true so the thread dies when app is closed
            t.daemon = True
            # start the thread
            t.start()
            self.coordinate_update = Clock.create_trigger(app.update_coordinates)
            self.coordinate_update()
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
    unit = ConfigParserProperty('mms',
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
        self.stage = stg.Stage(None)

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
        #config.setdefaults('Stage', {'speed': 50, 'speed_unit': 'ums', 'stage_limit_x':155})
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


    def on_controller_input(self, win, stickid, axisid, value):
        print(win, stickid, axisid, value)
        if  self.stage.is_busy():
            return
        if self.stage is not None:
            if self.stopevent is not None:
               Clock.unschedule(self.stopevent)
            #scale velocity
            v = self.vhigh*value/32767
            if v < self.vlow*0.25:
                self.stage.stop()
                self.coords = self.stage.get_position()
            else:
                direction = {0: (v,0,0),
                            1: (0,v,0),
                            4: (0,0,v)
                }
                if axisid in [0,1,4]:
                    self.stage.move_speed(direction[axisid], self.unit)
                    self.stopevent = Clock.schedule_once(lambda dt: self.stage.stop(), 0.1)
            

    # manage keyboard input for stage and focus
    def _keydown(self,  instance, key, scancode, codepoint, modifier):
        # use arrow key codes here. This might be OS dependent.
        print(key, scancode, codepoint, modifier)
        if 'shift' in modifier:
            v = self.vlow
        else:
            v = self.vhigh
        direction = {273: (0,-v,0),
                    274: (0,v,0),
                    275: (-v,0,0),
                    276: (v,0,0),
                    280: (0,0,-v),
                    281: (0,0,v)
        }
        if self.stage is not None:
            self.stage.move_speed(direction[key], self.unit)


    def _keyup(self, *args):
        if self.stage is not None:
            self.stage.stop()
            self.coords = self.stage.get_position()


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
        #event = Clock.create_trigger(self.im_to_texture())
        #event()
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
    
    
    def update_coordinates(self, dt):
        """get the current stage position."""
        if self.stage is not None:
            self.coords = self.stage.get_position()



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
    App = MacroscopeApp()
    App.run()  # This runs the App in an endless loop until it closes. At this point it will execute the code below
