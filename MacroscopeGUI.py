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
from kivy.graphics import Color, Ellipse, Line
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar
from kivy.factory import Factory
from kivy.uix.textinput import TextInput
from kivy.uix.slider import Slider
from kivy.properties import ObjectProperty, StringProperty, BoundedNumericProperty, NumericProperty, ConfigParserProperty, ListProperty
from kivy.clock import Clock
from threading import Thread
from functools import partial
import os
import time
import Zaber_control as stg
import Macroscope_macros as macro
import Basler_control as basler

import globals
# TODO: separate stage from camera in the programs
# TODO: Structure the GUI such that it shares variables across GUI components
# helper function

def im_to_texture(image):
    """helper function to create kivy textures from image arrays."""
    buf = image.tobytes()
    w,h = image.shape
    image_texture = Texture.create(
        size=(h,w), colorfmt="luminance"
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
        app = App.get_running_app()
        Window.bind(on_key_up=app._keyup)
        Window.bind(on_key_down=app._keydown)
        self._popup.dismiss()
    # popup camera file selector
    def show_load(self):
        content = LoadCameraProperties(load=self.load, cancel=self.dismiss_popup)
        content.ids.filechooser2.path = self.loadfile
        self._popup = Popup(title="Load camera file", content=content,
                            size_hint=(0.9, 0.9))
        #unbind keyboard events
        app = App.get_running_app()
        Window.unbind(on_key_up=app._keyup)
        Window.unbind(on_key_down=app._keydown)
        self._popup.open()
    # popup experiment dialog selector
    def show_save(self):
        content = SaveExperiment(save=self.save, cancel=self.dismiss_popup)
        content.ids.filechooser.path = self.savefile
        self._popup = Popup(title="Select save location", content=content,
                            size_hint=(0.9, 0.9))
        #unbind keyboard events
        app = App.get_running_app()
        Window.unbind(on_key_up=app._keyup)
        Window.unbind(on_key_down=app._keydown)
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
    
    #autofocus popup
    def show_autofocus(self):
        content = AutoFocus(run_autofocus = self.run_autofocus, cancel=self.dismiss_popup)
        self._popup = Popup(title="Focus the camera", content=content,
                            size_hint=(0.9, 0.9))
        #unbind keyboard events
        app = App.get_running_app()
        Window.unbind(on_key_up=app._keyup)
        Window.unbind(on_key_down=app._keydown)
        self._popup.open()
    
    # run autofocussing once on current location
    def run_autofocus(self):
        app = App.get_running_app()
        
        camera = app.camera
        stage = app.stage
        
        if camera is not None and stage.connection is not None:
            # stop grabbing
            app.root.ids.middlecolumn.ids.runtimecontrols.ids.recordbuttons.ids.liveviewbutton.state = 'normal'
            # get config values
            stepsize = app.config.getfloat('Autofocus', 'step_size')
            stepunits = app.config.get('Autofocus', 'step_units')
            nsteps = app.config.getint('Autofocus', 'nsteps')
            # run the autofocus
            _, imstack, _, focal_plane = macro.zFocus(stage, camera, stepsize, stepunits, nsteps)
            # update the images shown - delete old ones if rerunning
            self._popup.content.delete_images()
            self._popup.content.add_images(imstack, nsteps, focal_plane)

            

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
        #rebind keyboard events
        app = App.get_running_app()
        Window.bind(on_key_up=app._keyup)
        Window.bind(on_key_down=app._keydown)
        self._popup.dismiss()

    def show_recording_settings(self):
        """change recording settings."""
        fps = App.get_running_app().root.ids.leftcolumn.ids.camprops.framerate.value
        content = RecordingSettings(update=self.update, cancel=self.dismiss_popup, \
                            frames = self.nframes, framerate = str(fps), fileformat = self.fileformat)
        self._popup = Popup(title="Recording Settings", content=content,
                            size_hint=(0.5, 0.25))
        #unbind keyboard events
        app = App.get_running_app()
        Window.unbind(on_key_up=app._keyup)
        Window.unbind(on_key_down=app._keydown)
        self._popup.open()
    
    def update(self, frames):
        if frames is not None:
            self.nframes = frames
        self.dismiss_popup()
    
    def show_calibration(self):
        content = AutoCalibration(calibrate=self.calibrate, cancel=self.dismiss_popup, \
                            )
        self._popup = Popup(title="Autocalibration", content=content,
                            size_hint=(0.9, 0.75))
        self._popup.open()
    
    def calibrate(self):
        app = App.get_running_app()
        camera = app.camera
        stage = app.stage
        if camera is not None and stage.connection is not None:
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
            app.calibration_matrix =  macro.genCalibrationMatrix(pxsize, rotation)
            #update labels shown
            self._popup.content.ids.pxsize.text = f"Pixelsize ({app.config.get('Calibration', 'step_units')}/px)  {app.config.getfloat('Camera', 'pixelsize'):.2f}"
            self._popup.content.ids.rotation.text = f"Rotation (rad)  {app.config.getfloat('Camera', 'rotation'):.3f}"


class AutoCalibration(BoxLayout):
    calibrate = ObjectProperty(None)
    cancel = ObjectProperty(None)

# Stage controls
class XControls(BoxLayout):
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
    cancel = ObjectProperty(None)

# save location for images and meta data
class AutoFocus(BoxLayout):
    run_autofocus = ObjectProperty(None)
    cancel = ObjectProperty(None)
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



class LabelImage(BoxLayout):
    def __init__(self,  **kwargs):
        super(LabelImage, self).__init__(**kwargs)
        self.text = ''
        self.texture = ''

class MultipleImages(GridLayout):
    pass
       

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
    liveviewbutton = ObjectProperty(None)

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
        else:
            self.liveviewbutton.state = 'normal'
            
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
        else:
            self.recordbutton.state = 'normal'

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
                # store image as class variable - this will also trigger a canvas update
                App.get_running_app().image = img
                self.parent.framecounter.value += 1

    def record(self, dt):
        camera = App.get_running_app().camera
        nframes = int(App.get_running_app().root.ids.rightcolumn.nframes)
        if camera is not None:
            if camera.IsGrabbing() and self.parent.framecounter.value < nframes:
                ret, img = basler.retrieve_result(camera)
                if ret:
                    # show on GUI
                    # store image as class variable - this will also trigger a canvas update
                    App.get_running_app().image = img
                    # save image in thread
                    t = Thread(target=self.save, args = (img,))
                    # set daemon to true so the thread dies when app is closed
                    t.daemon = True
                    # start the thread
                    t.start()
                    self.parent.framecounter.value +=  1
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
    circle= ListProperty([0, 0, 0])
    offset = ListProperty([0, 0])

    def __init__(self,  **kwargs):
        super(PreviewImage, self).__init__(**kwargs)
        Window.bind(mouse_pos=self.mouse_pos)
        
    def mouse_pos(self, window, pos):
        # read mouse hover events and get image value
        if self.collide_point(*pos):
            # by default the touch coordinates are relative to GUI window
            wx, wy = self.to_widget(pos[0], pos[1], relative = True)
            image = App.get_running_app().image
            # get the image we last took
            if image is not None:
                texture_w, texture_h = self.norm_image_size
                #offset if the image is not fitting inside the widget
                cx, cy = self.to_widget(self.center_x, self.center_y, relative = True)
                ox, oy = cx - texture_w / 2., cy - texture_h/ 2
                h,w = image.shape
                imy, imx = int((wy-oy)*h/texture_h), int((wx-ox)*w/texture_w)
                if 0<=imy<=h and 0<=imx<=w:
                    val = image[imy,imx]
                    self.parent.parent.ids.pixelvalue.text = f'({imx},{imy},{val})'

    
    def captureCircle(self, pos):
        """define the capture circle and draw it."""
        wx, wy = self.to_widget(pos[0], pos[1], relative = True)
        image = App.get_running_app().image
        h,w = image.shape
        # paint a circle and make the coordinates available
        radius = App.get_running_app().config.getfloat('Tracking', 'capture_radius')
        # make the circle into pixel units
        r = radius/w*self.norm_image_size[0]#, radius/h*self.norm_image_size[1]
        self.circle = (*pos, r)
        # calculate in image units where the click was relative to image center and return that
        #offset if the image is not fitting inside the widget
        texture_w, texture_h = self.norm_image_size
        #offset if the image is not fitting inside the widget
        cx, cy = self.to_widget(self.center_x, self.center_y, relative = True)
        ox, oy = cx - texture_w / 2., cy - texture_h/ 2
        imy, imx = int((wy-oy)*h/texture_h), int((wx-ox)*w/texture_w)
        # offset of click from center of image
        self.offset = (imy-h//2, imx-w//2)
        

    def clearcircle(self):
        self.circle=(0,0,0)

    # # for reading mouse clicks
    def on_touch_down(self, touch):
        # if a click happens in this widget
        if self.collide_point(*touch.pos):
            #if tracking is active and not yet scheduled:
            if self.parent.parent.ids.runtimecontrols.trackingcheckbox.state =='down' and self.parent.parent.ids.runtimecontrols.trackingevent is None:
                #
                self.captureCircle(touch.pos)
                Clock.schedule_once(lambda dt: self.parent.parent.ids.runtimecontrols.center_image(), 0.5)

                # remove the circle 
                Clock.schedule_once(lambda dt: self.clearcircle(), 0.5)
                #self.circle = (0,0, 0)


#         # by default the touch coordinates are relative to GUI window
#         wx, wy = self.to_widget(touch.x, touch.y, relative = True)
#         image = App.get_running_app().image
#         # get the image we last took
#         print(wx, wy)
#         if image is not None:
#             texture_w, texture_h = self.norm_image_size
#             #offset if the image is not fitting inside the widget
#             ox, oy = self.to_widget(self.center_x - self.norm_image_size[0] / 2., self.center_y - self.norm_image_size[1] / 2., relative = True)
#             h,w = image.shape
#             imy, imx = int((wy-oy)*h/texture_h), int((wx-ox)*w/texture_w)
#             val = image[imy,imx]
#             self.parent.parent.ids.pixelvalue.text = f'({imx},{imy},{val})'
                
        
class RuntimeControls(BoxLayout):
    framecounter = ObjectProperty(rebind = True)
    autofocuscheckbox = ObjectProperty(rebind = True)
    trackingcheckbox = ObjectProperty(rebind = True)
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
        if camera is not None and stage.connection is not None and camera.IsGrabbing():
             # get config values
            focus_fps = App.get_running_app().config.getfloat('Livefocus', 'focus_fps')
            print("Focus Framerate:", focus_fps)
            z_step = App.get_running_app().config.getfloat('Livefocus', 'min_step')
            unit = App.get_running_app().config.get('Livefocus', 'step_units')
            factor = App.get_running_app().config.getfloat('Livefocus', 'factor')
            
            self.focusevent = Clock.schedule_interval(partial(self.focus,  z_step, unit, factor), 1.0 / focus_fps)
        else:
            self._popup = WarningPopup(title="Autofocus", text = 'Focus requires: \n - a stage \n - a camera \n - camera needs to be grabbing.',
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
        if len(self.focus_history)>1 and self.focus_motion !=0 :
            self.focus_motion = macro.calculate_focus_move(self.focus_motion, self.focus_history, z_step, focus_step_factor)
        else:
            self.focus_motion = z_step
        print('Move',self.focus_motion,unit)
        app.stage.move_z(self.focus_motion, unit)
        print('Move',self.focus_motion,unit)
        # throw away stuff
        self.focus_history = self.focus_history[-1:]
        
        print('Saving time: ',time.time() - start)
        return 

    def startTracking(self):
        app = App.get_running_app()
        # schedule a tracking routine
        camera = app.camera
        stage = app.stage

        if camera is not None:# and stage.connection is not None and camera.IsGrabbing():
             # get config values
            
            # find an animal and center it once by moving the stage
            self._popup = WarningPopup(title="Click on animal", text = 'Click on an animal to start tracking it.',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            # make a capture circle - all of this happens in Image Widget, and record offset from center, then dispatch the centering routine
            #schedule a tracking loop
        else:
            self._popup = WarningPopup(title="Tracking", text = 'Tracking requires a stage, a camera and the camera needs to be grabbing.',
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
    
    def center_image(self):
        app = App.get_running_app()
        # schedule a tracking routine
        camera = app.camera
        stage = app.stage
        # smaller FOV for the worm
        roiX, roiY  = app.config.getint('Tracking', 'roi_x'), app.config.getint('Tracking', 'roi_y')
        # move stage based on user input - happens here.
        print(self.parent.parent.ids.previewimage.offset)
        ystep, xstep = macro.getStageDistances(self.parent.parent.ids.previewimage.offset, app.calibration_matrix)
        print("Move stage (x,y)", xstep, ystep)
        stage.move_x(xstep, unit = 'um', wait_until_idle = False)
        stage.move_y(ystep, unit = 'um', wait_until_idle = False)
        # reset camera field of view to smaller size around center
        basler.cam_setROI(camera, roiX,roiY, center = True)
        # schedule the tracker
        focus_fps = app.config.getfloat('Livefocus', 'focus_fps')
        self.trackingevent = Clock.schedule_interval(self.tracking, 1.0 / focus_fps)

    def tracking(self, dt):
        # execute actual tracking code
        "we are tracking."
        pass
    
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
        if camera is not None:
            print('disconnecting')
            camera.Close()
        
    def connectStage(self):
        print('connecting Stage')
        stage = stg.Stage(port='/dev/ttyUSB0')
        
        if stage.connection is None:
            self.stage_connection.state = 'normal'
        else:
            App.get_running_app().stage = stage
            homing = App.get_running_app().config.getboolean('Stage', 'homing')
            startloc = [float(x) for x in App.get_running_app().config.get('Stage', 'start_loc').split(',')]
            limits = [float(x) for x in App.get_running_app().config.get('Stage', 'stage_limits').split(',')]
            # home stage - do this in a thread, it is slow
            t = Thread(target=App.get_running_app().stage.on_connect, args = (homing,  startloc, limits))
            # set daemon to true so the thread dies when app is closed
            t.daemon = True
            # start the thread
            t.start()
        
    def disconnectStage(self):
        print('disconnecting Stage')
        if App.get_running_app().stage.connection is None:
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
    # stage configuration properties - these will update when changed in config menu
    vhigh = ConfigParserProperty(20, 
                    'Stage', 'vhigh', 'app', val_type = float)
    vlow = ConfigParserProperty(20, 
                    'Stage', 'vlow', 'app', val_type = float)
    unit = ConfigParserProperty('mms', 
                    'Stage', 'speed_unit', 'app', val_type = str)
    # stage coordinates and current image
    texture = ObjectProperty(None, force_dispatch = True, rebind = True)
    image = ObjectProperty(None, force_dispatch = True, rebind = True)
    coords = ObjectProperty(None)
    #
    
    
    def __init__(self,  **kwargs):
        super(MacroscopeApp, self).__init__(**kwargs)
        # define settings menu style
        self.settings_cls = SettingsWithSidebar
        # bind key presses to stage motion - right now also happens in settings!
        Window.bind(on_key_up=self._keyup)
        Window.bind(on_key_down=self._keydown)
        
        # hardware
        self.camera = None
        self.stage = stg.Stage(None)
        
    
    
    def build(self):
        layout = Builder.load_file('layout.kv')
        # connect x-close button to action
        Window.bind(on_request_close=self.on_request_close)
        # load some stuff
        # other useful features
        pixelsize = self.config.getfloat('Camera', 'pixelsize')
        rotation= self.config.getfloat('Camera', 'rotation')
        self.calibration_matrix = macro.genCalibrationMatrix(pixelsize, rotation)
        return layout
    # 
    def build_config(self, config):
        """
        Set the default values for the configs sections.
        """
        config.read('macroscope.ini')
        #config.setdefaults('Stage', {'speed': 50, 'speed_unit': 'ums', 'stage_limit_x':155})

    # use custom settings for our GUI
    def build_settings(self, settings):
        """build the settings window"""
        settings.add_json_panel('Macroscope GUI', self.config, 'settings/gui_settings.json')

    # manage keyboard input for stage and focus
    def _keydown(self,  instance, key, scancode, codepoint, modifier):
        # use arrow key codes here. This might be OS dependenot.
        # left arrow - x axis
        if key == 276:
            if 'shift' in modifier:
                self.stage.move_speed((-self.vlow,0,0), self.unit)
            else: self.stage.move_speed((-self.vhigh,0,0), self.unit)
        #right arrow - x-axis
        if key == 275:
            if 'shift' in modifier:
                self.stage.move_speed((self.vlow,0,0), self.unit)
            else: self.stage.move_speed((self.vhigh,0,0), self.unit)
        # up and down arrow are y stage
        if key == 273:
            if 'shift' in modifier:
                self.stage.move_speed((0,-self.vlow,0), self.unit)
            else: self.stage.move_speed((0,-self.vhigh,0), self.unit)
        if key == 274:
            if 'shift' in modifier:
                self.stage.move_speed((0,self.vlow,0), self.unit)
            else: self.stage.move_speed((0,self.vhigh,0), self.unit)
        #focus keys -pg up and down for z
        print(key, scancode, codepoint, modifier)
        if key == 280:
            if 'shift' in modifier:
                self.stage.move_speed((0,0,-self.vlow), self.unit)
            else: self.stage.move_speed((0,0,-self.vhigh), self.unit)
        if key == 281:
            if 'shift' in modifier:
                self.stage.move_speed((0,0,self.vlow), self.unit)
            else: self.stage.move_speed((0,0,self.vhigh), self.unit)

    def _keyup(self, *args):
        self.stage.stop()
    
    def on_config_change(self, config, section, key, value):
        """if config changes, update certain things."""
        if config is self.config:
            token = (section, key)
            if token == ('Camera', 'pixelsize') or token == ('Camera', 'rotation'):
                print('updated matrix')
                pixelsize = self.config.getfloat('Camera', 'pixelsize')
                rotation= self.config.getfloat('Camera', 'rotation')
                self.calibration_matrix =  macro.genCalibrationMatrix(pixelsize, rotation)


    def on_image(self, instance, value):
        """update GUI texture when image changes."""
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
        self.stage.stop()
        self.stage.disconnect()
        if self.camera is not None:
            print('disconnecting')
            self.camera.Close()
        # stop the app
        self.stop()
    
    def im_to_texture(self):
        """helper function to create kivy textures from image arrays."""
        buf = self.image.tobytes()
        w,h = self.image.shape
        image_texture = Texture.create(
            size=(h,w), colorfmt="luminance"
        )
        image_texture.blit_buffer(buf, colorfmt="luminance", bufferfmt="ubyte")
        self.texture = image_texture

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
