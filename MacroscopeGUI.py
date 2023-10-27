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
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Line
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter
from kivy.uix.tabbedpanel import TabbedPanel
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
from kivy.clock import Clock, ClockEvent
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
import cv2
#from pypylon import pylon
#from pypylon import genicam
from typing import Tuple
from multiprocessing.pool import ThreadPool
from queue import Queue
import math
from overrides import override
from io import TextIOWrapper
# get the free clock (more accurate timing)
#Config.set('graphics', 'KIVY_CLOCK', 'free')
Config.set('modules', 'monitor', '')
# helper functions
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-{fname}'):
    """This creates a timestamped filename so we don't overwrite our good work."""
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def imageToTexture(image: np.ndarray) -> Texture:
    """A helper function to create kivy textures from numpy arrays.

    Args:
        image (np.ndarray): image

    Returns:
        texture (Texture): image as a Kivy Texture
    """    
    height, width = image.shape[0], image.shape[1]
    colorfmt = 'luminance' if image.ndim == 2 else 'rgb'
    bufferfmt = 'ubyte'
    
    # Create a new Kivy Texture
    image_texture = Texture.create(
        size=(width, height), colorfmt= colorfmt, bufferfmt= bufferfmt
    )

    # Kivy texture is in OpenGL corrindate which is btm-left origin so we need to flip texture coord once to match numpy's top-left
    image_texture.flip_vertical()
    
    # Upload data to texture
    buf = image.tobytes()
    image_texture.blit_buffer(buf, colorfmt= colorfmt, bufferfmt= bufferfmt)

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
            app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionbuttonlayout.ids.liveviewbutton.state = 'normal'
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
    

    def open_settings(self):
        # Disabled interaction with preview image widget
        App.root.ids.middlecolumn.ids.scalableimage.disabled = True
        # Call open settings
        App.open_settings()


    def show_recording_settings(self):
        """change recording settings."""
        content = RecordingSettings(ok=self.dismiss_popup)
        self._popup = Popup(title="Recording Settings", content=content,
                            size_hint=(0.5, 0.35))
        #unbind keyboard events
        App.get_running_app().unbind_keys()
        self._popup.open()


    def show_calibration(self):
        app: MacroscopeApp = App.get_running_app()
        camera: pylon.InstantCamera = app.camera
        stage: Stage = app.stage

        if camera is not None and stage is not None:
            calibrationTabPanel = CalibrationTabPanel()
            calibrationTabPanel.setCloseCallback(closeCallback= self.dismiss_popup)

            self._popup = Popup(title= '', separator_height= 0, content= calibrationTabPanel, size_hint= (0.9, 0.75))
            self._popup.open()
        
        else:
            self._popup = WarningPopup(title="Calibration", text='Autocalibration requires a stage and a camera. Connect a stage or use a calibration slide.',
                            size_hint=(0.5, 0.25))
            self._popup.open()


class CalibrationTabPanel(TabbedPanel):
    
    def setCloseCallback(self, closeCallback: callable) -> None:
        self.closeCallback = closeCallback
        self.ids.stagecalibration.setCloseCallback( closeCallback )
        self.ids.dualcolorcalibration.setCloseCallback( closeCallback )
    

class CameraAndStageCalibration(BoxLayout):

    closeCallback = ObjectProperty(None)
    
    def setCloseCallback( self, closeCallback: callable ) -> None:
        self.closeCallback = closeCallback
    

    def calibrate(self):
        app: MacroscopeApp = App.get_running_app()
        camera: pylon.InstantCamera = app.camera
        stage: Stage = app.stage

        if camera is None or stage is None:
            return
        
        # stop camera if already running
        liveViewButton: Button = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionbuttonlayout.ids.liveviewbutton
        prevLiveViewButtonState = liveViewButton.state
        liveViewButton.state = 'normal'
        
        # get config values
        stepsize = app.config.getfloat('Calibration', 'step_size')
        stepunits = app.config.get('Calibration', 'step_units')
        dualcolormode = app.config.get('DualColor', 'dualcolormode')
        mainside = app.config.get('DualColor', 'mainside')

        # Take calibration images
        cameraAndStageCalibrator = macro.CameraAndStageCalibrator()
        basisImageOrig, basisImageX, basisImageY = cameraAndStageCalibrator.takeCalibrationImage(
            camera,
            stage,
            stepsize,
            stepunits,
            dualcolormode,
            mainside 
        )
        
        # Update display calibration images
        self.ids.fixedimage.texture = imageToTexture(basisImageOrig)
        self.ids.movingimagex.texture = imageToTexture(basisImageX)
        self.ids.movingimagey.texture = imageToTexture(basisImageY)
            
        # Estimate camera to stage transformation parameters
        rotation, imageNormDir, pixelSize = cameraAndStageCalibrator.calibrateCameraAndStageTransform()
        app.config.set('Camera', 'rotation', rotation)
        app.config.set('Camera', 'imagenormaldir', '+Z' if imageNormDir == 1 else '-Z')
        app.config.set('Camera', 'pixelsize', pixelSize)
        
        # update calibration matrix
        app.imageToStageMat, app.imageToStageRotMat = macro.CameraAndStageCalibrator.genImageToStageMatrix(rotation, imageNormDir, pixelSize)

        # update labels shown
        self.ids.pxsize.text = f'Pixelsize ({stepunits}/px)  {pixelSize:.2f}'
        self.ids.rotation.text = f'Rotation (rad)  {rotation:.3f}'

        # save configs
        app.config.write()

        # Show axis figure
        stageToImageRotMat = np.linalg.inv(app.imageToStageRotMat)
        plotImage = macro.renderBasisImage(macro.swapMatXYOrder(stageToImageRotMat))
        self.ids.cameraandstageaxes.texture = imageToTexture(plotImage)

        # Resume the camera to previous state
        liveViewButton.state = prevLiveViewButtonState


class DualColorCalibration(BoxLayout):

    closeCallback = ObjectProperty(None)
    
    def setCloseCallback( self, closeCallback: callable ) -> None:
        self.closeCallback = closeCallback
    

    def calibrate(self) -> None:
        app: MacroscopeApp = App.get_running_app()
        camera: pylon.InstantCamera = app.camera
        stage: Stage = app.stage

        if camera is None or stage is None:
            return
        
        # stop camera if already running
        liveViewButton: Button = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionbuttonlayout.ids.liveviewbutton
        prevLiveViewButtonState = liveViewButton.state
        liveViewButton.state = 'normal'
        
        # Take a dual color image for calibration
        isSuccess, dualColorImage = basler.single_take(camera)

        if not isSuccess:
            return

        mainSide = app.config.get('DualColor', 'mainside')
        
        # Instantiate a dual color calibrator
        dualColorImageCalibrator = macro.DualColorImageCalibrator()

        # Process the dual color image
        mainSideImage, minorSideImage = dualColorImageCalibrator.processDualColorImage(
            dualColorImage= dualColorImage,
            mainSide= mainSide,
            cropWidth= 700,
            cropHeight= 700
        )
        
        # Update display image
        self.ids.mainsideimage.texture = imageToTexture(mainSideImage)
        self.ids.minorsideimage.texture = imageToTexture(minorSideImage)

        # Calibrate the transformation
        translation_x, translation_y, rotation = dualColorImageCalibrator.calibrateMinorToMainTransformationMatrix()

        # Save settines to config
        app.config.set('DualColor', 'translation_x', translation_x)
        app.config.set('DualColor', 'translation_y', translation_y)
        app.config.set('DualColor', 'rotation', rotation)
        app.config.write()

        # Update labels shown
        self.ids.translation.text = f"Translation (x,y): {translation_x:.2f}, {translation_y:.2f}"
        self.ids.rotation.text = f"Rotation (rad): {rotation:.3f}"
        
        # Compute minor to main calibration matrix
        minorToMainMat = dualColorImageCalibrator.genMinorToMainMatrix(translation_x, translation_y, rotation, mainSideImage.shape[1]/2, mainSideImage.shape[0]/2)

        # Apply transformation
        #   Use warpAffine() here instead of warpPerspective for a little faster computation.
        #   It also takes 2x3 mat, so we cut the last row out accordingly.
        translatedMinorSideImage = cv2.warpAffine(minorSideImage, minorToMainMat[:2,:], (minorSideImage.shape[1], minorSideImage.shape[0]))

        # Combine main and minor side
        combinedImage = np.zeros(shape= (mainSideImage.shape[1], mainSideImage.shape[0], 3), dtype= np.uint8)
        combinedImage[:,:,0] = mainSideImage
        combinedImage[:,:,1] = translatedMinorSideImage

        # Update the composite display image
        self.ids.calibratedimage.texture = imageToTexture(combinedImage)

        # Resume the camera to previous state
        liveViewButton.state = prevLiveViewButtonState


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
            tmp.image.texture = imageToTexture(imstack[i])
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


class ImageAcquisitionButton(ToggleButton):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app: MacroscopeApp | None = None
        self.camera: pylon.InstantCamera | None = None
        self.imageAcquisitionThread: Thread | None = None
        self.runtimeControls: RuntimeControls | None = None
        self.updateDisplayImageEvent: ClockEvent | None = None

    
    def on_state(self, widget, value):
        # On state change callback
        if value == 'down':
            self.startImageAcuisition()
        else:
            self.stopImageAcquisition()

    
    def startImageAcuisition(self) -> None:
        pass


    def stopImageAcquisition(self) -> None:
        pass
    

    def imageAcquisitionLoopingThread(self, grabArgs) -> None:
        app: MacroscopeApp = App.get_running_app()
        camera: pylon.InstantCamera = app.camera

        # Start grabbing images
        camera.MaxNumBuffer = grabArgs.bufferSize
        
        if grabArgs.numberOfImagesToGrab == -1:
            # Endless grabbing
            camera.StartGrabbing(grabArgs.grabStrategy)
        
        else:
            # Grab for a specific number of frames
            camera.StartGrabbingMax(grabArgs.numberOfImagesToGrab, grabArgs.grabStrategy)
            
        fps = camera.ResultingFrameRate()
        print("Grabbing Framerate:", fps)

        # Schedule a display update
        fps = app.config.getfloat('Camera', 'display_fps')
        self.updateDisplayImageEvent = Clock.schedule_interval(self.updateDisplayImage, 1.0 /fps)
        print(f'Displaying at {fps} fps')

        # Start image acquisition loop
        while self.acquisitionCondition():

            # retrieve an image
            isSuccess, img, timestamp, retrieveTimestamp = basler.retrieve_grabbing_result(camera)

            if isSuccess:
                # Trigger image callback
                self.receiveImageCallback(img, timestamp, retrieveTimestamp)
        
        self.finishAcquisitionCallback()


    def acquisitionCondition(self) -> bool:
        pass

    
    def receiveImageCallback(self, img: np.ndarray, timestamp: float, retrieveTimestamp: float) -> None:
        pass


    def finishAcquisitionCallback(self) -> None:
        pass


    def updateDisplayImage(self, dt):
        if self.app.lastframe is not None:
            self.app.image = self.app.lastframe


class LiveViewButton(ImageAcquisitionButton):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @override
    def startImageAcuisition(self) -> None:
        
        self.app: MacroscopeApp = App.get_running_app()
        self.camera: pylon.InstantCamera = self.app.camera

        if self.camera is None:
            self.state = 'normal'
            return
        
        # Setup image acquisition thread parameters
        self.runtimeControls = App.get_running_app().root.ids.middlecolumn.runtimecontrols

        grabArgs = basler.CameraGrabParameters(
            bufferSize= 16,
            numberOfImagesToGrab= -1,
            grabStrategy= pylon.GrabStrategy_LatestImageOnly
        )

        # Spawn image acquisition thread
        self.imageAcquisitionThread = Thread(
            target= self.imageAcquisitionLoopingThread,
            daemon= True,
            kwargs= {
                'grabArgs' : grabArgs
            }
        )
        self.imageAcquisitionThread.start()

        # Update image overlay
        self.app.updateDualColorOverlay()


    @override
    def stopImageAcquisition(self) -> None:

        print('stopping preview')
        
        if self.camera is None:
            return

        # Unschedule the display event thread
        Clock.unschedule(self.updateDisplayImageEvent)

        # Stop image acquisition thread
        if self.imageAcquisitionThread.is_alive():
            self.imageAcquisitionThread.join()
                
        # Stop grabbing
        basler.stop_grabbing(self.camera)

        # reset displayed framecounter
        self.runtimeControls.framecounter.value = 0
        
        # reset scale of image
        self.app.root.ids.middlecolumn.ids.scalableimage.reset()
    

    @override
    def acquisitionCondition(self) -> bool:
        return self.camera is not None and self.camera.IsGrabbing() and self.state == 'down'

    
    @override
    def receiveImageCallback(self, img: np.ndarray, timestamp: float, retrieveTimestamp: float) -> None:
        # Update app image data
        h, w = img.shape
        cropX, cropY = self.runtimeControls.cropX, self.runtimeControls.cropY
        self.app.lastframe = img[ cropY : h - cropY, cropX : w - cropX ]
        self.app.timestamp = timestamp
        self.app.retrieveTimestamp = retrieveTimestamp
        self.runtimeControls.framecounter.value += 1


class RecordButton(ImageAcquisitionButton):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.numberRecordframes: int = 0
        self.frameCounter: int = 0
        self.saveFilePath: str = ''
        self.coordinateFile: TextIOWrapper | None = None
        self.savingthread: Thread | None = None
        self.imageQueue: Queue | None = None
    
    
    @override
    def startImageAcuisition(self) -> None:
        
        self.app: MacroscopeApp = App.get_running_app()
        self.camera: pylon.InstantCamera = self.app.camera

        if self.camera is None:
            self.state = 'normal'
            return

        self.runtimeControls = App.get_running_app().root.ids.middlecolumn.runtimecontrols
        
        self.runtimeControls.framecounter.value = 0
        self.saveFilePath = self.app.root.ids.leftcolumn.savefile

        # stop camera if already running
        basler.stop_grabbing(self.camera)

        # open coordinate file
        self.coordinateFile = open(os.path.join(self.saveFilePath, timeStamped("coords.txt")), 'a')
        self.coordinateFile.write(f"Frame Time X Y Z \n")

        # Image data queue to share between recording and saving
        self.imageQueue = Queue()

        # Start a thread for saving images
        self.savingthread = Thread(target= self.imageSavingThreadManagerThread, args= [3,])
        self.savingthread.start()

        # Setup image acquisition thread parameters
        self.numberRecordframes, buffersize = self.initRecordingParams()
        self.frameCounter = 0

        grabArgs = basler.CameraGrabParameters(
            bufferSize= buffersize,
            numberOfImagesToGrab= self.numberRecordframes,
            grabStrategy= pylon.GrabStrategy_OneByOne
        )

        # Spawn image acquisition thread
        self.imageAcquisitionThread = Thread(
            target= self.imageAcquisitionLoopingThread,
            daemon= True,
            kwargs= {
                'grabArgs' : grabArgs,
            }
        )

        self.imageAcquisitionThread.start()

        # Update image overlay
        self.app.updateDualColorOverlay()


    @override
    def stopImageAcquisition(self) -> None:
        
        if self.camera is None:
            return
        
        # Unschedule the display event thread
        Clock.unschedule(self.updateDisplayImageEvent)

        # Schedule closing coordinate file a bit later
        Clock.schedule_once(lambda dt: self.coordinateFile.close(), 0.5)
        
        # Close saving threads
        self.savingthread.join()
        
        # Tell camera to stop grabbing mode
        basler.stop_grabbing(self.camera)

        # Reset frame counter
        self.runtimeControls.framecounter.value = 0

        # Update display buffer text
        self.runtimeControls.buffer.value = self.camera.MaxNumBuffer() - self.camera.NumQueuedBuffers()

        # Reset scale of display image
        self.app.root.ids.middlecolumn.ids.scalableimage.reset()

        # Set button state back to normal
        self.state = 'normal'
        
        print("Finished recording")
    

    @override
    def acquisitionCondition(self) -> bool:
        return self.camera is not None and self.frameCounter < self.numberRecordframes and self.state == 'down'

    
    @override
    def receiveImageCallback(self, img: np.ndarray, timestamp: float, retrieveTimestamp: float) -> None:

        # Update buffer display text
        self.runtimeControls.buffer.value = self.camera.MaxNumBuffer() - self.camera.NumQueuedBuffers()

        # Update app image data
        h, w = img.shape
        cropX, cropY = self.runtimeControls.cropX, self.runtimeControls.cropY
        self.app.lastframe = img[ cropY : h - cropY, cropX : w - cropX ]

        # write coordinate into file
        self.coordinateFile.write(f"{self.frameCounter} {timestamp} {self.app.coords[0]} {self.app.coords[1]} {self.app.coords[2]} \n")

        # Put image into the saving queue
        self.imageQueue.put([
            np.copy(self.app.lastframe),
            self.saveFilePath,
            self.image_filename.format(self.frameCounter)
        ])

        # update time and frame counter
        self.app.timestamp = timestamp
        self.app.retrieveTimestamp = retrieveTimestamp
        self.frameCounter += 1
        self.runtimeControls.framecounter.value = self.frameCounter
    

    @override
    def finishAcquisitionCallback(self) -> None:
        # Send signal to terminate recording workers
        self.imageQueue.put(None)

        print(f'Recorded {self.numberRecordframes} frames.')

        # Call to recording-stopping procedure
        self.stopImageAcquisition()
    

    def imageSavingThreadManagerThread(self, numMaxThreads: int | None = None) -> None:
        """An image saving thread manager that spawn a fix number of threads that iteratively consume an image from a queue and save it.

        Args:
            numMaxThreads (int | None, optional): Maximum number of small threads. Defaults to None.
        """

        def imageSavingThreadWorker(queue):

            # run until there is no more work
            while True:

                # retrieve one item from the queue
                queueItem = queue.get()

                # check for signal of no more work
                if queueItem is not None:
                    
                    img, imgPath, imgFileName = queueItem
                    # Save the image
                    imsave(os.path.join(imgPath, imgFileName), img, check_contrast=False,  plugin="tifffile")
                    
                else:
                    # put back on the queue for other consumers
                    queue.put(None)
                    # shutdown the thread
                    break
        
        
        # Create a thread pool
        consumerThreadPool = ThreadPool(processes= numMaxThreads)
        
        # Start spawn image saving workers
        for i in range(consumerThreadPool._processes):
            consumerThreadPool.apply_async(
                func= imageSavingThreadWorker, 
                args= (self.imageQueue,)
            )
        
        # Wait until all workers are done
        consumerThreadPool.close()
        consumerThreadPool.join()

        print(f'Finished saving all the images')
    

    def initRecordingParams(self) -> Tuple[ int, int ]:
        """Initialize recording parameters

        Returns:
            nframes (int): number of frames to grab
            buffersize (int): number of buffers to allocate, used for grabbing
        """
        # Setup grabbing with recording settings
        nframes = self.app.config.getint('Experiment', 'nframes')
        buffersize = self.app.config.getint('Experiment', 'buffersize')

        fps = self.app.root.ids.leftcolumn.ids.camprops.framerate
        print("Desired recording Framerate:", fps)

        fps = basler.set_framerate(self.app.camera, fps)
        print('Actual recording fps: ' + str(fps))

        # Update shown display settings, e.g. exposure, fps, gain values
        self.app.root.ids.leftcolumn.update_settings_display()

        # precalculate the filename
        ext = self.app.config.get('Experiment', 'extension')
        self.image_filename = timeStamped("basler_{}."+f"{ext}")
        return nframes, buffersize


class ImageAcquisitionButtonLayout(BoxLayout):
    recordbutton = ObjectProperty(None, rebind = True)
    liveviewbutton = ObjectProperty(None, rebind = True)
    snapbutton = ObjectProperty(None, rebind = True)

    def __init__(self,  **kwargs):
        super(ImageAcquisitionButtonLayout, self).__init__(**kwargs)

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
                basler.save_image(img, path, snap_filename)
                
        elif self.liveviewbutton.state == 'down':
            # If currently in live view mode
            #   then save the current image
            img = app.lastframe
            basler.save_image(img, path, snap_filename)
                

class ScalableImage(ScatterLayout):

    def on_touch_up(self, touch):
        
        # If the widget is enabled and interaction point is inside its bounding box
        if self.disabled or not self.collide_point(*touch.pos):
            return

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


class ImageOverlay(BoxLayout):

    def __init__(self,  **kwargs):
        super(ImageOverlay, self).__init__(**kwargs)
        self.hasDrawDualColorOverlay: bool = False
        self.label: Label | None = None


    def on_size(self, *args) -> None:
        """Update the position and size of the rectangle when the widget is resized
        """        
        if self.hasDrawDualColorOverlay:
            # Redraw the dual color overlay
            mainSide = App.get_running_app().config.get('DualColor', 'mainside')
            self.redrawDualColorOverlay(mainSide)


    def redrawDualColorOverlay(self, mainSide: str= 'Right'):
        """Redraw the dual color overlay by clear and draw.

        Args:
            mainSide (str, optional): Main side of the dual color mode. Defaults to 'Right'.
        """        
        self.clearDualColorOverlay()
        self.drawDualColorOverlay(mainSide)


    def drawDualColorOverlay(self, mainSide: str= 'Right'):
        """Draw the dual color overlay which are the middle seperate line and the label

        Args:
            mainSide (str, optional): Main side of the dual color mode. Defaults to 'Right'.
        """
        if self.hasDrawDualColorOverlay:
            return
        
        self.hasDrawDualColorOverlay = True

        app: MacroscopeApp = App.get_running_app()
        previewImage: PreviewImage = app.root.ids.middlecolumn.previewimage

        viewMode = app.config.get('DualColor', 'viewmode')

        if viewMode == 'Splitted':

            # Set the overlay size as the image size
            normImageSize = previewImage.get_norm_image_size()
            self.size = normImageSize

            # Set the overlay position to match the image position exactly.
            #   Note, this is a local position.
            imageWidgetSize = previewImage.size
            self.pos[0] = (imageWidgetSize[0] - normImageSize[0]) / 2
            self.pos[1] = (imageWidgetSize[1] - normImageSize[1]) / 2

            # 
            # Red line at the middle
            # 
            pos_center_local = self.to_local(self.center_x, self.center_y)
            p1 = (pos_center_local[0], pos_center_local[1] + self.height/2)
            p2 = (pos_center_local[0], pos_center_local[1] - self.height/2)
            self.canvas.add(Color(1., 0., 0., 0.5))
            self.canvas.add(Line(points= [p1[0], p1[1], p2[0], p2[1]], width= 1, cap= 'none'))

            # 
            # Label on the main side
            # 
            if self.label is None:
                # Create a Label and add it as a child
                self.label = Label(text= '', markup= True)        
                self.add_widget(self.label)
            else:
                # In this case, the self.canvas.clear() has been called so we have to redraw the label.
                #   Ideally, we would like to call self.canvas.add( some label draw instruction ) but I can't find it
                #   so we will mimick this by re-adding it again.
                self.remove_widget(self.label)
                self.add_widget(self.label)

            # Set Label position
            topPadding = 7
            leftPadding = 0
            wordSize = 33.0     # Word size is used to offset the text such that it is center aligned
            
            if mainSide == 'Left':
                leftPadding = normImageSize[0] * 1.0/4 - wordSize / 2

            elif mainSide == 'Right':
                leftPadding = normImageSize[0] * 3.0/4 - wordSize / 2

            # left, top, right, bottom
            self.label.text = '[color=8e0045]Main[/color]'
            self.label.text_size = self.size
            self.label.valign = 'top'
            self.label.halign = 'left'
            self.label.padding= [ leftPadding, topPadding, 0, 0 ]
        
        elif viewMode == 'Merged':

            # 
            # Label on the header
            # 
            if self.label is None:
                # Create a Label and add it as a child
                self.label = Label(text= '', markup= True)
                self.add_widget(self.label)
                
            else:
                # In this case, the self.canvas.clear() has been called so we have to redraw the label.
                #   Ideally, we would like to call self.canvas.add( some label draw instruction ) but I can't find it
                #   so we will mimick this by re-adding it again.
                self.remove_widget(self.label)
                self.add_widget(self.label)
            
            topPadding = 7

            # left, top, right, bottom
            self.label.text = '[color=8e0045]Dual Color: Merged[/color]'
            self.label.text_size = self.size
            self.label.valign = 'top'
            self.label.halign = 'center'
            self.label.padding= [ 0, topPadding, 0, 0 ]
        

    def clearDualColorOverlay(self):
        """Clear the canvas and set internal hasDraw flag to false
        """
        self.canvas.clear()
        self.hasDrawDualColorOverlay = False
    

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
        units = app.config.get('Calibration', 'step_units')
        minstep = app.config.getfloat('Tracking', 'min_step')
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        
        if not dualColorMode:
            # 
            # Move stage based on user input - happens here.
            # 
            ystep, xstep = macro.getStageDistances(app.root.ids.middlecolumn.previewimage.offset, app.imageToStageMat)
            print('Centering image',xstep, ystep, units)
            
            if xstep > minstep:
                stage.move_x(xstep, unit= units, wait_until_idle= True)
            if ystep > minstep:
                stage.move_y(ystep, unit= units, wait_until_idle= True)

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
        trackingMode =  app.config.get('Tracking', 'mode')
        area = app.config.getint('Tracking', 'area')
        threshold = app.config.getfloat('Tracking', 'threshold')

        # make a tracking thread 
        track_args = minstep, units, capture_radius, binning, dark_bg, area, threshold, trackingMode
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

        # Dual Color mode settings
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        mainSide = app.config.get('DualColor', 'mainside')
        
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
                # Wait for the stage to finished moving/centering at location in the
                # first time
                stage.wait_until_idle()

                img2_retrieveTimestamp = app.retrieveTimestamp
                estimated_next_timestamp = app.retrieveTimestamp

            # Get the latest image
            img2 = app.lastframe
            img2_retrieveTimestamp = app.retrieveTimestamp

            tracking_frame_start_time = time.perf_counter()

            # Crop for dual color
            if dualColorMode:
                h, w = img2.shape
                if mainSide == 'Left':
                    img2 = img2[:,:w//2]
                elif mainSide == 'Right':
                    img2 = img2[:,w//2:]

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
            # Invert Y because the coordinate is in image space which is top left, while the transformation matrix is in btm left
            ystep, xstep = macro.getStageDistances(np.array([-ystep, xstep]), app.imageToStageMat)
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


    def set_ROI(self, roiX, roiY):
        app = App.get_running_app()
        camera = app.camera
        rec = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionbuttonlayout.ids.recordbutton.state
        disp = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionbuttonlayout.ids.liveviewbutton.state
       
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

        # Load and gen camera&stage transformation matricies
        rotation = self.config.getfloat('Camera', 'rotation')
        imageNormalDir = self.config.get('Camera', 'imagenormaldir')
        imageNormalDir = +1 if imageNormalDir == '+Z' else -1
        pixelsize = self.config.getfloat('Camera', 'pixelsize')
        self.imageToStageMat, self.imageToStageRotMat = macro.CameraAndStageCalibrator.genImageToStageMatrix(rotation, imageNormalDir, pixelsize)

        # Load moveImageSpaceMode
        self.moveImageSpaceMode = self.config.getboolean('Stage', 'move_image_space_mode')

        return layout

    
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
        self.destroy_settings()
        self.bind_keys()
        # Enabled back the interaction with preview image widget
        self.root.ids.middlecolumn.ids.scalableimage.disabled = False
        # TODO: update device settings, i.e. stage limit
        # Check turning on or off dual color mode
        self.updateDualColorOverlay()
    

    def updateDualColorOverlay(self, isRedraw: bool = True):

        dualcolormode = self.config.getboolean('DualColor', 'dualcolormode')
        mainside = self.config.get('DualColor', 'mainside')
        
        # If in dual color mode then draw the overlay
        if dualcolormode:
            # Only redraw if nescessary
            if isRedraw:
                self.root.ids.middlecolumn.ids.imageoverlay.redrawDualColorOverlay(mainside)
                
        # If not in the dual color mode then clear the overlay
        else:
            self.root.ids.middlecolumn.ids.imageoverlay.clearDualColorOverlay()


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

        if 'shift' in modifier:
            v = self.vlow
        else:
            v = self.vhigh
        
        direction = {
            273: (0,v,0),  # up arrow
            274: (0,-v,0),   # down arrow
            275: (v,0,0),  # right arrow
            276: (-v,0,0),   # left arrow
            280: (0,0,-v),  # page up
            281: (0,0,v)    # page down
        }
        
        if key not in direction.keys():
            return
        
        # Stage movement mode
        if self.moveImageSpaceMode:
            move_img_space = direction[key]

            # Translate from image space into stage space
            translation_vec_img_space = np.array([move_img_space[1], move_img_space[0]], np.float32)
            translation_vec_stage_space = self.imageToStageRotMat @ translation_vec_img_space

            # Convert back to a 3D tuple
            translation_vec_stage_space = ( translation_vec_stage_space[1], translation_vec_stage_space[0], move_img_space[2] )
            
            self.stage.start_move(translation_vec_stage_space, self.unit)
        
        else:
            # Move 
            self.stage.start_move(direction[key], self.unit)


    def _keyup(self, instance, key, scancode) -> None:
        """Handle keyup callbacks. This is usually only for stopping axis movement"""
        if self.stage is None:
            return
        
        # Stopping axis depending on the movement mode
        if self.moveImageSpaceMode:
            
            # TODO: Improve this feature so that we can move in image space simultaneously
            #   in both X,Y axis. Will require additive velocity movement handling.
            if key in [273, 274, 275, 276, 280, 281]:
                self.stage.stop(stopAxis= AxisEnum.ALL)
                self.coords = self.stage.get_position()
        
        else:
        
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
        if config is not self.config:
            return
        
        token = (section, key)
        if token == ('Camera', 'pixelsize') or token == ('Camera', 'rotation'):
            print('updated calibration matrix')
            pixelsize = self.config.getfloat('Camera', 'pixelsize')
            rotation= self.config.getfloat('Camera', 'rotation')
            self.imageToStageMat, self.imageToStageRotMat = macro.genImageToStageMatrix(pixelsize, rotation)
        
        elif token == ('Experiment', 'exppath'):
            self.root.ids.leftcolumn.ids.saveloc.text = value
        
        elif token == ('Stage', 'move_image_space_mode'):
            # Token is a str of int or float, i.e. '0', '1' so we have to parse it to boolean
            self.moveImageSpaceMode = bool(int(value))


    def on_image(self, *args) -> None:
        """On im age change callback. Update image texture and GUI overlay
        """
        # 
        # Pre process image
        # 
        image = np.copy(self.image)

        dualcolorMode = self.config.getboolean('DualColor', 'dualcolormode')
        dualcolorViewMode = self.config.get('DualColor', 'viewmode')

        if dualcolorMode and dualcolorViewMode == 'Merged':

            # Split image into main and minor side
            mainSide = self.config.get('DualColor', 'mainside')

            h, w = image.shape
            mainSideImage = None
            minorSideImage = None

            if mainSide == 'Left':
                mainSideImage = image[:,:w//2]
                minorSideImage = image[:,w//2:]

            elif mainSide == 'Right':
                mainSideImage = image[:,w//2:]
                minorSideImage = image[:,:w//2]
            
            # Compute minor to main calibration matrix
            translation_x = self.config.getfloat('DualColor', 'translation_x')
            translation_y = self.config.getfloat('DualColor', 'translation_y')
            rotation = self.config.getfloat('DualColor', 'rotation')
            
            minorToMainMat = macro.DualColorImageCalibrator.genMinorToMainMatrix(translation_x, translation_y, rotation, mainSideImage.shape[1]/2, mainSideImage.shape[0]/2)

            # Apply transformation
            #   Use warpAffine() here instead of warpPerspective for a little faster computation.
            #   It also takes 2x3 mat, so we cut the last row out accordingly.
            translatedMinorSideImage = cv2.warpAffine(minorSideImage, minorToMainMat[:2,:], (minorSideImage.shape[1], minorSideImage.shape[0]))

            # Combine main and minor side
            combinedImage = np.zeros(shape= (mainSideImage.shape[0], mainSideImage.shape[1], 3), dtype= np.uint8)
            combinedImage[:,:,0] = mainSideImage
            combinedImage[:,:,1] = translatedMinorSideImage

            image = combinedImage
        

        # 
        # Upload image to texture
        # 
        
        imageHeight, imageWidth = image.shape[0], image.shape[1]
        imageColorFormat = 'rgb' if image.ndim == 3 else 'luminance'
        # Force unsign byte format
        imageDataFormat = 'ubyte'
        # 
        updateGUIFlag = False

        # Check if need to recreate texture
        if self.texture is None \
            or self.texture.width != imageWidth or self.texture.height != imageHeight \
            or self.texture.colorfmt != imageColorFormat \
            or self.texture.bufferfmt != imageDataFormat:
            
            # Recreate texture
            self.texture = Texture.create(
                size= (imageWidth, imageHeight),
                colorfmt= imageColorFormat
            )

            # Kivy texture is in OpenGL corrindate which is btm-left origin so we need to flip texture coord once to match numpy's top-left
            self.texture.flip_vertical()

            # Set flag update GUI
            updateGUIFlag = True
        
        # Upload image data to texture
        buf = image.tobytes()
        self.texture.blit_buffer(buf, colorfmt= imageColorFormat, bufferfmt= imageDataFormat)

        # 
        # Update GUI
        # 

        # Update GUI overlay
        self.updateDualColorOverlay(isRedraw= updateGUIFlag)
    

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
