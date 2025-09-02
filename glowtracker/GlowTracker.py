import os
# Suppress kivy normal initialization logs in the beginning
# for easier debugging
os.environ["KCFG_KIVY_LOG_LEVEL"] = "warning"
# Emulate camera
# os.environ["PYLON_CAMEMU"] = "1"

# 
# Kivy Imports
# 
import kivy
# Require modern version
kivy.require('2.0.0')
from kivy.app import App
from kivy.lang import Builder
from kivy.config import Config, ConfigParser
# get the free clock (more accurate timing)
# Config.set('graphics', 'KIVY_CLOCK', 'free')
# Config.set('modules', 'monitor', '')
Config.set('input', 'mouse', 'mouse,disable_multitouch')  # turns off the multi-touch emulation

from kivy.cache import Cache
from kivy.base import EventLoop
from kivy.core.window import Window
from kivy.graphics import Color, Line, Ellipse, Rectangle
from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty, BoundedNumericProperty, NumericProperty, ConfigParserProperty, ListProperty
from kivy.clock import Clock, ClockEvent, mainthread
from kivy.metrics import Metrics
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.scatter import Scatter
from kivy.uix.tabbedpanel import TabbedPanel
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.popup import Popup
from kivy.uix.settings import SettingsWithSidebar, SettingItem, SettingNumeric
from kivy.uix.textinput import TextInput
from kivy.uix.codeinput import CodeInput
from kivy.uix.slider import Slider
from kivy.uix.behaviors import DragBehavior
from kivy.uix.switch import Switch

# 
# IO, Utils
# 
import datetime
import time
from pathlib import Path
from threading import Thread, Lock
from multiprocessing.pool import ThreadPool
from functools import partial
from queue import Queue
from overrides import override
from typing import List, Tuple
from io import TextIOWrapper
import zaber_motion     # We need to import zaber_motion before pypylon to prevent environment crash
from pypylon import pylon
import platformdirs 
import shutil
from pyparsing import ParseException
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 
# Own classes
# 
from Zaber_control import Stage, AxisEnum
import Microscope_macros as macro
import Basler_control as basler
from MacroScript import MacroScriptExecutor
from AutoFocus import AutoFocusPID, FocusEstimationMethod

# 
# Math
# 
import math
import numpy as np
from skimage.io import imsave
import cv2
from scipy.stats import skew

# helper functions
def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S-%f-{fname}'):
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

    def __init__(self, text = 'warning', closeTime: float = 2, **kwargs):
        super(WarningPopup, self).__init__(**kwargs)
        self.text = text
        # call dismiss_popup in closeTime
        Clock.schedule_once(self.dismiss, closeTime)

    def ok(self):
        self.dismiss()


# main GUI
class MainWindow(GridLayout):
    pass


class LeftColumn(BoxLayout):
    # file saving and loading
    cameraConfigFile = ObjectProperty(None)
    savefile = StringProperty("")
    cameraprops = ObjectProperty(None)
    saveloc = ObjectProperty(None)
    #

    def __init__(self,  **kwargs):
        super(LeftColumn, self).__init__(**kwargs)

        self.app: GlowTrackerApp = App.get_running_app()

        # Camera config value
        self.cameraConfig: dict[str:any] = dict()
        
        Clock.schedule_once(self._do_setup)

    def _do_setup(self, *l):
        self.savefile = self.app.config.get("Experiment", "exppath")
        self.path_validate()
        self.cameraConfigFile = self.app.config.get('Camera', 'default_settings')
        self.apply_cam_settings()


    def path_validate(self):
        p = Path(self.saveloc.text)
        if p.exists() and p.is_dir():
            self.app.config.set("Experiment", "exppath", self.saveloc.text)
            self.app.config.write()
        # check if the parent dir exists, then create the folder
        elif p.parent.exists():
            p.mkdir(mode=0o777, parents=False, exist_ok=True)
        else:
            self.saveloc.text = self.savefile
        self.app.config.set("Experiment", "exppath", self.saveloc.text)
        self.app.config.write()
        self.savefile = self.app.config.get("Experiment", "exppath")
        # reset the stage keys
        self.app.bind_keys()
        print('saving path changed')


    def dismiss_popup(self):
        self.app.bind_keys()
        self._popup.dismiss()


    # popup camera file selector
    def show_load(self):
        content = LoadCameraProperties(load=self.load, cancel=self.dismiss_popup)
        content.ids.filechooser2.path = self.cameraConfigFile
        self._popup = Popup(title="Load camera file", content=content,
                            size_hint=(0.9, 0.9))
         #unbind keyboard events
        self.app.unbind_keys()
        self._popup.open()


    # popup experiment dialog selector
    def show_save(self):
        content = SaveExperiment(save=self.save, cancel=self.dismiss_popup)
        content.ids.filechooser.path = self.savefile
        self._popup = Popup(title="Select save location", content=content,
                            size_hint=(0.9, 0.9))
        #unbind keyboard events
        self.app.unbind_keys()
        self._popup.open()


    def load(self, path, filename):
        self.cameraConfigFile = os.path.join(path, filename[0])
        self.apply_cam_settings()
        self.dismiss_popup()


    def save(self, path, filename):
        self.savefile = os.path.join(path, filename)
        self.saveloc.text = (self.savefile)
        self.path_validate()
        self.dismiss_popup()


    def apply_cam_settings(self) -> None:
        """Read and apply the camera config file to the camera.
        """
        camera: basler.Camera = self.app.camera

        if camera is not None:

            if os.path.isfile(self.cameraConfigFile):
                # If the camera config file exists, load it into the camera.
                print('Updating camera settings')

                # Set the camera config as specified in the file.
                camera.updateProperties(self.cameraConfigFile)

                # Read and store the camera config separately for later use
                self.cameraConfig = basler.readPFSFile(self.cameraConfigFile)

                # Update display values on the GUI
                self.update_settings_display()

            else:
                # Otherwise, don't modify the camera and only copy attributes from camera object to dict
                self.cameraConfig = camera.getAllFeatures()


    # when file is loaded - update slider values which updates the camera
    def update_settings_display(self):
        # update slider value using ids
        camera = self.app.camera
        self.ids.camprops.exposure = camera.ExposureTime()
        self.ids.camprops.gain = camera.Gain()
        self.ids.camprops.framerate = camera.ResultingFrameRate()


    def autoFocusButtonCallback(self):
        """Move to the best focused distance by perform a sweep scan (z-axis) about the current position.
        """
        camera = self.app.camera
        stage = self.app.stage

        if camera is None or stage is None:
            # Popup a warning dialog
            self._popup = WarningPopup(title="Autofocus", text='Autofocus requires a stage and a running camera!',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            
        else:

            # Check if acquiring image
            imageAcquisitionManager: ImageAcquisitionManager = self.app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager

            # Stop camera if already running
            liveViewButton: Button = imageAcquisitionManager.ids.liveviewbutton
            prevLiveViewButtonState: str = liveViewButton.state
            liveViewButton.state = 'normal'

            #   Load settings
            depthoffield = self.app.config.getfloat('Camera', 'depthoffield')
            depthoffieldsearchdistance = self.app.config.getfloat('Calibration', 'depthoffieldsearchdistance')
            dualColorMode = self.app.config.getboolean('DualColor', 'dualcolormode')
            dualColorModeMainSide = self.app.config.get('DualColor', 'mainside')
            capturedRadius = self.app.config.getint('Tracking', 'capture_radius')
            focusEstimationMethod = FocusEstimationMethod(self.app.config.get('Autofocus', 'focusestimationmethod'))

            #   Reuse DepthOfFieldEstimator to scan and search for the best focus position
            depthOfFieldEstimator = macro.DepthOfFieldEstimator()
            numSamples = math.floor(depthoffieldsearchdistance / depthoffield) + 1
            depthOfFieldEstimator.takeCalibrationImages(camera, stage, depthoffieldsearchdistance, numSamples, focusEstimationMethod, dualColorMode, dualColorModeMainSide, capturedRadius)

            #   Get best-focused position
            bestFocusIndex = depthOfFieldEstimator.dofDataFrame['estimatedFocus'].idxmax()
            bestFocusPosition = depthOfFieldEstimator.dofDataFrame.iloc[bestFocusIndex]['pos_z']
            
            # Move to the best-focus position
            stagePosition = stage.get_position()
            stagePosition[2] = bestFocusPosition
            stage.move_abs(stagePosition, unit= 'mm')

            # Remember best focus value for later auto focus
            bestFocusValue = depthOfFieldEstimator.dofDataFrame.iloc[bestFocusIndex]['estimatedFocus']
            self.app.config.set('Autofocus', 'bestfocusvalue', bestFocusValue)
            self.app.config.write()

            # Return LiveView state
            liveViewButton.state = prevLiveViewButtonState


class MiddleColumn(BoxLayout, StencilView):
    runtimecontrols = ObjectProperty(None)
    previewimage = ObjectProperty(None)


class RightColumn(BoxLayout):

    def __init__(self,  **kwargs):
        super(RightColumn, self).__init__(**kwargs)
        # Class instance attributes
        self.app: GlowTrackerApp = App.get_running_app()


    def dismiss_popup(self):
        #rebind keyboard events
        self.app.bind_keys()
        self.app.root.ids.middlecolumn.ids.scalableimage.disabled = False
        self._popup.dismiss()
    

    def open_macro(self):
        """Open the macro script widget popup.
        """
        
        # Disabled interaction with preview image widget
        self.app.root.ids.middlecolumn.ids.scalableimage.disabled = True
        # Unbind keyboard events
        self.app.unbind_keys()

        # Create MacroScriptWidget Draggable Popup
        widget = MacroScriptWidget(app = self.app)
        widget.closeCallback = self.dismiss_popup
        self._popup = MacroScriptWidgetPopup(title= "Macro Script", content= widget, size_hint= (0.5, 0.7), auto_dismiss = False)
        self._popup.closeCallback = self.dismiss_popup

        # Open the widget
        self._popup.open()


    def open_settings(self):
        # Disabled interaction with preview image widget
        self.app.root.ids.middlecolumn.ids.scalableimage.disabled = True
        # Call open settings
        self.app.open_settings()


    def show_recording_settings(self):
        """change recording settings."""
        #unbind keyboard events
        self.app.unbind_keys()

        recordingSettings = RecordingSettings(ok= self.dismiss_popup)
        self._popup = Popup(title= "Recording Settings", content= recordingSettings, size_hint = (0.3, 0.45))
        self._popup.open()


    def show_calibration(self):
        """Show calibration window popup.
        """        
        camera = self.app.camera
        stage: Stage = self.app.stage

        if camera is not None and stage is not None:
            # Create the calibration widget
            calibrationTabPanel = CalibrationTabPanel()
            calibrationTabPanel.setCloseCallback(closeCallback= self.dismiss_popup)
            # Launch the widget inside a popup window
            self._popup = Popup(title= '', separator_height= 0, content= calibrationTabPanel, size_hint= (0.9, 0.75))
            self._popup.open()
        
        else:
            self._popup = WarningPopup(title="Calibration", text='Autocalibration requires a stage and a camera. Connect a stage or use a calibration slide.',
                            size_hint=(0.5, 0.25))
            self._popup.open()


class MacroScriptWidgetPopup(DragBehavior, Popup):

    closeCallback = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(MacroScriptWidgetPopup, self).__init__(**kwargs)


    @override
    def _align_center(self, *_args):
        # Override align_center function to not do naything
        pass


    @override
    def _handle_keyboard(self, _window, key, *_args):
        """Override handle_keyboard function to always close the widget when ESC is pressed,
        regardless whether the self.auto_dismiss is True or False
        """
        # ESC 
        if key == 27:
            # Call closing the popup procedure
            self.closeCallback()
            # Tell the caller to stop propagating keyboard event
            return True

    
    @override
    def on_touch_down(self, touch) -> bool:
        """Override on_touch_down function to check if the touch is inside the CodeInput region.
        If it is, then disable the drag behavior and allow the CodeInput to handle the touch instead.
        Returns:
            has_been_handled(bool): Flag to indicate if the touch event has been handled or not to stop propagation.
        """
        discardRegion: CodeInput = self.content.ids.macroscripttext
        
        if discardRegion.collide_point(*touch.pos):
            return discardRegion.on_touch_down(touch)
            
        else:
            return super().on_touch_down(touch)


class MacroScriptWidget(BoxLayout):
    """MacroScriptExecutor widget that holds the parser and the function handler
    """
    closeCallback = ObjectProperty(None)

    def __init__(self, **kwargs):

        # Intercept GlowTrackerApp reference object.
        #   There is a bug that if we call to get reference directly by App().get_running_app(),
        #   we would get a new GlowTrackerApp object that has different object id, and no config, root, etc. 
        #   like a completely new object.
        self.app: GlowTrackerApp = kwargs.pop('app', None)
        
        super(MacroScriptWidget, self).__init__(**kwargs)

        # Attributes
        self.macroScriptFile: str = self.ids.macroscriptfile.text
        self.macroScript: str = ''
        self._popup: Popup = None
        self.macroScriptExecutor = MacroScriptExecutor()

        # Initialize MacroScriptExecutor
        self.stage = self.app.stage
        self.camera = self.app.camera
        self.imageAcquisitionManager: ImageAcquisitionManager = self.app.root.ids.middlecolumn.ids.runtimecontrols.imageacquisitionmanager
        self.recordButton: RecordButton = self.imageAcquisitionManager.recordbutton
        position_unit = 'mm'

        # Register appropriate function handler
        self.macroScriptExecutor.registerFunctionHandler(
            move_abs_handle= lambda x, y, z: self.stage.move_abs((x,y,z), position_unit, wait_until_idle= True),
            move_rel_handle= lambda x, y, z: self.stage.move_rel((x,y,z), position_unit, wait_until_idle= True),
            snap_handle= lambda: self.imageAcquisitionManager.snap(),
            record_for_handle= lambda x: self._record_for_handle(x),
            start_recording_handle= self._start_recording_handle,
            stop_recording_handle= self._stop_recording_handle
        )

        # Load the recent script
        if self.macroScriptFile != '':
            self.loadMacroScript(self.macroScriptFile)


    def _record_for_handle(self, recordingTime: float):
        """Start recording for a certain amount of time.

        Args:
            recordingTime (float): recording duratino in seconds
        """

        # Check if still in recording mode, if so, overwrite it
        if self.recordButton.state == 'down':
            self.recordButton.state = 'normal'

            # Wait until the camera really stop grabbing
            while self.app.camera.IsGrabbing():
                time.sleep(0.01)
        
        # Set recording config
        self.app.config.set('Experiment', 'iscontinuous', False)

        framerate = self.app.config.getfloat('Experiment', 'framerate')
        nframes = int(recordingTime*framerate)

        # Unfortunately, the setting the nframes would invoke a cascading event
        #   that eventually want to update the GUI. Which is not allowed to do
        #   in a thread that is not a main thread.
        @mainthread
        def setNFrames():
            self.app.config.set('Experiment', 'nframes', nframes)
        
        setNFrames()

        self.app.config.write()
        
        # Start the recording mode
        self.recordButton.state = 'down'

    
    def _start_recording_handle(self):
        """Start the recording mode.
        """

        # Check if still in recording mode, if so, overwrite it
        if self.recordButton.state == 'down':
            self.recordButton.state = 'normal'

            # Wait until the camera really stop grabbing
            while self.app.camera.IsGrabbing():
                time.sleep(0.01)
        
        # Set recording config
        self.app.config.set('Experiment', 'iscontinuous', True)
        self.app.config.write()

        # Start the recording mode
        self.recordButton.state = 'down'


    def _stop_recording_handle(self):
        """Stop the recording mode.
        """
        self.recordButton.state = 'normal'


    def openLoadMacroScriptWidget(self):
        """Open a popup to load the macro script.
        """
        
        loadWidget = LoadMacroScriptWidget(load= self._loadScriptWidgetCallback)
        self._popup = Popup(title= "Load macro script file", content= loadWidget,
            size_hint= (0.9, 0.9), auto_dismiss= False)

        loadWidget.cancel = self._popup.dismiss
        self._popup.open()

    
    def _loadScriptWidgetCallback(self, selection: list[str]):
        """Load the macro script from a list of given file path. Will choose only the first file.
        Used for handler of LoadMacroScriptWidget.

        Args:
            selection (list[str]): list of script file path
        """
        # Close the loading widget
        if self._popup is not None:
            self._popup.dismiss()

        if len(selection) == 0:
            return
        
        self.loadMacroScript(selection[0])
    
    
    def loadMacroScript(self, filePath: str):
        """Load the macro script from a given file path.

        Args:
            filePath (str): _description_
        """
        # Get the absolute file path
        self.macroScriptFile = os.path.abspath(filePath)
        
        # Load the script text
        print(f'Loading the macro script {self.macroScriptFile}')

        try:
            with open(self.macroScriptFile, 'r') as file:
                self.macroScript = file.read()

        except FileNotFoundError:
            print(f'The file {self.macroScriptFile} was not found.')

        except IOError:
            print(f'An error occurred while reading the file {self.macroScriptFile}.')
        
        # Set display text
        self.ids.macroscriptfile.text = self.macroScriptFile
        self.ids.macroscripttext.text = self.macroScript

        # Set as recent script
        self.app.config.set('MacroScript', 'recentscript', self.macroScriptFile)
        self.app.config.write()
    

    def saveMacroScript(self):
        """Save the current macro script into the same file (overwrite if exists).
        """
        file_path = self.ids.macroscriptfile.text
        script = self.ids.macroscripttext.text

        try:
            # Convert to absolute path if it's a relative path
            abs_file_path = os.path.abspath(file_path)
            
            # Ensure the directory exists
            directory = os.path.dirname(abs_file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Open the file in overwrite mode, creating it if it doesn't exist
            with open(abs_file_path, 'w') as file:
                file.write(script)

            # Set as recent script
            self.app.config.set('MacroScript', 'recentscript', self.ids.macroscriptfile.text)

            print(f"Saved the script {file_path}")

        except IOError as e:
            print(f"Error saving to {file_path}: {e}")

        except Exception as e:
            print(f"Error saving macro script: {e}")
    

    def runMacroScript(self):
        """Run the current macro script.
        """

        print(f'Running the macro script {self.macroScriptFile}.')
        try:
            self.macroScriptExecutor.executeScript(self.ids.macroscripttext.text, self.finishedMacroScript)

            # Disable the run button
            self.ids.runbutton.disabled = True

        except ParseException as e:
            print(f"Parsing error: {e}")
        

    def finishedMacroScript(self):
        """Callback when the macro script is finished. Simply enable the run button back.
        """
        print('Finished running the macro script.')
        
        # Enable the run button
        self.ids.runbutton.disabled = False
        

    def stopMacroScript(self):
        """Stop running the macro script.
        """
        print('Stop running the macro script.')
        self.macroScriptExecutor.stop()


class LoadMacroScriptWidget(BoxLayout):
    """Camera settings loading widget
    """
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class CalibrationTabPanel(TabbedPanel):
    """Calibration widget that holds CameraAndStageCalibration, DualColorCalibration, and DepthOfFieldCalibration
    """    

    def setCloseCallback(self, closeCallback: callable) -> None:
        """API setting close callback event for children' tab.

        Args:
            closeCallback (callable): the closing callback event.
        """        
        self.closeCallback = closeCallback
        self.ids.stagecalibration.setCloseCallback( closeCallback )
        self.ids.dualcolorcalibration.setCloseCallback( closeCallback )
        self.ids.depthoffieldcalibration.setCloseCallback( closeCallback )


class CameraAndStageCalibration(BoxLayout):
    """Camera And Stage calibration widget that handles linking button callbacks and the calibration algorithm class.
    """    
    closeCallback = ObjectProperty(None)
    
    def setCloseCallback( self, closeCallback: callable ) -> None:
        """Set widget closing callback.

        Args:
            closeCallback (callable): the closing callback.
        """        
        self.closeCallback = closeCallback
    

    def calibrate(self):
        """Execute the camera and stage calibration process.
            1. Take calibration images.
            2. Estimate camera to stage transformation matrix.
            3. Display results.
        """        
        app: GlowTrackerApp = App.get_running_app()
        camera = app.camera
        stage = app.stage

        if camera is None or stage is None:
            return
        
        # stop camera if already running
        liveViewButton: Button = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager.ids.liveviewbutton
        prevLiveViewButtonState = liveViewButton.state
        liveViewButton.state = 'normal'
        
        # get config values
        stepsize = app.config.getfloat('Calibration', 'step_size')
        stepunits = app.config.get('Calibration', 'step_units')
        dualcolormode = app.config.getboolean('DualColor', 'dualcolormode')
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
        calibratedParameters = cameraAndStageCalibrator.calibrateCameraAndStageTransform()

        if calibratedParameters is None:

            # Display a popup message say to try again.
            warningPopup = WarningPopup(title="Calibration Failed", text='Calibration was unsuccessful. Try changing the image and calibrate again.',
                            size_hint=(0.35, 0.2), closeTime = 6)
            warningPopup.open()
            return
        
        rotation, imageNormDir, pixelSize = calibratedParameters
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
        plotImage = macro.CameraAndStageCalibrator.renderChangeOfBasisImage(macro.swapMatXYOrder(stageToImageRotMat))
        self.ids.cameraandstageaxes.texture = imageToTexture(plotImage)

        # Resume the camera to previous state
        liveViewButton.state = prevLiveViewButtonState


class DualColorCalibration(BoxLayout):
    """Dual color calibration widget that handles linking button callbacks and the calibration algorithm class.
    """    
    closeCallback = ObjectProperty(None)
    
    def setCloseCallback( self, closeCallback: callable ) -> None:
        """Set widget closing callback.

        Args:
            closeCallback (callable): the closing callback.
        """     
        self.closeCallback = closeCallback
    

    def calibrate(self) -> None:
        """Execute the dual color calibration process.
            1. Take a dual color image.
            2. Process the dual color image.
            3. Calibrate main side to minor side transformation matrix.
            4. Display results.
        """        
        app: GlowTrackerApp = App.get_running_app()
        camera: basler.Camera = app.camera
        stage: Stage = app.stage

        if camera is None or stage is None:
            return
        
        # stop camera if already running
        liveViewButton: Button = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager.ids.liveviewbutton
        prevLiveViewButtonState = liveViewButton.state
        liveViewButton.state = 'normal'
        
        # Take a dual color image for calibration
        isSuccess, dualColorImage = camera.singleTake()

        if not isSuccess:
            return

        mainSide = app.config.get('DualColor', 'mainside')
        
        # Instantiate a dual color calibrator
        dualColorImageCalibrator = macro.DualColorImageCalibrator()

        # Process the dual color image
        mainSideImage, minorSideImage = dualColorImageCalibrator.processDualColorImage(
            dualColorImage= dualColorImage,
            mainSide= mainSide
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
        combinedImage = np.zeros(shape= (mainSideImage.shape[0], mainSideImage.shape[1], 3), dtype= np.uint8)
        combinedImage[:,:,0] = mainSideImage
        combinedImage[:,:,1] = translatedMinorSideImage

        # Update the composite display image
        self.ids.calibratedimage.texture = imageToTexture(combinedImage)

        # Resume the camera to previous state
        liveViewButton.state = prevLiveViewButtonState


class DepthOfFieldCalibration(BoxLayout):
    """Camera And Stage calibration widget that handles linking button callbacks and the calibration algorithm class.
    """    
    closeCallback = ObjectProperty(None)
    
    def setCloseCallback( self, closeCallback: callable ) -> None:
        """Set widget closing callback.

        Args:
            closeCallback (callable): the closing callback.
        """        
        self.closeCallback = closeCallback
    

    def calibrate(self):
        """Estimate Depth of Field of the current optic system and display the results.
        """
        app: GlowTrackerApp = App.get_running_app()
        camera: basler.Camera = app.camera
        stage: Stage = app.stage

        # Safe guard
        if camera is None or stage is None:
            return
        
        # stop camera if already running
        liveViewButton: Button = app.root.ids.middlecolumn.ids.runtimecontrols.ids.imageacquisitionmanager.ids.liveviewbutton
        prevLiveViewButtonState = liveViewButton.state
        liveViewButton.state = 'normal'
        
        # get config values
        depthoffieldsearchdistance = app.config.getfloat('Calibration', 'depthoffieldsearchdistance')
        depthoffieldnumsampleimages = app.config.getint('Calibration', 'depthoffieldnumsampleimages')
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        mainSide = app.config.get('DualColor', 'mainside')
        capturedRadius = app.config.getint('Tracking', 'capture_radius')
        focusEstimationMethod = FocusEstimationMethod(app.config.get('Autofocus', 'focusestimationmethod'))


        # Take calibration images
        depthOfFieldEstimator = macro.DepthOfFieldEstimator()
        
        # Estimate DOF
        try:
            estimatedDof = depthOfFieldEstimator.estimate(camera, stage, depthoffieldsearchdistance, depthoffieldnumsampleimages, focusEstimationMethod, dualColorMode, mainSide, capturedRadius)

            # Display estimation plot
            estimatedDofPlotImage = depthOfFieldEstimator.genEstimatedDofPlot()
            self.ids.estimateddofplot.texture = imageToTexture(estimatedDofPlotImage)

            # Display best focus image
            bestFocusPosition, bestFocusImage, bestFocusValue = depthOfFieldEstimator.getBestFocusImage()
            self.ids.bestfocusimage.texture = imageToTexture(bestFocusImage)

            # Update display text
            self.ids.estimateddepthoffieldtext.text = f"Estimated Depth of Field: {estimatedDof:.5f} mm. Best in-focused position: {bestFocusPosition:.2f} mm."

            # Save to config
            app.config.set('Camera', 'depthoffield', estimatedDof)
            app.config.set('Autofocus', 'bestfocusvalue', bestFocusValue)
            app.config.write()

        except Exception as e:
            print(f'Failed to estimate depth of field: {e}')

        # Resume the camera to previous state
        liveViewButton.state = prevLiveViewButtonState


class StageAxisController(BoxLayout):
    """Template class for stage axis controller widget.
    """    

    def __init__(self,  **kwargs):
        super(StageAxisController, self).__init__(**kwargs)

    def disable_all(self):
        for id in self.ids:
            self.ids[id].disabled = True
    
    def enable_all(self):
        for id in self.ids:
            self.ids[id].disabled = False

class XControls(StageAxisController):

    def __init__(self,  **kwargs):
        super(XControls, self).__init__(**kwargs)


class YControls(StageAxisController):

    def __init__(self,  **kwargs):
        super(YControls, self).__init__(**kwargs)


class ZControls(StageAxisController):

    def __init__(self,  **kwargs):
        super(ZControls, self).__init__(**kwargs)


class LoadCameraProperties(BoxLayout):
    """Camera settings loading widget
    """    
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class SaveExperiment(GridLayout):
    """File saving location widget.
    """    
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)


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
    """Record settings widget
    """    
    ok = ObjectProperty(None)
    # store recording settings from popups
    nframes = ConfigParserProperty(5, 'Experiment', 'nframes', 'app', val_type=int)
    fileformat = ConfigParserProperty('jpg', 'Experiment', 'extension', 'app', val_type=str)
    duration = ConfigParserProperty(5, 'Experiment', 'duration', 'app', val_type=float)

    def __init__(self,  **kwargs):
        super(RecordingSettings, self).__init__(**kwargs)
        if self.framerate != 0:
            self.duration = self.nframes/self.framerate

    @property
    def framerate(self):
        """An alias property refers to CameraProperties.framerate for ease of use
        """
        return App.get_running_app().root.ids.leftcolumn.ids.camprops.framerate


class ContinuousSwitch(Switch):

    def __init__(self, **kwargs):
        super(ContinuousSwitch, self).__init__(**kwargs)
        self.app: GlowTrackerApp = App.get_running_app()
        self.active = self.app.config.getboolean('Experiment', 'iscontinuous')
    

    def on_touch_up(self, touch): 
        """On switch touch up callback. Update the config value 'iscontinuous',
            and disabled or enabled the recording 'duration' and 'frames' input field.

        Args:
            touch (Touch): touch input data.
        """
        super(ContinuousSwitch, self).on_touch_up(touch)

        recordingSettings: RecordingSettings = self.parent.parent

        if self.active:
            self.app.config.set('Experiment', 'iscontinuous', True) 
            recordingSettings.ids.duration.disabled = True
            recordingSettings.ids.frames.disabled = True

        else:
            self.app.config.set('Experiment', 'iscontinuous', False)
            recordingSettings.ids.duration.disabled = False
            recordingSettings.ids.frames.disabled = False


class CameraProperties(GridLayout):
    """Camera properties editor widget
    """   
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


@dataclass
class LiveAnalysisData():
    minBrightness: float = 0
    maxBrightness: float = 0
    meanBrightness: float = 0
    medianBrightness: float = 0
    skewness: float = 0
    percentile_5: float = 0
    percentile_95: float = 0


class ImageAcquisitionButton(ToggleButton):
    """A skeleton template class for a widget button that have image acquisition behavior.
    This class outline the common process of image acquisitions:
        1. Start Acquisition
        2. Acquisition loop
        3. Process acquired image callback
        4. Acquisition callback
        5. Finishing acquisition loop callback
        6. Stop Acquisition

        This class is incomplete in itself and the heir need to overrides the provided
    outline functions, most importantly:
        - startImageAcuisition()
        - acquisitionCondition()

    in order to be functionable.
    """    
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # Declar class's instance attributes
        self.app: GlowTrackerApp | None = None
        self.camera: basler.Camera | None = None
        self.imageAcquisitionThread: Thread | None = None
        self.runtimeControls: RuntimeControls | None = None
        self.updateDisplayImageEvent: ClockEvent | None = None
        self.image: np.ndarray = np.zeros((1,1))
        self.imageTimeStamp: float = 0
        self.imageRetrieveTimeStamp: float = 0
        self.dualColorMainSideImage: np.ndarray = np.zeros((1,1))
        self.dualColorMinorSideImage: np.ndarray = np.zeros((1,1))
        self.dualColorMinorToMainMat: np.ndarray | None = None

    
    def on_state(self, widget: Widget, state: str):
        """On state change callback

        Args:
            widget (Widget): the kivy widget, in this case is the same as the class instance itself.
            state (str): the new state
        """        
        if state == 'down':
            self.startImageAcquisition()
            
        else:
            self.stopImageAcquisition()

    
    def startImageAcquisition(self) -> None:
        """The starting image acquisition process. Needs to be overridden. The important steps is to spawn the imageAcquisitionLoopingThread.
        """        
        pass


    def stopImageAcquisition(self) -> None:
        """Stop the image acquisition process. Should be overridden or extended.
        The important steps are:
            - Stop the update display event.
            - Stop camera grabbing.
            - Stop the acquisition looping thread if not already.
        """        
        if self.camera is None:
            return
        
        # Unschedule the display event thread
        Clock.unschedule(self.updateDisplayImageEvent)

        # Stop grabbing
        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()

        # Flag recompute dual color transformation matrix
        self.dualColorMinorToMainMat = None
        
        # Reset displayed framecounter
        self.runtimeControls.framecounter.value = 0

        # reset scale of image
        self.app.root.ids.middlecolumn.ids.scalableimage.reset()

        # Set self button state to normal
        self.state = 'normal'
    

    def imageAcquisitionLoopingThread(self, grabArgs) -> None:
        """Image acquisition looping thread. This function should not be call directly 
        in the main thread but as a new thread instead for better performance.
        The procedure here is as follows:
            1. Start camera grabbing.
            2. Start a update display image event.
            3. Loop acquire image while the condition is True.
            4. Callback for each acquired image.
            5. Finished looping callback.
        """   

        if grabArgs.isContinuous:
            self.camera.StartGrabbing(grabArgs.grabStrategy)
        
        else:
            # Grab for a specific number of frames
            self.camera.StartGrabbingMax(grabArgs.numberOfImagesToGrab, grabArgs.grabStrategy)
            
        fps = self.camera.ResultingFrameRate()
        print("Grabbing Framerate:", fps)

        # Schedule a display update
        fps = self.app.config.getfloat('Camera', 'display_fps')
        self.updateDisplayImageEvent = Clock.schedule_interval(self.updateDisplayImage, 1.0 /fps)
        print(f'Displaying at {fps} fps')

        returnCameraOnHoldFlag = True if self.camera.isOnHold() else False

        # Start image acquisition loop
        while self.acquisitionCondition():

            # retrieve an image
            isSuccess, image, imageTimeStamp, imageRetrieveTimeStamp = self.camera.retrieveGrabbingResult()

            if isSuccess:

                if returnCameraOnHoldFlag:
                    self.camera.setIsOnHold(False)
                    returnCameraOnHoldFlag = False

                # Process the received image
                self.processImageCallback( image, imageTimeStamp, imageRetrieveTimeStamp )

                # Trigger image callback
                self.receiveImageCallback()

        self.finishAcquisitionCallback()


    def acquisitionCondition(self) -> bool:
        """Check if the acquisition is still True. Needs to be overrided by heir.

        Returns:
            isStillAcquiring (bool): is the acquisition is still True.
        """        
        pass

    
    def processImageCallback(self, image: np.ndarray, imageTimeStamp: float, imageRetrieveTimeStamp: float) -> None:
        """Process the acquired image by cropping per settings, and also dual color image 
        processing if the dual color mode is on.

        Args:
            image (np.ndarray): the acquired image
            imageTimeStamp (float): the acquired image's internal clock timestamp
            imageRetrieveTimeStamp (float): the timestamp when receiving image in the software.
        """        

        # Crop image
        h, w = image.shape
        cropX, cropY = self.runtimeControls.cropX, self.runtimeControls.cropY
        image = image[ cropY : h - cropY, cropX : w - cropX ]
        
        # Process image. For now this is only the case for dual color mode
        dualcolorMode = self.app.config.getboolean('DualColor', 'dualcolormode')
        mainSide = self.app.config.get('DualColor', 'mainside')
        dualcolorViewMode = self.app.config.get('DualColor', 'viewmode')

        if dualcolorMode:
            # If in dual color mode then post process the image

            # Split image into main and minor side
            if mainSide == 'Left':
                self.dualColorMainSideImage = image[:,:w//2]
                self.dualColorMinorSideImage = image[:,w//2:]

            elif mainSide == 'Right':
                self.dualColorMainSideImage = image[:,w//2:]
                self.dualColorMinorSideImage = image[:,:w//2]
            
            # Compute minor to main calibration matrix if first time
            if self.dualColorMinorToMainMat is None:
                translation_x = self.app.config.getfloat('DualColor', 'translation_x')
                translation_y = self.app.config.getfloat('DualColor', 'translation_y')
                rotation = self.app.config.getfloat('DualColor', 'rotation')
                
                self.dualColorMinorToMainMat = macro.DualColorImageCalibrator.genMinorToMainMatrix(translation_x, translation_y, rotation, self.dualColorMainSideImage.shape[1]/2, self.dualColorMainSideImage.shape[0]/2)

            # Apply transformation
            #   Use warpAffine() here instead of warpPerspective for a little faster computation.
            #   It also takes 2x3 mat, so we cut the last row out accordingly.
            self.dualColorMinorSideImage = cv2.warpAffine(self.dualColorMinorSideImage, self.dualColorMinorToMainMat[:2,:], (self.dualColorMinorSideImage.shape[1], self.dualColorMinorSideImage.shape[0]))

            if dualcolorViewMode == 'Merged':

                # Combine main and minor side
                combinedImage = np.zeros(shape= (self.dualColorMainSideImage.shape[0], self.dualColorMainSideImage.shape[1], 3), dtype= np.uint8)
                combinedImage[:,:,0] = self.dualColorMainSideImage
                combinedImage[:,:,1] = self.dualColorMinorSideImage

                self.image = combinedImage
            
            else:
                self.image = image
        
        else:
            # If not in dual color mode then simply pass on
            self.image = image
        
        self.imageTimeStamp = imageTimeStamp
        self.imageRetrieveTimeStamp = imageRetrieveTimeStamp

        # Compute live analysis data
        showliveanalysis = self.app.config.getboolean('LiveAnalysis', 'showliveanalysis')
        saveanalysistorecording = self.app.config.getboolean('LiveAnalysis', 'saveanalysistorecording')
        if showliveanalysis or saveanalysistorecording:
            self.computeLiveAnalysisValues()
    
    
    def computeLiveAnalysisValues(self):
        """Compute the following values from the current image
            - min
            - max
            - mean
            - skew
            - 5%, 95% percentiles
        """
        # Get current image
        dualcolorMode = self.app.config.getboolean('DualColor', 'dualcolormode')
        regionmode = self.app.config.get('LiveAnalysis', 'regionmode')

        image = self.image if not dualcolorMode else self.dualColorMainSideImage

        if regionmode == 'Tracking':
            # Crop to only the tracking region

            # Get tracking configs
            capture_radius = self.app.config.getint('Tracking', 'capture_radius')
            
            # Crop to tracking region
            image = macro.cropCenterImage(image, capture_radius * 2, capture_radius * 2)

        imageAcquisitionManager: ImageAcquisitionManager = self.parent
        # Do we need to crop on tracking region? 
        imageAcquisitionManager.liveAnalysisData.minBrightness = np.min(image, axis= None)
        imageAcquisitionManager.liveAnalysisData.maxBrightness = np.max(image, axis= None)
        imageAcquisitionManager.liveAnalysisData.meanBrightness = np.mean(image, axis= None)
        imageAcquisitionManager.liveAnalysisData.medianBrightness = np.median(image, axis= None)
        imageAcquisitionManager.liveAnalysisData.skewness = skew(image, axis= None, nan_policy= 'omit')
        imageAcquisitionManager.liveAnalysisData.percentile_5 = np.percentile(image, q= 5, axis= None)
        imageAcquisitionManager.liveAnalysisData.percentile_95 = np.percentile(image, q= 95, axis= None)
    

    def receiveImageCallback(self) -> None:
        """Callback after processed the image. Use for further updating.
        Update the parent (ImageAcquisitionManager) current image data.
        Can be extended.
        """    

        # Update parent (ImageAcquisitionManager) images
        self.parent.image = self.image
        self.parent.imageTimeStamp = self.imageTimeStamp
        self.parent.imageRetrieveTimeStamp = self.imageRetrieveTimeStamp
        self.parent.dualColorMainSideImage = self.dualColorMainSideImage
        # Update display frame value
        self.runtimeControls.framecounter.value += 1
        # Update live analysis data
        self.app.root.ids.middlecolumn.ids.liveanalysislabel.updateText(self.parent.liveAnalysisData)


    def finishAcquisitionCallback(self) -> None:
        """Finished the acquisition looping callback. Needs to be overridden.
        """        
        pass


    def updateDisplayImage(self, dt) -> None:
        """Update the app display image.

        Args:
            dt (float): addition delta time between each callback.
        """        
        self.app.image = self.image


class LiveViewButton(ImageAcquisitionButton):
    """A LiveView button that have image acquisition capability.
    """    
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @override
    def startImageAcquisition(self) -> None:
        """Start the image acquisition process by getting the grabbing parameters, spawn 
        image acquisition thread, and update the image GUI overlay.
        """        
        
        # Update the self-hold reference to the GlowTrackerApp object and the pylon camera object for each of access.
        self.app: GlowTrackerApp = App.get_running_app()
        self.camera = self.app.camera
        self.runtimeControls = App.get_running_app().root.ids.middlecolumn.runtimecontrols

        if self.camera is None:
            self.state = 'normal'
            return
        
        # Setup image acquisition thread parameters
        grabArgs = basler.CameraGrabParameters(
            bufferSize= 16,
            isContinuous= True,
            numberOfImagesToGrab= 1,
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
    

    @override
    def stopImageAcquisition(self) -> None:

        super().stopImageAcquisition()

        print('Stop live view')


    @override
    def acquisitionCondition(self) -> bool:

        return self.camera is not None \
            and (self.camera.IsGrabbing() or self.camera.isOnHold()) \
            and self.state == 'down'


class RecordButton(ImageAcquisitionButton):
    """A Record button that have image acquisition capability.
    """    

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        
        # Declare class instance attributes
        self.numberRecordframes: int = 0
        self.isContinuous: bool = False
        self.frameCounter: int = 0
        self.saveFilePath: str = ''
        self.coordinateFile: TextIOWrapper | None = None
        self.savingthread: Thread | None = None
        self.imageQueue: Queue | None = None
        self.isDualColorMode: bool = False
        self.dualColorRecordingMode: str = ''
        self.imageFilenameFormat: str = ''
        self.imageFilenameExtension: str = ''
        self.prevLiveViewButtonState: str = ''
        self.prevLiveAnalysisButtonState: str = ''
    

    @override
    def startImageAcquisition(self) -> None:
        """Start the image acquisition process:
            - getting the grabbing parameters.
            - spawn image acquisition thread.
            - spawn image saving thread.
            - update the image GUI overlay.
        """ 

        # Update the self-hold reference to the GlowTrackerApp object and the pylon camera object for each of access.
        self.app: GlowTrackerApp = App.get_running_app()
        self.camera = self.app.camera
        self.runtimeControls = App.get_running_app().root.ids.middlecolumn.runtimecontrols

        if self.camera is None:
            self.state = 'normal'
            return

        # Store relevent states
        imageAcquisitionManager: ImageAcquisitionManager = self.parent
        
        # Stop camera if already running and disable the LiveView button
        # If the camera is already running it in live view button,
        #   put the transition "OnHold" flag, and restart camera in Recording mode
        self.prevLiveViewButtonState = imageAcquisitionManager.liveviewbutton.state

        if self.prevLiveViewButtonState == 'down':
            self.camera.setIsOnHold(True)
            imageAcquisitionManager.liveviewbutton.stopImageAcquisition()

        # Disable LiveView
        imageAcquisitionManager.liveviewbutton.disabled = True

        # Store previous Live Analysis state and set to enable
        liveanalysisquickbutton: LiveAnalysisQuickButton = self.app.root.ids.middlecolumn.ids.runtimecontrols.ids.liveanalysisquickbutton

        self.prevLiveAnalysisButtonState = liveanalysisquickbutton.state
        if liveanalysisquickbutton.state == 'normal':
            liveanalysisquickbutton.state = 'down'

        self.runtimeControls.framecounter.value = 0

        self.saveFilePath = self.app.root.ids.leftcolumn.savefile
        self.isDualColorMode = self.app.config.getboolean('DualColor', 'dualcolormode')
        self.dualColorRecordingMode = self.app.config.get('DualColor', 'recordingmode')

        # Image data queue to share between recording and saving
        self.imageQueue = Queue()

        # Start a thread for saving images
        self.savingthread = Thread(target= macro.ImageSaver.startSavingImageInQueueThread, args= [self.imageQueue, 3])
        self.savingthread.start()

        # Setup image acquisition thread parameters
        self.initRecordingParams()
        self.frameCounter = 0

        grabArgs = basler.CameraGrabParameters(
            bufferSize= self.app.config.getint('Experiment', 'buffersize'),
            isContinuous= self.isContinuous,
            numberOfImagesToGrab= self.numberRecordframes,
            grabStrategy= pylon.GrabStrategy_OneByOne
        )

        # open coordinate file
        self.coordinateFile = self.initCoordinateFile()

        # Spawn image acquisition thread
        self.imageAcquisitionThread = Thread(
            target= self.imageAcquisitionLoopingThread,
            daemon= True,
            kwargs= {
                'grabArgs' : grabArgs,
            }
        )

        self.imageAcquisitionThread.start()


    def initCoordinateFile(self) -> TextIOWrapper:
        """Create a coordinate file and write the relevent recording settings into header.

        Returns:
            TextIOWrapper: the coordinate file handler
        """
        coordinateFile = open(os.path.join(self.saveFilePath, timeStamped("coords.txt")), 'a')

        # Recording
        coordinateFile.write(f'# Recording\n')
        #   duration
        duration = self.app.config.getfloat('Experiment', 'duration')
        coordinateFile.write(f'duration {duration}\n')
        #   frames
        nframes = self.app.config.getint('Experiment', 'nframes')
        coordinateFile.write(f'nframes {nframes}\n')

        # Camera 
        coordinateFile.write(f'# Camera\n')
        #   framerate
        framerate = self.camera.ResultingFrameRate()
        coordinateFile.write(f'framerate {framerate}\n')
        #   camera-stage transformation
        imageToStageMat_XYCoord = macro.swapMatXYOrder(self.app.imageToStageMat)
        coordinateFile.write(f'imageToStage {imageToStageMat_XYCoord[0,0]},{imageToStageMat_XYCoord[0,1]},{imageToStageMat_XYCoord[1,0]},{imageToStageMat_XYCoord[1,1]}\n')
        #   rotation
        rotation = self.app.config.getfloat('Camera', 'rotation')
        coordinateFile.write(f'rotation {rotation}\n')
        #   imagenormaldir
        imagenormaldir = self.app.config.get('Camera', 'imagenormaldir')
        coordinateFile.write(f'imagenormaldir {imagenormaldir}\n')
        #   pixelsize
        pixelsize = self.app.config.getfloat('Camera', 'pixelsize')
        coordinateFile.write(f'pixelsize {pixelsize}\n')

        #  Dual color
        coordinateFile.write(f'# Dual color\n')
        #   dualcolormode
        dualcolormode = self.app.config.getboolean('DualColor', 'dualcolormode')
        coordinateFile.write(f'dualcolormode {dualcolormode}\n')
        #   mainside
        mainside = self.app.config.get('DualColor', 'mainside')
        coordinateFile.write(f'mainside {mainside}\n')
        #   viewmode
        viewmode = self.app.config.get('DualColor', 'viewmode')
        coordinateFile.write(f'viewmode {viewmode}\n')
        #   recordingmode
        recordingmode = self.app.config.get('DualColor', 'recordingmode')
        coordinateFile.write(f'recordingmode {recordingmode}\n')
        #   translation_x
        translation_x = self.app.config.getfloat('DualColor', 'translation_x')
        coordinateFile.write(f'translation_x {translation_x}\n')
        #   translation_y
        translation_y = self.app.config.getfloat('DualColor', 'translation_y')
        coordinateFile.write(f'translation_y {translation_y}\n')
        #   rotation
        rotation = self.app.config.getfloat('DualColor', 'rotation')
        coordinateFile.write(f'rotation {rotation}\n')

        # Tracking
        coordinateFile.write(f'# Tracking\n')
        #   roi_x
        roi_x = self.app.config.getint('Tracking', 'roi_x')
        coordinateFile.write(f'roi_x {roi_x}\n')
        #   roi_y
        roi_y = self.app.config.getint('Tracking', 'roi_y')
        coordinateFile.write(f'roi_y {roi_y}\n')
        #   capture_radius
        capture_radius = self.app.config.getint('Tracking', 'capture_radius')
        coordinateFile.write(f'capture_radius {capture_radius}\n')
        #   min_step
        min_step = self.app.config.getint('Tracking', 'min_step')
        coordinateFile.write(f'min_step {min_step}\n')
        #   threshold
        threshold = self.app.config.getint('Tracking', 'threshold')
        coordinateFile.write(f'threshold {threshold}\n')
        #   binning
        binning = self.app.config.getint('Tracking', 'binning')
        coordinateFile.write(f'binning {binning}\n')
        #   dark_bg
        dark_bg = self.app.config.getboolean('Tracking', 'dark_bg')
        coordinateFile.write(f'dark_bg {dark_bg}\n')
        #   mode
        mode = self.app.config.get('Tracking', 'mode')
        coordinateFile.write(f'mode {mode}\n')
        #   area
        area = self.app.config.getint('Tracking', 'area')
        coordinateFile.write(f'area {area}\n')
        
        # Write recording header
        coordinateFile.write(f"# Frame Time X Y Z minBrightness maxBrightness meanBrightness medianBrightness skewness percentile_5 percentile_95\n")

        return coordinateFile

    
    @override
    def stopImageAcquisition(self) -> None:
        """Extend the stop image acquisition functionality: 
            - Stop the camera
            - Closing the coordinate file.
            - Closing the image saving thread.
            - Update display texts.
            - Un-disabled (enable if) the LiveView button
        """        

        if self.camera is None:
            return
        
        # If the live view button was previously running, 
        #   then set the transitioning "OnHold" flag.
        if self.prevLiveViewButtonState == 'down':
            self.camera.setIsOnHold(True)

        # Stop the camera and clear values
        super().stopImageAcquisition()

        # Schedule closing coordinate file a bit later
        Clock.schedule_once(lambda dt: self.coordinateFile.close(), 0.5)
        
        # Close saving threads
        if self.savingthread:
            self.imageQueue.put(None)
            self.savingthread.join()

        print("Stop recording")

        # Set LiveView button state back to enable.
        #   We need to do it here as a schedule event. Because this current function (stopImageAcquisition)
        #   is usually called inside a thread, if we call update LiveView button state, it would somehow
        #   invoke the LiveViewButton.on_state() inside this current thread, which can spawn more thread
        #   within itself meaning we will have a thread spawning another thread and is not the behavior
        #   that we want. The most likely reason why this happened is because the call to on_state is happened
        #   within the same Kivy render timeframe as this thread. By calling it through Clock.schedule_once,
        #   we essentially schedule the on_state to be call in the next Kivy render timeframe, ensuring that
        #   it is not invoked from a thread but from the main thread always.
        def resumeButtonsState(*args):
            # LiveView
            self.parent.liveviewbutton.disabled = False
            self.parent.liveviewbutton.state = self.prevLiveViewButtonState
            # LiveAnalysis
            liveanalysisquickbutton: LiveAnalysisQuickButton = self.app.root.ids.middlecolumn.ids.runtimecontrols.ids.liveanalysisquickbutton
            liveanalysisquickbutton.state = self.prevLiveAnalysisButtonState
        
        Clock.schedule_once( resumeButtonsState )
    

    @override
    def acquisitionCondition(self) -> bool:

        return self.camera is not None \
            and (self.camera.IsGrabbing() or self.camera.isOnHold()) \
            and self.isContinuous or (self.frameCounter < self.numberRecordframes) \
            and self.state == 'down'

    
    @override
    def processImageCallback(self, image, imageTimeStamp, imageRetrieveTimeStamp) -> None:

        super().processImageCallback(image, imageTimeStamp, imageRetrieveTimeStamp)

        # Additionaly, we need to take care of computing the Live Analysis values, even if the:
        #   - LiveAnalysisButton state is normal
        #   - showliveanalysis flag in the Settings is False.
        # In these cases, the base method would not have computed the Live Analysis values, so we just simply call it.
        showliveanalysis = self.app.config.getboolean('LiveAnalysis', 'showliveanalysis')
        if not showliveanalysis:
            self.computeLiveAnalysisValues()

    
    @override
    def receiveImageCallback(self) -> None:
        """Extended to further:
            - Save the coordinate data.
            - Put the image into an image saving queue.
        """

        # Write coordinate into file.
        try:
            self.coordinateFile.write(f"{self.frameCounter} \
{self.imageTimeStamp} \
{self.app.coords[0]} \
{self.app.coords[1]} \
{self.app.coords[2]} \
{self.parent.liveAnalysisData.minBrightness} \
{self.parent.liveAnalysisData.maxBrightness} \
{self.parent.liveAnalysisData.meanBrightness} \
{self.parent.liveAnalysisData.medianBrightness} \
{self.parent.liveAnalysisData.skewness} \
{self.parent.liveAnalysisData.percentile_5} \
{self.parent.liveAnalysisData.percentile_95} \n")

        #   Handle error from writing the file, such as ValueError: I/O operation on closed file.
        except ValueError as e:
            print(f'Error writing coordinateFile: {e}')

        # Put image(s) into the saving queue
        if not self.isDualColorMode or ( self.isDualColorMode and self.dualColorRecordingMode == 'Original' ):
            # Put the full image
            self.imageQueue.put([
                np.copy(self.image),
                self.saveFilePath,
                self.imageFilenameFormat.format(self.frameCounter)
            ])

        elif self.isDualColorMode and self.dualColorRecordingMode == 'Splitted':
            # Put the dual color main and minor images
            mainImageFileName = self.imageFilenameFormat.format(self.frameCounter)
            minorImageFileName = str(mainImageFileName)

            extensionLen = len(self.imageFilenameExtension)

            mainImageFileName = mainImageFileName[:-(extensionLen+1)] + '-main.' + self.imageFilenameExtension
            minorImageFileName = minorImageFileName[:-(extensionLen+1)] + '-minor.' + self.imageFilenameExtension

            mainImageFileName = mainImageFileName[:]
            self.imageQueue.put([
                np.copy(self.dualColorMainSideImage),
                self.saveFilePath,
                mainImageFileName
            ])
            self.imageQueue.put([
                np.copy(self.dualColorMinorSideImage),
                self.saveFilePath,
                minorImageFileName
            ])

        self.frameCounter += 1

        # If this is the last recording frame,
        #   the camera is going to shut itself off.
        #   Thus, we have to set the onHold transition flag if we're going to
        #   resume back in live view mode.
        if self.frameCounter == self.numberRecordframes \
            and not self.isContinuous \
            and self.prevLiveViewButtonState == 'down':
                
            self.camera.setIsOnHold(True)

        super().receiveImageCallback()
    

    @override
    def finishAcquisitionCallback(self) -> None:
        """Send stop signal to image saving threads and stop image acquisition.
        """        
        # Send signal to terminate recording workers
        self.imageQueue.put(None)

        print(f'Recorded {self.numberRecordframes} frames.')

        # Call to recording-stopping procedure
        self.stopImageAcquisition()
    

    def initRecordingParams(self):
        """Initialize the recording arguments
        """        
        # Setup grabbing with recording settings
        self.numberRecordframes = self.app.config.getint('Experiment', 'nframes')

        self.isContinuous = self.app.config.getboolean('Experiment', 'iscontinuous')

        # Get desired FPS from UI
        fps = self.app.root.ids.leftcolumn.ids.camprops.framerate
        print("Desired recording Framerate:", fps)

        # Get actual FPS from Camera
        fps = self.app.camera.setFramerate(fps)
        print('Actual recording fps: ' + str(fps))

        # Update shown display settings, e.g. exposure, fps, gain values
        self.app.root.ids.leftcolumn.update_settings_display()

        # pre-calculate the filename
        self.imageFilenameExtension = self.app.config.get('Experiment', 'extension')
        self.imageFilenameFormat = timeStamped("basler_{}."+f"{self.imageFilenameExtension}")
        

class ImageAcquisitionManager(BoxLayout):
    """An ImageAcquisition buttons holder widget. This class acts as a centralized contact
    point for accessing the acquired images.
    """    
    recordbutton: RecordButton = ObjectProperty(None, rebind = True)
    liveviewbutton: LiveViewButton = ObjectProperty(None, rebind = True)
    snapbutton: Button = ObjectProperty(None, rebind = True)
    # Class' attributes for centralized access of acquired images
    image: np.ndarray = np.zeros((1,1))
    imageTimeStamp: float = 0
    imageRetrieveTimeStamp: float = 0
    dualColorMainSideImage: np.ndarray = np.zeros((1,1))
    liveAnalysisData: LiveAnalysisData = LiveAnalysisData()


    def __init__(self,  **kwargs):
        super(ImageAcquisitionManager, self).__init__(**kwargs)

        self.app: GlowTrackerApp = App.get_running_app()


    def snap(self) -> None:
        """Callback for saving a single image from the Snap button.
        """
        ext = self.app.config.get('Experiment', 'extension')
        path = self.app.root.ids.leftcolumn.savefile
        snap_filename = timeStamped("snap."+f"{ext}")
        camera = self.app.camera
        
        if camera is None:
            return
        
        # Get an image appropriately acoording to current viewing mode
        if self.recordbutton.state == 'down' or self.liveviewbutton.state == 'down':
            #   save the current image
            basler.saveImage(self.image, path, snap_filename)

        else:
            # Call capture an image
            isSuccess, img = camera.singleTake()

            if isSuccess:
                basler.saveImage(img, path, snap_filename)

            else:
                print('An error occured when taking an image')


class StencilFloatLayout(FloatLayout, StencilView):

    def on_touch_down(self, touch):
        """Limits subsequent interactions to only be activated if it's within the StencilFloatLayout
        """

        if self.collide_point(*touch.pos):
            return super().on_touch_down(touch)
        else:
            return False
    
    def on_touch_up(self, touch):
        """Limits subsequent interactions to only be activated if it's within the StencilFloatLayout
        """
        if self.collide_point(*touch.pos):
            return super().on_touch_up(touch)
        else:
            return False

        
class ScalableImage(ScatterLayout):

    def on_touch_up(self, touch):
        
        # If the widget is enabled and interaction point is inside its bounding box
        if self.disabled or not self.collide_point(*touch.pos):
            return False

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
    circle = ListProperty([0, 0, 0])

    def __init__(self,  **kwargs):
        super(PreviewImage, self).__init__(**kwargs)
        Window.bind(mouse_pos=self.mouse_pos)

        self.mouse_pos_in_image_space: np.array = np.zeros((2,))
        self.mouse_pos_in_tex_coord: np.array = np.zeros((2,))

    def mouse_pos(self, window, pos):
        """Calculate relative mouse position to the preview image and update the
        inspect pixel value text at the bottom right corner of the GUI.
        """        
        if not hasattr(self, 'app'):
            self.app = App.get_running_app()
        
        image: np.ndarray = self.app.image
        
        if image is None:
            return
        
        mouse_pos = np.array(pos, np.float32)

        # Scale mouse position upto the display density factor.
        #   This is usually 1 for normal monitor. 
        #   But for higher density monitors like in modern laptop
        #   or smartphone, this factor will be more than 1.
        #   This is important because it affect the coordinate system down the line.
        mouse_pos *= Metrics.dp

        previewImage = self
        scalableImage = self.app.root.ids.middlecolumn.ids.scalableimage    # parent of the previewImage

        # Compute relative position in the scalableImage
        pos_in_scalableImage = scalableImage.to_local(mouse_pos[0], mouse_pos[1], relative= True)

        if not self.collide_point( pos_in_scalableImage[0], pos_in_scalableImage[1] ):
            return

        # Compute relative position in the image
        padding_x = (previewImage.size[0] - self.norm_image_size[0])/2
        padding_y = (previewImage.size[1] - self.norm_image_size[1])/2

        self.mouse_pos_in_image_space = pos_in_scalableImage - np.array([padding_x, padding_y])

        # Check if within the image bbox
        if 0 <= self.mouse_pos_in_image_space[0] <= self.norm_image_size[0] \
            and 0 <= self.mouse_pos_in_image_space[1] <= self.norm_image_size[1]:

            # Compute texture coordinate
            self.mouse_pos_in_tex_coord = self.mouse_pos_in_image_space / self.norm_image_size
            self.mouse_pos_in_tex_coord[0] *= image.shape[1]
            self.mouse_pos_in_tex_coord[1] *= image.shape[0]
            self.mouse_pos_in_tex_coord = np.floor(self.mouse_pos_in_tex_coord).astype(np.int32)

            # Get the pixel value. Need to flip Y as image coord is top-left.
            pixelVal = image[(image.shape[0] - 1) - self.mouse_pos_in_tex_coord[1], self.mouse_pos_in_tex_coord[0]]
            # Update info text
            self.app.root.ids.middlecolumn.ids.pixelvalue.text = f'({self.mouse_pos_in_tex_coord[0]}, {self.mouse_pos_in_tex_coord[1]}, {pixelVal})'

        return  


    def captureCircle(self, pos):
        """define the capture circle and draw it."""
        wx, wy = pos#self.to_widget(pos[0], pos[1])#, relative = True)
        image = App.get_running_app().image
        h, w = image.shape[0], image.shape[1]
        # paint a circle and make the coordinates available
        radius = App.get_running_app().config.getfloat('Tracking', 'capture_radius')
        # make the circle into pixel units
        r = radius/w*self.norm_image_size[0]#, radius/h*self.norm_image_size[1]
        self.circle = (*pos, r)


    def clearcircle(self):
        self.circle = (0, 0, 0)


    def on_touch_down(self, touch):
        runtimeControls = App.get_running_app().root.ids.middlecolumn.runtimecontrols
        # If a click happens in this widget
        # and tracking is active and not yet scheduled:
        if self.collide_point(*touch.pos) \
            and runtimeControls.trackingcheckbox.state == 'down' \
            and not runtimeControls.isTracking:

            # Check if within the image bbox
            if 0 <= self.mouse_pos_in_image_space[0] <= self.norm_image_size[0] \
                and 0 <= self.mouse_pos_in_image_space[1] <= self.norm_image_size[1]:

                print('Start tracking process')
                # Draw a red circle
                self.captureCircle(touch.pos)

                # Move stage to the starting position

                # Start tracking procedure
                Clock.schedule_once(lambda dt: runtimeControls.startTracking(self.mouse_pos_in_tex_coord), 0)
                
                # remove the circle 
                # Clock.schedule_once((lambda dt: self.circle = (0, 0, 0)), 0.5)
                Clock.schedule_once(lambda dt: self.clearcircle(), 0.5)


class LiveAnalysisLabel(Label):
    
    def __init__(self, **kwargs):
        super(LiveAnalysisLabel, self).__init__(**kwargs)
        self.updateText(LiveAnalysisData())


    def updateText(self, liveAnalysisData: LiveAnalysisData):

        # Get LiveAnalysisData from ImageAcquisition
        app: GlowTrackerApp = App.get_running_app()
        
        self.text = f"""Min: {liveAnalysisData.minBrightness:.2f}
Max: {liveAnalysisData.maxBrightness:.2f}
Mean: {liveAnalysisData.meanBrightness:.2f}
Median: {liveAnalysisData.medianBrightness:.2f}
Skewness: {liveAnalysisData.skewness:.2f}
5 percentile: {liveAnalysisData.percentile_5:.2f}
95 percentile: {liveAnalysisData.percentile_95:.2f}"""


class ImageOverlay(FloatLayout):
    """An image overlay class than handles drawing of GUI overlays ontop of the image.
    """    
    
    def __init__(self,  **kwargs):
        super(ImageOverlay, self).__init__(**kwargs)
        # Declare class instance's attributes
        self.hasDrawDualColorOverlay: bool = False
        self.label: Label | None = None

        self.trackingMaskLayout: FloatLayout | None = None
        self.trackingMask = Image()

        self.trackingBorder: Line | None = None
        self.cmsShape: Ellipse | None = None

        self.app = App.get_running_app()


    def resizeToImage(self) -> None:
        """Resize and move the overlay to match the display image exactly
        """
        previewImage: PreviewImage = self.app.root.ids.middlecolumn.previewimage

        # Set the overlay size as the image size
        normImageSize = previewImage.get_norm_image_size()
        self.size = normImageSize

        # Set the overlay position to match the image position exactly.
        #   Note, this is a local position.
        imageWidgetSize = previewImage.size
        self.pos[0] = (imageWidgetSize[0] - normImageSize[0]) / 2
        self.pos[1] = (imageWidgetSize[1] - normImageSize[1]) / 2
        
    
    def on_size(self, *args) -> None:
        """Called everytime the widget is resized. Resize the overlay to match the image and redraw.
        """        
        self.updateOverlay()
    

    @mainthread
    def updateOverlay(self) -> None:
        """Clear and redraw the overlay depending on the app config.
            1. Resize to match the image
            2. Clear all the overlay
            3. Redraw all the overlay
        """
        
        # If the app has just started with a logo then don't draw any overlay
        if self.app.image is None:
            return
        
        # Resize the overlay to match the image
        self.resizeToImage()

        # Clear all the overlay
        self.clearOverlay()

        # Update dual color overlay
        dualcolormode = self.app.config.getboolean('DualColor', 'dualcolormode')
        
        if dualcolormode:
            mainside = self.app.config.get('DualColor', 'mainside')
            self.drawDualColorOverlay(mainside)
                
        # Update tracking overlay
        showtrackingoverlay = self.app.config.getboolean('Tracking', 'showtrackingoverlay')
        
        if showtrackingoverlay:
            self.updateTrackingOverlay(doClear= False)

    
    def updateTrackingOverlay(self, doClear: bool = True):
        """Gather tracking overlay data and draw.

        Args:
            doClear (bool, optional): Clear the tracing overlay first before draw. Defaults to True.
        """

        cmsOffset_x, cmsOffset_y = 0, 0
        trackingMask = np.zeros(0)

        rtc: RuntimeControls = self.app.root.ids.middlecolumn.runtimecontrols
        if rtc.isTracking:
            # If tracking, the get the tracking data from RuntimeControls
            cmsOffset_x, cmsOffset_y, trackingMask = rtc.cmsOffset_x, rtc.cmsOffset_y, rtc.trackingMask

        else:
            # If not tracking, then we have to compute the tracking overlay data first
            cmsOffset_x, cmsOffset_y, trackingMask = rtc.computeTrackingCMS()

        if doClear:
            self.clearTrackingOverlay()

        self.drawTrackingOverlay(cmsOffset_x, cmsOffset_y, trackingMask)
    
    
    def computeTrackingOverlayBorderBBox(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the tracking overlay bounding box in the local widget space.

        Returns:
            center [np.ndarray]: center of the overlay in the local widget space
            btm_left [np.ndarray]: btm left corner of the overlay in the local widget space
            top_right [np.ndarray]: top right corner of the overlay in the local widget space
        """
        # Compute display scaling
        previewImage: PreviewImage = self.app.root.ids.middlecolumn.previewimage
        normImageSize = np.array(previewImage.get_norm_image_size())

        # Compute overlay center position
        center = self.to_local(self.center_x, self.center_y)
        center = np.array(center)

        dualColorMode = self.app.config.getboolean('DualColor', 'dualcolormode')
        dualColorViewMode = self.app.config.get('DualColor', 'viewmode')

        # If we are using the dual color and viewing the 'Splitted' mode, 
        #   then we have to shift the center of tracking border to the left ro right 
        #   side accordingly.
        if dualColorMode and dualColorViewMode == 'Splitted':
            mainSide = self.app.config.get('DualColor', 'mainside')
            
            if mainSide == 'Left':
                center[0] -= normImageSize[0]/4

            elif mainSide == 'Right':
                center[0] += normImageSize[0]/4
        
        # Compute the overlay bbox
        imageSize = previewImage.texture_size
        displayedScale = normImageSize[0] / imageSize[0]
        radius = self.app.config.getint('Tracking', 'capture_radius') * displayedScale
        btm_left = center - radius
        top_right = center + radius

        # Compute the bbox of the image in the overlay space
        image_btm_left = np.copy(center)
        image_top_right = np.copy(center)
        if dualColorMode and dualColorViewMode == 'Splitted':
            image_btm_left[0] -= normImageSize[0]/4
            image_btm_left[1] -= normImageSize[1]/2
            image_top_right[0] += normImageSize[0]/4
            image_top_right[1] += normImageSize[1]/2

        else:
            image_btm_left -= normImageSize/2
            image_top_right += normImageSize/2

        # Set the upper bound of the bbox to the image size
        btm_left = np.fmax(btm_left, image_btm_left)
        top_right = np.fmin(top_right, image_top_right)

        return center, btm_left, top_right
    
    
    def drawTrackingOverlay(self, cmsOffset_x: float | None = None, cmsOffset_y: float | None = None, trackingMask: np.ndarray | None = None) -> None:
        """Draw the tracking info overlay.
            1. Draw the tracking mask if provided
            2. Draw the tracking border
            3. Draw the tracking center of mass if provided

        Args:
            cmsOffset_x (float | None, optional): center of mass position as an ofset from the center of the image. Defaults to None.
            cmsOffset_y (float | None, optional): center of mass position as an ofset from the center of the image. Defaults to None.
            trackingMask (np.ndarray | None, optional): 2D uint8 numpy array representing the mask that is used for calculating the center of mass. Defaults to None.
        """
        
        # Frequently used 
        center, btm_left, top_right = self.computeTrackingOverlayBorderBBox()

        # 
        # Check if needs to draw tracking mask
        # 
        if trackingMask is not None:
            
            if self.trackingMaskLayout is None:

                # Create a FloatLayout
                self.trackingMaskLayout = FloatLayout()

                #   Set the position and size to fit the overlay
                self.trackingMaskLayout.pos = btm_left.tolist()
                self.trackingMaskLayout.size = (top_right - btm_left).tolist()

                # Add base FloatLayout to self
                self.add_widget(self.trackingMaskLayout)

                # Add the Image widget
                self.trackingMaskLayout.add_widget(self.trackingMask)
            
            if self.trackingMask.texture is None \
                or self.trackingMask.texture.width != trackingMask.shape[1] \
                or self.trackingMask.texture.height != trackingMask.shape[0]:

                # Create Texture
                self.trackingMask.texture = Texture.create(
                    size= (trackingMask.shape[1], trackingMask.shape[0]),
                    colorfmt= 'rgba'
                )
                # Kivy texture is in OpenGL corrindate which is btm-left origin so we need to flip texture coord once to match numpy's top-left
                self.trackingMask.texture.flip_vertical()

                # Set fit mode to fill so that it up-/down-scale to fit the trackingMask widget perfectly
                self.trackingMask.fit_mode = 'fill'

                # Unbind size callback from the parent.Very important!
                self.trackingMask.size_hint = (None, None)
                self.trackingMask.opacity = 0.5

                # Set the position and size to fit the overlay
                self.trackingMask.pos = btm_left.tolist()
                self.trackingMask.size = (top_right - btm_left).tolist()

            # Convert from grayscale to rgb and move to blue channel
            trackingMaskColor = np.zeros((trackingMask.shape[0], trackingMask.shape[1], 4), np.uint8)
            trackingMaskColor[:,:,2] = trackingMask
            trackingMaskColor[:,:,3][trackingMask>0] = 255  # alpha mask

            # Upload image data to texture
            imageByteBuffer: bytes = trackingMaskColor.tobytes()
            self.trackingMask.texture.blit_buffer(imageByteBuffer, colorfmt= 'rgba', bufferfmt= 'ubyte')

        # 
        # Check if needs to reconstruct the tracking border
        # 
        if self.trackingBorder is None:

            trackingBorderPoints = [
                btm_left[0], btm_left[1],
                btm_left[0], top_right[1],
                top_right[0], top_right[1],
                top_right[0], btm_left[1]
            ]

            # Construct the tracking border draw command
            self.trackingBorder = Line(points= trackingBorderPoints, width= 1, cap= 'none', joint= 'round', close= 'true')

            # Draw the tracking border as a red rectangle
            self.canvas.add(Color(1., 0., 0., 0.5))
            self.canvas.add(self.trackingBorder)

        # 
        # Draw tracking center of mass if provided
        # 
        if cmsOffset_x is not None and cmsOffset_y is not None:
            
            # Compute scaling
            previewImage: PreviewImage = self.app.root.ids.middlecolumn.previewimage
            normImageSize = np.array(previewImage.get_norm_image_size())
            imageSize = previewImage.texture_size
            displayedScale = normImageSize[0] / imageSize[0]
            
            # Compute cms draw position
            cms = center + np.array([cmsOffset_x, cmsOffset_y]) * displayedScale 

            pointRadius = 10 * displayedScale

            if self.cmsShape is None:
                # If the tracking shape is not yet created, create it and draw
                self.cmsShape = Ellipse(
                    pos= (cms[0] - pointRadius, cms[1] - pointRadius), 
                    size= (pointRadius * 2, pointRadius * 2)
                )

                # Draw the cms as a teal dot
                self.canvas.add(Color(0.435, 0.957, 1.0, 0.75))
                self.canvas.add(self.cmsShape)
            
            else:
                # Else just update the position
                self.cmsShape.pos = (cms[0] - pointRadius, cms[1] - pointRadius)


    def clearTrackingOverlay(self):
        """Clear the tracking info overlay
        """
        
        if self.trackingMaskLayout is not None:
            self.remove_widget(self.trackingMaskLayout)
            self.trackingMaskLayout.clear_widgets()
            self.trackingMaskLayout = None
            
        if self.trackingMask is not None:
            self.trackingMask.texture = None
            self.remove_widget(self.trackingMask)
        
        if self.trackingBorder is not None:
            self.canvas.remove(self.trackingBorder)
            self.trackingBorder = None

        if self.cmsShape is not None:
            self.canvas.remove(self.cmsShape)
            self.cmsShape = None
        

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

        previewImage: PreviewImage = self.app.root.ids.middlecolumn.previewimage

        viewMode = self.app.config.get('DualColor', 'viewmode')

        if viewMode == 'Splitted':

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
                self.label = Label(text= '[color=8e0045]Main[/color]', markup= True)  
                self.label.size_hint = [None, None]
                self.label.valign = 'top'
                self.label.halign = 'left'
                self.label.texture_update()
                self.add_widget(self.label)
            else:
                # In this case, the self.canvas.clear() has been called so we have to redraw the label.
                #   Ideally, we would like to call self.canvas.add( some label draw instruction ) but I can't find it
                #   so we will mimick this by re-adding it again.
                self.remove_widget(self.label)
                self.add_widget(self.label)
            
            self.label.size = self.label.texture_size

            # Compute label position
            normImageSize = previewImage.get_norm_image_size()
            labelDisplayedSize = np.array(self.label.texture_size) 
            
            labelOffset_x = pos_center_local[0] - labelDisplayedSize[0]/2
            if mainSide == 'Left':
                labelOffset_x -= normImageSize[0]/4

            elif mainSide == 'Right':
                labelOffset_x += normImageSize[0]/4
            
            #   Compute position at the top
            labelOffset_y = pos_center_local[1] + normImageSize[1]/2 - labelDisplayedSize[1]
            #   Further adjust to look prettier
            labelOffset_y -= labelDisplayedSize[1] * 0.75

            self.label.pos = [float(labelOffset_x), float(labelOffset_y)]

        
        elif viewMode == 'Merged':

            # 
            # Label on the header
            # 
            if self.label is None:
                # Create a Label and add it as a child
                self.label = Label(text= '[color=8e0045]Dual Color: Merged[/color]', markup= True)
                self.label.size_hint = [None, None]
                self.label.valign = 'top'
                self.label.halign = 'left'
                self.label.texture_update()
                self.add_widget(self.label)
                
            else:
                # In this case, the self.canvas.clear() has been called so we have to redraw the label.
                #   Ideally, we would like to call self.canvas.add( some label draw instruction ) but I can't find it
                #   so we will mimick this by re-adding it again.
                self.remove_widget(self.label)
                self.add_widget(self.label)
            
            self.label.size = self.label.texture_size

            # Compute label position
            normImageSize = previewImage.get_norm_image_size()
            labelDisplayedSize = np.array(self.label.texture_size) 
            
            #   Compute center position
            pos_center_local = self.to_local(self.center_x, self.center_y)
            labelOffset_x = pos_center_local[0] - labelDisplayedSize[0]/2
            
            #   Compute position at the top
            labelOffset_y = pos_center_local[1] + normImageSize[1]/2 - labelDisplayedSize[1]
            #   Further adjust to look prettier
            labelOffset_y -= labelDisplayedSize[1] * 0.75

            self.label.pos = [float(labelOffset_x), float(labelOffset_y)]
        

    def clearDualColorOverlay(self):
        """Clear the canvas and set internal hasDraw flag to false
        """
        self.canvas.clear()
        self.hasDrawDualColorOverlay = False

    
    def clearOverlay(self) -> None:
        """Clear both tracking and dual color overlay.
        """
        self.clearTrackingOverlay()
        self.clearDualColorOverlay()
    

class RuntimeControls(BoxLayout):
    framecounter = ObjectProperty(rebind=True)
    livefocuscheckbox = ObjectProperty(rebind=True)
    trackingcheckbox = ObjectProperty(rebind=True)
    imageacquisitionmanager: ImageAcquisitionManager = ObjectProperty(rebind=True)
    cropX = NumericProperty(0, rebind=True)
    cropY = NumericProperty(0, rebind=True)
    

    def __init__(self,  **kwargs):
        super(RuntimeControls, self).__init__(**kwargs)
        self.focus_history = []
        self.liveFocusThread = None
        self.focus_motion = 0
        self.isTracking = False
        self.coord_updateevent: ClockEvent | None = None
        # Center of Mass offset in current tracking frame
        self.cmsOffset_x: float | None = None
        self.cmsOffset_y: float | None = None
        self.trackingMask: np.ndarray | None = None


    def on_framecounter(self, instance, value):
        self.text = str(value)


    def startLiveFocus(self):
        """Initiate an autofocus thread.
        """
        camera: GlowTrackerApp = App.get_running_app().camera
        stage: Stage = App.get_running_app().stage
        
        # Sanity check
        if camera is not None and stage is not None and camera.IsGrabbing():

            # Load config values
            app: GlowTrackerApp = App.get_running_app()

            focusfps = app.config.getfloat('Autofocus', 'focusfps')

            print("Live focus Framerate:", focusfps)

            KP = app.config.getfloat('Autofocus', 'kp')
            KI = app.config.getfloat('Autofocus', 'ki')
            KD = app.config.getfloat('Autofocus', 'kd')
            SP = app.config.getfloat('Autofocus', 'bestfocusvalue')
            focusEstimationMethod = FocusEstimationMethod(app.config.get('Autofocus', 'focusestimationmethod'))
            dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
            capturedRadius = app.config.getint('Tracking', 'capture_radius')
            isshowgraph = app.config.getboolean('Autofocus', 'isshowgraph')
            depthoffield = app.config.getfloat('Camera', 'depthoffield')
            smoothingwindow = app.config.getint('Autofocus', 'smoothingwindow')
            minstepbeforechangedir = app.config.getint('Autofocus', 'minstepbeforechangedir')
            
            autoFocusPID = AutoFocusPID(
                KP= KP,
                KI= KI,
                KD= KD,
                SP= SP,
                focusEstimationMethod= FocusEstimationMethod(focusEstimationMethod),
                minStepDist= depthoffield,
                acceptableErrorPercentage= 0.05,
                integralLifeTime= 0,
                smoothingWindow= smoothingwindow,
                minStepBeforeChangeDir= minstepbeforechangedir
            )

            # Data handle from LiveFocus thread to plotting in main thread
            graph_x_data = list()
            graph_y_data = list()
            graph_data_lock = Lock()

            if isshowgraph:

                # Create a LiveFocus graph
                axisPlotHandle = None
                linePlotHandle = None

                # Init interactive plot handle
                plt.ion()

                fig, axisPlotHandle = plt.subplots()

                linePlotHandle, = axisPlotHandle.plot([], [], 'r-', label= 'Current Focus')

                bestFocusPlotHandle = axisPlotHandle.axhline(y= SP, color= 'g', linestyle= '-.', label= 'Estimated Maximum Focus')

                axisPlotHandle.set_title("Live Focus")
                axisPlotHandle.set_xlabel("Iteration")
                axisPlotHandle.set_ylabel("Estimated Focus")

                background = fig.canvas.copy_from_bbox(axisPlotHandle.bbox)

                plt.legend()
                plt.show()

                # Event to update the graph 
                def updateLiveFocusGraph( dt: float ):

                    # Empty guard
                    if len(graph_x_data) == 0 and len(graph_y_data) == 0:
                        return
                    
                    with graph_data_lock:
                        linePlotHandle.set_xdata(graph_x_data)
                        linePlotHandle.set_ydata(graph_y_data)

                    axisPlotHandle.set_xlim(min(graph_x_data), max(graph_x_data))
                    axisPlotHandle.set_ylim(min(graph_y_data), max(graph_y_data))

                    # restore background
                    fig.canvas.restore_region(background)

                    # redraw the line plots
                    axisPlotHandle.draw_artist(linePlotHandle)
                    axisPlotHandle.draw_artist(bestFocusPlotHandle)

                    # fill in the axes rectangle
                    fig.canvas.blit(axisPlotHandle.bbox)

                # Lunch an updating event. This has to be executed in the main thread so we can't multithread it.
                self.updateLiveFocusGraphEvent = Clock.schedule_interval(updateLiveFocusGraph, 1.0 / focusfps)

            # Pack args
            autoFocusArgs = autoFocusPID, camera, stage, dualColorMode, capturedRadius, isshowgraph, focusfps, graph_x_data, graph_y_data, graph_data_lock

            # Start the autofocus thread
            self.liveFocusThread = Thread(target= self._liveFocus, args= autoFocusArgs, daemon = True, name= 'LiveFocus')

            self.liveFocusThread.start()


        else:
            self._popup = WarningPopup(title="Autofocus", text='Focus requires: \n - a stage \n - a camera \n - camera needs to be grabbing.',
                            size_hint=(0.5, 0.25))
            self._popup.open()
            self.livefocuscheckbox.state = 'normal'

    
    def _liveFocus(self, autoFocusPID: AutoFocusPID, camera: basler.Camera, stage: Stage, dualColorMode: bool = False, capturedRadius: float = 0, isShowGraph: bool = False, fps: float = 10.0, graph_x_data: List[float] = list(), graph_y_data: List[float] = list(), graph_data_lock: Lock = None) -> None:
        """Autofocus loop to be executed inside a thread.

        Args:
            autoFocusPID (AutoFocusPID): Autofocus object
            camera (basler.Camera): camera
            stage (Stage): stage
            dualColorMode (bool, optional): is the image dual-colored. Defaults to False.
            capturedRadius (float, optional): radius from center of the image to square crop. Defaults to 0 means no cropping.
            isShowGraph (bool, optional): Is the live focus graph enabled. Defaults to False.
            fps (float, optional): Maximum frequency to perform autofocus per second. Defaults to 10.0.
            graph_x_data (List[float], optional): List object to append values in the x-axis to, to be shown on LiveFocus graph. Defaults to empty list().
            graph_y_data (List[float], optional): List object to append values in the y-axis to, to be shwown on LiveFocus graph. Defaults to empty list().
            graph_data_lock (Lock, optional): threading Lock object for modifying graph_data
        """
        app: GlowTrackerApp = App.get_running_app()
        spf = 1.0 / fps
        image = None

        print("Focus, Err, 1st, 2nd, 3rd, dist, new pos")

        # Continuously running as long as these conditions are met.
        while camera is not None and (camera.IsGrabbing() or camera.isOnHold()) and self.livefocuscheckbox.state == 'down':

            startTime = time.perf_counter()

            # Get current image
            if dualColorMode:
                image = self.imageacquisitionmanager.dualColorMainSideImage
            else:
                image = self.imageacquisitionmanager.image

            # Center-crop the image
            croppedImage = macro.cropCenterImage(image, capturedRadius * 2, capturedRadius * 2)

            # Get current position
            pos = app.coords[2]

            # Perform one autofocus step
            relPosZ = autoFocusPID.executePIDStep(croppedImage, pos= pos)

            # Move relative z-position
            stage.move_z(relPosZ, unit='mm', wait_until_idle= False)

            # Update App's internal stage coordinate
            app.coords[2] = app.coords[2] + relPosZ
            
            if isShowGraph:
                # Update live graph data
                with graph_data_lock:
                    graph_x_data.append(len(autoFocusPID.focusLog) - 1)
                    graph_y_data.append(autoFocusPID.focusLog[-1])
            
            endTime = time.perf_counter()

            elapsedTime = endTime - startTime

            waitTime = spf - elapsedTime

            # Wait until matching spf
            if waitTime > 0:
                time.sleep(waitTime)
        
        # The live focus has stopped
        self.livefocuscheckbox.state == 'normal'
    

    def stopLiveFocus(self):
        """Callback to stop LiveFocus mode
        """
        app: GlowTrackerApp = App.get_running_app()
        isshowgraph = app.config.getboolean('Autofocus', 'isshowgraph')
        if isshowgraph:
            Clock.unschedule(self.updateLiveFocusGraphEvent)
            plt.ioff()


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
    

    def startTracking(self, start_pos_tex_coord: np.array) -> None:
        """Start the tracking procedure by gathering variables, setting up the camera, and then spawn a tracking loop.

        Args:
            start_pos_tex_coord (np.array): Starting position in the image texture space (full image size). Used to move the stage to center at that position.
        """        
        app = App.get_running_app()
        stage = app.stage
        units = app.config.get('Calibration', 'step_units')
        minstep = app.config.getfloat('Tracking', 'min_step')
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        
        # 
        # Move stage by the user pointed starting position
        # 

        # Compute the offset from the center
        imageHeight, imageWidth = app.image.shape[0], app.image.shape[1]
        offset_from_center = np.zeros(2, np.float32)
        if dualColorMode:
            # Get the main side
            mainSide = app.config.get('DualColor', 'mainside')
        
            # Compute offset from the center of the main side
            if mainSide == 'Right':
                offset_from_center = start_pos_tex_coord - np.array([imageWidth*3.0/4, imageHeight/2])
                
            elif mainSide == 'Left':
                offset_from_center = start_pos_tex_coord - np.array([imageWidth*1.0/4, imageHeight/2])
            
        else:
            # In normal mode, compute from the image center
            offset_from_center = start_pos_tex_coord - np.array([imageWidth/2, imageHeight/2])
        
            # Set tracking ROI
            roiX, roiY  = app.config.getint('Tracking', 'roi_x'), app.config.getint('Tracking', 'roi_y')
            self.set_ROI(roiX, roiY)

        # Convert from texture coordinates to stage coordinates
        ystep, xstep = macro.getStageDistances(np.array([offset_from_center[1], offset_from_center[0]]), app.imageToStageMat)
        
        print('Stage centering image offset:',ystep, xstep, units)

        # Move the stage
        if abs(xstep) > minstep:
            stage.move_x(xstep, unit= units, wait_until_idle= True)
        if abs(ystep) > minstep:
            stage.move_y(ystep, unit= units, wait_until_idle= True)

        # Update stage coordinate in the app
        app.coords =  app.stage.get_position()

        # 
        # Start the tracking
        # 
        capture_radius = app.config.getint('Tracking', 'capture_radius')
        binning = app.config.getint('Tracking', 'binning')
        dark_bg = app.config.getboolean('Tracking', 'dark_bg')
        trackingMode =  app.config.get('Tracking', 'mode')
        area = app.config.getint('Tracking', 'area')
        threshold = app.config.getfloat('Tracking', 'threshold')
        min_brightness = app.config.getfloat('Tracking', 'min_brightness')
        max_brightness = app.config.getfloat('Tracking', 'max_brightness')

        # make a tracking thread 
        track_args = minstep, units, capture_radius, binning, dark_bg, area, threshold, trackingMode, min_brightness, max_brightness
        self.trackthread = Thread(target=self.tracking, args = track_args, daemon = True)
        self.trackthread.start()
        print('started tracking thread')

        # schedule occasional position check of the stage
        self.coord_updateevent = Clock.schedule_interval(lambda dt: stage.get_position(), 10)


    def set_ROI(self, roiX, roiY):
        app: GlowTrackerApp = App.get_running_app()
       
        hc, wc = app.camera.setROI(roiX, roiY, isCenter = True)

        print(hc, wc, roiX, roiY)

        # if desired FOV is smaller than allowed by camera, crop in GUI
        if wc > roiX:
            self.cropX = int((wc-roiX)//2)

        if hc > roiY:
            self.cropY = int((hc-roiY)//2)
    

    def tracking(self, minstep: int, units: str, capture_radius: int, binning: int, dark_bg: bool, area: int, threshold: int, mode: str, min_brightness: int, max_brightness: int) -> None:
        """Tracking function to be running inside a thread
        """
        app: GlowTrackerApp = App.get_running_app()
        stage = app.stage
        camera = app.camera

        # Compute second per frame to determine the lower bound waiting time
        camera_spf = 1 / camera.ResultingFrameRate()
        

        # Dual Color mode settings
        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        
        self.isTracking = True
        image: np.ndarray | None = None
        retrieveTimestamp: float = 0
        prevImage: np.ndarray | None = None
        scale = 1.0

        estimated_next_timestamp: float | None = None

        while camera is not None and (camera.IsGrabbing() or camera.isOnHold()) and self.trackingcheckbox.state == 'down':

            # Handling image cycle synchronization.
            # Because the recording and tracking thread are asynchronous
            # and doesn't have the same priority, it could be the case that
            # one thread get executed more than the other and the estimated time
            # became inaccurate.
            wait_time = 0
            if estimated_next_timestamp is not None:
                
                retrieveTimestamp = self.imageacquisitionmanager.imageRetrieveTimeStamp
                diff_estimated_time = estimated_next_timestamp - retrieveTimestamp

                # If the estimated time is approximately close to the image timestamp
                # then it's ok to use the current image. The epsilon in this case is 10% of the camera_spf
                if abs(diff_estimated_time)/camera_spf < 0.1:
                    pass
                else:
                    # If the estimated time is less than the current time
                    # then it is also ok to use the current image
                    if estimated_next_timestamp < retrieveTimestamp:
                        pass
                    # If the estimated time is more than the current image timestamp
                    # then compute the estimated next cycle time and wait
                    else:
                        current_time = time.perf_counter()

                        diff_time_factor = (current_time - retrieveTimestamp) / camera_spf
                        fractional_part, integer_part = math.modf(diff_time_factor)

                        wait_time = camera_spf * ( 1.0 - fractional_part )

                        time.sleep(wait_time)
            else:
                # Wait for the stage to finished moving/centering at location in the
                # first time
                stage.wait_until_idle()

                retrieveTimestamp = self.imageacquisitionmanager.imageRetrieveTimeStamp
                estimated_next_timestamp = self.imageacquisitionmanager.imageRetrieveTimeStamp

            # Get the latest image
            tracking_frame_start_time = time.perf_counter()

            if dualColorMode:
                image = self.imageacquisitionmanager.dualColorMainSideImage
            else:
                image = self.imageacquisitionmanager.image

            retrieveTimestamp = self.imageacquisitionmanager.imageRetrieveTimeStamp

            # If prev frame is empty then use the same as current
            if prevImage is None:
                prevImage = image

            # Extract worm position
            if mode=='Diff':
                ystep, xstep = macro.extractWormsDiff(prevImage, image, capture_radius, binning, area, threshold, dark_bg)
                
            elif mode=='Min/Max':
                ystep, xstep = macro.extractWorms(image, capture_radius = capture_radius,  bin_factor=binning, dark_bg = dark_bg, display = False)

            else:
                try:
                    ystep, xstep, self.trackingMask = macro.extractWormsCMS(image, capture_radius = capture_radius,  bin_factor=binning, dark_bg = dark_bg, display = False, min_brightness= min_brightness, max_brightness= max_brightness )

                except ValueError as e:
                    ystep, xstep = 0, 0
            
            # Record cms for tracking overlay
            self.cmsOffset_x = xstep
            self.cmsOffset_y = -ystep
            
            # Compute relative distancec in each axis
            # Invert Y because the coordinate is in image space which is top left, while the transformation matrix is in btm left
            ystep, xstep = macro.getStageDistances(np.array([-ystep, xstep]), app.imageToStageMat)
            ystep *= scale
            xstep *= scale

            # getting stage coord is slow so we will interpolate from movements
            if abs(xstep) > minstep:
                stage.move_x(xstep, unit=units, wait_until_idle =False)
                app.coords[0] += xstep/1000.
                prevImage = image
            if abs(ystep) > minstep:
                stage.move_y(ystep, unit=units, wait_until_idle = False)
                app.coords[1] += ystep/1000.
                prevImage = image

            tracking_frame_end_time = time.perf_counter()

            #   Wait for stage movement to finish to not get motion blur.
            #   This could be done by checking with stage.is_busy().
            #   However, that function call is very costly (~3 secs) 
            #   and is not good for loop checking.
            #   So we are going to just estimate it here.

            #   Delay from receing the image in recording and tracking it
            delay_receive_image_and_tracking_time = tracking_frame_start_time - retrieveTimestamp

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
        self.cmsOffset_x = None
        self.cmsOffset_y = None
        self.trackingMask = None


    def stopTracking(self):
        """Stop the tracking mode. Unschedule events. Reset camera parameters back. And then update the overlay.
        """
        app: GlowTrackerApp = App.get_running_app()
        camera = app.camera

        if camera is None:
            return
        
        self.isTracking = False
        self.cropX = 0
        self.cropY = 0

        if self.coord_updateevent is not None:
            Clock.unschedule(self.coord_updateevent)
            self.coord_updateevent = None

        dualColorMode = app.config.getboolean('DualColor', 'dualcolormode')
        # If in single color mode
        if not dualColorMode:

            # Reset the camera params back: Width, Height, OffsetX, OffsetY, center flag
            cameraConfig: dict = app.root.ids.leftcolumn.cameraConfig

            # Set camera on hold flag
            camera.setIsOnHold(True)
            # cam stop
            camera.AcquisitionStop.Execute()
            # Wait for camera acquisition to fully stop.
            time.sleep(2/camera.AcquisitionFrameRate()) # Wait 2 frame
            # grab unlock
            camera.TLParamsLocked = False

            # The Basler's camera have a feature-persistence feature, where the camera offset and width/height
            #   are checked against each other all the time so that the sum does not exceed the sensor's limit.
            #   Relaxing the offsets first allows setting any valid widhth/height.
            camera.OffsetX = 0
            camera.OffsetY = 0
            camera.Width = int(cameraConfig['Width'])
            camera.Height = int(cameraConfig['Height'])
            camera.CenterX = bool(int(cameraConfig['CenterX']))
            camera.CenterY = bool(int(cameraConfig['CenterY']))
            camera.OffsetX = int(cameraConfig['OffsetX'])
            camera.OffsetY = int(cameraConfig['OffsetY'])

            # grab lock
            camera.TLParamsLocked = True
            # cam start
            camera.AcquisitionStart.Execute()
            # Set camera on hold flag
            camera.setIsOnHold(False)

        # Update overlay
        app.root.ids.middlecolumn.ids.imageoverlay.updateOverlay()
    

    def computeTrackingCMS(self) -> Tuple[float, float, np.ndarray]:
        """Comput tracking mask and center off mass offsets that would be used for tracking, but just for analytic in this case

        Returns:
            offsetX (float): CMS offset X
            offsetY (float): CMS offset Y
            trackingMask (np.ndarray): boolean mask indicating which pixels are used to compute CMS offsets
        """
        app: GlowTrackerApp = App.get_running_app()

        # Get the current image
        image: np.ndarray = np.zeros(0)
        dualcolormode = app.config.getboolean('DualColor', 'dualcolormode')
        if dualcolormode:
            image = self.imageacquisitionmanager.dualColorMainSideImage

        else:
            image = self.imageacquisitionmanager.image

        # Get tracking configs
        capture_radius = app.config.getint('Tracking', 'capture_radius')
        binning = app.config.getint('Tracking', 'binning')
        dark_bg = app.config.getboolean('Tracking', 'dark_bg')
        min_brightness = app.config.getint('Tracking', 'min_brightness')
        max_brightness = app.config.getint('Tracking', 'max_brightness')

        try:
            offsetY, offsetX, trackingMask = macro.extractWormsCMS(image, capture_radius = capture_radius,  bin_factor=binning, dark_bg = dark_bg, min_brightness= min_brightness, max_brightness= max_brightness)

        except ValueError as e:
            offsetY, offsetX = 0, 0
            trackingMask = np.zeros(image.shape, image.dtype)
        
        finally:
            # Flip Y from the top-right corner to btm-left corner
            return offsetX, -offsetY, trackingMask


class TrackingOverlayQuickButton(ToggleButton):

    normalText = 'Tracking Overlay: [b][color=ff0000]Off[/color][/b]'
    downText = 'Tracking Overlay: [b][color=00ff00]On[/color][/b]'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.markup = True
        self.background_down = self.background_normal

        # Bind starting state to be the same as the config
        app = App.get_running_app()
        showtrackingoverlay = app.config.getboolean('Tracking', 'showtrackingoverlay')

        if showtrackingoverlay:
            self.state = 'down'
            self.text = self.downText

        else:
            self.state = 'normal'
            self.text = self.normalText
        

    def on_state(self, button: ToggleButton, state: 'str'):
        
        # Update config and setting
        app = App.get_running_app()
        configValue = '0'

        if state == 'normal':
            self.text = self.normalText
            configValue = '0'

        else:
            self.text = self.downText
            configValue = '1'
        
        app.config.set('Tracking', 'showtrackingoverlay', configValue)
        app.config.write()

        # Update overlay
        #   Prevent at startup
        if app.root is not None:
            app.root.ids.middlecolumn.ids.imageoverlay.updateOverlay()


class LiveAnalysisQuickButton(ToggleButton):

    normalText = 'Live analysis: [b][color=ff0000]Off[/color][/b]'
    downText = 'Live analysis: [b][color=00ff00]On[/color][/b]'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.markup = True
        self.background_down = self.background_normal

        # Bind starting state to be the same as the config
        app = App.get_running_app()
        showliveanalysis = app.config.getboolean('LiveAnalysis', 'showliveanalysis')

        if showliveanalysis:
            self.state = 'down'
            self.text = self.downText

        else:
            self.state = 'normal'
            self.text = self.normalText
        

    def on_state(self, button: ToggleButton, state: 'str'):
        
        # Update config and setting
        app = App.get_running_app()

        # Pass on start up
        if app.root is None:
            return
        
        liveanalysislabel: LiveAnalysisLabel = app.root.ids.middlecolumn.ids.liveanalysislabel

        if state == 'normal':
            self.text = self.normalText
            app.config.set('LiveAnalysis', 'showliveanalysis', 0)
            liveanalysislabel.disabled = True
            liveanalysislabel.opacity = 0

        else:
            self.text = self.downText
            app.config.set('LiveAnalysis', 'showliveanalysis', 1)
            liveanalysislabel.disabled = False
            liveanalysislabel.opacity = 1
        
        app.config.write()
    

class DualColorViewModeQuickButtonLayout(BoxLayout):
    
    dualcolorviewmodequickbutton = ObjectProperty(None)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.dualcolorviewmodequickbutton = DualColorViewModeQuickButton()

        app = App.get_running_app()
        dualcolormode = app.config.getboolean('DualColor', 'dualcolormode')

        if dualcolormode:
            self.showButton()

        else:
            self.hideButton()

    
    def hideButton(self):
        if self.dualcolorviewmodequickbutton in self.children:
            self.remove_widget(self.dualcolorviewmodequickbutton)


    def showButton(self):
        if not self.dualcolorviewmodequickbutton in self.children:
            self.add_widget(self.dualcolorviewmodequickbutton)

    
class DualColorViewModeQuickButton(ToggleButton):

    normalText = 'Dual Color: [b]Splitted[/b]'
    downText = 'Dual Color: [b]Merged[/b]'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.markup = True
        self.background_down = self.background_normal

        # Bind starting state to be the same as the config
        app = App.get_running_app()
        viewmode = app.config.get('DualColor', 'viewmode')

        if viewmode == 'Splitted':
            self.state = 'normal'
            self.text = self.normalText

        elif viewmode == 'Merged':
            self.state = 'down'
            self.text = self.downText
        

    def on_state(self, button: ToggleButton, state: 'str'):
        
        # Update config and setting
        app = App.get_running_app()
        configValue = str()

        if state == 'normal':
            configValue = 'Splitted'
            self.text = self.normalText

        else:
            configValue = 'Merged'
            self.text = self.downText
        
        app.config.set('DualColor', 'viewmode', configValue)
        app.config.write()

        # Update overlay
        #   Prevent at startup
        if app.root is not None:
            app.root.ids.middlecolumn.ids.imageoverlay.updateOverlay()


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
        print('Connecting Camera')
        # connect camera
        app = App.get_running_app()
        app.camera = basler.Camera.createAndConnectCamera()

        if app.camera is None:
            self.cam_connection.state = 'normal'

        else:
            # load and apply the camera settings
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
            app.stage: Stage = stage # type: ignore
            
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


class SettingsCustomNumeric(SettingNumeric):

    @override
    def _validate(self, instance):
        # Close the popup
        self._dismiss()
        
        value_float = float(0)

        # Check if input is a number
        try:
            value_float = float(self.textinput.text)

        except ValueError:
            # The value is not a number
            return
        
        # Check if should display text in integer style or floating point style
        try:
            value_int = int(self.textinput.text)
            self.value = str(value_int)

        except ValueError:
            # We are here because we couldn't cast the value string to int, thus the value is a float
            self.value = str(value_float)

        return
    

# load the layout
class GlowTrackerApp(App):
    # stage configuration properties - these will update when changed in config menu
    vhigh = ConfigParserProperty(20,
                    'Stage', 'vhigh', 'app', val_type=float)
    vlow = ConfigParserProperty(20,
                    'Stage', 'vlow', 'app', val_type=float)
    unit = ConfigParserProperty('mm/s',
                    'Stage', 'speed_unit', 'app', val_type=str)
    # stage coordinates and current image
    texture = ObjectProperty(None, force_dispatch=True, rebind=True)
    image = ObjectProperty(None, force_dispatch=True, rebind=True)
    coords = ListProperty([0, 0, 0])
    frameBuffer = list()


    def __init__(self,  **kwargs):
        super(GlowTrackerApp, self).__init__(**kwargs)
        # Declare config file path
        self.configFile = self.getDefaultUserConfigFilePath()
        # define settings menu style
        self.settings_cls = SettingsWithSidebar
        # bind key presses to stage motion - right now also happens in settings!
        self.bind_keys()
        # hardware
        self.camera: basler.Camera | None = None
        self.stage: Stage = Stage(None)
        self.updateFpsEvent = None
    

    def getDefaultUserConfigFilePath(self) -> str:
        """Get the default glowtrackeer app config file path from the user local machine.
        The default location depends on the username and the OS. Create a new one by 
        copying from the default template if it doesn't exist.

        Returns:
            configFile (str): The default config file path.
        """
        configFileName = 'glowtracker.ini'
        configDir = platformdirs.user_config_dir(appname= 'GlowTracker', appauthor= 'Monika Scholz')

        # Join the directory path and file name for a complete file path.
        configFullPath = os.path.join(configDir, configFileName)
        
        # If the config file doesn't exist, create a new one.
        if not os.path.exists(configFullPath):

            try:
                # Create a directory if not yet exist.
                os.makedirs(configDir, exist_ok= True)

                # Copy the template file to the target directory.
                shutil.copy(configFileName, configDir)
                
            except Exception as e:
                print(e)

        return configFullPath


    def build_config(self, config: ConfigParser):
        """Set the default values for the configs sections.

        Unfortunately, the caller of this function, which is Kivi.app.App.load_config(),
        forces the default config '<Workspcae>/glowtracker/glowtracker.ini' on to the config object
        eventhough we have specifically specified to load the config file from the user default location.

        The next function, which is self.build(), will have to reload it again.

        Thus, we will skip the loading here and pass the responsibility to self.build() to load instead.
        """
        # Set the config defaults 
        config.setdefaults('Stage', {
            'speed_unit': 'mm/s',
            'vhigh': '30.0',
            'vlow': '1.0',
            'port': '/dev/ttyUSB0',
            'move_start': 'false',
            'homing': 'false',
            'stage_limits': '160,160,180',
            'start_loc': '0,0,0',
            'maxspeed': '20',
            'maxspeed_unit': 'mm/s',
            'acceleration': '60',
            'acceleration_unit': 'mm/s^2',
            'move_image_space_mode': 'false'
        })

        config.setdefaults('Camera', {
            'default_settings': 'settings/defaults.pfs',
            'display_fps': '15',
            'rotation': '0',
            'imagenormaldir': '+Z',
            'pixelsize': '1',
            'depthoffield': '0.0001'
        })

        config.setdefaults('Autofocus', {
            'kp': '0.00001',
            'ki': '0.000000001',
            'kd': '0.000000001',
            'focusestimationmethod': 'SumOfHighDCT',
            'smoothingwindow': '1',
            'minstepbeforechangedir': '0',
            'bestfocusvalue': 2000,
            'focusfps': '15',
            'isshowgraph': 'false',
        })

        config.setdefaults('Calibration', {
            'step_size': '300',
            'step_units': 'um',
            'depthoffieldsearchdistance': '2',
            'depthoffieldnumsampleimages' : '50'
        })

        config.setdefaults('DualColor', {
            'dualcolormode': 'false',
            'mainside': 'Right',
            'viewmode': 'Splitted',
            'recordingmode': 'Original',
            'translation_x': '0',
            'translation_y': '0',
            'rotation': '0'
        })

        config.setdefaults('Tracking', {
            'showtrackingoverlay': 'true',
            'roi_x': '1800',
            'roi_y': '1800',
            'capture_radius': '400',
            'min_step': '1',
            'threshold': '30',
            'binning': '4',
            'dark_bg': 'true',
            'mode': 'CMS',
            'area': '400',
            'min_brightness': '0',
            'max_brightness': '255'
        })

        config.setdefaults('LiveAnalysis', {
            'showliveanalysis': 'true',
            'saveanalysistorecording': 'false',
            'regionmode': 'Full'
        })

        config.setdefaults('Experiment', {
            'exppath': '',
            'nframes': '7500',
            'extension': 'tiff',
            'iscontinuous': 'true',
            'framerate': '50.0',
            'duration': '150.0',
            'buffersize': '3000'
        })

        config.setdefaults('MacroScript', {
            'recentscript': ''
        })

        config.setdefaults('Developer', {
            'showfps': 'false'
        })


    def build(self):

        # Load user's config
        self.config.update_config(self.configFile, overwrite= True)

        # Also save the new updated config back to the user's. In case there are new config fields that the user doesn't have.
        self.config.filename = self.configFile
        self.config.write()

        # Set app name
        self.title = 'GlowTracker'
        # Set app icon
        self.icon = 'icons/glowtracker_gray_logo.png'
        # Load app layout
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
    

    def on_start(self):
        '''Event handler for the `on_start` event which is fired after
        initialization (after build() has been called) but before the
        application has started running.
        '''
        # Display FPS label if enabled
        showfps = self.config.getboolean('Developer', 'showfps')
        if showfps:
            self.startShowFpsEvent()
    
    
    # use custom settings for our GUI
    def build_settings(self, settings: SettingsWithSidebar):
        """build the settings window"""
        # Register custom types
        settings.register_type('custom_numeric', SettingsCustomNumeric)

        # Create settings panel from json
        settings.add_json_panel('GlowTracker', self.config, 'settings/gui_settings.json')
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

        self.config.read(self.configFile)
        
        settings = self.settings_cls()
        self.build_settings(settings)
        
        self.unbind_keys()

        settings.bind(
            on_close= self.close_settings,
            on_config_change= self.on_config_change
        )
        
        return settings


    @override
    def close_settings(self, *args) -> bool:
        """Override the `App.close_settings()` to also handle binding interactions back afterward.

        Returns:
            hasClosedSettings (bool): Return True if has successfully closed the settings.
        """
        # Get window and settings
        win = self._app_window
        settings = self._app_settings

        # Safe-guard if no window or settings
        if win is None \
            or settings is None \
            or settings not in win.children:
            return False

        # Remove the settings widget
        win.remove_widget(settings)

        # Destroy the settings widget
        self._app_settings = None

        # Bind back the keys
        self.bind_keys()
        
        # Enabled back the interaction with preview image widget
        self.root.ids.middlecolumn.ids.scalableimage.disabled = False

        return True
        

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
        
        # print(key, scancode, codepoint, modifier)

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


    def on_config_change(self, settingsWidget: SettingsWithSidebar, config: ConfigParser, section: str, key: str, value: str):

        if config is not self.config:
            return
        
        updateSettingsWidgetFlag = False
        updateOverlayFlag = False

        if section == 'Stage':

            if self.stage is not None:

                # Update the stage settings
                if key == 'stage_limits':
                    # Set the stage limit
                    limits = [float(x) for x in value.split(',')]
                    limits = self.stage.set_rangelimits(limits)
                    # Get back the current value and set back to settings in case the input value is invalid
                    # Round to 2 digis and convert to a str of tuple of char
                    limits = ','.join([str(round(x,2)) for x in limits])
                    self.config.set('Stage', 'stage_limits', limits)
                    updateSettingsWidgetFlag = True
                
                elif key == 'maxspeed':
                    # Set the stage maxspeed
                    maxspeed = float(value)
                    maxspeed_unit = self.config.get('Stage', 'maxspeed_unit')
                    maxspeed = self.stage.set_maxspeed(maxspeed, maxspeed_unit)
                    maxspeed = round(maxspeed, 2)
                    # Get back the current value and set back to settings in case the input value is invalid
                    self.config.set('Stage', 'maxspeed', maxspeed)
                    self.config.write()
                    updateSettingsWidgetFlag = True
                    
                elif key == 'acceleration':
                    # Set the stage acceleration speed
                    acceleration = float(value)
                    acceleration_unit = self.config.get('Stage', 'acceleration_unit')
                    acceleration = self.stage.set_accel(acceleration, acceleration_unit)
                    acceleration = round(acceleration, 2)
                    # Get back the current value and set back to settings in case the input value is invalid
                    self.config.set('Stage', 'acceleration', acceleration)
                    self.config.write()
                    updateSettingsWidgetFlag = True
                
                elif key == 'move_image_space_mode':
                    # value is a str of int or float, i.e. '0', '1' so we have to parse it to boolean
                    self.moveImageSpaceMode = bool(int(value))

        elif section == 'Camera':

            if key in ['pixelsize', 'rotation']:

                print('Updated calibration matrix')
                pixelsize = self.config.getfloat('Camera', 'pixelsize')
                imageNormalDir = self.config.get('Camera', 'imagenormaldir')
                imageNormalDir = 1 if imageNormalDir == '+Z' else -1
                rotation = self.config.getfloat('Camera', 'rotation')

                self.imageToStageMat, self.imageToStageRotMat = macro.CameraAndStageCalibrator.genImageToStageMatrix(pixelsize, imageNormalDir, rotation)
        
        elif section == 'DualColor':
            
            if key == 'dualcolormode':
                updateOverlayFlag = True

                # Also update the DualColorViewMode Quick Button Layout
                dualcolormode = bool(int(value))
                dualColorViewModeQuickButtonLayout: DualColorViewModeQuickButtonLayout = self.root.ids.middlecolumn.ids.runtimecontrols.ids.dualcolorviewmodequickbuttonlayout
                if dualcolormode:
                    dualColorViewModeQuickButtonLayout.showButton()
                    
                else:
                    dualColorViewModeQuickButtonLayout.hideButton()
            
            elif key == 'mainside':
                updateOverlayFlag = True
            
            elif key == 'viewmode':
                updateOverlayFlag = True
            
                # Also update the DualColorViewMode Quick Button
                button = self.root.ids.middlecolumn.ids.runtimecontrols.ids.dualcolorviewmodequickbuttonlayout.dualcolorviewmodequickbutton
                button.state = 'down' if value == 'Merged' else 'normal'
        
        elif section == 'Tracking':

            if key == 'showtrackingoverlay':
                updateOverlayFlag = True

                # Also update the TrackingOverlay Quick Button
                showtrackingoverlay = bool(int(value))
                self.root.ids.middlecolumn.ids.runtimecontrols.ids.trackingoverlayquickbutton.state = \
                    'down' if showtrackingoverlay else 'normal'
                
            elif key == 'capture_radius':
                updateOverlayFlag = True

            elif key == 'min_brightness':
                
                min_brightness = int(value)
                max_brightness = self.config.getint('Tracking', 'max_brightness')

                # Bound the value between [0, max_brightness]
                min_brightness = max(0, min(min_brightness, max_brightness))

                self.config.set('Tracking', 'min_brightness', min_brightness)
                self.config.write()
                updateSettingsWidgetFlag = True
            
            elif key == 'max_brightness':
                
                max_brightness = int(value)
                min_brightness = self.config.getint('Tracking', 'min_brightness')

                # Bound the value between [min_brightness, 255]
                max_brightness = max(min_brightness, min(max_brightness, 255))

                self.config.set('Tracking', 'max_brightness', max_brightness)
                self.config.write()
                updateSettingsWidgetFlag = True
            
        elif section == 'Experiment':

            if key == 'exppath':
                self.root.ids.leftcolumn.ids.saveloc.text = value
        
        elif section == 'LiveAnalysis':

            if key == 'showliveanalysis':
                updateOverlayFlag = True

                # Also update the LiveAnalysis Quick Button
                showliveanalysis = bool(int(value))
                self.root.ids.middlecolumn.ids.runtimecontrols.ids.liveanalysisquickbutton.state = \
                    'down' if showliveanalysis else 'normal'
            

        elif section == 'Developer':

            if key == 'showfps':
                # Convert from text to boolean
                showfps = bool(int(value))
                if showfps:
                    self.startShowFpsEvent()

                else:
                    self.stopShowFpsEvent()

                updateSettingsWidgetFlag = True

            
        # Update setting widget value to reflect the setting file
        if updateSettingsWidgetFlag:
            panels = settingsWidget.interface.content.panels
    
            # For every setting items in the panel
            for panel in panels.values():        
                for child in panel.children:
                    
                    if isinstance(child, SettingItem):                    
                        child.value = panel.get_value(child.section, child.key)
        
        # Update overlay
        if updateOverlayFlag:
            self.root.ids.middlecolumn.ids.imageoverlay.updateOverlay()
        

    def startShowFpsEvent(self):
        # Bring up the FPS label
        self.root.ids.fpslabel.disabled = False
        self.root.ids.fpslabel.opacity = 1

        if self.updateFpsEvent is not None:
            # Stop current event first
            Clock.unschedule(self.updateFpsEvent)
            self.updateFpsEvent = None

        def updateFpsText(dt):
            self.root.ids.fpslabel.text = f'FPS: {Clock.get_fps():.1f}'

        # Start an update event
        self.updateFpsEvent = Clock.schedule_interval(
            updateFpsText
            , 1.0
        )

    def stopShowFpsEvent(self):
        self.root.ids.fpslabel.disabled = True
        self.root.ids.fpslabel.opacity = 0
        # Stop an update event
        Clock.unschedule(self.updateFpsEvent)
        self.updateFpsEvent = None


    def on_image(self, *args) -> None:
        """On image change callback. Update image texture and GUI overlay
        """
        imageHeight, imageWidth = self.image.shape[0], self.image.shape[1]
        imageColorFormat = 'rgb' if self.image.ndim == 3 else 'luminance'
        # Force unsign byte format
        imageDataFormat = 'ubyte'

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

            # Update overlay
            self.root.ids.middlecolumn.ids.imageoverlay.updateOverlay()

        # Upload image data to texture
        imageByteBuffer: bytes = self.image.tobytes()
        self.texture.blit_buffer(imageByteBuffer, colorfmt= imageColorFormat, bufferfmt= imageDataFormat)

        # Update tracking overlay if the option is enabled
        if self.config.getboolean('Tracking', 'showtrackingoverlay'):
            self.root.ids.middlecolumn.ids.imageoverlay.updateTrackingOverlay(doClear= False)
    

    # ask for confirmation of closing
    def on_request_close(self, *args, **kwargs):
        content = ExitApp(stop=self.graceful_exit, cancel=self.dismiss_popup)
        self._popup = Popup(title="Exit GlowTracker", content=content,
                            size_hint=(0.5, 0.2))
        self._popup.open()
        return True


    def dismiss_popup(self):
        self._popup.dismiss()


    def graceful_exit(self):
        # disconnect hardware
        # stop remaining stage motion
        if self.stage is not None:
            print('Disconnecting Stage')
            self.stage.stop()
            self.stage.disconnect()
        
        if self.camera is not None:
            print('Disconnecting Camera')
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


def main():
    reset()
    Window.size = (1280, 800)
    Config.set('graphics', 'position', 'custom')
    Config.set('graphics', 'top', '0') 
    Config.set('graphics', 'left', '0') 

    # Last barrier for catching unhandled exception.
    try:
        App = GlowTrackerApp()
        App.run()  # This runs the App in an endless loop until it closes. At this point it will execute the code below

    except Exception as e:
        print(f'Kivy App error: {e}')
        return None


if __name__ == '__main__':
    main()