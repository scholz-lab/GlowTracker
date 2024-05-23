from pypylon import genicam, pylon
import os
import time
from skimage.io import imsave
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraGrabParameters:
    bufferSize: int
    grabStrategy: pylon.GrabStrategy_OneByOne | pylon.GrabStrategy_LatestImageOnly
    numberOfImagesToGrab: int = -1
    

class Camera(pylon.InstantCamera):

    def __init__(self):

        # Class variable
        #   Cannot inject a new variable directly onto the pylon.InstantCamera class
        #   because they implemented conditions into their getter, setters.
        #   But they still allow for private property starts with "__" so we create
        #   a private variable and our getter, setter instead

        #   isOnHold flag use to expres a behavior where the camera is currently
        #   on image acquisition mode, but is on a pause.
        self.__isOnHold__ = True

        # Create an instant camera object with the camera device found first.
        super().__init__(pylon.TlFactory.GetInstance().CreateFirstDevice())

        try:
            # Register an image event handler that accesses the chunk data.
            class ImageEventPrinter(pylon.ImageEventHandler):
                """A simple dummy class for passing image event
                """

                def OnImagesSkipped(self, camera, countOfSkippedImages):
                    print(countOfSkippedImages, " images have been skipped.")

                def OnImageGrabbed(self, camera, grabResult):
                    return True
            
            self.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
            
            # Open the connection
            self.Open()
            
            # Print the model name of the camera.
            print("Using device ", self.GetDeviceInfo().GetModelName())

        except genicam.GenericException as exception:
            print(exception)

    # Getters, Setters
    @property
    def isOnHold(self):
        return self.__isOnHold__
    

    @isOnHold.setter
    def isOnHold(self, value):
        self.__isOnHold__ = value
    

    # Class functions
    def updateProperties(self, propfile):
        """read psf file and alter camera properties.
            propfile: path to .pfs file"""
        try:
            pylon.FeaturePersistence.Load(propfile, self.GetNodeMap(), True)

        except genicam.RuntimeException as e:
            print(f"Camera features could not be loaded. {e}")


    def retrieveGrabbingResult(self) -> Tuple[ bool, np.ndarray, int, int]:
        """Retrieve a grabbed image from a camera

        Returns:
            isSuccess (bool): boolean indicate if the retrieving is successful
            img (np.array): the retrieved image
            timestamp (int): time stamp when the result is captured by camera internal clock
            retrieveTimestamp (int): time stamp when the result is received via time.perf_counter() 
        """
        if not self.IsGrabbing():
            return False, None, None, None

        else:
            try:
                # Retrieve an image
                grabResult: pylon.GrabResult = self.RetrieveResult(1000, pylon.TimeoutHandling_Return)

                if grabResult.GrabSucceeded():

                    img = grabResult.Array
                    retrieveTimestamp = time.perf_counter()
                    conversion_factor = 1e6  # for conversion in ms
                    timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
                    grabResult.Release()

                    return True, img, timestamp, retrieveTimestamp
                    
            except genicam.RuntimeException as e:
                # Handle a RuntimeException here because
                #   when closing the app while in a grabbing mode,
                #   this thread will still trying to access the camera result
                print(e)
                return False, None, None, None
       

    def singleTake(self) -> Tuple[ bool, np.ndarray ]:
        """Take and return a single image.

        Returns:
            isSuccess (bool): is the capturing image successful
            img (np.ndarray): the resulting image
        """
        self.StartGrabbingMax(1)
        isSuccess, img, _, _ = self.retrieveGrabbingResult()
        return isSuccess, img


    def setROI(self, ROI_w: int, ROI_h: int, isCenter: bool= True) -> Tuple[int, int]:
        """Set the ROI of a camera.

        Args:
            ROI_w (int): ROI width
            ROI_h (int): ROI height
            isCenter (bool, optional): If true then compute offset such that the ROI is center of the camera. This does not set the CenterX nor CenterY bool flag of the camera.

        Returns:
            height (int): the actual camera ROI width that has been set
            width (int): the actual camera ROI width that has been set
        """
        
        if ROI_w <= self.Width.Max and ROI_h <= self.Height.Max:
            # cam stop
            self.AcquisitionStop.Execute()
            # grab unlock
            self.TLParamsLocked = False

            prevCameraWidth = self.Width()
            prevCameraHeight = self.Height()

            self.Width = max(ROI_w, self.Width.Min)
            self.Height = max(ROI_h, self.Height.Min)

            if isCenter:
                
                # Compute additional offset from the previous offset 
                additionalOffsetX = (prevCameraWidth - self.Width())//2
                additionalOffsetY = (prevCameraHeight - self.Height())//2

                offsetX = self.OffsetX() + additionalOffsetX
                offsetY = self.OffsetY() + additionalOffsetY

                # Round offsets to be multiples of 4
                offsetX = int(round(offsetX / 4) * 4)
                offsetY = int(round(offsetY / 4) * 4)

                # Bound by minimum of 4
                offsetX = max(offsetX, 4)
                offsetY = max(offsetY, 4)

                # Set the camera offset
                self.OffsetX = offsetX
                self.OffsetY = offsetY
                
            # grab lock
            self.TLParamsLocked = True
            # cam start
            self.AcquisitionStart.Execute()

        return self.Height(), self.Width()


    def resetROI(self) -> Tuple[int, int]:
        """set the ROI for a camera to full sensor size.

        Returns:
            width(int): camera sensor width
            height(int): camera sensor height
        """
        # cam stop
        self.AcquisitionStop.Execute()
        # grab unlock
        self.TLParamsLocked = False

        self.OffsetX = 0
        self.OffsetY = 0
        self.Width = self.Width.Max
        self.Height = self.Height.Max

        # grab lock
        self.TLParamsLocked = True
        # cam start
        self.AcquisitionStart.Execute()

        return self.Width.GetValue(), self.Height.GetValue()


    def setFramerate(self, fps: float) -> float:
        """Change acquisition framerate. Returns real framerate achievable with settings.

        Args:
            fps (float): desired framerate

        Returns:
            fps (float): the resulting framerate
        """
        
        self.AcquisitionFrameRateEnable = True
        self.AcquisitionFrameRate = float(fps)
        return self.ResultingFrameRate()


# Utility functions
def saveImage(im: np.ndarray, path: str, fname: str, isFlipY: bool= False) -> None:
    """Save image in path using fname and ext as extension.

    Args:
        im (np.ndarray): the image
        path (str): image path
        fname (str): image file name
        isFlipY (bool, optional): _description_. Defaults to False.
    """    
    img = im

    if isFlipY:
        img = np.flip(img, axis= 0)

    try:
        imsave(os.path.join(path, fname), img, check_contrast=False)
        
    except FileNotFoundError as e:
        print(e)


def readPFSFile(filepath: str) -> Dict[str, str] | None:
    """Read a camera config file ".pfs" and parse as a dict

    Args:
        filepath (str): the .pfs file path

    Returns:
        Dict[str, str] | None: A string dictionary contains
        the configuration key and value. The value is always parsed
        as a string, so if it is number or other type, it would need
        to be converted manully before use. Return None if the reading or
        parsing is unsuccessfull.
    """
    
    parsedDict = {}
    
    try:
        with open(filepath, 'r') as file:
            
            for line in file:

                # Strip unnescessary spaces
                line = line.strip()
                
                # Skip comment
                if line.startswith('#'):
                    continue

                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, value = parts
                    parsedDict[key] = value
                
                else:
                    print(f'Error: Parsing {filepath}. Format is not supported')
                    return None
        
            return parsedDict

    except FileNotFoundError:
        print(f"Error: File '{filepath}' is not found.")
        
    except Exception as e:
        print(e)
    
    return None

