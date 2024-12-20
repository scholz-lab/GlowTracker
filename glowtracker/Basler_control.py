from __future__ import annotations
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
    isContinuous: bool = True
    numberOfImagesToGrab: int = 1
    

class Camera(pylon.InstantCamera):

    @classmethod
    def createAndConnectCamera(cls) -> Camera | None:
        """Create a Camera class object. Return the object if the connection to the actual camera
        is successful. Otherwise, return None.

        Returns:
            camera (Camera|None): Camera(pylon.InstantCamera) class if successful, otherwise None.
        """

        try:
            # Create an instant camera object with the camera device found first.
            pylonCameraHandle = pylon.TlFactory.GetInstance().CreateFirstDevice()
            # Create the Camera class object
            camera = cls(pylonCameraHandle)

            # Register an image event handler that accesses the chunk data.
            class ImageEventPrinter(pylon.ImageEventHandler):
                """A simple dummy class for passing image event
                """

                def OnImagesSkipped(self, camera, countOfSkippedImages):
                    print(countOfSkippedImages, " images have been skipped.")

                def OnImageGrabbed(self, camera, grabResult):
                    return True
            
            camera.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
            
            # Open the connection
            camera.Open()
            
            # Print the model name of the camera.
            print("Using device ", camera.GetDeviceInfo().GetModelName())

            return camera

        except genicam.GenericException as exception:
            # Cannot connect to the camera
            print(exception)
            return None
    

    def __init__(self, *args):
        # WARNING: Outsider should not use this as a way to create and connect to camera.
        #   use the method createAndConnectCamera() instead.
        super().__init__(*args)

        # Class variable
        #   Cannot inject a new variable directly onto the pylon.InstantCamera class
        #   because they implemented conditions into their getter, setters.
        #   But they still allow for private property starts with "__" so we create
        #   a private variable and our getter, setter instead

        #   isOnHold flag use to expres a behavior where the camera is currently
        #   on image acquisition mode, but is on a pause.
        self.__isOnHold__ = True

        
    # Getters, Setters
    def isOnHold(self) -> bool:
        return self.__isOnHold__
    

    def setIsOnHold(self, value):
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
        isSuccess = False
        img = None
        timestamp = None
        retrieveTimestamp = None
        
        if self.IsGrabbing():
            try:
                # Retrieve an image
                #   The function pylon.InstantCamera is not well-ported to Python API.
                #   If the grab is succeeded it will return pylon.GrabResult object.
                #   Otherwise, it will return False.
                grabResult: pylon.GrabResult | bool = self.RetrieveResult(1000, pylon.TimeoutHandling_Return)

                if isinstance(grabResult, bool) and grabResult == False:
                    pass

                else:
                    # Need to double check
                    if grabResult.GrabSucceeded():

                        isSuccess = True
                        img = grabResult.Array
                        retrieveTimestamp = time.perf_counter()
                        conversion_factor = 1e6  # for conversion in ms
                        timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
                        grabResult.Release()

            except genicam.RuntimeException as e:
                # An exception is thrown here when trying to access a grab result while the camera 
                #   aquisition is being shut down. This can happen when the acquisition is happening
                #   in a thread and failed to synchronize with the main thread in time.
                pass
                
            except Exception as e:
                # Report other error behaviors for better handling
                print(f'Camera::retrieveGrabbingResult -- {e}')

        return isSuccess, img, timestamp, retrieveTimestamp

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

            # Set camera on hold flag
            self.setIsOnHold(True)
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
            # Set camera on hold flag
            self.setIsOnHold(False)

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
    

    def getAllFeatures(self) -> dict[str, any]:
        """Get all current camera's features.

        Returns:
            features(dict[str, any]): Dict of camera features
        """
        # Get all camera IValues
        IValues: Tuple[genicam.IValue] = self.NodeMap.GetNodes()

        features: dict[str, any] = dict()

        for IValue in IValues:

            try:
                
                # Check if it's one of the type we're interested in
                if type(IValue) in [genicam.IBoolean, genicam.IInteger, genicam.IBoolean, genicam.IString]:

                    # Check if the node that holds the value is a feature node
                    node: genicam.INode = IValue.GetNode()
                    
                    if node.IsFeature():

                        try:
                            # Get feature name and value
                            featureName = node.GetName()
                            value = IValue.Value
                            features[featureName] = value

                        except genicam.AccessException as e:
                            # Continue if node or value is not accessable.
                            pass

            except Exception as e:
                print(f'Some weird exception {e}')

        return features


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

