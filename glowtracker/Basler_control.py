from pypylon import genicam, pylon
import os
import time
from skimage.io import imsave
from typing import Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class CameraGrabParameters:
    bufferSize: int
    grabStrategy: pylon.GrabStrategy_OneByOne | pylon.GrabStrategy_LatestImageOnly
    numberOfImagesToGrab: int = -1
    

#%% Camera initialization
def camera_init():
    """Initialize a basler camera.

    Returns:
        camera object (GenICam)
    """
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Register an image event handler that accesses the chunk data.
        camera.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        return camera
    except genicam.GenericException as exception:
        # Error handling.
        print("An exception occurred.")
        print(exception)
        #print(e.GetDescription())
    return None


def update_props(camera, propfile):
    """read psf file and alter camera properties.
        camera: pylon.InstantCamera
        propfile: path to .pfs file"""
    try:
        pylon.FeaturePersistence.Load(propfile, camera.GetNodeMap(), True)
    except genicam.RuntimeException:
        print("Camera features could not be loaded.")
       

def single_take(camera: pylon.InstantCamera) -> Tuple[ bool, np.ndarray ]:
    """Take and return a single image.

    Args:
        camera (pylon.InstantCamera): the camera use for capture

    Returns:
        isSuccess (bool): is the capturing image successful
        img (np.ndarray): the resulting image
    """
    camera.StartGrabbingMax(1)
    isSuccess, img, _, _ = retrieve_grabbing_result(camera)
    return isSuccess, img



def stop_grabbing(camera):
    """start grabbing with the camera"""
    camera.StopGrabbing()


def retrieve_grabbing_result(camera: pylon.InstantCamera) -> Tuple[ bool, np.ndarray, int, int]:
    """Retrieve a grabbed image from a camera

    Args:
        camera (pylon.InstantCamera): camera to retrieve result

    Returns:
        isSuccess (bool): boolean indicate if the retrieving is successful
        img (np.array): the retrieved image
        timestamp (int): time stamp when the result is captured by camera internal clock
        retrieveTimestamp (int): time stamp when the result is received via time.perf_counter() 
    """
    if camera.IsGrabbing():
        try:
            grabResult: pylon.GrabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
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


class ImageEventPrinter(pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        #print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        #print()

    def OnImageGrabbed(self, camera, grabResult):
        # print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())

        # # Image grabbed successfully?
        # if grabResult.GrabSucceeded():
        #     print("SizeX: ", grabResult.GetWidth())
        #     print("SizeY: ", grabResult.GetHeight())
        # else:
        #     print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())
        return True


def save_image(im: np.ndarray, path: str, fname: str, isFlipY: bool= False) -> None:
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
    imsave(os.path.join(path, fname), img, check_contrast=False)


def cam_setROI(camera: pylon.InstantCamera, ROI_w: int, ROI_h: int, center: bool= True) -> Tuple[int, int]:
    """Set the ROI of a camera.

    Args:
        camera (pylon.InstantCamera): the camera to set
        ROI_w (int): ROI width
        ROI_h (int): ROI height
        center (bool, optional): If the ROI is at the center of the camera. Defaults to True.

    Returns:
        height (int): the actual camera ROI width that has been set
        width (int): the actual camera ROI width that has been set
    """
     
    if ROI_w <= camera.Width.Max and ROI_h <= camera.Height.Max:
        # cam stop
        camera.AcquisitionStop.Execute()
        # grab unlock
        camera.TLParamsLocked = False
        camera.Width = max(ROI_w, camera.Width.Min)
        camera.Height = max(ROI_h, camera.Height.Min)
        if center:
            # Round offsets to be multiples of 4
            offsetX = (camera.Width.Max - camera.Width())//2
            offsetX = int(round(offsetX / 4) * 4)
            offsetY = (camera.Height.Max - camera.Height())//2
            offsetY = int(round(offsetY / 4) * 4)
            camera.OffsetX = max(offsetX, 4)
            camera.OffsetY = max(offsetY, 4)
        # grab lock
        camera.TLParamsLocked = True
        # cam start
        camera.AcquisitionStart.Execute()

    return camera.Height(), camera.Width()


def cam_resetROI(camera):
    """set the ROI for a camera to full sensor size."""
    # cam stop
    camera.AcquisitionStop.Execute()
    # grab unlock
    camera.TLParamsLocked = False
    camera.OffsetX = 0
    camera.OffsetY = 0
    camera.Width = camera.Width.Max
    camera.Height = camera.Height.Max
    # grab lock
    camera.TLParamsLocked = True
    # cam start -- do not!
    camera.AcquisitionStart.Execute()
    return camera.Height.GetValue(), camera.Width.GetValue()


def set_framerate(camera, fps):
    """change acquisition framerate. Returns real framerate achievable with settings."""
    camera.AcquisitionFrameRateEnable = True
    camera.AcquisitionFrameRate = float(fps)
    return camera.ResultingFrameRate()


def get_shape(camera):
    """return current field of view size."""
    return  camera.Height.GetValue(), camera.Width.GetValue()
