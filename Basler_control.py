from pypylon import genicam, pylon
# for saving
from PIL import Image
import os
import time
from skimage.io import imsave
#import cv2
#from libtiff import TIFF

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
       

def single_take(camera):
    """take and return a single image."""
    camera.StartGrabbingMax(1)
    ret, img,_ = retrieve_result(camera)
    return ret, img


def start_grabbing(camera, numberOfImagesToGrab=100, record=False, buffersize=16):
    """start grabbing with the camera"""
    camera.MaxNumBuffer = buffersize
    if record:
        camera.StartGrabbingMax(numberOfImagesToGrab, pylon.GrabStrategy_OneByOne)
    else:
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


def stop_grabbing(camera):
    """start grabbing with the camera"""
    camera.StopGrabbing()


def retrieve_result(camera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
    if grabResult.GrabSucceeded():
        conversion_factor = 1e6  # for conversion in ms
        img = grabResult.Array.copy()
        time = round(grabResult.TimeStamp/conversion_factor, 1)
        grabResult.Release()
        return True, img, time
    else:
        return False, None, None


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


def save_image(im, path, fname):
    """save image in path using fname and ext as extension."""
    #using PIL - slow
    #im = Image.fromarray(im)
    #im.save(os.path.join(path, fname), quality = 100)
    #using skimage
    imsave(os.path.join(path, fname), im, check_contrast=False,  plugin="tifffile")
    #cv2.imwrite(os.path.join(path, fname), im)
    #tiff = TIFF.open(os.path.join(path, fname), mode='w')
    #tiff.write_image(im)
    #tiff.close()

def cam_setROI(camera, w, h, center=True):
    """set the ROI for a camera. ox, oy are offsets, w,h are the width and height in pixel, respectively."""
    if w <= camera.Width.Max and h <= camera.Height.Max:
        # cam stop
        camera.AcquisitionStop.Execute()
        # grab unlock
        camera.TLParamsLocked = False
        camera.Width = max(w, camera.Width.Min)
        camera.Height = max(h, camera.Height.Min)
        if center:
            camera.OffsetX = max(int(camera.Width.Max - camera.Width())//2, 4)
            camera.OffsetY = max(int(camera.Height.Max - camera.Height())//2, 4)
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
    # cam start
    camera.AcquisitionStart.Execute()
    return camera.Height(), camera.Width()


def set_framerate(camera, fps):
    """change acquisition framerate. Returns real framerate achievable with settings."""
    camera.AcquisitionFrameRateEnable = True
    camera.AcquisitionFrameRate = float(fps)
    return camera.ResultingFrameRate()
