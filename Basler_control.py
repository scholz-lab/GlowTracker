from pypylon import genicam, pylon
# for saving
from PIL import Image
import os
import time
from skimage.io import imsave

#%% Camera initialization
def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Register an image event handler that accesses the chunk data.
        camera.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e)
        #print(e.GetDescription())
    return None


def update_props(camera, propfile):
    """read psf file and alter camera properties.
        camera: pylon.InstantCamera
        propfile: path to .pfs file"""
    pylon.FeaturePersistence.Load(propfile, camera.GetNodeMap(), True)


def single_take(camera):
    """take and return a single image."""
    camera.StartGrabbingMax(1)
    ret, img = retrieve_result(camera)
    return ret, img

def start_grabbing(camera, numberOfImagesToGrab = 100, record = False):
    """start grabbing with the camera"""
    
    if record:
        camera.StartGrabbingMax(numberOfImagesToGrab,pylon.GrabStrategy_OneByOne)#, pylon.GrabLoop_ProvidedByInstantCamera)
    else:
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

def stop_grabbing(camera):
    """start grabbing with the camera"""
    camera.StopGrabbing()


def retrieve_result(camera):
    """start grabbing with the camera"""
    #camera.StartGrabbing(numberOfImagesToGrab, pylon.GrabStrategy_LatestImageOnly)
    # while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
    
    if grabResult.GrabSucceeded():
        img = grabResult.Array
        grabResult.Release()
        return True, img
    else:
        return False, None

class ImageEventPrinter(pylon.ImageEventHandler):
    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print("OnImagesSkipped event for device ", camera.GetDeviceInfo().GetModelName())
        print(countOfSkippedImages, " images have been skipped.")
        print()

    def OnImageGrabbed(self, camera, grabResult):
        print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())

        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            print("SizeX: ", grabResult.GetWidth())
            print("SizeY: ", grabResult.GetHeight())
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())
        return True


def save_image(im,path,fname):
    """save image in path using fname and ext as extension."""
    start = time.time()
    #using PIL - slow
    #im = Image.fromarray(im)
    #im.save(os.path.join(path, fname), quality = 100)
    #using skimage
    imsave(os.path.join(path, fname), im, check_contrast=False,  plugin="tifffile")
    print('Saving time: ',time.time() - start)


def cam_setROI(camera, w,h,center = True):
    """set the ROI for a camera. ox, oy are offsets, w,h are the width and height in pixel, respectively."""
    if w <= camera.Width.Max and h <= camera.Height.Max:
        # cam stop
        camera.AcquisitionStop.Execute()
        # grab unlock
        camera.TLParamsLocked = False
        camera.Width = max(w, camera.Width.Min)
        camera.Height = max(h, camera.Height.Min)
        if center:
            print(camera.Width.Max , camera.Width())
            camera.OffsetX = max(int(camera.Width.Max - camera.Width())//2, 4)
            camera.OffsetY = max(int(camera.Height.Max - camera.Height())//2, 4)
        
        # grab lock
        camera.TLParamsLocked = True
        # cam start
        camera.AcquisitionStart.Execute()

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