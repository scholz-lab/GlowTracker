from pypylon import genicam, pylon
# for saving
from PIL import Image
import os


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
        # Loading settings
        #print("Possible place to read the file defining the camera's settings...")
        #node_file = "FluorescenceTestLawn.pfs"
        #pylon.FeaturePersistence.Load(node_file, camera.GetNodeMap(), True)
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


def start_grabbing(camera, numberOfImagesToGrab = 100):
    """start grabbing with the camera"""
    camera.StartGrabbing(numberOfImagesToGrab, pylon.GrabStrategy_LatestImageOnly)

def stop_grabbing(camera):
    """start grabbing with the camera"""
    camera.StopGrabbing()


def retrieve_result(camera):
    """start grabbing with the camera"""
    #camera.StartGrabbing(numberOfImagesToGrab, pylon.GrabStrategy_LatestImageOnly)
    # while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

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


def save_image(im,path,fname):
    """save image in path using fname and ext as extension."""
    im = Image.fromarray(im)
    im.save(os.path.join(path, fname))


