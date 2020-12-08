from pypylon import genicam, pylon



#%% Camera initialization
def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        #camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        # Get the transport layer factory.
        tlFactory = pylon.TlFactory.GetInstance()
        # Get all attached devices and exit application if no device is found.
        devices = tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise pylon.RuntimeException("No camera present.")

        # Create and attach the first Pylon Devices.
        camera = pylon.InstantCamera(tlFactory.CreateDevice(devices[0]))

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
