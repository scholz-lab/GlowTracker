import pypylon.pylon as py
import pypylon.genicam as geni
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pandas as pd


def testSingleGrab(cam: py.InstantCamera):
    
    print("Waiting for trigger signal")
    grabResult: py.GrabResult = cam.GrabOne(py.waitForever)

    if grabResult.GrabSucceeded():

        print("Successful!")

        img = grabResult.Array
        retrieveTimestamp = time.perf_counter()
        conversion_factor = 1e6  # for conversion in ms
        timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
        grabResult.Release()

        plt.figure()
        plt.imshow(img)
        plt.legend()

        plt.show()

    else:
        print("Unsuccessful")


class TriggeredImage(py.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.grab_times = []

        self.images = []
        self.retrieveTimestamps = []
        self.timestamps = []
        
        
    def OnImageGrabbed(self, camera, grabResult: py.GrabResult):

        if grabResult.GrabSucceeded():
            print("Got an image!")

            self.grab_times.append(grabResult.TimeStamp)
            
            self.images.append(grabResult.Array)

            self.retrieveTimestamps.append(time.perf_counter())
            
            conversion_factor = 1e6  # for conversion in ms
            timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
            self.timestamps.append(timestamp)

            grabResult.Release()

        else:
            print("Grab didn't succeed")
    

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print(countOfSkippedImages, " images have been skipped.")


def testLoopGrab(cam: py.InstantCamera):

    # create event handler instance
    image_timestamps = TriggeredImage()

    # register handler
    # remove all other handlers
    cam.RegisterImageEventHandler(image_timestamps, 
                                py.RegistrationMode_ReplaceAll, 
                                py.Cleanup_None)

    # start grabbing with background loop
    cam.StartGrabbingMax(10, py.GrabStrategy_LatestImages, py.GrabLoop_ProvidedByInstantCamera)

    print("Start grabbing 10 images")

    # wait ... or do something relevant
    while cam.IsGrabbing():
        time.sleep(0.1)

    # stop grabbing
    cam.StopGrabbing()

    print("Finished grabbping")

    # Plotting some result
    plt.figure()
    frame_delta_s = np.diff(image_timestamps.grab_times)/1.e9
    plt.plot(frame_delta_s, ".")
    plt.axhline(np.mean(frame_delta_s))
    plt.title("grab_times")

    plt.figure()
    timestampsDif = np.diff(image_timestamps.timestamps)
    plt.plot(timestampsDif, ".")
    plt.axhline(np.mean(timestampsDif))
    plt.title("timestamps (cam)")


    plt.figure()
    retrieveTimestampsDif = np.diff(image_timestamps.retrieveTimestamps)
    plt.plot(retrieveTimestampsDif, ".")
    plt.axhline(np.mean(retrieveTimestampsDif))
    plt.title("timestmps (host)")

    plt.show()


def testBurstGrab(cam: py.InstantCamera):

    cam.TriggerSelector.Value = "FrameBurstStart"

    cam.AcquisitionBurstFrameCount.Value = 10

    # create event handler instance
    image_timestamps = TriggeredImage()

    # register handler
    # remove all other handlers
    cam.RegisterImageEventHandler(image_timestamps, 
                                py.RegistrationMode_ReplaceAll, 
                                py.Cleanup_None)

    # start grabbing with background loop
    # cam.StartGrabbingMax(10, , )
    # cam.StartGrabbing(15, py.GrabStrategy_LatestImages, py.GrabLoop_ProvidedByInstantCamera)
    cam.StartGrabbing(py.GrabStrategy_OneByOne)

    print("Start grabbing 10 images")

    # wait ... or do something relevant
    while cam.IsGrabbing():
        time.sleep(0.1)

    # stop grabbing
    cam.StopGrabbing()

    print("Finished grabbping")

    # Plotting some result
    plt.figure()
    frame_delta_s = np.diff(image_timestamps.grab_times)/1.e9
    plt.plot(frame_delta_s, ".")
    plt.axhline(np.mean(frame_delta_s))
    plt.title("grab_times")

    plt.figure()
    timestampsDif = np.diff(image_timestamps.timestamps)
    plt.plot(timestampsDif, ".")
    plt.axhline(np.mean(timestampsDif))
    plt.title("timestamps (cam)")


    plt.figure()
    retrieveTimestampsDif = np.diff(image_timestamps.retrieveTimestamps)
    plt.plot(retrieveTimestampsDif, ".")
    plt.axhline(np.mean(retrieveTimestampsDif))
    plt.title("timestmps (host)")

    plt.show()


if __name__ == '__main__':

    # open the camera
    tlf = py.TlFactory.GetInstance()
    cam = py.InstantCamera(tlf.CreateFirstDevice())
    cam.Open()

    # Set acquisition mode to hardware trigger
    cam.TriggerMode.Value = "On"

    # Set Line1 to be input
    cam.LineSelector.Value = "Line1"
    cam.LineMode.Value = "Input"
    
    # Set trigger source to Line1
    cam.TriggerSource.Value = "Line1"
    cam.TriggerSelector.Value = "FrameStart"
    cam.TriggerActivation.Value = "RisingEdge"
    
    
    # testSingleGrab(cam)

    # testLoopGrab(cam)

    # TODO:
    testBurstGrab(cam)

    cam.Close()
