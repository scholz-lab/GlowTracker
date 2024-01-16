#!/usr/bin/env python3
# First attempt at running a basler cam on raspberry Pi using pypylon.
# @author: monika
# @date: October 2019
import sys
sys.path.append('/home/pi/.local/lib/python3.7/site-packages/')
from pypylon import pylon
import platform
import time
import datetime

import subprocess
# this is a way to get pylon into the root PYTHONPATH. Not preferred but works for now.

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
            #img = grabResult.GetArray()
            #saveImage(img, filename.format(grabResult.TimeStamp))
            #print("Gray values of first row: ", img[0])
            #print()
            #img.Release()
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())


if __name__ == "__main__":
    ###########################################################
    #
    # Relevant acquisition parameters
    #
    ###########################################################
    duration = 1 # in minutes
    frequency = 30 # Hz
    NumberOfPictures = frequency*60*duration
    fname = "img_{}.tif"
    ###########################################################
    #
    # start cam and load configuration
    #
    ###########################################################
    img = pylon.PylonImage()
    tlf = pylon.TlFactory.GetInstance()
    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    # Print the model name of the cam.
    print("Using device ", cam.GetDeviceInfo().GetModelName())
    ###########################################################
    #
    # make cam ready for a hardware trigger
    #
    ###########################################################
    cam.Open()
    cam.TriggerMode = 'Off'
    #cam.ExposureMode = 'TriggerControlled'
    # set exposure time to 100 microseconds works correctly,
    cam.ExposureTime = 15000
    # set binning to get reasonable data rates
    cam.BinningHorizontal = 1  # 4
    cam.BinningVertical = 1  # 4
    cam.BinningVerticalMode = "Sum"
    cam.BinningHorizontalMode = "Sum"
    # 
    print(f'Actual framerate: {cam.ResultingFrameRate}')
    # set Acquisition framerate
    cam.AcquisitionFrameRateEnable = True
    cam.properties.AcquisitionFrameRate = frequency
    ###########################################################
    #
    # save data
    #
    ###########################################################
    
    # For demonstration purposes only, register another image event handler.
    #cam.RegisterImageEventHandler(ImageEventPrinter(), pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

        # Start the grabbing using the grab loop thread, by setting the grabLoopType parameter
        # to GrabLoop_ProvidedByInstantCamera. The grab results are delivered to the image event handlers.
        # The GrabStrategy_OneByOne default grab strategy is used.
        #camera.StartGrabbing(pylon.GrabStrategy_OneByOne, pylon.GrabLoop_ProvidedByInstantCamera)
   
    cam.StartGrabbing(pylon.GrabStrategy_OneByOne)
    now = datetime.datetime.now()
    times = 0
    buffersInQueue = 0
    print("Started Grabbing")
    while (buffersInQueue < NumberOfPictures):
        grabResult = cam.RetrieveResult(5000, pylon.TimeoutHandling_Return)
        if not grabResult.GrabSucceeded():
            print('Grab failed')
            break
        print((grabResult.TimeStamp - times)/10e6) # should be in ms now
        times = grabResult.TimeStamp
            #print("Skipped ", grabResult.GetNumberOfSkippedImages(), " images.")
        #timestamp = grabResult.TimeStamp
        
        # Calling AttachGrabResultBuffer creates another reference to the
        # grab result buffer. This prevents the buffer's reuse for grabbing.
        img.AttachGrabResultBuffer(grabResult)
        img.Save(pylon.ImageFileFormat_Tiff,  fname.format(grabResult.TimeStamp))
        
        # In order to make it possible to reuse the grab result for grabbing
        # again, we have to release the image (effectively emptying the
        # image object).
        img.Release()
        
        buffersInQueue += 1
        
    print("Retrieved ", buffersInQueue, " grab results from output queue.")
    ###########################################################
    #
    #  cleanup
    #
    ###########################################################
    cam.StopGrabbing()
    cam.Close()
    then = datetime.datetime.now()
    execution_time = then - now

    print("Execution time {} ms".format(execution_time.microseconds/1000))
    
