# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:04:15 2020

@author: hofmann
"""
from matplotlib import pyplot
import sys
from pypylon import pylon
from pypylon import genicam


def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        # Loading settings
        print("Possible place to read the file defining the camera's settings...")
        nodeFile = "FluorescenceTestWormsSmallROI.pfs"
        pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exit_code = 1
    sys.exit(exit_code)


def live_update_demo(blit=False):
    
    # set up a figure
    fig = pyplot.figure(figsize=(5,5))
    ax1 = fig.add_subplot(1, 1, 1)
    
    # open cam and grab one image
    camera = camera_init()
    camera.StartGrabbingMax(1)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = grabResult.Array
    img_display = ax1.imshow(img, cmap = 'gray', vmin=0, vmax=150)
    pyplot.axis('off')
    fig.canvas.draw()   # note that the first draw comes before setting data 
    pyplot.show(block=False)
    
    #start a grab loop
    camera.StartGrabbingMax(400)
    
    while camera.IsGrabbing():
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img = grabResult.Array          # accessing image data
        img_display.set_data(img)
        
        if blit:
            # redraw just the points
            ax1.draw_artist(img_display)

        else:
            # redraw everything
            fig.canvas.draw()
        fig.canvas.flush_events()
    pyplot.close()
    camera.Close()
