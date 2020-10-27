# ===============================================================================
#   Attempt to generate a live image and display it using Matplotlib
#   and detect key events
# ===============================================================================
from pypylon import pylon
from pypylon import genicam
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cannon_communication_protocol as canon
import time
import cv2


def init_camera():
    # TODO: load a specific parameter set
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        # Loading settings
        print("Possible place to read the file defining the camera's settings...")
        nodeFile = "FluorescenceTestLawn.pfs"
        pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exit_code = 1
    sys.exit(exit_code)

def main():
    #%% Hardware initialization
    camera = init_camera()  # camera initialization
    clock, signal, out = canon.initialize_DAQ()  # DAQ initialization
    print('Initialization DAQ completed')
    canon.align_communication(clock, signal, out)  # Objective initialization
    print('Canon bit alignment completed')
    canon.full_protocol(clock, signal, out, '6')
    time.sleep(0.4)
    
    #%% Taking an image to initialize display
    camera.StartGrabbingMax(1)
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = grabResult.Array  # Access the image data.
    focus_steps = 30
    stack = np.zeros((img.shape[0], img.shape[1], focus_steps), dtype='uint16')
    stack[:, :, 0] = img
    
    # Image display
    cv2.namedWindow('Live Image', cv2.WINDOW_NORMAL)  # used to bring to frontmost cv2
    scale = 0.5
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Live Image', cv2.equalizeHist(resized_image))
    cv2.waitKey(1)
    # Repeat for all focal planes
    step_command = '44 00 21'
    for frame in np.arange(1, focus_steps):
        canon.full_protocol(clock, signal, out, step_command)  # move objective
        #time.sleep(0.05)
        camera.StartGrabbingMax(1)
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # Access the image data.
        img = grabResult.Array
        stack[:, :, frame] = img
        # Display
        resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow('Live Image', cv2.equalizeHist(resized_image))
        cv2.waitKey(1)
        grabResult.Release()
    cv2.destroyAllWindows()
    
    #%% Looking for the focal plane using the frame of maximal variance
    average_frame_intensity = np.mean(stack, axis=(0, 1))
    normalized_stack = stack.astype(float)/average_frame_intensity
    stack_variance = np.var(normalized_stack, axis=(0, 1))
    focal_plane = np.argmax(stack_variance)
    #  Graphical display of the variance and the selected focal plane
    fig2, axs2 = plt.subplots(figsize=(6, 6), nrows=2, ncols=1)
    axs2[0].imshow(stack[:, :, focal_plane])  # displaying the selected focal plane
    axs2[0].axes.axis('off')
    axs2[1].plot(stack_variance)  # displaying the variance curve
    axs2[1].set_ylabel('Variance of pixel intensity')
    axs2[1].set_xlabel('Frame #')
    plt.draw()
    plt.show()
    
    #%% Moving to best position
    canon.full_protocol(clock, signal, out, '6')
    time.sleep(0.4)
    for frame in np.arange(0, focal_plane):
        canon.full_protocol(clock, signal, out, step_command)  # move objective
    
    # acquire picture
    camera.StartGrabbingMax(1)
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = grabResult.Array  # Access the image data.
    # Image display
    cv2.namedWindow('Image at focus', cv2.WINDOW_NORMAL)  # used to bring to frontmost cv2
    resized_image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Image at focus', cv2.equalizeHist(resized_image))
    cv2.waitKey(1)
    grabResult.Release()
    
    #%% Displaying a mosaic of all frames
    fig3, ax3 = plt.subplots(figsize=(15, 15), nrows=5, ncols=6, tight_layout=True)
    for index_row in range(5):
        for index_column in range(6):
            frame = index_row*6 + index_column
            ax3[index_row,index_column].imshow(normalized_stack[:, :, frame])
            ax3[index_row,index_column].annotate(str(frame), xy=(100, 200), color='black', fontsize=14)
            ax3[index_row, index_column].axes.axis('off')
    plt.draw()
    plt.show()
    
    #%% Data saving could go here
    # os.chdir("C:\Data\CElegans\Behaviour Microscope\AcquisitionFolder")  # Go to folder to save data
    # image.AttachGrabResultBuffer(grabResult)
    # image.Save(pylon.ImageFileFormat_Tiff, fname.format(grabResult.TimeStamp))
    
    #%% disconnecting from the camera and the DAQ card
    camera.Close()
    canon.close_DAQ(clock, signal, out)
    
    # TODO: trigger to avoid artifacts due to movement of the objective while imaging and to increase speed
    # Further development could: Load settings from a .pfs file. Alternatively, it could do an autoexposure/ gain
    # until it detects the lawn.
    # An additional tool could be designed to select the area where the maximal variance should be exploited for autofocus
    # Improve the way the objective reaches again the final desired focal plane
