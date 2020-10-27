import sys
import numpy as np
from pypylon import genicam
from zaber_motion import Units
from zaber_motion.ascii import Connection
from pypylon import pylon
import matplotlib.pyplot as plt

def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        # Loading settings
        print("Possible place to read the file defining the camera's settings...")
        nodeFile = "FluorescenceTestWorms.pfs"
        pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exit_code = 1
    sys.exit(exit_code)

def zFocus():
    # Hardware initialization
    connection =  Connection.open_serial_port('com3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()

    device3 = device_list[2]
    axis_z = device3.get_axis(1)
    
    camera = camera_init()
    
    # Taking an image to initialize display
    camera.StartGrabbingMax(1)
    # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = grabResult.Array  # Access the image data.
    focus_steps = 25
    stack = np.zeros((img.shape[0], img.shape[1], focus_steps), dtype='uint16')
    stack[:, :, 0] = img
    
    zPosition = []
    for frame in np.arange(1, focus_steps):
        camera.StartGrabbingMax(1)
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # Access the image data.
        img = grabResult.Array
        stack[:, :, frame] = img
        grabResult.Release()
        #List of z axis positions
        CurrentPos = axis_z.get_position(Units.LENGTH_MILLIMETRES)
        zPosition.append(CurrentPos)
        # Move z axis
        axis_z.move_relative(280, Units.LENGTH_MICROMETRES, wait_until_idle=True)
    
    # Looking for the focal plane using the frame of maximal variance
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
    
    # Second sweep in smaller range
    axis_z.move_absolute(zPosition[focal_plane-2], Units.LENGTH_MILLIMETRES, wait_until_idle=True)
    newstack = np.zeros((img.shape[0], img.shape[1], focus_steps), dtype='uint16')
    newstack[:, :, 0] = img
    zPosition = []
    for frame in np.arange(1, focus_steps):
        camera.StartGrabbingMax(1)
        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        # Access the image data.
        img = grabResult.Array
        newstack[:, :, frame] = img
        grabResult.Release()
        #List of z axis positions
        CurrentPos = axis_z.get_position(Units.LENGTH_MILLIMETRES)
        zPosition.append(CurrentPos)
        # Move z axis
        axis_z.move_relative(56, Units.LENGTH_MICROMETRES, wait_until_idle=True)
    
    # Looking for the focal plane using the frame of maximal variance
    average_frame_intensity = np.mean(newstack, axis=(0, 1))
    normalized_stack = newstack.astype(float)/average_frame_intensity
    stack_variance = np.var(normalized_stack, axis=(0, 1))
    focal_plane = np.argmax(stack_variance)
    #  Graphical display of the variance and the selected focal plane
    fig2, axs2 = plt.subplots(figsize=(6, 6), nrows=2, ncols=1)
    axs2[0].imshow(newstack[:, :, focal_plane])  # displaying the selected focal plane
    axs2[0].axes.axis('off')
    axs2[1].plot(stack_variance)  # displaying the variance curve
    axs2[1].set_ylabel('Variance of pixel intensity')
    axs2[1].set_xlabel('Frame #')
    plt.draw()
    plt.show()
    
    # Moving to best position
    axis_z.move_absolute(zPosition[focal_plane], Units.LENGTH_MILLIMETRES, wait_until_idle=False)

    # Displaying a mosaic of all frames
    fig3, ax3 = plt.subplots(figsize=(15, 15), nrows=5, ncols=5, tight_layout=True)
    for index_row in range(5):
        for index_column in range(5):
            frame = index_row*5 + index_column
            ax3[index_row,index_column].imshow(normalized_stack[:, :, frame])
            ax3[index_row,index_column].annotate(str(frame), xy=(100, 200), color='black', fontsize=14)
            ax3[index_row, index_column].axes.axis('off')
    plt.draw()
    plt.show()
    
    camera.Close()
    connection.close()
    
zFocus()
