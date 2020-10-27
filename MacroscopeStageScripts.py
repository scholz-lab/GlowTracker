# Macroscope Stage Module
import math
import sys
import numpy as np
import os
import csv
import matplotlib as plt
from pypylon import genicam, pylon
from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.measure import regionprops, label
from skimage.feature import register_translation as phase_cross_correlation
from skimage.registration import phase_cross_correlation
from skimage import io
from tifffile import imsave, TiffWriter
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

Library.toggle_device_db_store(True)

#%% Camera initialization
def camera_init():
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



#%% Stage homing & moving to starting position
def home_stage():
    '''
    homes all connected devices & moves axes to starting positions
    necessary if device was disconnected from power source
    '''
    with Connection.open_serial_port('COM3') as connection:
        device_list = connection.detect_devices()
        connection.renumber_devices(first_address = 1)
        
        for device in device_list:
            device.all_axes.home()
        
        device1 = device_list[0]
        device2 = device_list[1]
        axis_x = device1.get_axis(1)
        axis_y = device2.get_axis(1)
        if len(device_list) > 2:
            device3 = device_list[2]
            axis_z = device3.get_axis(1)
        
        print('Moving axix_x to starting position...')
        axis_x.move_absolute(20, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
        print('Moving axis_y to starting position...')
        axis_y.move_absolute(75, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
        print('Moving axis_z to starting position...')
        axis_z.move_absolute(130, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
        

#%% Range limits for axes
def set_rangelimits():
    '''
    sets limit for every device axis separately
    necessary to avoid collision with other set-up elements
    '''
    with Connection.open_serial_port('COM3') as connection:
        device_list = connection.detect_devices()
        connection.renumber_devices(first_address = 1)

        # assign connected devices to IDs & axes variables    
        device1 = device_list[0]
        device2 = device_list[1]
        axis_x = device1.get_axis(1)
        axis_y = device2.get_axis(1)
        if len(device_list) > 2:
            device3 = device_list[2]
            axis_z = device3.get_axis(1)
        
        # set axes limits in millimetres (max. value is )
        axis_x.settings.set('limit.max', 160, Units.LENGTH_MILLIMETRES)
        axis_y.settings.set('limit.max', 160, Units.LENGTH_MILLIMETRES)
        axis_z.settings.set('limit.max', 155, Units.LENGTH_MILLIMETRES)
        
#%% Stage Calibration
def genCalibrationMatrix(shift):
    '''Calculating calibration matrix from extracted coordinates
    input: list of coordinates from images taken between stage movements
    output: calibration matrix, translating image distances to stage & correcting rotation
    '''
    xTranslation = shift[1]
    yTranslation = shift[0]
    print(shift)
    # Length translation
    xPixelSize = 500/xTranslation # units of um/px
    yPixelSize = 500/yTranslation # units of um/px
    # Rotation angle
    rotation = math.atan2(xTranslation, yTranslation) - math.pi/4 
    print(math.degrees(rotation))
    # Create calibration matrix (Rotation matrix reordered y, x)
    calibrationMatrix = np.zeros((2,2))
    calibrationMatrix[0][0] = -yPixelSize*math.sin(rotation)
    calibrationMatrix[0][1] = yPixelSize*math.cos(rotation)
    calibrationMatrix[1][0] = xPixelSize*math.cos(rotation)
    calibrationMatrix[1][1] = xPixelSize*math.sin(rotation)

    return calibrationMatrix


def getImgs(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff'):
    # Hardware initialization
    connection =  Connection.open_serial_port('COM3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()
    
    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)

    camera = camera_init()
    
    # grabbing images before & after stage movement
    counter = 0
    while counter < 2:
        print('Grabbing image...')
        camera.StartGrabbingMax(1)      # taking image
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img = grabResult.Array          # accessing image data  
        if counter == 0:
            imsave(im0name, img)
            axis_x.move_relative(500, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            axis_y.move_relative(500, Units.LENGTH_MICROMETRES, wait_until_idle=True)
        elif counter == 1:
            imsave(im1name, img)
        counter += 1
    # return to origin
    axis_x.move_relative(-500, Units.LENGTH_MICROMETRES, wait_until_idle=False)
    axis_y.move_relative(-500, Units.LENGTH_MICROMETRES, wait_until_idle=True)
    camera.Close()
    connection.close()
    

def getCalibrationMatrix(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff'):
    #getImgs()
    im0 = io.imread(os.path.join(os.getcwd(), im0name))
    im1 = io.imread(os.path.join(os.getcwd(), im1name))
    # calculate the shift using FFT correlation
    shift = phase_cross_correlation(im0, im1, upsample_factor=1, space='real', return_error=0, overlap_ratio=0.1)    
    # Generate calibration matrix
    calibrationMatrix = genCalibrationMatrix(shift)         
    print('Calibration completed.')
    return calibrationMatrix


def writeCalibration(calibrationMatrix, fname = 'calibrationMatrix.txt'):
    np.savetxt(os.path.join(os.getcwd(), fname), calibrationMatrix)


def readCalibration(fname):
    return np.loadtxt(os.path.join(os.getcwd(), fname))
    
def stageCalibration():
    calibmatrix = getCalibrationMatrix(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff')
    writeCalibration(calibmatrix, fname = 'calibrationMatrix.txt')
        

#%% Autofocus using z axis
def zFocus():
    # Hardware initialization
    connection =  Connection.open_serial_port('COM3')
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


#%% Functions used for centering stage
def extractWorms(img, area = 0, binfactor = 4, li_init = 10):
    '''
    use otsu threshold to obtain mask of pharynx & label them
    input: image of shape (N,M) 
    output: array of worm coordinates.
    '''
    img = img[::binfactor, ::binfactor]
    mask = img>threshold_otsu(img)
    labeled = label(mask)
    coords = []
    for region in regionprops(labeled):
        if area <= region.area:
            y, x = region.centroid
            coords.append([y*binfactor,x*binfactor])
    return np.array(coords)


def getDistanceToCenter(coords, imshape):
    '''
    calculates distance of worms to center, keeps x/y-distances for worm closest to center
    input: list of all worm coordinates & tuple of image shape
    output: list of distances(px) of closest worm to center
    '''
    centerCoords = [px/2 for px in imshape]
    # calculate maximal distance from center in field of view
    smallestDistanceToCenter = math.sqrt((imshape[0] - centerCoords[0])**2 + (imshape[1] - centerCoords[1])**2)
    for worm in coords:
        distanceToCenter = math.sqrt((centerCoords[0] - worm[0])**2 + (centerCoords[1] - worm[1])**2)
        if distanceToCenter < smallestDistanceToCenter:
            smallestDistanceToCenter = distanceToCenter    
            deltaY = worm[0] - centerCoords[0] # y coordinates are inverted relative to axis
            deltaX = centerCoords[1] - worm[1]
            deltaCoords = [deltaY, deltaX]
    return deltaCoords


def getStageDistances(deltaCoords, calibrationMatrix):
    '''
    translates distance of worm to center from px to um
    input: distance of worm to center in px
    output: distance of worm to center in um
    '''
    stageDistances = np.dot(calibrationMatrix, deltaCoords)
    return stageDistances


#%% Center stage once
def center_once():
    # Hardware Initialization
    connection = Connection.open_serial_port('COM3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()
    
    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)
        
    camera = camera_init()
    
    # image retrieval & stage motion
    print('Grabbing image...')
    camera.StartGrabbingMax(1)      # taking image
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    img = grabResult.Array          # accessing image data
    print('Starting image analysis...')
    imshape = img.shape
    print('Getting coordinates...')
    coords = extractWorms(img)      # extracting lawn coordinates from image
    print('Number of coordinates: ', len(coords))
    print('Completing plots...')
    plt.figure()
    plt.imshow(img)
    [plt.plot(x,y, 'rx') for (y,x) in coords]
    print('Image analysis finished.')
    print('Calculating smallest distance(px) to center...')
    deltaCoords = getDistanceToCenter(coords, imshape)      # calculating distance of worm coordinates from center of image
    print('Translating distances on image to stage motion...')
    calibrationMatrix = readCalibration(fname = 'calibrationMatrix.txt')   # read in calibration matrix
    stageDistances = getStageDistances(deltaCoords, calibrationMatrix) # get distances for stage axes
    print('Centering...')
    axis_x.move_relative(stageDistances[1], Units.LENGTH_MICROMETRES, wait_until_idle=False)
    axis_y.move_relative(stageDistances[0], Units.LENGTH_MICROMETRES, wait_until_idle=True)

    camera.Close()        
    connection.close()      
    print('Image centered.')


#%% Tracking Worm
def trackWorm():
    connection =  Connection.open_serial_port('COM3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()
    
    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)
    
    # get calibration matrix from text file
    calibrationMatrix = readCalibration(fname = 'calibrationMatrix.txt')
                
    my_tiff = TiffWriter('TestTracking', bigtiff=True, append=True)

    camera = camera_init()
    count = 0
    with open('TrackingLog.csv', 'w') as logfile:
        writer = csv.writer(logfile, lineterminator='\n')
        writer.writerow(['Iteration #', 'x Coord', 'y Coord'])
        while count < 100:
            print('Grabbing image...')
            camera.StartGrabbingMax(1)      # taking image
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            img = grabResult.Array          # accessing image data
            my_tiff.save(img)               # appending image to tiff file
            imshape = img.shape
            coords = extractWorms(img)      # extracting worm coordinates from image
            deltaCoords = getDistanceToCenter(coords, imshape)      # calculating distance of worm coordinates from center of image
            stageDistances = getStageDistances(deltaCoords, calibrationMatrix)  # get distances for stage axes
            # write a string into the open file
            writer.writerow([count, stageDistances[1], stageDistances[0]])
            print('Centering worm...')
            axis_x.move_relative(stageDistances[1], Units.LENGTH_MICROMETRES, wait_until_idle=False)
            axis_y.move_relative(stageDistances[0], Units.LENGTH_MICROMETRES, wait_until_idle=False)
            count += 1
        camera.Close()
        my_tiff.close()
        connection.close()
        print('Tracking finished.')

